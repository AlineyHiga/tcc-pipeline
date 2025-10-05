"""LangGraph tool node to apply generated patches safely."""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Iterable, List, Optional

from app.a2a.protocol import State
from app.patcher import PatchApplicationError, apply_patch

LOGGER = logging.getLogger(__name__)

_LINE_NUMBER_PATTERN = re.compile(r"^\s*\d{1,5}[:\-]\s?")
_HUNK_HEADER_PATTERN = re.compile(r"@@\s+-([0-9]+)(?:,[0-9]+)?\s+\+([0-9]+)(?:,[0-9]+)?\s+@@")
_SANITIZE_STRIP_TOKENS = {
    "*** Begin Patch",
    "*** End Patch",
    "```",
    "```diff",
    "```patch",
    "Patch:",
}


def _sanitize_diff(raw_diff: str) -> str:
    """Remove common artefacts that break unified diffs."""

    if not raw_diff or not raw_diff.strip():
        raise PatchApplicationError("Diff vazio recebido do Fixer")

    lines: list[str] = []
    for raw_line in raw_diff.splitlines():
        stripped = raw_line.strip()
        if stripped in _SANITIZE_STRIP_TOKENS:
            continue
        cleaned = _LINE_NUMBER_PATTERN.sub("", raw_line)
        if "diff --git" in cleaned and not cleaned.lstrip().startswith("diff --git"):
            cleaned = cleaned[cleaned.index("diff --git") :]
        lines.append(cleaned)

    sanitized = "\n".join(lines).strip()
    header_idx = sanitized.find("diff --git")
    if header_idx == -1:
        raise PatchApplicationError("Diff do Fixer não contém header `diff --git`")
    sanitized = sanitized[header_idx:]
    if not sanitized.endswith("\n"):
        sanitized += "\n"
    return sanitized


def _strip_git_prefix(path: str) -> str:
    if path.startswith("a/") or path.startswith("b/"):
        return path[2:]
    return path


def _rehydrate_diff_with_repo(diff_text: str) -> str:
    """Reinsert exact original lines for deletions and context."""

    repo_root = Path(os.getenv("A2A_REPO_ROOT", Path.cwd())).resolve()
    current_file: Optional[Path] = None
    current_lines: list[str] = []
    src_index = 0
    in_hunk = False
    rebuilt: list[str] = []

    for line in diff_text.splitlines():
        if line.startswith("diff --git"):
            in_hunk = False
            rebuilt.append(line)
            continue
        if line.startswith("--- "):
            rebuilt.append(line)
            source_path = line[4:].strip()
            if source_path != "/dev/null":
                rel = _strip_git_prefix(source_path)
                current_file = repo_root / rel
                try:
                    current_lines = current_file.read_text().splitlines()
                except FileNotFoundError:
                    current_lines = []
            else:
                current_file = None
                current_lines = []
            continue
        if line.startswith("+++ "):
            rebuilt.append(line)
            continue
        if line.startswith("@@"):
            rebuilt.append(line)
            in_hunk = True
            match = _HUNK_HEADER_PATTERN.match(line)
            if not match:
                raise PatchApplicationError(f"Header de hunk inválido: {line}")
            src_index = int(match.group(1)) - 1
            continue
        if not in_hunk or current_file is None:
            rebuilt.append(line)
            continue
        if line.startswith("-") and not line.startswith("---"):
            if src_index >= len(current_lines):
                raise PatchApplicationError("Diff não corresponde ao arquivo atual (remoção fora do intervalo)")
            rebuilt.append("-" + current_lines[src_index])
            src_index += 1
            continue
        if line.startswith(" "):
            if src_index >= len(current_lines):
                raise PatchApplicationError("Diff não corresponde ao arquivo atual (contexto fora do intervalo)")
            rebuilt.append(" " + current_lines[src_index])
            src_index += 1
            continue
        if line.startswith("+"):
            rebuilt.append(line)
            continue
        rebuilt.append(line)

    return "\n".join(rebuilt) + "\n"


def _iter_file_diffs(diff_text: str) -> Iterable[List[str]]:
    lines = diff_text.splitlines()
    current: List[str] = []
    for line in lines:
        if line.startswith("diff --git "):
            if current:
                yield current
                current = []
        current.append(line)
    if current:
        yield current


_DEF_PATTERN = re.compile(r"def\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(")


def _extract_function_name(lines: Iterable[str]) -> Optional[str]:
    for raw in lines:
        stripped = raw.lstrip(" +-\t")
        if stripped.startswith("def "):
            match = _DEF_PATTERN.search(stripped)
            if match:
                return match.group("name")
    return None


def _apply_python_function_fallback(diff_text: str) -> bool:
    repo_root = Path(os.getenv("A2A_REPO_ROOT", Path.cwd())).resolve()
    applied_any = False

    for block in _iter_file_diffs(diff_text):
        header = block[0]
        parts = header.split()
        if len(parts) < 4:
            continue
        target_path = _strip_git_prefix(parts[3])
        file_path = repo_root / target_path
        if file_path.suffix != ".py" or not file_path.exists():
            continue

        func_name = _extract_function_name(block)
        if not func_name:
            continue

        new_body = [
            line[1:]
            for line in block
            if line.startswith("+") and not line.startswith("+++")
        ]
        if not new_body:
            continue

        try:
            applied_any = _replace_function_body(file_path, func_name, new_body) or applied_any
        except PatchApplicationError:
            raise
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug(
                "Python fallback failed para %s (%s): %s", func_name, file_path, exc
            )
            continue

    return applied_any


def _replace_function_body(file_path: Path, func_name: str, new_body: List[str]) -> bool:
    lines = file_path.read_text().splitlines()

    start_idx: Optional[int] = None
    indent_level = 0
    def_prefix = f"def {func_name}"

    for idx, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(def_prefix):
            start_idx = idx
            indent_level = len(line) - len(stripped)
            break

    if start_idx is None:
        raise PatchApplicationError(
            f"Função {func_name} não encontrada em {file_path.as_posix()}"
        )

    end_idx = start_idx + 1
    while end_idx < len(lines):
        stripped = lines[end_idx].lstrip()
        current_indent = len(lines[end_idx]) - len(stripped)
        if (
            stripped.startswith("def ")
            or stripped.startswith("class ")
            or (stripped and current_indent < indent_level)
        ):
            break
        end_idx += 1

    existing_body = lines[start_idx + 1 : end_idx]
    if existing_body == new_body:
        LOGGER.info("Corpo da função %s já está atualizado", func_name)
        return False

    replacement: List[str] = []
    replacement.extend(lines[: start_idx + 1])
    replacement.extend(new_body)
    if replacement and (end_idx >= len(lines) or lines[end_idx].strip()):
        replacement.append("")
    replacement.extend(lines[end_idx:])
    file_path.write_text("\n".join(replacement) + "\n")
    return True


def _is_patch_already_applied(diff_text: str) -> bool:
    repo_root = Path(os.getenv("A2A_REPO_ROOT", Path.cwd())).resolve()

    for block in _iter_file_diffs(diff_text):
        header = block[0]
        parts = header.split()
        if len(parts) < 4:
            return False
        target_path = _strip_git_prefix(parts[3])
        file_path = repo_root / target_path
        if not file_path.exists():
            return False

        content_lines = file_path.read_text().splitlines()
        plus_lines = [
            line[1:]
            for line in block
            if line.startswith("+") and not line.startswith("+++")
        ]
        minus_lines = [
            line[1:]
            for line in block
            if line.startswith("-") and not line.startswith("---")
        ]

        for line in plus_lines:
            if line and line not in content_lines:
                return False
        for line in minus_lines:
            if line and line in content_lines:
                return False

    LOGGER.debug("Verificação de patch já aplicado retornou verdadeiro")
    return True


def _apply_with_patch_ng(diff_text: str, repo_root: Path) -> bool:
    try:
        from patch_ng import fromstring as patch_fromstring
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("patch-ng indisponível: %s", exc)
        return False

    try:
        from unidiff import PatchSet
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("unidiff indisponível: %s", exc)
        return False

    try:
        patchset = PatchSet(diff_text)
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("Falha ao interpretar diff com unidiff: %s", exc)
        return False

    try:
        patch_obj = patch_fromstring(diff_text)
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("patch-ng não conseguiu criar patch: %s", exc)
        return False

    try:
        applied = bool(patch_obj.apply(root=str(repo_root)))
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("patch-ng falhou ao aplicar: %s", exc)
        return False

    if not applied:
        LOGGER.debug("patch-ng retornou falso ao aplicar diff")
        return False

    _stage_with_gitpython(repo_root, patchset)
    LOGGER.info("Patch aplicado com patch-ng")
    return True


def _stage_with_gitpython(repo_root: Path, patchset: "PatchSet") -> None:  # type: ignore[name-defined]
    try:
        from git import Repo
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("GitPython indisponível para staging: %s", exc)
        return

    try:
        repo = Repo(str(repo_root))
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("GitPython não conseguiu abrir o repo: %s", exc)
        return

    add_paths: List[str] = []
    remove_paths: List[str] = []

    for patched_file in patchset:
        new_path = _strip_git_prefix(getattr(patched_file, "path", ""))
        source_path = _strip_git_prefix(getattr(patched_file, "source_file", ""))

        if getattr(patched_file, "is_removed_file", False):
            if new_path:
                remove_paths.append(new_path)
        elif getattr(patched_file, "is_rename", False):
            if source_path:
                remove_paths.append(source_path)
            if new_path:
                add_paths.append(new_path)
        else:
            if new_path:
                add_paths.append(new_path)

    try:
        if add_paths:
            repo.index.add(add_paths)
        if remove_paths:
            repo.index.remove(remove_paths, working_tree=True)
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("GitPython staging falhou: %s", exc)
    else:
        if add_paths:
            LOGGER.debug("GitPython adicionou arquivos ao index: %s", add_paths)
        if remove_paths:
            LOGGER.debug("GitPython removeu arquivos do index: %s", remove_paths)


def apply_patch_node(state: State) -> State:
    """Apply the patch stored in the graph state using the patcher utility."""
    raw_patch: Optional[str] = state.get("patch")
    if not raw_patch:
        message = "Fixer não forneceu diff para aplicar"
        LOGGER.error(message)
        state.update({
            "fixer_summary": message,
            "fix_failed": True,
        })
        return state

    LOGGER.info("Patch recebido do Fixer (primeiros 200 chars): %s", raw_patch[:200])

    try:
        sanitized = _sanitize_diff(raw_patch)
    except PatchApplicationError as exc:
        LOGGER.error("Diff inválido recebido do Fixer: %s", exc)
        state.update({
            "fixer_summary": str(exc),
            "fix_failed": True,
        })
        return state

    LOGGER.debug("Diff sanitizado (primeiros 200 chars): %s", sanitized[:200])

    repo_root = Path(os.getenv("A2A_REPO_ROOT", Path.cwd())).resolve()

    # First try: use unidiff + patch-ng (+ GitPython) if disponíveis
    rehydrated_for_patch_ng = sanitized
    try:
        rehydrated_for_patch_ng = _rehydrate_diff_with_repo(sanitized)
        LOGGER.debug(
            "Diff reidratado para patch-ng (primeiros 200 chars): %s",
            rehydrated_for_patch_ng[:200],
        )
    except PatchApplicationError as exc:
        LOGGER.debug("Rehidratação para patch-ng falhou: %s", exc)

    if _apply_with_patch_ng(rehydrated_for_patch_ng, repo_root):
        state.update({
            "fixer_summary": "Patch aplicado com patch-ng",
            "fix_failed": False,
        })
        return state

    try:
        patch = _rehydrate_diff_with_repo(sanitized)
        LOGGER.debug(
            "Diff reidratado final (primeiros 200 chars): %s",
            patch[:200],
        )
        apply_patch(patch)
        state.update({
            "fixer_summary": "Patch aplicado com sucesso",
            "fix_failed": False,
        })
        return state
    except PatchApplicationError as exc:
        LOGGER.warning("Aplicação direta do diff falhou: %s", exc)

        if _is_patch_already_applied(sanitized):
            LOGGER.info("Patch já aplicado anteriormente; nada a fazer")
            state.update({
                "fixer_summary": "Patch já aplicado anteriormente",
                "fix_failed": False,
            })
            return state

        try:
            if _apply_python_function_fallback(sanitized):
                LOGGER.info("Diff aplicado via fallback específico para função Python")
                state.update({
                    "fixer_summary": "Patch aplicado com fallback Python",
                    "fix_failed": False,
                })
                return state
            LOGGER.debug("Fallback Python não modificou arquivos; mantendo falha")
        except PatchApplicationError:
            raise
        except Exception as fallback_exc:  # noqa: BLE001
            LOGGER.debug("Fallback Python inesperado: %s", fallback_exc)

        state.update({
            "fixer_summary": f"Falha ao aplicar patch: {exc}",
            "fix_failed": True,
        })
    return state


__all__ = ["apply_patch_node"]
