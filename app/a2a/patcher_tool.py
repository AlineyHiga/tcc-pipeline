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

_METADATA_PREFIXES = (
    "index ",
    "new file mode ",
    "deleted file mode ",
    "old mode ",
    "new mode ",
    "similarity index ",
    "rename from ",
    "rename to ",
    "copy from ",
    "copy to ",
)


def _component_to_path(component: Optional[str]) -> Optional[str]:
    if not component:
        return None
    if ":" in component:
        _, rel = component.split(":", 1)
    else:
        rel = component
    rel = rel.strip()
    if not rel:
        return None
    return rel.replace("\\", "/")


def _derive_fallback_path(issue: Optional[object], repo_root: Path) -> Optional[str]:
    if issue is None:
        return None
    component = getattr(issue, "component", None)
    rel = _component_to_path(component)
    if not rel:
        return None
    candidate = Path(rel)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (repo_root / candidate).resolve()
    try:
        return resolved.relative_to(repo_root).as_posix()
    except ValueError:
        return resolved.as_posix()


def _normalize_candidate_path(value: Optional[str], repo_root: Optional[Path]) -> Optional[str]:
    if not value:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    if stripped == "/dev/null":
        return "/dev/null"
    normalized = stripped.replace("\\", "/")
    if normalized.startswith("a/") or normalized.startswith("b/"):
        normalized = normalized[2:]
    normalized = normalized.lstrip("./")
    if repo_root:
        candidate_path = Path(normalized)
        if candidate_path.is_absolute():
            resolved = candidate_path.resolve()
        else:
            resolved = (repo_root / normalized).resolve()
        try:
            return resolved.relative_to(repo_root).as_posix()
        except ValueError:
            return resolved.as_posix()
    return normalized


def _format_old_line(
    raw_value: Optional[str],
    file_path: str,
    has_deletions: bool,
    repo_root: Optional[Path],
) -> str:
    normalized = _normalize_candidate_path(raw_value, repo_root)
    if normalized == "/dev/null" or (normalized is None and not has_deletions):
        return "--- /dev/null"
    return f"--- a/{normalized or file_path}"


def _format_new_line(
    raw_value: Optional[str],
    file_path: str,
    has_additions: bool,
    repo_root: Optional[Path],
) -> str:
    normalized = _normalize_candidate_path(raw_value, repo_root)
    if normalized == "/dev/null" or (normalized is None and not has_additions):
        return "+++ /dev/null"
    return f"+++ b/{normalized or file_path}"


def _select_file_path(
    new_path: Optional[str],
    old_path: Optional[str],
    fallback_path: Optional[str],
    repo_root: Optional[Path],
) -> Optional[str]:
    for candidate in (new_path, old_path, fallback_path):
        normalized = _normalize_candidate_path(candidate, repo_root)
        if normalized and normalized != "/dev/null":
            return normalized
    return None


def _is_metadata_line(value: str) -> bool:
    stripped = value.lstrip()
    return any(stripped.startswith(prefix) for prefix in _METADATA_PREFIXES)


def _build_block(
    block_lines: List[str],
    repo_root: Optional[Path],
    fallback_path: Optional[str],
) -> List[str]:
    if not block_lines:
        return []

    idx = 0
    metadata: List[str] = []
    while idx < len(block_lines) and not block_lines[idx].startswith("--- "):
        metadata.append(block_lines[idx])
        idx += 1

    if idx >= len(block_lines):
        raise PatchApplicationError(
            "Bloco de diff sem linha inicial '---' não pôde ser interpretado"
        )

    old_line = block_lines[idx]
    idx += 1
    new_line: Optional[str] = None
    if idx < len(block_lines) and block_lines[idx].startswith("+++ "):
        new_line = block_lines[idx]
        idx += 1

    body = block_lines[idx:]
    has_additions = any(line.startswith("+") and not line.startswith("+++") for line in body)
    has_deletions = any(line.startswith("-") and not line.startswith("---") for line in body)

    old_path_raw = old_line[4:].strip()
    new_path_raw = new_line[4:].strip() if new_line else None

    file_path = _select_file_path(new_path_raw, old_path_raw, fallback_path, repo_root)
    if not file_path:
        raise PatchApplicationError(
            "Diff do Fixer não contém header `diff --git` e arquivo alvo não pôde ser inferido"
        )

    header = f"diff --git a/{file_path} b/{file_path}"
    formatted_old = _format_old_line(old_path_raw, file_path, has_deletions, repo_root)
    formatted_new = _format_new_line(new_path_raw, file_path, has_additions, repo_root)

    result: List[str] = [header]
    result.extend(metadata)
    result.append(formatted_old)
    result.append(formatted_new)
    result.extend(body)
    return result


def _reconstruct_headers(
    lines: List[str],
    repo_root: Optional[Path],
    fallback_path: Optional[str],
) -> str:
    if not lines:
        raise PatchApplicationError("Diff vazio recebido do Fixer")

    result: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            result.append(line)
            i += 1
            continue
        if line.startswith("--- "):
            metadata_prefix: List[str] = []
            while result and _is_metadata_line(result[-1]):
                metadata_prefix.insert(0, result.pop())
            block: List[str] = metadata_prefix
            while i < len(lines) and not (
                lines[i].startswith("diff --git ") or (i != 0 and lines[i].startswith("--- "))
            ):
                block.append(lines[i])
                i += 1
            block_result = _build_block(block, repo_root, fallback_path)
            result.extend(block_result)
            continue
        if line.startswith("diff --git "):
            result.extend(lines[i:])
            break
        result.append(line)
        i += 1

    if not any(item.startswith("diff --git ") for item in result):
        normalized_fallback = _normalize_candidate_path(fallback_path, repo_root)
        if not normalized_fallback or normalized_fallback == "/dev/null":
            raise PatchApplicationError(
                "Diff do Fixer não contém header `diff --git` e arquivo alvo não pôde ser inferido"
            )
        has_additions = any(
            line.startswith("+") and not line.startswith("+++") for line in result
        )
        has_deletions = any(
            line.startswith("-") and not line.startswith("---") for line in result
        )
        header_block = [
            f"diff --git a/{normalized_fallback} b/{normalized_fallback}",
            "--- /dev/null" if not has_deletions else f"--- a/{normalized_fallback}",
            "+++ /dev/null" if not has_additions else f"+++ b/{normalized_fallback}",
        ]
        result = header_block + result

    return "\n".join(result)


def _sanitize_diff(raw_diff: str, repo_root: Path, fallback_path: Optional[str] = None) -> str:
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
    line_list = sanitized.splitlines()
    for idx, line in enumerate(line_list):
        if line.startswith("diff --git "):
            trimmed = "\n".join(line_list[idx:])
            if not trimmed.endswith("\n"):
                trimmed += "\n"
            return trimmed

    reconstructed = _reconstruct_headers(line_list, repo_root, fallback_path)
    sanitized = reconstructed.strip()
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

    repo_root = Path(os.getenv("A2A_REPO_ROOT", Path.cwd())).resolve()
    fallback_path = _derive_fallback_path(state.get("issue"), repo_root)
    if fallback_path:
        LOGGER.debug("Fallback de caminho do issue: %s", fallback_path)

    try:
        sanitized = _sanitize_diff(raw_patch, repo_root, fallback_path)
    except PatchApplicationError as exc:
        LOGGER.error("Diff inválido recebido do Fixer: %s", exc)
        state.update({
            "fixer_summary": str(exc),
            "fix_failed": True,
        })
        return state

    LOGGER.debug("Diff sanitizado (primeiros 200 chars): %s", sanitized[:200])

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
