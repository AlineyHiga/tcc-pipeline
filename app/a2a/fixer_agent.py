"""Fixer agent: generates and applies patches based on Requester context."""
from __future__ import annotations

import ast
import difflib
import hashlib
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Union

from langchain_core.prompts import ChatPromptTemplate

from app.a2a.protocol import State
from app.llm_client import LLMClient

LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = """
Você é o Fixer Agent. Receba o contexto preparado pelo Requester e devolva o arquivo Python completo já refatorado com as correções solicitadas.
Instruções:
- Analise o problema reportado pelo SonarQube
- Corrija APENAS o problema específico mencionado
- Reflita as correções em TODO o arquivo, mantendo a estrutura e funcionalidades existentes
- Retorne SOMENTE o código Python final dentro de um bloco ```python``` (sem explicações, diffs ou comentários extras)
- Preserve a indentação e o formato do arquivo
"""


class FixerAgent:
    def __init__(
        self,
        temperature: float = 0.1,
        repo_root: Optional[Union[Path, str]] = None,
    ) -> None:
        self.llm = LLMClient(role="fixer", temperature=temperature)
        env_root = os.getenv("A2A_REPO_ROOT")
        if repo_root:
            base = Path(repo_root)
        elif env_root:
            base = Path(env_root)
        else:
            base = Path.cwd()
        self.repo_root = base.resolve()
        LOGGER.debug("Fixer repo root set to %s", self.repo_root)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "human",
                    "Você receberá um issue do SonarQube e deve responder com o arquivo Python completo corrigido.\n"
                    "Regra: {issue_rule}\n"
                    "Mensagem: {issue_message}\n"
                    "Arquivo: {target_path}\n"
                    "Contexto adicional:\n{requester_context}\n\n"
                    "Código original:\n{original_code}\n\n"
                    "Retorne apenas um bloco ```python``` contendo o arquivo completo com as correções aplicadas.",
                )
            ]
        )

    def invoke(self, state: State) -> State:
        context = state.get("context", "")
        issues_for_file: List = list(state.get("issues_for_file") or [])
        issue = state.get("issue")

        LOGGER.debug("Fixer received context with %d chars", len(context))
        if context:
            LOGGER.debug("Fixer context preview: %s", context[:500].replace("\n", "\\n"))
        if issues_for_file:
            LOGGER.debug("Fixer issues for file: %s", [item.key for item in issues_for_file])

        file_hint = state.get("file_path") or self._extract_file_path(context)
        LOGGER.debug("Fixer target file path hint: %s", file_hint)
        resolved_path = self._resolve_file_path(file_hint)
        if not resolved_path:
            LOGGER.error("Fixer could not resolve target file: %s", file_hint)
            state.update({
                "fixer_summary": "Arquivo não encontrado",
                "fix_failed": True,
            })
            return state

        diff_path = self._format_diff_path(resolved_path, file_hint)
        LOGGER.debug("Fixer resolved target file to %s (diff path %s)", resolved_path, diff_path)
        original_content = resolved_path.read_text()
        LOGGER.debug("Original file size: %d chars", len(original_content))

        issues_block = self._render_issue_list(issues_for_file or ([issue] if issue else []))
        issue_rules = [
            getattr(item, "rule", "")
            for item in (issues_for_file or [])
            if getattr(item, "rule", "")
        ]

        LOGGER.info("Fixer gerando patch único para %s", diff_path)
        prompt_input = {
            "issue_rule": ", ".join(dict.fromkeys(issue_rules)) or getattr(issue, "rule", ""),
            "issue_message": issues_block,
            "target_path": diff_path,
            "requester_context": context or "(Contexto indisponível)",
            "original_code": original_content,
        }
        prompt_value = self.prompt.format_prompt(**prompt_input)
        user_prompt = prompt_value.to_messages()[0].content
        LOGGER.debug("Prompt size sent to LLM: %d chars", len(user_prompt))
        raw_response = self.llm.invoke(SYSTEM_PROMPT, user_prompt)
        LOGGER.debug("Raw LLM response (first 1k chars): %s", raw_response[:1000])
        fixed_content = raw_response.strip()
        LOGGER.debug("Raw LLM response size: %d chars", len(fixed_content))

        fixed_content = self._clean_code_response(fixed_content)
        LOGGER.debug("Cleaned LLM response size: %d chars", len(fixed_content))
        if fixed_content:
            preview = fixed_content[:200].replace("\n", "\\n")
            LOGGER.debug("Sanitized fixer output preview: %s%s", preview, "..." if len(fixed_content) > 200 else "")
        original_digest = hashlib.sha256(original_content.encode("utf-8")).hexdigest()
        fixed_digest = hashlib.sha256(fixed_content.encode("utf-8")).hexdigest() if fixed_content else "EMPTY"
        LOGGER.debug("Content digests — original=%s fixed=%s", original_digest, fixed_digest)

        if self._looks_like_diff_response(fixed_content):
            LOGGER.warning("Fixer detected diff-like output, sanitizing.")
            fixed_content = self._sanitize_diff_like_response(
                fixed_content, diff_path, original_content
            )
            LOGGER.debug("Sanitized diff-like response size: %d chars", len(fixed_content))

        is_valid_python, syntax_error = self._is_valid_python_code(fixed_content)
        if not is_valid_python:
            LOGGER.error(
                "Fixer produced invalid Python source for %s: %s",
                diff_path,
                syntax_error or "syntax error",
            )
            state["patch"] = ""
            state.update({
                "fixer_summary": (
                    f"Código inválido retornado pelo Fixer: {syntax_error}"
                    if syntax_error
                    else "Fixer retornou código inválido"
                ),
                "fix_failed": True,
            })
            return state

        candidate_patch = self._generate_patch(original_content, fixed_content, diff_path)
        LOGGER.info("Generated patch (first 1k chars): %s", candidate_patch[:1000])
        LOGGER.debug("Generated patch size: %d chars", len(candidate_patch))
        if candidate_patch:
            LOGGER.debug("Candidate patch line count: %d", candidate_patch.count("\n"))
        else:
            LOGGER.debug("Candidate patch vazio para %s", diff_path)

        if not candidate_patch or "@@" not in candidate_patch:
            LOGGER.error("Generated patch missing diff hunks (@@) for %s", diff_path)
            state["patch"] = ""
            state.update({
                "fixer_summary": "Falha ao gerar patch válido",
                "fix_failed": True,
            })
            return state

        applies, patch_error = self._patch_applies(candidate_patch)
        if not applies:
            LOGGER.error(
                "Generated patch failed to apply in dry-run for %s: %s",
                diff_path,
                patch_error or "git apply --check retornou código não zero",
            )
            state["patch"] = ""
            state.update({
                "fixer_summary": (
                    "Falha ao validar patch: "
                    f"{patch_error}" if patch_error else "Falha ao validar patch gerado"
                ),
                "fix_failed": True,
            })
            return state

        state["patch"] = candidate_patch
        state.update({
            "fixer_summary": "Patch gerado com sucesso",
            "fix_failed": False,
        })
        return state

    # ---------------------------------------------------------------------

    def _extract_file_path(self, context: str) -> str:
        """Extract file path from context."""
        lines = context.split('\n')
        for line in lines:
            if 'Arquivo alvo:' in line:
                return line.split(':', 1)[1].strip()
        return ""

    def _render_issue_list(self, issues: List) -> str:
        if not issues:
            return ""
        lines: List[str] = []
        for idx, item in enumerate(issues, start=1):
            line = getattr(item, "line", None)
            message = getattr(item, "message", "")
            rule = getattr(item, "rule", "")
            key = getattr(item, "key", f"ISSUE-{idx}")
            lines.append(
                f"{idx}. ({key}) Linha {line if line is not None else '?'} — {message} ({rule})"
            )
        return "\n".join(lines)

    def _resolve_file_path(self, reference: str) -> Optional[Path]:
        """Resolve file references relative to the configured repo root."""
        if not reference:
            return None
        candidate_paths = []
        raw = Path(reference)
        if raw.is_absolute():
            candidate_paths.append(raw)
        candidate_paths.append(self.repo_root / raw)
        candidate_paths.append(Path.cwd() / raw)

        seen = set()
        for candidate in candidate_paths:
            try:
                resolved = candidate.resolve()
            except FileNotFoundError:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            if resolved.exists():
                return resolved
        return None

    def _format_diff_path(self, absolute: Path, hint: str) -> str:
        """Return a path string appropriate for unified diff headers."""
        try:
            return absolute.relative_to(self.repo_root).as_posix()
        except ValueError:
            if hint:
                return Path(hint).as_posix()
            return absolute.as_posix()

    def _clean_code_response(self, response: str) -> str:
        """Remove code fences and extra text from LLM response."""
        cleaned = response.strip()
        if not cleaned:
            return ""

        # Remove possíveis tokens sentinela devolvidos pelo modelo (ex: <|im_end|]>).
        cleaned = cleaned.replace("<|im_end|]>", "").strip()

        fence_pattern = re.compile(r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
        match = fence_pattern.search(cleaned)
        code_block = ""
        if match:
            code_block = match.group(1).strip()
            LOGGER.debug("Código extraído de bloco markdown ```python``` com %d chars", len(code_block))
        else:
            LOGGER.warning("Nenhum bloco ```python``` encontrado; aplicando heurística de fallback.")
            if '```' in cleaned:
                parts = cleaned.split('```')
                for part in parts:
                    trimmed = part.strip()
                    if not trimmed:
                        continue
                    if 'import ' in trimmed or 'def ' in trimmed or 'class ' in trimmed:
                        code_block = trimmed
                        break
            if not code_block:
                code_block = cleaned

        if code_block.lower().startswith("python"):
            lines = code_block.splitlines()
            if lines and lines[0].strip().lower() == 'python':
                code_block = '\n'.join(lines[1:]).strip()

        return code_block.strip()

    def _looks_like_diff_response(self, response: str) -> bool:
        """Heuristic to detect diff snippets in the LLM output."""
        if not response:
            return False
        diff_pattern = re.compile(r"^(diff --git|--- |\+\+\+ |@@)", re.MULTILINE)
        match = bool(diff_pattern.search(response))
        LOGGER.debug("Diff-like response detected=%s", match)
        return match

    def _sanitize_diff_like_response(self, response: str, file_path: str, original: str) -> str:
        """
        Corrige respostas onde o LLM retornou um pseudo-diff em vez de código puro.
        Retorna código Python 'limpo' que pode ser usado para gerar um diff válido.
        """
        lines = response.splitlines()
        LOGGER.debug(
            "Sanitizing diff-like response for %s: %d raw lines",
            file_path,
            len(lines),
        )
        clean_lines = []
        skipped_headers = skipped_code_prefix = skipped_fences = 0
        for line in lines:
            # remove headers típicos de diff
            stripped = line.strip()
            if re.match(r"^(diff|---|\+\+\+|@@)", stripped):
                skipped_headers += 1
                continue
            # remove prefixos incorretos como '+python'
            if stripped.startswith("+python"):
                skipped_code_prefix += 1
                continue
            # remove cercas de código
            if stripped.startswith("```"):
                skipped_fences += 1
                continue
            clean_lines.append(line + "\n")

        cleaned = "".join(clean_lines).strip()
        LOGGER.debug(
            "Sanitize diff-like response kept %d lines (headers=%d, +python=%d, fences=%d)",
            len(clean_lines),
            skipped_headers,
            skipped_code_prefix,
            skipped_fences,
        )
        if not cleaned or cleaned == original.strip():
            LOGGER.warning("Sanitize produced no new code; fallback to original content.")
            return original
        return cleaned

    def _is_valid_python_code(self, code: str) -> tuple[bool, str | None]:
        """Return whether code is valid Python and, if not, the syntax error."""
        if not code.strip():
            return False, "resposta vazia"
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as exc:
            message = f"{exc.msg} (linha {exc.lineno}, coluna {exc.offset})"
            LOGGER.debug("SyntaxError while parsing fixer output: %s", message)
            return False, message

    def _generate_patch(self, original: str, fixed: str, file_path: str) -> str:
        """Generate unified diff patch leveraging GitPython when available."""
        LOGGER.debug(
            "Generating patch for %s: original=%d chars fixed=%d chars",
            file_path,
            len(original),
            len(fixed),
        )
        if not fixed.strip() or fixed.strip() == original.strip():
            LOGGER.warning("Fixer produced identical content or empty fix.")
            return ""

        try:
            return self._generate_patch_with_gitpython(original, fixed, file_path)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "GitPython falhou ao gerar diff para %s: %s. Recuando para difflib.",
                file_path,
                exc,
            )
            return self._generate_patch_with_difflib(original, fixed, file_path)

    def _generate_patch_with_gitpython(self, original: str, fixed: str, file_path: str) -> str:
        """Use GitPython (git diff --no-index) to produce a unified diff."""
        try:
            from git import Git  # lazy import to keep optional dependency
        except ImportError as exc:  # pragma: no cover - handled by fallback
            raise RuntimeError("GitPython não disponível") from exc

        target_path = Path(file_path)
        normalized_path = target_path.as_posix()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            original_rel = Path("orig") / target_path
            fixed_rel = Path("new") / target_path
            original_path = tmp_root / original_rel
            fixed_path = tmp_root / fixed_rel
            original_path.parent.mkdir(parents=True, exist_ok=True)
            fixed_path.parent.mkdir(parents=True, exist_ok=True)
            original_path.write_text(original, encoding="utf-8")
            fixed_path.write_text(fixed, encoding="utf-8")

            git_cli = Git(str(tmp_root))
            cmd = [
                "git",
                "--no-pager",
                "diff",
                "--no-index",
                "--text",
                "--no-ext-diff",
                "--color=never",
                "--unified=3",
                "--",
                original_rel.as_posix(),
                fixed_rel.as_posix(),
            ]
            status, stdout, stderr = git_cli.execute(
                cmd,
                with_extended_output=True,
                with_exceptions=False,
            )

        status = int(str(status).strip())
        if status not in (0, 1):
            stderr_text = (stderr or "").strip()
            raise RuntimeError(
                f"git diff retornou status {status} ao gerar patch: {stderr_text or 'erro desconhecido'}"
            )

        diff_output = stdout or ""
        if not diff_output.strip():
            LOGGER.debug("git diff não encontrou mudanças entre snapshots para %s", file_path)
            return ""

        normalized_diff = diff_output.replace(
            f"a/{original_rel.as_posix()}",
            f"a/{normalized_path}",
        ).replace(
            f"b/{fixed_rel.as_posix()}",
            f"b/{normalized_path}",
        )
        if not normalized_diff.endswith("\n"):
            normalized_diff += "\n"
        LOGGER.debug(
            "GitPython generated diff for %s totals %d chars",
            file_path,
            len(normalized_diff),
        )
        return normalized_diff

    def _generate_patch_with_difflib(self, original: str, fixed: str, file_path: str) -> str:
        """Fallback unified diff generation using difflib."""
        original_lines = original.splitlines(keepends=True)
        fixed_lines = fixed.splitlines(keepends=True)

        diff_lines = list(
            difflib.unified_diff(
                original_lines,
                fixed_lines,
                fromfile=f"a/{file_path}",
                tofile=f"b/{file_path}",
                lineterm="\n",
            )
        )

        if not diff_lines:
            LOGGER.error("Unified diff generation (difflib) returned empty output.")
            return ""

        header = f"diff --git a/{file_path} b/{file_path}\n"
        diff_text = "".join(diff_lines)
        if not diff_text.endswith("\n"):
            diff_text += "\n"
        full_diff = header + diff_text
        LOGGER.debug(
            "difflib generated diff for %s totals %d chars",
            file_path,
            len(full_diff),
        )
        return full_diff

    def _patch_applies(self, patch: str) -> tuple[bool, Optional[str]]:
        """Run git apply --check to verify patch validity without modifying files."""
        if not patch.strip():
            return False, "patch vazio"
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".patch") as handle:
                handle.write(patch)
                tmp_path = Path(handle.name)
            result = subprocess.run(
                ["git", "apply", "--check", str(tmp_path)],
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                return True, None
            combined = (result.stdout or "") + (result.stderr or "")
            return False, combined.strip() or "git apply --check retornou código não zero"
        finally:
            if tmp_path and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    LOGGER.debug("Não foi possível remover arquivo temporário %s", tmp_path)
