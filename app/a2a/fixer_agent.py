"""Fixer agent: generates and applies patches based on Requester context."""
from __future__ import annotations

import difflib
import logging
import os
from pathlib import Path
from typing import Optional, Union

from app.a2a.protocol import State
from app.llm_client import LLMClient

LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = """
Você é o Fixer Agent. Receba o contexto preparado pelo Requester e gere o código Python corrigido.
Instruções:
- Analise o problema reportado pelo SonarQube
- Corrija APENAS o problema específico mencionado
- Mantenha toda a estrutura e funcionalidade existente
- Retorne APENAS o código Python corrigido, sem explicações
- Não adicione comentários extras
- Preserve a indentação original
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

    def invoke(self, state: State) -> State:
        context = state.get("context", "")
        issue = state.get("issue")

        LOGGER.debug("Fixer received context with %d chars", len(context))
        if context:
            LOGGER.debug("Fixer context preview: %s", context[:500].replace("\n", "\\n"))
        if issue is not None:
            LOGGER.debug(
                "Fixer issue: message=%s rule=%s", getattr(issue, "message", ""), getattr(issue, "rule", "")
            )

        file_hint = self._extract_file_path(context)
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

        prompt = (
            f"{SYSTEM_PROMPT}\n\nProblema: {getattr(issue, 'message', '')}\nRegra: {getattr(issue, 'rule', '')}"
            f"\n\nCódigo original:\n{original_content}"
        )
        LOGGER.debug("Prompt size sent to LLM: %d chars", len(prompt))
        raw_response = self.llm.invoke(prompt, context)
        LOGGER.debug("Raw LLM response (first 1k chars): %s", raw_response[:1000])
        fixed_content = raw_response.strip()
        LOGGER.debug("Raw LLM response size: %d chars", len(fixed_content))

        fixed_content = self._clean_code_response(fixed_content)
        LOGGER.debug("Cleaned LLM response size: %d chars", len(fixed_content))

        patch = self._generate_patch(original_content, fixed_content, diff_path)
        LOGGER.info("Generated patch (first 1k chars): %s", patch[:1000])
        LOGGER.debug("Generated patch size: %d chars", len(patch))
        state["patch"] = patch

        if not patch or "@@" not in patch:
            LOGGER.error("Fixer produced invalid patch for %s", diff_path)
            state.update({
                "fixer_summary": "Falha ao gerar patch válido",
                "fix_failed": True,
            })
        else:
            state.update({
                "fixer_summary": "Patch gerado com sucesso",
                "fix_failed": False,
            })

        return state

    def _extract_file_path(self, context: str) -> str:
        """Extract file path from context."""
        lines = context.split('\n')
        for line in lines:
            if 'Arquivo alvo:' in line:
                return line.split(':', 1)[1].strip()
        return ""

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
        if '```' in response:
            parts = response.split('```')
            for part in parts:
                trimmed = part.strip()
                if not trimmed:
                    continue
                if 'import' in trimmed or 'def ' in trimmed or 'class ' in trimmed:
                    cleaned = trimmed
                    break
        if cleaned.lower().startswith('python'):
            lines = cleaned.splitlines()
            if lines and lines[0].strip().lower() == 'python':
                cleaned = '
'.join(lines[1:])
        return cleaned.strip()

    def _generate_patch(self, original: str, fixed: str, file_path: str) -> str:
        """Generate unified diff patch."""
        if original == fixed:
            return ""

        original_lines = original.splitlines(keepends=True)
        fixed_lines = fixed.splitlines(keepends=True)

        diff_lines = list(
            difflib.unified_diff(
                original_lines,
                fixed_lines,
                fromfile=f"a/{file_path}",
                tofile=f"b/{file_path}",
                lineterm='\n',
            )
        )
        if not diff_lines:
            return ""
        diff_text = ''.join(diff_lines)
        header = f"diff --git a/{file_path} b/{file_path}\n"
        if not diff_text.endswith('\n'):
            diff_text += '\n'
        return header + diff_text
