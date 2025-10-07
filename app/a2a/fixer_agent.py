"""Fixer agent: generates and applies patches based on Requester context."""
from __future__ import annotations

import difflib
import logging
import os
import re
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

        # Limpa resposta de cercas de código ou prefixos
        fixed_content = self._clean_code_response(fixed_content)
        LOGGER.debug("Cleaned LLM response size: %d chars", len(fixed_content))

        # Se o LLM retornou um diff parcial (sem header git), sanitiza antes
        if self._looks_like_diff_response(fixed_content):
            LOGGER.warning("Fixer detected diff-like output, sanitizing.")
            fixed_content = self._sanitize_diff_like_response(
                fixed_content, diff_path, original_content
            )

        # Gera diff unificado completo
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

    def _generate_patch(self, original: str, fixed: str, file_path: str) -> str:
        """Generate unified diff patch with guaranteed valid header."""
        LOGGER.debug(
            "Generating patch for %s: original=%d chars fixed=%d chars",
            file_path,
            len(original),
            len(fixed),
        )
        if not fixed.strip() or fixed.strip() == original.strip():
            LOGGER.warning("Fixer produced identical content or empty fix.")
            return ""

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
            LOGGER.error("Unified diff generation returned empty output.")
            return ""

        header = f"diff --git a/{file_path} b/{file_path}\n"
        diff_text = "".join(diff_lines)
        LOGGER.debug(
            "Unified diff for %s has %d lines before header",
            file_path,
            len(diff_lines),
        )
        if not diff_text.endswith("\n"):
            diff_text += "\n"
        full_diff = header + diff_text
        LOGGER.debug(
            "Generated patch for %s totals %d chars",
            file_path,
            len(full_diff),
        )
        return full_diff
