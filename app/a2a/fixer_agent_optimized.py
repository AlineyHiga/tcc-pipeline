"""Optimized Fixer agent with minimal context and no validation."""
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
from typing import Iterable, List, Optional, Union

from langchain_core.prompts import ChatPromptTemplate

from app.a2a.protocol import State
from app.llm_client import LLMClient
from app.utils import with_line_numbers

LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = """
Fix the specific SonarQube issue mentioned in the context.

Instructions:
- Fix ONLY the reported issue with minimal changes
- Return the complete corrected Python file in a ```python``` block
- Preserve indentation and formatting
- Do not include line numbers in your response
- Focus on the specific function/area mentioned in the issue
"""


class OptimizedFixerAgent:
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
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Fix the SonarQube issue in this Python file.\n"
                "File: {target_path}\n"
                "Issue: {issue_message}\n"
                "Context: {requester_context}\n"
                "Code to fix:\n{original_code}\n\n"
                "Return the complete corrected file in a ```python``` block. "
                "Focus on fixing the specific issue mentioned. "
            )
        ])

    def invoke(self, state: State) -> State:
        context = state.get("context", "")
        issues_for_file: List = list(state.get("issues_for_file") or [])
        issue = state.get("issue")

        LOGGER.debug("Fixer received context with %d chars", len(context))

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
        original_content = resolved_path.read_text()
        LOGGER.debug("Original file size: %d chars", len(original_content))

        issues_block = self._render_issue_list(issues_for_file or ([issue] if issue else []))

        LOGGER.info("Fixer gerando patch único para %s", diff_path)
        
        # Always provide complete file with line numbers
        code_context = with_line_numbers(original_content)
        
        # Use RAG to identify target function but send complete file
        try:
            from rag_service.service import RAGService
            rag_service = RAGService(str(self.repo_root / ".rag_index"))
            
            if issues_for_file:
                issue = issues_for_file[0]
                result = rag_service.retrieve_for_issue(
                    file_path=file_hint,
                    line=getattr(issue, 'line', 1),
                    rule=getattr(issue, 'rule', ''),
                    message=getattr(issue, 'message', ''),
                    k=1
                )
                
                target_function = result['target']['symbol']
                LOGGER.info(f"Fixer targeting function: {target_function} for issue at line {getattr(issue, 'line', '?')}")
                
        except Exception as e:
            LOGGER.debug("RAG not available, using complete file: %s", e)
        
        prompt_input = {
            "issue_message": issues_block,
            "target_path": diff_path,
            "requester_context": (context or "(Contexto indisponível)")[:600],
            "original_code": code_context,
        }
        prompt_value = self.prompt.format_prompt(**prompt_input)
        user_prompt = prompt_value.to_messages()[0].content
        LOGGER.debug("Prompt size sent to LLM: %d chars", len(user_prompt))
        raw_response = self.llm.invoke(SYSTEM_PROMPT, user_prompt)
        
        # Extract code from response
        fixed_content = self._clean_code_response(raw_response)
        if not fixed_content:
            LOGGER.error("Failed to extract code from LLM response")
            state["patch"] = ""
            state.update({
                "fixer_summary": "Falha ao interpretar resposta do LLM",
                "fix_failed": True,
            })
            return state

        # Basic validation
        is_valid_python, syntax_error = self._is_valid_python_code(fixed_content)
        if not is_valid_python:
            LOGGER.error("Fixer produced invalid Python source: %s", syntax_error)
            state["patch"] = ""
            state.update({
                "fixer_summary": f"Código inválido: {syntax_error}",
                "fix_failed": True,
            })
            return state

        candidate_patch = self._generate_patch(original_content, fixed_content, diff_path)
        LOGGER.info("Generated patch size: %d chars", len(candidate_patch))

        if not candidate_patch or "@@" not in candidate_patch:
            LOGGER.error("Generated patch missing diff hunks")
            state["patch"] = ""
            state.update({
                "fixer_summary": "Falha ao gerar patch válido",
                "fix_failed": True,
            })
            return state

        state["patch"] = candidate_patch
        state.update({
            "fixer_summary": "Patch gerado com sucesso",
            "fix_failed": False,
        })
        return state

    def _extract_file_path(self, context: str) -> str:
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
            lines.append(f"{idx}. Linha {line if line is not None else '?'} — {message} ({rule})")
        return "\n".join(lines)

    def _resolve_file_path(self, reference: str) -> Optional[Path]:
        if not reference:
            return None

        cleaned = reference.strip().replace("\\", "/")
        if not cleaned:
            return None

        # Try direct path first
        candidate = self.repo_root / cleaned
        if candidate.exists():
            return candidate.resolve()

        # Search in common locations
        for root in [self.repo_root, Path.cwd()]:
            for candidate in root.rglob(Path(cleaned).name):
                if candidate.exists():
                    return candidate.resolve()
        
        return None

    def _format_diff_path(self, absolute: Path, hint: str) -> str:
        try:
            return absolute.relative_to(self.repo_root).as_posix()
        except ValueError:
            return absolute.as_posix()

    def _clean_code_response(self, response: str) -> str:
        cleaned = response.strip()
        if not cleaned:
            return ""

        # Remove end tokens
        cleaned = cleaned.replace("<|im_end|]>", "").strip()
        cleaned = cleaned.replace("&gt;", ">").replace("&lt;", "<").replace("&quot;", '"')

        # Extract from ```python blocks
        fence_pattern = re.compile(r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
        match = fence_pattern.search(cleaned)
        
        if match:
            code_block = match.group(1).strip()
            LOGGER.debug("Extracted code from ```python``` block: %d chars", len(code_block))
            return code_block

        # Fallback
        return cleaned

    def _is_valid_python_code(self, code: str) -> tuple[bool, str | None]:
        if not code.strip():
            return False, "resposta vazia"
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as exc:
            message = f"{exc.msg} (linha {exc.lineno}, coluna {exc.offset})"
            return False, message

    def _generate_patch(self, original: str, fixed: str, file_path: str) -> str:
        if not fixed.strip() or fixed.strip() == original.strip():
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
            return ""

        header = f"diff --git a/{file_path} b/{file_path}\n"
        diff_text = "".join(diff_lines)
        if not diff_text.endswith("\n"):
            diff_text += "\n"
        
        return header + diff_text