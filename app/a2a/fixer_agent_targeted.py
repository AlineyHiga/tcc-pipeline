"""Targeted Fixer agent that generates patches only for modified functions."""
from __future__ import annotations

import ast
import difflib
import logging
import os
import re
from pathlib import Path
from typing import List, Optional, Union

from langchain_core.prompts import ChatPromptTemplate

from app.a2a.protocol import State
from app.llm_client import LLMClient
from app.utils import with_line_numbers

LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = """
Fix the specific SonarQube issue in the given function.

Instructions:
- Fix ONLY the reported issue with minimal changes
- Return ONLY the corrected function in a ```python``` block
- Preserve indentation and formatting
- Do not include line numbers in your response
- Focus on the specific issue mentioned
"""


class TargetedFixerAgent:
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
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Fix the SonarQube issue in this function.\n"
                "File: {target_path}\n"
                "Issue: {issue_message}\n"
                "Function to fix:\n{function_code}\n\n"
                "Return ONLY the corrected function in a ```python``` block."
            )
        ])

    def invoke(self, state: State) -> State:
        context = state.get("context", "")
        issues_for_file: List = list(state.get("issues_for_file") or [])
        issue = state.get("issue")

        file_hint = state.get("file_path") or self._extract_file_path(context)
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
        issues_block = self._render_issue_list(issues_for_file or ([issue] if issue else []))

        # Get target function from requester state or use RAG as fallback
        target_function_code = None
        target_function_name = state.get('target_function')
        
        if target_function_name:
            LOGGER.info(f"Using target function from requester: {target_function_name}")
            # Extract the specific function from the file
            target_function_code = self._extract_function_by_name(original_content, target_function_name)
        
        # Fallback to RAG if no target function specified
        if not target_function_code:
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
                    
                    target_id = f"{result['target']['path']}::{result['target']['symbol']}"
                    code_map = rag_service.get_code_for_symbols([target_id])
                    
                    if code_map:
                        target_function_code = list(code_map.values())[0]
                        target_function_name = result['target']['symbol']
                        LOGGER.info(f"RAG fallback targeting function: {target_function_name}")
                    
            except Exception as e:
                LOGGER.debug("RAG not available: %s", e)

        # Final fallback: extract function by line number
        if not target_function_code and issues_for_file:
            issue_line = getattr(issues_for_file[0], 'line', 1)
            target_function_code, target_function_name = self._extract_function_by_line(
                original_content, issue_line
            )
            if target_function_name:
                LOGGER.info(f"Line-based fallback targeting function: {target_function_name}")

        if not target_function_code:
            LOGGER.error("Could not extract target function")
            state.update({
                "fixer_summary": "Não foi possível extrair função alvo",
                "fix_failed": True,
            })
            return state

        # Generate fix for the specific function
        prompt_input = {
            "issue_message": issues_block,
            "target_path": diff_path,
            "function_code": target_function_code,
        }
        
        prompt_value = self.prompt.format_prompt(**prompt_input)
        user_prompt = prompt_value.to_messages()[0].content
        LOGGER.debug("Prompt size: %d chars", len(user_prompt))
        
        raw_response = self.llm.invoke(SYSTEM_PROMPT, user_prompt)
        fixed_function = self._clean_code_response(raw_response)
        
        if not fixed_function:
            LOGGER.error("Failed to extract fixed function from LLM response")
            state.update({
                "fixer_summary": "Falha ao interpretar resposta do LLM",
                "fix_failed": True,
            })
            return state

        # Validate the fixed function
        if not self._is_valid_python_code(fixed_function)[0]:
            LOGGER.error("Fixed function is not valid Python")
            state.update({
                "fixer_summary": "Função corrigida contém erros de sintaxe",
                "fix_failed": True,
            })
            return state

        # Generate patch by replacing the original function with the fixed one
        fixed_content = self._replace_function_in_file(
            original_content, target_function_code, fixed_function
        )
        
        if fixed_content == original_content:
            LOGGER.warning("No changes detected in function replacement")
            state.update({
                "fixer_summary": "Nenhuma mudança detectada",
                "fix_failed": True,
            })
            return state

        # Generate the patch
        candidate_patch = self._generate_patch(original_content, fixed_content, diff_path)
        
        if not candidate_patch or "@@" not in candidate_patch:
            LOGGER.error("Failed to generate valid patch")
            state.update({
                "fixer_summary": "Falha ao gerar patch válido",
                "fix_failed": True,
            })
            return state

        state["patch"] = candidate_patch
        state.update({
            "fixer_summary": f"Patch gerado para função {target_function_name}",
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
        lines = []
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
        
        candidate = self.repo_root / cleaned
        if candidate.exists():
            return candidate.resolve()
        
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

    def _extract_function_by_name(self, content: str, function_name: str) -> Optional[str]:
        """Extract function by name from file content."""
        lines = content.splitlines()
        function_start = None
        
        # Find function definition
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(('def ', 'async def ')):
                match = re.match(r'(?:async\s+)?def\s+(\w+)', stripped)
                if match and match.group(1) == function_name:
                    function_start = i
                    break
        
        if function_start is None:
            return None
        
        # Find function end
        indent_level = len(lines[function_start]) - len(lines[function_start].lstrip())
        function_end = len(lines)
        
        for i in range(function_start + 1, len(lines)):
            line = lines[i]
            if line.strip():  # Non-empty line
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= indent_level:
                    function_end = i
                    break
        
        function_lines = lines[function_start:function_end]
        return "\n".join(function_lines)
    
    def _extract_function_by_line(self, content: str, line_number: int) -> tuple[Optional[str], Optional[str]]:
        """Extract function containing the given line number."""
        lines = content.splitlines()
        if line_number <= 0 or line_number > len(lines):
            return None, None

        # Find function start by looking backwards
        function_start = None
        function_name = None
        
        for i in range(line_number - 1, -1, -1):
            if i < len(lines):
                stripped = lines[i].strip()
                if stripped.startswith(('def ', 'async def ')):
                    function_start = i
                    # Extract function name
                    match = re.match(r'(?:async\s+)?def\s+(\w+)', stripped)
                    if match:
                        function_name = match.group(1)
                    break

        if function_start is None:
            return None, None

        # Find function end
        indent_level = len(lines[function_start]) - len(lines[function_start].lstrip())
        function_end = len(lines)
        
        for i in range(function_start + 1, len(lines)):
            line = lines[i]
            if line.strip():  # Non-empty line
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= indent_level:
                    function_end = i
                    break

        function_lines = lines[function_start:function_end]
        return "\n".join(function_lines), function_name

    def _replace_function_in_file(self, original_content: str, old_function: str, new_function: str) -> str:
        """Replace the old function with the new function in the file content."""
        # Normalize whitespace for comparison
        old_normalized = "\n".join(line.rstrip() for line in old_function.splitlines())
        
        # Try exact replacement first
        if old_normalized in original_content:
            return original_content.replace(old_normalized, new_function.strip(), 1)
        
        # Try fuzzy replacement by finding function boundaries
        lines = original_content.splitlines()
        old_lines = old_function.splitlines()
        
        if not old_lines:
            return original_content
            
        # Find the function definition line
        first_line = old_lines[0].strip()
        
        for i, line in enumerate(lines):
            if line.strip() == first_line:
                # Found potential start, check if it matches
                match_lines = min(len(old_lines), len(lines) - i)
                if all(lines[i + j].strip() == old_lines[j].strip() for j in range(match_lines)):
                    # Replace the function
                    new_lines = lines[:i] + new_function.splitlines() + lines[i + len(old_lines):]
                    return "\n".join(new_lines)
        
        return original_content

    def _clean_code_response(self, response: str) -> str:
        cleaned = response.strip()
        if not cleaned:
            return ""

        # Remove end tokens and HTML entities
        cleaned = cleaned.replace("<|im_end|]>", "").strip()
        cleaned = cleaned.replace("&gt;", ">").replace("&lt;", "<").replace("&quot;", '"')

        # Extract from ```python blocks
        fence_pattern = re.compile(r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
        match = fence_pattern.search(cleaned)
        
        if match:
            return match.group(1).strip()
        
        return cleaned

    def _is_valid_python_code(self, code: str) -> tuple[bool, Optional[str]]:
        if not code.strip():
            return False, "código vazio"
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as exc:
            return False, f"{exc.msg} (linha {exc.lineno})"

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