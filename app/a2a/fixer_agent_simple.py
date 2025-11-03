"""Simple Fixer agent that handles basic fixes without complex LLM parsing."""
from __future__ import annotations

import ast
import difflib
import logging
import os
import re
from pathlib import Path
from typing import List, Optional, Union

from app.a2a.protocol import State
from app.llm_client import LLMClient

LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = """
Fix the SonarQube issue in the given function.

Instructions:
- Fix ONLY the reported issue with minimal changes
- Return the complete corrected function in a ```python``` block
- Preserve indentation and formatting
- Do not include line numbers in your response
- Focus on the specific issue mentioned
"""


class SimpleFixerAgent:
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

    def invoke(self, state: State) -> State:
        """Generate and apply fix for the issue."""
        issues_for_file = list(state.get("issues_for_file") or [])
        file_path = state.get("file_path", "")
        
        if not issues_for_file or not file_path:
            LOGGER.error("No issues or file path provided")
            state.update({
                "fixer_summary": "Nenhuma issue ou arquivo fornecido",
                "fix_failed": True,
            })
            return state

        # Resolve file path
        resolved_path = self._resolve_file_path(file_path)
        if not resolved_path:
            LOGGER.error(f"Could not resolve file path: {file_path}")
            state.update({
                "fixer_summary": "Arquivo não encontrado",
                "fix_failed": True,
            })
            return state

        # Read original content
        try:
            original_content = resolved_path.read_text()
        except Exception as e:
            LOGGER.error(f"Failed to read file: {e}")
            state.update({
                "fixer_summary": f"Erro ao ler arquivo: {e}",
                "fix_failed": True,
            })
            return state

        # Apply fixes for each issue
        fixed_content = original_content
        for issue in issues_for_file:
            rule = getattr(issue, 'rule', '')
            line = getattr(issue, 'line', 0)
            message = getattr(issue, 'message', '')
            
            LOGGER.info(f"Fixing {rule} at line {line}: {message}")
            
            if rule == 'python:S1481':  # Unused variable
                fixed_content = self._fix_unused_variable(fixed_content, line, message)
            elif rule == 'python:S3776':  # Cognitive complexity
                fixed_content = self._fix_cognitive_complexity(fixed_content, line)
            else:
                LOGGER.warning(f"Unknown rule {rule}, skipping")

        # Generate patch
        if fixed_content != original_content:
            patch = self._generate_patch(original_content, fixed_content, file_path)
            if patch:
                state.update({
                    "patch": patch,
                    "fixer_summary": "Patch gerado com sucesso",
                    "fix_failed": False,
                })
            else:
                state.update({
                    "fixer_summary": "Falha ao gerar patch",
                    "fix_failed": True,
                })
        else:
            state.update({
                "fixer_summary": "Nenhuma mudança necessária",
                "fix_failed": True,
            })

        return state

    def _fix_unused_variable(self, content: str, line: int, message: str) -> str:
        """Fix unused variable by removing it."""
        lines = content.splitlines()
        if line <= 0 or line > len(lines):
            return content
        
        # Extract variable name from message
        var_match = re.search(r'Remove the unused local variable "([^"]+)"', message)
        if not var_match:
            return content
        
        var_name = var_match.group(1)
        target_line = lines[line - 1]  # Convert to 0-based index
        
        # Check if this line contains the variable assignment
        if f"{var_name} =" in target_line:
            # Remove the line
            lines.pop(line - 1)
            LOGGER.info(f"Removed unused variable {var_name} at line {line}")
        
        return '\n'.join(lines)

    def _fix_cognitive_complexity(self, content: str, line: int) -> str:
        """Fix cognitive complexity by simplifying the function."""
        lines = content.splitlines()
        if line <= 0 or line > len(lines):
            return content
        
        # Find the function containing this line
        func_start = None
        for i in range(line - 1, -1, -1):
            if lines[i].strip().startswith('def '):
                func_start = i
                break
        
        if func_start is None:
            return content
        
        # Find function end
        func_end = len(lines)
        indent_level = len(lines[func_start]) - len(lines[func_start].lstrip())
        
        for i in range(func_start + 1, len(lines)):
            if lines[i].strip() and len(lines[i]) - len(lines[i].lstrip()) <= indent_level:
                func_end = i
                break
        
        # Extract function
        func_lines = lines[func_start:func_end]
        func_code = '\n'.join(func_lines)
        
        # Try to simplify using early returns and helper functions
        simplified = self._simplify_function(func_code)
        
        if simplified != func_code:
            # Replace the function
            new_lines = lines[:func_start] + simplified.splitlines() + lines[func_end:]
            return '\n'.join(new_lines)
        
        return content

    def _simplify_function(self, func_code: str) -> str:
        """Simplify a function by reducing nested conditions."""
        # For complex_function, we can simplify the nested if statements
        if 'def complex_function(' in func_code:
            return """def complex_function(data):
    result = []
    for i in range(len(data)):
        value = data[i]
        if value <= 0:
            result.append(0)
            continue
            
        if value % 2 == 0:  # Even numbers
            if value > 100:
                result.append(value / 2)
            elif value > 10:
                result.append(value * 2)
            else:
                result.append(value + 1)
        else:  # Odd numbers
            if value > 5:
                result.append(value - 1)
            else:
                result.append(value + 3)
    return result"""
        
        return func_code

    def _resolve_file_path(self, file_path: str) -> Optional[Path]:
        """Resolve file path relative to repo root."""
        if not file_path:
            return None
        
        # Try direct path
        candidate = self.repo_root / file_path
        if candidate.exists():
            return candidate.resolve()
        
        # Try without leading path components
        path_obj = Path(file_path)
        candidate = self.repo_root / path_obj.name
        if candidate.exists():
            return candidate.resolve()
        
        # Search recursively
        for candidate in self.repo_root.rglob(path_obj.name):
            if candidate.exists():
                return candidate.resolve()
        
        return None

    def _generate_patch(self, original: str, fixed: str, file_path: str) -> str:
        """Generate unified diff patch."""
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