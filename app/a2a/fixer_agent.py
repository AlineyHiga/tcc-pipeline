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
from typing import Iterable, List, Optional, Union

from langchain_core.prompts import ChatPromptTemplate

from app.a2a.protocol import State
from app.llm_client import LLMClient
from app.utils import with_line_numbers

LOGGER = logging.getLogger(__name__)

REMOVAL_KEYWORD_PATTERN = re.compile(
    r"\b(remove|remover|remova|delete|deletar|eliminate|eliminar|drop|excluir|exclua)\b",
    re.IGNORECASE,
)

SYSTEM_PROMPT = """
You are the programmer. Consume the context prepared by the Requester and return the full Python file with the requested corrections applied.
Instructions:
- Analyse the SonarQube issue that was reported.
- Fix ONLY the specific problem that is mentioned.
- Keep all unaffected code exactly as it is; make the minimal edits necessary to resolve the issue.
- Return ONLY the final Python source inside a single ```python``` block (no explanations, diffs, ou texto adicional). O bloco deve conter o arquivo completo, linha por linha, incluindo partes que não foram alteradas; nunca use reticências, placeholders ou comentários como “restante inalterado”.
- Preserve indentation and formatting.
- Preserve every existing top-level function and class along with their original names; do not remove or rename them unless the issue explicitly requires it. You may add new helpers if needed, mas mantenha tudo o que já existe.
- Ignore any line numbers in the provided source and return code sem numeração.
- do not change the names of the functions variable
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
                    "system",
                    "You will receive a SonarQube issue and must reply with the fully corrected Python file.\n"
                    "File: {target_path}\n"
                    "Existing top-level definitions (they must remain and keep the same names): {required_definitions}\n"
                    "\n{requester_context}\n\n"
                    "Original code (line numbers included for reference):\n{original_code}\n\n"
                    "Return only one ```python``` block containing the entire file with the applied fix. "
                    "Do not include line numbers in your response. "
                    "Do not change the names of the functions. "
                    "You may add new helper definitions if needed, but never remove or rename the existing ones unless explicitly instructed. "
                    "The block must include the entire file content, without ellipses or omitted sections. "
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
        original_with_numbers = with_line_numbers(original_content)
        original_defs = self._collect_top_level_defs(original_content)
        required_definitions = ", ".join(sorted(original_defs)) if original_defs else "(nenhuma)"

        issues_block = self._render_issue_list(issues_for_file or ([issue] if issue else []))
        issue_rules = [
            getattr(item, "rule", "")
            for item in (issues_for_file or [])
            if getattr(item, "rule", "")
        ]

        LOGGER.info("Fixer gerando patch único para %s", diff_path)
        
        # Use enhanced RAG for minimal targeted context
        try:
            from app.rag_builder import auto_build_rag_index
            auto_build_rag_index(self.repo_root)
            
            from rag_service.service import RAGService
            rag_service = RAGService(str(self.repo_root / ".rag_index"))
            
            if issues_for_file:
                issue = issues_for_file[0]
                result = rag_service.retrieve_for_issue(
                    file_path=file_hint,
                    line=getattr(issue, 'line', 1),
                    rule=getattr(issue, 'rule', ''),
                    message=getattr(issue, 'message', ''),
                    k=3  # Minimal context
                )
                
                # Get only the target symbol code (most relevant)
                target_id = f"{result['target']['path']}::{result['target']['symbol']}"
                code_map = rag_service.get_code_for_symbols([target_id])
                
                if code_map:
                    target_code = list(code_map.values())[0]
                    code_context = f"# Target function: {result['target']['symbol']}\n{target_code[:1200]}"
                    LOGGER.info(f"Fixer using RAG context: {len(target_code)} chars for {result['target']['symbol']}")
                else:
                    code_context = original_with_numbers[:1200]
            else:
                code_context = original_with_numbers[:2000]
                
        except (ImportError, Exception) as e:
            LOGGER.debug("Enhanced RAG not available, using optimization fallback: %s", e)
            # Apply token optimizations
            try:
                from app.optimizations import TokenOptimizer
                optimizer = TokenOptimizer()
                
                # Use targeted region if we have issues with line numbers
                if issues_for_file and hasattr(issues_for_file[0], 'line'):
                    first_issue_line = getattr(issues_for_file[0], 'line', 1)
                    code_context = optimizer.extract_target_region(original_content, first_issue_line)
                    
                    # If the extracted region is too small, expand it
                    if len(code_context) < 500:
                        code_context = original_with_numbers[:2000]  # Reduced context
                else:
                    code_context = original_with_numbers[:2000]  # Reduced limit
            except ImportError:
                code_context = original_with_numbers[:2000]  # Reduced fallback
        
        prompt_input = {
            "issue_rule": ", ".join(dict.fromkeys(issue_rules)) or getattr(issue, "rule", ""),
            "issue_message": issues_block,
            "target_path": diff_path,
            "required_definitions": required_definitions,
            "requester_context": (context or "(Contexto indisponível)")[:600],  # Further limit context
            "original_code": code_context[:1500],  # Limit code context
        }
        prompt_value = self.prompt.format_prompt(**prompt_input)
        user_prompt = prompt_value.to_messages()[0].content
        LOGGER.debug("Prompt size sent to LLM: %d chars", len(user_prompt))
        raw_response = self.llm.invoke(SYSTEM_PROMPT, user_prompt)
        LOGGER.debug("Raw LLM response (first 1k chars): %s", raw_response[:1000])
        
        # Parse the changes format response
        changes = self._parse_changes_response(raw_response)
        if not changes:
            LOGGER.error("Failed to parse changes from LLM response")
            state["patch"] = ""
            state.update({
                "fixer_summary": "Falha ao interpretar resposta do LLM",
                "fix_failed": True,
            })
            return state
        
        # Apply changes to original content
        fixed_content = self._apply_changes_to_content(original_content, changes)
        if not fixed_content:
            LOGGER.error("Failed to apply changes to original content")
            state["patch"] = ""
            state.update({
                "fixer_summary": "Falha ao aplicar mudanças no código original",
                "fix_failed": True,
            })
            return state
        LOGGER.debug("Applied changes, result size: %d chars", len(fixed_content))
        
        # Additional cleaning for HTML entities that might remain
        fixed_content = fixed_content.replace('&gt;', '>').replace('&lt;', '<').replace('&quot;', '"').replace('&amp;', '&')
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

        if self._has_placeholder_markers(fixed_content):
            LOGGER.error("Fixer output for %s contém placeholders indicando omissões.", diff_path)
            state["patch"] = ""
            state.update({
                "fixer_summary": (
                    "Fixer retornou código com trechos omitidos (ex: 'rest of the functions remain unchanged')."
                ),
                "fix_failed": True,
            })
            return state

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

        # Skip top-level definition validation - allow fixer to focus on the specific issue
        fixed_defs = self._collect_top_level_defs(fixed_content)
        extra_defs = sorted(fixed_defs - original_defs)
        if extra_defs:
            LOGGER.info(
                "Fixer output for %s adicionou novas definições de topo de arquivo: %s",
                diff_path,
                ", ".join(extra_defs),
            )

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

        cleaned = reference.strip().replace("\\", "/")
        if not cleaned:
            return None

        hints: List[str] = []
        if ":" in cleaned:
            _, suffix = cleaned.split(":", 1)
            suffix = suffix.lstrip("/")
            if suffix:
                hints.append(suffix)
        hints.append(cleaned)

        candidate_paths: List[Path] = []
        for hint in hints:
            raw = Path(hint)
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

        roots = self._candidate_roots()
        for hint in hints:
            fallback = self._search_by_suffix(roots, Path(hint))
            if fallback:
                return fallback
        return None

    def _candidate_roots(self) -> List[Path]:
        roots: List[Path] = []

        def register(path: Optional[Union[str, Path]]) -> None:
            if not path:
                return
            resolved = Path(path).expanduser().resolve()
            if resolved not in roots:
                roots.append(resolved)

        register(self.repo_root)
        for parent in self.repo_root.parents:
            register(parent)
        register(Path.cwd())
        env_root = os.getenv("A2A_REPO_ROOT")
        if env_root:
            register(env_root)
        return roots

    def _search_by_suffix(self, roots: Iterable[Path], path_hint: Path) -> Optional[Path]:
        parts = tuple(part for part in path_hint.parts if part not in {"", "."})
        if not parts:
            return None
        for root in roots:
            try:
                for candidate in root.rglob(parts[-1]):
                    rel_parts = tuple(part for part in candidate.relative_to(root).parts if part not in {"", "."})
                    if not rel_parts:
                        continue
                    if len(parts) <= len(rel_parts) and rel_parts[-len(parts) :] == parts:
                        return candidate.resolve()
            except (OSError, RuntimeError) as exc:  # noqa: BLE001
                LOGGER.debug("Fixer: falha ao buscar %s em %s (%s)", path_hint, root, exc)
                continue
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

    def _has_placeholder_markers(self, content: str) -> bool:
        """Detect placeholder phrases that indicate the file was truncated."""
        if not content:
            return False
        lowered = content.lower()
        patterns = [
            "rest of the functions remain unchanged",
            "rest of the file remains unchanged",
            "rest of the code remains unchanged",
            "omitted for brevity",
            "remaining code unchanged",
        ]
        for marker in patterns:
            if marker in lowered:
                return True
        return False

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

    def _blocked_definition_removals(
        self,
        missing_defs: List[str],
        issues: List,
        context: str,
    ) -> List[str]:
        """Return the subset of missing definitions that cannot be removed."""
        if not missing_defs:
            return []

        text_sources: List[str] = []
        for item in issues:
            message = getattr(item, "message", "")
            if message:
                text_sources.append(message)
            rule = getattr(item, "rule", "")
            if rule:
                text_sources.append(rule)
        if context:
            text_sources.append(context)

        blocked: List[str] = []
        for name in missing_defs:
            if not self._removal_requested_for(name, text_sources):
                blocked.append(name)
        return blocked

    def _removal_requested_for(self, definition_name: str, sources: Iterable[str]) -> bool:
        """Return True when any source explicitly requests removing `definition_name`."""
        if not definition_name:
            return False
        normalized_targets = {definition_name.lower()}
        if "_" in definition_name:
            normalized_targets.add(definition_name.replace("_", " ").lower())
            normalized_targets.add(definition_name.replace("_", "").lower())

        for raw in sources:
            text = (raw or "").lower()
            if not text:
                continue
            if not REMOVAL_KEYWORD_PATTERN.search(text):
                continue
            if any(target in text for target in normalized_targets):
                return True
        return False

    def _collect_top_level_defs(self, source: str) -> set[str]:
        """Collect names of top-level functions and classes."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return set()
        names: set[str] = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
        return names

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

    def _parse_changes_response(self, response: str) -> List[dict]:
        """Parse the changes format response from LLM."""
        changes = []
        
        # Clean up response first
        cleaned_response = self._clean_llm_response(response)
        LOGGER.debug("Cleaned response preview: %s", cleaned_response[:200])
        
        # Try to find ```changes blocks first
        changes_pattern = re.compile(r'```changes\s*(.*?)```', re.DOTALL | re.IGNORECASE)
        matches = changes_pattern.findall(cleaned_response)
        
        # If no ```changes blocks, try to parse direct ORIGINAL/FIXED format
        if not matches:
            LOGGER.debug("No ```changes``` blocks found, trying direct ORIGINAL/FIXED parsing")
            matches = [cleaned_response]  # Treat entire response as one block
        
        for match in matches:
            # Look for ORIGINAL: section with ```python block
            original_pattern = r'ORIGINAL:\s*```python\s*(.*?)```'
            original_match = re.search(original_pattern, match, re.DOTALL | re.IGNORECASE)
            
            # Look for FIXED: section with ```python block  
            fixed_pattern = r'FIXED:\s*```python\s*(.*?)```'
            fixed_match = re.search(fixed_pattern, match, re.DOTALL | re.IGNORECASE)
            
            if not original_match or not fixed_match:
                LOGGER.warning("Could not find ORIGINAL and FIXED sections with ```python blocks")
                LOGGER.debug("Match content: %s", match[:500])
                # Try to extract just the code from ```python blocks as fallback
                return self._parse_full_code_response(cleaned_response)
            
            original_code = original_match.group(1).strip()
            fixed_code = fixed_match.group(1).strip()
            
            if original_code and fixed_code:
                changes.append({
                    "original": original_code,
                    "fixed": fixed_code
                })
                LOGGER.debug("Parsed change: %d chars original -> %d chars fixed", 
                           len(original_code), len(fixed_code))
        
        return changes
    
    def _clean_llm_response(self, response: str) -> str:
        """Clean up LLM response by removing unwanted tokens and unescaping HTML."""
        # Remove end tokens - handle both escaped and unescaped versions
        cleaned = re.sub(r'<\|im_end\|\]>', '', response)
        cleaned = re.sub(r'&lt;\|im_end\|\]&gt;', '', cleaned)
        
        # Unescape HTML entities
        cleaned = cleaned.replace('&quot;', '"').replace('&gt;', '>').replace('&lt;', '<').replace('&amp;', '&')
        
        return cleaned.strip()
    
    def _parse_full_code_response(self, response: str) -> List[dict]:
        """Parse response that contains full fixed code instead of changes format."""
        LOGGER.debug("Trying to parse full code response")
        
        # Extract code from ```python blocks
        code_block = self._clean_code_response(response)
        if not code_block:
            LOGGER.error("No code block found in response")
            return []
        
        # Return as a single "change" that replaces entire content
        return [{
            "original": "FULL_FILE_REPLACEMENT",  # Special marker
            "fixed": code_block
        }]
    
    def _apply_changes_to_content(self, original_content: str, changes: List[dict]) -> str:
        """Apply the parsed changes to the original content."""
        if not changes:
            return ""
        
        # Handle full file replacement
        if len(changes) == 1 and changes[0]["original"] == "FULL_FILE_REPLACEMENT":
            LOGGER.debug("Applying full file replacement")
            return changes[0]["fixed"]
        
        result = original_content
        
        for change in changes:
            original_code = change["original"]
            fixed_code = change["fixed"]
            
            if original_code not in result:
                LOGGER.warning("Original code not found in file content")
                # Try fuzzy matching
                result = self._fuzzy_replace(result, original_code, fixed_code)
                if result == original_content:  # No change was made
                    LOGGER.error("Failed to apply change - original code not found")
                    return ""
                continue
            
            # Replace the original code with fixed code
            result = result.replace(original_code, fixed_code, 1)
            LOGGER.debug("Applied change: replaced %d chars with %d chars", 
                        len(original_code), len(fixed_code))
        
        return result

    def _fuzzy_replace(self, content: str, original: str, fixed: str) -> str:
        """Try to find and replace code with fuzzy matching."""
        # Remove leading/trailing whitespace and normalize
        original_lines = [line.strip() for line in original.splitlines() if line.strip()]
        content_lines = content.splitlines()
        
        # Try to find a sequence of lines that match
        for i in range(len(content_lines) - len(original_lines) + 1):
            match_lines = [content_lines[i + j].strip() for j in range(len(original_lines))]
            
            if match_lines == original_lines:
                # Found a match, replace the lines
                new_lines = content_lines[:i] + fixed.splitlines() + content_lines[i + len(original_lines):]
                return "\n".join(new_lines)
        
        return content  # Return original if no match found

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
