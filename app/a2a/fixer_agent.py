"""Fixer agent for generating minimal code patches."""
import re
import os
import ast
import hashlib
import difflib
import json
import textwrap
import time
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from ..llm_client import get_llm_client
from ..rag.retriever import RAGRetriever
from ..codemod_apply import apply_spec
from ..logging_setup import log_event


def _extract_signature_block(func_src: str, func_name: str) -> Tuple[List[str], str, str, str]:
    """Extract decorators, signature line, base indent and body indent from original function."""
    decorators: List[str] = []
    signature_line = ""
    base_indent = ""
    body_indent = ""
    lines = func_src.splitlines()

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped and not signature_line:
            continue

        if not signature_line:
            if stripped.startswith("@"):
                decorators.append(line.rstrip())
                continue

            if stripped.startswith("def ") or stripped.startswith("async def "):
                if re.search(rf"\b{re.escape(func_name)}\b", stripped):
                    signature_line = line.rstrip()
                    base_indent = line[: len(line) - len(line.lstrip())]
                    # Infer body indent from first non-empty line after signature
                    for body_line in lines[idx + 1 :]:
                        body_stripped = body_line.strip()
                        if not body_stripped:
                            continue
                        body_indent = body_line[: len(body_line) - len(body_line.lstrip())]
                        break
                    if not body_indent:
                        body_indent = base_indent + "    "
                    return decorators, signature_line, base_indent, body_indent
                else:
                    # Skip nested/unexpected defs; continue scanning
                    continue
        else:
            break

    if not signature_line:
        raise ValueError("could not extract original function signature")

    if not body_indent:
        body_indent = base_indent + "    "

    return decorators, signature_line, base_indent, body_indent


def _extract_func_from_llm(txt: str, func_name: str, original_src: str | None = None) -> str:
    """Extract function from LLM response with multiple fallbacks."""
    original_text = txt.strip()
    sentinel_re = re.compile(r"<<(?:FUNC_NEW|END_FUNC(?:_NEW)?)>>")
    # 1) Marcadores
    m = re.search(r"<<FUNC_NEW>>(.*?)(?:<<END_FUNC>>|<<END_FUNC_NEW>>|$)", txt, re.S)
    cand = m.group(1).strip() if m else ""

    # 2) Bloco ```python
    if not cand:
        m = re.search(r"```(?:python)?\s*(.*?)```", txt, re.S | re.I)
        cand = m.group(1).strip() if m else ""

    # 3) Varre texto cru por def <func_name>(...):
    if not cand or f"def {func_name}" not in cand:
        pat = rf"(?ms)^\s*def\s+{re.escape(func_name)}\s*\([^)]*\):.*?(?=^\s*def\s+|\Z)"
        m = re.search(pat, txt)
        cand = m.group(0).strip() if m else ""

    cand = cand.strip().lstrip("`").rstrip("`").strip()
    cand = sentinel_re.sub("", cand)
    if not cand and original_text:
        cand = sentinel_re.sub("", original_text)
    if not cand:
        raise ValueError("LLM response is empty")

    pattern = rf"^\s*(?:async\s+)?def\s+{re.escape(func_name)}\s*\("
    if re.search(pattern, cand, re.MULTILINE):
        cand = cand.rstrip()
        return cand + ("\n" if not cand.endswith("\n") else "")

    if original_src is None:
        raise ValueError("LLM response did not contain a function body")

    decorators, signature_line, _base_indent, body_indent = _extract_signature_block(original_src, func_name)

    body_text = sentinel_re.sub("", cand).strip("\n")
    if not body_text:
        body_text = "pass"

    dedented_body = textwrap.dedent(body_text).rstrip()
    if not dedented_body:
        dedented_body = "pass"

    dedented_body += "\n"
    indented_body = textwrap.indent(dedented_body, body_indent)

    header_lines = []
    if decorators:
        header_lines.extend(decorators)
    header_lines.append(signature_line.rstrip())
    header = "\n".join(header_lines)

    func_block = f"{header}\n{indented_body}"
    return func_block if func_block.endswith("\n") else func_block + "\n"


def build_unified_diff(path: str, old_text: str, new_text: str) -> str:
    """Generate unified diff locally using difflib."""
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        n=3
    )
    return "".join(diff)


def validate_diff(text: str) -> tuple[bool, str | None]:
    """Validate diff text for obvious issues."""
    if "..." in text:  # típico de preview do LLM
        return False, "ellipsis_in_diff"
    if "--- a/" not in text or "+++ b/" not in text or "\n@@ " not in text:
        return False, "missing_headers_or_hunks"
    return True, None
try:
    from unidiff import PatchSet
except ImportError:
    PatchSet = None
try:
    import libcst as cst
    
    class PrintToLoggerTransformer(cst.CSTTransformer):
        """Transform print() calls to logger calls."""
        
        def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.Call:
            if isinstance(updated_node.func, cst.Name) and updated_node.func.value == "print":
                return updated_node.with_changes(
                    func=cst.Attribute(
                        value=cst.Name("logger"),
                        attr=cst.Name("info")
                    )
                )
            return updated_node
    
except ImportError:
    cst = None
    PrintToLoggerTransformer = None


class FixerAgent:
    """Generates minimal unified diffs to fix Sonar issues."""
    
    def __init__(self):
        self.llm = get_llm_client()
        self.retriever = RAGRetriever()
        self.last_llm_response: str = ""
        self.fallback_model = os.getenv("LLM_FALLBACK_MODEL")
        try:
            self.max_function_rewrite_attempts = max(
                1, int(os.getenv("FUNC_REWRITE_MAX_ATTEMPTS", "3"))
            )
        except ValueError:
            self.max_function_rewrite_attempts = 3

    def _llm_generate(
        self,
        prompt: str | None = None,
        messages: List[Dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str:
        """Wrapper that records the last LLM response."""
        model = kwargs.get("model") or getattr(self.llm, "model", "")
        payload_text = ""
        if prompt:
            payload_text = prompt
        elif messages:
            payload_text = "\n\n".join(
                f"{msg.get('role', 'user')}:{msg.get('content', '')}" for msg in messages
            )

        if payload_text:
            payload_preview = payload_text[:400]
            log_event(
                "fixer.llm.request",
                model=model,
                prompt_sha=self._hash_text(payload_text),
                prompt_preview=payload_preview,
                prompt_len=len(payload_text),
                temperature=kwargs.get("temperature"),
                max_tokens=kwargs.get("max_tokens"),
            )

        response = self.llm.generate(prompt=prompt, messages=messages, **kwargs)
        self.last_llm_response = response or ""
        if response:
            log_event(
                "fixer.llm.response",
                model=model,
                content_sha=self._hash_text(response),
                content_preview=response[:400],
                content_len=len(response),
            )
        else:
            log_event(
                "fixer.llm.response",
                model=model,
                content_sha="",
                content_preview="",
                content_len=0,
                note="empty_response",
            )
        return response
    
    def fix_plan(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate deterministic fix plan from Sonar issues and AST analysis."""
        current_lot = state.get("current_lot", {})
        repo_root = state.get("repo_path", ".")
        
        rule_key = current_lot.get("ruleKey", "")
        issues = current_lot.get("issues", [])
        
        if not issues:
            return {"goals": [], "steps": [], "touched_files": [], "targets": []}
        
        # Build deterministic targets from issues (sorted by severity + gap)
        targets = self._build_fix_goals(repo_root, current_lot)
        
        # Filter touched_files to only real files (no globs)
        touched_files = []
        for issue in issues:
            component = issue.get("component", "")
            file_path = self._component_rel(component)
            if file_path:
                # Check if file exists
                full_path = Path(repo_root) / file_path if not Path(file_path).is_absolute() else Path(file_path)
                if full_path.exists() and full_path.is_file():
                    touched_files.append(str(Path(file_path).as_posix()))
        
        # Generate goals text from targets (not LLM)
        goals_text = []
        for t in targets:
            if t.get("from_to") and t["from_to"][0] is not None:
                goals_text.append(f"Refactor {t['file']}::{t['func_name']} from {t['from_to'][0]} to {t['from_to'][1]} ({t['rule']})")
            else:
                goals_text.append(f"Fix {t['rule']} in {t['file']}::{t['func_name']}")
        
        # Generate summary from targets
        summary = []
        for t in targets:
            if t.get("from_to") and t["from_to"][0] is not None:
                summary.append(f"{t['file']}::{t['func_name']} {t['from_to'][0]}→{t['from_to'][1]}")
            else:
                summary.append(f"{t['file']}::{t['func_name']}")
        
        return {
            "targets": targets,
            "summary": summary,
            "goals": goals_text,
            "steps": ["Apply minimal changes to resolve issue"],
            "touched_files": touched_files
        }
    
    def generate_patch(self, state: Dict[str, Any]) -> str:
        """Generate unified diff patch using semantic specs with safe fallback."""
        import logging
        from ..logging_setup import log_event

        logger = logging.getLogger(__name__)

        fix_plan = state.get("fix_plan", {})
        current_lot = state.get("current_lot", {})
        rag_ctx = state.get("rag_ctx", {})

        if not fix_plan or not current_lot:
            return ""

        targets = fix_plan.get("targets", [])
        if not targets:
            logger.warning("No targets in fix_plan")
            return ""

        target = targets[0]
        log_event(
            "fix.select",
            file=target["file"],
            func=target["func_name"],
            from_to=target["from_to"],
            reason="highest_priority",
        )

        rule_key = current_lot.get("ruleKey", "")
        repo_root = state.get("repo_path", ".")
        budget = current_lot.get("budget", 300)

        # Read only the target file
        file_contents: Dict[str, str] = {}
        target_file = target["file"]
        try:
            full_path = (
                Path(repo_root) / target_file
                if not Path(target_file).is_absolute()
                else Path(target_file)
            )
            content = full_path.read_text(encoding="utf-8")
            file_contents[target_file] = content
            logger.debug(f"Read {len(content)} chars from {target_file}")
        except Exception as e:
            logger.error(f"Error reading target file {target_file}: {e}")
            return ""

        # 1) Primary path: semantic codemod spec
        try:
            spec_patch = self._try_semantic_codemod(
                target,
                file_contents,
                repo_root,
                budget,
                current_lot,
                rag_ctx,
                rule_key,
            )
            if spec_patch:
                self._log_patch_generation("semantic_spec", spec_patch, target_file)
                return spec_patch
        except Exception as e:
            logger.warning(f"Semantic codemod failed: {e}")

        # 2) Function-level rewrite with AST swap
        if target.get("func_name"):
            func_patch = self._try_function_rewrite_target(target, file_contents, repo_root)
            if func_patch:
                error = self._enforce_allowlist_and_budget(func_patch, budget, current_lot)
                if error:
                    print(f"Function rewrite rejected: {error}")
                else:
                    return func_patch

        # 3) Mechanical codemod fallback (non-LLM)
        fallback_patch = self._fallback_codemod(target["file"], rule_key)
        if fallback_patch:
            fallback_error = self._enforce_allowlist_and_budget(fallback_patch, budget, current_lot)
            if fallback_error:
                print(f"Mechanical fallback rejected: {fallback_error}")
                return ""
            self._log_patch_generation("mechanical_codemod", fallback_patch, target_file)
            return fallback_patch

        return ""

    def _try_semantic_codemod(
        self,
        target: Dict[str, Any],
        file_contents: Dict[str, str],
        repo_root: str,
        budget: int,
        current_lot: Dict[str, Any],
        rag_ctx: Dict[str, Any],
        rule_key: str,
    ) -> str:
        """Attempt to build a diff from a semantic codemod spec."""
        prompt = self._build_spec_prompt(target, file_contents, rag_ctx, rule_key)
        response = self._llm_generate(
            prompt=prompt,
            max_tokens=800,
            temperature=0,
        )

        spec = self._parse_spec_response(response)
        if not spec:
            return ""

        # Limit spec to the current target file for safety.
        target_file = target["file"]
        files = spec.get("files", [])
        if not files:
            return ""

        def _normalize_path(path: str) -> str:
            return str(Path(path).as_posix())

        normalized_target = _normalize_path(target_file)
        filtered_files = [
            entry
            for entry in files
            if _normalize_path(entry.get("path", "")) == normalized_target
        ]
        if not filtered_files:
            print("Spec ignored: no operations for target file")
            return ""
        safe_spec = {"files": []}
        for entry in filtered_files:
            try:
                sanitized_ops = self._sanitize_replace_function_ops(
                    entry.get("operations", []),
                    entry.get("path", target_file),
                )
            except ValueError as exc:
                log_event(
                    "fixer.spec.invalid",
                    file=target_file,
                    error=str(exc),
                )
                return ""
            safe_spec["files"].append(
                {
                    "path": entry["path"],
                    "operations": sanitized_ops,
                }
            )

        updated_sources = apply_spec(safe_spec, repo_root=repo_root, original_sources=file_contents)
        if not updated_sources:
            return ""

        diffs: List[str] = []
        for rel_path, new_text in updated_sources.items():
            original = file_contents.get(rel_path)
            if original is None:
                original_path = Path(repo_root) / rel_path
                original = original_path.read_text(encoding="utf-8")
            if original == new_text:
                continue
            diff = build_unified_diff(rel_path, original, new_text)
            if diff:
                diffs.append(diff)

        if not diffs:
            return ""

        patch = "\n".join(diffs)
        ok, why = validate_diff(patch)
        if not ok:
            print(f"Semantic codemod diff invalid: {why}")
            return ""

        error = self._enforce_allowlist_and_budget(patch, budget, current_lot)
        if error:
            print(f"Semantic codemod rejected: {error}")
            return ""

        return patch

    def _build_spec_prompt(
        self,
        target: Dict[str, Any],
        file_contents: Dict[str, str],
        rag_ctx: Dict[str, Any],
        rule_key: str,
    ) -> str:
        """Build prompt requesting a JSON spec for deterministic codemod."""
        file_path = target["file"]
        func_name = target.get("func_name", "unknown")
        content = file_contents.get(file_path, "")

        if len(content) > 2400:
            content = content[:2400] + "\n# … truncated for prompt"

        few_shots = rag_ctx.get("few_shots", [])
        contexts = rag_ctx.get("contexts", [])
        examples = "\n\n".join(few_shots[:2]) if few_shots else "\n".join(contexts[:1])

        spec_schema = json.dumps(
            {
                "files": [
                    {
                        "path": file_path,
                        "operations": [
                            {
                                "type": "replace_function",
                                "name": func_name,
                                "signature": "def sample(arg):",
                                "decorators": ["@example"],
                                "body": [
                                    "if condition:",
                                    "    return value",
                                    "return fallback",
                                ],
                            }
                        ],
                    }
                ]
            },
            indent=2,
        )

        sentinel = "<<END_SPEC>>"

        return f"""You are a deterministic refactoring planner. Respond ONLY with valid JSON followed by the sentinel `{sentinel}`.

Sonar rule: {rule_key}
Target file: {file_path}
Target function: {func_name}

Current file content (read-only):
```python
{content}
```

Reference fixes / guidance:
{examples or '(none)'}

Schema (example values, keep same structure):
{spec_schema}

Rules:
- Output valid JSON matching the schema above (list of files → operations).
- Use operation type "replace_function" and provide the exact `signature`, optional `decorators`, and a `body` list with each statement as a string.
- Keep imports/decorators intact unless they violate the rule. Preserve the existing signature unless a change is required by the rule.
- Limit modifications to the target file only.
- Do not include markdown fences, commentary or explanatory text.

Finish the response with the sentinel on a separate line: {sentinel}
If no change is needed return: {{"files": []}}
Do not output anything after the sentinel."""

    def _parse_spec_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON spec from LLM response."""
        if not response:
            return {}

        text = response.strip()
        if "<<END_SPEC>>" in text:
            text = text.split("<<END_SPEC>>", 1)[0].strip()
        text = text.replace("```json", "```")

        candidates = []
        fence_pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
        matches = fence_pattern.findall(text)
        if matches:
            candidates.extend(match.strip() for match in matches if match.strip())
        else:
            candidates.append(text)

        for candidate in candidates:
            try:
                data = json.loads(candidate)
                if isinstance(data, dict) and "files" in data:
                    return data
            except json.JSONDecodeError:
                continue

        return {}
    
    def _validate_and_reconstruct_diff(self, text: str, target_file: str) -> str:
        """Validate and reconstruct diff with proper headers."""
        t = text.strip()
        
        # Remove markdown fences
        t = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", t, flags=re.M)
        
        # Early validation: must have @@ hunks
        if "@@" not in t:
            print("Diff validation failed: no hunks found")
            return ""
        
        if not PatchSet:
            print("Warning: unidiff not available, using raw diff")
            return t
        
        try:
            # Try to parse as-is first
            ps = PatchSet(t.splitlines(keepends=True))
            
            # Validate at least one valid hunk
            total_hunks = sum(len(p) for p in ps)
            if total_hunks == 0:
                print("Diff validation failed: no valid hunks parsed")
                return ""
            
            # If no git headers, reconstruct them
            if "diff --git " not in t:
                rebuilt = []
                for p in ps:
                    a = f"a/{p.path}"
                    b = f"b/{p.path}"
                    rebuilt.append(f"diff --git {a} {b}\n--- {a}\n+++ {b}\n")
                    for h in p:
                        rebuilt.append(str(h))
                t = "".join(rebuilt) if rebuilt else t
            
            # Final validation
            PatchSet(t.splitlines(keepends=True))
            return t
            
        except Exception as e:
            print(f"Diff validation failed: {e}")
            return ""  # Return empty to trigger fallback
    
    def _enforce_allowlist_and_budget(self, patch_diff: str, budget: int = None, current_lot: dict = None) -> str:
        """Enforce allowlist and budget constraints."""
        if not PatchSet or not patch_diff:
            return ""
        
        try:
            ps = PatchSet(patch_diff.splitlines(keepends=True))
            touched = {p.path for p in ps}
            
            # Check allowlist
            allowed_prefixes = ("src/", "tests/", "tests_prop/")
            if not all(p.startswith(allowed_prefixes) for p in touched):
                return f"patch touches disallowed paths: {touched}"
            
            # Check lot-specific files
            if current_lot:
                issue_paths = set()
                for issue in current_lot.get("issues", []):
                    component = issue.get("component", "")
                    if ":" in component:
                        path = component.split(":", 1)[1]
                    else:
                        path = component
                    if path:
                        issue_paths.add(path)
                
                if issue_paths and not touched.issubset(issue_paths):
                    return f"patch touches files outside current lot: {touched - issue_paths}"
            
            # Check budget if provided
            if budget is not None:
                loc_changed = sum(h.added + h.removed for p in ps for h in p)
                if loc_changed > budget:
                    return f"LOC budget exceeded ({loc_changed}>{budget})"
            
            return ""  # No errors
        except Exception as e:
            return f"Patch validation failed: {e}"
    
    def _fallback_codemod(self, file_path: str, rule_key: str) -> str:
        """Apply mechanical refactoring for common patterns."""
        if not Path(file_path).exists():
            return ""
        
        try:
            with open(file_path, 'r') as f:
                original = f.read()
            
            modified = original
            
            # S106: Replace print() with logger
            if rule_key == "python:S106":
                modified = self._fix_print_statements(modified)
            
            # File not closed: wrap with context manager
            elif "file" in rule_key.lower() and "close" in rule_key.lower():
                modified = self._fix_file_handling(modified)
            
            # S3776: Complexity - flatten if pyramids
            elif rule_key == "python:S3776":
                modified = self._fix_complexity_s3776(modified)
            
            if modified != original:
                # Generate unified diff
                import difflib
                diff_lines = list(difflib.unified_diff(
                    original.splitlines(keepends=True),
                    modified.splitlines(keepends=True),
                    fromfile=f"a/{file_path}",
                    tofile=f"b/{file_path}"
                ))
                return "".join(diff_lines)
            
        except Exception as e:
            print(f"Codemod fallback failed: {e}")
        
        return ""
    
    def _fix_print_statements(self, code: str) -> str:
        """Replace print() with logger calls."""
        if cst is None:
            # Simple regex fallback
            if "import logging" not in code:
                code = "import logging\n" + code
            if "logger = logging.getLogger(__name__)" not in code:
                # Insert after imports
                lines = code.split('\n')
                insert_idx = 0
                for i, line in enumerate(lines):
                    if line.startswith('import ') or line.startswith('from '):
                        insert_idx = i + 1
                lines.insert(insert_idx, "logger = logging.getLogger(__name__)")
                code = '\n'.join(lines)
            
            # Replace print calls
            code = re.sub(r'print\(([^)]+)\)', r'logger.info(\1)', code)
            return code
        
        # Use libcst for more robust transformation
        try:
            tree = cst.parse_module(code)
            transformer = PrintToLoggerTransformer()
            modified_tree = tree.visit(transformer)
            return modified_tree.code
        except:
            return code
    
    def _fix_file_handling(self, code: str) -> str:
        """Wrap file operations with context managers."""
        # Simple pattern matching for open() calls
        pattern = r'(\w+)\s*=\s*open\(([^)]+)\)'
        
        def replace_open(match):
            var_name = match.group(1)
            args = match.group(2)
            return f"with open({args}) as {var_name}:"
        
        return re.sub(pattern, replace_open, code)
    
    def _fix_complexity_s3776(self, code: str) -> str:
        """Flatten complex if pyramids with guard clauses."""
        lines = code.split('\n')
        result = []
        
        for line in lines:
            # Simple heuristic: convert nested ifs to guard clauses
            stripped = line.strip()
            if stripped.startswith('if ') and line.count('    ') > 1:
                # Convert to guard clause
                condition = stripped[3:].rstrip(':')
                indent = '    ' * (line.count('    ') - 1)
                result.append(f"{indent}if not ({condition}):")
                result.append(f"{indent}    return")
            else:
                result.append(line)
        
        return '\n'.join(result)
    
    def _component_rel(self, c: str) -> str:
        """Extract relative path from component."""
        return c.split(":", 1)[-1] if ":" in c else c
    
    def _hash_text(self, text: str) -> str:
        """Compute stable hash for drift detection."""
        return hashlib.sha1(text.encode("utf-8")).hexdigest()
    
    def _log_patch_generation(self, strategy: str, patch_text: str, file_path: str) -> None:
        """Emit structured log for generated patches."""
        if not patch_text:
            return
        log_event(
            "fixer.patch.generated",
            strategy=strategy,
            file=file_path,
            diff_sha=self._hash_text(patch_text),
            diff_lines=patch_text.count("\n"),
        )
    
    def _find_function_by_name(self, text: str, func_name: str) -> Optional[tuple[str, int, int]]:
        """Locate function definition by name using AST."""
        try:
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == func_name:
                    start = getattr(node, "lineno", None)
                    end = getattr(node, "end_lineno", None)
                    if start and end:
                        return node.name, start - 1, end
        except Exception as exc:
            print(f"AST search failed for {func_name}: {exc}")
        return None
    
    def _apply_function_swap(
        self,
        file_path: str,
        original_text: str,
        func_name: str,
        new_func_src: str,
    ) -> str:
        """Apply function replacement via LibCST spec."""
        spec = {
            "files": [
                {
                    "path": file_path,
                    "operations": [
                        {
                            "type": "replace_function",
                            "name": func_name,
                            "code": new_func_src,
                        }
                    ],
                }
            ]
        }
        try:
            updated = apply_spec(spec, original_sources={file_path: original_text})
            return updated.get(file_path, "")
        except Exception as exc:
            print(f"Function swap via LibCST failed for {func_name}: {exc}")
            return ""

    def _sanitize_replace_function_ops(
        self,
        operations: List[Dict[str, Any]],
        file_path: str,
    ) -> List[Dict[str, Any]]:
        """Normalize replace_function operations to a safe code-only form."""
        sanitized: List[Dict[str, Any]] = []
        for op in operations or []:
            if op.get("type") != "replace_function":
                raise ValueError(f"unsupported operation type: {op.get('type')}")

            func_name = op.get("name")
            if not func_name:
                raise ValueError("replace_function operation missing 'name'")

            func_code = op.get("code")
            source = "code"
            if func_code:
                func_text = textwrap.dedent(str(func_code)).strip("\n") + "\n"
            else:
                signature_raw = str(op.get("signature") or "").strip()
                if not signature_raw:
                    raise ValueError("replace_function operation missing 'signature'")

                signature_lines = signature_raw.splitlines()
                signature_line = signature_lines[0].strip()
                if not signature_line.endswith(":"):
                    signature_line += ":"
                if not re.match(r"^\s*def\s+\w+\s*\(.*\)\s*:\s*$", signature_line):
                    raise ValueError(f"invalid function signature: {signature_line}")

                decorators = [
                    str(deco).strip()
                    for deco in op.get("decorators", [])
                    if str(deco or "").strip()
                ]

                body_raw = op.get("body", "")
                if isinstance(body_raw, list):
                    body_text = "\n".join(str(line) for line in body_raw)
                else:
                    body_text = str(body_raw or "")
                body_text = re.sub(r"<<(?:FUNC_NEW|END_FUNC(?:_NEW)?)>>", "", body_text)

                extra_header = "\n".join(signature_lines[1:]).strip("\n")
                if extra_header:
                    extra_header = textwrap.dedent(extra_header).strip("\n")
                    body_text = (
                        f"{extra_header}\n{body_text}" if body_text.strip() else extra_header
                    )

                body_text = textwrap.dedent(body_text).strip("\n")
                if not body_text.strip():
                    body_text = "pass"

                clean_body_lines = [line.rstrip() for line in body_text.splitlines()]
                indented_body = textwrap.indent("\n".join(clean_body_lines) + "\n", "    ")

                pieces: List[str] = []
                pieces.extend(decorators)
                pieces.append(signature_line)
                pieces.append(indented_body.rstrip("\n"))
                func_text = "\n".join(pieces) + "\n"
                source = "signature+body"

            try:
                ast.parse(func_text)
            except SyntaxError as exc:
                raise ValueError(f"invalid function code for {func_name}: {exc}") from exc

            log_event(
                "fixer.spec.sanitized",
                file=file_path,
                func=func_name,
                code_sha=self._hash_text(func_text),
                source=source,
            )
            sanitized.append(
                {
                    "type": "replace_function",
                    "name": func_name,
                    "code": func_text,
                }
            )

        return sanitized
    
    def _stable_issue_key(self, issue: Dict[str, Any]) -> str:
        """Stable identifier for issue deduplication."""
        line = issue.get("textRange", {}).get("startLine") or issue.get("line") or 0
        message = issue.get("message", "")
        rule = issue.get("rule", "")
        component = issue.get("component", "")
        message_hash = hashlib.sha1(message.encode("utf-8")).hexdigest()[:8] if message else "nomsg"
        return f"{rule}:{component}:{line}:{message_hash}"
    
    def _find_enclosing_function(self, text: str, line_no: int) -> tuple[str, int, int] | None:
        """Find function enclosing the given line number."""
        try:
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    s = getattr(node, "lineno", None)
                    e = getattr(node, "end_lineno", None)
                    if s and e and s <= line_no <= e:
                        return node.name, s-1, e  # (name, start 0-based, end)
        except:
            pass
        return None
    
    def _find_best_func_by_line(self, text: str, line_no: int):
        """Find closest function when line doesn't fall within any function."""
        try:
            tree = ast.parse(text)
            candidates = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    s = getattr(node, "lineno", None)
                    e = getattr(node, "end_lineno", None)
                    if s and e:
                        dist = 0 if (s <= line_no <= e) else min(abs(line_no-s), abs(line_no-e))
                        candidates.append((dist, node.name, s-1, e))
            return min(candidates)[1:] if candidates else None
        except:
            return None
    
    def _parse_from_to(self, msg: str) -> tuple[int, int] | None:
        """Parse complexity reduction target from message."""
        m = re.search(r'from\s+(\d+)\s+to\s+(?:the\s+)?(\d+)', msg, re.I)
        if m:
            return int(m.group(1)), int(m.group(2))
        return None
    
    def _build_fix_goals(self, repo_root: str, lot: dict) -> list[dict]:
        """Build detailed fix goals from issues."""
        goals = []
        seen_keys: set[str] = set()
        for iss in lot.get("issues", []):
            issue_key = self._stable_issue_key(iss)
            if issue_key in seen_keys:
                continue
            seen_keys.add(issue_key)
            rel = self._component_rel(iss.get("component", ""))
            path = Path(repo_root) / rel
            if not path.exists():
                continue
            
            try:
                text = path.read_text(encoding="utf-8")
                line = iss.get("textRange", {}).get("startLine") or iss.get("line")
                
                func = None
                if line:
                    func = self._find_enclosing_function(text, int(line))
                    if not func:
                        func = self._find_best_func_by_line(text, int(line))
                
                func_hash = None
                if func:
                    text_lines = text.splitlines(True)
                    func_src_text = "".join(text_lines[func[1]:func[2]])
                    func_hash = self._hash_text(func_src_text)

                frm_to = self._parse_from_to(iss.get("message", ""))
                
                goals.append({
                    "rule": iss.get("rule", ""),
                    "file": rel,
                    "func_name": func[0] if func else None,
                    "func_span": [func[1], func[2]] if func else None,
                    "func_hash": func_hash,
                    "from_to": frm_to or [None, 15],
                    "severity": iss.get("severity", "")
                })
            except Exception as e:
                print(f"Error processing issue in {rel}: {e}")
                continue
        
        # Sort by severity first, then by complexity gap (highest first)
        SEV_W = {"BLOCKER": 0, "CRITICAL": 1, "MAJOR": 2, "MINOR": 3, "INFO": 4}
        
        def gap(t):
            f, to = t.get("from_to") or (None, 15)
            return (f or 0) - (to or 15)
        
        goals.sort(key=lambda t: (SEV_W.get(t.get("severity"), 9), -gap(t)))
        return goals
    
    def _window_lines(self, text: str, start: int, end: int, k: int = 10) -> str:
        """Extract window of lines around function."""
        lines = text.splitlines(True)
        a = max(0, start - k)
        b = min(len(lines), end + k)
        return "".join(lines[a:b])
    
    def _make_unified_diff(self, rel_path: str, before: str, after: str, ctx: int = 3) -> str:
        """Generate unified diff with git headers."""
        diff_lines = list(difflib.unified_diff(
            before.splitlines(True), after.splitlines(True),
            fromfile=f"a/{rel_path}", tofile=f"b/{rel_path}",
            n=ctx, lineterm=""
        ))
        if diff_lines:
            return f"diff --git a/{rel_path} b/{rel_path}\n" + "\n".join(diff_lines)
        return ""
    
    def _merge_func(self, text: str, start: int, end: int, new_func_src: str) -> str:
        """Merge new function source into original text."""
        if not new_func_src.endswith("\n"):
            new_func_src += "\n"
        lines = text.splitlines(True)
        return "".join(lines[:start]) + new_func_src + "".join(lines[end:])
    
    def _prompt_rewrite_func(self, rule_key: str, func_name: str, ctx_text: str, func_src: str) -> str:
        """Build prompt for function rewriting (function-only, with hard format guarantees)."""
        
        def _extract_signature(src: str) -> str:
            # pega a linha 'def ...:' original, incluindo decorators se houver
            import re
            decos = ""
            for m in re.finditer(r'^\s*@\w[^\n]*\n', src, re.M):
                decos = m.group(0)
            m = re.search(r'^\s*def\s+[^(]+\([^)]*\):', src, re.M)
            if not m:
                raise ValueError("could not extract function signature")
            return (decos + m.group(0)).strip()
        
        func_sig = _extract_signature(func_src)
        
        escaped_sig = func_sig.replace('(', r'\(').replace(')', r'\)')
        
        return f"""Respond ONLY with the complete function body starting with `{func_sig}`.
Do not include comments, markdown/backticks, examples, or any text outside the code.
Enclose your entire output between `<<FUNC_NEW>>` and `<<END_FUNC>>`. Output must match:
`^<<FUNC_NEW>>\\s*{escaped_sig}[\\s\\S]*<<END_FUNC>>$`

You are a Python refactoring expert. Sonar rule: {rule_key}.
Reduce cognitive complexity of {func_name} without changing public behavior.

Constraints:
- Keep EXACT same signature (name, parameters, defaults) and decorators; do NOT convert to @staticmethod/@classmethod.
- Same return types/semantics; no I/O (no print/log/input), no new imports/globals, no changes in other functions.
- Prefer guard-clauses and small local helpers defined inside the function.
- No placeholders or ellipses (...).

Context (read-only, do not modify):
<<CTX>>
{ctx_text}
<<END_CTX>>

Original function:
<<FUNC_ORIG>>
{func_src}
<<END_FUNC_ORIG>>

Now output ONLY the replacement function body:

<<FUNC_NEW>>
{func_sig}""".rstrip()
    
    def _try_function_rewrite_target(self, target: dict, file_contents: dict, repo_root: str) -> str:
        """Try function-level rewriting with strict validation and retries."""
        if not target.get("func_name") or not target.get("func_span"):
            return ""

        file_path = target["file"]
        func_name = target["func_name"]
        start, end = target["func_span"]

        if file_path not in file_contents:
            return ""

        max_attempts = self.max_function_rewrite_attempts
        fallback_model = self.fallback_model
        total_attempts = max_attempts + (1 if fallback_model else 0)

        try:
            text = file_contents[file_path]
            lines = text.splitlines(True)
            func_src = "".join(lines[start:end])

            expected_hash = target.get("func_hash")
            current_hash = self._hash_text(func_src) if func_src else ""
            if expected_hash and current_hash and current_hash != expected_hash:
                relocated = self._find_function_by_name(text, func_name)
                if relocated:
                    start = relocated[1]
                    end = relocated[2]
                    func_src = "".join(lines[start:end])
                    target["func_span"] = [start, end]
                    current_hash = self._hash_text(func_src)
                    print(f"Function {func_name} drift detected; updated span to {start}:{end}.")
                else:
                    print(f"Function {func_name} drifted and could not be located.")
                    return ""

            if func_src:
                target["func_hash"] = self._hash_text(func_src)

            ctx_text = self._window_lines(text, start, end, k=5)

            # Build prompt for function rewrite
            prompt = self._prompt_rewrite_func(target["rule"], func_name, ctx_text, func_src)
            base_messages = [{"role": "user", "content": prompt}]

            attempt_reasons: List[str] = []

            for attempt in range(total_attempts):
                use_fallback_model = bool(fallback_model) and attempt >= max_attempts
                messages = list(base_messages)

                if attempt_reasons:
                    last_reason = attempt_reasons[-1]
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Retry #{attempt + 1}: previous response was invalid ({last_reason}). "
                                "Return ONLY the full function body between <<FUNC_NEW>> and <<END_FUNC>>."
                            ),
                        }
                    )

                request_kwargs: Dict[str, Any] = {
                    "messages": messages,
                    "temperature": 0,
                    "max_tokens": 900,
                    "stop": ["<<END_FUNC>>", "<<END_FUNC_NEW>>", "```"],
                }
                if use_fallback_model:
                    print(f"Function rewrite switching to fallback model {fallback_model}")
                    request_kwargs["model"] = fallback_model

                resp_raw = self._llm_generate(**request_kwargs).strip()

                if not resp_raw:
                    attempt_reasons.append("empty_response")
                    self._sleep_backoff(attempt, total_attempts)
                    continue

                try:
                    new_func = _extract_func_from_llm(resp_raw, func_name, func_src)
                except ValueError as e:
                    attempt_reasons.append(f"parse_error:{e}")
                    self._sleep_backoff(attempt, total_attempts)
                    continue

                valid, reason = self._validate_llm_response(resp_raw, func_name)
                if not valid:
                    log_event(
                        "fixer.llm.response.invalid_format",
                        attempt=attempt + 1,
                        reason=reason or "unknown",
                        func=func_name,
                        file=file_path,
                    )

                new_file = self._apply_function_swap(file_path, text, func_name, new_func)
                if not new_file:
                    new_file = self._merge_func(text, start, end, new_func)
                    if not new_file:
                        attempt_reasons.append("apply_swap_failed")
                        self._sleep_backoff(attempt, total_attempts)
                        continue

                patch = build_unified_diff(file_path, text, new_file)

                ok, why = validate_diff(patch)
                if not ok:
                    attempt_reasons.append(f"diff_invalid:{why}")
                    self._sleep_backoff(attempt, total_attempts)
                    continue

                if not patch:
                    attempt_reasons.append("empty_patch")
                    self._sleep_backoff(attempt, total_attempts)
                    continue

                attempt_label = (
                    f"attempt {attempt + 1} (fallback model)" if use_fallback_model else f"attempt {attempt + 1}"
                )
                print(f"Generated function-level patch for {func_name} on {attempt_label}")
                self._log_patch_generation("function_rewrite", patch, file_path)
                return patch

            if attempt_reasons:
                print(
                    f"Function rewrite exhausted retries for {func_name}: "
                    f"{'; '.join(attempt_reasons[-3:])}"
                )

        except Exception as e:
            print(f"Function rewrite failed for {func_name}: {e}")

        return ""

    def _sleep_backoff(self, attempt: int, total_attempts: int) -> None:
        """Simple exponential backoff between retries."""
        if attempt + 1 < total_attempts:
            time.sleep(min(1.5, 0.5 * (attempt + 1)))

    def _validate_llm_response(self, resp: str, func_name: str) -> tuple[bool, str | None]:
        """Validate LLM response format before extraction."""
        text = resp.strip()
        if not text:
            return False, "empty_response"
        if "```" in text:
            return False, "contains_code_fence"
        if "..." in text:
            return False, "contains_ellipsis"

        has_markers = "<<FUNC_NEW>>" in text and (
            "<<END_FUNC>>" in text or "<<END_FUNC_NEW>>" in text
        )
        if not has_markers:
            return False, "missing_markers"

        try:
            inner = text.split("<<FUNC_NEW>>", 1)[1]
            inner = inner.rsplit("<<END_FUNC", 1)[0]
        except IndexError:
            return False, "marker_parse_error"

        signature_pattern = rf"^\s*(?:async\s+)?def\s+{re.escape(func_name)}\s*\("
        if not re.search(signature_pattern, inner, flags=re.MULTILINE):
            return False, "missing_signature"

        closing_pattern = r"<<END_FUNC(?:_NEW)?>>\s*$"
        if not re.search(closing_pattern, text, flags=re.MULTILINE):
            return False, "missing_closing_marker"

        return True, None
    
