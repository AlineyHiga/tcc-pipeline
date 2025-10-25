"""Fixer agent for generating minimal code patches."""
import re
import ast
import difflib
from typing import Dict, Any
from pathlib import Path
from ..llm_client import get_llm_client
from ..rag.retriever import RAGRetriever


def _extract_func_from_llm(txt: str, func_name: str) -> str:
    """Extract function from LLM response with multiple fallbacks."""
    # 1) Marcadores
    m = re.search(r"<<FUNC_NEW>>(.*?)(?:<<END_FUNC>>|$)", txt, re.S)
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
    if not cand.startswith("def "):
        raise ValueError("LLM response did not contain a function body")

    return cand + ("" if cand.endswith("\n") else "\n")


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
        """Generate unified diff patch using deterministic target selection."""
        import logging
        from ..logging_setup import log_event
        logger = logging.getLogger(__name__)
        
        fix_plan = state.get("fix_plan", {})
        current_lot = state.get("current_lot", {})
        rag_ctx = state.get("rag_ctx", {})
        
        if not fix_plan or not current_lot:
            return ""
        
        # Always select targets[0] (highest priority)
        targets = fix_plan.get("targets", [])
        if not targets:
            logger.warning("No targets in fix_plan")
            return ""
        
        target = targets[0]
        log_event("fix.select", 
                 file=target["file"], 
                 func=target["func_name"], 
                 from_to=target["from_to"],
                 reason="highest_priority")
        
        issues = current_lot.get("issues", [])
        rule_key = current_lot.get("ruleKey", "")
        
        # Read only the target file (not all issue files)
        file_contents = {}
        target_file = target["file"]
        repo_root = state.get("repo_path", ".")
        
        try:
            full_path = Path(repo_root) / target_file if not Path(target_file).is_absolute() else Path(target_file)
            content = full_path.read_text(encoding="utf-8")
            file_contents[target_file] = content
            logger.debug(f"Read {len(content)} chars from {target_file}")
        except Exception as e:
            logger.error(f"Error reading target file {target_file}: {e}")
            return ""

        
        # Generate patch prompt for selected target
        prompt = self._build_patch_prompt_target(target, file_contents, rag_ctx)
        
        try:
            # Prefer function rewriting for complexity rules (S3776)
            if rule_key == "python:S3776" and target.get("func_name"):
                func_patch = self._try_function_rewrite_target(target, file_contents, repo_root)
                if func_patch:
                    return func_patch
            
            # Fallback to diff-based approach
            patch = self.llm.generate(prompt)
            
            # Validate and reconstruct diff
            sanitized_patch = self._validate_and_reconstruct_diff(patch, target_file)
            
            # Enforce allowlist and budget
            current_lot = state.get("current_lot", {})
            budget = current_lot.get("budget", 300)
            error = self._enforce_allowlist_and_budget(sanitized_patch, budget, current_lot)
            
            # If diff validation failed or has errors, trigger fallback immediately
            if not sanitized_patch or error:
                print(f"Patch validation failed: {error or 'invalid diff'}")
                return self._fallback_to_rewrite(target, file_contents, rule_key)
            
            # Test patch application
            from ..patcher import SafePatcher
            patcher = SafePatcher()
            result = patcher.apply_patch(sanitized_patch, repo_root, budget=budget)
            
            if not result["applied"]:
                print(f"Patch apply failed: {result.get('error', 'unknown')}")
                return self._fallback_to_rewrite(target, file_contents, rule_key)
            
            return sanitized_patch
                
        except Exception as e:
            print(f"Patch generation failed: {e}")
            return ""
    
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
    
    def _fallback_to_rewrite(self, target: dict, file_contents: dict, rule_key: str) -> str:
        """Fallback to function/file rewriting when diff fails."""
        print("Falling back to function/file rewrite...")
        
        # Try function rewrite first
        if target.get("func_name") and target.get("func_span"):
            func_patch = self._try_function_rewrite_target(target, file_contents, "")
            if func_patch:
                print("Function rewrite successful")
                return func_patch
        # Para S3776 (complexidade), evite reescrever arquivo inteiro; use codemod mecânico
        if rule_key == "python:S3776":
            fallback_patch = self._fallback_codemod(target["file"], rule_key)
            if fallback_patch:
                print(f"Using fallback codemod for {rule_key}")
                return fallback_patch
            # Para S3776, não faça full file rewrite
            return ""

        # Try full file rewrite (outras regras)
        full_req = self._build_full_file_prompt_target(target, file_contents)
        new_texts = self.llm.generate(full_req)
        rebuilt_diff = self._make_diff_from_texts(file_contents, new_texts)
        
        if rebuilt_diff:
            print("Full file rewrite successful")
            return rebuilt_diff
        
        # Last resort: mechanical codemod
        fallback_patch = self._fallback_codemod(target["file"], rule_key)
        if fallback_patch:
            print(f"Using fallback codemod for {rule_key}")
            return fallback_patch
        
        return ""
    
    def _component_rel(self, c: str) -> str:
        """Extract relative path from component."""
        return c.split(":", 1)[-1] if ":" in c else c
    
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
        for iss in lot.get("issues", []):
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
                
                frm_to = self._parse_from_to(iss.get("message", ""))
                
                goals.append({
                    "rule": iss.get("rule", ""),
                    "file": rel,
                    "func_name": func[0] if func else None,
                    "func_span": [func[1], func[2]] if func else None,
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
        """Build prompt for function rewriting."""
        return f"""You are a Python refactoring expert. Sonar rule: {rule_key}.
Reduce cognitive complexity of {func_name} without changing public behavior.

Context (read-only):
<<CTX>>
{ctx_text}
<<END_CTX>>

Original function:
<<FUNC_ORIG>>
{func_src}
<<END_FUNC_ORIG>>

Instructions:
- Same signature and contracts
- Prefer guard-clauses and small local helpers
- Do not modify other functions
- Respond ONLY with the complete function body starting with def {func_name}(...):. Do not include comments, markdown/backticks, or any text outside the code.
- Wrap your response between <<FUNC_NEW>> and <<END_FUNC>> markers.

<<FUNC_NEW>>
def {func_name}""".rstrip()
    
    def _try_function_rewrite_target(self, target: dict, file_contents: dict, repo_root: str) -> str:
        """Try function-level rewriting for single target."""
        if not target.get("func_name") or not target.get("func_span"):
            return ""
        
        file_path = target["file"]
        func_name = target["func_name"]
        start, end = target["func_span"]
        
        if file_path not in file_contents:
            return ""
        
        try:
            text = file_contents[file_path]
            
            # Extract function and context
            ctx_text = self._window_lines(text, start, end, k=5)
            lines = text.splitlines(True)
            func_src = "".join(lines[start:end])
            
            # Build prompt for function rewrite
            prompt = self._prompt_rewrite_func(target["rule"], func_name, ctx_text, func_src)
            
            # Generate new function
            resp = self.llm.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=600,
                stop=["<<END_FUNC>>"]
            ).strip()
            
            # Extract function using robust helper
            try:
                new_func = _extract_func_from_llm(resp, func_name)
            except ValueError as e:
                print(f"Function extraction failed: {e}")
                return ""
            
            # Merge and create diff locally
            new_file = self._merge_func(text, start, end, new_func)
            patch = build_unified_diff(file_path, text, new_file)
            
            # Validate diff before returning
            ok, why = validate_diff(patch)
            if not ok:
                print(f"Generated diff validation failed: {why}")
                return ""
            
            if patch:
                print(f"Generated function-level patch for {func_name}")
                return patch
                
        except Exception as e:
            print(f"Function rewrite failed for {func_name}: {e}")
        
        return ""
    
    def _build_full_file_prompt_target(self, target: dict, file_contents: dict) -> str:
        """Build prompt for full file rewrite of target file."""
        file_path = target["file"]
        content = file_contents.get(file_path, "")
        rule_key = target["rule"]
        func_name = target.get("func_name", "unknown")
        
        return f"""Rewrite the complete file to fix Sonar rule {rule_key} in function {func_name}.

Current file:
=== {file_path} ===
{content}

Return the complete rewritten file content.
Format: === {file_path} === followed by complete file content.
Make minimal changes focused on the specific function.

Rewritten file:"""
    
    def _make_diff_from_texts(self, original_contents: dict, new_text: str) -> str:
        """Generate unified diff from original and new file contents."""
        import difflib
        import re
        
        # Parse new_text for file sections
        sections = re.split(r'=== ([^=]+) ===', new_text)
        if len(sections) < 3:
            return ""
        
        diffs = []
        for i in range(1, len(sections), 2):
            if i + 1 >= len(sections):
                break
            
            filename = sections[i].strip()
            new_content = sections[i + 1].strip()
            
            if filename in original_contents:
                original_lines = original_contents[filename].splitlines(keepends=True)
                new_lines = new_content.splitlines(keepends=True)
                
                diff = list(difflib.unified_diff(
                    original_lines,
                    new_lines,
                    fromfile=f"a/{filename}",
                    tofile=f"b/{filename}",
                    n=3  # 3 lines of context
                ))
                
                if diff:
                    diffs.extend(diff)
        
        return "".join(diffs)
    

    
    def _build_patch_prompt_target(self, target: dict, file_contents: dict, rag_ctx: dict) -> str:
        """Build prompt for patch generation focused on single target."""
        few_shots = rag_ctx.get("few_shots", [])
        contexts = rag_ctx.get("contexts", [])
        
        # Prioritize few_shots, fallback to contexts
        examples = "\n\n".join(few_shots[:3]) if few_shots else "\n".join(contexts[:2])
        
        file_path = target["file"]
        content = file_contents.get(file_path, "")
        
        # Limit content size for prompt
        if len(content) > 2000:
            content = content[:2000] + "\n... (truncated)"
        
        rule_key = target["rule"]
        func_name = target.get("func_name", "unknown")
        from_to = target.get("from_to", [None, 15])
        
        target_info = f"Focus on function {func_name} in {file_path}"
        if from_to[0] is not None:
            target_info += f" (reduce complexity from {from_to[0]} to {from_to[1]})"
        
        return f"""Generate a minimal unified diff to fix Sonar rule {rule_key}.

{target_info}

Current file:
--- {file_path} ---
{content}

Fix examples and patterns:
{examples}

IMPORTANT REQUIREMENTS:
- Generate ONLY unified diff format (--- +++ @@ lines)
- Include ≥3 lines of unchanged context before and after each change
- Do not create hunks that start at line 1 (avoid @@ -1,3)
- Make minimal changes focused on the specific function
- Focus on code quality improvements

Diff:"""