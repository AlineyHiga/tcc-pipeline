"""Safe patch application with path allowlisting."""
import os
import subprocess
from pathlib import Path
from typing import List, Dict, Any
import tempfile
import pathspec
try:
    from unidiff import PatchSet
except ImportError:
    PatchSet = None


class SafePatcher:
    """Applies patches with security constraints."""
    
    def __init__(self, allowed_patterns: List[str] = None):
        if allowed_patterns is None:
            allowed_patterns = [
                "src/**",
                "tests/**", 
                "tests_prop/**",
                "README.md",
                "sonar-project.properties",
                ".coveragerc",
                ".github/workflows/*.yml",
                ".env.example"
            ]
        self.spec = pathspec.PathSpec.from_lines('gitwildmatch', allowed_patterns)
    
    def apply_patch(self, diff: str, repo_path: str = ".", budget: int = None) -> Dict[str, Any]:
        """Apply unified diff with path validation."""
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"apply_patch called with repo_path: {repo_path}")
        logger.debug(f"Diff content: {diff[:500]}...")
        
        repo_path = Path(repo_path).resolve()
        
        # Validate paths in diff
        affected_files = self._extract_paths_from_diff(diff)
        logger.debug(f"Extracted paths from diff: {affected_files}")
        
        blocked_files = [f for f in affected_files if not self._is_path_allowed(f)]
        logger.debug(f"Blocked files: {blocked_files}")
        
        if blocked_files:
            return {
                "applied": False,
                "files_changed": 0,
                "rejected": blocked_files,
                "error": f"Blocked paths: {blocked_files}"
            }
        
        # Check LOC budget if provided
        if budget is not None:
            loc_changed = self._count_loc_changes(diff)
            logger.debug(f"LOC changed: {loc_changed}, budget: {budget}")
            
            if loc_changed > budget:
                return {
                    "applied": False,
                    "files_changed": 0,
                    "rejected": [],
                    "error": f"LOC budget exceeded ({loc_changed}>{budget}). Please restrict scope."
                }
        
        # Log first 5 lines of target files for debugging
        self._log_target_files(affected_files, repo_path)
        
        # Validate diff first
        from .a2a.fixer_agent import validate_diff
        ok, why = validate_diff(diff)
        if not ok:
            return {
                "applied": False,
                "files_changed": 0,
                "rejected": [],
                "error": f"Invalid diff: {why}",
                "reason": "invalid_diff"
            }
        
        # Apply patch with resilient sequence
        try:
            phase_errors = {}
            with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as f:
                f.write(diff)
                patch_file = f.name
            
            # Try 1: Standard check with whitespace tolerance
            result = subprocess.run(
                ["git", "apply", "--check", "--whitespace=nowarn", patch_file],
                cwd=repo_path,
                capture_output=True,
                text=True
            )
            phase_errors["check"] = result.stderr
            
            if result.returncode != 0:
                return {
                    "applied": False,
                    "files_changed": 0,
                    "rejected": [],
                    "error": "Invalid diff (--check failed)",
                    "error_detail": phase_errors,
                    "reason": "invalid_diff"
                }
            
            # Try 2: Standard apply with whitespace tolerance
            result = subprocess.run(
                ["git", "apply", "--whitespace=nowarn", patch_file],
                cwd=repo_path,
                capture_output=True,
                text=True
            )
            phase_errors["apply"] = result.stderr
            
            if result.returncode == 0:
                # Validate Python syntax after applying patch
                syntax_errors = self._validate_python_syntax(affected_files, repo_path)
                if syntax_errors:
                    # Rollback the patch
                    subprocess.run(
                        ["git", "checkout", "--"] + affected_files,
                        cwd=repo_path,
                        capture_output=True
                    )
                    return {
                        "applied": False,
                        "files_changed": 0,
                        "rejected": [],
                        "error": f"Patch creates syntax errors: {syntax_errors}",
                        "reason": "syntax_error"
                    }
                
                return {
                    "applied": True,
                    "files_changed": len(affected_files),
                    "rejected": [],
                    "error": None
                }
            
            # Standard apply failed, return clear error
            logger.warning(f"Standard apply failed: {result.stderr}")
            return {
                "applied": False,
                "files_changed": 0,
                "rejected": [],
                "error": "Patch apply failed",
                "error_detail": phase_errors,
                "reason": "patch_failed"
            }
            

            
        except Exception as e:
            return {
                "applied": False,
                "files_changed": 0,
                "rejected": [],
                "error": str(e)
            }
        finally:
            if 'patch_file' in locals():
                os.unlink(patch_file)
    
    def _extract_paths_from_diff(self, diff: str) -> List[str]:
        """Extract file paths from unified diff."""
        import re
        paths = []
        for line in diff.split('\n'):
            if line.startswith('--- ') or line.startswith('+++ '):
                # Extract path, handle a/ and b/ prefixes
                path = line[4:].strip()
                if path.startswith('a/') or path.startswith('b/'):
                    path = path[2:]
                if path and path != '/dev/null':
                    paths.append(path)
        return list(set(paths))
    
    def _count_loc_changes(self, diff: str) -> int:
        """Count lines of code changed in diff."""
        if PatchSet:
            try:
                ps = PatchSet(diff.splitlines(keepends=True))
                return sum(h.added + h.removed for p in ps for h in p)
            except:
                pass
        
        # Fallback to simple counting
        added = 0
        removed = 0
        
        for line in diff.split('\n'):
            if line.startswith('+') and not line.startswith('+++'):
                added += 1
            elif line.startswith('-') and not line.startswith('---'):
                removed += 1
        
        return added + removed
    
    def _log_target_files(self, affected_files: list, repo_path: str) -> None:
        """Log first 5 lines of target files for debugging."""
        import logging
        logger = logging.getLogger(__name__)
        
        for file_path in affected_files[:3]:  # Limit to 3 files
            full_path = Path(repo_path) / file_path
            if full_path.exists():
                try:
                    with open(full_path, 'r') as f:
                        lines = f.readlines()[:5]
                    logger.debug(f"Target file {file_path} first 5 lines:")
                    for i, line in enumerate(lines, 1):
                        logger.debug(f"  {i:2d} | {line.rstrip()}")
                except Exception as e:
                    logger.debug(f"Could not read {file_path}: {e}")
    
    def _validate_python_syntax(self, affected_files: List[str], repo_path: Path) -> List[str]:
        """Validate Python syntax for affected .py files."""
        import ast
        syntax_errors = []
        
        for file_path in affected_files:
            if not file_path.endswith('.py'):
                continue
                
            full_path = repo_path / file_path
            if not full_path.exists():
                continue
                
            try:
                content = full_path.read_text(encoding='utf-8')
                ast.parse(content)
            except SyntaxError as e:
                syntax_errors.append(f"{file_path}:{e.lineno}: {e.msg}")
            except Exception as e:
                syntax_errors.append(f"{file_path}: {str(e)}")
        
        return syntax_errors
    
    def _is_path_allowed(self, path: str) -> bool:
        """Check if path is in allowlist."""
        return self.spec.match_file(path)