"""Symbol locator for finding owner symbols by file/line."""
import ast
from pathlib import Path
from typing import Optional, Tuple

class SymbolLocator:
    """Locates owner symbols by file and line number."""
    
    def locate_owner(self, file_path: str, line: int) -> Optional[Tuple[str, str, Tuple[int, int]]]:
        """Find the owner symbol (class/def) for a given file and line.
        
        Returns: (symbol_name, kind, (start_line, end_line)) or None
        """
        try:
            path = Path(file_path)
            if not path.exists():
                return None
            
            content = path.read_text(encoding='utf-8')
            tree = ast.parse(content)
            
            # Find all symbols with their line ranges
            symbols = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    start_line = node.lineno
                    end_line = node.end_lineno or start_line + 10
                    symbols.append((node.name, "def", start_line, end_line))
                elif isinstance(node, ast.ClassDef):
                    start_line = node.lineno
                    end_line = node.end_lineno or start_line + 50
                    symbols.append((node.name, "class", start_line, end_line))
            
            # Find the symbol that contains the target line
            for symbol_name, kind, start_line, end_line in symbols:
                if start_line <= line <= end_line:
                    return symbol_name, kind, (start_line, end_line)
            
            return None
            
        except Exception:
            return None