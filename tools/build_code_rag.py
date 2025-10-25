#!/usr/bin/env python3
import argparse
import ast
import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional, Set

@dataclass
class Chunk:
    id: str
    path: str
    symbol: str
    kind: str
    lines: Tuple[int, int]
    summary_text: str
    code_text: str
    imports: List[str]
    called_symbols: List[str]
    neighbors: List[str]
    sonar_issue_keys: List[str]

def find_symbols(py_text: str) -> List[Tuple[str, str, int, int]]:
    """Extract function/class symbols with line ranges and kinds."""
    try:
        tree = ast.parse(py_text)
    except SyntaxError:
        return [("__file__", "module", 1, len(py_text.splitlines()))]
    
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
    
    return symbols or [("__file__", "module", 1, len(py_text.splitlines()))]

def extract_imports(py_text: str) -> List[str]:
    """Extract import statements."""
    imports = []
    for match in re.finditer(r'^(import .+|from .+ import .+)', py_text, re.MULTILINE):
        imports.append(match.group(0))
    return imports

def get_context_lines(lines: List[str], start: int, end: int, context: int = 15) -> str:
    """Get code with context lines."""
    ctx_start = max(0, start - 1 - context)
    ctx_end = min(len(lines), end + context)
    return '\n'.join(lines[ctx_start:ctx_end])

def summarize_code(code: str, symbol: str) -> str:
    """Generate simple summary."""
    lines = code.strip().splitlines()
    if not lines:
        return f"Empty {symbol}"
    
    first_line = lines[0].strip()
    if first_line.startswith('def '):
        return f"Function {symbol}: {first_line}"
    elif first_line.startswith('class '):
        return f"Class {symbol}: {first_line}"
    else:
        return f"Symbol {symbol}: {first_line[:100]}"

def extract_called_symbols(py_text: str) -> Set[str]:
    """Extract symbols called in the code."""
    called = set()
    try:
        tree = ast.parse(py_text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    called.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    called.add(node.func.attr)
    except SyntaxError:
        pass
    return list(called)

def find_test_neighbors(path: Path, symbol: str) -> List[str]:
    """Find test files that might test this symbol."""
    neighbors = []
    test_dirs = ["tests", "test"]
    
    for test_dir in test_dirs:
        test_path = path.parent / test_dir
        if test_path.exists():
            for test_file in test_path.rglob("test_*.py"):
                try:
                    content = test_file.read_text(encoding="utf-8", errors="ignore")
                    if symbol in content:
                        neighbors.append(f"{test_file}::test_{symbol}")
                except Exception:
                    continue
    
    return neighbors

def chunk_file(path: Path) -> List[Chunk]:
    """Extract chunks from a Python file."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    
    lines = text.splitlines()
    symbols = find_symbols(text)
    imports = extract_imports(text)
    chunks = []
    
    for symbol, kind, start, end in symbols:
        # Skip very large symbols
        if end - start > 300:
            continue
            
        symbol_code = '\n'.join(lines[start-1:end])
        context_code = get_context_lines(lines, start, end)
        called_symbols = extract_called_symbols(symbol_code)
        neighbors = find_test_neighbors(path, symbol)
        
        chunks.append(Chunk(
            id=f"{path}::{symbol}",
            path=str(path),
            symbol=symbol,
            kind=kind,
            lines=(start, end),
            summary_text=summarize_code(symbol_code, symbol),
            code_text=context_code,
            imports=imports,
            called_symbols=called_symbols,
            neighbors=neighbors,
            sonar_issue_keys=[]
        ))
    
    return chunks

def main():
    parser = argparse.ArgumentParser(description="Build enhanced code RAG index")
    parser.add_argument("--src", required=True, help="Source directory")
    parser.add_argument("--out", default=".rag_index", help="Output directory")
    args = parser.parse_args()

    src_path = Path(args.src)
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    for py_file in src_path.rglob("*.py"):
        if py_file.name.startswith('.'):
            continue
        chunks = chunk_file(py_file)
        all_chunks.extend(chunks)

    # Write chunks to JSONL
    chunks_file = out_path / "chunks.jsonl"
    with open(chunks_file, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    
    # Build dependency graph
    graph_file = out_path / "graph.jsonl"
    with open(graph_file, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            if chunk.called_symbols:
                graph_entry = {
                    "from": chunk.id,
                    "to": [f"{chunk.path}::{sym}" for sym in chunk.called_symbols]
                }
                f.write(json.dumps(graph_entry, ensure_ascii=False) + "\n")

    print(f"Indexed {len(all_chunks)} chunks to {chunks_file}")
    print(f"Built dependency graph in {graph_file}")

if __name__ == "__main__":
    main()