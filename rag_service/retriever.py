"""Enhanced RAG retriever with dependency graph and hybrid search."""
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Set

from .models import Chunk, Target, ContextSymbol, RetrievalResult
from .locator import SymbolLocator

LOGGER = logging.getLogger(__name__)

class EnhancedRAGRetriever:
    """Enhanced RAG retriever with symbol-level precision."""
    
    def __init__(self, index_path: str = ".rag_index"):
        self.index_path = Path(index_path)
        self.chunks: Dict[str, Chunk] = {}
        self.graph: Dict[str, List[str]] = {}
        self.locator = SymbolLocator()
        self._load_index()
    
    def _load_index(self):
        """Load chunks and dependency graph."""
        chunks_file = self.index_path / "chunks.jsonl"
        graph_file = self.index_path / "graph.jsonl"
        
        # Load chunks
        if chunks_file.exists():
            with open(chunks_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    chunk = Chunk(
                        id=data['id'],
                        path=data['path'],
                        symbol=data['symbol'],
                        kind=data.get('kind', 'def'),
                        lines=tuple(data['lines']),
                        code=data.get('code_text', data.get('code', '')),
                        summary=data.get('summary_text', data.get('summary', '')),
                        imports=data.get('imports', []),
                        called_symbols=data.get('called_symbols', []),
                        neighbors=data.get('neighbors', [])
                    )
                    self.chunks[chunk.id] = chunk
        
        # Load graph (if exists)
        if graph_file.exists():
            with open(graph_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    self.graph[data['from']] = data.get('to', [])
        
        LOGGER.info(f"Loaded {len(self.chunks)} chunks and {len(self.graph)} graph edges")
    
    def locate_owner(self, file_path: str, line: int) -> Optional[Target]:
        """Locate the owner symbol for a file/line."""
        result = self.locator.locate_owner(file_path, line)
        if not result:
            return None
        
        symbol_name, kind, (start_line, end_line) = result
        return Target(
            path=file_path,
            symbol=symbol_name,
            kind=kind,
            lines=(start_line, end_line)
        )
    
    def retrieve_for_issue(self, file_path: str, line: int, rule: str, message: str, 
                          failing_tests: List[str] = None, k: int = 5, deps_hops: int = 1) -> RetrievalResult:
        """Retrieve context for a Sonar issue."""
        # 1. Locate owner symbol
        target = self.locate_owner(file_path, line)
        if not target:
            # Fallback: create target from file info
            target = Target(
                path=file_path,
                symbol="__file__",
                kind="module",
                lines=(max(1, line - 10), line + 10)
            )
        
        # 2. Find relevant chunks
        context_symbols = []
        
        # Add the target symbol itself
        target_id = f"{target.path}::{target.symbol}"
        if target_id in self.chunks:
            context_symbols.append(ContextSymbol(
                path=target.path,
                symbol=target.symbol,
                kind=target.kind,
                lines=target.lines,
                reason="owner"
            ))
        
        # 3. Add dependencies (1-hop)
        if deps_hops > 0:
            dependencies = self._get_dependencies(target_id, hops=deps_hops)
            for dep_id in dependencies[:k-1]:  # Reserve space for target
                if dep_id in self.chunks:
                    chunk = self.chunks[dep_id]
                    context_symbols.append(ContextSymbol(
                        path=chunk.path,
                        symbol=chunk.symbol,
                        kind=chunk.kind,
                        lines=chunk.lines,
                        reason="called_by_target"
                    ))
        
        # 4. Add tests that touch the symbol
        test_symbols = self._find_touching_tests(target_id, failing_tests or [])
        for test_id in test_symbols[:2]:  # Max 2 tests
            if test_id in self.chunks:
                chunk = self.chunks[test_id]
                context_symbols.append(ContextSymbol(
                    path=chunk.path,
                    symbol=chunk.symbol,
                    kind="test",
                    lines=chunk.lines,
                    reason="touches_target"
                ))
        
        explain = f"Found {len(context_symbols)} symbols for {target.symbol} in {target.path}"
        
        return RetrievalResult(
            target=target,
            context=context_symbols[:k],
            explain=explain
        )
    
    def _get_dependencies(self, symbol_id: str, hops: int = 1) -> List[str]:
        """Get dependencies within N hops."""
        if hops <= 0 or symbol_id not in self.chunks:
            return []
        
        visited = set()
        queue = [(symbol_id, 0)]
        dependencies = []
        
        while queue:
            current_id, current_hops = queue.pop(0)
            if current_id in visited or current_hops >= hops:
                continue
            
            visited.add(current_id)
            
            # Add direct dependencies from called_symbols
            if current_id in self.chunks:
                chunk = self.chunks[current_id]
                for called_symbol in chunk.called_symbols:
                    # Try to find the full ID for the called symbol
                    for chunk_id, chunk_data in self.chunks.items():
                        if chunk_data.symbol == called_symbol and chunk_id not in visited:
                            dependencies.append(chunk_id)
                            if current_hops + 1 < hops:
                                queue.append((chunk_id, current_hops + 1))
        
        return dependencies
    
    def _find_touching_tests(self, symbol_id: str, failing_tests: List[str]) -> List[str]:
        """Find tests that touch the target symbol."""
        touching_tests = []
        
        if symbol_id not in self.chunks:
            return touching_tests
        
        target_chunk = self.chunks[symbol_id]
        target_symbol = target_chunk.symbol
        
        # Look for tests that reference the symbol
        for chunk_id, chunk in self.chunks.items():
            if chunk.kind == "test" or "test" in chunk.path.lower():
                # Check if test references the target symbol
                if (target_symbol in chunk.code or 
                    target_symbol in chunk.called_symbols or
                    chunk_id in target_chunk.neighbors):
                    touching_tests.append(chunk_id)
        
        # Prioritize failing tests
        prioritized = []
        for test_name in failing_tests:
            for test_id in touching_tests:
                if test_name in test_id:
                    prioritized.append(test_id)
        
        # Add remaining tests
        for test_id in touching_tests:
            if test_id not in prioritized:
                prioritized.append(test_id)
        
        return prioritized