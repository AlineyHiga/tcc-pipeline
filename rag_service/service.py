"""RAG service API."""
from typing import Dict, List, Optional
from .retriever import EnhancedRAGRetriever
from .models import Target, RetrievalResult

class RAGService:
    """Internal RAG service for the pipeline."""
    
    def __init__(self, index_path: str = ".rag_index"):
        self.retriever = EnhancedRAGRetriever(index_path)
    
    def locate_owner(self, file_path: str, line: int) -> Optional[Dict]:
        """Locate owner symbol for file/line."""
        target = self.retriever.locate_owner(file_path, line)
        if not target:
            return None
        
        return {
            "path": target.path,
            "symbol": target.symbol,
            "kind": target.kind,
            "lines": list(target.lines)
        }
    
    def retrieve_for_issue(self, file_path: str, line: int, rule: str, message: str,
                          failing_tests: List[str] = None, k: int = 5, deps_hops: int = 1) -> Dict:
        """Retrieve context for a Sonar issue."""
        result = self.retriever.retrieve_for_issue(
            file_path=file_path,
            line=line,
            rule=rule,
            message=message,
            failing_tests=failing_tests or [],
            k=k,
            deps_hops=deps_hops
        )
        
        return {
            "target": {
                "path": result.target.path,
                "symbol": result.target.symbol,
                "kind": result.target.kind,
                "lines": list(result.target.lines)
            },
            "context": [
                {
                    "path": ctx.path,
                    "symbol": ctx.symbol,
                    "kind": ctx.kind,
                    "lines": list(ctx.lines),
                    "reason": ctx.reason
                }
                for ctx in result.context
            ],
            "explain": result.explain
        }
    
    def get_code_for_symbols(self, symbol_ids: List[str]) -> Dict[str, str]:
        """Get code content for specific symbols."""
        code_map = {}
        for symbol_id in symbol_ids:
            if symbol_id in self.retriever.chunks:
                chunk = self.retriever.chunks[symbol_id]
                code_map[symbol_id] = chunk.code
        return code_map