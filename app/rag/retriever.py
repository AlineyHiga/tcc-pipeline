"""RAG retrieval with hybrid search."""
from pathlib import Path
from typing import List, Dict, Any, Iterable, Optional
import chromadb


class RAGRetriever:
    """Hybrid retriever combining vector and keyword search."""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection("autofix_kb")
    
    def retrieve(
        self,
        query: str,
        k: int = 8,
        filters: Dict[str, Any] = None,
        file_path: str | Path | None = None,
        symbol: str | None = None,
        repo_root: str | Path | None = None,
    ) -> Dict[str, Any]:
        """Retrieve relevant contexts for a query."""
        where = dict(filters or {})
        norm_path = self._normalize_path(file_path, repo_root)
        if norm_path:
            where["source"] = norm_path
        if symbol:
            where["symbol"] = symbol
        
        try:
            # ChromaDB requires explicit operators for multiple filters
            if where and len(where) > 1:
                # Convert to $and operator
                where = {"$and": [{k: v} for k, v in where.items()]}
            
            results = self.collection.query(
                query_texts=[query or ""],
                n_results=k,
                where=where or None,
            )
        except Exception as e:
            print(f"RAG retrieval error: {e}")
            return {"contexts": [], "citations": [], "few_shots": [], "metadata": []}
        
        contexts = results.get("documents", [[]])
        metadatas = results.get("metadatas", [[]])
        docs = contexts[0] if contexts else []
        metas = metadatas[0] if metadatas else []
        
        citations = [self._format_citation(meta) for meta in metas]
        few_shots = self._extract_few_shots(docs, metas)
        
        return {
            "contexts": docs,
            "citations": citations,
            "few_shots": few_shots,
            "metadata": metas,
        }
    
    def code_chunks(
        self,
        file_path: str | Path,
        repo_root: str | Path | None = None,
        line: Optional[int] = None,
        symbol: Optional[str] = None,
        limit: int = 4,
    ) -> List[Dict[str, Any]]:
        norm_path = self._normalize_path(file_path, repo_root)
        if not norm_path:
            return []
        
        where = {"source": norm_path}
        if symbol:
            where["symbol"] = symbol
        
        try:
            # ChromaDB requires explicit operators for multiple filters
            if where and len(where) > 1:
                where = {"$and": [{k: v} for k, v in where.items()]}
            
            results = self.collection.get(where=where)
        except Exception as e:
            print(f"RAG code retrieval error: {e}")
            return []
        
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])
        pairs: List[tuple] = list(zip(docs, metas))
        
        if line is not None:
            pairs = [
                (doc, meta)
                for doc, meta in pairs
                if meta.get("start_line") is None
                or (
                    meta.get("start_line") <= line <= meta.get("end_line", meta.get("start_line"))
                )
            ]
        
        pairs.sort(key=lambda item: (item[1].get("start_line", 0), item[1].get("symbol", "")))
        trimmed = pairs[:limit]
        
        return [
            {
                "content": doc,
                "source": meta.get("source"),
                "symbol": meta.get("symbol"),
                "kind": meta.get("kind"),
                "start_line": meta.get("start_line"),
                "end_line": meta.get("end_line"),
                "file_type": meta.get("file_type"),
            }
            for doc, meta in trimmed
        ]
    
    def _extract_few_shots(self, docs: Iterable[str], metas: Iterable[Dict[str, Any]]) -> List[str]:
        shots: List[str] = []
        for doc, meta in zip(docs, metas):
            text = doc or ""
            lowered = text.lower()
            kind = (meta or {}).get("kind", "")
            if any(keyword in lowered for keyword in ["fix", "before", "after", "diff", "---", "+++"]):
                if text not in shots:
                    shots.append(text)
            elif kind in {"function", "method"}:
                if text not in shots:
                    shots.append(text)
            if len(shots) >= 3:
                break
        return shots[:3]
    
    def _format_citation(self, meta: Dict[str, Any]) -> str:
        if not meta:
            return "unknown"
        source = meta.get("source", "unknown")
        start = meta.get("start_line")
        end = meta.get("end_line")
        symbol = meta.get("symbol")
        if start and end:
            base = f"{source}:{start}-{end}"
        else:
            base = source
        if symbol and symbol != "__file__":
            return f"{base} ({symbol})"
        return base
    
    def _normalize_path(self, file_path: str | Path | None, repo_root: str | Path | None) -> str | None:
        if not file_path:
            return None
        raw = str(file_path)
        if ":" in raw:
            raw = raw.split(":", 1)[-1]
        candidate = Path(raw)
        base_path = Path(repo_root).resolve() if repo_root else None
        if base_path and not candidate.is_absolute():
            candidate = base_path / candidate
        try:
            if base_path:
                relative = candidate.resolve().relative_to(base_path)
                return relative.as_posix()
        except Exception:
            pass
        return candidate.resolve().as_posix() if candidate.exists() else candidate.as_posix()
