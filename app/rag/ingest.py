"""RAG ingestion for code and documentation."""
import ast
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Iterable

import chromadb
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


class RAGIngestor:
    """Indexes source code and documentation for retrieval."""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection("autofix_kb")
        
        if LANGCHAIN_AVAILABLE:
            try:
                self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
                self.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    separators=["\n\n", "\n", " ", ""]
                )
            except Exception as e:
                print(f"Warning: Could not initialize LangChain components: {e}")
                self.embeddings = None
                self.text_splitter = None
        else:
            self.embeddings = None
            self.text_splitter = None
    
    def ingest_directory(self, directory: Path, patterns: List[str] = None) -> Dict[str, int]:
        """Ingest files from directory matching patterns."""
        if patterns is None:
            patterns = ["*.py", "*.md", "*.txt", "*.yml", "*.yaml"]
        
        docs_indexed = 0
        repo_root = directory.resolve()
        
        for pattern in patterns:
            for file_path in directory.rglob(pattern):
                if self._should_skip(file_path):
                    continue
                
                try:
                    rel_path = self._make_relative(file_path, repo_root)
                    content = file_path.read_text(encoding="utf-8")
                    self.collection.delete(where={"source": rel_path})
                    
                    segments = self._segment_file(file_path, rel_path, content)
                    if not segments:
                        continue
                    
                    for segment in segments:
                        self.collection.add(
                            documents=[segment["document"]],
                            metadatas=[segment["metadata"]],
                            ids=[segment["id"]],
                        )
                    
                    docs_indexed += len(segments)
                except Exception as e:
                    print(f"Error ingesting {file_path}: {e}")
        
        return {"docs_indexed": docs_indexed}
    
    def _should_skip(self, file_path: Path) -> bool:
        skip_dirs = {".git", "__pycache__", ".pytest_cache", "node_modules", ".venv"}
        skip_files = {".pyc", ".pyo", ".pyd"}
        
        if any(part in skip_dirs for part in file_path.parts):
            return True
        
        if file_path.suffix in skip_files:
            return True
        
        return False
    
    def _segment_file(self, file_path: Path, rel_path: str, content: str) -> List[Dict[str, Any]]:
        suffix = file_path.suffix.lower()
        if suffix == ".py":
            segments = self._segment_python(rel_path, content)
            if segments:
                return segments
        return self._segment_generic(rel_path, content, suffix)
    
    def _segment_python(self, rel_path: str, content: str) -> List[Dict[str, Any]]:
        segments: List[Dict[str, Any]] = []
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return segments
        
        for node, qualified, kind in self._iter_python_symbols(tree):
            segment = self._node_source(content, node)
            if not segment.strip():
                continue
            start = getattr(node, "lineno", 1)
            end = getattr(node, "end_lineno", start)
            metadata = {
                "source": rel_path,
                "symbol": qualified,
                "kind": kind,
                "start_line": start,
                "end_line": end,
                "file_type": ".py",
            }
            segments.append(
                {
                    "document": segment[:2500],
                    "metadata": metadata,
                    "id": self._build_id(rel_path, qualified, start, end),
                }
            )
        
        if segments:
            full_meta = {
                "source": rel_path,
                "symbol": "__file__",
                "kind": "file",
                "start_line": 1,
                "end_line": content.count("\n") + 1,
                "file_type": ".py",
            }
            segments.append(
                {
                    "document": content[:4000],
                    "metadata": full_meta,
                    "id": self._build_id(rel_path, "__file__", 1, full_meta["end_line"]),
                }
            )
        
        return segments
    
    def _segment_generic(self, rel_path: str, content: str, suffix: str) -> List[Dict[str, Any]]:
        chunks: Iterable[str]
        if self.text_splitter:
            chunks = self.text_splitter.split_text(content)
        else:
            step = 800
            chunks = [content[i:i + 1000] for i in range(0, len(content), step)]
        
        segments: List[Dict[str, Any]] = []
        for idx, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            metadata = {
                "source": rel_path,
                "symbol": f"chunk_{idx}",
                "kind": "document",
                "start_line": idx * 80 + 1,
                "end_line": (idx + 1) * 80,
                "file_type": suffix,
            }
            segments.append(
                {
                    "document": chunk,
                    "metadata": metadata,
                    "id": self._build_id(rel_path, metadata["symbol"], metadata["start_line"], metadata["end_line"]),
                }
            )
        return segments
    
    def _iter_python_symbols(self, tree: ast.AST) -> Iterable[tuple]:
        def walk(node: ast.AST, parents: List[str]) -> Iterable[tuple]:
            for child in getattr(node, "body", []):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qualified = ".".join(parents + [child.name]) if parents else child.name
                    kind = "method" if parents else "function"
                    yield child, qualified, kind
                if isinstance(child, ast.ClassDef):
                    qualified = ".".join(parents + [child.name]) if parents else child.name
                    yield child, qualified, "class"
                    yield from walk(child, parents + [child.name])
        yield from walk(tree, [])
    
    def _node_source(self, content: str, node: ast.AST) -> str:
        try:
            segment = ast.get_source_segment(content, node)
            if segment:
                return segment
        except AttributeError:
            pass
        lines = content.splitlines()
        start = getattr(node, "lineno", 1) - 1
        end = getattr(node, "end_lineno", start + 1)
        return "\n".join(lines[start:end])
    
    def _build_id(self, rel_path: str, symbol: str, start: int, end: int) -> str:
        payload = f"{rel_path}:{symbol}:{start}:{end}"
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()
    
    def _make_relative(self, path: Path, repo_root: Path) -> str:
        try:
            rel = path.resolve().relative_to(repo_root)
        except ValueError:
            rel = path.name
        return rel.as_posix()
