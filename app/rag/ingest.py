"""RAG ingestion for code and documentation."""
import os
from pathlib import Path
from typing import List, Dict, Any

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
        
        for pattern in patterns:
            for file_path in directory.rglob(pattern):
                if self._should_skip(file_path):
                    continue
                    
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Use text splitter if available, otherwise simple chunking
                    if self.text_splitter:
                        chunks = self.text_splitter.split_text(content)
                    else:
                        chunks = [content[i:i+1000] for i in range(0, len(content), 800)]
                    
                    for i, chunk in enumerate(chunks):
                        doc_id = f"{file_path}#{i}"
                        self.collection.add(
                            documents=[chunk],
                            metadatas=[{
                                "source": str(file_path),
                                "chunk_id": i,
                                "file_type": file_path.suffix
                            }],
                            ids=[doc_id]
                        )
                    
                    docs_indexed += len(chunks)
                    
                except Exception as e:
                    print(f"Error ingesting {file_path}: {e}")
        
        return {"docs_indexed": docs_indexed}
    
    def _should_skip(self, file_path: Path) -> bool:
        """Check if file should be skipped during ingestion."""
        skip_dirs = {".git", "__pycache__", ".pytest_cache", "node_modules", ".venv"}
        skip_files = {".pyc", ".pyo", ".pyd"}
        
        if any(part in skip_dirs for part in file_path.parts):
            return True
        
        if file_path.suffix in skip_files:
            return True
            
        return False