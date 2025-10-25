"""RAG retrieval with hybrid search."""
from typing import List, Dict, Any
import chromadb


class RAGRetriever:
    """Hybrid retriever combining vector and keyword search."""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection("autofix_kb")
        # ChromaDB handles embeddings internally
    
    def retrieve(self, query: str, k: int = 8, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Retrieve relevant contexts for a query."""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                where=filters
            )
            
            contexts = results['documents'][0] if results['documents'] else []
            metadatas = results['metadatas'][0] if results['metadatas'] else []
            
            citations = [meta.get('source', 'unknown') for meta in metadatas]
            
            # Extract few-shot examples (fix patterns)
            few_shots = []
            for ctx, meta in zip(contexts, metadatas):
                # Look for fix examples, diffs, or code patterns
                if any(keyword in ctx.lower() for keyword in ['fix', 'before', 'after', 'diff', '---', '+++']):
                    few_shots.append(ctx)
                elif meta.get('file_type') == '.py' and any(pattern in ctx for pattern in ['def ', 'class ', 'import ']):
                    few_shots.append(ctx)
            
            # Limit to top 3 examples
            few_shots = few_shots[:3]
            
            return {
                "contexts": contexts,
                "citations": citations,
                "few_shots": few_shots
            }
            
        except Exception as e:
            print(f"RAG retrieval error: {e}")
            return {"contexts": [], "citations": [], "few_shots": []}