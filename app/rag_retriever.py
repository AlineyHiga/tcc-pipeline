"""Minimal code RAG retriever for agent context."""
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Set

LOGGER = logging.getLogger(__name__)

@dataclass
class CodeChunk:
    id: str
    path: str
    symbol: str
    lines: tuple
    summary_text: str
    code_text: str
    imports: List[str]
    sonar_issue_keys: List[str]
    neighbors: List[str]

class CodeRAGRetriever:
    """Simple code retriever using lexical matching."""
    
    def __init__(self, index_path: str = ".rag_index"):
        self.index_path = Path(index_path)
        self.chunks: List[CodeChunk] = []
        self._load_index()
    
    def _load_index(self):
        """Load chunks from JSONL file."""
        chunks_file = self.index_path / "chunks.jsonl"
        if not chunks_file.exists():
            LOGGER.warning(f"RAG index not found at {chunks_file}")
            return
        
        with open(chunks_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                chunk = CodeChunk(
                    id=data['id'],
                    path=data['path'],
                    symbol=data['symbol'],
                    lines=tuple(data['lines']),
                    summary_text=data['summary_text'],
                    code_text=data['code_text'],
                    imports=data['imports'],
                    sonar_issue_keys=data['sonar_issue_keys'],
                    neighbors=data['neighbors']
                )
                self.chunks.append(chunk)
        
        LOGGER.info(f"Loaded {len(self.chunks)} code chunks")
    
    def retrieve(self, 
                 file_paths: Optional[List[str]] = None,
                 sonar_rules: Optional[List[str]] = None,
                 symbols: Optional[List[str]] = None,
                 max_chunks: int = 5) -> List[CodeChunk]:
        """Retrieve relevant code chunks."""
        if not self.chunks:
            return []
        
        scored_chunks = []
        
        for chunk in self.chunks:
            score = 0
            
            # File path matching
            if file_paths:
                for file_path in file_paths:
                    if file_path in chunk.path:
                        score += 10
            
            # Symbol matching
            if symbols:
                for symbol in symbols:
                    if symbol.lower() in chunk.symbol.lower():
                        score += 8
                    if symbol.lower() in chunk.summary_text.lower():
                        score += 4
            
            # Sonar rule matching
            if sonar_rules:
                for rule in sonar_rules:
                    if rule in chunk.sonar_issue_keys:
                        score += 6
                    if rule.lower() in chunk.summary_text.lower():
                        score += 3
            
            if score > 0:
                scored_chunks.append((score, chunk))
        
        # Sort by score and return top chunks
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored_chunks[:max_chunks]]
    
    def get_context_for_issues(self, issues: List) -> str:
        """Get context for a list of issues."""
        if not issues:
            return ""
        
        # Extract info from issues
        file_paths = []
        sonar_rules = []
        
        for issue in issues:
            if hasattr(issue, 'component'):
                file_paths.append(issue.component)
            if hasattr(issue, 'rule'):
                sonar_rules.append(issue.rule)
        
        # Retrieve relevant chunks
        chunks = self.retrieve(
            file_paths=file_paths,
            sonar_rules=sonar_rules,
            max_chunks=7
        )
        
        if not chunks:
            return ""
        
        # Format context
        context_lines = ["Código relevante encontrado:"]
        for chunk in chunks:
            context_lines.append(f"\n## {chunk.path}::{chunk.symbol} (linhas {chunk.lines[0]}-{chunk.lines[1]})")
            context_lines.append(chunk.code_text)
        
        return "\n".join(context_lines)