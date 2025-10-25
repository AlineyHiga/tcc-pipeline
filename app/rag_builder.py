"""Auto-build RAG index for pipeline execution."""
import logging
import os
import subprocess
import sys
from pathlib import Path

LOGGER = logging.getLogger(__name__)

def auto_build_rag_index(repo_root: Path = None) -> bool:
    """Automatically build RAG index if needed."""
    if repo_root is None:
        repo_root = Path(__file__).parent.parent
    
    rag_index_dir = repo_root / ".rag_index"
    chunks_file = rag_index_dir / "chunks.jsonl"
    
    # Check if index exists and is recent
    if chunks_file.exists():
        LOGGER.debug("RAG index already exists")
        return True
    
    # Find source directories to index
    src_candidates = [
        repo_root / "../src",
        repo_root / "src", 
        repo_root / "app",
        repo_root.parent / "src"
    ]
    
    src_dir = None
    for candidate in src_candidates:
        if candidate.exists() and any(candidate.glob("*.py")):
            src_dir = candidate
            break
    
    if not src_dir:
        LOGGER.warning("No Python source directory found for RAG indexing")
        return False
    
    # Build the index
    try:
        indexer_script = repo_root / "tools" / "build_code_rag.py"
        if not indexer_script.exists():
            LOGGER.warning("RAG indexer script not found")
            return False
        
        cmd = [
            sys.executable,
            str(indexer_script),
            "--src", str(src_dir),
            "--out", str(rag_index_dir)
        ]
        
        LOGGER.info(f"Auto-building RAG index from {src_dir}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            LOGGER.info("RAG index built successfully")
            return True
        else:
            LOGGER.error(f"Failed to build RAG index: {result.stderr}")
            return False
            
    except Exception as e:
        LOGGER.error(f"Error building RAG index: {e}")
        return False