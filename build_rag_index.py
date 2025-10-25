#!/usr/bin/env python3
"""Build RAG index for the pipeline."""
import subprocess
import sys
from pathlib import Path

def main():
    """Build RAG index for the current project."""
    repo_root = Path(__file__).parent
    src_dir = repo_root / "../src"
    
    if not src_dir.exists():
        print(f"Source directory not found: {src_dir}")
        return 1
    
    # Run the indexer
    cmd = [
        sys.executable,
        str(repo_root / "tools" / "build_code_rag.py"),
        "--src", str(src_dir),
        "--out", str(repo_root / ".rag_index")
    ]
    
    print(f"Building RAG index from {src_dir}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(result.stdout)
        print("RAG index built successfully!")
        return 0
    else:
        print(f"Error building RAG index: {result.stderr}")
        return 1

if __name__ == "__main__":
    sys.exit(main())