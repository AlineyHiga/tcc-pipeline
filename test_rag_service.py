#!/usr/bin/env python3
"""Test the enhanced RAG service."""
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from rag_service.service import RAGService

def test_rag_service():
    """Test the RAG service functionality."""
    rag = RAGService(".rag_index")
    
    # Test locate_owner
    print("=== Testing locate_owner ===")
    target = rag.locate_owner("../src/test-pipeline/src/vulnerable_app.py", 57)
    if target:
        print(f"Found owner: {target}")
    else:
        print("No owner found")
    
    # Test retrieve_for_issue
    print("\n=== Testing retrieve_for_issue ===")
    result = rag.retrieve_for_issue(
        file_path="../src/test-pipeline/src/vulnerable_app.py",
        line=57,
        rule="python:S5547",
        message="Division by zero error",
        k=3
    )
    
    print(f"Target: {result['target']}")
    print(f"Context symbols: {len(result['context'])}")
    for ctx in result['context']:
        print(f"  - {ctx['symbol']} ({ctx['kind']}) - {ctx['reason']}")
    
    # Test get_code_for_symbols
    print("\n=== Testing get_code_for_symbols ===")
    symbol_ids = [f"{ctx['path']}::{ctx['symbol']}" for ctx in result['context'][:2]]
    code_map = rag.get_code_for_symbols(symbol_ids)
    
    for symbol_id, code in code_map.items():
        print(f"\n--- {symbol_id} ---")
        print(code[:200] + "..." if len(code) > 200 else code)

if __name__ == "__main__":
    test_rag_service()