#!/usr/bin/env python3
"""
Start DeepSeek Coder LLM server
"""
import subprocess
import sys

def start_server():
    print("Starting Simple Code Fix LLM server...")
    subprocess.run([sys.executable, "llm_server.py"])

if __name__ == "__main__":
    start_server()