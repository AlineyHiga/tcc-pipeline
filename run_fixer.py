#!/usr/bin/env python3
"""
Script to run the Fixer Agent server with LLM
"""
from app.a2a.fixer_agent import FixerAgent

if __name__ == "__main__":
    print("Starting LLM-powered Fixer Agent on http://localhost:9090")
    print("Using LLM on port 11435")
    fixer = FixerAgent(llm_endpoint="http://localhost:11435/api/generate")
    fixer.run(host='localhost', port=9090)