#!/usr/bin/env python3
"""
Test script for LLM-powered A2A agents
"""
import time
import requests
from app.sonarqube_client import SonarIssue
from app.a2a.requester_agent import RequesterAgent
from rich import print

def test_llm_agents():
    print("[blue]🤖 Testing A2A agents with local LLM")
    
    # Mock issue
    issue = SonarIssue(
        key="TEST-001",
        rule="python:S1481", 
        severity="MAJOR",
        component="src/sample_module.py",
        line=2,
        message="Remove this unused local variable 'unused_var'",
        textRange=None
    )
    
    # Read sample module code
    with open('src/sample_module.py', 'r') as f:
        source_code = f.read()
    
    print(f"[yellow]Issue: {issue.key} - {issue.message}")
    print(f"[cyan]Source code:\n{source_code}")
    
    # Test requester agent
    requester = RequesterAgent("http://localhost:9090")
    
    try:
        print("\n[blue]Sending fix request to A2A agent...")
        patch = requester.request_fix(issue, source_code, "src/sample_module.py")
        
        if patch:
            print(f"[green]✅ Received patch:\n{patch}")
        else:
            print("[red]❌ No patch received")
            
    except Exception as e:
        print(f"[red]Error: {e}")

if __name__ == "__main__":
    test_llm_agents()