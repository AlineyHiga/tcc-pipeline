#!/usr/bin/env python3
"""
LLM server using Ollama API
"""
from flask import Flask, request, jsonify
import requests as req
import json

app = Flask(__name__)

@app.route('/api/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')
    
    # Use Ollama with CodeLlama
    try:
        print(f"[cyan]Calling Ollama at http://localhost:11434/api/generate")
        print("[magenta]Prompt sent to Ollama:\n" + prompt)
        ollama_response = req.post('http://localhost:11434/api/generate', json={
            "model": "deepseek-coder:1.3b",
            "prompt": prompt,
            "stream": False
        }, timeout=300)
        
        print(f"[yellow]Ollama response status: {ollama_response.status_code}")
        if ollama_response.status_code == 200:
            result = ollama_response.json().get('response', '')
            print(f"[green]Ollama response (first 200 chars): {result[:200]}...")
            print("[magenta]Full Ollama response:\n" + result)
            return jsonify({"response": result})
        else:
            print(f"[red]Ollama error response: {ollama_response.text}")
    except Exception as e:
        print(f"[red]Ollama error: {e}")
    
    # Fallback
    return jsonify({"response": "# LLM unavailable"})

if __name__ == '__main__':
    print("Ollama LLM Proxy Server running on http://localhost:11435")
    print("Make sure Ollama is running: ollama serve")
    print("And CodeLlama is installed: ollama pull codellama:7b-code")
    app.run(host='0.0.0.0', port=11435, debug=False)
