#!/usr/bin/env python3
import requests

endpoints = [
    "http://localhost:11435/v1/completions",
    "http://localhost:11435/api/generate", 
    "http://localhost:11435/generate",
    "http://localhost:11435/v1/chat/completions",
    "http://localhost:11435/"
]

for endpoint in endpoints:
    try:
        print(f"Testing {endpoint}...")
        response = requests.get(endpoint, timeout=5)
        print(f"  Status: {response.status_code}")
        if response.status_code == 200:
            print(f"  Response: {response.text[:100]}...")
    except Exception as e:
        print(f"  Error: {e}")
    print()