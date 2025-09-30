from flask import Flask, request, jsonify
from .protocol import A2AMessage, create_fix_response
import requests
import json
import re

class FixerAgent:
    def __init__(self, llm_endpoint="http://localhost:11435/api/generate"):
        self.app = Flask(__name__)
        self.llm_endpoint = llm_endpoint
        self.setup_routes()

    def setup_routes(self):
        @self.app.route('/fix', methods=['POST'])
        def handle_fix_request():
            data = request.json
            message = A2AMessage(
                type=data['type'],
                content=data['content'],
                metadata=data.get('metadata')
            )
            
            patch = self.generate_patch(message)
            response = create_fix_response(patch, "Auto-generated fix")
            
            return jsonify({
                "type": response.type,
                "content": response.content,
                "metadata": response.metadata
            })

    def generate_patch(self, message: A2AMessage) -> str:
        content = message.content
        return self._generate_llm_patch(content)
    
    def _generate_llm_patch(self, content: dict) -> str:
        prompt = self._build_prompt(content)
        file_path = content.get('file_path', 'file.py')
        result = self._call_llm(prompt)
        patch = self._extract_patch(result, file_path)
        if patch:
            return patch

        retry_prompt = (
            prompt
            + "\n\nYou must output only the unified diff patch now. Repeat the diff format exactly as shown."
        )
        retry_result = self._call_llm(retry_prompt)
        patch = self._extract_patch(retry_result, file_path)
        if patch:
            return patch

        print("[yellow]Using fallback - no valid patch in LLM response after retry")
        return self._fallback_fix(content)

    def _build_prompt(self, content: dict) -> str:
        return f"""Fix this SonarQube issue: {content.get('message', '')}

File: {content.get('file_path', 'file.py')}
Code:
{content.get('source_code', '')}

Output ONLY the unified diff patch. No explanations.

--- a/{content.get('file_path', 'file.py')}
+++ b/{content.get('file_path', 'file.py')}
@@ -line,count +line,count @@
-old line
+new line
"""
    
    def _call_llm(self, prompt: str) -> str:
        try:
            print(f"[cyan]Calling LLM at {self.llm_endpoint}...")
            print("[magenta]Prompt sent to LLM:\n" + prompt)
            response = requests.post(
                self.llm_endpoint,
                json={
                    "model": "llama",
                    "prompt": prompt,
                    "max_tokens": 300,
                    "temperature": 0.0,
                    "stream": False,
                },
                timeout=120,
            )
            print(f"[yellow]LLM response status: {response.status_code}")
            if response.status_code == 200:
                result = response.json().get("response", "")
                print(f"[green]LLM generated response (first 100 chars): {result[:100]}...")
                print("[magenta]Full LLM response:\n" + result)
                return result
        except Exception as e:
            print(f"[red]LLM error: {e}")
        return ""

    def _extract_patch(self, result: str, file_path: str) -> str:
        if not result:
            return ""

        if '---' not in result or '+++' not in result:
            return ""

        cleaned_lines = []
        for raw_line in result.split('\n'):
            stripped = raw_line.strip()
            if stripped.startswith('```'):
                continue
            normalized = re.sub(r'^\s*\d+[\.)]\s*', '', raw_line.rstrip('\n'))
            cleaned_lines.append(normalized)

        patch_lines = []
        in_patch = False
        allowed_prefixes = ('---', '+++', '@@', '-', '+', ' ')
        for raw_line in cleaned_lines:
            detection_line = raw_line.lstrip()
            if detection_line.startswith('---'):
                in_patch = True
            if in_patch:
                if detection_line and not detection_line.startswith(allowed_prefixes):
                    break
                if detection_line.startswith(allowed_prefixes):
                    candidate = detection_line
                else:
                    candidate = ' ' + detection_line
                patch_lines.append(candidate)

        patch = '\n'.join(patch_lines).strip()
        if not (patch.startswith('---') and '+++' in patch and '@@' in patch):
            return ""

        normalized = patch.split('\n')
        expected_old = f"--- a/{file_path}"
        expected_new = f"+++ b/{file_path}"
        if normalized:
            normalized[0] = expected_old
        if len(normalized) > 1:
            normalized[1] = expected_new

        sanitized_lines = []
        for line in normalized:
            # Reject patches with explanations or commentary
            if any(word in line.lower() for word in ['sure', 'here', 'patch', 'fixes', 'note', 'however', 'explanation']):
                return ""
            if line.startswith(('-', '+')):
                # reject lines with obvious narrative artifacts
                if '//' in line or 'http' in line or 'Added missing' in line or 'Removed' in line:
                    return ""
            sanitized_lines.append(line)

        return '\n'.join(sanitized_lines)

    def _fallback_fix(self, content: dict) -> str:
        print("[yellow]Using fallback fix (not LLM)")
        source_code = content.get('source_code', '')
        file_path = content.get('file_path', '')
        
        # Simple fallback: add docstring to functions
        lines = source_code.split('\n')
        fixed_lines = []
        
        for line in lines:
            fixed_lines.append(line)
            if line.strip().startswith('def ') and ':' in line:
                indent = len(line) - len(line.lstrip())
                docstring = ' ' * (indent + 4) + '"""Function docstring."""'
                fixed_lines.append(docstring)
        
        fixed_code = '\n'.join(fixed_lines)
        return self._create_unified_diff(source_code, fixed_code, file_path)

    def _create_unified_diff(self, original: str, fixed: str, file_path: str) -> str:
        if original == fixed:
            return ""
        
        original_lines = original.split('\n')
        fixed_lines = fixed.split('\n')
        
        diff = f"--- a/{file_path}\n+++ b/{file_path}\n"
        diff += f"@@ -1,{len(original_lines)} +1,{len(fixed_lines)} @@\n"
        
        for line in original_lines:
            diff += f"-{line}\n"
        for line in fixed_lines:
            diff += f"+{line}\n"
        
        return diff

    def run(self, host='localhost', port=9090):
        self.app.run(host=host, port=port, debug=True)
