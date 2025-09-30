from flask import Flask, request, jsonify
from .protocol import A2AMessage, create_test_response
import subprocess
import os

class TesterAgent:
    def __init__(self):
        self.app = Flask(__name__)
        self.setup_routes()

    def setup_routes(self):
        @self.app.route('/test', methods=['POST'])
        def handle_test_request():
            data = request.json
            message = A2AMessage(
                type=data['type'],
                content=data['content'],
                metadata=data.get('metadata')
            )
            
            result = self.run_tests(message)
            response = create_test_response(result)
            
            return jsonify({
                "type": response.type,
                "content": response.content,
                "metadata": response.metadata
            })

    def run_tests(self, message: A2AMessage) -> dict:
        content = message.content
        test_dir = content.get('test_dir', 'tests/')
        
        result = subprocess.run(
            ['python3', '-m', 'pytest', '-v', test_dir],
            capture_output=True,
            text=True,
            cwd='.'
        )
        
        return {
            'exit_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'passed': result.returncode == 0
        }

    def run(self, host='localhost', port=9091):
        self.app.run(host=host, port=port, debug=True)