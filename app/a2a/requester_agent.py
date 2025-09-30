import requests
from .protocol import A2AMessage, create_fix_request
from ..sonarqube_client import SonarIssue

class RequesterAgent:
    def __init__(self, fixer_endpoint: str):
        self.fixer_endpoint = fixer_endpoint

    def request_fix(self, issue: SonarIssue, source_code: str, file_path: str) -> str:
        message = create_fix_request(
            issue_key=issue.key,
            rule=issue.rule,
            severity=issue.severity,
            file_path=file_path,
            source_code=source_code,
            line_number=issue.line,
            message=issue.message
        )
        
        try:
            response = requests.post(
                f"{self.fixer_endpoint}/fix",
                json={
                    "type": message.type,
                    "content": message.content,
                    "metadata": message.metadata
                },
                timeout=120
            )
            response.raise_for_status()
            fix_response = response.json()
            return fix_response.get("content", {}).get("patch", "")
        except requests.Timeout:
            print("[red]Fixer agent timed out while processing the request.")
        except requests.RequestException as exc:
            print(f"[red]Fixer agent request failed: {exc}")
        return ""
