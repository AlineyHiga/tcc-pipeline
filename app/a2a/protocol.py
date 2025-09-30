from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class A2AMessage:
    type: str
    content: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

def create_fix_request(issue_key: str, rule: str, severity: str, 
                      file_path: str, source_code: str, 
                      line_number: Optional[int] = None,
                      message: Optional[str] = None) -> A2AMessage:
    return A2AMessage(
        type="fix_request",
        content={
            "issue_key": issue_key,
            "rule": rule,
            "severity": severity,
            "file_path": file_path,
            "source_code": source_code,
            "line_number": line_number,
            "message": message or ""
        }
    )

def create_fix_response(patch: str, explanation: str = "") -> A2AMessage:
    return A2AMessage(
        type="fix_response",
        content={
            "patch": patch,
            "explanation": explanation
        }
    )

def create_test_response(test_result: dict) -> A2AMessage:
    return A2AMessage(
        type="test_response",
        content=test_result,
        metadata={"timestamp": "2024-01-01T00:00:00Z"}
    )
