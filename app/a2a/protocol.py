"""Common data structures and message constants for the A2A agents."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, TypedDict


@dataclass
class Issue:
    key: str
    rule: str
    severity: str
    component: str
    message: str
    line: Optional[int]


class SessionState(TypedDict, total=False):
    issue: Issue
    attempt: int
    max_rounds: int
    context: str
    feedback_log: str
    patch: str
    test_logs: str
    test_passed: bool
    tester_summary: str
    sonar_passed: bool
    sonar_summary: str
    fixer_summary: str
    branch: str
    pr_url: str
    result: str


MessageType = Literal[
    "start_issue",
    "fix_request",
    "fix_failed",
    "test_request",
    "tester_feedback",
    "sonar_request",
    "sonar_feedback",
    "pr_request",
    "session_end",
]


__all__ = ["Issue", "SessionState", "MessageType"]
