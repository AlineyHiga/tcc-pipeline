"""Common data structures for A2A agent communication."""
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


class State(TypedDict, total=False):
    issue: Issue
    context: str
    patch: str
    attempt: int
    fixer_summary: str
    tester_summary: str
    test_logs: str
    test_passed: bool
    sonar_summary: str
    sonar_passed: bool
    pr_url: str
    branch: str
    feedback_log: str
    fix_failed: bool


Role = Literal["requester", "fixer", "tester", "sonar", "open_pr"]
