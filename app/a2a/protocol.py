"""Common data structures for A2A agent communication."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, TypedDict


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
    issues: List[Issue]
    issues_for_file: List[Issue]
    plan_summary: str
    multi_issue_summary: str
    context: str
    patch: str
    fixer_summary: str
    executor_summary: str
    execution_failed: bool
    test_output: str
    tester_summary: str
    test_passed: bool
    lint_passed: bool
    sonar_summary: str
    sonar_passed: bool
    deployment_summary: str
    file_path: str
    pr_url: str
    branch: str
    feedback_log: str
    fix_failed: bool
    deployment_failed: bool


Role = Literal["requester", "fixer", "tester", "sonar", "deployment"]
