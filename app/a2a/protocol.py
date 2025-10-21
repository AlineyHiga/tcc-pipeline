"""Common data structures for A2A agent communication."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, TypedDict


@dataclass
class Issue:
    key: str
    rule: str
    severity: str
    component: str
    message: str
    line: Optional[int]


class State(TypedDict, total=False):
    abstract_properties: List["AbstractProperty"]
    property_tests_passed: bool
    property_attempts: int
    property_component: str
    property_processed_components: List[str]
    property_test_files: List[str]
    property_generation_summary: str
    property_generation_failed: bool
    property_file_preview: str
    property_absolute_path: str
    property_checks: List["PropertyCheck"]
    property_check_errors: List[str]
    property_generators: List[str]
    property_inputs: List["PropertyInput"]
    property_failures: List[Dict[str, Any]]
    property_summary: str
    public_examples: List["PropertyExample"]
    known_failures: List["PropertyExample"]
    issue: Issue
    issues: List[Issue]
    issues_for_file: List[Issue]
    issues_scoped: List[Issue]
    repo_root: str
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
    sonar_remaining_issues: List[Issue]
    sonar_feedback_attempts: int
    deployment_summary: str
    file_path: str
    tester_file_contents: str
    tester_issue_summary: str
    tester_feedback_cases: List[Dict[str, Any]]
    tester_primary_failure: Dict[str, Any]
    tester_feedback_summary: str
    tester_generated_test_files: List[str]
    pr_url: str
    branch: str
    feedback_log: str
    fix_failed: bool
    deployment_failed: bool
    metrics: "Metrics"


Role = Literal["requester", "fixer", "tester", "sonar", "deployment"]


class AbstractProperty(TypedDict, total=False):
    name: str
    description: str
    target: str
    function: str
    inputs: List[str]
    context: str


class PropertyExample(TypedDict):
    property_name: Optional[str]
    inputs: Dict[str, Any]
    output: Any


class PropertyInput(TypedDict, total=False):
    property_name: str
    inputs: Dict[str, Any]


class PropertyCheck(TypedDict, total=False):
    name: str
    code: str
    function_name: str
    description: str


class Metrics(TypedDict, total=False):
    timings: Dict[str, float]
    attempts: Dict[str, int]
    counters: Dict[str, int]
