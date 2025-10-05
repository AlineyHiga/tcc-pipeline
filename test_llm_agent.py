#!/usr/bin/env python3
"""Quick smoke-test for the local LLM pipeline."""
from __future__ import annotations

from rich import print

from app.a2a import FixerAgent, Issue, RequesterAgent, TesterAgent


def main() -> None:
    issue = Issue(
        key="TEST-001",
        rule="python:S1481",
        severity="MAJOR",
        component="src/sample_module.py",
        message="Remove this unused local variable 'unused_var'",
        line=2,
    )

    state = {"issue": issue, "attempt": 1, "feedback_log": ""}

    print("[blue]🤖 Running requester")
    state = RequesterAgent().invoke(state)
    print(state.get("context"))

    print("\n[blue]🤖 Running fixer")
    state = FixerAgent().invoke(state)
    print(state.get("patch"))

    print("\n[blue]🤖 Running tester")
    state = TesterAgent().invoke(state)
    print(state.get("tester_summary"))


if __name__ == "__main__":
    main()
