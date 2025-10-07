"""Planner agent: collects issues from SonarQube and seeds the pipeline state."""
from __future__ import annotations

import logging
from typing import List

from app.a2a.protocol import Issue, State
from app.sonarqube_client import SonarIssue, SonarQubeClient

LOGGER = logging.getLogger(__name__)


def _coerce_issue(item: SonarIssue) -> Issue:
    return Issue(
        key=item.key,
        rule=item.rule,
        severity=item.severity,
        component=item.component,
        message=item.message,
        line=item.line,
    )


def invoke(state: State) -> State:
    """Fetch all open issues from SonarQube and store them in the shared state."""

    client = SonarQubeClient()
    issues = client.search_issues()
    LOGGER.info("Planner recuperou %d issue(s) do SonarQube", len(issues))

    normalized: List[Issue] = [_coerce_issue(item) for item in issues]

    summary = (
        "Nenhuma issue pendente no SonarQube"
        if not normalized
        else f"{len(normalized)} issue(s) recuperadas do SonarQube"
    )

    state.update(
        {
            "issues": normalized,
            "plan_summary": summary,
        }
    )
    if normalized:
        state["issue"] = normalized[0]
    return state


__all__ = ["invoke"]
