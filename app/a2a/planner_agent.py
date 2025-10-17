"""Planner agent: collects issues from SonarQube and seeds the pipeline state."""
from __future__ import annotations

import logging
import os
from typing import Iterable, List, Optional, Sequence, Tuple

from app.a2a.protocol import Issue, State
from app.sonarqube_client import SonarIssue, SonarQubeClient

LOGGER = logging.getLogger(__name__)

DEFAULT_STATUSES: Tuple[str, ...] = ("OPEN", "REOPENED", "CONFIRMED")
RESOLVED_STATUSES: Tuple[str, ...] = ("CLOSED", "RESOLVED")


def _coerce_issue(item: SonarIssue) -> Issue:
    return Issue(
        key=item.key,
        rule=item.rule,
        severity=item.severity,
        component=item.component,
        message=item.message,
        line=item.line,
    )


def _parse_env_list(value: Optional[str]) -> Tuple[str, ...]:
    if not value:
        return ()
    parts = [element.strip().upper() for element in value.split(",")]
    return tuple(part for part in parts if part)


def _filter_closed(issues: Iterable[SonarIssue]) -> List[SonarIssue]:
    remaining: List[SonarIssue] = []
    skipped = 0
    for item in issues:
        status = (item.status or "").upper()
        if status in RESOLVED_STATUSES:
            skipped += 1
            LOGGER.debug("Planner ignorando issue %s com status resolvido %s", item.key, status)
            continue
        remaining.append(item)
    if skipped:
        LOGGER.info("Planner descartou %d issue(s) resolvidas", skipped)
    return remaining


def invoke(state: State) -> State:
    """Fetch open issues from SonarQube, respecting configured severities."""

    severities_env = os.getenv("ISSUE_SEVERITIES")
    severities: Sequence[str] = _parse_env_list(severities_env)

    client = SonarQubeClient()
    issues = client.search_issues(
        severities=severities or None,
        statuses=DEFAULT_STATUSES,
        resolved=False,
    )
    LOGGER.info("Planner recuperou %d issue(s) do SonarQube (pré-filtragem)", len(issues))

    filtered = _filter_closed(issues)
    LOGGER.info("Planner mantém %d issue(s) após filtrar resolvidas", len(filtered))

    normalized: List[Issue] = [_coerce_issue(item) for item in filtered]

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
