"""Sonar agent: reruns analysis and validates that issues were resolved."""
from __future__ import annotations

import logging
from typing import List

from app.a2a.protocol import Issue, State
from app.sonarqube_client import SonarQubeClient
from app.utils import run_sonar_scanner

LOGGER = logging.getLogger(__name__)


def invoke(state: State) -> State:
    LOGGER.info("Sonar agent executando nova análise")
    run_sonar_scanner()
    client = SonarQubeClient()
    issues = client.search_issues()

    target_components = {
        issue.component for issue in (state.get("issues_for_file") or []) if issue
    }
    if not target_components and state.get("issue"):
        target_components = {state["issue"].component}

    remaining: List[Issue] = []
    for item in issues:
        if item.component in target_components:
            remaining.append(
                Issue(
                    key=item.key,
                    rule=item.rule,
                    severity=item.severity,
                    component=item.component,
                    message=item.message,
                    line=item.line,
                )
            )

    if remaining:
        formatted = "\n".join(
            f"[{iss.severity}] {iss.rule} @ {iss.component}:{iss.line} — {iss.message}"
            for iss in remaining
        )
        summary = f"Issues remanescentes no arquivo:\n{formatted}"
    else:
        summary = "0 issues restantes para o arquivo alvo"

    state.update(
        {
            "sonar_passed": not remaining,
            "sonar_summary": summary,
        }
    )
    LOGGER.info("Sonar agent finalizado: %s", summary)
    return state


__all__ = ["invoke"]
