"""Sonar agent: reruns analysis to validate patches."""
from __future__ import annotations

import logging
from typing import Iterable, Optional

from app.a2a_runtime import Agent, Message, Session

from app.sonarqube_client import SonarIssue, SonarQubeClient
from app.utils import format_issues_for_prompt, run_sonar_scanner

LOGGER = logging.getLogger(__name__)


class SonarAgent(Agent):
    def __init__(self, severities: Optional[Iterable[str]] = None) -> None:
        super().__init__(name="Sonar")
        self.severities = list(severities) if severities else None

    def on_message(self, message: Message, session: Session) -> None:
        if message.type != "sonar_request":
            LOGGER.debug("Sonar ignorou mensagem %s", message.type)
            return
        issue = session.state.get("issue")
        if not issue:
            raise ValueError("Issue não presente no estado da sessão")
        LOGGER.info("Sonar executando scanner para validar issue %s", issue.key)
        try:
            run_sonar_scanner()
            client = SonarQubeClient()
            issues = client.search_issues(severities=self.severities)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Falha ao executar sonar-scanner: %s", exc)
            summary = f"Falha ao executar sonar-scanner: {exc}"
            session.state["sonar_passed"] = False
            session.state["sonar_summary"] = summary
            session.send(
                Message(
                    type="sonar_feedback",
                    from_=self.name,
                    to="Requester",
                    body={"summary": summary},
                )
            )
            return
        remaining = [item for item in issues if self._matches_issue(item, issue.key)]
        if not remaining:
            LOGGER.info("Sonar não encontrou mais a issue %s", issue.key)
            session.state["sonar_passed"] = True
            session.state["sonar_summary"] = "Issue resolvido segundo SonarQube"
            session.send(
                Message(
                    type="pr_request",
                    from_=self.name,
                    to="PR",
                    body={},
                )
            )
            return
        summary = format_issues_for_prompt(
            {
                "severity": item.severity,
                "rule": item.rule,
                "component": item.component,
                "line": item.line,
                "message": item.message,
            }
            for item in remaining
        )
        session.state["sonar_passed"] = False
        session.state["sonar_summary"] = summary
        session.send(
            Message(
                type="sonar_feedback",
                from_=self.name,
                to="Requester",
                body={"summary": summary},
            )
        )

    @staticmethod
    def _matches_issue(issue: SonarIssue, key: str) -> bool:
        return issue.key == key
