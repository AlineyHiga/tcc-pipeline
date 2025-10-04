"""Orchestrator for the AutoFix SonarQube + A2A pipeline."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

from app.a2a_runtime import Message, Session

from app.a2a import PRAgent, FixerAgent, Issue, RequesterAgent, SessionState, SonarAgent, TesterAgent
from app.sonarqube_client import SonarIssue, SonarQubeClient, format_issue
from app.utils import Utils

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_env_file(path: str | Path) -> None:
    """Populate missing environment variables from a .env file."""

    env_path = Path(path)
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


load_env_file(PROJECT_ROOT / ".env")

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

MAX_ROUNDS = int(os.getenv("MAX_ROUNDS", "3"))
ISSUE_SEVERITIES = [s.strip() for s in os.getenv("ISSUE_SEVERITIES", "MAJOR,CRITICAL").split(",") if s.strip()]
BASE_BRANCH = os.getenv("BASE_BRANCH", "main")
AUTO_BRANCH_PREFIX = os.getenv("AUTO_BRANCH_PREFIX", "autofix")


def sonar_issue_to_issue(issue: SonarIssue) -> Issue:
    return Issue(
        key=issue.key,
        rule=issue.rule,
        severity=issue.severity,
        component=issue.component,
        message=issue.message,
        line=issue.line,
    )


def run_session_for_issue(issue: Issue) -> SessionState:
    session = Session()
    session.state.update({
        "issue": issue,
        "attempt": 1,
        "max_rounds": MAX_ROUNDS,
        "feedback_log": "",
    })
    requester = RequesterAgent()
    fixer = FixerAgent()
    tester = TesterAgent()
    sonar = SonarAgent(ISSUE_SEVERITIES or None)
    pr_agent = PRAgent(base_branch=BASE_BRANCH, auto_branch_prefix=AUTO_BRANCH_PREFIX)
    for agent in (requester, fixer, tester, sonar, pr_agent):
        session.register(agent)
    session.send(Message(type="start_issue", from_="system", to=requester.name, body={}))
    return session.state


def run_pipeline() -> List[SessionState]:
    client = SonarQubeClient()
    LOGGER.info("Executando sonar-scanner inicial")
    scan_exit = Utils.run_sonar_scanner()
    if scan_exit is None:
        LOGGER.warning("sonar-scanner não encontrado; pulando scan inicial")
    elif scan_exit != 0:
        LOGGER.error("sonar-scanner falhou (exit=%d). Abortando pipeline", scan_exit)
        return []
    
    issues = client.search_issues(severities=ISSUE_SEVERITIES or None)
    LOGGER.info("%d issue(s) encontradas", len(issues))
    results: List[SessionState] = []
    for issue in issues:
        LOGGER.info("Iniciando sessão para %s", format_issue(issue).strip().replace("\n", " | "))
        state = run_session_for_issue(sonar_issue_to_issue(issue))
        results.append(state)
    return results


if __name__ == "__main__":
    try:
        outcomes = run_pipeline()
        LOGGER.info("Pipeline concluído para %d issue(s)", len(outcomes))
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Falha durante pipeline: %s", exc)
        raise
