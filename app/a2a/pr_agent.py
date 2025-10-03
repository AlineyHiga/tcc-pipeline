"""PR agent: finalises the flow by committing and opening a pull request."""
from __future__ import annotations

import logging
from typing import Optional

from app.a2a_runtime import Agent, Message, Session

from app.a2a.protocol import Issue
from app.utils import create_pull_request, ensure_git_branch, git_commit_all

LOGGER = logging.getLogger(__name__)


class PRAgent(Agent):
    def __init__(self, base_branch: str = "main", auto_branch_prefix: str = "autofix") -> None:
        super().__init__(name="PR")
        self.base_branch = base_branch
        self.auto_branch_prefix = auto_branch_prefix

    def on_message(self, message: Message, session: Session) -> None:
        if message.type != "pr_request":
            LOGGER.debug("PR ignorou mensagem %s", message.type)
            return
        issue: Optional[Issue] = session.state.get("issue")  # type: ignore[assignment]
        if issue is None:
            raise ValueError("Issue não presente no estado da sessão")
        branch_name = session.state.get("branch") or f"{self.auto_branch_prefix}/{issue.key}"
        LOGGER.info("PR agent preparando branch %s", branch_name)
        try:
            ensure_git_branch(str(branch_name))
            git_commit_all(f"fix: auto remediation for {issue.key}")
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Falha ao preparar commit/branch: %s", exc)
            session.state["result"] = f"commit_failed:{exc}"
            session.end()
            return
        session.state["branch"] = str(branch_name)
        try:
            pr = create_pull_request(
                title=f"fix: AutoFix for {issue.key}",
                body=self._build_body(session),
                head=str(branch_name),
                base=self.base_branch,
            )
            pr_url = pr.get("html_url", "")
            session.state["pr_url"] = pr_url
            session.state["result"] = "pr_created"
            LOGGER.info("PR criado: %s", pr_url)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Falha ao criar PR: %s", exc)
            session.state["result"] = f"pr_failed:{exc}"
        finally:
            session.end()

    def _build_body(self, session: Session) -> str:
        issue: Issue = session.state.get("issue")  # type: ignore[assignment]
        tester_logs = session.state.get("test_logs", "")
        sonar_summary = session.state.get("sonar_summary", "")
        body_lines = [
            "Correção automática via pipeline AutoFix.",
            "",
            f"Issue Sonar: `{issue.key}`",
            "",
            "Resumo Sonar:",
            sonar_summary or "-",
            "",
            "Logs do Tester:",
            "```",
            tester_logs,
            "```",
        ]
        return "\n".join(body_lines)
