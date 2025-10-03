"""Requester agent: prepares Fixer context inside an A2A session."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from app.a2a_runtime import Agent, Message, Session
from app.a2a.protocol import Issue, SessionState

LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = """
Você é o Requester Agent em um fluxo AutoFix.
Dado um issue do SonarQube, feedback acumulado e o conteúdo do arquivo envolvido,
produza um resumo conciso contendo:
- Descrição legível do problema.
- Trechos relevantes do código.
- Metas de correção para o Fixer Agent.
Inclua feedback recebido do Tester ou do Sonar quando disponível.
Responda em português.
"""


def _component_to_path(component: str) -> Optional[Path]:
    if ":" in component:
        _, rel = component.split(":", 1)
        return Path(rel)
    return Path(component)


class RequesterAgent(Agent):
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.1, repo_root: Optional[Path] = None) -> None:
        from app.llm_client import LLMClient

        super().__init__(name="Requester")
        self.llm = LLMClient(role="requester", temperature=temperature)
        self.repo_root = repo_root or Path.cwd()

    # Public API --------------------------------------------------------------
    def on_message(self, message: Message, session: Session) -> None:
        state = session.state
        if message.type == "start_issue":
            LOGGER.info("Requester recebeu start_issue")
            self._dispatch_fix_request(session)
        elif message.type == "tester_feedback":
            LOGGER.info("Requester recebeu tester_feedback")
            feedback = message.body.get("summary", "")
            logs = message.body.get("logs", "")
            state["tester_summary"] = feedback
            state["test_logs"] = logs
            self._enqueue_feedback(state, f"Falhas do Tester:\n{feedback}")
            self._maybe_retry(session, reason="tester")
        elif message.type == "sonar_feedback":
            LOGGER.info("Requester recebeu sonar_feedback")
            summary = message.body.get("summary", "")
            state["sonar_summary"] = summary
            self._enqueue_feedback(state, f"Feedback Sonar:\n{summary}")
            self._maybe_retry(session, reason="sonar")
        elif message.type == "fix_failed":
            LOGGER.warning("Requester recebeu fix_failed")
            details = message.body.get("error", "Patch inválido gerado")
            self._enqueue_feedback(state, f"Falha ao aplicar patch:\n{details}")
            self._maybe_retry(session, reason="fix_failed")
        elif message.type == "session_end":
            LOGGER.info("Requester encerrando sessão (motivo externo)")
            session.end()
        else:
            LOGGER.debug("Requester ignorou mensagem %s", message.type)

    # Internal helpers -------------------------------------------------------
    def _enqueue_feedback(self, state: SessionState, text: str) -> None:
        log = state.get("feedback_log", "")
        if log:
            log += "\n\n"
        state["feedback_log"] = log + text

    def _maybe_retry(self, session: Session, reason: str) -> None:
        state = session.state
        attempt = int(state.get("attempt", 1))
        max_rounds = int(state.get("max_rounds", 1))
        if attempt >= max_rounds:
            LOGGER.info("Limite de tentativas atingido após %s", reason)
            state["result"] = f"failed_{reason}_max_rounds"
            session.end()
            return
        state["attempt"] = attempt + 1
        self._dispatch_fix_request(session)

    def _dispatch_fix_request(self, session: Session) -> None:
        state = session.state
        issue = state.get("issue")
        if not isinstance(issue, Issue):
            raise ValueError("Requester precisa do issue no estado da sessão")
        feedback = state.get("feedback_log", "")
        attempt = int(state.get("attempt", 1))
        context = self._build_context(issue, feedback, attempt)
        state["context"] = context
        LOGGER.info("Requester preparando tentativa %s", attempt)
        session.send(
            Message(
                type="fix_request",
                from_=self.name,
                to="Fixer",
                body={"context": context, "attempt": attempt},
            )
        )

    def _build_context(self, issue: Issue, feedback: str, attempt: int) -> str:
        file_contents = self._load_file_snippet(issue)
        prompt = (
            f"Tentativa: {attempt}\n"
            f"Issue: {issue.severity} - {issue.rule}\n"
            f"Local: {issue.component}:{issue.line}\n"
            f"Mensagem: {issue.message}\n"
            f"Feedback acumulado: {feedback or 'N/A'}\n"
            "Arquivo:\n" + file_contents
        )
        return self.llm.invoke(SYSTEM_PROMPT, prompt)

    def _load_file_snippet(self, issue: Issue) -> str:
        path = _component_to_path(issue.component)
        if not path:
            return "(Arquivo não encontrado)"
        full_path = self.repo_root / path
        if not full_path.exists():
            LOGGER.warning("Arquivo da issue %s não encontrado: %s", issue.key, full_path)
            return "(Arquivo não encontrado)"
        try:
            return full_path.read_text()
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Erro ao ler %s: %s", full_path, exc)
            return "(Erro ao ler arquivo)"
