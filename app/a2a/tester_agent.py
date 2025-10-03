"""Tester agent: runs pytest/Hypothesis and feeds back the results."""
from __future__ import annotations

import logging
import subprocess

from app.a2a_runtime import Agent, Message, Session

LOGGER = logging.getLogger(__name__)

SYSTEM_TESTER = """
Você é um engenheiro de testes.
Receberá logs de execução de testes baseados em propriedades.
Resuma falhas em linguagem clara para o Fixer.
Explique inputs que quebraram invariantes.
Se todos passaram, apenas confirme a validação.
"""


class TesterAgent(Agent):
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0) -> None:
        from app.llm_client import LLMClient

        super().__init__(name="Tester")
        self.llm = LLMClient(role="tester", temperature=temperature)

    def on_message(self, message: Message, session: Session) -> None:
        if message.type != "test_request":
            LOGGER.debug("Tester ignorou mensagem %s", message.type)
            return
        attempt = message.body.get("attempt")
        LOGGER.info("Tester executando pytest (tentativa %s)", attempt)
        passed, logs = self.run_tests()
        session.state["test_passed"] = passed
        session.state["test_logs"] = logs
        summary = self.llm.invoke(SYSTEM_TESTER, f"Logs de teste (pytest/Hypothesis):\n{logs}")
        session.state["tester_summary"] = summary
        if passed:
            session.send(
                Message(
                    type="sonar_request",
                    from_=self.name,
                    to="Sonar",
                    body={},
                )
            )
        else:
            session.send(
                Message(
                    type="tester_feedback",
                    from_=self.name,
                    to="Requester",
                    body={"summary": summary, "logs": logs},
                )
            )

    def run_tests(self) -> tuple[bool, str]:
        proc = subprocess.run(
            ["pytest", "-q", "--disable-warnings"],
            capture_output=True,
            text=True,
            check=False,
        )
        logs = (proc.stdout or "") + (proc.stderr or "")
        LOGGER.debug("pytest logs: %s", logs)
        return proc.returncode == 0, logs
