"""Tester agent: executes pytest/Hypothesis and explains the results."""
from __future__ import annotations

import logging
import subprocess

from app.a2a.protocol import State
from app.llm_client import LLMClient

LOGGER = logging.getLogger(__name__)

SYSTEM_TESTER = """
Você é um engenheiro de testes.
Receberá logs de execução de testes baseados em propriedades.
Resuma falhas em linguagem clara para o Fixer.
Explique inputs que quebraram invariantes.
Se todos passaram, apenas confirme a validação.
"""


class TesterAgent:
    def __init__(self, temperature: float = 0.0) -> None:
        self.llm = LLMClient(role="tester", temperature=temperature)

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

    def invoke(self, state: State) -> State:
        passed, logs = self.run_tests()
        prompt = "Logs de teste (pytest/Hypothesis):\n" + logs
        summary = self.llm.invoke(SYSTEM_TESTER, prompt)
        state.update({
            "test_passed": passed,
            "test_logs": logs,
            "tester_summary": summary,
        })
        return state
