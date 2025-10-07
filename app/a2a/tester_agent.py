"""Tester agent: executes pytest/Hypothesis and explains the results."""
from __future__ import annotations

import logging
import subprocess
from typing import List, Tuple

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

    def _run_command(self, command: List[str]) -> Tuple[bool, str, bool]:
        LOGGER.debug("Tester executando comando: %s", command)
        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            missing = command[0]
            message = f"Comando '{missing}' não encontrado; lint será pulado. Detalhes: {exc}"
            LOGGER.info(message)
            return False, message, True
        output = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode == 0, output, False

    def run_suite(self) -> Tuple[bool, bool, str]:
        pytest_ok, pytest_logs, pytest_missing = self._run_command(["pytest", "-q", "--disable-warnings"])
        ruff_ok, ruff_logs, ruff_missing = self._run_command(["ruff", "check", "."])
        black_ok, black_logs, black_missing = self._run_command(["black", "--check", "."])

        if ruff_missing:
            ruff_logs = (ruff_logs.strip() + "\nComando ausente; lint pulado.").strip()
        if black_missing:
            black_logs = (black_logs.strip() + "\nComando ausente; lint pulado.").strip()

        aggregate_logs = "\n\n".join(
            [
                "pytest -q:\n" + pytest_logs.strip(),
                "ruff check .:\n" + ruff_logs.strip(),
                "black --check .:\n" + black_logs.strip(),
            ]
        ).strip()

        lint_passed = (ruff_ok or ruff_missing) and (black_ok or black_missing)
        if pytest_missing:
            LOGGER.error("Pytest não está disponível; marcando suíte como falha.")
            pytest_ok = False
        return pytest_ok, lint_passed, aggregate_logs

    def invoke(self, state: State) -> State:
        tests_ok, lint_ok, logs = self.run_suite()
        prompt = "Logs de teste e lint:\n" + logs
        summary = self.llm.invoke(SYSTEM_TESTER, prompt)
        state.update({
            "test_passed": tests_ok,
            "lint_passed": lint_ok,
            "test_output": logs,
            "test_logs": logs,
            "tester_summary": summary,
        })
        return state
