"""Simple Tester agent that just runs pytest without property generation."""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple, Union

from app.a2a.protocol import State

LOGGER = logging.getLogger(__name__)


class SimpleTesterAgent:
    def __init__(self, temperature: float = 0.0, repo_root: Optional[Union[Path, str]] = None) -> None:
        root = repo_root or os.getenv("AUTOFIX_TARGET_ROOT") or os.getenv("A2A_REPO_ROOT") or Path.cwd()
        self.repo_root = Path(root).expanduser().resolve()
        run_linters_env = (os.getenv("TESTER_RUN_LINTERS") or "").strip().lower()
        if run_linters_env:
            self.run_linters = run_linters_env not in {"0", "false", "no"}
        else:
            self.run_linters = False
        LOGGER.debug("SimpleTester repo root set to %s", self.repo_root)

    def _run_command(self, command: List[str]) -> Tuple[bool, str, bool]:
        LOGGER.debug("SimpleTester executando comando: %s (cwd=%s)", command, self.repo_root)
        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                cwd=str(self.repo_root),
            )
        except FileNotFoundError as exc:
            missing = command[0]
            message = f"Comando '{missing}' não encontrado; teste será pulado. Detalhes: {exc}"
            LOGGER.info(message)
            return False, message, True
        output = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode == 0, output, False

    def run_suite(self) -> Tuple[bool, bool, str]:
        """Run pytest and optional linters."""
        pytest_ok, pytest_logs, pytest_missing = self._run_command(["pytest", "-q", "--disable-warnings"])

        lint_passed = True
        lint_sections: List[str] = []
        
        if self.run_linters:
            ruff_ok, ruff_logs, ruff_missing = self._run_command(["ruff", "check", "."])
            black_ok, black_logs, black_missing = self._run_command(["black", "--check", "."])

            if ruff_missing:
                ruff_logs = (ruff_logs.strip() + "\nComando ausente; lint pulado.").strip()
            if black_missing:
                black_logs = (black_logs.strip() + "\nComando ausente; lint pulado.").strip()

            lint_passed = (ruff_ok or ruff_missing) and (black_ok or black_missing)
            lint_sections.extend([
                "ruff check .:\n" + ruff_logs.strip(),
                "black --check .:\n" + black_logs.strip(),
            ])
        else:
            lint_sections.append("linters:\nLinters desativados (TESTER_RUN_LINTERS=0).")

        sections = ["pytest -q:\n" + pytest_logs.strip()] + lint_sections
        aggregate_logs = "\n\n".join(section.strip() for section in sections if section).strip()

        if pytest_missing:
            LOGGER.error("Pytest não está disponível; marcando suíte como falha.")
            pytest_ok = False
            
        return pytest_ok, lint_passed, aggregate_logs

    def invoke(self, state: State) -> State:
        """Run tests and update state."""
        LOGGER.info("SimpleTester executando testes...")
        
        # Run the test suite
        pytest_ok, lint_ok, logs = self.run_suite()
        overall_tests_ok = pytest_ok
        
        # Create summary
        if overall_tests_ok:
            summary = "Testes executados com sucesso."
        else:
            summary = "Testes falharam. Verifique os logs para detalhes."
            
        # Update state
        state.update({
            "test_passed": overall_tests_ok,
            "lint_passed": lint_ok,
            "test_output": logs,
            "test_logs": logs,
            "tester_summary": summary,
            "property_tests_passed": True,  # Skip property tests
            "property_summary": "Property tests skipped in simple mode",
        })
        
        LOGGER.info("SimpleTester resultado: pytest_ok=%s, lint_ok=%s", pytest_ok, lint_ok)
        return state