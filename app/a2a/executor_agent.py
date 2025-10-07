"""Executor agent: applies the patch and runs tests inside a sandbox container."""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

from app.a2a.protocol import State
LOGGER = logging.getLogger(__name__)


class ExecutorAgent:
    def __init__(self, repo_root: Optional[Path] = None) -> None:
        env_root = os.getenv("A2A_REPO_ROOT")
        base = Path(repo_root) if repo_root else Path(env_root) if env_root else Path.cwd()
        self.repo_root = base.resolve()
        self.docker_image = os.getenv("AUTOFIX_DOCKER_IMAGE", "").strip()

    def invoke(self, state: State) -> State:
        patch = state.get("patch")
        if not patch:
            message = "Patch ausente; executor não pode prosseguir"
            LOGGER.error(message)
            state.update(
                {
                    "executor_summary": message,
                    "execution_failed": True,
                }
            )
            return state

        LOGGER.info("Executor iniciando testes no sandbox Docker")
        command = self._build_command(patch)
        try:
            proc = subprocess.run(
                command,
                cwd=str(self.repo_root),
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:  # docker ausente
            LOGGER.warning("Docker não encontrado (%s); executando fallback local", exc)
            return self._run_local(state, patch)

        output = (proc.stdout or "") + (proc.stderr or "")
        success = proc.returncode == 0
        LOGGER.info("Executor Docker finalizado (status=%s)", proc.returncode)

        if not success:
            self._revert_patch(patch)

        state.update(
            {
                "executor_summary": "Testes executados no Docker" if success else "Falha ao executar testes no Docker",
                "execution_failed": not success,
                "test_output": output.strip(),
            }
        )
        return state

    # Helpers --------------------------------------------------------------
    def _build_command(self, patch: str) -> list[str]:
        if not self.docker_image:
            LOGGER.info("AUTOFIX_DOCKER_IMAGE não definido; executando testes localmente")
            return self._local_command(patch)

        script = (
            "set -euo pipefail\n"
            "cd /workspace\n"
            "git apply - <<'AUTOFIX_PATCH'\n"
            f"{patch}\n"
            "AUTOFIX_PATCH\n"
            "if [ -f requirements.txt ]; then pip install -r requirements.txt >/tmp/pip.log 2>&1 || { cat /tmp/pip.log; exit 1; }; fi\n"
            "pytest -q\n"
        )

        docker_cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{self.repo_root.as_posix()}:/workspace",
            "-w",
            "/workspace",
            self.docker_image,
            "bash",
            "-lc",
            script,
        ]
        LOGGER.debug("Executor docker command: %s", docker_cmd)
        return docker_cmd

    def _local_command(self, patch: str) -> list[str]:
        script = (
            "set -euo pipefail\n"
            "git apply - <<'AUTOFIX_PATCH'\n"
            f"{patch}\n"
            "AUTOFIX_PATCH\n"
            "if [ -f requirements.txt ]; then pip install -r requirements.txt >/tmp/pip.log 2>&1 || { cat /tmp/pip.log; exit 1; }; fi\n"
            "pytest -q\n"
        )
        return ["bash", "-lc", script]

    def _run_local(self, state: State, patch: str) -> State:
        LOGGER.info("Executor fallback: executando testes localmente")
        command = self._local_command(patch)
        proc = subprocess.run(
            command,
            cwd=str(self.repo_root),
            check=False,
            capture_output=True,
            text=True,
        )
        output = (proc.stdout or "") + (proc.stderr or "")
        success = proc.returncode == 0
        if not success:
            self._revert_patch(patch)
        state.update(
            {
                "executor_summary": "Testes executados localmente" if success else "Falha ao executar testes localmente",
                "execution_failed": not success,
                "test_output": output.strip(),
            }
        )
        return state

    def _revert_patch(self, patch: str) -> None:
        try:
            subprocess.run(
                ["git", "apply", "-R", "-"],
                input=patch,
                cwd=str(self.repo_root),
                check=False,
                text=True,
                capture_output=True,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Não foi possível reverter patch após falha: %s", exc)


__all__ = ["ExecutorAgent"]
