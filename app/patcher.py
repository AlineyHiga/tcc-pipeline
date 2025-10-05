"""Utilities for applying patches."""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

LOGGER = logging.getLogger(__name__)


class PatchApplicationError(RuntimeError):
    """Raised when a git patch fails to apply."""


def apply_patch(patch: str, cwd: str | Path | None = None) -> None:
    """Apply a unified diff to the working tree using `git apply`."""
    if not patch.strip():
        raise PatchApplicationError("Empty patch content")
    root = Path(cwd) if cwd else Path(os.getenv("A2A_REPO_ROOT", Path.cwd())).resolve()
    attempts = [
        ["git", "apply", "--whitespace=nowarn", "-"],
        ["git", "apply", "--ignore-space-change", "--whitespace=nowarn", "-"],
        ["git", "apply", "--ignore-whitespace", "--whitespace=nowarn", "-"],
    ]
    errors: list[str] = []

    for args in attempts:
        LOGGER.debug("Applying patch with command %s", " ".join(args[:-1]))
        proc = subprocess.run(  # noqa: PL subprocess-run
            args,
            input=patch.encode(),
            cwd=str(root),
            capture_output=True,
            check=False,
        )
        if proc.returncode == 0:
            LOGGER.info("Patch applied successfully with %s", " ".join(args[:-1]))
            return
        stderr = proc.stderr.decode()
        stdout = proc.stdout.decode()
        message = stderr or stdout or f"git apply failed with {' '.join(args[:-1])}"
        errors.append(message.strip())
        LOGGER.warning(
            "git apply attempt failed (exit=%s) using %s", proc.returncode, " ".join(args[:-1])
        )
    numbered_patch = "\n".join(
        f"{line_no:04d}: {line}" for line_no, line in enumerate(patch.splitlines(), start=1)
    )
    LOGGER.error("All git apply attempts failed. Patch content (line numbered):\n%s", numbered_patch)
    combined = "\n".join(errors)
    raise PatchApplicationError(combined or "git apply failed")
