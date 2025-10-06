
"""Utility helpers for the AutoFix pipeline."""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Mapping, Optional

import requests

LOGGER = logging.getLogger(__name__)


def run_sonar_scanner(extra_env: Optional[Mapping[str, str]] = None, cwd: str | Path = ".") -> None:
    """Execute `sonar-scanner` locally or via the official Docker image."""
    from dotenv import load_dotenv
    load_dotenv()
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    scanner_path = shutil.which("sonar-scanner")
    if scanner_path:
        cmd = [scanner_path]
    else:
        LOGGER.info("sonar-scanner not found, falling back to Docker image")
        cmd = [
            "docker",
            "run",
            "--rm",
            "--network=host",
            "-e",
            f"SONAR_HOST_URL={env.get('SONARQUBE_URL')}",
            "-e",
            f"SONAR_TOKEN={env.get('SONARQUBE_TOKEN')}",
            "-e",
            f"SONAR_PROJECT_KEY={env.get('SONAR_PROJECT_KEY')}",
            "-e",
            f"SONAR_PROJECT_NAME={env.get('SONAR_PROJECT_NAME', env.get('SONAR_PROJECT_KEY'))}",
            "-e",
            f"SONAR_SOURCES={env.get('SONAR_SOURCES', 'src')}",
            "-v",
            f"{Path(cwd).resolve()}:/usr/src",
            "sonarsource/sonar-scanner-cli",
        ]
    LOGGER.info("Project key: %s", env.get('SONAR_PROJECT_KEY'))
    LOGGER.info("SonarQube URL: %s", env.get('SONARQUBE_URL'))
    LOGGER.debug("Executing command: %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, check=False)
    if proc.returncode != 0:
        LOGGER.error("Sonar scanner failed: %s", proc.stderr.decode())
        raise RuntimeError(proc.stderr.decode())
    LOGGER.info("Sonar scanner completed successfully")


def ensure_git_branch(branch_name: str, cwd: str | Path = ".") -> None:
    """Create or check out the working branch used for the PR."""
    proc = subprocess.run(["git", "rev-parse", "--verify", branch_name], cwd=str(cwd), capture_output=True, check=False)
    if proc.returncode == 0:
        subprocess.run(["git", "checkout", branch_name], cwd=str(cwd), check=True)
    else:
        subprocess.run(["git", "checkout", "-b", branch_name], cwd=str(cwd), check=True)


def git_commit_all(message: str, cwd: str | Path = ".") -> None:
    subprocess.run(["git", "add", "-A"], cwd=str(cwd), check=True)
    proc = subprocess.run(["git", "diff", "--cached", "--stat"], cwd=str(cwd), capture_output=True, check=True)
    if not proc.stdout.strip():
        LOGGER.info("No changes to commit")
        return
    subprocess.run(["git", "commit", "-m", message], cwd=str(cwd), check=True)


def create_pull_request(
    title: str,
    body: str,
    head: str,
    base: str = "main",
    repository: Optional[str] = None,
    token: Optional[str] = None,
) -> dict:
    """Open a pull request using GitHub's REST API."""
    repo = repository or os.getenv("GITHUB_REPOSITORY")
    if not repo:
        raise ValueError("GITHUB_REPOSITORY not configured")
    api_url = f"https://api.github.com/repos/{repo}/pulls"
    payload = {"title": title, "body": body, "head": head, "base": base}
    headers = {"Accept": "application/vnd.github+json"}
    token = token or os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    LOGGER.debug(
        "Creating pull request payload=%s headers=%s",
        payload,
        {k: ('***' if k.lower() == 'authorization' else v) for k, v in headers.items()},
    )
    resp = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=30)
    LOGGER.debug("GitHub create PR response status=%s body=%s", resp.status_code, resp.text)
    if resp.status_code >= 400:
        raise RuntimeError(f"Failed to create PR: {resp.status_code} {resp.text}")
    return resp.json()


def format_issues_for_prompt(issues: Iterable[Mapping[str, object]]) -> str:
    """Render Sonar issues in a human prompt friendly format."""
    lines: list[str] = []
    for issue in issues:
        severity = issue.get("severity") or "UNKNOWN"
        rule = issue.get("rule") or ""
        component = issue.get("component") or ""
        message = issue.get("message") or ""
        line = issue.get("line")
        location = f"{component}:{line}" if line else str(component)
        lines.append(f"[{severity}] {rule} @ {location}\n{message}")
    return "\n\n".join(lines)
