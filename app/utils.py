"""Utility helpers for the AutoFix pipeline."""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Mapping, Optional
from urllib.parse import urlparse, urlunparse

import requests

LOGGER = logging.getLogger(__name__)


def run_sonar_scanner(extra_env: Optional[Mapping[str, str]] = None, cwd: str | Path = ".") -> None:
    """Execute `sonar-scanner` locally or via the official Docker image."""
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    host_url = env.get("SONAR_HOST_URL") or env.get("SONARQUBE_URL")
    token = env.get("SONAR_TOKEN") or env.get("SONARQUBE_TOKEN")
    if host_url:
        env.setdefault("SONAR_HOST_URL", host_url)
    if token:
        env.setdefault("SONAR_TOKEN", token)
    scanner_path = shutil.which("sonar-scanner")
    if scanner_path:
        # Use system Java instead of bundled Java
        env["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-amd64"
        cmd = [scanner_path]
        # Add required properties as command line arguments
        project_key = env.get("SONAR_PROJECT_KEY")
        if project_key:
            cmd.extend(["-Dsonar.projectKey=" + project_key])
        if host_url:
            cmd.extend(["-Dsonar.host.url=" + host_url])
        if token:
            cmd.extend(["-Dsonar.token=" + token])
    else:
        LOGGER.info("sonar-scanner not found, falling back to Docker image")
        host_url = env.get("SONAR_HOST_URL")
        # When falling back to Docker, always use the original SONARQUBE_URL so we can adjust it for container networking.
        docker_host_url = os.getenv("SONARQUBE_URL") or env.get("SONAR_HOST_URL")
        if docker_host_url:
            parsed = urlparse(docker_host_url)
            if parsed.hostname in {"localhost", "127.0.0.1", "::1"}:
                parsed = parsed._replace(netloc=parsed.netloc.replace(parsed.hostname or "localhost", "host.docker.internal"))
            env["SONAR_HOST_URL"] = urlunparse(parsed)
        cmd = [
            "docker",
            "run",
            "--rm",
            "--add-host",
            "host.docker.internal:host-gateway",
            "-e",
            "SONAR_HOST_URL",
            "-e",
            "SONAR_TOKEN",
            "-v",
            f"{Path(cwd).resolve()}:/usr/src",
            "sonarsource/sonar-scanner-cli",
        ]
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
    resp = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=30)
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
