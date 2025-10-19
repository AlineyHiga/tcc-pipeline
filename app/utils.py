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
    repo_root = Path(__file__).resolve().parents[2]
    root_env = repo_root / ".env"
    local_env = repo_root / "tcc-pipeline" / ".env"
    if root_env.exists():
        load_dotenv(dotenv_path=root_env, override=False)
    load_dotenv(dotenv_path=local_env, override=True)
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    scanner_candidates: list[Path] = []
    explicit_bin = env.get("SONAR_SCANNER_BIN") or os.getenv("SONAR_SCANNER_BIN")
    if explicit_bin:
        scanner_candidates.append(Path(explicit_bin))

    system_binary = shutil.which("sonar-scanner")
    if system_binary:
        scanner_candidates.append(Path(system_binary))

    repo_root = Path(__file__).resolve().parents[2]
    bundled_binary = repo_root / "sonar-scanner-5.0.1.3006-linux" / "bin" / "sonar-scanner"
    if bundled_binary.exists():
        scanner_candidates.append(bundled_binary)

    scanner_path: Optional[Path] = None
    for candidate in scanner_candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            scanner_path = candidate
            break

    property_args: list[str] = []
    def _collect_property(prop: str, *env_keys: str) -> Optional[str]:
        for key in env_keys:
            value = env.get(key)
            if value:
                property_args.append(f"-D{prop}={value}")
                return value
        return None

    sonar_url = env.get("SONARQUBE_URL") or env.get("SONAR_HOST_URL")
    if not sonar_url:
        raise RuntimeError(
            "SONARQUBE_URL ou SONAR_HOST_URL não configurado. Defina o endpoint do SonarQube."
        )

    sonar_token = env.get("SONARQUBE_TOKEN") or env.get("SONAR_TOKEN")
    if sonar_token:
        env.setdefault("SONAR_TOKEN", sonar_token)

    project_key = _collect_property("sonar.projectKey", "SONAR_PROJECT_KEY", "SONARQUBE_PROJECT_KEY")
    if not project_key:
        config_path = Path(cwd).resolve() / "sonar-project.properties"
        if config_path.exists():
            LOGGER.debug("Usando sonar-project.properties em %s para definir sonar.projectKey", config_path)
        else:
            raise RuntimeError(
                "SONAR_PROJECT_KEY não definido e nenhum arquivo sonar-project.properties encontrado em "
                f"{config_path}. Configure a chave do projeto antes de executar o scanner."
            )
    _collect_property("sonar.projectName", "SONAR_PROJECT_NAME", "SONARQUBE_PROJECT_NAME")
    _collect_property("sonar.projectVersion", "SONAR_PROJECT_VERSION")
    _collect_property("sonar.sources", "SONAR_SOURCES")
    _collect_property("sonar.tests", "SONAR_TESTS")
    _collect_property("sonar.language", "SONAR_LANGUAGE")
    _collect_property("sonar.sourceEncoding", "SONAR_SOURCE_ENCODING")
    _collect_property("sonar.host.url", "SONAR_HOST_URL", "SONARQUBE_URL")

    if scanner_path:
        LOGGER.info("Usando sonar-scanner localizado em %s", scanner_path)
        cmd = [str(scanner_path)]
    else:
        docker_path = shutil.which("docker")
        if not docker_path:
            message = (
                "Não foi possível localizar o binário sonar-scanner nem Docker. "
                "Instale o sonar-scanner ou habilite Docker para executar a análise."
            )
            LOGGER.error(message)
            raise RuntimeError(message)
        LOGGER.info("sonar-scanner não encontrado; executando via imagem Docker oficial")
        sonar_host_for_container = sonar_url.replace("localhost", "host.docker.internal")
        cmd = [
            docker_path,
            "run",
            "--rm",
            "--network=host",
            "-e",
            f"SONAR_HOST_URL={sonar_host_for_container}",
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
    cmd.extend(property_args)
    LOGGER.info("Project key: %s", env.get('SONAR_PROJECT_KEY'))
    LOGGER.info("SonarQube URL: %s", sonar_url)
    LOGGER.debug("Executing command: %s", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        stdout = proc.stdout.strip()
        combined = "\n".join(part for part in [stdout, stderr] if part).strip()
        LOGGER.error("Sonar scanner failed: %s", combined or "(sem saída)")
        raise RuntimeError(combined or "sonar-scanner retornou código não zero")
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


def with_line_numbers(text: str, start: int = 1, delimiter: str = " | ") -> str:
    """Annotate text with line numbers for prompting contexts."""
    if not text:
        return ""
    lines = text.splitlines()
    if text.endswith("\n"):
        lines.append("")
    width = len(str(start + len(lines) - 1)) if lines else len(str(start))
    numbered = []
    for idx, line in enumerate(lines, start=start):
        numbered.append(f"{str(idx).rjust(width)}{delimiter}{line}")
    return "\n".join(numbered)
