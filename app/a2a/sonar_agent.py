"""Sonar agent: reruns analysis and validates that issues were resolved."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List

from app.a2a.protocol import Issue, State
from app.sonarqube_client import SonarQubeClient, format_issue
from app.utils import run_sonar_scanner

LOGGER = logging.getLogger(__name__)


def _resolve_repo_root(state: State) -> Path:
    root = state.get("repo_root") or os.getenv("AUTOFIX_TARGET_ROOT") or os.getenv("A2A_REPO_ROOT") or Path.cwd()
    return Path(root).expanduser().resolve()


def _read_report_metadata(repo_root: Path) -> Dict[str, str]:
    report_path = repo_root / ".scannerwork" / "report-task.txt"
    metadata: Dict[str, str] = {}
    try:
        with report_path.open("r", encoding="utf-8") as handler:
            for line in handler:
                line = line.strip()
                if not line or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                metadata[key] = value
    except FileNotFoundError:
        LOGGER.warning("Arquivo report-task.txt não encontrado em %s", report_path)
    except OSError as exc:  # pragma: no cover - defensive logging
        LOGGER.warning("Falha ao ler report-task.txt em %s: %s", report_path, exc)
    return metadata


def invoke(state: State) -> State:
    LOGGER.info("Sonar agent executando nova análise")
    repo_root = _resolve_repo_root(state)
    LOGGER.debug("Sonar agent running with repo root %s", repo_root)
    try:
        run_sonar_scanner(cwd=repo_root)
    except RuntimeError as exc:
        message = f"Falha ao executar sonar-scanner: {exc}"
        LOGGER.error(message)
        state.update(
            {
                "sonar_passed": False,
                "sonar_summary": message,
            }
        )
        return state
    metadata = _read_report_metadata(repo_root)
    client = SonarQubeClient()
    ce_task_id = metadata.get("ceTaskId")
    if ce_task_id:
        try:
            client.wait_for_ce_task(ce_task_id)
        except (RuntimeError, TimeoutError) as exc:
            message = f"Falha ao aguardar processamento do Sonar: {exc}"
            LOGGER.error(message)
            state.update(
                {
                    "sonar_passed": False,
                    "sonar_summary": message,
                }
            )
            return state
    else:
        LOGGER.warning("Metadados do Sonar não contém ceTaskId; prosseguindo sem aguardar a fila")

    issues = client.search_issues(statuses=("OPEN", "REOPENED", "CONFIRMED"), resolved=False)
    if issues:
        formatted_issues = "\n\n".join(format_issue(issue) for issue in issues)
        LOGGER.debug("Issues retornadas pelo Sonar:\n%s", formatted_issues)
    else:
        LOGGER.debug("Nenhuma issue retornada pelo Sonar.")

    target_components = {
        issue.component for issue in (state.get("issues_for_file") or []) if issue
    }
    if not target_components and state.get("issue"):
        target_components = {state["issue"].component}

    remaining: List[Issue] = []
    for item in issues:
        if item.component in target_components:
            status = (item.status or "").upper()
            if status in {"CLOSED", "RESOLVED"}:
                LOGGER.debug("Ignorando issue resolvida %s com status %s", item.key, status)
                continue
            remaining.append(
                Issue(
                    key=item.key,
                    rule=item.rule,
                    severity=item.severity,
                    component=item.component,
                    message=item.message,
                    line=item.line,
                )
            )

    if remaining:
        formatted = "\n".join(
            f"[{iss.severity}] {iss.rule} @ {iss.component}:{iss.line} — {iss.message}"
            for iss in remaining
        )
        summary = f"Issues remanescentes no arquivo:\n{formatted}"
    else:
        summary = "0 issues restantes para o arquivo alvo"

    state.update(
        {
            "sonar_passed": not remaining,
            "sonar_summary": summary,
            "sonar_remaining_issues": remaining,
        }
    )

    if remaining:
        LOGGER.info(
            "Sonar identificou %d issue(s) remanescente(s) para o arquivo alvo",
            len(remaining),
        )
        state["issues_scoped"] = remaining
        state["issues_for_file"] = remaining
        state["issue"] = remaining[0]
    else:
        state.pop("issues_scoped", None)
        state["issues_for_file"] = []
        state.pop("sonar_remaining_issues", None)

    LOGGER.info("Sonar agent finalizado: %s", summary)
    return state


__all__ = ["invoke"]
