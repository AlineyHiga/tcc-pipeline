"""Sonar agent: reruns analysis and validates that issues were resolved."""
from __future__ import annotations

import logging
import os
import time
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
        LOGGER.info("Executando sonar-scanner...")
        run_sonar_scanner(cwd=repo_root)
        LOGGER.info("Sonar-scanner concluído, aguardando processamento...")
        time.sleep(2)  # Give SonarQube time to process
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
            LOGGER.info(f"Aguardando processamento da tarefa CE: {ce_task_id}")
            client.wait_for_ce_task(ce_task_id)
            LOGGER.info("Processamento CE concluído")
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
        LOGGER.warning("Metadados do Sonar não contém ceTaskId; aguardando 3s antes de buscar issues")
        time.sleep(3)  # Extra wait when no CE task ID

    LOGGER.info("Buscando issues do SonarQube...")
    issues = client.search_issues(statuses=("OPEN", "REOPENED", "CONFIRMED"), resolved=False)
    LOGGER.info(f"SonarQube retornou {len(issues)} issues no total")
    if issues:
        formatted_issues = "\n\n".join(format_issue(issue) for issue in issues)
        LOGGER.debug("Issues retornadas pelo Sonar:\n%s", formatted_issues)
    else:
        LOGGER.debug("Nenhuma issue retornada pelo Sonar.")

    # Get the current file being processed
    current_file = state.get("file_path", "")
    if not current_file and state.get("issue"):
        current_file = getattr(state["issue"], "component", "")
    
    LOGGER.info(f"Sonar checking issues for file: {current_file}")
    
    remaining: List[Issue] = []
    for item in issues:
        # Include all issues from the current file being processed
        if current_file and (item.component == current_file or item.component.endswith(current_file)):
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
            LOGGER.debug(f"Found issue: {item.rule} @ line {item.line}: {item.message}")

    if remaining:
        formatted = "\n".join(
            f"[{iss.severity}] {iss.rule} @ {iss.component}:{iss.line} — {iss.message}"
            for iss in remaining
        )
        summary = f"Issues remanescentes no arquivo {current_file}:\n{formatted}"
        LOGGER.info(f"Found {len(remaining)} remaining issues in {current_file}")
    else:
        summary = f"0 issues restantes para o arquivo {current_file}"
        LOGGER.info(f"All issues resolved in {current_file}")

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
