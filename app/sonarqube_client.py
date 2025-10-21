"""Client helpers for interacting with SonarQube's REST API."""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional

import requests

LOGGER = logging.getLogger(__name__)


@dataclass
class SonarIssue:
    """Minimal representation of a SonarQube issue."""

    key: str
    rule: str
    severity: str
    component: str
    message: str
    line: Optional[int]
    status: str
    effort: Optional[str]
    type: Optional[str]


class SonarQubeClient:
    """Thin wrapper around SonarQube's issue search API."""

    def __init__(self, host_url: Optional[str] = None, token: Optional[str] = None, project_key: Optional[str] = None) -> None:
        self.host_url = host_url or os.getenv("SONARQUBE_URL")
        self.token = token or os.getenv("SONARQUBE_TOKEN")
        self.project_key = project_key or os.getenv("SONAR_PROJECT_KEY")
        if not self.host_url:
            raise ValueError("SONARQUBE_URL is required")
        if not self.token:
            raise ValueError("SONARQUBE_TOKEN is required")
        if not self.project_key:
            raise ValueError("SONAR_PROJECT_KEY is required")
        self._session = requests.Session()
        self._session.auth = (self.token, "")

    def _url(self, path: str) -> str:
        return f"{self.host_url.rstrip('/')}{path}" if path.startswith("/") else f"{self.host_url.rstrip('/')}/{path}"

    def search_issues(
        self,
        severities: Iterable[str] | None = None,
        statuses: Iterable[str] | None = None,
        resolved: Optional[bool] = None,
    ) -> List[SonarIssue]:
        """Fetch project issues respecting severity, status and resolution filters."""
        params = {
            "componentKeys": self.project_key,
            "p": 1,
            "ps": 500,
        }
        if severities:
            params["severities"] = ",".join(severities)
        if statuses:
            params["statuses"] = ",".join(statuses)
        if resolved is not None:
            params["resolved"] = "true" if resolved else "false"
        issues: List[SonarIssue] = []
        while True:
            LOGGER.debug("Fetching Sonar issues page %s", params["p"])
            resp = self._session.get(self._url("/api/issues/search"), params=params, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
            raw_issues = payload.get("issues", [])
            for entry in raw_issues:
                text_range = entry.get("textRange") or {}
                issues.append(
                    SonarIssue(
                        key=entry.get("key", ""),
                        rule=entry.get("rule", ""),
                        severity=entry.get("severity", "UNKNOWN"),
                        component=entry.get("component", ""),
                        message=entry.get("message", ""),
                        line=text_range.get("startLine"),
                        status=entry.get("status", "UNKNOWN"),
                        effort=entry.get("effort"),
                        type=entry.get("type"),
                    )
                )
            paging = payload.get("paging", {})
            page_index = paging.get("pageIndex", params["p"])
            page_size = paging.get("pageSize", len(raw_issues))
            total = paging.get("total", len(raw_issues))
            fetched = page_index * page_size
            if fetched >= total or not raw_issues:
                break
            params["p"] += 1
        LOGGER.info("Fetched %d Sonar issues", len(issues))
        return issues

    def wait_for_ce_task(
        self,
        ce_task_id: str,
        *,
        timeout: float = 120.0,
        poll_interval: float = 2.0,
    ) -> dict:
        """Block until the Sonar background task reaches a terminal state."""
        if not ce_task_id:
            raise ValueError("ce_task_id must be provided")

        deadline = time.monotonic() + timeout
        last_status: Optional[str] = None

        while True:
            resp = self._session.get(
                self._url("/api/ce/task"),
                params={"id": ce_task_id},
                timeout=30,
            )
            resp.raise_for_status()
            payload = resp.json() or {}
            task = payload.get("task") or {}
            status = (task.get("status") or "").upper()

            if status == "SUCCESS":
                LOGGER.info("Sonar background task %s finalized with SUCCESS", ce_task_id)
                return task

            if status in {"FAILED", "CANCELED"}:
                message = task.get("errorMessage") or f"Status {status}"
                raise RuntimeError(
                    f"Sonar background task {ce_task_id} ended with {status}: {message}"
                )

            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timeout waiting for Sonar background task {ce_task_id}"
                    f" (last status: {status or 'UNKNOWN'})"
                )

            if status and status != last_status:
                LOGGER.debug("Sonar background task %s status -> %s", ce_task_id, status)
                last_status = status

            time.sleep(poll_interval)


def format_issue(issue: SonarIssue) -> str:
    """Readable representation for prompting."""
    location = f"{issue.component}:{issue.line}" if issue.line else issue.component
    return "\n".join([
        f'[{issue.severity}] {issue.rule} @ {location}',
        f'Status: {issue.status}',
        f'Message: {issue.message}',
    ])
