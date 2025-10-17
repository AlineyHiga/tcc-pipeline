"""Requester agent: groups issues per file and builds rich context for the Fixer."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.a2a.protocol import Issue, State
from app.llm_client import LLMClient

LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are the Requester Agent in an AutoFix pipeline.
Given a list of SonarQube issues for a single file and the file contents, craft a concise English briefing.
Structure the response with three short sections:
- Overview: one sentence summarising the affected file and the number of issues.
- Issues: bullet list where each item follows "Line <number>: <short description> (<rule>)".
- Notes: include only if there is tester or sonar feedback worth highlighting.
Keep the result under 150 words and avoid repetition.
"""


def _component_to_path(component: str) -> Optional[Path]:
    cleaned = component.strip()
    if not cleaned:
        return None
    cleaned = cleaned.replace("\\", "/")
    if ":" in cleaned:
        _, rel = cleaned.split(":", 1)
    else:
        rel = cleaned
    rel = rel.lstrip("/")
    if not rel:
        return None
    return Path(rel)


class RequesterAgent:
    """Aggregates multiple issues for a file and prepares Fixer context."""

    def __init__(self, temperature: float = 0.1, repo_root: Optional[Path] = None) -> None:
        self.llm = LLMClient(role="requester", temperature=temperature)
        env_root = os.getenv("A2A_REPO_ROOT")
        base = Path(repo_root) if repo_root else Path(env_root) if env_root else Path.cwd()
        self.repo_root = base.resolve()
        self.max_file_chars = int(os.getenv("REQUESTER_MAX_FILE_CHARS", "6000"))
        self.max_context_chars = int(os.getenv("REQUESTER_MAX_CONTEXT_CHARS", "9000"))
        self._last_file_path: Optional[Path] = None

    # Public API ----------------------------------------------------------
    def invoke(self, state: State) -> State:
        issues = list(state.get("issues") or [])
        if not issues:
            LOGGER.info("Requester não encontrou issues pendentes")
            state.update(
                {
                    "context": "No pending issues found by planner.",
                    "multi_issue_summary": "",
                }
            )
            return state

        processed: set[str] = set(state.get("processed_components") or [])
        component, grouped = self._select_component(issues, processed)
        if component is None:
            LOGGER.info("Requester: nenhum arquivo pendente após filtrar componentes processados")
            state.update(
                {
                    "context": "All files have already been processed.",
                    "multi_issue_summary": "",
                }
            )
            return state

        display_path, file_text = self._load_file(component)
        summary_items = self._format_issue_list(grouped)
        multi_issue_summary = "\n".join(summary_items)

        tester_feedback = state.get("tester_summary")
        tester_focus = state.get("tester_feedback_summary")
        sonar_feedback = state.get("sonar_summary")
        tester_generated = list(state.get("tester_generated_test_files") or [])

        prompt = self._build_prompt(
            display_path,
            grouped,
            file_text,
            tester_feedback,
            tester_focus,
            tester_generated,
            sonar_feedback,
        )
        llm_summary = self.llm.invoke(SYSTEM_PROMPT, prompt)

        tester_cases = list(state.get("tester_feedback_cases") or [])

        context = self._build_context(
            display_path,
            grouped,
            file_text,
            llm_summary,
            tester_focus,
            tester_cases,
            tester_generated,
        )

        processed.add(component)
        state.update(
            {
                "issue": grouped[0],
                "issues_for_file": grouped,
                "file_path": display_path,
                "context": self._truncate(context),
                "multi_issue_summary": multi_issue_summary,
                "processed_components": list(processed),
            }
        )
        return state

    # Internal helpers ----------------------------------------------------
    def _select_component(
        self, issues: Iterable[Issue], processed: set[str]
    ) -> Tuple[Optional[str], List[Issue]]:
        ordered = sorted(issues, key=lambda item: (item.component, item.line or 0, item.key))
        for issue in ordered:
            if issue.component in processed:
                continue
            component = issue.component
            grouped = [candidate for candidate in ordered if candidate.component == component]
            LOGGER.debug("Requester selecionou componente %s com %d issue(s)", component, len(grouped))
            return component, grouped
        return None, []

    def _load_file(self, component: str) -> Tuple[str, str]:
        path_hint = _component_to_path(component)
        if not path_hint:
            LOGGER.warning("Requester: componente %s sem caminho identificável", component)
            return component, "(Arquivo não encontrado)"

        candidates = self._candidate_roots()
        for root in candidates:
            candidate = (root / path_hint).resolve()
            if candidate.exists():
                try:
                    text = candidate.read_text()
                except Exception as exc:  # noqa: BLE001
                    LOGGER.error("Falha ao ler %s: %s", candidate, exc)
                    return self._display_path(candidate), "(Erro ao ler arquivo)"
                self._last_file_path = candidate
                trimmed = self._trim_file(text)
                return self._display_path(candidate), trimmed

        # Fallback for components that include extra directory prefixes.
        fallback = self._search_by_suffix(candidates, path_hint)
        if fallback:
            try:
                text = fallback.read_text()
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("Falha ao ler %s: %s", fallback, exc)
                return self._display_path(fallback), "(Erro ao ler arquivo)"
            self._last_file_path = fallback
            trimmed = self._trim_file(text)
            return self._display_path(fallback), trimmed

        LOGGER.warning("Requester não encontrou o arquivo para componente %s", component)
        return component, "(Arquivo não encontrado)"

    def _candidate_roots(self) -> List[Path]:
        roots: List[Path] = []

        def _register(path: Optional[Path]) -> None:
            if not path:
                return
            resolved = path.resolve()
            if resolved not in roots:
                roots.append(resolved)

        _register(self.repo_root)
        for parent in self.repo_root.parents:
            _register(parent)
        env_root = os.getenv("A2A_REPO_ROOT")
        if env_root:
            _register(Path(env_root))
        _register(Path.cwd())
        try:
            _register(Path(__file__).resolve().parents[3])
        except IndexError:  # pragma: no cover - defensive guard
            pass
        return roots

    def _search_by_suffix(self, roots: Iterable[Path], path_hint: Path) -> Optional[Path]:
        parts = tuple(part for part in path_hint.parts if part not in {"", "."})
        if not parts:
            return None
        for root in roots:
            try:
                for candidate in root.rglob(parts[-1]):
                    rel_parts = candidate.relative_to(root).parts
                    if not rel_parts:
                        continue
                    if tuple(rel_parts) == parts[-len(rel_parts) :]:
                        return candidate.resolve()
            except (OSError, RuntimeError) as exc:  # noqa: BLE001
                LOGGER.debug("Requester: falha ao buscar %s em %s (%s)", path_hint, root, exc)
                continue
        return None

    def _trim_file(self, text: str) -> str:
        if len(text) <= self.max_file_chars:
            return text
        LOGGER.debug("Requester truncating file content from %d chars", len(text))
        return text[: self.max_file_chars] + "\n... (conteúdo truncado)"

    def _display_path(self, path: Path) -> str:
        try:
            return path.relative_to(self.repo_root).as_posix()
        except ValueError:
            return path.as_posix()

    def _format_issue_list(self, issues: Iterable[Issue]) -> List[str]:
        rows: List[str] = []
        for idx, issue in enumerate(issues, start=1):
            line = issue.line if issue.line is not None else "?"
            message = " ".join(issue.message.split())
            rows.append(f"{idx}. [{issue.key}] Linha {line} — {message} ({issue.rule})")
        return rows

    def _format_feedback_case(self, case: Dict[str, Any]) -> str:
        if not isinstance(case, dict):
            return f"- {case}"
        case_type = case.get("type")
        message = str(case.get("message") or "").strip()
        if case_type == "property":
            prop = case.get("property") or "(propriedade)"
            stage = case.get("stage") or "property_violation"
            inputs = case.get("inputs") or {}
            return f"- {prop} [{stage}]: inputs={inputs} -> {message or 'violação detectada'}"
        if case_type == "pytest":
            return f"- Pytest: {message or 'falha registrada pelo pytest'}"
        return f"- {message or 'Falha reportada pelo tester'}"

    def _build_prompt(
        self,
        file_path: str,
        issues: List[Issue],
        file_text: str,
        tester_feedback: Optional[str],
        tester_focus: Optional[str],
        tester_generated: List[str],
        sonar_feedback: Optional[str],
    ) -> str:
        issue_lines = "\n".join(self._format_issue_list(issues))
        feedback_lines = []
        if tester_feedback:
            feedback_lines.append(f"Tester feedback: {tester_feedback.strip()}")
        if tester_focus:
            feedback_lines.append(f"Tester foco: {tester_focus.strip()}")
        if tester_generated:
            generated_text = ", ".join(tester_generated)
            feedback_lines.append(f"Testes gerados: {generated_text}")
        if sonar_feedback:
            feedback_lines.append(f"Sonar feedback: {sonar_feedback.strip()}")
        feedback_block = "\n".join(feedback_lines) if feedback_lines else ""
        return "\n".join(
            filter(
                None,
                (
                    f"Target file: {file_path}",
                    f"Issues detected ({len(issues)}):\n{issue_lines}",
                    feedback_block,
                    "File contents:\n" + file_text,
                ),
            )
        )

    def _build_context(
        self,
        file_path: str,
        issues: List[Issue],
        file_text: str,
        llm_summary: str,
        tester_focus: Optional[str],
        tester_cases: List[Dict[str, Any]],
        tester_generated: List[str],
    ) -> str:
        lines = [
            f"Arquivo alvo: {file_path}",
            f"Total de issues: {len(issues)}",
        ]
        lines.extend(self._format_issue_list(issues))
        if llm_summary.strip():
            lines.append("Requester summary:\n" + llm_summary.strip())
        if tester_focus:
            lines.append("Resumo do tester:\n" + tester_focus.strip())
        if tester_cases:
            formatted_cases = [self._format_feedback_case(case) for case in tester_cases[:3]]
            if formatted_cases:
                lines.append("Casos destacados pelo tester:\n" + "\n".join(formatted_cases))
        if tester_generated:
            lines.append("Arquivos de testes gerados:\n" + "\n".join(f"- {path}" for path in tester_generated))
        lines.append("Código completo:\n" + file_text)
        return "\n".join(lines)

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_context_chars:
            return text
        LOGGER.debug(
            "Requester truncating context from %d to %d chars",
            len(text),
            self.max_context_chars,
        )
        return text[: self.max_context_chars] + "\n... (contexto truncado)"


__all__ = ["RequesterAgent"]
