"""Requester agent: groups issues per file and builds rich context for the Fixer."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.a2a.protocol import Issue, State
from app.llm_client import LLMClient
from app.utils import with_line_numbers

LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are the Requester Agent in an AutoFix pipeline.
Given a list of SonarQube issues for a single file and the file contents, craft a concise English briefing.
Structure the response with three short sections:
- Overview: one sentence summarising the affected file and the number of issues.
- Issues: bullet list where each item follows "Line <number>: <short description> (<rule>)".
- Instructions: a few sentences on how to fix the issues, focusing on the most critical ones.

- Notes: include tester, sonar, or fixer feedback when available (e.g., failures from previous fixer attempts).
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
        self.max_file_chars = int(os.getenv("REQUESTER_MAX_FILE_CHARS", "2000"))  # Reduced from 6000
        self.max_context_chars = int(os.getenv("REQUESTER_MAX_CONTEXT_CHARS", "4000"))  # Reduced from 9000
        self._last_file_path: Optional[Path] = None
        
        # Initialize enhanced RAG service with auto-build
        try:
            from app.rag_builder import auto_build_rag_index
            auto_build_rag_index(self.repo_root)
            
            from rag_service.service import RAGService
            self.rag_service = RAGService(str(self.repo_root / ".rag_index"))
        except (ImportError, Exception) as e:
            self.rag_service = None
            LOGGER.warning(f"Enhanced RAG service not available: {e}")

    # Public API ----------------------------------------------------------
    def invoke(self, state: State) -> State:
        issues = list(state.get("issues_scoped") or state.get("issues") or [])
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
        preferred_component = state.pop("property_component", None)
        ordered = sorted(issues, key=lambda item: (item.component, item.line or 0, item.key))
        grouped: List[Issue] = []
        component: Optional[str] = None
        if preferred_component:
            grouped = [candidate for candidate in ordered if candidate.component == preferred_component]
            if grouped:
                component = preferred_component
                LOGGER.debug(
                    "Requester reutilizando componente %s vindo do PropertyAgent",
                    preferred_component,
                )
            else:
                LOGGER.warning(
                    "Requester ignorou componente preferencial %s (não encontrado no conjunto de issues)",
                    preferred_component,
                )
        if component is None:
            component, grouped = self._select_component(ordered, processed)
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
        numbered_file_text = with_line_numbers(file_text)
        summary_items = self._format_issue_list(grouped)
        multi_issue_summary = "\n".join(summary_items)

        tester_feedback = self._compact_text(state.get("tester_summary"))
        tester_focus = self._compact_text(state.get("tester_feedback_summary"))
        sonar_feedback = self._compact_text(state.get("sonar_summary"))
        tester_generated = self._deduplicate_preserve_order(
            list(state.get("tester_generated_test_files") or [])
        )
        fixer_feedback = None
        if state.get("fix_failed"):
            fixer_feedback = self._compact_text(state.get("fixer_summary"))

        prompt = self._build_prompt(
            display_path,
            grouped,
            numbered_file_text,
            tester_feedback,
            tester_focus,
            tester_generated,
            sonar_feedback,
            fixer_feedback,
        )
        llm_summary = self._compact_text(self.llm.invoke(SYSTEM_PROMPT, prompt)) or ""

        tester_cases = list(state.get("tester_feedback_cases") or [])

        context = self._build_context(
            display_path,
            grouped,
            numbered_file_text,
            llm_summary,
            tester_focus,
            tester_cases,
            tester_generated,
            fixer_feedback,
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
                    rel_parts = tuple(part for part in candidate.relative_to(root).parts if part not in {"", "."})
                    if not rel_parts:
                        continue
                    if len(parts) <= len(rel_parts) and rel_parts[-len(parts) :] == parts:
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

    def _compact_text(self, text: Optional[str]) -> Optional[str]:
        if text is None:
            return None
        lines = text.splitlines()
        seen: set[str] = set()
        compacted: List[str] = []
        last_blank = True
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if not last_blank and compacted:
                    compacted.append("")
                last_blank = True
                continue
            normalized = " ".join(stripped.split()).lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            compacted.append(stripped)
            last_blank = False
        while compacted and not compacted[-1]:
            compacted.pop()
        return "\n".join(compacted) if compacted else None

    def _deduplicate_preserve_order(self, items: Iterable[str]) -> List[str]:
        seen: set[str] = set()
        ordered: List[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            ordered.append(item)
        return ordered

    def _build_prompt(
        self,
        file_path: str,
        issues: List[Issue],
        file_text_numbered: str,
        tester_feedback: Optional[str],
        tester_focus: Optional[str],
        tester_generated: List[str],
        sonar_feedback: Optional[str],
        fixer_feedback: Optional[str],
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
        if fixer_feedback:
            feedback_lines.append(f"Fixer falhou anteriormente: {fixer_feedback.strip()}")
        feedback_block = "\n".join(feedback_lines) if feedback_lines else ""
        return "\n".join(
            filter(
                None,
                (
                    f"Target file: {file_path}",
                    f"Issues detected ({len(issues)}):\n{issue_lines}",
                    feedback_block,
                    "File contents with line numbers:\n" + file_text_numbered,
                ),
            )
        )

    def _build_context(
        self,
        file_path: str,
        issues: List[Issue],
        file_text_numbered: str,
        llm_summary: str,
        tester_focus: Optional[str],
        tester_cases: List[Dict[str, Any]],
        tester_generated: List[str],
        fixer_feedback: Optional[str],
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
        if fixer_feedback:
            lines.append("Falha anterior do Fixer:\n" + fixer_feedback.strip())
        # Use enhanced RAG for minimal context
        if self.rag_service and issues:
            try:
                issue = issues[0]
                result = self.rag_service.retrieve_for_issue(
                    file_path=file_path,
                    line=getattr(issue, 'line', 1),
                    rule=getattr(issue, 'rule', ''),
                    message=getattr(issue, 'message', ''),
                    k=2  # Reduced from 5 to 2
                )
                
                # Get code for target symbol only (most relevant)
                target_id = f"{result['target']['path']}::{result['target']['symbol']}"
                code_map = self.rag_service.get_code_for_symbols([target_id])
                
                if code_map:
                    target_code = list(code_map.values())[0]
                    lines.append(f"Função alvo: {result['target']['symbol']}\n{target_code}")
                    LOGGER.info(f"RAG forneceu {len(target_code)} chars para {result['target']['symbol']}")
                else:
                    # Fallback to function extraction
                    lines.append("Código com numeração de linhas:\n" + file_text_numbered[:1500])
            except Exception as e:
                LOGGER.debug(f"RAG service error: {e}")
                # Fallback to function extraction
                lines.append("Código com numeração de linhas:\n" + file_text_numbered[:1200])
        else:
            if file_text_numbered.strip():
                lines.append("Código com numeração de linhas:\n" + file_text_numbered[:1200])  # Reduced from 2000
            else:
                lines.append("Código do arquivo indisponível.")
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
