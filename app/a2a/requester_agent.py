"""Requester agent: assembles rich context for the Fixer agent."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

from app.a2a.protocol import Issue, State
from app.llm_client import LLMClient

LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = """
Você é o Requester Agent em um fluxo AutoFix.
Dado um issue do SonarQube e o conteúdo do arquivo envolvido, produza uma resposta extremamente concisa usando inglês.
Organize a saída em no máximo três seções curtas:
- Problem: descreva o bug em uma única frase clara.
- Fix: explique em até duas frases objetivas o que precisa ser alterado.
- Notes: inclua somente se houver feedback relevante do Tester ou do Sonar.
Limite-se a ~120 palavras e evite redundâncias.
"""


def _component_to_path(component: str) -> Optional[Path]:
    if ":" in component:
        _, rel = component.split(":", 1)
        return Path(rel)
    return Path(component)


class RequesterAgent:
    def __init__(self, temperature: float = 0.1, repo_root: Optional[Path] = None) -> None:
        self.llm = LLMClient(role="requester", temperature=temperature)
        env_root = os.getenv("A2A_REPO_ROOT")
        base = Path(repo_root) if repo_root else Path(env_root) if env_root else Path.cwd()
        self.repo_root = base.resolve()
        self.max_file_chars = int(os.getenv("REQUESTER_MAX_FILE_CHARS", "6000"))
        self.context_radius = int(os.getenv("REQUESTER_CONTEXT_RADIUS", "120"))
        self.max_context_chars = int(os.getenv("REQUESTER_MAX_CONTEXT_CHARS", "8000"))
        self._last_file_path: Optional[Path] = None
        self._last_snippet: Optional[str] = None

    def invoke(self, state: State) -> State:
        issue = state["issue"]
        attempt = int(state.get("attempt", 1))
        feedback_log = state.get("feedback_log", "")
        tester_feedback = state.get("tester_summary")
        sonar_feedback = state.get("sonar_summary")

        prompt = self._build_prompt(issue, attempt, feedback_log, tester_feedback, sonar_feedback)
        summary = self.llm.invoke(SYSTEM_PROMPT, prompt)
        combined_context = self._build_concise_context(issue, attempt, feedback_log, tester_feedback, sonar_feedback, summary)
        combined_context = self._truncate_context(combined_context)

        state.update({
            "context": combined_context,
            "attempt": attempt,
            "feedback_log": feedback_log,
        })
        if "patch" in state:
            LOGGER.debug("Requester clearing stale patch before fixer run")
            state.pop("patch")
        return state

    # Internal helpers -------------------------------------------------------
    def _build_prompt(
        self,
        issue: Issue,
        attempt: int,
        feedback_log: str,
        tester_feedback: Optional[str],
        sonar_feedback: Optional[str],
    ) -> str:
        file_contents = self._load_file_snippet(issue)
        path_hint = _component_to_path(issue.component)
        target_path = self._format_target_path(path_hint)
        lines = [
            f"Tentativa: {attempt}",
            f"Issue: {issue.severity} - {issue.rule}",
            f"Local: {issue.component}:{issue.line}",
            (f"Arquivo alvo: {target_path}" if target_path else ""),
            f"Mensagem: {issue.message}",
            f"Feedback acumulado: {feedback_log or 'N/A'}",
            f"Feedback tester: {tester_feedback or 'N/A'}",
            f"Feedback sonar: {sonar_feedback or 'N/A'}",
            "Arquivo:\n" + file_contents,
        ]
        return "\n".join(filter(None, lines))

    def _build_concise_context(
        self,
        issue: Issue,
        attempt: int,
        feedback_log: str,
        tester_feedback: Optional[str],
        sonar_feedback: Optional[str],
        summary: str,
    ) -> str:
        target_path = self._format_target_path(_component_to_path(issue.component))
        location = target_path or issue.component
        if issue.line:
            location = f"{location}:{issue.line}"

        def _clean(value: Optional[str]) -> Optional[str]:
            if not value:
                return None
            cleaned = value.strip()
            if not cleaned or cleaned.upper() == "N/A":
                return None
            return cleaned

        parts: List[str] = [
            f"Attempt {attempt}",
            f"Issue {issue.key}: {issue.rule} ({issue.severity})",
            f"Location: {location}",
            f"Message: {issue.message.strip()}",
        ]
        if target_path:
            parts.append(f"Arquivo alvo: {target_path}")

        for label, value in (
            ("Prior feedback", _clean(feedback_log)),
            ("Tester feedback", _clean(tester_feedback)),
            ("Sonar feedback", _clean(sonar_feedback)),
        ):
            if value:
                parts.append(f"{label}: {value}")

        snippet = (self._last_snippet or "").strip() or "(Trecho indisponível)"
        summary_block = summary.strip()
        if summary_block:
            parts.append("Requester summary:\n" + summary_block)
        parts.append("Code excerpt:\n" + snippet)
        return "\n".join(parts)

    def _load_file_snippet(self, issue: Issue) -> str:
        self._last_file_path = None
        self._last_snippet = None
        path = _component_to_path(issue.component)
        if not path:
            missing = "(Arquivo não encontrado)"
            self._last_snippet = missing
            return missing
        candidates = self._candidate_roots()
        for root in candidates:
            full_path = (root / path).resolve()
            if not full_path.exists():
                continue
            try:
                self._last_file_path = full_path
                snippet = self._build_file_snippet(full_path.read_text(), issue.line)
                self._last_snippet = snippet
                return snippet
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("Failed to read %s: %s", full_path, exc)
                self._last_file_path = None
                error_msg = "(Erro ao ler arquivo)"
                self._last_snippet = error_msg
                return error_msg
        LOGGER.warning(
            "File for issue %s not found under roots: %s",
            issue.key,
            ", ".join(str(root) for root in candidates),
        )
        self._last_file_path = None
        missing = "(Arquivo não encontrado)"
        self._last_snippet = missing
        return missing

    def _candidate_roots(self) -> List[Path]:
        roots: List[Path] = []

        def _add(root: Optional[Path]) -> None:
            if not root:
                return
            resolved = root.resolve()
            if resolved not in roots:
                roots.append(resolved)

        _add(self.repo_root)
        for parent in self.repo_root.parents:
            _add(parent)
        env_root = os.getenv("A2A_REPO_ROOT")
        if env_root:
            _add(Path(env_root))
        _add(Path.cwd())
        try:
            _add(Path(__file__).resolve().parents[3])
        except IndexError:  # pragma: no cover - defensive guard
            pass
        return roots

    def _format_target_path(self, path_hint: Optional[Path]) -> Optional[str]:
        path = self._last_file_path
        if path:
            try:
                return path.relative_to(self.repo_root).as_posix()
            except ValueError:
                return path.as_posix()
        if path_hint:
            return path_hint.as_posix()
        return None

    def _build_file_snippet(self, file_text: str, line: Optional[int]) -> str:
        if len(file_text) <= self.max_file_chars:
            return file_text
        lines = file_text.splitlines()
        total_lines = len(lines)
        if line and 1 <= line <= total_lines:
            idx = line - 1
            radius = max(0, self.context_radius)
            start = max(0, idx - radius)
            end = min(total_lines, idx + radius + 1)
            snippet_lines = lines[start:end]
            snippet = "\n".join(snippet_lines)
            header = f"... (trecho de linhas {start + 1}-{end} de {total_lines})\n"
            footer = "\n... (conteúdo truncado; veja o arquivo completo no repositório)"
            result = header + snippet + footer
        else:
            result = file_text[: self.max_file_chars]
            result += "\n... (conteúdo truncado; veja o arquivo completo no repositório)"
        if len(result) > self.max_file_chars:
            result = result[: self.max_file_chars] + "\n... (trecho truncado)"
        return result

    def _truncate_context(self, context: str) -> str:
        if len(context) <= self.max_context_chars:
            return context
        truncated = context[: self.max_context_chars]
        return truncated + "\n... (contexto adicional truncado)"
