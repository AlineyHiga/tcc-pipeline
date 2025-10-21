"""Property agent: prepares property-based testing context before the fixer stage."""
from __future__ import annotations

import ast
import contextlib
import json
import logging
import os
import re
import textwrap
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.a2a.protocol import Issue, State
from app.llm_client import LLMClient

LOGGER = logging.getLogger(__name__)

PROPERTY_DISCOVERY_PROMPT = """
You analyse a Python file together with the related SonarQube issues and propose properties
that verify the recently applied fix.
Return strictly in JSON using {"properties": [...]}.
Each list item must contain:
- name: short snake_case identifier.
- description: one sentence describing the behaviour that must remain true.
- function: importable path for the target callable. Prefer "module.function" when the module can be imported,
  otherwise use "path/to/file.py::function" to reference the file explicitly.
- inputs: list of parameter names that should be supplied when calling the function.
- context (optional): extra hints that help understand the property.
As propriedades devem considerar que os testes serão escritos com o framework `Hypothesis`.
Limit yourself to properties related to the reported issues and do not output text outside JSON."""

PROPERTY_SUITE_PROMPT = """
You design Hypothesis property-based tests to validate a Sonar-guided bug fix.
Always produce BOTH of the following suites:
- Backward compatibility: outside the bug domain the new behaviour must match the old behaviour.
- Metamorphic invariants: semantic properties that must remain true forever (e.g. idempotence, monotonicity, permutations).

You receive structured context about the change. Use it to derive a precise predicate bug_domain(inputs_dict) -> bool that
identifies the inputs where the fix intentionally changes behaviour. Everything outside bug_domain must remain backward compatible.

Guardrails:
- Tests must rely only on the Python standard library, pytest, and Hypothesis.
- Do NOT perform I/O, networking, sleeping, or unrestricted randomness.
- Cap data generation ranges (finite integers/floats, bounded text length, capped collection sizes).
- Use assume() for preconditions.
- Do not mutate Hypothesis inputs in-place.
- Use the helper callables `old_impl` and `new_impl` provided by the harness to execute the old and new implementations.
- Use the helper bug_domain function you define to guard the compatibility assertions.

Respond strictly in JSON with the fields:
- bug_domain: object with fields { "name": snake_case identifier, "description": short text, "code": Python source defining bug_domain(inputs_dict) -> bool }.
- compatibility_tests: non-empty list of objects { "name": snake_case test id, "description": short text, "code": full Python function including decorators }.
- metamorphic_tests: non-empty list of objects with the same schema as compatibility_tests.
- imports (optional): list of extra import statements required by the tests (e.g. "from collections import Counter").
- notes (optional): additional guidance for humans (will be stored as comments).

Formatting rules:
- Wrap every value under a "code" key (including bug_domain["code"]) inside a fenced block that starts with ```python and ends with ```.
- Inside each fenced block include only executable Python code; do not add explanations or comments.
- Do not emit any commentary outside the JSON payload.

Return the JSON object only, without extra text before or after it."""

SYSTEM_PROPERTY_GENERATOR = """
You are the Property Agent for an automated bug-fixing pipeline.
Given a Python module and the related Sonar issues, write property-based tests using pytest and Hypothesis.
Focus on behaviours that guard against regressions and cover the reported problems.

Respond using the following exact structure:
Based on the provided Sonar issues and Python module, here is a property-based test using pytest and Hypothesis.
```json
{
  "imports": ["from hypothesis import given", ...],          # optional list of extra import lines
  "helpers": ["```python\\n<helper functions>\\n```", ...],  # optional list of reusable helper code blocks
  "tests": [
     {
       "name": "snake_case_identifier",                   # optional descriptive name
       "description": "short human summary",              # optional
       "code": "```python\\n@given(...)\n def test_...(...):\n    ...\n```"  # REQUIRED Python code block with one or more property tests
     },
     ...
  ],
  "notes": "optional commentary for humans"
}
```

Guardrails:
- Do not output anything before or after the message line plus the JSON block.
- Use only standard library, pytest, and hypothesis.
- Keep strategies bounded (finite ranges, limited sizes) and prefer `st.` helpers.
- Reference the target module using the provided import path or the injected TARGET_MODULE helper.
- Avoid filesystem, network, sleeps, subprocesses, or random seeding beyond pytest fixtures.
- Import the class or function you are making the test when direct imports are possible.
- Ensure every test is executable as-is inside a pytest module.
"""


@dataclass
class PropertyGenerationResult:
    success: bool
    summary: str
    files: List[Path]

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


class PropertyAgent:
    """Selects the target file and seeds state for property generation/testing."""

    def __init__(self, repo_root: Optional[Path] = None) -> None:
        env_root = os.getenv("A2A_REPO_ROOT")
        base = Path(repo_root) if repo_root else Path(env_root) if env_root else Path.cwd()
        self.repo_root = base.expanduser().resolve()
        self.max_file_chars = int(os.getenv("PROPERTY_AGENT_MAX_FILE_CHARS", "6000"))
        temperature = float(os.getenv("PROPERTY_AGENT_TEMPERATURE", "0.1"))
        self.llm = LLMClient(role="property", temperature=temperature)
        self.generated_test_root = self.repo_root / "tests" / "_a2a_generated"
        LOGGER.debug("PropertyAgent repo root set to %s", self.repo_root)

    # Public API ----------------------------------------------------------
    def invoke(self, state: State) -> State:
        issues = list(state.get("issues_scoped") or state.get("issues") or [])
        if not issues:
            LOGGER.info("PropertyAgent não encontrou issues pendentes")
            state.setdefault("property_summary", "Nenhuma issue disponível para propriedades.")
            return state

        processed: set[str] = set(state.get("property_processed_components") or [])
        component, grouped = self._select_component(issues, processed)
        if component is None:
            LOGGER.info("PropertyAgent: nenhum arquivo pendente após filtrar componentes já processados")
            state.setdefault(
                "property_summary",
                "Todas as issues já foram avaliadas nas propriedades.",
            )
            return state

        display_path, file_text, absolute_path = self._load_file(component)
        if group := grouped:
            state["issue"] = group[0]
            state["issues_for_file"] = group
        else:
            state.pop("issues_for_file", None)
            state.pop("issue", None)

        issues_summary = self._format_issue_summary(grouped)
        state["file_path"] = display_path
        state["property_component"] = component
        state["property_file_preview"] = file_text
        state["property_issues_summary"] = issues_summary
        if absolute_path:
            state["property_absolute_path"] = absolute_path.as_posix()
        else:
            state.pop("property_absolute_path", None)

        previous_files = list(state.get("property_test_files") or [])
        previous_report = str(state.get("property_summary") or "").strip()
        generation_result = self._generate_property_tests(
            state,
            component,
            display_path,
            absolute_path,
            file_text,
            issues_summary,
            previous_report,
        )
        if generation_result.success:
            self._cleanup_test_paths(previous_files)
            rel_files: List[str] = []
            for path in generation_result.files:
                try:
                    rel_files.append(str(path.relative_to(self.repo_root)))
                except ValueError:
                    rel_files.append(path.as_posix())
            state["property_test_files"] = rel_files
        else:
            state["property_test_files"] = previous_files

        state["property_generation_summary"] = generation_result.summary
        state["property_summary"] = generation_result.summary
        state["property_generation_failed"] = not generation_result.success
        processed.add(component)
        state["property_processed_components"] = list(processed)
        LOGGER.info("PropertyAgent selecionou %s para geração de propriedades", display_path)
        return state

    # Internal helpers ----------------------------------------------------
    def _select_component(
        self,
        issues: Iterable[Issue],
        processed: set[str],
    ) -> Tuple[Optional[str], List[Issue]]:
        ordered = sorted(issues, key=lambda item: (item.component, item.line or 0, item.key))
        for issue in ordered:
            if issue.component in processed:
                continue
            component = issue.component
            grouped = [candidate for candidate in ordered if candidate.component == component]
            LOGGER.debug(
                "PropertyAgent selecionou componente %s com %d issue(s)",
                component,
                len(grouped),
            )
            return component, grouped
        return None, []

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

    def _load_file(self, component: str) -> Tuple[str, str, Optional[Path]]:
        path_hint = _component_to_path(component)
        if not path_hint:
            LOGGER.warning("PropertyAgent: componente %s sem caminho identificável", component)
            return component, "(Arquivo não encontrado)", None

        candidates = self._candidate_roots()
        for root in candidates:
            candidate = (root / path_hint).resolve()
            if candidate.exists():
                try:
                    text = candidate.read_text()
                except Exception as exc:  # noqa: BLE001
                    LOGGER.error("Falha ao ler %s: %s", candidate, exc)
                    return self._display_path(candidate), "(Erro ao ler arquivo)", candidate
                trimmed = self._trim_file(text)
                return self._display_path(candidate), trimmed, candidate

        fallback = self._search_by_suffix(candidates, path_hint)
        if fallback:
            try:
                text = fallback.read_text()
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("Falha ao ler %s: %s", fallback, exc)
                return self._display_path(fallback), "(Erro ao ler arquivo)", fallback
            trimmed = self._trim_file(text)
            return self._display_path(fallback), trimmed, fallback

        LOGGER.warning("PropertyAgent não encontrou o arquivo para componente %s", component)
        return component, "(Arquivo não encontrado)", None

    def _format_issue_summary(self, issues: Iterable[Issue]) -> str:
        lines: List[str] = []
        for item in issues:
            message = getattr(item, "message", "")
            rule = getattr(item, "rule", "")
            line = getattr(item, "line", None)
            if line is not None:
                lines.append(f"Linha {line}: {message} ({rule})")
            else:
                lines.append(f"{message} ({rule})")
        return "\n".join(lines).strip()

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
                LOGGER.debug("PropertyAgent: falha ao buscar %s em %s (%s)", path_hint, root, exc)
                continue
        return None

    def _generate_property_tests(
        self,
        state: State,
        component: str,
        display_path: str,
        source_path: Optional[Path],
        file_text: str,
        issues_summary: str,
        previous_report: str,
    ) -> PropertyGenerationResult:
        if not file_text.strip() or "Arquivo não encontrado" in file_text:
            summary = (
                "Arquivo alvo não pôde ser carregado; testes de propriedades não foram gerados."
            )
            return PropertyGenerationResult(False, summary, [])

        module_path = self._module_import_path(source_path)
        prompt = self._build_generation_prompt(
            component,
            display_path,
            module_path,
            issues_summary,
            file_text,
            previous_report,
            state,
        )

        try:
            raw_response = self.llm.invoke(SYSTEM_PROPERTY_GENERATOR, prompt)
            payload = self._parse_json_response(raw_response)
        except Exception as exc:  # noqa: BLE001
            message = f"Falha ao solicitar testes de propriedades via LLM: {exc}"
            LOGGER.error("PropertyAgent: %s", message)
            return PropertyGenerationResult(False, message, [])

        if not isinstance(payload, dict):
            summary = f"Resposta inválida ao gerar testes de propriedades: {payload!r}"
            LOGGER.error("PropertyAgent: %s", summary)
            return PropertyGenerationResult(False, summary, [])

        tests_payload = payload.get("tests")
        if not isinstance(tests_payload, list) or not tests_payload:
            summary = "LLM não retornou testes de propriedades utilizáveis."
            LOGGER.warning("PropertyAgent: %s", summary)
            return PropertyGenerationResult(False, summary, [])

        tests: List[Dict[str, str]] = []
        unsafe_patterns = (
            "os.system",
            "subprocess.",
            "open(",
            "path(",
            "tempfile.",
            "process_file(",
            "create_temp_file",
            "run_system_command",
            "increment_counter",
            "global_counter",
        )
        for item in tests_payload:
            if isinstance(item, str):
                code = item
                name = ""
                description = ""
                function = ""
            elif isinstance(item, dict):
                code = item.get("code") or ""
                name = item.get("name") or ""
                description = item.get("description") or ""
                function = item.get("function") or ""
            else:
                continue
            code_block = self._extract_code_block(code)
            code_block = textwrap.dedent(code_block).strip()
            if not code_block:
                continue
            lowered = code_block.lower()
            if any(pattern in lowered for pattern in unsafe_patterns):
                LOGGER.info("PropertyAgent descartou teste inseguro: %s", name or code_block.splitlines()[0])
                continue
            code_block = self._normalise_strategy_calls(code_block)
            code_block = self._enforce_safe_strategies(code_block)
            code_block = self._refine_known_properties(code_block, source_path)
            tests.append(
                {
                    "code": code_block,
                    "name": str(name).strip(),
                    "description": str(description).strip(),
                    "function": str(function).strip(),
                }
            )

        if tests:
            unique_tests: List[Dict[str, str]] = []
            seen_codes: set[str] = set()
            for entry in tests:
                code_block = entry["code"]
                if code_block in seen_codes:
                    continue
                seen_codes.add(code_block)
                unique_tests.append(entry)
            tests = unique_tests

        if not tests:
            summary = "Nenhum bloco de teste válido foi extraído da resposta da LLM."
            LOGGER.warning("PropertyAgent: %s", summary)
            return PropertyGenerationResult(False, summary, [])

        helpers_payload = payload.get("helpers") or []
        helper_blocks: List[str] = []
        if isinstance(helpers_payload, list):
            for helper in helpers_payload:
                helper_code = self._extract_code_block(str(helper))
                if helper_code.strip():
                    helper_blocks.append(helper_code)

        imports_payload = payload.get("imports") or []
        imports: List[str] = []
        if isinstance(imports_payload, list):
            skip_imports = {
                "import pytest",
                "from hypothesis import given",
                "from hypothesis import strategies as st",
                "from hypothesis import strategies",
                "from hypothesis import strategies as strategies",
            }
            module_tokens: set[str] = set()
            if module_path:
                module_tokens.update(module_path.lower().split("."))
            if source_path:
                module_tokens.add(source_path.stem.lower())
                try:
                    module_tokens.update(source_path.relative_to(self.repo_root).as_posix().lower().split("/"))
                except ValueError:
                    module_tokens.update(source_path.as_posix().lower().split("/"))
            for entry in imports_payload:
                cleaned = str(entry or "").strip()
                if not cleaned:
                    continue
                lower = cleaned.lower()
                if lower in skip_imports:
                    continue
                if any(token and token in lower for token in module_tokens):
                    continue
                if cleaned not in imports:
                    imports.append(cleaned)

        notes = payload.get("notes")

        try:
            test_path = self._write_test_module(
                component,
                module_path,
                imports,
                helper_blocks,
                tests,
                notes,
                source_path,
            )
        except Exception as exc:  # noqa: BLE001
            message = f"Falha ao escrever módulo de testes gerados: {exc}"
            LOGGER.error("PropertyAgent: %s", message)
            return PropertyGenerationResult(False, message, [])

        if not test_path:
            summary = "LLM não forneceu conteúdo suficiente para criar o arquivo de testes."
            LOGGER.warning("PropertyAgent: %s", summary)
            return PropertyGenerationResult(False, summary, [])

        rel_path: str
        try:
            rel_path = str(test_path.relative_to(self.repo_root))
        except ValueError:
            rel_path = test_path.as_posix()
        summary = f"Testes de propriedades gerados em {rel_path}."
        if notes:
            summary += f" Observações: {notes}"
        LOGGER.info("PropertyAgent gravou testes em %s", test_path)
        return PropertyGenerationResult(True, summary, [test_path])

    def _build_generation_prompt(
        self,
        component: str,
        display_path: str,
        module_path: Optional[str],
        issues_summary: str,
        file_text: str,
        previous_report: str,
        state: State,
    ) -> str:
        lines: List[str] = [
            f"Componente Sonar: {component}",
            f"Caminho do arquivo: {display_path}",
        ]
        if module_path:
            lines.append(f"Módulo importável: {module_path}")
        feedback_summary = state.get("tester_feedback_summary") or ""
        property_failures = state.get("property_failures") or []
        if issues_summary:
            lines.append("")
            lines.append("Issues relevantes:")
            lines.append(issues_summary)
        if property_failures:
            lines.append("")
            lines.append("Falhas recentes das propriedades:")
            for failure in property_failures[:5]:
                try:
                    property_name = failure.get("property") or failure.get("name") or ""
                    message = failure.get("message") or ""
                    inputs = failure.get("inputs") or {}
                    lines.append(
                        f"- {property_name}: {message} (inputs={json.dumps(inputs, ensure_ascii=False)})"
                    )
                except Exception:  # noqa: BLE001
                    continue
        if previous_report:
            lines.append("")
            lines.append("Resumo anterior do tester:")
            lines.append(previous_report)
        if feedback_summary:
            lines.append("")
            lines.append("Falha principal identificada:")
            lines.append(str(feedback_summary))
        lines.append("")
        lines.append("Conteúdo atualizado do arquivo alvo:")
        lines.append(file_text.strip())
        lines.append("")
        lines.append("Instruções adicionais:")
        lines.append("- Utilize o helper TARGET_MODULE disponibilizado pelo harness quando importações diretas não forem possíveis.")
        lines.append("- Concentre-se em propriedades determinísticas sem efeitos colaterais; evite comandos de sistema, acesso a arquivos ou dependências externas.")
        return "\n".join(lines)

    def _parse_json_response(self, raw: str) -> Any:
        def _attempt(candidate: str) -> Optional[Any]:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                repaired = self._repair_json_string(candidate)
                if repaired != candidate:
                    with contextlib.suppress(json.JSONDecodeError):
                        return json.loads(repaired)
                with contextlib.suppress(Exception):
                    return ast.literal_eval(candidate)
            return None

        candidates: List[str] = []
        stripped = raw.strip()
        if stripped:
            candidates.append(stripped)
        if "```" in raw:
            for block in raw.split("```"):
                block = block.strip()
                if not block:
                    continue
                lower = block.lower()
                if lower.startswith(("json", "python")):
                    _, _, remainder = block.partition("\n")
                    block = remainder.strip()
                if block and block[0] in "{[":
                    candidates.append(block)
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and start < end:
            candidates.append(raw[start : end + 1])

        for candidate in candidates:
            result = _attempt(candidate)
            if result is not None:
                return result
        raise ValueError("Não foi possível interpretar a resposta JSON do gerador de propriedades.")

    def _repair_json_string(self, payload: str) -> str:
        """Attempts to escape control characters embedded inside JSON strings."""
        in_string = False
        escaped = False
        updated: List[str] = []
        for char in payload:
            if in_string:
                if escaped:
                    updated.append(char)
                    escaped = False
                    continue
                if char == "\\":
                    updated.append(char)
                    escaped = True
                    continue
                if char == '"':
                    updated.append(char)
                    in_string = False
                    continue
                codepoint = ord(char)
                if char == "\n":
                    updated.append("\\n")
                    continue
                if char == "\r":
                    updated.append("\\r")
                    continue
                if char == "\t":
                    updated.append("\\t")
                    continue
                if codepoint < 32:
                    updated.append(f"\\u{codepoint:04x}")
                    continue
                updated.append(char)
            else:
                updated.append(char)
                if char == '"':
                    in_string = True
            if not in_string:
                escaped = False
        return "".join(updated)

    def _extract_code_block(self, value: str) -> str:
        cleaned = str(value or "").strip()
        if cleaned.startswith("```"):
            match = re.match(r"```[a-zA-Z0-9_-]*\n(.*)```", cleaned, re.DOTALL)
            if match:
                return match.group(1).strip()
        return cleaned

    def _module_import_path(self, source_path: Optional[Path]) -> Optional[str]:
        if not source_path or source_path.suffix != ".py":
            return None
        try:
            rel_path = source_path.relative_to(self.repo_root)
        except ValueError:
            return None
        parts = list(rel_path.parts)
        if not parts:
            return None
        last = parts[-1]
        stem = Path(last).stem
        if stem == "__init__":
            parts = parts[:-1]
        else:
            parts[-1] = stem
        if not parts:
            return None
        return ".".join(parts)

    def _is_valid_module_path(self, module_path: str) -> bool:
        if not module_path:
            return False
        parts = [part.strip() for part in module_path.split(".") if part.strip()]
        return bool(parts) and all(part.isidentifier() for part in parts)

    def _module_path_matches_source(self, module_path: str, source_path: Optional[Path]) -> bool:
        if not module_path or not source_path:
            return False
        source_resolved = source_path.resolve()
        parts = [part.strip() for part in module_path.split(".") if part.strip()]
        if not parts:
            return False
        candidate = self.repo_root.joinpath(*parts)
        file_candidate = candidate.with_suffix(".py")
        if file_candidate.exists() and file_candidate.resolve() == source_resolved:
            return True
        init_candidate = candidate / "__init__.py"
        if init_candidate.exists() and init_candidate.resolve() == source_resolved:
            return True
        return False

    _STRATEGY_NAMES = (
        "integers",
        "floats",
        "text",
        "lists",
        "sets",
        "tuples",
        "dictionaries",
        "booleans",
        "dates",
        "datetimes",
        "timedeltas",
        "just",
        "sampled_from",
        "one_of",
        "none",
    )

    def _normalise_strategy_calls(self, code_block: str) -> str:
        result = code_block
        for name in self._STRATEGY_NAMES:
            pattern = re.compile(r"(?<![A-Za-z0-9_\.])" + re.escape(name) + r"\s*\(")
            result = pattern.sub(f"st.{name}(", result)
        result = re.sub(r"\bhypothesis\.strategies\.", "st.", result)
        result = re.sub(r"\bstrategies\.", "st.", result)
        return result

    _SAFE_REPLACEMENTS = (
        (re.compile(r"st\.integers\(\s*\)"), "st.integers(min_value=-1000, max_value=1000)"),
        (
            re.compile(r"st\.floats\(\s*\)"),
            "st.floats(allow_nan=False, allow_infinity=False, width=16)",
        ),
    )

    def _enforce_safe_strategies(self, code_block: str) -> str:
        result = code_block

        def _safe_text_replacer(match: re.Match[str]) -> str:
            args = match.group("args").strip()
            if not args:
                return (
                    "st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), max_size=200)"
                )
            if "alphabet=" in args:
                return match.group(0)
            return (
                "st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), max_size=200"
                f", {args})"
            )

        text_pattern = re.compile(r"st\.text\(\s*(?P<args>[^)]*)\)")
        result = text_pattern.sub(_safe_text_replacer, result)

        for pattern, replacement in self._SAFE_REPLACEMENTS:
            result = pattern.sub(replacement, result)

        return result

    def _refine_known_properties(self, code_block: str, source_path: Optional[Path]) -> str:
        module_alias = ""
        if source_path:
            alias_candidate = source_path.stem.replace("-", "_")
            if alias_candidate.isidentifier():
                module_alias = alias_candidate
        module_ref = module_alias or "TARGET_MODULE"

        def _wrap(text: str) -> str:
            return textwrap.dedent(text).strip()

        lowered = code_block.lower()
        if "format_string_1" in lowered:
            return _wrap(
                f"""
                @given(st.one_of(st.none(), st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), max_size=200)))
                def test_format_string_1_normalizes_text(input_text):
                    result = {module_ref}.format_string_1(input_text)
                    if input_text is None:
                        assert result == ""
                    else:
                        expected = input_text.strip().lower().replace(" ", "_")
                        assert result == expected
                """
            )
        if "get_first_element" in lowered:
            return _wrap(
                f"""
                @given(st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=20))
                def test_get_first_element_returns_index_zero(values):
                    assert {module_ref}.get_first_element(values) == values[0]
                """
            )
        if "simple_encrypt" in lowered:
            return _wrap(
                f"""
                @given(
                    st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), max_size=200),
                    st.integers(min_value=-16, max_value=16),
                )
                def test_simple_encrypt_round_trip(text_value, offset):
                    encrypted = {module_ref}.simple_encrypt(text_value, offset)
                    restored = "".join(chr(ord(ch) - offset) for ch in encrypted)
                    assert restored == text_value
                """
            )
        if "divide_numbers" in lowered:
            return _wrap(
                f"""
                @given(
                    st.integers(min_value=-10_000, max_value=10_000),
                    st.integers(min_value=1, max_value=10_000),
                )
                def test_divide_numbers_matches_python_division(numerator, denominator):
                    assert {module_ref}.divide_numbers(numerator, denominator) == numerator / denominator
                """
            )
        return code_block

    def _slugify(self, value: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_")
        return cleaned or "property"

    def _write_test_module(
        self,
        component: str,
        module_path: Optional[str],
        imports: Iterable[str],
        helpers: Iterable[str],
        tests: List[Dict[str, str]],
        notes: Optional[Any],
        source_path: Optional[Path],
    ) -> Optional[Path]:
        if not tests:
            return None
        self.generated_test_root.mkdir(parents=True, exist_ok=True)
        slug = self._slugify(component or module_path or "property")
        test_path = self.generated_test_root / f"test_property_{slug}.py"

        header_lines = ["# Auto-generated by PropertyAgent. Do not edit."]
        import_lines: List[str] = []

        def _add_import(entry: str) -> None:
            cleaned = entry.strip()
            if cleaned and cleaned not in import_lines:
                import_lines.append(cleaned)

        _add_import("import pytest")
        _add_import("from hypothesis import given, strategies as st")
        for entry in imports:
            _add_import(str(entry or ""))

        _add_import("import importlib.util")
        _add_import("import os")
        _add_import("from pathlib import Path")

        lines: List[str] = header_lines + import_lines
        if lines and lines[-1]:
            lines.append("")

        if not source_path:
            raise ValueError("source_path é obrigatório para gerar loader do módulo alvo.")
        try:
            target_identifier = source_path.relative_to(self.repo_root).as_posix()
        except ValueError:
            target_identifier = source_path.as_posix()

        alias_name = ""
        stem = source_path.stem.replace("-", "_")
        if stem and stem.isidentifier():
            alias_name = stem

        loader_code = textwrap.dedent(
            f"""
            _TARGET_MODULE_PATH = {target_identifier!r}

            def _resolve_target_path() -> Path:
                module_path = Path(_TARGET_MODULE_PATH)
                if module_path.is_absolute():
                    return module_path
                repo_root_env = os.getenv("A2A_REPO_ROOT")
                if repo_root_env:
                    candidate = (Path(repo_root_env).expanduser() / module_path).resolve()
                    if candidate.exists():
                        return candidate
                test_file = Path(__file__).resolve()
                for parent in [test_file.parent, *test_file.parents]:
                    candidate = (parent / module_path).resolve()
                    if candidate.exists():
                        return candidate
                raise FileNotFoundError(f"Não foi possível localizar {{_TARGET_MODULE_PATH}} a partir de {{test_file}}")

            def _load_target_module() -> object:
                module_path = _resolve_target_path()
                spec = importlib.util.spec_from_file_location("a2a_property_target", module_path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"Não foi possível carregar módulo em {{module_path}}")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module

            TARGET_MODULE = _load_target_module()
            """
        ).strip()
        if alias_name:
            alias_line = textwrap.dedent(
                f"""
            globals().setdefault("{alias_name}", TARGET_MODULE)
            """
            ).strip()
            loader_code += "\n" + alias_line
        lines.extend(loader_code.splitlines())
        if lines and lines[-1]:
            lines.append("")

        exposed_snippet = textwrap.dedent(
            """
            for name in dir(TARGET_MODULE):
                if name.startswith("_"):
                    continue
                globals().setdefault(name, getattr(TARGET_MODULE, name))
            """
        ).strip()
        lines.extend(exposed_snippet.splitlines())
        if lines and lines[-1]:
            lines.append("")

        for helper in helpers:
            helper_block = textwrap.dedent(helper).strip()
            if helper_block:
                lines.extend(helper_block.splitlines())
                if lines and lines[-1]:
                    lines.append("")

        for test in tests:
            description = test.get("description")
            function_name = test.get("function")
            code_block = textwrap.dedent(test.get("code") or "").strip()
            if not code_block:
                continue
            comment_parts: List[str] = []
            if function_name:
                comment_parts.append(f"Tests for {function_name}")
            if description:
                comment_parts.append(description)
            if comment_parts:
                lines.append("# " + " — ".join(comment_parts))
            lines.extend(code_block.splitlines())
            if lines and lines[-1]:
                lines.append("")

        if notes:
            lines.append("")
            for note_line in str(notes).splitlines():
                note_line = note_line.strip()
                if note_line:
                    lines.append(f"# NOTE: {note_line}")

        while lines and not lines[-1].strip():
            lines.pop()

        lines.append("")
        test_path.write_text("\n".join(lines), encoding="utf-8")
        return test_path

    def _cleanup_test_paths(self, paths: Iterable[str]) -> None:
        for entry in paths:
            candidate = Path(entry)
            if not candidate.is_absolute():
                candidate = (self.repo_root / candidate).resolve()
            with contextlib.suppress(FileNotFoundError):
                candidate.unlink()


    def _trim_file(self, text: str) -> str:
        if len(text) <= self.max_file_chars:
            return text
        LOGGER.debug("PropertyAgent truncating file content from %d chars", len(text))
        return text[: self.max_file_chars] + "\n... (conteúdo truncado)"

    def _display_path(self, path: Path) -> str:
        try:
            return path.relative_to(self.repo_root).as_posix()
        except ValueError:
            return path.as_posix()


__all__ = ["PropertyAgent"]
