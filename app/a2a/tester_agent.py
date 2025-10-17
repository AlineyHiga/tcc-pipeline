"""Tester agent: executes pytest/Hypothesis and explains the results."""
from __future__ import annotations

import ast
import contextlib
import importlib
import importlib.util
import json
import logging
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from app.a2a.protocol import (
    AbstractProperty,
    PropertyCheck,
    PropertyExample,
    PropertyInput,
    State,
)
from app.llm_client import LLMClient

LOGGER = logging.getLogger(__name__)

SYSTEM_TESTER = """
You are a testing engineer focused on validating fixes.
You will receive logs from property-based test runs.
Report failures in clear language, explaining which inputs violated the property or test.
Only suggest new tests or validation scenarios when they are necessary.
Do not propose code changes or refactorings to the patched file.
If every test passes, confirm that validation succeeded.
"""

PROPERTY_SYSTEM_PROMPT = """
You convert abstract properties into executable checkers.
For each property, produce a Python snippet with a single function that receives
`inputs: dict` and `output: Any` and returns True when the property holds, otherwise False.
Respond strictly in JSON with the fields:
- name: short snake_case identifier for the property.
- description: one-sentence summary of what the checker enforces.
- function_name: name of the function defined in the code.
- code: Python code (standard library only) defining the requested function.
Do not include text outside the JSON payload.
"""

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
Limit yourself to properties related to the reported issues and do not output text outside JSON."""

PBT_INPUT_PROMPT = """
You synthesise additional input data to test properties.
You will receive the context of a validated property and must answer in JSON with the field `inputs`,
containing a list of dictionaries that represent extra calls exploring diverse scenarios.
Keep values as Python literals (int, float, bool, str, simple lists and dicts).
Do not include text outside JSON.
"""


class TesterAgent:
    def __init__(self, temperature: float = 0.0, repo_root: Optional[Union[Path, str]] = None) -> None:
        self.llm = LLMClient(role="tester", temperature=temperature)
        root = repo_root or os.getenv("AUTOFIX_TARGET_ROOT") or os.getenv("A2A_REPO_ROOT") or Path.cwd()
        self.repo_root = Path(root).expanduser().resolve()
        self.generated_test_root = self.repo_root / "tests" / "_a2a_generated"
        self._active_test_files: List[Path] = []
        run_linters_env = (os.getenv("TESTER_RUN_LINTERS") or "").strip().lower()
        if run_linters_env:
            self.run_linters = run_linters_env not in {"0", "false", "no"}
        else:
            self.run_linters = False
        LOGGER.debug("Tester repo root set to %s", self.repo_root)

    @contextlib.contextmanager
    def _syspath_context(self) -> Iterable[None]:
        extras: List[str] = []
        for candidate in {self.repo_root, self.repo_root / "src"}:
            if not candidate.exists():
                continue
            candidate_str = str(candidate)
            if candidate_str in sys.path:
                continue
            sys.path.insert(0, candidate_str)
            extras.append(candidate_str)
        try:
            yield
        finally:
            for item in extras:
                with contextlib.suppress(ValueError):
                    sys.path.remove(item)

    def _run_command(self, command: List[str]) -> Tuple[bool, str, bool]:
        LOGGER.debug("Tester executando comando: %s (cwd=%s)", command, self.repo_root)
        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                cwd=str(self.repo_root),
            )
        except FileNotFoundError as exc:
            missing = command[0]
            message = f"Comando '{missing}' não encontrado; lint será pulado. Detalhes: {exc}"
            LOGGER.info(message)
            return False, message, True
        output = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode == 0, output, False

    # Property-based testing helpers -------------------------------------
    def _parse_json_response(self, raw: str, label: str) -> Any:
        def _attempt(candidate: str) -> Optional[Any]:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                with contextlib.suppress(Exception):
                    return ast.literal_eval(candidate)
            return None

        def _collect_candidate(candidate: str, holder: List[str]) -> None:
            candidate = candidate.strip()
            if candidate and candidate not in holder:
                holder.append(candidate)

        def _extract_balanced_segment(text: str, opener: str, closer: str) -> Optional[str]:
            start = text.find(opener)
            while start != -1:
                depth = 0
                for index in range(start, len(text)):
                    char = text[index]
                    if char == opener:
                        depth += 1
                    elif char == closer:
                        depth -= 1
                        if depth == 0:
                            return text[start : index + 1]
                start = text.find(opener, start + 1)
            return None

        candidates: List[str] = []
        stripped = raw.strip()
        if stripped:
            _collect_candidate(stripped, candidates)

        if "```" in raw:
            for block in raw.split("```"):
                block = block.strip()
                if not block:
                    continue
                lower = block.lower()
                if lower.startswith(("json", "python")):
                    _, _, remainder = block.partition("\n")
                    block = remainder.strip() or ""
                if block and block[0] in "{[":
                    _collect_candidate(block, candidates)

        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and start < end:
            snippet = raw[start : end + 1]
            _collect_candidate(snippet, candidates)

        list_snippet = _extract_balanced_segment(raw, "[", "]")
        if list_snippet:
            _collect_candidate(list_snippet, candidates)

        for candidate in candidates:
            result = _attempt(candidate)
            if result is not None:
                return result

        raise ValueError(
            f"Resposta inválida ao gerar {label}: não foi possível interpretar o resultado."
        )

    def _filter_examples(self, property_name: str, items: Iterable[PropertyExample]) -> List[PropertyExample]:
        results: List[PropertyExample] = []
        for item in items:
            target = item.get("property_name")
            if target and target != property_name:
                continue
            if "inputs" not in item or "output" not in item:
                continue
            results.append(item)
        return results

    def _format_examples_for_prompt(self, examples: Iterable[PropertyExample], limit: int = 3) -> str:
        lines: List[str] = []
        for index, example in enumerate(examples):
            if index >= limit:
                break
            inputs = example.get("inputs", {})
            output = example.get("output")
            lines.append(
                f"- inputs={json.dumps(inputs, ensure_ascii=False)} -> output={json.dumps(output, ensure_ascii=False)}"
            )
        return "\n".join(lines) or "- (nenhum exemplo disponível)"

    def _render_property_prompt(
        self,
        prop: AbstractProperty,
        positive_examples: Iterable[PropertyExample],
        negative_examples: Iterable[PropertyExample],
    ) -> str:
        name = prop.get("name") or prop.get("function") or prop.get("target") or "unknown_property"
        description = prop.get("description") or "(sem descrição fornecida)"
        target = prop.get("function") or prop.get("target") or "(desconhecido)"
        inputs = ", ".join(prop.get("inputs") or []) or "(não informado)"
        context = prop.get("context") or ""
        formatted_positive = self._format_examples_for_prompt(positive_examples)
        formatted_negative = self._format_examples_for_prompt(negative_examples)
        prompt = textwrap.dedent(
            f"""
            Nome: {name}
            Descrição: {description}
            Função alvo: {target}
            Parâmetros esperados: {inputs}
            Contexto adicional: {context or '(vazio)'}

            Exemplos válidos conhecidos:
            {formatted_positive}

            Exemplos que devem falhar (saída incorreta):
            {formatted_negative}
            """
        ).strip()
        return prompt

    def _property_identifier(self, prop: AbstractProperty) -> str:
        name = (prop.get("name") or "").strip()
        target = (prop.get("function") or prop.get("target") or "").strip()
        if name and target:
            return f"{name}::{target}"
        return name or target or "property"

    def _property_cache_key(self, property_name: str) -> str:
        return property_name.strip().lower()

    def _deduplicate_property_inputs(self, items: Iterable[PropertyInput]) -> List[PropertyInput]:
        seen: set[Tuple[str, str]] = set()
        result: List[PropertyInput] = []
        for item in items:
            prop_name = (item.get("property_name") or "").strip()
            key = (prop_name, json.dumps(item.get("inputs", {}), sort_keys=True, ensure_ascii=False))
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
        return result

    def _maybe_generate_properties(
        self,
        state: State,
        file_contents: str,
        issues_summary: str,
    ) -> None:
        if state.get("abstract_properties"):
            return
        target_file = str(state.get("file_path") or "").strip()
        prompt = textwrap.dedent(
            f"""
            Arquivo alvo: {target_file or '(desconhecido)'}
            Issues relevantes:
            {issues_summary.strip() or '(nenhuma issue disponível)'}

            Código corrigido:
            {file_contents.strip()}
            """
        ).strip()
        try:
            raw = self.llm.invoke(PROPERTY_DISCOVERY_PROMPT, prompt)
            data = self._parse_json_response(raw, "propriedades PBT")
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Falha ao gerar propriedades via LLM: %s", exc)
            return

        candidate_props: Any = data.get("properties") if isinstance(data, dict) else data
        if not isinstance(candidate_props, list):
            LOGGER.warning("LLM não retornou lista de propriedades: %s", candidate_props)
            return

        normalized: List[AbstractProperty] = []
        for item in candidate_props:
            if not isinstance(item, dict):
                continue
            function_field = item.get("function") or item.get("target")
            function_ref = ""
            if isinstance(function_field, str):
                function_ref = function_field.strip()
            elif isinstance(function_field, (list, tuple)):
                for element in function_field:
                    if isinstance(element, str) and element.strip():
                        function_ref = element.strip()
                        break
            if not function_ref:
                LOGGER.debug("Ignorando propriedade sem função alvo: %s", item)
                continue
            if isinstance(function_field, (list, tuple)) and len(
                [el for el in function_field if isinstance(el, str) and el.strip()]
            ) > 1:
                LOGGER.debug("Discarding property with multiple target functions: %s", function_field)
                continue
            name = str(item.get("name") or f"property_{len(normalized) + 1}").strip()
            description = str(item.get("description") or "").strip()
            inputs_value = item.get("inputs") or []
            if isinstance(inputs_value, (list, tuple)):
                inputs = [str(element) for element in inputs_value]
            else:
                inputs = []
            prop: AbstractProperty = {
                "name": name,
                "description": description,
                "function": function_ref,
                "inputs": inputs,
            }
            context_text = item.get("context") or item.get("notes")
            if context_text:
                prop["context"] = str(context_text)
            normalized.append(prop)

        if not normalized:
            LOGGER.warning("LLM não forneceu propriedades utilizáveis para %s", target_file or "(arquivo)")
            return

        LOGGER.info("Geradas %d propriedade(s) PBT via LLM para %s", len(normalized), target_file or "(arquivo)")
        state["abstract_properties"] = normalized
        state.setdefault("public_examples", [])
        state.setdefault("known_failures", [])

    def _generate_property_check(
        self,
        prop: AbstractProperty,
        pos_examples: List[PropertyExample],
        neg_examples: List[PropertyExample],
        property_name: str,
    ) -> PropertyCheck:
        prompt = self._render_property_prompt(prop, pos_examples, neg_examples)
        raw_response = self.llm.invoke(PROPERTY_SYSTEM_PROMPT, prompt)
        payload = self._parse_json_response(raw_response, f"verificador da propriedade {prop.get('name') or prop.get('function')}")

        name = payload.get("name") or prop.get("name") or "property_check"
        code = payload.get("code")
        function_name = payload.get("function_name")
        description = payload.get("description") or prop.get("description") or ""
        if not code:
            raise ValueError("Resposta não contém o campo 'code'")
        if not function_name:
            raise ValueError("Resposta não contém o campo 'function_name'")
        check = PropertyCheck(
            name=name,
            description=description,
            function_name=function_name,
            code=code,
        )
        check["property_name"] = property_name
        target = (prop.get("function") or prop.get("target") or "").strip()
        if target:
            check["property_target"] = target  # type: ignore[assignment]
        return check

    def _compile_property_check(self, check: PropertyCheck) -> Callable[[Dict[str, Any], Any], bool]:
        namespace: Dict[str, Any] = {
            "Any": Any,
            "Dict": Dict,
            "List": List,
            "Optional": Optional,
            "Tuple": Tuple,
            "Union": Union,
        }
        try:
            exec(check["code"], namespace)  # noqa: S102 - executa código gerado
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Falha ao compilar verificador {check.get('name')}: {exc}") from exc
        func_name = check.get("function_name")
        candidate = None
        if func_name and func_name in namespace and callable(namespace[func_name]):
            candidate = namespace[func_name]
        else:
            for value in namespace.values():
                if callable(value):
                    candidate = value
                    break
        if not candidate:
            raise ValueError(f"Verificador {check.get('name')} não definiu função executável")
        return candidate  # type: ignore[return-value]

    def _validate_property_examples(
        self,
        property_name: str,
        check_func: Callable[[Dict[str, Any], Any], Any],
        pos_examples: List[PropertyExample],
        neg_examples: List[PropertyExample],
    ) -> List[str]:
        errors: List[str] = []
        for example in pos_examples:
            inputs = example.get("inputs", {})
            output = example.get("output")
            try:
                if not bool(check_func(inputs, output)):
                    errors.append(
                        f"[{property_name}] Falhou em exemplo conhecido: inputs={inputs}, output={output}"
                    )
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    f"[{property_name}] Exceção ao validar exemplo conhecido {inputs}: {exc}"
                )
        for example in neg_examples:
            inputs = example.get("inputs", {})
            output = example.get("output")
            try:
                if bool(check_func(inputs, output)):
                    errors.append(
                        f"[{property_name}] Não detectou saída incorreta: inputs={inputs}, output={output}"
                    )
            except Exception as exc:  # noqa: BLE001
                # Caso a função lance exceção, consideramos que detectou a falha.
                LOGGER.debug(
                    "Verificador %s lançou exceção esperada ao validar saída incorreta: %s",
                    property_name,
                    exc,
                )
        return errors

    def _generate_property_inputs(
        self,
        prop: AbstractProperty,
        check: PropertyCheck,
        property_name: str,
    ) -> Tuple[List[PropertyInput], Optional[str]]:
        prompt_lines = [
            f"Propriedade: {prop.get('name') or check.get('name')}",
            f"Descrição: {prop.get('description') or check.get('description')}",
            f"Função alvo: {prop.get('function') or prop.get('target')}",
            f"Parâmetros: {', '.join(prop.get('inputs') or []) or '(não informado)'}",
            "Código do verificador validado:",
            check.get("code", ""),
        ]
        prompt = "\n".join(prompt_lines)
        raw_response = self.llm.invoke(PBT_INPUT_PROMPT, prompt)
        data = self._parse_json_response(raw_response, f"entradas PBT da propriedade {prop.get('name')}")
        payload_inputs: List[Dict[str, Any]] = []
        generator_script: Optional[str] = None
        if isinstance(data, dict):
            payload_inputs = data.get("inputs") or []
            generator_script = data.get("generator")
        elif isinstance(data, list):
            payload_inputs = data
        else:
            LOGGER.warning(
                "Resposta inesperada ao gerar inputs para propriedade %s: %r",
                prop.get("name"),
                data,
            )
            payload_inputs = []
        collected: List[PropertyInput] = []
        for item in payload_inputs:
            if not isinstance(item, dict):
                continue
            collected.append(
                PropertyInput(
                    property_name=property_name,
                    inputs=item,
                )
            )
        return collected, generator_script

    def _resolve_target_callable(self, prop: AbstractProperty, property_name: str) -> Callable[..., Any]:
        target_ref = (prop.get("function") or prop.get("target") or "").strip()
        if not target_ref:
            raise ValueError(f"[{property_name}] Propriedade não informa caminho da função alvo")
        if "::" in target_ref:
            candidate = self._load_callable_from_file(target_ref, property_name)
        else:
            module_path, _, attr = target_ref.rpartition(".")
            if not module_path or not attr:
                raise ValueError(f"[{property_name}] Caminho inválido para função alvo: {target_ref}")
            with self._syspath_context():
                module = importlib.import_module(module_path)
            candidate = getattr(module, attr, None)
        if not callable(candidate):
            raise ValueError(f"[{property_name}] Função alvo {target_ref} não é chamável")
        return candidate  # type: ignore[return-value]

    def _load_callable_from_file(self, reference: str, property_name: str) -> Callable[..., Any]:
        file_part, _, attr = reference.partition("::")
        if not attr:
            raise ValueError(f"[{property_name}] Formato inválido para referência de arquivo: {reference}")
        path = Path(file_part.strip())
        if not path.is_absolute():
            path = (self.repo_root / path).resolve()
        if not path.exists():
            raise ValueError(f"[{property_name}] Arquivo alvo não encontrado: {path}")
        module_name = f"_tester_module_{abs(hash((str(path), attr)))}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if not spec or not spec.loader:
            raise ValueError(f"[{property_name}] Não foi possível carregar módulo para {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"[{property_name}] Falha ao carregar {path}: {exc}") from exc
        candidate = getattr(module, attr, None)
        if not callable(candidate):
            raise ValueError(f"[{property_name}] Função {attr} não encontrada em {path}")
        return candidate  # type: ignore[return-value]

    def _execute_property_runtime(
        self,
        prop: AbstractProperty,
        property_name: str,
        check: PropertyCheck,
        check_func: Callable[[Dict[str, Any], Any], Any],
        generated_inputs: List[PropertyInput],
        pos_examples: List[PropertyExample],
    ) -> Tuple[bool, List[str], Optional[Dict[str, Any]]]:
        logs: List[str] = []
        runtime_inputs: List[Dict[str, Any]] = []
        for example in pos_examples:
            runtime_inputs.append(example.get("inputs", {}))
        for item in generated_inputs:
            if item.get("property_name") and item["property_name"] != property_name:
                continue
            runtime_inputs.append(item.get("inputs", {}))
        if not runtime_inputs:
            logs.append(f"[{property_name}] Nenhum input adicional para execução do property-based test.")
            return True, logs, None
        try:
            target_callable = self._resolve_target_callable(prop, property_name)
        except Exception as exc:  # noqa: BLE001
            message = f"[{property_name}] Não foi possível resolver função alvo: {exc}"
            logs.append(message)
            failure_detail = {
                "type": "property",
                "property": property_name,
                "property_description": prop.get("description"),
                "check_name": check.get("name"),
                "stage": "resolve_target",
                "inputs": {},
                "output": None,
                "message": message,
            }
            return False, logs, failure_detail

        for index, inputs in enumerate(runtime_inputs, start=1):
            try:
                result = target_callable(**inputs)
            except Exception as exc:  # noqa: BLE001
                message = f"[{property_name}] Execução falhou para inputs {inputs}: {exc}"
                logs.append(message)
                failure_detail = {
                    "type": "property",
                    "property": property_name,
                    "property_description": prop.get("description"),
                    "check_name": check.get("name"),
                    "stage": "callable_execution",
                    "inputs": inputs,
                    "output": None,
                    "message": message,
                }
                return False, logs, failure_detail
            try:
                satisfied = bool(check_func(inputs, result))
            except Exception as exc:  # noqa: BLE001
                message = f"[{property_name}] Verificador lançou exceção para inputs {inputs}: {exc}"
                logs.append(message)
                failure_detail = {
                    "type": "property",
                    "property": property_name,
                    "property_description": prop.get("description"),
                    "check_name": check.get("name"),
                    "stage": "checker_execution",
                    "inputs": inputs,
                    "output": result,
                    "message": message,
                }
                return False, logs, failure_detail
            if not satisfied:
                message = f"[{property_name}] Propriedade violada para inputs {inputs} (iteração {index})"
                logs.append(message)
                failure_detail = {
                    "type": "property",
                    "property": property_name,
                    "property_description": prop.get("description"),
                    "check_name": check.get("name"),
                    "stage": "property_violation",
                    "inputs": inputs,
                    "output": result,
                    "message": message,
                }
                return False, logs, failure_detail
        logs.append(
            f"[{property_name}] Propriedade satisfeita para {len(runtime_inputs)} entradas avaliadas."
        )
        return True, logs, None

    def _slugify(self, text: str) -> str:
        text = (text or "property").lower()
        text = re.sub(r"[^a-z0-9]+", "_", text)
        text = text.strip("_") or "property"
        return text[:80]

    def _render_target_loader(
        self,
        prop: AbstractProperty,
        property_name: str,
        slug: str,
    ) -> Tuple[str, str]:
        target_ref = (prop.get("function") or prop.get("target") or "").strip()
        if not target_ref:
            raise ValueError(f"[{property_name}] Propriedade não informa função alvo.")
        if "::" in target_ref:
            file_part, _, attr = target_ref.partition("::")
            if not attr:
                raise ValueError(f"[{property_name}] Formato inválido para referência: {target_ref}")
            path = Path(file_part.strip())
            if not path.is_absolute():
                path = (self.repo_root / path).resolve()
            loader = textwrap.dedent(
                f"""
                import importlib.util
                from pathlib import Path

                _spec = importlib.util.spec_from_file_location("_a2a_target_{slug}", Path({str(path)!r}))
                _module = importlib.util.module_from_spec(_spec)
                _spec.loader.exec_module(_module)
                _target_callable = getattr(_module, {attr!r})
                """
            ).strip()
            return loader, "_target_callable"
        module_path, _, attr = target_ref.rpartition(".")
        if not module_path or not attr:
            raise ValueError(f"[{property_name}] Caminho inválido para função alvo: {target_ref}")
        loader = f"from {module_path} import {attr} as _target_callable"
        return loader, "_target_callable"

    def _write_property_test_module(
        self,
        prop: AbstractProperty,
        property_name: str,
        check: PropertyCheck,
        inputs: List[Dict[str, Any]],
    ) -> Optional[Path]:
        if not inputs:
            return None
        slug = self._slugify(property_name)
        self.generated_test_root.mkdir(parents=True, exist_ok=True)
        test_path = self.generated_test_root / f"test_property_{slug}.py"
        loader_code, target_expr = self._render_target_loader(prop, property_name, slug)
        check_code = textwrap.dedent(check.get("code") or "").strip()

        lines: List[str] = ["# Auto-generated by TesterAgent. Do not edit.", "import pytest"]
        if loader_code:
            lines.extend(loader_code.splitlines())
        if check_code:
            if lines and lines[-1]:
                lines.append("")
            lines.extend(check_code.splitlines())
        if lines and lines[-1]:
            lines.append("")
        lines.append("_TEST_INPUTS = [")
        for item in inputs:
            lines.append(f"    {repr(item)},")
        lines.append("]")
        lines.append("")
        description = check.get("description") or prop.get("description")
        if description:
            lines.append(f"# {description}")
        lines.append(f"@pytest.mark.parametrize('inputs', _TEST_INPUTS)")
        lines.append(f"def test_property_{slug}(inputs):")
        lines.append(f"    result = {target_expr}(**inputs)")
        lines.append(f"    assert bool({check.get('function_name')}(inputs, result))")

        test_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        if test_path not in self._active_test_files:
            self._active_test_files.append(test_path)
        return test_path

    def _cleanup_generated_tests(self, success: bool) -> None:
        if not success:
            return
        for path in list(self._active_test_files):
            with contextlib.suppress(FileNotFoundError):
                path.unlink()
        self._active_test_files.clear()
        with contextlib.suppress(FileNotFoundError, OSError):
            if self.generated_test_root.exists() and not any(self.generated_test_root.iterdir()):
                self.generated_test_root.rmdir()

    def _execute_property_checks(
        self,
        state: State,
    ) -> Tuple[
        bool,
        str,
        List[PropertyCheck],
        List[PropertyInput],
        List[str],
        List[str],
        List[Dict[str, Any]],
        List[Path],
    ]:
        abstract_props = list(state.get("abstract_properties") or [])
        if not abstract_props:
            return (
                True,
                "Nenhuma propriedade fornecida; testes baseados em propriedades não executados.",
                [],
                [],
                [],
                [],
                [],
                [],
            )

        public_examples = list(state.get("public_examples") or [])
        known_failures = list(state.get("known_failures") or [])
        existing_checks_map: Dict[str, PropertyCheck] = {}
        for existing_check in state.get("property_checks") or []:
            prop_name = (existing_check.get("property_name") or existing_check.get("name") or "").strip()
            if not prop_name:
                continue
            existing_checks_map[self._property_cache_key(prop_name)] = existing_check
        existing_inputs_map: Dict[str, List[PropertyInput]] = {}
        for existing_input in state.get("property_inputs") or []:
            prop_name = (existing_input.get("property_name") or "").strip()
            if not prop_name:
                continue
            existing_inputs_map.setdefault(self._property_cache_key(prop_name), []).append(existing_input)
        generator_scripts: List[str] = list(state.get("property_generators") or [])

        valid_checks_order: List[str] = []
        valid_checks_map: Dict[str, PropertyCheck] = {}
        generated_inputs: List[PropertyInput] = []
        errors: List[str] = []
        log_lines: List[str] = []
        overall_ok = True
        failure_details: List[Dict[str, Any]] = []
        generated_test_files: List[Path] = []

        for prop in abstract_props:
            property_name = self._property_identifier(prop)
            cache_key = self._property_cache_key(property_name)

            pos_examples = self._filter_examples(property_name, public_examples)
            neg_examples = self._filter_examples(property_name, known_failures)

            cached_check = existing_checks_map.get(cache_key)
            check: Optional[PropertyCheck] = None
            if cached_check:
                check = cached_check
                log_lines.append(f"[{property_name}] Verificador reaproveitado de execução anterior.")

            check_func: Optional[Callable[[Dict[str, Any], Any], Any]] = None
            if check:
                try:
                    check_func = self._compile_property_check(check)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning(
                        "[%s] Verificador reaproveitado inválido; regenerando. Motivo: %s",
                        property_name,
                        exc,
                    )
                    check = None

            if not check:
                try:
                    check = self._generate_property_check(prop, pos_examples, neg_examples, property_name)
                except Exception as exc:  # noqa: BLE001
                    message = f"[{property_name}] Falha ao gerar verificador: {exc}"
                    LOGGER.error(message)
                    log_lines.append(message)
                    errors.append(message)
                    overall_ok = False
                    continue
                try:
                    check_func = self._compile_property_check(check)
                except Exception as exc:  # noqa: BLE001
                    message = f"[{property_name}] Verificador inválido: {exc}"
                    LOGGER.error(message)
                    log_lines.append(message)
                    errors.append(message)
                    overall_ok = False
                    continue
            else:
                if not check.get("property_name"):
                    check["property_name"] = property_name  # type: ignore[index]

            if check_func is None:
                message = f"[{property_name}] Verificador não pôde ser compilado."
                LOGGER.error(message)
                log_lines.append(message)
                errors.append(message)
                overall_ok = False
                continue

            validation_errors = self._validate_property_examples(property_name, check_func, pos_examples, neg_examples)
            if validation_errors:
                overall_ok = False
                errors.extend(validation_errors)
                log_lines.extend(validation_errors)
                continue

            valid_checks_map[cache_key] = check
            if cache_key not in valid_checks_order:
                valid_checks_order.append(cache_key)
            log_lines.append(
                f"[{property_name}] Verificador validado com {len(pos_examples)} exemplo(s) correto(s) e "
                f"{len(neg_examples)} contraexemplo(s)."
            )

            cached_inputs = list(existing_inputs_map.get(cache_key, []))
            if cached_inputs:
                generated_inputs.extend(cached_inputs)
                log_lines.append(f"[{property_name}] Entradas PBT reaproveitadas ({len(cached_inputs)}).")
                new_inputs = cached_inputs
            else:
                new_inputs, generator_script = self._generate_property_inputs(prop, check, property_name)
                if new_inputs:
                    generated_inputs.extend(new_inputs)
                    log_lines.append(f"[{property_name}] {len(new_inputs)} entrada(s) sintetizada(s) para PBT.")
                else:
                    log_lines.append(f"[{property_name}] Nenhuma entrada adicional sintetizada.")
                if generator_script:
                    if generator_script not in generator_scripts:
                        generator_scripts.append(str(generator_script))

            candidate_inputs: List[Dict[str, Any]] = []
            for item in new_inputs:
                inputs_dict = item.get("inputs") if isinstance(item, dict) else None
                if isinstance(inputs_dict, dict) and inputs_dict not in candidate_inputs:
                    candidate_inputs.append(inputs_dict)
            if not candidate_inputs:
                for sample in pos_examples:
                    inputs_dict = sample.get("inputs", {})
                    if isinstance(inputs_dict, dict) and inputs_dict not in candidate_inputs:
                        candidate_inputs.append(inputs_dict)
            try:
                test_file = self._write_property_test_module(prop, property_name, check, candidate_inputs)
                if test_file:
                    generated_test_files.append(test_file)
                    try:
                        test_location = test_file.relative_to(self.repo_root)
                    except ValueError:
                        test_location = test_file
                    log_lines.append(f"[{property_name}] Testes gerados em {test_location}.")
            except Exception as exc:  # noqa: BLE001
                message = f"[{property_name}] Falha ao escrever arquivo de testes gerados: {exc}"
                LOGGER.error(message)
                log_lines.append(message)

            runtime_ok, runtime_logs, violation_detail = self._execute_property_runtime(
                prop,
                property_name,
                check,
                check_func,
                new_inputs,
                pos_examples,
            )
            log_lines.extend(runtime_logs)
            if violation_detail:
                failure_details.append(violation_detail)
            if not runtime_ok:
                overall_ok = False

        summary = "\n".join(log_lines).strip()
        if not summary:
            summary = "Nenhum log gerado durante a execução das propriedades."
        ordered_checks = [valid_checks_map[key] for key in valid_checks_order]
        dedup_inputs = self._deduplicate_property_inputs(generated_inputs)
        generator_scripts = list(dict.fromkeys(generator_scripts))
        errors = list(dict.fromkeys(errors))
        return (
            overall_ok,
            summary,
            ordered_checks,
            dedup_inputs,
            generator_scripts,
            errors,
            failure_details,
            generated_test_files,
        )

    def _load_tested_file_contents(self, state: State) -> Optional[str]:
        file_ref = state.get("file_path")
        if not file_ref or not isinstance(file_ref, str):
            return None
        candidate = Path(file_ref)
        if not candidate.is_absolute():
            candidate = (self.repo_root / candidate).resolve()
        try:
            if candidate.exists():
                return candidate.read_text()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Tester não conseguiu ler arquivo %s: %s", candidate, exc)
        return None

    def _summarize_issues(self, state: State) -> str:
        issues = list(state.get("issues_for_file") or state.get("issues") or [])
        if not issues:
            return "Nenhuma issue disponível."
        lines = []
        for item in issues:
            message = getattr(item, "message", "")
            rule = getattr(item, "rule", "")
            line = getattr(item, "line", None)
            if line is not None:
                lines.append(f"Linha {line}: {message} ({rule})")
            else:
                lines.append(f"{message} ({rule})")
        return "\n".join(lines)

    def _format_inputs(self, inputs: Dict[str, Any]) -> str:
        if not inputs:
            return "{}"
        parts = []
        for key, value in inputs.items():
            parts.append(f"{key}={repr(value)}")
        return ", ".join(parts)

    def _failure_score(self, case: Dict[str, Any]) -> int:
        inputs = case.get("inputs") or {}
        if inputs:
            try:
                return len(json.dumps(inputs, ensure_ascii=False, sort_keys=True))
            except TypeError:
                return len(repr(inputs))
        message = case.get("message")
        if message:
            return len(str(message))
        return 10

    def _extract_pytest_failure(self, logs: str) -> Optional[Dict[str, Any]]:
        if not logs.strip():
            return None
        lines = logs.splitlines()
        snippet_lines: List[str] = []
        capture = False
        for line in lines:
            if line.startswith("E   ") or line.startswith("F   "):
                capture = True
            if capture:
                snippet_lines.append(line)
                if not line.strip():
                    break
        snippet = "\n".join(snippet_lines).strip()
        if not snippet:
            marker = "short test summary info"
            if marker in logs:
                snippet = logs.split(marker, 1)[-1].strip()
        if not snippet:
            return None
        return {
            "type": "pytest",
            "stage": "unit_test",
            "inputs": {},
            "output": None,
            "message": snippet,
        }

    def _format_failure_summary(self, case: Dict[str, Any]) -> str:
        if not case:
            return ""
        case_type = case.get("type")
        if case_type == "property":
            property_name = case.get("property") or "(propriedade)"
            stage = case.get("stage") or "property_violation"
            inputs_str = self._format_inputs(case.get("inputs") or {})
            output = case.get("output")
            message = case.get("message") or ""
            if output is not None:
                return f"[{property_name}] {stage} com {inputs_str} -> output={repr(output)}. {message}".strip()
            return f"[{property_name}] {stage} com {inputs_str}. {message}".strip()
        if case_type == "pytest":
            return f"Falha do pytest: {case.get('message', '').strip()}"
        return case.get("message") or "Falha não especificada."

    def _prepare_feedback_cases(
        self,
        property_failures: List[Dict[str, Any]],
        pytest_ok: bool,
        pytest_logs: str,
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], str]:
        cases: List[Dict[str, Any]] = []
        for failure in property_failures:
            case = dict(failure)
            case["score"] = self._failure_score(case)
            cases.append(case)
        if not pytest_ok:
            pytest_case = self._extract_pytest_failure(pytest_logs)
            if pytest_case:
                pytest_case["score"] = self._failure_score(pytest_case)
                cases.append(pytest_case)
        cases.sort(key=lambda item: item.get("score", float("inf")))
        primary = cases[0] if cases else None
        summary = self._format_failure_summary(primary) if primary else ""
        return cases, primary, summary

    def run_suite(self, extra_sections: Optional[List[Tuple[str, str]]] = None) -> Tuple[bool, bool, str]:
        pytest_ok, pytest_logs, pytest_missing = self._run_command(["pytest", "-q", "--disable-warnings"])

        lint_passed = True
        lint_sections: List[str] = []
        if self.run_linters:
            ruff_ok, ruff_logs, ruff_missing = self._run_command(["ruff", "check", "."])
            black_ok, black_logs, black_missing = self._run_command(["black", "--check", "."])

            if ruff_missing:
                ruff_logs = (ruff_logs.strip() + "\nComando ausente; lint pulado.").strip()
            if black_missing:
                black_logs = (black_logs.strip() + "\nComando ausente; lint pulado.").strip()

            lint_passed = (ruff_ok or ruff_missing) and (black_ok or black_missing)
            lint_sections.extend(
                [
                    "ruff check .:\n" + ruff_logs.strip(),
                    "black --check .:\n" + black_logs.strip(),
                ]
            )
        else:
            lint_sections.append("linters:\nLinters desativados (TESTER_RUN_LINTERS=0).")

        sections: List[str] = []
        if extra_sections:
            for title, content in extra_sections:
                sections.append(f"{title}:\n{content.strip()}")
        sections.append("pytest -q:\n" + pytest_logs.strip())
        sections.extend(lint_sections)

        aggregate_logs = "\n\n".join(section.strip() for section in sections if section).strip()

        if pytest_missing:
            LOGGER.error("Pytest não está disponível; marcando suíte como falha.")
            pytest_ok = False
        return pytest_ok, lint_passed, aggregate_logs

    def invoke(self, state: State) -> State:
        tested_code = self._load_tested_file_contents(state) or ""
        issues_summary = self._summarize_issues(state)
        self._maybe_generate_properties(state, tested_code, issues_summary)

        (
            property_ok,
            property_logs,
            property_checks,
            property_inputs,
            generator_scripts,
            property_errors,
            property_failures,
            generated_test_files,
        ) = self._execute_property_checks(state)

        extra_sections: List[Tuple[str, str]] = [("issues", issues_summary)]
        if tested_code:
            extra_sections.append(("arquivo-corrigido", tested_code))
        extra_sections.append(("property-tests", property_logs))

        pytest_ok, lint_ok, logs = self.run_suite(extra_sections=extra_sections)
        overall_tests_ok = pytest_ok and property_ok

        feedback_cases, primary_case, feedback_summary = self._prepare_feedback_cases(
            property_failures,
            pytest_ok,
            logs,
        )
        generated_files_rel: List[str] = []
        for path in generated_test_files:
            try:
                generated_files_rel.append(str(path.relative_to(self.repo_root)))
            except ValueError:
                generated_files_rel.append(str(path))
        self._cleanup_generated_tests(overall_tests_ok)
        feedback_cases_clean = [dict(item) for item in feedback_cases]
        for item in feedback_cases_clean:
            item.pop("score", None)
        primary_case_clean = dict(primary_case) if primary_case else None
        if primary_case_clean:
            primary_case_clean.pop("score", None)

        prompt = "Logs de teste e lint:\n" + logs
        summary = self.llm.invoke(SYSTEM_TESTER, prompt)
        if feedback_summary:
            summary = summary.strip()
            if summary:
                summary += "\n\n"
            summary += f"Falha principal identificada: {feedback_summary}"
        if not overall_tests_ok and generated_files_rel:
            summary = summary.strip()
            if summary:
                summary += "\n\n"
            summary += "Testes gerados mantidos em: " + ", ".join(generated_files_rel)
        elif overall_tests_ok and generated_files_rel:
            summary = summary.strip()
            if summary:
                summary += "\n\n"
            summary += "Testes gerados foram limpos após execução bem-sucedida."

        state.update({
            "test_passed": overall_tests_ok,
            "lint_passed": lint_ok,
            "test_output": logs,
            "test_logs": logs,
            "tester_summary": summary,
            "property_summary": property_logs,
            "property_checks": property_checks,
            "property_inputs": property_inputs,
            "property_generators": generator_scripts,
            "property_check_errors": property_errors,
            "property_failures": property_failures,
            "tester_file_contents": tested_code,
            "tester_issue_summary": issues_summary,
            "tester_feedback_cases": feedback_cases_clean,
            "tester_primary_failure": primary_case_clean or {},
            "tester_feedback_summary": feedback_summary,
            "tester_generated_test_files": generated_files_rel,
        })
        return state
