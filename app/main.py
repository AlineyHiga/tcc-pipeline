"""LangGraph orchestrator for the AutoFix pipeline following the TCC plan."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Literal

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.a2a.deployment_agent import DeploymentAgent
from app.a2a.fixer_agent_targeted import TargetedFixerAgent as FixerAgent
from app.a2a.patcher_tool import apply_patch_node
from app.a2a.planner_agent import invoke as planner_invoke
from app.a2a.property_agent import PropertyAgent
from app.a2a.protocol import State
from app.a2a.requester_agent_optimized import OptimizedRequesterAgent as RequesterAgent
from app.a2a.sonar_agent import invoke as sonar_invoke
from app.a2a.tester_agent_simple import SimpleTesterAgent as TesterAgent

_ROOT_ENV = Path(__file__).resolve().parents[2] / ".env"
_LOCAL_ENV = Path(__file__).resolve().parents[1] / ".env"

# Load root env first (project-level overrides) then pipeline-specific env.
if _ROOT_ENV.exists():
    load_dotenv(dotenv_path=_ROOT_ENV, override=False)
load_dotenv(dotenv_path=_LOCAL_ENV, override=True)

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

MAX_FEEDBACK_LOOPS = 2
MAX_PROPERTY_ATTEMPTS = int(os.getenv("MAX_PROPERTY_ATTEMPTS", "3"))
GRAPH_RECURSION_LIMIT = int(os.getenv("LANGGRAPH_RECURSION_LIMIT", "3"))


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    return value


def _resolve_repo_root() -> Path:
    candidates = [
        ("AUTOFIX_TARGET_ROOT", os.getenv("AUTOFIX_TARGET_ROOT")),
        ("A2A_TARGET_ROOT", os.getenv("A2A_TARGET_ROOT")),
        ("A2A_REPO_ROOT", os.getenv("A2A_REPO_ROOT")),
    ]
    for name, value in candidates:
        if not value:
            continue
        path = Path(value).expanduser()
        if path.exists():
            resolved = path.resolve()
            LOGGER.info("AutoFix utilizando diretório alvo (%s): %s", name, resolved)
            return resolved
        LOGGER.warning("Diretório alvo configurado em %s não existe: %s", name, value)
    default_root = Path.cwd().resolve()
    LOGGER.info("AutoFix utilizando diretório atual como alvo: %s", default_root)
    return default_root


def _ensure_metrics_container(state: State) -> dict[str, Any]:
    metrics = state.get("metrics")
    if not metrics:
        metrics = {"timings": {}, "attempts": {}, "counters": {}}
        state["metrics"] = metrics
    else:
        metrics.setdefault("timings", {})
        metrics.setdefault("attempts", {})
        metrics.setdefault("counters", {})
    return metrics


def _metrics_increment(state: State, counter: str, amount: int = 1) -> None:
    metrics = _ensure_metrics_container(state)
    counters = metrics["counters"]
    counters[counter] = counters.get(counter, 0) + amount


def _invoke_with_metrics(state: State, label: str, func: Callable[[State], State]) -> State:
    metrics = _ensure_metrics_container(state)
    attempts = metrics["attempts"]
    attempts[label] = attempts.get(label, 0) + 1
    start = perf_counter()
    try:
        return func(state)
    finally:
        elapsed = perf_counter() - start
        timings = metrics["timings"]
        timings[label] = timings.get(label, 0.0) + elapsed


def build_graph(repo_root: Path | None = None) -> StateGraph:
    repo_root = Path(repo_root).resolve() if repo_root else _resolve_repo_root()
    os.environ["A2A_REPO_ROOT"] = str(repo_root)

    requester = RequesterAgent(repo_root=repo_root)
    fixer = FixerAgent(repo_root=repo_root)
    tester = TesterAgent(repo_root=repo_root)
    deployment = DeploymentAgent(repo_root=repo_root)
    property_agent = PropertyAgent(repo_root=repo_root)

    def planner_node(state: State) -> State:
        state.setdefault("repo_root", repo_root.as_posix())
        return _invoke_with_metrics(state, "planner", planner_invoke)

    def property_node(state: State) -> State:
        state.setdefault("repo_root", repo_root.as_posix())
        return _invoke_with_metrics(state, "property", property_agent.invoke)

    def property_tester_node(state: State) -> State:
        state.setdefault("repo_root", repo_root.as_posix())
        state["tester_mode"] = "properties"
        result_state = _invoke_with_metrics(state, "tester_properties", tester.invoke)
        LOGGER.info(
            "property_tester_node: property_tests_passed=%s, keys=%s",
            result_state.get("property_tests_passed"),
            list(result_state.keys()),
        )
        return result_state

    def property_abort_node(state: State) -> State:
        LOGGER.info("Pipeline interrompida antes do requester devido a falha nas propriedades")
        state.setdefault(
            "property_summary",
            state.get("property_summary") or "Testes de propriedades falharam.",
        )
        state.setdefault("deployment_summary", "Pipeline interrompida nas propriedades.")
        state.setdefault("deployment_failed", True)
        return state

    def property_router(state: State) -> Literal["property", "property_abort", "requester"]:
        property_passed = state.get("property_tests_passed")
        LOGGER.info(
            "Router de propriedades avaliando resultado: property_tests_passed=%s, attempts=%s",
            property_passed,
            state.get("property_attempts"),
        )
        if property_passed:
            state["property_attempts"] = 0
            return "requester"
        attempts = int(state.get("property_attempts") or 0) + 1
        state["property_attempts"] = attempts
        component = state.get("property_component")
        processed = set(state.get("property_processed_components") or [])
        if component and component in processed:
            processed.discard(component)
            state["property_processed_components"] = list(processed)
        _metrics_increment(state, "property_failures")
        if attempts >= MAX_PROPERTY_ATTEMPTS:
            LOGGER.error(
                "Testes de propriedades falharam após %d tentativa(s); interrompendo pipeline.",
                attempts,
            )
            return "property_abort"
        summary = (state.get("property_summary") or "").strip()
        if summary:
            max_log_chars = 4000
            if len(summary) > max_log_chars:
                summary = summary[:max_log_chars] + "\n...[truncado]"
            LOGGER.warning(
                "Resumo da tentativa %d dos testes de propriedade:\n%s",
                attempts,
                summary,
            )
        else:
            LOGGER.warning(
                "Resumo da tentativa %d dos testes de propriedade está vazio.",
                attempts,
            )
        LOGGER.warning(
            "Testes de propriedades falharam (tentativa %d); regenerando propriedades.",
            attempts,
        )
        return "property"

    def requester_node(state: State) -> State:
        state.setdefault("repo_root", repo_root.as_posix())
        return _invoke_with_metrics(state, "requester", requester.invoke)

    def fixer_node(state: State) -> State:
        state.setdefault("repo_root", repo_root.as_posix())
        return _invoke_with_metrics(state, "fixer", fixer.invoke)

    def tester_node(state: State) -> State:
        state.setdefault("repo_root", repo_root.as_posix())
        if state.get("fix_failed"):
            LOGGER.info("Tester pulado: patch falhou ao ser aplicado")
            state.setdefault("tester_summary", "Patch não aplicado; tester aguardando nova tentativa")
            state.setdefault("test_passed", False)
            state.setdefault("lint_passed", False)
            _metrics_increment(state, "tester_skipped")
            return state
        return _invoke_with_metrics(state, "tester", tester.invoke)

    def feedback_node(state: State) -> State:
        state.setdefault("repo_root", repo_root.as_posix())
        def _logic(current: State) -> State:
            fix_failed = current.get("fix_failed", False)
            tests_failed = current.get("test_passed") is False
            loops = current.get("feedback_loops", 0)
            should_retry = False

            failure_source: str | None = None
            if fix_failed:
                failure_source = "fixer"
            elif tests_failed:
                failure_source = "tester"

            if failure_source:
                if loops < MAX_FEEDBACK_LOOPS:
                    loops += 1
                    current["feedback_loops"] = loops
                    if failure_source == "tester":
                        feedback_text = (
                            current.get("test_output")
                            or current.get("tester_summary")
                            or ""
                        ).strip()
                        if not feedback_text:
                            feedback_text = "Testes falharam, mas nenhuma saída foi capturada."
                        _metrics_increment(current, "tester_failures")
                    else:
                        feedback_text = (current.get("fixer_summary") or "").strip()
                        if not feedback_text:
                            feedback_text = "Fixer não conseguiu aplicar o patch."
                        _metrics_increment(current, "fixer_failures")
                    entry_header = f"Tentativa {loops} falhou ({failure_source})"
                    new_entry = f"{entry_header}:\n{feedback_text}"
                    feedback_log = current.get("feedback_log") or ""
                    current["feedback_log"] = (
                        f"{feedback_log}\n\n{new_entry}" if feedback_log else new_entry
                    )
                    LOGGER.info(
                        "Feedback loop #%d acionado após falha (%s)",
                        loops,
                        failure_source,
                    )
                    if failure_source == "tester":
                        current["tester_summary"] = current.get("tester_summary") or feedback_text
                    processed = set(current.get("processed_components") or [])
                    current_issue = current.get("issue")
                    if current_issue and getattr(current_issue, "component", None) in processed:
                        processed.discard(current_issue.component)
                        LOGGER.debug(
                            "Feedback loop removendo componente %s de processed_components",
                            current_issue.component,
                        )
                    current["processed_components"] = list(processed)
                    focused_issues = list(current.get("issues_for_file") or [])
                    if focused_issues:
                        component_name = getattr(current_issue, "component", None)
                        LOGGER.debug(
                            "Feedback loop restringindo issues à componente %s para nova tentativa",
                            component_name or "(desconhecido)",
                        )
                        current["issues_scoped"] = focused_issues
                    base_context = (current.get("context") or "").rstrip()
                    if base_context:
                        base_context += "\n\n"
                    current["context"] = f"{base_context}[Feedback da tentativa {loops}]\n{feedback_text}"
                    current["patch"] = ""
                    should_retry = True
                    current["next_after_requester"] = "fixer"
                    _metrics_increment(current, "feedback_loops")
                else:
                    limit_msg = "Limite de tentativas do feedback loop alcançado."
                    feedback_log = current.get("feedback_log") or ""
                    current["feedback_log"] = (
                        f"{feedback_log}\n\n{limit_msg}" if feedback_log else limit_msg
                    )
                    LOGGER.info("Feedback loop atingiu o limite de %d tentativas", MAX_FEEDBACK_LOOPS)
            else:
                if loops:
                    LOGGER.debug("Feedback loop resetado após sucesso dos testes (antes=%d)", loops)
                current["feedback_loops"] = 0
                current["retry_requested"] = False
                current.pop("next_after_requester", None)
                current.pop("issues_scoped", None)
                return current

            current["retry_requested"] = should_retry
            return current

        return _invoke_with_metrics(state, "feedback", _logic)

    def feedback_router(state: State) -> Literal["requester", "sonar"]:
        return "requester" if state.get("retry_requested") else "sonar"

    def requester_router(state: State) -> Literal["fixer", "tester"]:
        next_step = state.pop("next_after_requester", None)
        if next_step == "tester":
            return "tester"
        return "fixer"

    def tester_router(state: State) -> Literal["feedback", "sonar"]:
        return "sonar" if state.get("test_passed") else "feedback"

    def patcher_node(state: State) -> State:
        state.setdefault("repo_root", repo_root.as_posix())
        result = _invoke_with_metrics(state, "patcher", apply_patch_node)
        if state.get("fix_failed"):
            state.setdefault("test_passed", False)
        return result

    def patcher_router(state: State) -> Literal["tester", "feedback"]:
        return "feedback" if state.get("fix_failed") else "tester"

    def sonar_node(state: State) -> State:
        state.setdefault("repo_root", repo_root.as_posix())
        if not state.get("test_passed"):
            LOGGER.info("Sonar pulado: testes não passaram")
            state.setdefault("sonar_passed", False)
            state.setdefault("sonar_summary", "Sonar não executado devido a falha nos testes")
            _metrics_increment(state, "sonar_skipped")
            return state
        return _invoke_with_metrics(state, "sonar", sonar_invoke)

    def sonar_feedback_node(state: State) -> State:
        state.setdefault("repo_root", repo_root.as_posix())

        def _logic(current: State) -> State:
            attempts = int(current.get("sonar_feedback_attempts") or 0) + 1
            current["sonar_feedback_attempts"] = attempts

            summary = (current.get("sonar_summary") or "Sonar ainda reporta issues abertas.").strip()
            if summary:
                base_context = (current.get("context") or "").rstrip()
                if base_context:
                    base_context += "\n\n"
                marker = f"[Sonar tentativa {attempts}]"
                current["context"] = f"{base_context}{marker}\n{summary}".strip()

            remaining = list(current.get("sonar_remaining_issues") or [])
            if remaining:
                current["issues_scoped"] = remaining
                current["issues_for_file"] = remaining
                current["issue"] = remaining[0]

            processed = set(current.get("processed_components") or [])
            current_issue = current.get("issue")
            component_name = getattr(current_issue, "component", None) if current_issue else None
            if component_name and component_name in processed:
                processed.discard(component_name)
            current["processed_components"] = list(processed)
            current["patch"] = ""
            LOGGER.warning(
                "Sonar identificou issues após tentativa de correção (tentativa %d); retornando ao requester.",
                attempts,
            )
            _metrics_increment(current, "sonar_feedback")
            return current

        return _invoke_with_metrics(state, "sonar_feedback", _logic)

    def sonar_router(state: State) -> Literal["deployment", "sonar_feedback"]:
        if state.get("sonar_passed"):
            if state.get("sonar_feedback_attempts"):
                LOGGER.debug(
                    "Sonar recuperou sucesso após %d tentativa(s)",
                    state.get("sonar_feedback_attempts"),
                )
            state["sonar_feedback_attempts"] = 0
            return "deployment"
        attempts = int(state.get("sonar_feedback_attempts") or 0)
        if attempts >= MAX_FEEDBACK_LOOPS:
            LOGGER.error(
                "Sonar falhou após %d tentativa(s); prosseguindo para deployment mesmo assim.",
                attempts,
            )
            return "deployment"
        LOGGER.info("Sonar detectou issues remanescentes; iniciando feedback para nova tentativa.")
        _metrics_increment(state, "sonar_failures")
        return "sonar_feedback"

    def deployment_node(state: State) -> State:
        state.setdefault("repo_root", repo_root.as_posix())
        return _invoke_with_metrics(state, "deployment", deployment.invoke)

    builder = StateGraph(State)
    builder.add_node("planner", planner_node)
    builder.add_node("property", property_node)
    builder.add_node("property_tester", property_tester_node)
    builder.add_node("property_abort", property_abort_node)
    builder.add_node("requester", requester_node)
    builder.add_node("feedback", feedback_node)
    builder.add_node("tester", tester_node)
    builder.add_node("fixer", fixer_node)
    builder.add_node("patcher", patcher_node)
    builder.add_node("sonar", sonar_node)
    builder.add_node("sonar_feedback", sonar_feedback_node)
    builder.add_node("deployment", deployment_node)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "property")
    builder.add_edge("property", "property_tester")
    builder.add_conditional_edges("property_tester", property_router)
    builder.add_conditional_edges("requester", requester_router)
    builder.add_conditional_edges("tester", tester_router)
    builder.add_conditional_edges("feedback", feedback_router)
    builder.add_edge("fixer", "patcher")
    builder.add_conditional_edges("patcher", patcher_router)
    builder.add_conditional_edges("sonar", sonar_router)
    builder.add_edge("sonar_feedback", "requester")
    builder.add_edge("property_abort", END)
    builder.add_edge("deployment", END)

    return builder


def _persist_metrics(state: State, repo_root: Path) -> None:
    metrics = state.get("metrics")
    if not metrics:
        LOGGER.info("Nenhuma métrica registrada para persistir.")
        return

    output_hint = os.getenv("PIPELINE_METRICS_PATH")
    if output_hint:
        target = Path(output_hint)
        if target.is_dir():
            target = target / "metrics.jsonl"
    else:
        target = repo_root / "artifacts" / "metrics.jsonl"

    target.parent.mkdir(parents=True, exist_ok=True)

    issue = state.get("issue")
    record = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "repo_root": repo_root.as_posix(),
        "active_issue": getattr(issue, "key", None) if issue else None,
        "issues_processed": len(state.get("processed_components") or []),
        "feedback_loops": state.get("feedback_loops", 0),
        "tests_passed": state.get("test_passed"),
        "sonar_passed": state.get("sonar_passed"),
        "metrics": metrics,
    }

    with target.open("a", encoding="utf-8") as handler:
        handler.write(json.dumps(record, ensure_ascii=False) + "\n")

    LOGGER.info("Métricas registradas em %s", target)


def run_pipeline() -> State:
    from app.utils import run_sonar_scanner
    
    repo_root = _resolve_repo_root()
    os.environ["A2A_REPO_ROOT"] = str(repo_root)
    
    # Execute sonar-scanner first
    LOGGER.info("Executando sonar-scanner inicial em %s", repo_root)
    try:
        run_sonar_scanner(cwd=repo_root)
        LOGGER.info("Sonar-scanner executado com sucesso")
    except Exception as e:
        LOGGER.error("Falha no sonar-scanner: %s", e)
        return {"error": str(e)}
    
    pipeline_start = perf_counter()
    graph_builder = build_graph(repo_root=repo_root)
    graph = graph_builder.compile(checkpointer=MemorySaver())
    LOGGER.info("Iniciando pipeline AutoFix")
    final_state: State = graph.invoke(
        {},
        config={
            "recursion_limit": GRAPH_RECURSION_LIMIT,
            "configurable": {"thread_id": "autofix"},
        },
    )
    LOGGER.info("Pipeline concluída")
    pipeline_elapsed = perf_counter() - pipeline_start

    metrics = _ensure_metrics_container(final_state)
    timings = metrics["timings"]
    timings["pipeline_total_seconds"] = timings.get("pipeline_total_seconds", 0.0) + pipeline_elapsed
    counters = metrics["counters"]
    counters["pipeline_runs"] = counters.get("pipeline_runs", 0) + 1

    serializable = {k: _serialize(v) for k, v in final_state.items()}
    LOGGER.debug("Estado final:\n%s", json.dumps(serializable, ensure_ascii=False, indent=2))
    _persist_metrics(final_state, repo_root)
    return final_state


if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Falha durante pipeline: %s", exc)
        raise
