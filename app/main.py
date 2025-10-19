"""LangGraph orchestrator for the AutoFix pipeline following the TCC plan."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.a2a.deployment_agent import DeploymentAgent
from app.a2a.executor_agent import ExecutorAgent
from app.a2a.fixer_agent import FixerAgent
from app.a2a.planner_agent import invoke as planner_invoke
from app.a2a.protocol import State
from app.a2a.requester_agent import RequesterAgent
from app.a2a.sonar_agent import invoke as sonar_invoke
from app.a2a.tester_agent import TesterAgent

_ROOT_ENV = Path(__file__).resolve().parents[2] / ".env"
_LOCAL_ENV = Path(__file__).resolve().parents[1] / ".env"

# Load root env first (project-level overrides) then pipeline-specific env.
if _ROOT_ENV.exists():
    load_dotenv(dotenv_path=_ROOT_ENV, override=False)
load_dotenv(dotenv_path=_LOCAL_ENV, override=True)

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

MAX_FEEDBACK_LOOPS = 2


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


def build_graph(repo_root: Path | None = None) -> StateGraph:
    repo_root = Path(repo_root).resolve() if repo_root else _resolve_repo_root()
    os.environ["A2A_REPO_ROOT"] = str(repo_root)

    requester = RequesterAgent(repo_root=repo_root)
    fixer = FixerAgent(repo_root=repo_root)
    executor = ExecutorAgent(repo_root=repo_root)
    tester = TesterAgent(repo_root=repo_root)
    deployment = DeploymentAgent(repo_root=repo_root)

    def planner_node(state: State) -> State:
        state.setdefault("repo_root", repo_root.as_posix())
        return planner_invoke(state)

    def requester_node(state: State) -> State:
        state.setdefault("repo_root", repo_root.as_posix())
        return requester.invoke(state)

    def fixer_node(state: State) -> State:
        state.setdefault("repo_root", repo_root.as_posix())
        return fixer.invoke(state)

    def executor_node(state: State) -> State:
        state.setdefault("repo_root", repo_root.as_posix())
        return executor.invoke(state)

    def feedback_node(state: State) -> State:
        state.setdefault("repo_root", repo_root.as_posix())
        execution_failed = state.get("execution_failed", False)
        tests_failed = state.get("test_passed") is False
        loops = state.get("feedback_loops", 0)
        should_retry = False

        failure_source: str | None = None
        if execution_failed:
            failure_source = "executor"
        elif tests_failed:
            failure_source = "tester"

        if failure_source:
            if loops < MAX_FEEDBACK_LOOPS:
                loops += 1
                state["feedback_loops"] = loops
                test_output = (
                    state.get("test_output")
                    or state.get("tester_summary")
                    or ""
                ).strip()
                if not test_output:
                    test_output = "Testes falharam, mas nenhuma saída foi capturada."
                entry_header = f"Tentativa {loops} falhou ({failure_source})"
                new_entry = f"{entry_header}:\n{test_output}"
                feedback_log = state.get("feedback_log") or ""
                state["feedback_log"] = (
                    f"{feedback_log}\n\n{new_entry}" if feedback_log else new_entry
                )
                LOGGER.info(
                    "Feedback loop #%d acionado após falha (%s)",
                    loops,
                    failure_source,
                )
                state["tester_summary"] = state.get("tester_summary") or test_output
                processed = set(state.get("processed_components") or [])
                current_issue = state.get("issue")
                if current_issue and getattr(current_issue, "component", None) in processed:
                    processed.discard(current_issue.component)
                    LOGGER.debug(
                        "Feedback loop removendo componente %s de processed_components",
                        current_issue.component,
                    )
                state["processed_components"] = list(processed)
                base_context = (state.get("context") or "").rstrip()
                if base_context:
                    base_context += "\n\n"
                state["context"] = f"{base_context}[Feedback da tentativa {loops}]\n{test_output}"
                state["patch"] = ""
                should_retry = True
            else:
                limit_msg = "Limite de tentativas do feedback loop alcançado."
                feedback_log = state.get("feedback_log") or ""
                state["feedback_log"] = (
                    f"{feedback_log}\n\n{limit_msg}" if feedback_log else limit_msg
                )
                LOGGER.info("Feedback loop atingiu o limite de %d tentativas", MAX_FEEDBACK_LOOPS)
        else:
            if loops:
                LOGGER.debug("Feedback loop resetado após sucesso dos testes (antes=%d)", loops)
            state["feedback_loops"] = 0
            state["execution_failed"] = False
            state["retry_requested"] = False
            return state

        state["retry_requested"] = should_retry
        return state

    def feedback_router(state: State) -> Literal["requester", "tester"]:
        return "requester" if state.get("retry_requested") else "tester"

    def tester_node(state: State) -> State:
        state.setdefault("repo_root", repo_root.as_posix())
        if state.get("execution_failed"):
            LOGGER.info("Tester pulado: executor falhou")
            state.setdefault("tester_summary", "Executor falhou; testes não executados")
            state.setdefault("test_passed", False)
            state.setdefault("lint_passed", False)
            return state
        return tester.invoke(state)

    def sonar_node(state: State) -> State:
        state.setdefault("repo_root", repo_root.as_posix())
        if not state.get("test_passed"):
            LOGGER.info("Sonar pulado: testes não passaram")
            state.setdefault("sonar_passed", False)
            state.setdefault("sonar_summary", "Sonar não executado devido a falha nos testes")
            return state
        return sonar_invoke(state)

    def deployment_node(state: State) -> State:
        state.setdefault("repo_root", repo_root.as_posix())
        return deployment.invoke(state)

    builder = StateGraph(State)
    builder.add_node("planner", planner_node)
    builder.add_node("requester", requester_node)
    builder.add_node("fixer", fixer_node)
    builder.add_node("executor", executor_node)
    builder.add_node("feedback", feedback_node)
    builder.add_node("tester", tester_node)
    builder.add_node("sonar", sonar_node)
    builder.add_node("deployment", deployment_node)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "requester")
    builder.add_edge("requester", "fixer")
    builder.add_edge("fixer", "executor")
    builder.add_edge("executor", "feedback")
    builder.add_conditional_edges("feedback", feedback_router)

    def tester_router(state: State) -> Literal["feedback", "sonar"]:
        if state.get("test_passed"):
            return "sonar"
        loops = state.get("feedback_loops", 0)
        if loops >= MAX_FEEDBACK_LOOPS:
            LOGGER.info(
                "Tester falhou após atingir limite de feedback loops; seguindo para sonar."
            )
            state["retry_requested"] = False
            return "sonar"
        return "feedback"

    builder.add_conditional_edges("tester", tester_router)
    builder.add_edge("sonar", "deployment")
    builder.add_edge("deployment", END)

    return builder


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
    
    graph_builder = build_graph(repo_root=repo_root)
    graph = graph_builder.compile(checkpointer=MemorySaver())
    LOGGER.info("Iniciando pipeline AutoFix")
    final_state: State = graph.invoke({}, config={"configurable": {"thread_id": "autofix"}})
    LOGGER.info("Pipeline concluída")

    serializable = {k: _serialize(v) for k, v in final_state.items()}
    LOGGER.debug("Estado final:\n%s", json.dumps(serializable, ensure_ascii=False, indent=2))
    return final_state


if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Falha durante pipeline: %s", exc)
        raise
