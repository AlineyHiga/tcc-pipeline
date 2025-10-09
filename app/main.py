"""LangGraph orchestrator for the AutoFix pipeline following the TCC plan."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

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

load_dotenv()

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


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


def build_graph() -> StateGraph:
    repo_root = _resolve_repo_root()
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
    builder.add_node("tester", tester_node)
    builder.add_node("sonar", sonar_node)
    builder.add_node("deployment", deployment_node)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "requester")
    builder.add_edge("requester", "fixer")
    builder.add_edge("fixer", "executor")
    builder.add_edge("executor", "tester")
    builder.add_edge("tester", "sonar")
    builder.add_edge("sonar", "deployment")
    builder.add_edge("deployment", END)

    return builder


def run_pipeline() -> State:
    graph_builder = build_graph()
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
