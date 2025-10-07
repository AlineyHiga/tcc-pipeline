"""LangGraph orchestrator for the AutoFix pipeline following the TCC plan."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, is_dataclass
from typing import Any

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.a2a.executor_agent import ExecutorAgent
from app.a2a.fixer_agent import FixerAgent
from app.a2a.planner_agent import invoke as planner_invoke
from app.a2a.protocol import State
from app.a2a.requester_agent import RequesterAgent
from app.a2a.sonar_agent import invoke as sonar_invoke
from app.a2a.tester_agent import TesterAgent
from app.a2a.deployment_agent import deployment_node

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


def build_graph() -> StateGraph:
    requester = RequesterAgent()
    fixer = FixerAgent()
    executor = ExecutorAgent()
    tester = TesterAgent()

    def requester_node(state: State) -> State:
        return requester.invoke(state)

    def fixer_node(state: State) -> State:
        return fixer.invoke(state)

    def executor_node(state: State) -> State:
        return executor.invoke(state)

    def tester_node(state: State) -> State:
        if state.get("execution_failed"):
            LOGGER.info("Tester pulado: executor falhou")
            state.setdefault("tester_summary", "Executor falhou; testes não executados")
            state.setdefault("test_passed", False)
            state.setdefault("lint_passed", False)
            return state
        return tester.invoke(state)

    def sonar_node(state: State) -> State:
        if not state.get("test_passed"):
            LOGGER.info("Sonar pulado: testes não passaram")
            state.setdefault("sonar_passed", False)
            state.setdefault("sonar_summary", "Sonar não executado devido a falha nos testes")
            return state
        return sonar_invoke(state)

    builder = StateGraph(State)
    builder.add_node("planner", planner_invoke)
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
