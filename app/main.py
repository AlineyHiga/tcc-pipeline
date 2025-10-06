"""Orchestrator for the AutoFix SonarQube + A2A pipeline."""
from __future__ import annotations

import json
import logging
import os
from typing import List

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.a2a import FixerAgent, Issue, RequesterAgent, State, TesterAgent
from app.a2a.deployment_agent import deployment_node
from app.a2a.patcher_tool import apply_patch_node
from app.sonarqube_client import SonarIssue, SonarQubeClient, format_issue
from app.utils import format_issues_for_prompt, run_sonar_scanner

load_dotenv()

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

MAX_ROUNDS = int(os.getenv("MAX_ROUNDS", "3"))
ISSUE_SEVERITIES = [s.strip() for s in os.getenv("ISSUE_SEVERITIES", "MAJOR,CRITICAL").split(",") if s.strip()]


def sonar_issue_to_issue(issue: SonarIssue) -> Issue:
    return Issue(
        key=issue.key,
        rule=issue.rule,
        severity=issue.severity,
        component=issue.component,
        message=issue.message,
        line=issue.line,
    )


def append_feedback(state: State, header: str, body: str) -> None:
    if not body:
        return
    entry = f"{header}:\n{body.strip()}"
    log = state.get("feedback_log", "")
    state["feedback_log"] = (log + "\n\n" + entry).strip() if log else entry
    context = state.get("context", "")
    state["context"] = (context + "\n\n" + entry).strip() if context else entry


def requester_node(state: State) -> State:
    agent = RequesterAgent()
    return agent.invoke(state)


def fixer_node(state: State) -> State:
    agent = FixerAgent()
    return agent.invoke(state)


def tester_node(state: State) -> State:
    if state.get("fix_failed"):
        summary = state.get("fixer_summary", "Falha ao aplicar patch")
        state.update({
            "test_passed": False,
            "test_logs": "Patch não aplicado; testes não executados.",
            "tester_summary": summary,
            "fix_failed": False,
        })
        return state
    agent = TesterAgent()
    return agent.invoke(state)


def sonar_node(state: State) -> State:
    run_sonar_scanner()
    client = SonarQubeClient()
    issues = client.search_issues(severities=ISSUE_SEVERITIES or None)
    issue_key = state["issue"].key
    remaining = [issue for issue in issues if issue.key == issue_key]
    if not remaining:
        state.update({
            "sonar_passed": True,
            "sonar_summary": "Issue resolvido segundo SonarQube",
        })
    else:
        formatted = format_issues_for_prompt(
            {
                "severity": item.severity,
                "rule": item.rule,
                "component": item.component,
                "line": item.line,
                "message": item.message,
            }
            for item in remaining
        )
        state.update({
            "sonar_passed": False,
            "sonar_summary": formatted,
        })
    return state


def after_patch_router(state: State) -> str:
    if state.get("fix_failed"):
        append_feedback(state, "Falha do Fixer", state.get("fixer_summary", ""))
        attempt = int(state.get("attempt", 1))
        if attempt >= MAX_ROUNDS:
            return "end"
        state["attempt"] = attempt + 1
        state["fix_failed"] = False
        return "requester"
    return "tester"


def after_tester_router(state: State) -> str:
    attempt = int(state.get("attempt", 1))
    if state.get("test_passed"):
        return "sonar"
    append_feedback(state, "Falhas do Tester", state.get("tester_summary", ""))
    if attempt >= MAX_ROUNDS:
        return "end"
    state["attempt"] = attempt + 1
    return "fixer"


def after_sonar_router(state: State) -> str:
    if state.get("sonar_passed"):
        return "deployment"
    append_feedback(state, "Feedback Sonar", state.get("sonar_summary", ""))
    attempt = int(state.get("attempt", 1))
    if attempt >= MAX_ROUNDS:
        return "end"
    state["attempt"] = attempt + 1
    return "fixer"


def run_graph_for_issue(issue: Issue) -> State:
    builder = StateGraph(State)
    builder.add_node("requester", requester_node)
    builder.add_node("fixer", fixer_node)
    builder.add_node("patcher", apply_patch_node)
    builder.add_node("tester", tester_node)
    builder.add_node("sonar", sonar_node)
    builder.add_node("deployment", deployment_node)

    builder.add_edge(START, "requester")
    builder.add_edge("requester", "fixer")
    builder.add_edge("fixer", "patcher")
    builder.add_conditional_edges("patcher", after_patch_router, {
        "tester": "tester",
        "requester": "requester",
        "end": END,
    })
    builder.add_conditional_edges("tester", after_tester_router, {
        "sonar": "sonar",
        "fixer": "fixer",
        "end": END,
    })
    builder.add_conditional_edges("sonar", after_sonar_router, {
        "deployment": "deployment",
        "fixer": "fixer",
        "end": END,
    })
    builder.add_edge("deployment", END)

    graph = builder.compile(checkpointer=MemorySaver())
    initial_state: State = {"issue": issue, "attempt": 1, "feedback_log": ""}
    config = {
        "configurable": {"thread_id": issue.key},
        "recursion_limit": max(25, MAX_ROUNDS * 10),
    }
    final_state: State = graph.invoke(initial_state, config=config)
    LOGGER.info("Pipeline finalizado para issue %s", issue.key)
    LOGGER.debug("Estado final: %s", json.dumps({k: v for k, v in final_state.items() if k != "issue"}, ensure_ascii=False, indent=2))
    return final_state


def run_pipeline() -> List[State]:
    client = SonarQubeClient()
    LOGGER.info("Executando sonar-scanner inicial")
    run_sonar_scanner()
    issues = client.search_issues(severities=ISSUE_SEVERITIES or None)
    LOGGER.info("%d issue(s) encontradas", len(issues))
    results: List[State] = []
    for issue in issues:
        LOGGER.info("Iniciando fluxo para %s", format_issue(issue).strip().replace("\n", " | "))
        result = run_graph_for_issue(sonar_issue_to_issue(issue))
        results.append(result)
    return results


if __name__ == "__main__":
    try:
        outcomes = run_pipeline()
        LOGGER.info("Pipeline concluído para %d issue(s)", len(outcomes))
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Falha durante pipeline: %s", exc)
        raise
