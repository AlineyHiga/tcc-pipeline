"""Utility CLI to materialize the LangGraph builder and inspect the pipeline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver

from app.main import build_graph


def _uml_node_name(node_id: str) -> str:
    if node_id == "__start__":
        return "start"
    if node_id == "__end__":
        return "stop"
    return f"\"{node_id}\""


def _build_plantuml(graph) -> str:
    """Create a PlantUML activity diagram describing the flow."""
    lines = ["@startuml", "skinparam ArrowColor #5b4ede", "skinparam ActivityBackgroundColor #f2f0ff", "skinparam ActivityBorderColor #5b4ede", "start"]
    for edge in graph.edges:
        source = _uml_node_name(edge.source)
        target = _uml_node_name(edge.target)
        lines.append(f"{source} --> {target}")
    lines.append("stop")
    lines.append("@enduml")
    return "\n".join(lines)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def run(output_dir: Path) -> None:
    """Build the pipeline graph, emit artifacts, and print a quick overview."""
    graph_builder = build_graph()
    compiled = graph_builder.compile(checkpointer=MemorySaver())
    graph = compiled.get_graph()

    mermaid = graph.draw_mermaid()
    plantuml = _build_plantuml(graph)
    try:
        ascii_map = graph.draw_ascii()
        ascii_note = None
    except ImportError as exc:
        ascii_map = (
            "ASCII indisponivel: instale grandalf para habilitar este formato.\n"
            f"Detalhes: {exc}"
        )
        ascii_note = "ASCII exige dependencia opcional grandalf."

    mermaid_path = output_dir / "autofix_graph.mmd"
    plantuml_path = output_dir / "autofix_graph.puml"
    ascii_path = output_dir / "autofix_graph.txt"

    _write_text(mermaid_path, mermaid)
    _write_text(plantuml_path, plantuml)
    _write_text(ascii_path, ascii_map)

    print("Builder pronto. Artefatos gerados:")
    print(f"- Mermaid: {mermaid_path}")
    print(f"- PlantUML: {plantuml_path}")
    print(f"- ASCII  : {ascii_path}")
    if ascii_note:
        print(f"  > {ascii_note}")
    print("\nPrevia ASCII:\n")
    print(ascii_map)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera artefatos de visualização para o LangGraph AutoFix."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Diretório onde os arquivos .mmd e .txt serão escritos.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    run(args.output_dir)


if __name__ == "__main__":
    main()
