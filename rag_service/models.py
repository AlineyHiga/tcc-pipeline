"""Data models for RAG service."""
from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class Chunk:
    id: str
    path: str
    symbol: str
    kind: str  # "class", "def", "test"
    lines: Tuple[int, int]
    code: str
    summary: str
    imports: List[str]
    called_symbols: List[str]
    neighbors: List[str]

@dataclass
class Target:
    path: str
    symbol: str
    kind: str
    lines: Tuple[int, int]

@dataclass
class ContextSymbol:
    path: str
    symbol: str
    kind: str
    lines: Tuple[int, int]
    reason: str  # "owner", "called_by_target", "imports", "touches_target"

@dataclass
class RetrievalResult:
    target: Target
    context: List[ContextSymbol]
    explain: str

@dataclass
class RequesterBriefing:
    issue_overview: dict
    target: Target
    context_symbols: List[ContextSymbol]
    change_plan: List[dict]
    constraints: dict
    test_hints: List[str]
    unknowns: List[str]