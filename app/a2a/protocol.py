"""Agent2Agent protocol definitions and state management."""
from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class AgentState(BaseModel):
    """Central state for the AutoFix pipeline."""
    
    # Configuration
    project_key: str
    repo_path: str
    sonar_server: str
    sonar_token: str
    max_rounds: int = 3
    current_round: int = 0
    lot_index: int = 0
    
    # Pipeline data
    issues: List[Dict[str, Any]] = []
    lots: List[Dict[str, Any]] = []
    current_lot: Optional[Dict[str, Any]] = None
    
    # Agent outputs
    rag_ctx: Dict[str, Any] = {}
    prop_spec: Dict[str, Any] = {}
    prop_gen: Dict[str, Any] = {}  # Separated from prop_result
    prop_result: Dict[str, Any] = {}
    fix_plan: Dict[str, Any] = {}
    patch_diff: Optional[str] = None
    test_report: Dict[str, Any] = {}
    sonar_rescan: Dict[str, Any] = {}
    pr_urls: List[str] = []
    
    # Flow control
    next_action: str = "continue"  # "retry", "next_lot", "finish"
    fallback_tried: bool = False
    prev_issue_hash: str = ""
    attempts_same_issues: int = 0
    lot_start_time: Optional[float] = None
    
    # Feedback loop
    feedback: List[str] = []
    
    class Config:
        arbitrary_types_allowed = True


class A2AMessage(BaseModel):
    """Message format for agent communication."""
    
    sender: str
    receiver: str
    content: str
    metadata: Dict[str, Any] = {}
    timestamp: Optional[str] = None
