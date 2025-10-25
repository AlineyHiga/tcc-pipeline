"""Optimized Requester agent with reduced token usage."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Union

from langchain_core.prompts import ChatPromptTemplate

from app.a2a.protocol import State
from app.llm_client import LLMClient
from app.utils import with_line_numbers
from rag_service.service import RAGService

LOGGER = logging.getLogger(__name__)

# Optimized system prompt - much shorter
SYSTEM_PROMPT = """
Analyze the SonarQube issue and provide specific fix instructions.
Include:
1. Which function/method needs changes
2. What specific changes to make
3. Why this fixes the issue
Be direct and technical.
"""

class OptimizedRequesterAgent:
    def __init__(
        self,
        temperature: float = 0.1,
        repo_root: Optional[Union[Path, str]] = None,
    ) -> None:
        self.llm = LLMClient(role="requester", temperature=temperature)
        env_root = os.getenv("A2A_REPO_ROOT")
        if repo_root:
            base = Path(repo_root)
        elif env_root:
            base = Path(env_root)
        else:
            base = Path.cwd()
        self.repo_root = base.resolve()
        
        # Initialize RAG service
        try:
            rag_index_path = self.repo_root / ".rag_index"
            if rag_index_path.exists():
                self.rag_service = RAGService(str(rag_index_path))
                LOGGER.info("RAG service initialized successfully")
            else:
                self.rag_service = None
                LOGGER.warning("RAG index not found, falling back to full context")
        except Exception as e:
            self.rag_service = None
            LOGGER.warning(f"Failed to initialize RAG service: {e}")
        
        # Optimized prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "File: {file_path}\nIssues: {issues_summary}\nCode: {code_context}\n\nProvide specific fix instructions including which function/lines to modify and how."),
        ])

    def invoke(self, state: State) -> State:
        issues = list(state.get("issues_for_file") or [])
        file_path = state.get("file_path", "")
        
        if not issues:
            state["context"] = "No issues to fix."
            return state

        # Create minimal context
        issues_summary = self._create_minimal_summary(issues)
        code_context = self._get_minimal_code_context(state)
        
        prompt_input = {
            "file_path": file_path,
            "issues_summary": issues_summary,
            "code_context": code_context,
        }
        
        prompt_value = self.prompt.format_prompt(**prompt_input)
        user_prompt = prompt_value.to_messages()[0].content
        
        LOGGER.debug("Optimized requester prompt size: %d chars", len(user_prompt))
        
        response = self.llm.invoke(SYSTEM_PROMPT, user_prompt)
        state["context"] = response.strip()
        
        return state

    def _create_minimal_summary(self, issues: List) -> str:
        """Create ultra-concise issue summary."""
        if not issues:
            return "No issues"
        
        # Group by rule type
        rules = {}
        for issue in issues[:3]:  # Max 3 issues
            rule = getattr(issue, 'rule', 'Unknown')
            line = getattr(issue, 'line', '?')
            if rule not in rules:
                rules[rule] = []
            rules[rule].append(str(line))
        
        summary_parts = []
        for rule, lines in rules.items():
            lines_str = ','.join(lines[:3])  # Max 3 lines per rule
            summary_parts.append(f"{rule} L{lines_str}")
        
        return "; ".join(summary_parts)

    def _get_minimal_code_context(self, state: State) -> str:
        """Get complete file context and identify target function using RAG."""
        issues = list(state.get('issues_for_file', []))
        file_content = state.get('property_file_preview', '')
        
        if not issues:
            return "No issues to analyze"
        
        # Always send complete file with line numbers
        if file_content:
            complete_file = with_line_numbers(file_content)
        else:
            complete_file = "File content unavailable"
        
        # Use RAG to identify target function and add to state
        target_function = None
        if self.rag_service:
            try:
                issue = issues[0]
                file_path = state.get('file_path', '')
                line = getattr(issue, 'line', 1)
                rule = getattr(issue, 'rule', '')
                message = getattr(issue, 'message', '')
                
                result = self.rag_service.retrieve_for_issue(
                    file_path=file_path,
                    line=line,
                    rule=rule,
                    message=message,
                    k=1
                )
                
                target_function = result['target']['symbol']
                LOGGER.info(f"RAG identified target function: {target_function}")
                
                # Store target function in state for fixer
                state['target_function'] = target_function
                
            except Exception as e:
                LOGGER.warning(f"RAG retrieval failed: {e}")
        
        # Return complete file with target function info
        if target_function:
            return f"Target function to fix: {target_function}\n\nComplete file:\n{complete_file}"
        else:
            return f"Complete file:\n{complete_file}"