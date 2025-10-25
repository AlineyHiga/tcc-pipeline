"""Smoke tests for the AutoFix pipeline."""
import pytest
from pathlib import Path
from app.main import AutoFixPipeline
from app.a2a.protocol import AgentState
from app.a2a.property_agent import PropertyAgent


def test_pipeline_graph_builds():
    """Test that the LangGraph compiles without errors."""
    pipeline = AutoFixPipeline()
    assert pipeline.graph is not None


def test_agent_state_creation():
    """Test AgentState model validation."""
    state = AgentState(
        project_key="test-project",
        repo_path=".",
        sonar_server="http://localhost:9000",
        sonar_token="test-token"
    )
    
    assert state.project_key == "test-project"
    assert state.max_rounds == 3
    assert state.current_round == 0
    assert len(state.issues) == 0


def test_property_agent_spec_generation():
    """Test property specification generation."""
    agent = PropertyAgent()
    
    # Mock state with a simple issue
    state = {
        "current_lot": {
            "ruleKey": "S106",
            "issues": [{
                "message": "Replace this use of System.out by a logger",
                "component": "src/example.py",
                "line": 10
            }]
        }
    }
    
    spec = agent.prop_spec(state)
    
    # Should return some form of specification
    assert isinstance(spec, dict)
    assert "invariants" in spec or "metamorphisms" in spec or "oracles" in spec


def test_property_generation():
    """Test that property test files can be generated."""
    agent = PropertyAgent()
    
    state = {
        "prop_spec": {
            "invariants": ["no_exception_on_valid_input"],
            "metamorphisms": [],
            "oracles": ["type_preservation"]
        },
        "current_lot": {
            "issues": [{
                "component": "src/example.py",
                "line": 10
            }]
        }
    }
    
    result = agent.prop_gen(state)
    
    assert isinstance(result, dict)
    assert "files_generated" in result
    assert "test_files" in result


@pytest.mark.integration
def test_rag_retrieval():
    """Test RAG retrieval functionality."""
    from app.rag.retriever import RAGRetriever
    
    retriever = RAGRetriever()
    result = retriever.retrieve("python function testing")
    
    assert isinstance(result, dict)
    assert "contexts" in result
    assert "citations" in result
    assert "few_shots" in result