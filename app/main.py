"""Main orchestrator using LangGraph for AutoFix pipeline."""
import os
import logging
import uuid
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv
from .logging_setup import setup_logging, set_ctx, log_event, Span, save_artifact, get_run_artifacts_dir

from langgraph.graph import StateGraph, END
# from langgraph.checkpoint.sqlite import SqliteSaver  # Optional checkpointer

from .a2a.protocol import AgentState
from .sonarqube_client import SonarQubeClient
from .a2a.property_agent import PropertyAgent
from .a2a.tester_agent import TesterAgent
from .a2a.fixer_agent import FixerAgent
from .patcher import SafePatcher
from .rag.retriever import RAGRetriever
from .utils import run_sonar_scanner, mask_secrets

# Setup structured logging
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()


class AutoFixPipeline:
    """LangGraph-based AutoFix pipeline orchestrator."""
    
    def __init__(self):
        self.sonar_client = SonarQubeClient()
        self.property_agent = PropertyAgent()
        self.tester_agent = TesterAgent()
        self.fixer_agent = FixerAgent()
        self.patcher = SafePatcher()
        self.retriever = RAGRetriever()
        
        # Build graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("sonar_ingest", self.sonar_ingest)
        workflow.add_node("plan", self.plan)
        workflow.add_node("rule_rag", self.rule_rag)
        workflow.add_node("prop_spec", self.prop_spec)
        workflow.add_node("prop_gen", self.prop_gen)
        workflow.add_node("prop_run", self.prop_run)
        workflow.add_node("fix_plan", self.fix_plan)
        workflow.add_node("patch", self.patch)
        workflow.add_node("tests", self.tests)
        workflow.add_node("sonar_rescan", self.sonar_rescan)
        workflow.add_node("lot_gate", self.lot_gate)
        workflow.add_node("update_lot", self.update_lot)
        workflow.add_node("pr_builder", self.pr_builder)
        
        # Add edges
        workflow.set_entry_point("sonar_ingest")
        workflow.add_edge("sonar_ingest", "plan")
        workflow.add_edge("plan", "rule_rag")
        workflow.add_edge("rule_rag", "prop_spec")
        workflow.add_edge("prop_spec", "prop_gen")
        workflow.add_edge("prop_gen", "prop_run")
        workflow.add_edge("prop_run", "fix_plan")
        workflow.add_edge("fix_plan", "patch")
        workflow.add_edge("patch", "tests")
        workflow.add_edge("tests", "sonar_rescan")
        workflow.add_edge("sonar_rescan", "lot_gate")
        
        # Conditional edges
        workflow.add_conditional_edges(
            "lot_gate",
            self._should_continue,
            {
                "continue": "rule_rag",  # Loop back for refinement
                "next_lot": "update_lot",  # Update lot then continue
                "finish": "pr_builder"
            }
        )
        workflow.add_edge("update_lot", "rule_rag")
        workflow.add_edge("pr_builder", END)
        
        return workflow.compile()
    
    def run(self) -> Dict[str, Any]:
        """Execute the AutoFix pipeline."""
        # Initialize run context
        run_id = str(uuid.uuid4())
        set_ctx(run_id=run_id, node="START")
        
        # Initialize state
        initial_state = AgentState(
            project_key=os.getenv("SONAR_PROJECT_KEY", ""),
            repo_path=os.getenv("AUTOFIX_TARGET_ROOT", "."),
            sonar_server=os.getenv("SONARQUBE_URL", ""),
            sonar_token=os.getenv("SONARQUBE_TOKEN", ""),
            max_rounds=int(os.getenv("MAX_ROUNDS", "3"))
        )
        
        log_event("pipeline.start", 
                 project_key=initial_state.project_key,
                 repo_path=initial_state.repo_path,
                 max_rounds=initial_state.max_rounds)
        
        try:
            with Span(logger, "pipeline.execute"):
                # Execute graph
                result = self.graph.invoke(initial_state)
            
            pr_count = len(result.get('pr_urls', []))
            log_event("pipeline.complete", pr_count=pr_count, pr_urls=result.get('pr_urls', []))
            
            return result
            
        except Exception as e:
            log_event("pipeline.error", error=str(e))
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise
    
    # Node implementations
    def sonar_ingest(self, state: AgentState) -> Dict[str, Any]:
        """Run initial scan and collect issues from SonarQube."""
        set_ctx(node="SONAR_INGEST")
        
        try:
            # Run initial sonar scan
            with Span(logger, "sonar.initial_scan"):
                run_sonar_scanner(cwd=state.repo_path)
            
            with Span(logger, "sonar.collect_issues"):
                issues = self.sonar_client.list_issues(state.project_key)
            
            log_event("sonar.issues", count=len(issues), project_key=state.project_key)
            
            # Save issues artifact
            issues_path = save_artifact(
                f"{get_run_artifacts_dir()}/sonar",
                "issues.json",
                json.dumps(issues, indent=2)
            )
            
            # Normalize issues
            normalized_issues = []
            for issue in issues:
                normalized_issue = self._normalize_issue(issue, state.repo_path)
                normalized_issues.append(normalized_issue)
            
            log_event("sonar.ingest.complete", 
                     normalized_count=len(normalized_issues),
                     artifact_path=issues_path)
            
            return {"issues": normalized_issues}
            
        except Exception as e:
            log_event("sonar.ingest.error", error=str(e))
            logger.error(f"SonarQube ingest failed: {e}")
            return {"issues": []}
    
    def plan(self, state: AgentState) -> Dict[str, Any]:
        """Group issues into lots by rule and directory."""
        set_ctx(node="PLAN")
        
        issues = self._dedupe_issues(state.issues)
        lots = {}
        
        for issue in issues:
            rule_key = issue.get("rule", "unknown")
            component = issue.get("component", "")
            directory = str(Path(component).parent) if component else "."
            
            lot_key = f"{rule_key}#{directory}"
            
            if lot_key not in lots:
                lots[lot_key] = {
                    "ruleKey": rule_key,
                    "directory": directory,
                    "issues": [],
                    "budget": 300  # LOC limit
                }
            
            lots[lot_key]["issues"].append(issue)
        
        lots_list = list(lots.values())
        
        # Set lot_id for first lot
        if lots_list:
            first_lot = lots_list[0]
            lot_id = hashlib.sha256(f"{first_lot['ruleKey']}:{first_lot['directory']}".encode()).hexdigest()[:8]
            set_ctx(lot_id=lot_id)
        
        log_event("plan.complete", 
                 total_issues=len(issues),
                 lots_count=len(lots_list),
                 lots=[{"rule": lot["ruleKey"], "dir": lot["directory"], "count": len(lot["issues"])} for lot in lots_list])
        
        return {
            "issues": issues,
            "lots": lots_list,
            "lot_index": 0,
            "current_lot": lots_list[0] if lots_list else None
        }
    
    def rule_rag(self, state: AgentState) -> Dict[str, Any]:
        """Retrieve context for the current rule."""
        current_lot = state.current_lot
        if not current_lot:
            return {"rag_ctx": {}}
        
        rule_key = current_lot.get("ruleKey", "")
        logger.info(f"Retrieving context for rule {rule_key}")
        
        query = f"sonar rule {rule_key} fix examples"
        rag_ctx = self.retriever.retrieve(query)
        
        return {"rag_ctx": rag_ctx}
    
    def prop_spec(self, state: AgentState) -> Dict[str, Any]:
        """Generate property specifications."""
        logger.info("Generating property specifications")
        
        prop_spec = self.property_agent.prop_spec(state.model_dump())
        
        return {"prop_spec": prop_spec}
    
    def prop_gen(self, state: AgentState) -> Dict[str, Any]:
        """Generate Hypothesis test files."""
        logger.info("Generating property test files")
        
        result = self.property_agent.prop_gen(state.model_dump())
        
        return {"prop_gen": result}
    
    def prop_run(self, state: AgentState) -> Dict[str, Any]:
        """Execute property tests."""
        logger.info("Running property tests")
        
        prop_result = self.tester_agent.prop_run(state.repo_path)
        
        failed = prop_result.get("failed")
        if failed is None:
            # fallback: use exit_code
            failed = 1 if prop_result.get("exit_code", 1) != 0 else 0
        if failed > 0:
            logger.warning("Property tests failed - counterexamples available")
        else:
            logger.info("Property tests passed")
        
        return {"prop_result": prop_result}
    
    def fix_plan(self, state: AgentState) -> Dict[str, Any]:
        """Generate fix plan."""
        logger.info("Creating fix plan")
        
        fix_plan = self.fixer_agent.fix_plan(state.model_dump())
        
        return {"fix_plan": fix_plan}
    
    def patch(self, state: AgentState) -> Dict[str, Any]:
        """Generate and apply patch."""
        logger.info("Generating and applying patch")
        
        patch_diff = self.fixer_agent.generate_patch(state.model_dump())
        
        if patch_diff:
            # Get budget from current lot
            budget = None
            if state.current_lot:
                budget = state.current_lot.get("budget", 300)
            
            result = self.patcher.apply_patch(patch_diff, state.repo_path, budget=budget)
            if result["applied"]:
                logger.info(f"Patch applied successfully ({result['files_changed']} files)")
                return {"patch_diff": patch_diff, "patch_applied": True}
            else:
                logger.error(f"Patch failed: {result['error']}")
                return {
                    "patch_diff": patch_diff, 
                    "patch_applied": False,
                    "patch_error": result["error"],
                    "patch_reason": result.get("reason", "unknown")
                }
        else:
            last_resp = (self.fixer_agent.last_llm_response or "").strip()
            if last_resp:
                logger.error("No patch generated; last LLM response:\n%s", last_resp)
            else:
                logger.error("No patch generated; last LLM response was empty.")
            return {
                "patch_diff": "",
                "patch_applied": False,
                "patch_error": "No patch generated",
                "patch_llm_response": last_resp,
            }
    
    def tests(self, state: AgentState) -> Dict[str, Any]:
        """Run full test suite with coverage."""
        # Skip tests if patch was not applied
        if not state.model_dump().get("patch_applied", True):
            logger.warning("Skipping tests - patch was not applied")
            return {
                "test_report": {
                    "status": "skipped",
                    "exit_code": 0,
                    "failed": 0,
                    "tests": 0,
                    "skipped": 0,
                    "stdout": "",
                    "stderr": "Tests skipped due to patch failure",
                    "coverage_xml_exists": False,
                    "junit_path": None
                }
            }
        
        logger.info("Running full test suite with coverage")
        
        test_report = self.tester_agent.run_tests_with_coverage(state.repo_path)
        
        failed = test_report.get("failed")
        if failed is None:
            # fallback: use exit_code
            failed = 1 if test_report.get("exit_code", 1) != 0 else 0
        if failed > 0:
            logger.warning(f"Tests failed: {failed} failures")
        else:
            logger.info("All tests passed")
        
        return {"test_report": test_report}
    
    def sonar_rescan(self, state: AgentState) -> Dict[str, Any]:
        """Re-run SonarQube scanner."""
        logger.info("Re-scanning with SonarQube")
        
        try:
            # Run scanner with coverage
            run_sonar_scanner(
                cwd=state.repo_path,
                extra_args=["-Dsonar.python.coverage.reportPaths=coverage.xml"]
            )
            
            # Get fresh issues (OPEN status only)
            fresh_raw = self.sonar_client.list_issues(state.project_key, status="OPEN")
            fresh_issues = self._dedupe_issues(fresh_raw)
            normalized_fresh = [
                self._normalize_issue(issue, state.repo_path) for issue in fresh_issues
            ]
            
            # Build lots keyed by rule/directory with existing budgets preserved
            existing_budgets = {
                f"{lot['ruleKey']}#{lot['directory']}": lot.get("budget", 300)
                for lot in (state.lots or [])
            }
            
            lots_by_key: Dict[str, Dict[str, Any]] = {}
            for issue in normalized_fresh:
                rule_key = issue.get("rule", "unknown")
                component = issue.get("component", "")
                directory = str(Path(component).parent) if component else "."
                lot_key = f"{rule_key}#{directory}"
                lot_entry = lots_by_key.setdefault(
                    lot_key,
                    {
                        "ruleKey": rule_key,
                        "directory": directory,
                        "issues": [],
                        "budget": existing_budgets.get(lot_key, 300),
                    },
                )
                lot_entry["issues"].append(issue)
            
            ordered_keys = [
                f"{lot['ruleKey']}#{lot['directory']}" for lot in (state.lots or [])
            ]
            new_lots: List[Dict[str, Any]] = []
            for key in ordered_keys:
                if key in lots_by_key:
                    new_lots.append(lots_by_key.pop(key))
            new_lots.extend(lots_by_key.values())
            
            updated_current_lot = None
            if state.current_lot:
                current_key = f"{state.current_lot.get('ruleKey')}#{state.current_lot.get('directory')}"
                for lot in new_lots:
                    lot_key = f"{lot.get('ruleKey')}#{lot.get('directory')}"
                    if lot_key == current_key:
                        updated_current_lot = lot
                        break
                if updated_current_lot is None:
                    # Lot resolved; keep metadata but drop issues
                    updated_current_lot = dict(state.current_lot)
                    updated_current_lot["issues"] = []
            else:
                updated_current_lot = None
            
            lot_issues = updated_current_lot.get("issues", []) if updated_current_lot else []
            
            # Check quality gate
            qg = self.sonar_client.quality_gate(state.project_key)
            
            payload: Dict[str, Any] = {
                "sonar_rescan": {
                    "quality_gate": qg.get("projectStatus", {}).get("status", "ERROR"),
                    "lot_clean": len(lot_issues) == 0,
                    "total_issues": len(normalized_fresh)
                }
            }
            payload["issues"] = normalized_fresh
            payload["lots"] = new_lots
            if updated_current_lot is not None:
                payload["current_lot"] = updated_current_lot
            
            return payload
            
        except Exception as e:
            logger.error(f"Sonar rescan failed: {e}")
            return {
                "sonar_rescan": {
                    "quality_gate": "ERROR",
                    "lot_clean": False,
                    "error": str(e)
                }
            }
    
    def lot_gate(self, state: AgentState) -> Dict[str, Any]:
        """Decide whether to continue, process next lot, or finish."""
        sonar_rescan = state.sonar_rescan
        ok_gate = sonar_rescan.get("quality_gate") == "OK"
        lot_clean = sonar_rescan.get("lot_clean", False)
        
        if ok_gate and lot_clean:
            # próximo lote ou finalizar
            if state.lot_index + 1 < len(state.lots):
                return {"next_action": "next_lot"}
            else:
                return {"next_action": "finish"}
        
        if state.current_round + 1 < state.max_rounds:
            return {
                "next_action": "retry",
                "current_round": state.current_round + 1
            }
        
        # Max rounds reached - skip to next lot or finish
        if state.lot_index + 1 < len(state.lots):
            return {"next_action": "next_lot"}
        else:
            return {"next_action": "finish"}
    
    def pr_builder(self, state: AgentState) -> Dict[str, Any]:
        """Create pull request."""
        logger.info("Creating pull request")
        
        try:
            pr_result = self._create_pr(state)
            return {"pr_urls": [pr_result.get("pr_url", "")]}
        except Exception as e:
            logger.error(f"PR creation failed: {e}")
            return {"pr_urls": []}
    
    def _create_pr(self, state: AgentState) -> Dict[str, Any]:
        """Create a real pull request using git and gh CLI."""
        import hashlib
        import uuid
        from .utils import run_cmd
        
        lot = state.current_lot
        if not lot:
            raise ValueError("No current lot available for PR creation")
        
        # Generate unique branch name
        rule_key = lot.get("ruleKey", "unknown")
        directory = lot.get("directory", "")
        suffix = hashlib.sha1(f'{rule_key}:{directory}'.encode()).hexdigest()[:8]
        branch = f'fix/sonar-{rule_key.lower().replace(":", "-")}-{suffix}'
        
        repo_path = state.repo_path

        # Ensure branch name is unique per run
        exists = run_cmd(["git", "rev-parse", "--verify", "--quiet", branch], workdir=repo_path)
        if exists["exit_code"] == 0:
            unique_suffix = uuid.uuid4().hex[:6]
            branch = f"{branch}-{unique_suffix}"
            logger.info(f"Target branch already exists, using fallback branch {branch}")
        
        # Create and checkout branch
        result = run_cmd(["git", "checkout", "-b", branch], workdir=repo_path)
        if result["exit_code"] != 0:
            raise RuntimeError(f"Failed to create branch: {result['stderr']}")
        
        # Stage all changes
        result = run_cmd(["git", "add", "-A"], workdir=repo_path)
        if result["exit_code"] != 0:
            raise RuntimeError(f"Failed to stage changes: {result['stderr']}")
        
        # Check if there are changes to commit
        result = run_cmd(["git", "diff", "--cached", "--quiet"], workdir=repo_path)
        if result["exit_code"] == 0:
            logger.info("No changes to commit - skipping PR creation")
            return {"branch": branch, "pr_url": ""}
        
        # Commit changes
        commit_msg = f'fix(sonar): {rule_key} — lote {directory}'
        result = run_cmd(["git", "commit", "-m", commit_msg], workdir=repo_path)
        if result["exit_code"] != 0:
            raise RuntimeError(f"Failed to commit: {result['stderr']}")
        
        # Push branch
        result = run_cmd(["git", "push", "-u", "origin", branch], workdir=repo_path)
        if result["exit_code"] != 0:
            raise RuntimeError(f"Failed to push branch: {result['stderr']}")
        
        # Create PR using gh CLI
        pr_title = f"fix(sonar): Resolve {rule_key} violations"
        pr_body = self._generate_pr_body(lot, state)
        
        result = run_cmd([
            "gh", "pr", "create",
            "--title", pr_title,
            "--body", pr_body,
            "--head", branch,
            "--base", os.getenv("BASE_BRANCH", "main")
        ], workdir=repo_path)
        
        if result["exit_code"] != 0:
            raise RuntimeError(f"Failed to create PR: {result['stderr']}")
        
        pr_url = result["stdout"].strip()
        logger.info(f"Created PR: {pr_url}")
        
        return {"branch": branch, "pr_url": pr_url}
    
    def _generate_pr_body(self, lot: Dict[str, Any], state: AgentState) -> str:
        """Generate PR description body."""
        rule_key = lot.get("ruleKey", "unknown")
        issues = lot.get("issues", [])
        directory = lot.get("directory", "")
        
        body = f"""## AutoFix: {rule_key}

**Rule**: {rule_key}
**Directory**: {directory}
**Issues Fixed**: {len(issues)}

### Changes Made

This PR automatically fixes SonarQube violations for rule `{rule_key}` in the `{directory}` directory.

### Issues Addressed

"""
        
        for i, issue in enumerate(issues[:5], 1):  # Limit to 5 issues
            component = issue.get("component_path") or issue.get("component", "")
            line = issue.get("line", "")
            message = issue.get("message", "")
            body += f"{i}. **{component}:{line}** - {message}\n"
        
        if len(issues) > 5:
            body += f"\n... and {len(issues) - 5} more issues\n"
        
        body += "\n### Quality Gate\n\n"
        
        sonar_rescan = state.sonar_rescan
        if sonar_rescan:
            qg_status = sonar_rescan.get("quality_gate", "UNKNOWN")
            lot_clean = sonar_rescan.get("lot_clean", False)
            body += f"- Quality Gate: {qg_status}\n"
            body += f"- Lot Clean: {'✅ Yes' if lot_clean else '❌ No'}\n"
        
        body += "\n---\n*Generated by AutoFix Pipeline*"
        
        return body
    
    def _normalize_issue(self, issue: Dict[str, Any], repo_path: str) -> Dict[str, Any]:
        """Normalize issue with resolved paths and metadata."""
        import os
        
        # Resolve component path
        component = issue.get("component", "")
        if ":" in component:
            rel_path = component.split(":", 1)[1]
        else:
            rel_path = component
        
        component_path = os.path.normpath(os.path.join(repo_path, rel_path))
        directory = os.path.dirname(component_path)
        
        # Add normalized fields
        issue["ruleKey"] = issue.get("rule", "")
        issue["type"] = issue.get("type", "CODE_SMELL")  # Bug/Vulnerability/Code_Smell
        issue["component_path"] = component_path
        issue["dir"] = directory
        
        return issue
    
    def _issue_stable_key(self, issue: Dict[str, Any]) -> str:
        """Build a stable identifier for deduplicating Sonar issues."""
        rule = issue.get("rule") or issue.get("ruleKey") or ""
        component = issue.get("component", "")
        line = (
            issue.get("textRange", {}).get("startLine")
            or issue.get("line")
            or 0
        )
        message = issue.get("message", "")
        message_hash = hashlib.sha1(message.encode("utf-8")).hexdigest()[:8] if message else "nomsg"
        return f"{rule}:{component}:{line}:{message_hash}"
    
    def _dedupe_issues(self, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate issues based on a stable key."""
        seen: set[str] = set()
        deduped: List[Dict[str, Any]] = []
        for issue in issues or []:
            key = self._issue_stable_key(issue)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(issue)
        return deduped
    
    def update_lot(self, state: AgentState) -> Dict[str, Any]:
        """Update to next lot."""
        new_index = state.lot_index + 1
        if new_index < len(state.lots):
            new_lot = state.lots[new_index]
            logger.info(f"Moving to lot {new_index}: {new_lot.get('ruleKey', 'unknown')}")
            return {
                "lot_index": new_index,
                "current_lot": new_lot,
                "current_round": 0
            }
        return {}
    
    def _should_continue(self, state: AgentState) -> str:
        """Determine next step based on lot gate status."""
        if state.next_action == "retry":
            return "continue"
        if state.next_action == "next_lot":
            # Update lot index and current lot
            new_index = state.lot_index + 1
            if new_index < len(state.lots):
                # This will be handled by updating the state in the graph
                return "next_lot"
            else:
                return "finish"
        return "finish"


def main():
    """Main entry point."""
    # Setup logging first
    setup_logging()
    
    pipeline = AutoFixPipeline()
    result = pipeline.run()
    
    print(f"Pipeline completed. PRs: {result.get('pr_urls', [])}")


if __name__ == "__main__":
    main()
