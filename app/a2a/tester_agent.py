"""Testing agent for running property tests and full test suite."""
import subprocess
import json
import tempfile
import shutil
import os
import re
from pathlib import Path
from typing import Dict, Any
from ..utils import run_cmd


class TesterAgent:
    """Executes property tests and full test suite with coverage."""
    
    def __init__(self):
        self.isolation_mode = os.getenv("TEST_ISOLATION_MODE", "docker")
        self.iso_image = os.getenv("TEST_ISO_IMAGE", "python:3.11-slim")
        self.timeout_secs = int(os.getenv("TEST_TIMEOUT_SECS", "300"))
        self.cov_target = os.getenv("COV_TARGET", "src")
    
    def _ensure_reports_dir(self, repo_path: str) -> None:
        """Ensure reports directory exists."""
        reports_dir = Path(repo_path) / "reports"
        reports_dir.mkdir(exist_ok=True)
    
    def _run_isolated_docker(self, cmd: list[str], repo_path: str) -> Dict[str, Any]:
        """Run command in Docker container."""
        docker_cmd = [
            "docker", "run", "--rm",
            "--network=none",  # No network access
            "-v", f"{Path(repo_path).resolve()}:/workspace",
            "-w", "/workspace",
            self.iso_image,
            "sh", "-c",
            f"pip install -q pytest hypothesis pytest-cov coverage && {' '.join(cmd)}"
        ]
        
        return run_cmd(docker_cmd, timeout_s=self.timeout_secs, workdir=repo_path)
    
    def _run_isolated_local(self, cmd: list[str], repo_path: str) -> Dict[str, Any]:
        """Run command in local environment."""
        return run_cmd(cmd, timeout_s=self.timeout_secs, workdir=repo_path)
    
    def _run_isolated(self, cmd: list[str], repo_path: str) -> Dict[str, Any]:
        """Run command in isolated environment."""
        if self.isolation_mode == "docker" and shutil.which("docker"):
            return self._run_isolated_docker(cmd, repo_path)
        else:
            return self._run_isolated_local(cmd, repo_path)
    
    def prop_run(self, repo_path: str = ".") -> Dict[str, Any]:
        """Run only property tests and capture results."""
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"prop_run called with repo_path: {repo_path}")
        
        self._ensure_reports_dir(repo_path)
        
        cmd = [
            "python", "-m", "pytest", "-v", "tests_prop",
            "--maxfail=1",
            "--junitxml=reports/junit_prop.xml",
            "--tb=short"
        ]
        
        result = self._run_isolated(cmd, repo_path)
        
        # Normalize exit codes
        exit_code = result["exit_code"]
        if exit_code == 5:  # No tests collected
            status = "skipped"
            passed, failed = 0, 0
        elif exit_code == 0:
            status = "passed"
            passed, failed = self._parse_junit_results(Path(repo_path) / "reports" / "junit_prop.xml")
        else:
            status = "failed"
            passed, failed = self._parse_junit_results(Path(repo_path) / "reports" / "junit_prop.xml")
        
        # Extract counterexamples
        examples = self._extract_counterexamples(result["stdout"] + "\n" + result["stderr"])
        
        return {
            "status": status,
            "passed": passed,
            "failed": failed,
            "examples": examples,
            "exit_code": exit_code,
            "stdout": result["stdout"],
            "stderr": result["stderr"]
        }
    
    def run_tests_with_coverage(self, repo_path: str = ".") -> Dict[str, Any]:
        """Run full test suite with coverage generation."""
        import logging
        logger = logging.getLogger(__name__)
        
        self._ensure_reports_dir(repo_path)
        
        cmd = [
            "python", "-m", "pytest", "-v", "--maxfail=1",
            f"--cov={self.cov_target}",
            "--cov-report=xml:coverage.xml",
            "--junitxml=reports/junit.xml",
            "--tb=short"
        ]
        
        result = self._run_isolated(cmd, repo_path)
        
        # Normalize exit codes
        exit_code = result["exit_code"]
        if exit_code == 5:  # No tests collected
            status = "skipped"
        elif exit_code == 0:
            status = "passed"
        else:
            status = "failed"
        
        # Parse JUnit results with full counts
        junit_path = Path(repo_path) / "reports" / "junit.xml"
        counts = self._junit_counts(str(junit_path)) if junit_path.exists() else {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
        
        # Verify coverage.xml exists
        coverage_xml_path = Path(repo_path) / "coverage.xml"
        coverage_xml_exists = coverage_xml_path.exists()
        
        if not coverage_xml_exists:
            logger.warning("coverage.xml not generated - running coverage manually")
            # Fallback: run coverage manually
            cov_cmd = ["python", "-m", "coverage", "xml"]
            self._run_isolated(cov_cmd, repo_path)
            coverage_xml_exists = coverage_xml_path.exists()
        
        return {
            "status": status,
            "exit_code": exit_code,
            "failed": counts["failures"] + counts["errors"],  # <- expected key
            "tests": counts["tests"],
            "skipped": counts["skipped"],
            "stdout": result["stdout"][:20000],
            "stderr": result["stderr"][:20000],
            "coverage_xml_exists": coverage_xml_exists,
            "junit_path": str(junit_path) if junit_path.exists() else None,
        }
    
    def _extract_counterexamples(self, output: str) -> list:
        """Extract Hypothesis counterexamples from test output."""
        examples = []
        
        # Pattern for Hypothesis falsifying examples
        falsifying_pattern = r"Falsifying example:\s*([^\n]+(?:\n\s+[^\n]+)*)"
        matches = re.finditer(falsifying_pattern, output, re.MULTILINE)
        
        for match in matches:
            example_text = match.group(1).strip()
            examples.append({
                "type": "counterexample",
                "value": example_text,
                "shrunk": True
            })
        
        # Also look for assertion errors with context
        assertion_pattern = r"AssertionError: ([^\n]+)"
        assertion_matches = re.finditer(assertion_pattern, output)
        
        for match in assertion_matches:
            examples.append({
                "type": "assertion",
                "value": match.group(1).strip(),
                "shrunk": False
            })
        
        return examples
    
    def _junit_counts(self, junit_path: str) -> dict:
        """Parse JUnit XML to extract test counts."""
        counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
        try:
            import xml.etree.ElementTree as ET
            root = ET.parse(junit_path).getroot()
            # junitxml: <testsuite tests="..." failures="..." errors="..." skipped="...">
            ts = root if root.tag == "testsuite" else root.find("testsuite")
            if ts is not None:
                for k in counts:
                    if k in ts.attrib:
                        counts[k] = int(ts.attrib[k])
        except Exception:
            pass
        return counts
    
    def _parse_junit_results(self, junit_file: Path) -> tuple[int, int]:
        """Parse JUnit XML to extract pass/fail counts."""
        if not junit_file.exists():
            return 0, 0
        
        counts = self._junit_counts(str(junit_file))
        failed = counts["failures"] + counts["errors"]
        passed = counts["tests"] - failed - counts["skipped"]
        return max(0, passed), failed