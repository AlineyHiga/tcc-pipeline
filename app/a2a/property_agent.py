"""Property-based testing agent for generating Hypothesis tests."""
import ast
import os
import logging
import json
import re
from pathlib import Path
from textwrap import dedent
from typing import Dict, Any, List
from ..llm_client import get_llm_client
from ..rag.retriever import RAGRetriever

# GBNF grammar for property objects
PROP_GBNF = r"""
root        ::= ws "[" ws obj (ws "," ws obj)* ws "]" ws
obj         ::= "{" ws kv (ws "," ws kv)* ws "}"
kv          ::= name_kv | type_kv | desc_kv | domain_kv | fallback_kv
name_kv     ::= "\"name\"" ws ":" ws str
type_kv     ::= "\"type\"" ws ":" ws ("\"invariant\"" | "\"metamorphic\"" | "\"oracle\"")
desc_kv     ::= "\"description\"" ws ":" ws str
domain_kv   ::= "\"domain\"" ws ":" ws str
fallback_kv ::= "\"fallback\"" ws ":" ws str
str         ::= "\"" chars "\""
chars       ::= { ANY - ["\"\\] } | ("\\" ["\"\\bfnrt/])
ws          ::= { " " | "\n" | "\r" | "\t" }
"""


class PropertyAgent:
    """Generates property-based tests before code modifications."""
    
    def __init__(self):
        self.llm = get_llm_client()
        self.retriever = RAGRetriever()
    
    def prop_spec(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate property specifications from issue and code context."""
        logger = logging.getLogger(__name__)
        logger.debug(f"prop_spec called with state keys: {list(state.keys())}")
        
        current_lot = state.get("current_lot")
        if not current_lot:
            logger.warning("No current_lot found in state")
            return {"invariants": [], "metamorphisms": [], "oracles": []}
            
        rule_key = current_lot.get("ruleKey", "")
        issues = current_lot.get("issues", [])
        logger.debug(f"Processing rule {rule_key} with {len(issues)} issues")
        
        if not issues:
            return {"invariants": [], "metamorphisms": [], "oracles": []}
        
        # Get RAG context for the rule
        rag_ctx = self.retriever.retrieve(f"sonar rule {rule_key} property testing")
        
        # Build prompt for property specification
        repo_path = state.get("repo_path", ".")
        prompt = self._build_prop_spec_prompt(rule_key, issues, rag_ctx, repo_path)
        
        try:
            logger.debug(f"Sending prompt to LLM: {prompt[:200]}...")
            
            # Try with JSON constraints
            response = self.llm.generate(
                prompt,
                max_tokens=512,
                temperature=0,
                stop=["<<END_JSON>>"],
                seed=42,
                # Uncomment ONE of these if your server supports:
                # response_format={"type": "json_object"},  # Change format to {"properties":[...]}
                # grammar={"type": "gbnf", "value": PROP_GBNF},
            )
            logger.debug(f"LLM response: {response[:500]}...")
            
            # Robust JSON parsing with sentinels, arrays and loose objects
            text = response.strip()
            
            # 1) Prefer sentinels
            sent = re.search(r"<<JSON>>(.*)$", text, re.DOTALL)
            if sent:
                text = sent.group(1).strip()
            # Remove eventual fence final before END
            text = text.replace("```json", "```")
            
            def _extract_blocks(s: str):
                blocks = []
                # Code fences
                for m in re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```", s, re.IGNORECASE):
                    blocks.append(m.group(1).strip())
                # Arrays inline
                for m in re.finditer(r"\[\s*{[\s\S]*?}\s*\]", s):
                    blocks.append(m.group(0))
                # Objects inline
                for m in re.finditer(r"{\s*\"name\"\s*:[\s\S]*?}", s):
                    blocks.append(m.group(0))
                return blocks or [s]
            
            def _coerce_list(js: str):
                try:
                    val = json.loads(js)
                    return val if isinstance(val, list) else [val]
                except Exception:
                    return None
            
            items = []
            for cand in _extract_blocks(text):
                val = _coerce_list(cand)
                if val:
                    items.extend(val)
            
            # Deduplicate by name to avoid duplicates
            seen_names = set()
            props = []
            for p in items:
                if (isinstance(p, dict) and 
                    {"name", "type", "description", "domain", "fallback"} <= set(p.keys()) and
                    p["name"] not in seen_names):
                    props.append(p)
                    seen_names.add(p["name"])
            
            # Retry short if nothing valid
            if not props:
                logger.warning("No valid properties found, retrying...")
                retry = self.llm.generate(
                    "Return ONLY a JSON array of property objects with keys [name,type,description,domain,fallback].",
                    max_tokens=256,
                    temperature=0,
                    stop=["\n\n"]
                ).strip()
                retry_items = []
                for cand in _extract_blocks(retry):
                    val = _coerce_list(cand)
                    if val:
                        retry_items.extend(val)
                
                # Deduplicate retry items too
                for p in retry_items:
                    if (isinstance(p, dict) and 
                        {"name", "type", "description", "domain", "fallback"} <= set(p.keys()) and
                        p["name"] not in seen_names):
                        props.append(p)
                        seen_names.add(p["name"])
            
            if not props:
                raise ValueError("No valid property objects found in LLM response")
            
            properties_list = props
            logger.debug(f"Successfully parsed {len(properties_list)} properties")
            
            # Convert array format to expected state format
            spec = {
                "invariants": [p["name"] for p in properties_list if p.get("type") == "invariant"],
                "metamorphisms": [p["name"] for p in properties_list if p.get("type") == "metamorphic"],
                "oracles": [p["name"] for p in properties_list if p.get("type") == "oracle"],
                "properties_detail": properties_list
            }
            logger.debug(f"Converted to spec format: {spec}")
            return spec
                
        except Exception as e:
            logger.error(f"Property spec generation failed: {e}")
            logger.error(f"Raw response was: {response if 'response' in locals() else 'No response'}")
            # Fallback: basic sanity properties
            fallback = {
                "invariants": ["no_exception_on_valid_input"],
                "metamorphisms": [],
                "oracles": ["type_preservation"],
                "properties_detail": [
                    {
                        "name": "no_exception_on_valid_input",
                        "type": "invariant",
                        "description": "Function should not raise exceptions on valid inputs",
                        "domain": "valid inputs according to function signature",
                        "fallback": "Skip test if input validation fails"
                    },
                    {
                        "name": "type_preservation",
                        "type": "oracle",
                        "description": "Function should maintain expected return types",
                        "domain": "all valid inputs",
                        "fallback": "Allow any type if signature unclear"
                    }
                ]
            }
            logger.debug(f"Using fallback spec: {fallback}")
            return fallback
    
    def prop_gen(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Hypothesis test files from specifications."""
        prop_spec = state.get("prop_spec", {})
        current_lot = state.get("current_lot", {})
        
        if not prop_spec or not current_lot:
            return {"files_generated": 0, "test_files": []}
        
        if not current_lot or not current_lot.get("issues"):
            return {"test_files": [], "generated": 0}
        
        # Get first issue to determine file path
        first_issue = current_lot["issues"][0]
        component = first_issue.get("component", "")
        
        if not component.endswith(".py"):
            return {"test_files": [], "generated": 0}
        
        # Resolve file path
        repo_path = state.get("repo_path", ".")
        file_path = self._resolve_component_path(repo_path, component)
        
        if not os.path.exists(file_path):
            logging.getLogger(__name__).warning(f"File not found: {file_path}")
            return {"test_files": [], "generated": 0}
        
        # Extract targets from file
        targets = self._extract_targets(file_path)
        
        test_files = []
        for target in targets[:3]:  # Limit to 3 functions
            rel_path = os.path.relpath(file_path, repo_path)
            content = self._render_property_test(rel_path, target["name"])
            
            out_path = Path(repo_path) / "tests_prop" / f"test_{Path(file_path).stem}_{target['name']}_props.py"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(content, encoding="utf-8")
            
            test_files.append(str(out_path))
        
        return {
            "test_files": test_files,
            "generated": len(test_files)
        }
    
    def _build_prop_spec_prompt(self, rule_key: str, issues: List[Dict], rag_ctx: Dict, repo_path: str = ".") -> str:
        """Build prompt for property specification."""
        contexts = "\n".join(rag_ctx.get("contexts", [])[:3])
        
        # Extract code snippet from first issue
        code_snippet = ""
        issue_description = ""
        issue_type = "CODE_SMELL"
        if issues:
            issue = issues[0]
            issue_type = issue.get('type', 'CODE_SMELL')
            issue_description = f"{issue.get('message', '')} (Rule: {rule_key}, Type: {issue_type})"
            # Try to get code context if available
            component_path = issue.get('component_path') or issue.get('component', '')
            if component_path:
                try:
                    # Use resolved component path if available
                    if not component_path.startswith('/'):
                        file_path = self._resolve_component_path(repo_path, component_path)
                    else:
                        file_path = component_path
                    
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                        line_num = issue.get('line', 1) - 1
                        start = max(0, line_num - 5)
                        end = min(len(lines), line_num + 5)
                        code_snippet = ''.join(lines[start:end])
                except:
                    code_snippet = "Code not available"
        
        few_shots = '''[
  {
    "name": "preserve_length",
    "type": "invariant",
    "description": "The function must return a list of the same length as the input.",
    "domain": "non-empty list of integers",
    "fallback": "Return an empty list if input is invalid"
  },
  {
    "name": "idempotent_operation",
    "type": "metamorphic",
    "description": "Applying the function twice should yield the same result as applying it once.",
    "domain": "valid string inputs",
    "fallback": "Skip test if input is malformed"
  },
  {
    "name": "type_safety",
    "type": "oracle",
    "description": "Function should raise TypeError for invalid input types.",
    "domain": "invalid type inputs (int when string expected)",
    "fallback": "Pass test if no exception raised"
  }
]'''
        
        # Dynamic part (can use f-string)
        header = dedent(f"""You are a software engineer specializing in generating property-based tests using Hypothesis for Python.

## Goal:
Given the code and the issue description (e.g. from SonarQube), identify testable properties that describe the expected behavior of the function. These properties will guide test generation and help detect bugs.

---

## Target Code (partial):

{code_snippet}

---

## Reported Issue:

{issue_description}

---

## Additional Context (Sonar rule doc, fix examples, neighborhood code):

{contexts}

---

## Previous Examples (few-shot property-based tests):

{few_shots}

---

## Instructions:

1. Carefully analyze the function and issue.
2. Issue type is {issue_type}. For BUG/VULNERABILITY, properties are MANDATORY and should be strict. For CODE_SMELL, properties can be more lenient.
3. Identify up to 3 relevant **properties** that:
   - Describe behavior for valid input (✅ invariants)
   - Compare inputs/outputs with transformations (🔁 metamorphic relations)
   - Expect errors on invalid inputs (💥 oracles)
4. For each property, include:
   - `name`: short identifier
   - `type`: one of ["invariant", "metamorphic", "oracle"]
   - `description`: human-readable summary
   - `domain`: expected input domain
   - `fallback`: how to behave if inputs are out of domain

---""")
        
        # Static part with JSON example — NO f-string here!
        json_guide = dedent("""## Response Format (JSON ONLY — no prose, no markdown)
Return ONLY a JSON array **between the markers** below:
<<JSON>>
[
  {
    "name": "preserve_length",
    "type": "invariant",
    "description": "The function must return a list of the same length as the input.",
    "domain": "non-empty list of integers",
    "fallback": "Return an empty list if input is invalid"
  }
]
<<END_JSON>>""")
        
        return header + "\n" + json_guide
    
    def _resolve_component_path(self, repo_root: str, component: str) -> str:
        """Resolve component path from SonarQube format."""
        # "myproj:src/mod.py" -> "src/mod.py"
        rel = component.split(":", 1)[-1]
        p = os.path.normpath(os.path.join(repo_root, rel))
        return p
    
    def _extract_targets(self, py_path: str) -> List[Dict]:
        """Extract target functions from Python file using AST."""
        try:
            src = Path(py_path).read_text(encoding="utf-8")
            tree = ast.parse(src, filename=py_path)
            targets = []
            for n in tree.body:
                if isinstance(n, ast.FunctionDef) and not n.name.startswith("_"):
                    targets.append({
                        "type": "function",
                        "name": n.name,
                        "lineno": n.lineno
                    })
            return targets
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to parse {py_path}: {e}")
            return []
    
    def _render_property_test(self, module_path: str, func_name: str) -> str:
        """Render property test for a specific function."""
        mod_import = module_path.replace("/", ".").removesuffix(".py")
        return f'''# generated property tests
from hypothesis import given, strategies as st, settings, HealthCheck, assume
from {mod_import} import {func_name}

@settings(deadline=None, max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(st.integers(min_value=-10_000, max_value=10_000), max_size=200))
def test_{func_name}_no_exception_on_valid_input(xs):
    """Property: Function should not raise exceptions on valid input."""
    {func_name}(xs)  # deve não lançar

@settings(deadline=None, max_examples=200)
@given(st.lists(st.integers(min_value=-1_000, max_value=1_000), max_size=100))
def test_{func_name}_idempotent_on_repeat(xs):
    """Property: Function should be idempotent when applied twice."""
    try:
        out1 = {func_name}(xs)
        out2 = {func_name}(out1)
        assert out2 == out1
    except (TypeError, ValueError):
        # Skip if function doesn't support this property
        assume(False)
'''