"""Semantic codemod application utilities."""
from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import libcst as cst  # type: ignore[assignment]

# Compat helper for older LibCST versions without AsyncFunctionDef
AsyncFunctionDefType = getattr(cst, "AsyncFunctionDef", None)
if AsyncFunctionDefType:
    FUNCTION_NODE_TYPES: Tuple[type, ...] = (cst.FunctionDef, AsyncFunctionDefType)
else:
    FUNCTION_NODE_TYPES = (cst.FunctionDef,)


@dataclass
class CodemodOperationResult:
    """Holds metadata about an applied codemod operation."""

    operation: Dict[str, Any]
    applied: bool
    detail: str = ""


class ReplaceFunctionTransformer(cst.CSTTransformer):
    """Replace a target function (sync or async) with a new definition."""

    def __init__(self, func_name: str, new_node: cst.CSTNode):
        self.func_name = func_name
        self.new_node = new_node
        self.applied = False

    def _matches(self, name: str) -> bool:
        return name == self.func_name

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.CSTNode:
        if self._matches(original_node.name.value):
            self.applied = True
            return self.new_node
        return updated_node

    if AsyncFunctionDefType:

        def leave_AsyncFunctionDef(  # type: ignore[override]
            self, original_node: cst.CSTNode, updated_node: cst.CSTNode
        ) -> cst.CSTNode:
            if isinstance(original_node, AsyncFunctionDefType) and self._matches(
                original_node.name.value
            ):
                self.applied = True
                return self.new_node
            return updated_node


def _parse_function_node(code: str) -> cst.CSTNode:
    """Parse function code into a LibCST node."""
    normalized = textwrap.dedent(code).strip()
    if not normalized:
        raise ValueError("function code is empty")

    # Ensure trailing newline for LibCST parser.
    module = cst.parse_module(normalized + ("\n" if not normalized.endswith("\n") else ""))
    if len(module.body) != 1:
        raise ValueError("function code must contain exactly one function definition")

    func_node = module.body[0]
    if not isinstance(func_node, FUNCTION_NODE_TYPES):
        raise ValueError("provided code is not a function definition")

    return func_node


def _build_function_from_parts(
    signature: str, body: Iterable[str], decorators: Optional[Iterable[str]] = None
) -> str:
    """Reconstruct a function definition from signature, decorators and body lines."""
    decorators = list(decorators or [])
    body_lines = list(body)
    if not signature.startswith("def ") and not signature.startswith("async def "):
        raise ValueError("function signature must start with def/async def")
    if not body_lines:
        body_lines = ["pass"]

    # Ensure body is indented.
    body_block = "\n".join(body_lines)
    body_block = textwrap.dedent(body_block)
    if not body_block.endswith("\n"):
        body_block += "\n"
    indented_body = textwrap.indent(body_block, "    ")

    pieces = []
    pieces.extend(decorators)
    pieces.append(signature.rstrip())
    pieces.append(indented_body.rstrip("\n"))
    return "\n".join(pieces) + "\n"


def _apply_operations_to_module(
    module: cst.Module, operations: List[Dict[str, Any]]
) -> Tuple[cst.Module, List[CodemodOperationResult]]:
    """Apply a sequence of operations to a LibCST module."""
    _require_libcst()
    results: List[CodemodOperationResult] = []
    updated_module = module

    for op in operations:
        op_type = op.get("type")
        if op_type == "replace_function":
            func_name = op.get("name")
            func_code = op.get("code")
            if not func_name:
                raise ValueError("replace_function operation missing 'name'")

            if func_code:
                new_node = _parse_function_node(func_code)
            else:
                signature = op.get("signature")
                body = op.get("body")
                decorators = op.get("decorators", [])
                if not signature or body is None:
                    raise ValueError(
                        "replace_function operation requires either 'code' or both 'signature' and 'body'"
                    )

                if not isinstance(signature, str):
                    raise ValueError("function signature must be a string")
                if isinstance(body, str):
                    body_lines = body.splitlines()
                else:
                    body_lines = list(body)
                signature_text = signature.rstrip()
                signature_lines = signature_text.splitlines()
                signature_line = signature_lines[0].strip()
                extra_header_lines = signature_lines[1:]
                if extra_header_lines:
                    extra_block = textwrap.dedent("\n".join(extra_header_lines)).splitlines()
                    body_lines = extra_block + body_lines
                rebuilt_code = _build_function_from_parts(signature_line, body_lines, decorators)
                new_node = _parse_function_node(rebuilt_code)

            transformer = ReplaceFunctionTransformer(func_name, new_node)
            updated_module = updated_module.visit(transformer)
            results.append(
                CodemodOperationResult(
                    operation=op,
                    applied=transformer.applied,
                    detail="function replaced" if transformer.applied else "function not found",
                )
            )
            if not transformer.applied:
                raise ValueError(f"function '{func_name}' not found for replacement")
        else:
            raise ValueError(f"unsupported operation type: {op_type}")

    return updated_module, results


def apply_spec(
    spec: Dict[str, Any],
    repo_root: str = ".",
    original_sources: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Apply a semantic codemod specification and return updated file contents."""
    _require_libcst()
    if not isinstance(spec, dict):
        raise ValueError("spec must be a dictionary")

    files = spec.get("files", [])
    if not isinstance(files, list):
        raise ValueError("spec['files'] must be a list")

    repo_root_path = Path(repo_root).resolve()
    updated_contents: Dict[str, str] = {}

    for file_entry in files:
        if not isinstance(file_entry, dict):
            raise ValueError("file entry must be a dictionary")

        rel_path = file_entry.get("path")
        if not rel_path:
            raise ValueError("file entry missing 'path'")

        operations = file_entry.get("operations", [])
        if not isinstance(operations, list) or not operations:
            raise ValueError("file entry must include a non-empty 'operations' list")

        original_text: Optional[str] = None
        if original_sources and rel_path in original_sources:
            original_text = original_sources[rel_path]
        else:
            full_path = repo_root_path / rel_path
            if not full_path.exists():
                raise FileNotFoundError(f"file '{rel_path}' not found")
            original_text = full_path.read_text(encoding="utf-8")

        module = cst.parse_module(original_text)
        updated_module, _ = _apply_operations_to_module(module, operations)
        updated_contents[rel_path] = updated_module.code

    return updated_contents


__all__ = ["apply_spec", "CodemodOperationResult"]


def _require_libcst() -> None:
    """Ensure LibCST is available."""
    if cst is None:
        raise ImportError("libcst is required for semantic codemod operations")
