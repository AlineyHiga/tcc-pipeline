"""Compatibility wrapper that re-exports the public helpers from ``../src``."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType


def _load_reference_module() -> ModuleType:
    """Load the canonical implementation defined in the repository root."""
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "src" / "sample_module.py"
    spec = spec_from_file_location("autofix_reference.sample_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to locate reference sample_module at {module_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_reference = _load_reference_module()

Calculator = _reference.Calculator
divide_numbers = _reference.divide_numbers
safe_divide = _reference.safe_divide
sum_non_negative = _reference.sum_non_negative

__all__ = ["Calculator", "divide_numbers", "safe_divide", "sum_non_negative"]
