"""Compatibility wrapper that re-exports the public helpers from `src`."""

from src.sample_module import (  # type: ignore[F401]
    Calculator,
    divide_numbers,
    safe_divide,
    sum_non_negative,
)

__all__ = [
    "Calculator",
    "divide_numbers",
    "safe_divide",
    "sum_non_negative",
]
