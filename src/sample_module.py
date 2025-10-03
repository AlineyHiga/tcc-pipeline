"""Sample module used by the AutoFix pipeline."""
from __future__ import annotations

from typing import Iterable


def safe_divide(a: float, b: float) -> float:
    """Return a / b protecting against division by zero."""
    if b == 0:
        return float("inf")
    return a / b


def sum_non_negative(values: Iterable[float]) -> float:
    """Sum values but treat negatives as zero (idempotent example)."""
    total = 0.0
    for value in values:
        total += value if value >= 0 else 0.0
    return total
