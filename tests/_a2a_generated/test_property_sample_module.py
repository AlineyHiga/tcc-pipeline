from __future__ import annotations

import math

import pytest
from hypothesis import assume, given, strategies as st

from src.sample_module import safe_divide, sum_non_negative


def bug_domain_safe_divide(inputs: dict) -> bool:
    return inputs["denominator"] == 0


finite_floats = st.floats(
    allow_nan=False,
    allow_infinity=False,
    width=32,
    min_value=-1e6,
    max_value=1e6,
)


@given(numerator=finite_floats, denominator=finite_floats)
def test_safe_divide_matches_standard_division(numerator: float, denominator: float) -> None:
    inputs = {"numerator": numerator, "denominator": denominator}
    assume(not bug_domain_safe_divide(inputs))
    assume(denominator != 0)
    result = safe_divide(numerator, denominator)
    assert result == pytest.approx(numerator / denominator)


@given(numerator=finite_floats)
def test_safe_divide_zero_denominator_returns_positive_infinity(numerator: float) -> None:
    result = safe_divide(numerator, 0.0)
    assert math.isinf(result)
    assert result > 0


@given(
    numerator=finite_floats,
    denominator=finite_floats.filter(lambda value: value != 0),
    scale=finite_floats.filter(lambda value: value not in (0, 1, -1)),
)
def test_safe_divide_scale_invariance(numerator: float, denominator: float, scale: float) -> None:
    assume(not bug_domain_safe_divide({"numerator": numerator, "denominator": denominator}))
    scaled_result = safe_divide(numerator * scale, denominator * scale)
    original_result = safe_divide(numerator, denominator)
    assert scaled_result == pytest.approx(original_result)


list_of_floats = st.lists(finite_floats, min_size=0, max_size=30)


@given(values=list_of_floats)
def test_sum_non_negative_matches_manual_filter(values: list[float]) -> None:
    result = sum_non_negative(values)
    expected = sum(value for value in values if value >= 0)
    assert result == pytest.approx(expected)
    assert result >= 0


@given(values=list_of_floats, prefix_length=st.integers(min_value=0, max_value=30))
def test_sum_non_negative_additivity(values: list[float], prefix_length: int) -> None:
    assume(prefix_length <= len(values))
    left = values[:prefix_length]
    right = values[prefix_length:]
    combined = sum_non_negative(values)
    assert combined == pytest.approx(sum_non_negative(left) + sum_non_negative(right))


@given(values=st.lists(finite_floats.map(lambda value: -abs(value)), min_size=1, max_size=30))
def test_sum_non_negative_all_negative_returns_zero(values: list[float]) -> None:
    assert sum_non_negative(values) == 0
