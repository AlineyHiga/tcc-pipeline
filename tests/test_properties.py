from __future__ import annotations

from math import isfinite

import hypothesis.strategies as st
from hypothesis import given

from src.sample_module import safe_divide, sum_non_negative


@given(
    st.floats(allow_infinity=False, allow_nan=False),
    st.floats(allow_infinity=False, allow_nan=False),
)
def test_safe_divide_behaviour(a: float, b: float) -> None:
    result = safe_divide(a, b)
    if b == 0:
        assert result == float("inf")
    else:
        assert result == a / b


@given(
    st.lists(
        st.floats(
            min_value=-1000,
            max_value=1000,
            allow_nan=False,
            allow_infinity=False,
        )
    )
)
def test_sum_non_negative_is_finite(values: list[float]) -> None:
    total = sum_non_negative(values)
    assert total >= 0
    assert isfinite(total)
