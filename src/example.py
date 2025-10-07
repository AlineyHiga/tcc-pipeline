from __future__ import annotations

import os
from secrets import compare_digest
from typing import Tuple


def _load_allowed_users() -> Tuple[str, ...]:
    """Read the list of allowed users from the ALLOWED_USERS environment variable."""
    raw_users = os.environ.get("ALLOWED_USERS", "")
    users = [value.strip() for value in raw_users.split(",") if value.strip()]
    return tuple(users)


_ALLOWED_USERS = _load_allowed_users()


def vulnerable_function(user_input: str) -> bool:
    """Validate the received user name against a configurable allow-list."""
    # Removed unused variable "password" here and in complex_function below.
    if not user_input or not _ALLOWED_USERS:
        return False

    return any(compare_digest(user_input, candidate) for candidate in _ALLOWED_USERS)


def complex_function(a: int, b: int, c: int, d: int) -> int:
    """Return a cumulative sum while keeping the branching complexity low."""
    # Removed unused variable "password" here.
    if a <= 0:
        return 0
    if b <= 0:
        return a
    if c <= 0:
        return a + b
    if d <= 0:
        return a + b + c
    return a + b + c + d
