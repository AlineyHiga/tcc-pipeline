from __future__ import annotations

import importlib
import os
import stat
import string
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import patch

import pytest
from hypothesis import assume, given, strategies as st


TEST_PIPELINE_ROOT = Path(__file__).resolve().parents[3] / "src" / "test-pipeline"
if TEST_PIPELINE_ROOT.exists():
    root_str = str(TEST_PIPELINE_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    import src  # noqa: E402

    candidate = str(TEST_PIPELINE_ROOT / "src")
    if candidate not in src.__path__:
        src.__path__.append(candidate)

    utils = importlib.import_module("src.utils")  # noqa: E402
else:  # pragma: no cover - defensive guard for missing dependency tree
    raise ImportError("Unable to locate test-pipeline sources for src.utils")


@given(
    content_a=st.text(min_size=1, max_size=64),
    content_b=st.text(min_size=1, max_size=64),
)
def test_create_temp_file_generates_unique_names(content_a: str, content_b: str) -> None:
    assume(content_a != content_b)
    path_a = utils.create_temp_file(content_a)
    path_b = utils.create_temp_file(content_b)
    try:
        assert path_a != path_b, "Temporary file names must be unique per invocation"
    finally:
        for path in {path_a, path_b}:
            if os.path.exists(path):
                os.remove(path)


@given(content=st.text())
def test_create_temp_file_is_not_world_writable(content: str) -> None:
    path = utils.create_temp_file(content)
    try:
        mode = stat.S_IMODE(os.stat(path).st_mode)
        world_writable = mode & 0o002 or mode & 0o020
        assert not world_writable, "Temporary files should not grant write access to group or world"
    finally:
        if os.path.exists(path):
            os.remove(path)


@given(
    key=st.text(alphabet=st.characters(min_codepoint=97, max_codepoint=122), min_size=1, max_size=12),
    value=st.integers(min_value=-1_000_000, max_value=-1),
)
def test_process_file_parses_negative_integers(key: str, value: int) -> None:
    with NamedTemporaryFile("w", delete=False) as handle:
        handle.write(f"{key} = {value}\n")
        path = handle.name
    try:
        parsed = utils.process_file(path)
        assert (key, value) in parsed, "Configuration loader should return negative integer values as ints"
    finally:
        os.remove(path)


dangerous_alphabet = string.ascii_letters + string.digits + " _-;|&><'\"$`\n\t"
dangerous_chars = {";", "|", "&", ">", "<", "`", "$"}


@given(user_input=st.text(alphabet=dangerous_alphabet, min_size=1, max_size=40))
def test_run_system_command_sanitizes_shell_metacharacters(user_input: str) -> None:
    assume(any(char in user_input for char in dangerous_chars))
    with patch("src.utils.os.system") as mock_system:
        utils.run_system_command(user_input)
    executed_command = mock_system.call_args[0][0]
    assert not any(
        char in executed_command for char in dangerous_chars
    ), "Shell metacharacters must be sanitized before execution"


@given(st.lists(st.integers(), max_size=20))
def test_get_first_element_handles_empty_lists_safely(values: list[int]) -> None:
    if values:
        assert utils.get_first_element(values) == values[0]
    else:
        with pytest.raises(IndexError, match="list index out of range"):
            utils.get_first_element(values)
