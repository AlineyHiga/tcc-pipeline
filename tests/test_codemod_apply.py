import textwrap

import pytest

pytest.importorskip("libcst")

from app.codemod_apply import apply_spec


def test_replace_function_with_full_code():
    original = textwrap.dedent(
        """
        def add(a, b):
            return a - b
        """
    ).strip() + "\n"

    spec = {
        "files": [
            {
                "path": "module.py",
                "operations": [
                    {
                        "type": "replace_function",
                        "name": "add",
                        "code": "def add(a, b):\n    return a + b\n",
                    }
                ],
            }
        ]
    }

    result = apply_spec(spec, original_sources={"module.py": original})
    assert "module.py" in result
    assert "return a + b" in result["module.py"]
    assert "return a - b" not in result["module.py"]


def test_replace_function_with_parts():
    original = textwrap.dedent(
        """
        @decorator
        async def fetch(url):
            return await http_get(url)
        """
    ).strip() + "\n"

    spec = {
        "files": [
            {
                "path": "client.py",
                "operations": [
                    {
                        "type": "replace_function",
                        "name": "fetch",
                        "signature": "async def fetch(url):",
                        "decorators": ["@decorator"],
                        "body": [
                            "if not url:",
                            "    raise ValueError('url is required')",
                            "return await http_get(url)",
                        ],
                    }
                ],
            }
        ]
    }

    result = apply_spec(spec, original_sources={"client.py": original})
    assert "raise ValueError" in result["client.py"]
    assert result["client.py"].startswith("@decorator")
    assert "async def fetch(url):" in result["client.py"]


def test_invalid_operation_type_raises():
    original = "def foo():\n    pass\n"
    spec = {
        "files": [
            {
                "path": "module.py",
                "operations": [{"type": "rename"}],
            }
        ]
    }

    with pytest.raises(ValueError):
        apply_spec(spec, original_sources={"module.py": original})


def test_signature_with_embedded_body_lines():
    original = textwrap.dedent(
        """
        def validate_user_data(data):
            return False
        """
    ).strip() + "\n"

    spec = {
        "files": [
            {
                "path": "db.py",
                "operations": [
                    {
                        "type": "replace_function",
                        "name": "validate_user_data",
                        "signature": textwrap.dedent(
                            """\
                            def validate_user_data(data):
                                if not isinstance(data, dict):
                                    return False
                            """
                        ),
                        "body": [
                            "if 'username' not in data:",
                            "    return False",
                            "return True",
                        ],
                    }
                ],
            }
        ]
    }

    result = apply_spec(spec, original_sources={"db.py": original})
    replacement = result["db.py"]
    assert "def validate_user_data(data):" in replacement
    assert "if not isinstance(data, dict):" in replacement
    assert "return True" in replacement
