"""Validation helpers for the correction Instruct-Validate-Repair loop.

Two concerns, kept separate:
1. Validation (for the correction IVR loop): pytest only — the single source of
   truth for whether generated code solves a problem.
2. Tagging (post-hoc classification): flake8-qiskit-migration detects deprecated
   APIs in broken code. Used to enrich prompts, NOT as a validation gate. It is
   imported in-process (unlike qiskit) and ships in the ``science``/``all``
   extras; the import is guarded, so without it tagging degrades to a no-op and
   every row reports ``has_deprecated_apis=false``.

The pytest subprocess mirrors ``scripts/pass_at_k_eval.py``: the generated tests
are ``unittest.TestCase`` classes ending in ``unittest.main()``, and a minority
``import pytest`` (for ``pytest.raises``), so pytest must be *importable* in the
run environment. Run this package under ``uv run --extra all --with pytest`` so
qiskit + all addons + pytest resolve.
"""

from __future__ import annotations

import ast
import logging
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable

try:
    from flake8_qiskit_migration.plugin import Plugin as QiskitMigrationPlugin
except ImportError:
    QiskitMigrationPlugin = None

log = logging.getLogger(__name__)

MAX_ERROR_LENGTH = 2000
PYTEST_TIMEOUT_SECONDS = 60


def extract_code_from_markdown(text: str) -> str:
    """Extract Python code from markdown fenced code blocks.

    Returns the first ```python or ``` block found, or the raw text if none.
    """
    pattern = r"```(?:python|py)?\s*(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    return text.strip()


def find_deprecated_apis(code: str) -> tuple[bool, str]:
    """Detect deprecated Qiskit APIs (for classification, not validation).

    Returns ``(has_deprecated, warning_text)``: ``(True, messages)`` listing the
    QKT violations found, else ``(False, "")``. The polarity matches the
    ``has_deprecated_apis`` / ``deprecated_api_warnings`` row fields it feeds, so
    callers never re-invert it.

    Code that cannot be analysed reports ``(False, "")`` and logs the reason
    rather than being labelled deprecated: inputs here are already-broken model
    attempts, so a SyntaxError is expected and says nothing about deprecation.
    Letting it through as a "deprecation" would put syntax-error text in
    ``deprecated_api_warnings``, which must hold deprecation messages only.
    Also a no-op when flake8-qiskit-migration is not installed — see the module
    docstring for how to enable it.
    """
    if QiskitMigrationPlugin is None:
        return False, ""

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        log.debug("Skipping deprecation tagging, code does not parse: %s", e)
        return False, ""

    try:
        plugin = QiskitMigrationPlugin(tree)
        errors = list(plugin.run())
    except Exception as e:
        log.warning("Skipping deprecation tagging, flake8 plugin failed: %s", e)
        return False, ""

    if not errors:
        return False, ""

    messages = [message for _line, _col, message, _type in errors]
    return True, "\n".join(messages)


_SOLUTION_SRC_MARKER = "# synthetic unittests"
# Matches an *empty* (or whitespace-only) string literal assigned to
# __solution_source__. Triple-quoted forms are tried first so `""""""` is not
# partially consumed as `""` — that used to leave a dangling `""""` behind and
# turn the rewritten test file into a SyntaxError. The trailing lookahead pins
# the match to the end of the statement so no longer literal is ever clipped.
_SOLUTION_SRC_STUB = re.compile(
    r"""__solution_source__[ \t]*=[ \t]*"""
    r"""(?:\"{3}\s*\"{3}|'{3}\s*'{3}|\"[ \t]*\"|'[ \t]*')"""
    r"""(?=[ \t]*(?:$|[#;]))""",
    re.MULTILINE,
)
_SOLUTION_SRC_SELFREAD = (
    '__solution_source__ = __import__("pathlib").Path(__file__)'
    '.read_text().split("# synthetic unittests")[0]'
)


def _populate_solution_source(unittest_code: str) -> str:
    """Fix the ``__solution_source__ = ""`` stub some synthetic tests ship with.

    A subset of datasets (e.g. the gemma-filtered set) hardcode
    ``__solution_source__ = ""`` and expect the run harness to fill it in. Their
    ``test_required_names/modules/substrings/call_assigned_to`` checks
    ``ast.parse(__solution_source__)`` — with an empty string that parses to an
    empty AST, so EVERY structural check fails even on correct code, tanking the
    solve rate to ~0. We rewrite the stub to self-read the solution portion of
    the assembled file (imports+code, everything before the marker) — the same
    mechanism other synthetic datasets already use. No-op when the stub is
    absent, so self-reading datasets are unaffected.

    Empty single-, double- and triple-quoted stubs are recognised, including
    whitespace-only ones (``= " "``); a stub holding real code is left alone.
    """
    if _SOLUTION_SRC_MARKER not in unittest_code:
        return unittest_code
    return _SOLUTION_SRC_STUB.sub(_SOLUTION_SRC_SELFREAD, unittest_code, count=1)


def validate_pytest(
    code: str,
    imports: str,
    unittest_code: str,
    test_python: str | None = None,
    timeout: int = PYTEST_TIMEOUT_SECONDS,
) -> tuple[bool, str]:
    """Run imports + code + unittest via pytest in a subprocess.

    Combines the three parts into a single file (matching test.sh behavior) and
    runs pytest. Returns (True, "") on pass or (False, truncated_output).

    ``test_python`` selects the interpreter used to run the tests (defaults to
    ``sys.executable``), matching pass_at_k_eval.py's ``--test-python`` — pass a
    distinct interpreter when generation and testing use different environments.
    """
    unittest_code = _populate_solution_source(unittest_code)
    combined = f"{imports}\n\n{code}\n\n{unittest_code}\n"

    py = test_python or sys.executable
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".py", prefix="qiskit_test_")
        os.write(fd, combined.encode())
        os.close(fd)

        result = subprocess.run(
            [py, "-m", "pytest", tmp_path, "-x", "-q", "--tb=short", "--no-header"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode == 0:
            return True, ""

        output = (result.stdout + "\n" + result.stderr).strip()
        if len(output) > MAX_ERROR_LENGTH:
            output = output[:MAX_ERROR_LENGTH] + "\n... (truncated)"
        return False, output

    except subprocess.TimeoutExpired:
        return False, f"Pytest timed out after {timeout}s"
    except Exception as e:
        return False, f"Pytest execution error: {e}"
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def make_composite_validator(
    imports: str,
    unittest_code: str,
    test_python: str | None = None,
) -> Callable[[str], tuple[bool, str]]:
    """Return a closure that validates model output via pytest only.

    flake8 is intentionally NOT part of the validation chain: deprecated APIs
    are removed in Qiskit 2.x, so they cause ImportError/AttributeError at
    runtime which pytest already catches. Running flake8 as a gate would
    short-circuit and hide whether the code also has logic bugs underneath.
    """

    def validator(text: str) -> tuple[bool, str]:
        code = extract_code_from_markdown(text)
        if not code.strip():
            return False, "Empty code output"

        return validate_pytest(code, imports, unittest_code, test_python=test_python)

    return validator
