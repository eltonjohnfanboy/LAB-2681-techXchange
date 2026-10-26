"""Prompt/error text scrubbing shared by the correction and gold stages.

The broken code in a correction prompt is accompanied by the pytest failure that
produced it. Raw pytest output quotes the assertion source, which hands the model
the test's expected values — it could satisfy the check without writing correct
Qiskit. One filter, used by both the stage that builds the prompt
(``stages.correction``) and the stage that re-scrubs an already-built prompt
(``stages.gold``), so the two can never drift apart.
"""

from __future__ import annotations


def strip_assertion_internals(text: str) -> str:
    """Drop pytest assertion lines that leak test logic, line by line.

    Removes:
      - ``>       assert ...`` (the quoted assertion source)
      - indented ``assert ... ==``/``isinstance`` lines (``--tb=short`` format)
      - ``E   assert ...`` (pytest's rewritten comparison values)
      - ``FAILED``/``ERROR`` summary lines ending in ``- assert ...``
    Keeps:
      - ``E   AssertionError: Wrong entanglement`` (the custom message — useful)
      - every non-assertion error (TypeError, ImportError, … — always useful)
      - the traceback structure (helps locate the error)
    """
    filtered = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith(">") and "assert " in stripped:
            continue
        if stripped.startswith("assert ") and ("==" in stripped or "isinstance" in stripped):
            continue
        if stripped.startswith("E") and stripped.lstrip("E").strip().startswith("assert "):
            continue
        if ("FAILED" in stripped or "ERROR" in stripped) and " - assert " in stripped:
            continue
        filtered.append(line)
    return "\n".join(filtered)
