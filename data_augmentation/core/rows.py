"""Output-row shaping shared by every stage.

All three stages write JSONL whose leading keys are the canonical
``prompt, imports, unittest, template`` (what generate_harbor_tasks.py reads),
followed by that stage's provenance fields. Keeping the reorder in one place
guarantees the three outputs order their keys identically.
"""

from __future__ import annotations

from typing import Any

from .. import CANONICAL_KEYS


def canonicalize_row(row: dict[str, Any]) -> dict[str, Any]:
    """Return ``row`` with the canonical keys first, then everything else.

    Insertion order is what ``json.dumps`` writes, so this is purely about the
    on-disk key order. Raises KeyError if a canonical key is missing — every
    stage's dataclass declares all four, so an absent one is a bug worth failing on.
    """
    out = {k: row[k] for k in CANONICAL_KEYS}
    out.update({k: v for k, v in row.items() if k not in CANONICAL_KEYS})
    return out
