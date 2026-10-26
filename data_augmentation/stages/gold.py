"""Gold stage: extract gold correction entries from the correction hub.

Gold = failing attempts from problems that were eventually solved. These are the
highest-value correction training data for GRPO: the model has demonstrably
fixable broken code, and the original unit tests provide the reward signal.

Output is JSONL compatible with generate_harbor_tasks.py (canonical keys first).
This module is pure library code; the CLI lives in ``data_augmentation.cli``.
"""

from __future__ import annotations

import json
import os
from collections import Counter

from ..core.rows import canonicalize_row
from ..core.text import strip_assertion_internals


def sanitize_error_in_prompt(prompt: str) -> str:
    """Strip pytest assertion internals from the error section of a built prompt.

    Applies ``core.text.strip_assertion_internals`` to everything after the
    ``Errors encountered:`` marker only, leaving the task text and the broken code
    block untouched (an ``assert`` inside the broken code is part of the problem).
    Removes lines like:
      >       assert SU2.entanglement == "full", "Wrong entanglement"
          assert SU2.entanglement == "full", "Wrong entanglement"
      E   assert 'linear' == 'full'
    Keeps:
      E   AssertionError: Wrong entanglement
    """
    marker = "Errors encountered:\n"
    idx = prompt.find(marker)
    if idx == -1:
        return prompt

    split_at = idx + len(marker)
    return prompt[:split_at] + strip_assertion_internals(prompt[split_at:])


def inject_imports_into_code_block(prompt: str, imports: str) -> str:
    """Prepend imports into the broken code block so it looks model-generated.

    The original model saw imports in the prompt and omitted them from its
    output. We prepend them into the code block so the correction prompt
    presents complete code — the model learns to fix the real bug, not
    missing imports.
    """
    imports = imports.strip()
    if not imports:
        return prompt

    marker = "```python\n"
    idx = prompt.find(marker)
    if idx == -1:
        return prompt

    insert_at = idx + len(marker)
    return prompt[:insert_at] + imports + "\n\n" + prompt[insert_at:]


def extract_gold(input_path: str, output_path: str) -> int:
    """Filter correction JSONL to gold entries and write them. Returns the count."""
    entries = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    gold = [
        e
        for e in entries
        if e.get("solved")
        and e.get("attempt_number", 0) > 0
        and e.get("attempt_number", 0) < e.get("solved_at_turn", 0)
    ]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for entry in gold:
            row = canonicalize_row(entry)
            row["prompt"] = inject_imports_into_code_block(row["prompt"], row["imports"])
            row["prompt"] = sanitize_error_in_prompt(row["prompt"])
            f.write(json.dumps(row) + "\n")

    template_counts = Counter(e["template"] for e in gold)
    error_counts = Counter(e["error_type"] for e in gold)
    attempt_counts = Counter(e["attempt_number"] for e in gold)
    source_problems = len({e["source_problem_idx"] for e in gold})

    print(f"Input:  {len(entries)} total entries from {input_path}")
    print(f"Output: {len(gold)} gold entries to {output_path}")
    print(f"Source problems: {source_problems}")
    print()
    print("By attempt number (which turn failed):")
    for k in sorted(attempt_counts):
        print(f"  attempt {k}: {attempt_counts[k]}")
    print()
    print("By error type:")
    for err, count in error_counts.most_common():
        print(f"  {err}: {count}")
    print()
    print("By template (top 15):")
    for tmpl, count in template_counts.most_common(15):
        print(f"  {tmpl}: {count}")

    return len(gold)
