"""Correction stage: generate "fix this broken code" data via mellea's IVR loop.

Runs mellea's Instruct-Validate-Repair loop over existing Qiskit problems —
the exact engine that produced the v5/v6 datasets. mellea owns the loop:
``session.instruct(...)`` with ``RepairTemplateStrategy`` calls the model, runs the
pytest validator, and on failure re-solves. Because the session uses
``SimpleContext`` (no conversation history), each repair turn re-renders the
ORIGINAL instruction with only the raw validator failure appended — the model never
sees its own broken code and re-solves fresh (this is what distinguishes the loop
from a naive "here is your broken code, fix it" retry, and what the recorded data
distribution depends on).

Intermediate failing attempts become "fix this broken code" problems that reuse the
original unit tests as the reward signal; the ``gold`` stage later keeps the ones
from eventually-solved problems.

Output JSONL is compatible with generate_harbor_tasks.py (canonical keys first).
This module is pure library code; the CLI lives in ``data_augmentation.cli``.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from ..core.rows import canonicalize_row
from ..core.text import strip_assertion_internals
from ..core.validation import (
    extract_code_from_markdown,
    find_deprecated_apis,
    make_composite_validator,
)
from ..models import CorrectionEntry, GenConfig

if TYPE_CHECKING:
    from mellea import MelleaSession

log = logging.getLogger(__name__)

MAX_ERROR_IN_PROMPT = 1500


def _iter_jsonl(path: str) -> Iterator[dict]:
    """Yield parsed rows one line at a time (never holds the whole file)."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_problems(input_path: str, allowlist_path: str | None = None) -> list[dict]:
    # Streamed rather than readlines() so only the parsed dicts are resident —
    # the raw text of a large source file is never held alongside them.
    problems = list(_iter_jsonl(input_path))

    if allowlist_path:
        allowed = set()
        with open(allowlist_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    allowed.add(line)
        original_count = len(problems)
        problems = [p for p in problems if p.get("template", "") in allowed]
        log.info(
            "Allowlist: %d templates, filtered %d → %d problems",
            len(allowed),
            original_count,
            len(problems),
        )

    return problems


def sanitize_error_message(error_msg: str) -> str:
    """Strip pytest assertion internals that leak test logic from a raw error.

    Thin alias over ``core.text.strip_assertion_internals`` — the gold stage
    applies the same filter to already-built prompts.
    """
    return strip_assertion_internals(error_msg)


def format_correction_prompt(
    broken_code: str,
    original_prompt: str,
    error_msg: str,
    deprecated_warnings: str,
) -> str:
    """Build the correction prompt. Includes deprecation context when detected."""
    error_section = sanitize_error_message(error_msg)[:MAX_ERROR_IN_PROMPT]

    parts = [
        "The following Qiskit code doesn't work. Fix it so all tests pass.\n",
        f"```python\n{broken_code}\n```\n",
        f"Original task: {original_prompt}\n",
        f"Errors encountered:\n{error_section}",
    ]

    if deprecated_warnings:
        parts.append(f"\nNote: This code also uses deprecated Qiskit APIs:\n{deprecated_warnings}")

    return "\n".join(parts)


def classify_error(error_msg: str) -> str:
    """Classify the pytest error by inspecting the error message content."""
    lower = error_msg.lower()
    if "importerror" in lower or "modulenotfounderror" in lower:
        return "import_error"
    if "attributeerror" in lower:
        return "attribute_error"
    if "syntaxerror" in lower or "invalid syntax" in lower:
        return "syntax_error"
    if "assertionerror" in lower or "assert" in lower:
        return "assertion_error"
    if "nameerror" in lower:
        return "name_error"
    if "typeerror" in lower:
        return "type_error"
    if "timed out" in lower:
        return "timeout"
    if "empty code" in lower:
        return "empty"
    return "other"


def process_single_problem(
    problem: dict,
    problem_idx: int,
    session_factory: Callable[[], MelleaSession],
    cfg: GenConfig,
) -> list[CorrectionEntry]:
    """Run one problem through mellea's IVR loop and collect attempts.

    mellea owns the loop: with ``SimpleContext`` there is no conversation history,
    so each repair turn re-renders the ORIGINAL instruction with only the raw
    validator failure appended and the model re-solves fresh (it never sees its own
    broken code). ``RepairTemplateStrategy(loop_budget=N)`` allows N total attempts
    (1 initial + N-1 repairs) and breaks on the first pass. One mellea session is
    built per problem via ``session_factory`` and cleaned up in ``finally``.
    """
    from mellea.stdlib.requirements import req, simple_validate
    from mellea.stdlib.sampling import RepairTemplateStrategy

    imports = problem.get("imports", "")
    unittest_code = problem["unittest"]

    # Instruct prompt: task augmented with the imports available in the test env
    # (matches generate_correction_data.py).
    augmented_prompt = problem["prompt"]
    if imports.strip():
        augmented_prompt = (
            f"{augmented_prompt}\n\n"
            f"The following imports are available in the test environment:\n"
            f"```python\n{imports.strip()}\n```"
        )

    validator = make_composite_validator(imports, unittest_code, test_python=cfg.test_python)

    # Each element: (code, passed, error_msg) — the shape the recorded-row logic below expects.
    generations: list[tuple[str, bool, str]] = []
    session: MelleaSession | None = None
    try:
        session = session_factory()
        result = session.instruct(
            augmented_prompt,
            requirements=[
                req(
                    "Code must pass all pytest tests",
                    validation_fn=simple_validate(validator),
                )
            ],
            strategy=RepairTemplateStrategy(loop_budget=cfg.loop_budget),
            return_sampling_results=True,
        )

        # Adapt mellea's SamplingResult into (code, passed, error_msg) tuples so the
        # recorded-row logic below is identical to the legacy generator's.
        for gen, vals in zip(result.sample_generations, result.sample_validations):
            passed = all(v[1].as_bool() for v in vals)
            code = extract_code_from_markdown(str(gen.value or ""))
            if passed:
                error_msg = ""
            else:
                reasons = [v[1].reason for v in vals if not v[1].as_bool() and v[1].reason]
                error_msg = "\n".join(reasons) if reasons else "Unknown error"
            generations.append((code, passed, error_msg))
    except Exception:
        log.exception("Problem %d failed", problem_idx)
        return []
    finally:
        if session is not None:
            try:
                session.cleanup()
            except Exception:
                pass

    # First passing generation → the eventual solution.
    correct_code = ""
    solved_at_turn = 0
    for turn_idx, (code, passed, _error) in enumerate(generations):
        if passed and code.strip():
            correct_code = code
            solved_at_turn = turn_idx + 1
            break

    solved = bool(correct_code.strip())

    corrections: list[CorrectionEntry] = []
    for attempt_idx, (code, passed, error_msg) in enumerate(generations):
        if passed:
            continue
        if not code.strip():
            continue

        # (has_deprecated, warnings) already in row polarity — never re-invert.
        has_deprecated, deprecated_warnings = find_deprecated_apis(code)

        corrections.append(
            CorrectionEntry(
                prompt=format_correction_prompt(
                    code,
                    problem["prompt"],
                    error_msg,
                    deprecated_warnings,
                ),
                imports=imports,
                unittest=unittest_code,
                template="correction/" + problem.get("template", "unknown"),
                source_problem_idx=problem_idx,
                attempt_number=attempt_idx + 1,
                error_type=classify_error(error_msg),
                error_message=error_msg,
                has_deprecated_apis=has_deprecated,
                deprecated_api_warnings=deprecated_warnings,
                solved=solved,
                original_prompt=problem["prompt"],
                broken_code=code,
                correct_code=correct_code,
                solved_at_turn=solved_at_turn,
                turns_to_fix=(max(0, solved_at_turn - (attempt_idx + 1)) if solved_at_turn else 0),
            )
        )

    if correct_code.strip():
        corrections.append(
            CorrectionEntry(
                prompt=problem["prompt"],
                imports=imports,
                unittest=unittest_code,
                template=problem.get("template", "unknown"),
                source_problem_idx=problem_idx,
                attempt_number=0,
                error_type="none",
                error_message="",
                has_deprecated_apis=False,
                deprecated_api_warnings="",
                solved=True,
                original_prompt=problem["prompt"],
                broken_code="",
                correct_code=correct_code,
                solved_at_turn=solved_at_turn,
                turns_to_fix=0,
            )
        )

    log.info(
        "Problem %d: %d corrections, solved=%s, attempts=%d",
        problem_idx,
        len(corrections),
        solved,
        len(generations),
    )
    return corrections


def save_checkpoint(corrections: list[CorrectionEntry], batch_id: int, checkpoint_dir: str) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"checkpoint_{batch_id:06d}.jsonl")
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        for entry in corrections:
            f.write(json.dumps(asdict(entry)) + "\n")
    os.rename(tmp_path, path)
    log.info("Checkpoint saved: %s (%d entries)", path, len(corrections))


def load_checkpoints(checkpoint_dir: str) -> tuple[list[CorrectionEntry], set[int]]:
    entries = []
    seen_indices = set()

    if not os.path.isdir(checkpoint_dir):
        return entries, seen_indices

    for path in sorted(Path(checkpoint_dir).glob("checkpoint_*.jsonl")):
        try:
            with open(path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    entries.append(
                        CorrectionEntry(
                            **{
                                k: v
                                for k, v in data.items()
                                if k in CorrectionEntry.__dataclass_fields__
                            }
                        )
                    )
                    seen_indices.add(data["source_problem_idx"])
        except Exception:
            log.warning("Skipping corrupted checkpoint: %s", path)

    log.info(
        "Loaded %d entries from checkpoints, %d source problems done",
        len(entries),
        len(seen_indices),
    )
    return entries, seen_indices


def write_output(corrections: list[CorrectionEntry], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for entry in corrections:
            f.write(json.dumps(canonicalize_row(asdict(entry))) + "\n")
    log.info("Wrote %d correction problems to %s", len(corrections), output_path)


def print_summary(corrections: list[CorrectionEntry], total_problems: int, elapsed: float) -> None:
    problems_with_corrections = len({e.source_problem_idx for e in corrections})
    template_counts = Counter(e.template for e in corrections)
    error_counts = Counter(e.error_type for e in corrections)
    deprecated_count = sum(1 for e in corrections if e.has_deprecated_apis)
    solved_count = sum(1 for e in corrections if e.solved)
    unsolved_count = len(corrections) - solved_count

    print("\n" + "=" * 60)
    print("CORRECTION DATA GENERATION SUMMARY")
    print("=" * 60)
    print(f"Source problems processed:           {total_problems}")
    print(
        f"Problems that produced corrections:   {problems_with_corrections} "
        f"({100 * problems_with_corrections / max(total_problems, 1):.1f}%)"
    )
    print(f"Total correction problems generated: {len(corrections)}")
    print(f"  from solved problems (safe):       {solved_count}")
    print(f"  from unsolved problems (harder):   {unsolved_count}")
    print(f"  with deprecated APIs:              {deprecated_count}")
    print(f"Avg corrections per source problem:  {len(corrections) / max(total_problems, 1):.2f}")
    print(f"Time elapsed:                        {elapsed / 60:.1f} min")

    print("\nBy error type:")
    for err_type, count in error_counts.most_common():
        print(f"  {err_type}: {count}")

    print("\nBy template (top 15):")
    for template, count in template_counts.most_common(15):
        print(f"  {template}: {count}")

    print("=" * 60)


def run_pipeline(
    input_path: str,
    output_path: str,
    session_factory: Callable[[], MelleaSession],
    cfg: GenConfig,
    max_workers: int = 4,
    template_allowlist: str | None = None,
    checkpoint_dir: str | None = None,
    checkpoint_every: int = 50,
) -> int:
    """Run the correction IVR loop over all problems. Returns entries written."""
    problems = load_problems(input_path, template_allowlist)
    if not problems:
        log.error("No problems loaded")
        return 0

    log.info("Loaded %d problems", len(problems))

    existing_corrections: list[CorrectionEntry] = []
    completed_indices: set[int] = set()
    if checkpoint_dir:
        existing_corrections, completed_indices = load_checkpoints(checkpoint_dir)
        if completed_indices:
            log.info("Resuming: %d problems already done", len(completed_indices))

    all_corrections = list(existing_corrections)
    remaining = [(idx, p) for idx, p in enumerate(problems) if idx not in completed_indices]
    log.info("Processing %d remaining problems with %d workers", len(remaining), max_workers)

    start_time = time.time()
    processed = 0

    for batch_start in range(0, len(remaining), checkpoint_every):
        batch = remaining[batch_start : batch_start + checkpoint_every]
        batch_corrections: list[CorrectionEntry] = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_single_problem, problem, idx, session_factory, cfg): idx
                for idx, problem in batch
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    # Guard against a wedged worker, not a hard bound: a timeout
                    # here abandons the result but cannot cancel the thread, which
                    # keeps its slot until the process exits. The real bounds are
                    # the per-request timeout/num_retries and the pytest timeout.
                    batch_corrections.extend(future.result(timeout=600))
                except Exception:
                    log.exception("Problem %d: executor error", idx)

        all_corrections.extend(batch_corrections)
        processed += len(batch)

        if checkpoint_dir and batch_corrections:
            save_checkpoint(batch_corrections, batch_start, checkpoint_dir)

        log.info(
            "Progress: %d/%d problems, %d total corrections so far",
            processed + len(completed_indices),
            len(problems),
            len(all_corrections),
        )

    elapsed = time.time() - start_time
    write_output(all_corrections, output_path)
    print_summary(all_corrections, len(problems), elapsed)
    return len(all_corrections)
