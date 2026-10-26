"""Fill-in-the-blank stage: build incomplete-code data from solved code.

Takes correction-hub entries with correct_code and produces variants with
targeted deletions:
  - Import deletion: remove an import line, model must add it back
  - Function call deletion: remove a key API call line, model must complete it
  - Combined deletion: remove an import AND its corresponding usage line

Output is JSONL compatible with generate_harbor_tasks.py (canonical keys first).
This module is pure library code; the CLI lives in ``data_augmentation.cli``.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict

from ..core.rows import canonicalize_row
from ..core.validation import _populate_solution_source
from ..models import FillInBlankEntry

TRIVIAL_IMPORTS = {
    "from qiskit import QuantumCircuit",
    "import numpy as np",
    "from math import pi",
    "from math import sqrt",
}

IMPORT_RE = re.compile(r"^\s*(from\s+\S+\s+import\s+.+|import\s+\S+.*)$")

KEY_CALL_PATTERNS = [
    re.compile(r"^\s*\w+\s*=\s*\w+\.run\("),
    re.compile(r"^\s*\w+\s*=\s*generate_preset_pass_manager\("),
    re.compile(r"^\s*\w+\s*=\s*\w+\.run\(\["),
    re.compile(r"^\s*\w+\s*=\s*(Estimator|EstimatorV2|Sampler|SamplerV2)\("),
    re.compile(r"^\s*\w+\s*=\s*AerSimulator\("),
    re.compile(r"^\s*\w+\s*=\s*Fake\w+\("),
    re.compile(r"^\s*\w+\s*=\s*SparsePauliOp\("),
    re.compile(r"^\s*\w+\s*=\s*(EfficientSU2|TwoLocal|RealAmplitudes)\("),
    re.compile(r"^\s*\w+\s*=\s*QuantumCircuit\("),
    re.compile(r"^\s*\w+\.measure_all\("),
    re.compile(r"^\s*\w+\.compose\("),
    re.compile(r"^\s*\w+\s*=\s*qasm[23]\.(loads|dumps)\("),
    re.compile(r"^\s*\w+\s*=\s*\w+\.result\("),
]

PYTEST_TIMEOUT = 30

FILL_BLANK_PROMPT = (
    "The following Qiskit code is incomplete. Write the complete working version so all tests pass."
)


def extract_imported_names(import_line: str) -> list[str]:
    """Extract the names introduced by an import line.

    'from qiskit import QuantumCircuit, Aer' → ['QuantumCircuit', 'Aer']
    'from qiskit import QuantumCircuit as QC' → ['QC']
    'import numpy as np' → ['np']
    'import json' → ['json']
    """
    line = import_line.strip()
    if line.startswith("from "):
        m = re.match(r"from\s+\S+\s+import\s+(.+)", line)
        if not m:
            return []
        names_part = m.group(1)
        names = []
        for item in names_part.split(","):
            item = item.strip()
            if " as " in item:
                names.append(item.split(" as ")[-1].strip())
            else:
                names.append(item.strip())
        return [n for n in names if n]
    elif line.startswith("import "):
        m = re.match(r"import\s+(.+)", line)
        if not m:
            return []
        if " as " in m.group(1):
            return [m.group(1).split(" as ")[-1].strip()]
        module = m.group(1).split(",")[0].strip()
        return [module.split(".")[0]]
    return []


def imported_names_used_in_code(import_line: str, code: str) -> bool:
    """Check if any name from the import is actually referenced in non-import code lines."""
    names = extract_imported_names(import_line)
    if not names:
        return True
    code_lines = []
    for line in code.split("\n"):
        stripped = line.strip()
        if stripped and not stripped.startswith("import ") and not stripped.startswith("from "):
            code_lines.append(stripped)
    body = "\n".join(code_lines)
    return any(re.search(r"\b" + re.escape(name) + r"\b", body) for name in names)


def find_instantiation_lines(import_line: str, code: str) -> list[tuple[int, str, str]]:
    """Find lines that instantiate/call an imported class or function.

    Returns (line_idx, line_text, variable_name) for lines like:
      sampler = StatevectorSampler()
      pm = generate_preset_pass_manager(...)
    """
    names = extract_imported_names(import_line)
    if not names:
        return []
    results = []
    lines = code.split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("import ") or stripped.startswith("from "):
            continue
        for name in names:
            pattern = rf"^(\w+)\s*=\s*{re.escape(name)}\s*[\(.]"
            m = re.match(pattern, stripped)
            if m:
                results.append((i, stripped, m.group(1)))
                break
    return results


def variable_used_after(var_name: str, code: str, after_line_idx: int) -> bool:
    """Check if a variable is referenced in code after a given line."""
    lines = code.split("\n")
    for i in range(after_line_idx + 1, len(lines)):
        stripped = lines[i].strip()
        if not stripped or stripped.startswith("#"):
            continue
        if re.search(r"\b" + re.escape(var_name) + r"\b", stripped):
            return True
    return False


def parse_import_lines(code: str) -> list[tuple[int, str]]:
    """Return (line_index, line_text) for each import line in the code."""
    results = []
    for i, line in enumerate(code.split("\n")):
        if IMPORT_RE.match(line) and line.strip():
            results.append((i, line.strip()))
    return results


def parse_key_call_lines(code: str) -> list[tuple[int, str]]:
    """Return (line_index, line_text) for key API call lines."""
    results = []
    for i, line in enumerate(code.split("\n")):
        stripped = line.strip()
        if (
            not stripped
            or stripped.startswith("#")
            or stripped.startswith("import")
            or stripped.startswith("from")
        ):
            continue
        for pattern in KEY_CALL_PATTERNS:
            if pattern.match(stripped):
                results.append((i, stripped))
                break
    return results


def remove_line(code: str, line_idx: int) -> str:
    """Remove a line from code by index, collapsing extra blank lines."""
    lines = code.split("\n")
    lines.pop(line_idx)
    result = "\n".join(lines)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def remove_lines(code: str, line_indices: list[int]) -> str:
    """Remove multiple lines from code by indices, collapsing extra blank lines."""
    lines = code.split("\n")
    for idx in sorted(line_indices, reverse=True):
        lines.pop(idx)
    result = "\n".join(lines)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def is_syntactically_valid(code: str) -> bool:
    """Check if code is syntactically valid Python."""
    try:
        compile(code, "<string>", "exec")
        return True
    except SyntaxError:
        return False


def remove_import_from_field(imports_field: str, deleted_import: str) -> str:
    """Remove a matching import line from the imports field.

    Matches by comparing the imported names/modules, not exact string,
    to handle formatting differences.
    """
    deleted_stripped = deleted_import.strip()
    lines = imports_field.split("\n")
    result = []
    for line in lines:
        if line.strip() == deleted_stripped:
            continue
        if _imports_equivalent(line.strip(), deleted_stripped):
            continue
        result.append(line)
    return "\n".join(result)


def _imports_equivalent(a: str, b: str) -> bool:
    """Check if two import lines import the same thing."""

    def _normalize(s):
        s = re.sub(r"\s+", " ", s.strip())
        return s

    return _normalize(a) == _normalize(b)


def validate_deletion_causes_failure(
    modified_code: str,
    imports: str,
    unittest_code: str,
    test_python: str | None = None,
) -> bool:
    """Run pytest on modified code to confirm the deletion causes test failure."""
    unittest_code = _populate_solution_source(unittest_code)
    combined = f"{imports}\n{modified_code}\n{unittest_code}"
    py = test_python or sys.executable
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix="fill_validate_", delete=False
    ) as f:
        f.write(combined)
        tmp_path = f.name
    try:
        result = subprocess.run(
            [py, "-m", "pytest", tmp_path, "-x", "-q", "--tb=line", "--no-header"],
            capture_output=True,
            text=True,
            timeout=PYTEST_TIMEOUT,
        )
        return result.returncode != 0
    except subprocess.TimeoutExpired:
        return True
    except Exception:
        return False
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _make_prompt(modified_code: str, original_task: str) -> str:
    return (
        f"{FILL_BLANK_PROMPT}\n\n```python\n{modified_code}\n```\n\nOriginal task: {original_task}"
    )


def _template_base(entry: dict) -> str:
    template = entry.get("template", "unknown")
    if template.startswith("correction/"):
        template = template[len("correction/") :]
    return template


def generate_import_deletions(
    entry: dict, validate: bool = False, test_python: str | None = None
) -> list[FillInBlankEntry]:
    """Generate import deletion variants from an entry with correct_code."""
    code = entry["correct_code"]
    import_lines = parse_import_lines(code)

    if not import_lines:
        return []

    original_task = entry.get("original_prompt", entry.get("prompt", ""))
    results = []
    for line_idx, import_line in import_lines:
        if import_line in TRIVIAL_IMPORTS:
            continue

        if not imported_names_used_in_code(import_line, code):
            continue

        modified_code = remove_line(code, line_idx)
        modified_imports = remove_import_from_field(entry["imports"], import_line)

        if validate:
            if not validate_deletion_causes_failure(
                modified_code, modified_imports, entry["unittest"], test_python
            ):
                continue

        results.append(
            FillInBlankEntry(
                prompt=_make_prompt(modified_code, original_task),
                imports=modified_imports,
                unittest=entry["unittest"],
                template=f"fill_import/{_template_base(entry)}",
                source_problem_idx=entry.get("source_problem_idx", -1),
                deletion_type="import",
                deleted_line=import_line,
                correct_code=code,
                original_prompt=original_task,
            )
        )

    return results


def generate_call_deletions(
    entry: dict, validate: bool = False, test_python: str | None = None
) -> list[FillInBlankEntry]:
    """Generate function call deletion variants from an entry with correct_code."""
    code = entry["correct_code"]
    call_lines = parse_key_call_lines(code)

    if not call_lines:
        return []

    original_task = entry.get("original_prompt", entry.get("prompt", ""))
    results = []
    for line_idx, call_line in call_lines:
        modified_code = remove_line(code, line_idx)

        if validate:
            if not validate_deletion_causes_failure(
                modified_code, entry["imports"], entry["unittest"], test_python
            ):
                continue

        results.append(
            FillInBlankEntry(
                prompt=_make_prompt(modified_code, original_task),
                imports=entry["imports"],
                unittest=entry["unittest"],
                template=f"fill_call/{_template_base(entry)}",
                source_problem_idx=entry.get("source_problem_idx", -1),
                deletion_type="call",
                deleted_line=call_line,
                correct_code=code,
                original_prompt=original_task,
            )
        )

    return results


def generate_combined_deletions(
    entry: dict, validate: bool = False, test_python: str | None = None
) -> list[FillInBlankEntry]:
    """Generate combined import + usage deletion variants.

    Removes an import line AND the line that instantiates/calls the imported
    class or function. The remaining code still references the created variable,
    hinting at what's missing.
    """
    code = entry["correct_code"]
    import_lines = parse_import_lines(code)

    if not import_lines:
        return []

    original_task = entry.get("original_prompt", entry.get("prompt", ""))
    results = []
    for import_idx, import_line in import_lines:
        if import_line in TRIVIAL_IMPORTS:
            continue
        if not imported_names_used_in_code(import_line, code):
            continue

        instantiations = find_instantiation_lines(import_line, code)
        if not instantiations:
            continue

        for inst_idx, inst_line, var_name in instantiations:
            if not variable_used_after(var_name, code, inst_idx):
                continue

            modified_code = remove_lines(code, [import_idx, inst_idx])
            if not is_syntactically_valid(modified_code):
                continue

            modified_imports = remove_import_from_field(entry["imports"], import_line)

            if validate:
                if not validate_deletion_causes_failure(
                    modified_code, modified_imports, entry["unittest"], test_python
                ):
                    continue

            results.append(
                FillInBlankEntry(
                    prompt=_make_prompt(modified_code, original_task),
                    imports=modified_imports,
                    unittest=entry["unittest"],
                    template=f"fill_combined/{_template_base(entry)}",
                    source_problem_idx=entry.get("source_problem_idx", -1),
                    deletion_type="combined",
                    deleted_line=f"{import_line}\n{inst_line}",
                    correct_code=code,
                    original_prompt=original_task,
                )
            )

    return results


def _cap_variants(variants: list[FillInBlankEntry], max_per_type: int) -> list[FillInBlankEntry]:
    """Truncate to at most ``max_per_type`` variants, keeping the first and last.

    Candidates arrive in source order, so the first and last deletion are the
    most dissimilar pair available (e.g. an early import vs. a late call); when
    the cap allows more than two, the gap is filled from the front. Returns
    ``variants`` unchanged when it already fits under the cap.
    """
    if max_per_type <= 0:
        return []
    if len(variants) <= max_per_type:
        return list(variants)
    if max_per_type == 1:
        return [variants[0]]
    return [*variants[: max_per_type - 1], variants[-1]]


def deduplicate_variants(
    variants: list[FillInBlankEntry], max_per_type: int = 2
) -> list[FillInBlankEntry]:
    """Remove redundant variants:
    1. Drop import-only when combined covers the same import for the same problem
    2. Cap at max_per_type variants per (problem, deletion_type)
    """
    by_problem: dict[int, list[FillInBlankEntry]] = {}
    for v in variants:
        by_problem.setdefault(v.source_problem_idx, []).append(v)

    result = []
    for pvariants in by_problem.values():
        combined_imports = set()
        for v in pvariants:
            if v.deletion_type == "combined":
                combined_imports.add(v.deleted_line.split("\n")[0])

        filtered = []
        for v in pvariants:
            if v.deletion_type == "import" and v.deleted_line in combined_imports:
                continue
            filtered.append(v)

        by_type: dict[str, list[FillInBlankEntry]] = {}
        for v in filtered:
            by_type.setdefault(v.deletion_type, []).append(v)

        for tvariants in by_type.values():
            result.extend(_cap_variants(tvariants, max_per_type))

    return result


def ensure_imports_in_code(code: str, imports_field: str) -> str:
    """Prepend imports to correct_code if it doesn't already have them."""
    if not imports_field.strip():
        return code
    has_imports = any(
        line.strip().startswith("import ") or line.strip().startswith("from ")
        for line in code.split("\n")[:10]
    )
    if has_imports:
        return code
    return f"{imports_field.strip()}\n\n{code}"


def load_entries_with_correct_code(input_path: str) -> list[dict]:
    """Load entries that have non-empty correct_code, deduplicated by source problem."""
    seen = {}
    patched = 0
    with open(input_path) as f:
        for line in f:
            if not line.strip():
                continue
            e = json.loads(line)
            if not e.get("correct_code", "").strip():
                continue
            idx = e.get("source_problem_idx", id(e))
            if idx not in seen:
                original = e["correct_code"]
                e["correct_code"] = ensure_imports_in_code(e["correct_code"], e.get("imports", ""))
                if e["correct_code"] != original:
                    patched += 1
                seen[idx] = e
    if patched:
        print(f"  Patched {patched} entries with missing imports in correct_code")
    return list(seen.values())


def run_pipeline(
    input_path: str,
    output_path: str,
    types: str = "import,call,combined",
    max_per_type: int = 2,
    validate: bool = False,
    max_workers: int = 1,
    test_python: str | None = None,
) -> int:
    """Generate, (optionally) validate, dedupe and write fill-in-blank variants.

    Returns the number of variants written.
    """
    entries = load_entries_with_correct_code(input_path)
    if not entries:
        print("No entries with correct_code found")
        return 0

    print(f"Loaded {len(entries)} problems with correct_code from {input_path}")

    type_set = set(types.split(","))
    if validate:
        print("Validation enabled — will run pytest for each deletion (slow)")

    def _process_entry(entry: dict) -> list[FillInBlankEntry]:
        """Generate all requested deletion variants for one problem.

        Independent per problem, so validation pytest runs (the slow part)
        can be parallelized across problems with a thread pool — pytest is a
        subprocess, so it releases the GIL and threads give real parallelism.
        """
        out: list[FillInBlankEntry] = []
        if "import" in type_set:
            out.extend(generate_import_deletions(entry, validate, test_python))
        if "call" in type_set:
            out.extend(generate_call_deletions(entry, validate, test_python))
        if "combined" in type_set:
            out.extend(generate_combined_deletions(entry, validate, test_python))
        return out

    all_variants: list[FillInBlankEntry] = []
    if max_workers <= 1:
        for i, entry in enumerate(entries):
            all_variants.extend(_process_entry(entry))
            if (i + 1) % 100 == 0:
                print(
                    f"  Processed {i + 1}/{len(entries)} problems, "
                    f"{len(all_variants)} variants so far",
                    flush=True,
                )
    else:
        print(f"Parallel validation with {max_workers} workers", flush=True)
        done = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_process_entry, e) for e in entries]
            for fut in as_completed(futures):
                try:
                    all_variants.extend(fut.result())
                except Exception as exc:
                    print(f"  entry failed: {exc}", flush=True)
                done += 1
                if done % 100 == 0:
                    print(
                        f"  Processed {done}/{len(entries)} problems, "
                        f"{len(all_variants)} variants so far",
                        flush=True,
                    )

    raw_count = len(all_variants)
    all_variants = deduplicate_variants(all_variants, max_per_type=max_per_type)
    print(
        f"\nDeduplication: {raw_count} → {len(all_variants)} "
        f"(removed {raw_count - len(all_variants)} redundant variants)"
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for entry in all_variants:
            f.write(json.dumps(canonicalize_row(asdict(entry))) + "\n")

    print(f"Wrote {len(all_variants)} fill-in-the-blank variants to {output_path}")

    type_counts = Counter(e.deletion_type for e in all_variants)
    template_counts = Counter(e.template for e in all_variants)

    print("\nBy deletion type:")
    for t, count in type_counts.most_common():
        print(f"  {t}: {count}")

    print("\nBy template (top 15):")
    for tmpl, count in template_counts.most_common(15):
        print(f"  {tmpl}: {count}")

    return len(all_variants)
