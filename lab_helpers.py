"""Helpers for the LAB-2681 notebook.

Keeps the notebook cells short: loading the sample prompts, running one prompt through
the Mellea Instruct-Validate-Repair loop with a full per-attempt trace, and the plots.
The data generation itself is quantum-data-forge's ``data_augmentation`` package,
unchanged; this file only wraps it for a classroom.
"""

from __future__ import annotations

import difflib
import html
import json
import os
import re
import sys
import textwrap
import time
from collections import Counter
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent
# `data_augmentation/` (the pipeline) ships inside this folder: an unmodified copy of
# quantum-data-forge's package (commit c653954), so the lab needs nothing else.
if str(LAB_DIR) not in sys.path:
    sys.path.insert(0, str(LAB_DIR))

# The pipeline writes each test to a temp file and runs pytest on it with the temp dir as
# rootdir. On a shared machine, pytest then scans every other user's files in /tmp and fails
# with PermissionError before running a single test, so give this kernel its own temp dir.
_PRIVATE_TMP = Path(os.environ.get("LAB_TMPDIR", f"/tmp/lab2681-{os.getuid()}"))
_PRIVATE_TMP.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(_PRIVATE_TMP)
import tempfile  # noqa: E402

tempfile.tempdir = None  # re-read TMPDIR

# Jupyter (ipykernel) sets FORCE_COLOR=1 for child processes, so pytest prints ANSI colour
# codes into its report, and those codes would end up inside the correction prompts.
# Turn colour off for every pytest the pipeline runs from this kernel.
os.environ.pop("FORCE_COLOR", None)
os.environ["NO_COLOR"] = "1"
os.environ["PY_COLORS"] = "0"
# Mellea logs a banner with the whole system prompt for every session; keep only warnings.
os.environ.setdefault("MELLEA_LOG_LEVEL", "ERROR")   # also hides a harmless litellm "may drop api_base" notice

# Run every test with an empty HOME, as the RL reward does. Model code that calls
# QiskitRuntimeService() at import time would otherwise pick up a saved IBM Quantum account
# from ~/.qiskit and contact the real service, so results would depend on whose machine runs
# them (and use that person's account).
_TEST_HOME = _PRIVATE_TMP / "home"
_TEST_HOME.mkdir(exist_ok=True)
os.environ["HOME"] = str(_TEST_HOME)
for _var in ("QISKIT_IBM_TOKEN", "QISKIT_IBM_INSTANCE", "QISKIT_IBM_CHANNEL", "QISKIT_IBM_URL"):
    os.environ.pop(_var, None)

# Plot colours, one per data modality, reused in every chart.
MODALITY_COLORS = {"QA": "#4C78A8", "Correction": "#F58518", "Fill-in-the-blank": "#54A24B"}
MODEL_COLORS = {"base": "#9D9DA0", "tuned": "#7B4FE0"}


# --------------------------------------------------------------------------------------
# Loading and describing the input prompts
# --------------------------------------------------------------------------------------
def load_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows: list[dict], path: str | Path) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def topic_of(template: str) -> str:
    """'template_examples/quantum_info/random_unitary.yaml' -> 'quantum_info'."""
    parts = template.split("/")
    return parts[1] if len(parts) > 2 else parts[0]


def count_tests(unittest_code: str) -> int:
    return len(re.findall(r"^\s*def test_", unittest_code, re.M))


def describe_prompts(rows: list[dict]):
    import pandas as pd

    return pd.DataFrame(
        {
            "topic": [topic_of(r["template"]) for r in rows],
            "template": [Path(r["template"]).stem for r in rows],
            "prompt": [textwrap.shorten(r["prompt"], 90) for r in rows],
            "prompt_chars": [len(r["prompt"]) for r in rows],
            "unit_tests": [count_tests(r["unittest"]) for r in rows],
        }
    )


# --------------------------------------------------------------------------------------
# Pretty printing
# --------------------------------------------------------------------------------------
def show_code(code: str, title: str | None = None, lang: str = "python") -> None:
    from IPython.display import Markdown, display

    head = f"**{title}**\n\n" if title else ""
    display(Markdown(f"{head}```{lang}\n{code.rstrip()}\n```"))


def show_text(text: str, title: str | None = None, max_chars: int = 2500) -> None:
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... (truncated)"
    show_code(text, title, lang="text")


def show_diff(before: str, after: str, title: str = "") -> None:
    """Side-by-side diff of two code snippets (red = removed, green = added)."""
    from IPython.display import HTML, display

    table = difflib.HtmlDiff(wrapcolumn=70).make_table(
        before.splitlines(), after.splitlines(), "attempt that failed", "attempt that passed", context=True, numlines=2
    )
    style = (
        "<style>.lab-diff table{font-family:monospace;font-size:12px;border-collapse:collapse}"
        ".lab-diff td{padding:0 6px}.lab-diff .diff_add{background:#d7f5d7}"
        ".lab-diff .diff_sub{background:#fbdcdc}.lab-diff .diff_chg{background:#fff3c4}"
        ".lab-diff .diff_header{color:#888}</style>"
    )
    display(HTML(f"{style}<div class='lab-diff'><b>{html.escape(title)}</b>{table}</div>"))


# --------------------------------------------------------------------------------------
# Talking to a model directly (one plain call, no Mellea)
# --------------------------------------------------------------------------------------
def ask(base_url: str, model: str, prompt: str, system: str | None = None, temperature: float = 0.6,
        max_tokens: int = 4096) -> str:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=os.environ.get("LITELLM_API_KEY", "EMPTY"))
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    out = client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
    return out.choices[0].message.content or ""


def extract_code(text: str) -> str:
    """First ```python block of a model answer (or the whole answer if there is none)."""
    from data_augmentation.core.validation import extract_code_from_markdown

    return extract_code_from_markdown(text)


def run_tests(code: str, problem: dict, test_python: str | None = None) -> tuple[bool, str]:
    """Run a problem's unit tests against a code string, exactly as the pipeline does."""
    from data_augmentation.core.validation import validate_pytest

    return validate_pytest(code, problem.get("imports", ""), problem["unittest"], test_python=test_python)


# --------------------------------------------------------------------------------------
# The IVR loop, one prompt at a time, with every attempt kept for display
# --------------------------------------------------------------------------------------
_SITE_PACKAGES = re.compile(r"\S*?/(?:site|dist)-packages/")
_STDLIB = re.compile(r"\S*?/lib/python3\.\d+/(?!site-packages)")
_TMP_TEST_FILE = re.compile(r"(?:\S*/)?(qiskit_test_)\w+(\.py)")
_DURATION = re.compile(r"((?:passed|failed|error|errors|skipped|warnings?)) in \d+(?:\.\d+)?s\b")


def neutral_paths(text: str) -> str:
    """Make a pytest report machine-independent.

    The report names the temp test file (random name, per-user folder) and the Python install path.
    Those land inside the correction prompts, so without this the same failure would read differently
    on every machine: noisy for training data, and replay could not match a prompt to its recording.
    """
    text = _SITE_PACKAGES.sub("site-packages/", text)
    text = _STDLIB.sub("python/lib/", text)
    text = _DURATION.sub(r"\1", text)  # "1 failed in 0.31s": the timing differs on every run
    return _TMP_TEST_FILE.sub(r"\1solution\2", text)


def answer_key(prompt: str) -> str:
    """Key for looking up a recorded answer to a prompt.

    A correction prompt pastes in the pytest report, which still varies by machine in small ways
    (line wrapping follows the terminal width, objects print their memory address). Everything
    before the pasted errors (the broken code and the task) is identical everywhere, so that part
    is the key.
    """
    return prompt.split("\nErrors encountered:", 1)[0].strip()


def ivr_trace(problem: dict, problem_idx: int, session_factory, cfg, tries: int = 3) -> dict:
    """Run one prompt through Mellea's Instruct-Validate-Repair loop.

    Same call as ``data_augmentation.stages.correction.process_single_problem`` (same
    instruction, validator and repair strategy), but it also returns every attempt, so
    the notebook can show what the model wrote and what the tests said at each turn.

    A dropped connection to the model (it happens now and then under parallel load) is
    retried; if every try fails, the prompt comes back with no attempts and a ``skipped``
    reason instead of raising, so one bad request can't break the rest of the notebook.
    """
    last_error = None
    for _ in range(tries):
        try:
            return _ivr_trace_once(problem, problem_idx, session_factory, cfg)
        except Exception as e:  # noqa: BLE001 - network/endpoint errors from litellm
            last_error = f"{type(e).__name__}: {e}"
            time.sleep(2)
    return {"problem_idx": problem_idx, "template": problem["template"], "instruction": problem["prompt"],
            "attempts": [], "solved_at_turn": 0, "seconds": 0.0, "skipped": last_error}


def _ivr_trace_once(problem: dict, problem_idx: int, session_factory, cfg) -> dict:
    from mellea.stdlib.requirements import req, simple_validate
    from mellea.stdlib.sampling import RepairTemplateStrategy

    from data_augmentation.core.validation import extract_code_from_markdown, find_deprecated_apis, make_composite_validator
    from data_augmentation.stages.correction import classify_error

    imports = problem.get("imports", "")
    instruction = problem["prompt"]
    if imports.strip():
        instruction += (
            "\n\nThe following imports are available in the test environment:\n"
            f"```python\n{imports.strip()}\n```"
        )
    validator = make_composite_validator(imports, problem["unittest"], test_python=cfg.test_python)

    t0 = time.time()
    session = session_factory()
    try:
        result = session.instruct(
            instruction,
            requirements=[req("Code must pass all pytest tests", validation_fn=simple_validate(validator))],
            strategy=RepairTemplateStrategy(loop_budget=cfg.loop_budget),
            return_sampling_results=True,
        )
    finally:
        session.cleanup()

    attempts = []
    for turn, (gen, vals) in enumerate(zip(result.sample_generations, result.sample_validations), start=1):
        passed = all(v[1].as_bool() for v in vals)
        code = extract_code_from_markdown(str(gen.value or ""))
        error = "" if passed else neutral_paths("\n".join(v[1].reason for v in vals if not v[1].as_bool() and v[1].reason))
        deprecated, warnings = find_deprecated_apis(code)
        attempts.append(
            {
                "turn": turn,
                "passed": passed,
                "code": code,
                "error": error,
                "error_type": "none" if passed else classify_error(error),
                "deprecated": deprecated,
                "deprecated_warnings": warnings,
            }
        )
    solved_at = next((a["turn"] for a in attempts if a["passed"]), 0)
    return {
        "problem_idx": problem_idx,
        "template": problem["template"],
        "instruction": instruction,
        "attempts": attempts,
        "solved_at_turn": solved_at,
        "seconds": round(time.time() - t0, 1),
    }


def print_trace(trace: dict, show_all_code: bool = False) -> None:
    from IPython.display import Markdown, display

    n = len(trace["attempts"])
    if trace.get("skipped"):
        display(Markdown(f"### Prompt #{trace['problem_idx']}: skipped, the model could not be reached\n\n`{trace['skipped'][:300]}`"))
        return
    verdict = f"solved at turn {trace['solved_at_turn']}" if trace["solved_at_turn"] else f"not solved in {n} turns"
    display(Markdown(f"### Prompt #{trace['problem_idx']}: `{Path(trace['template']).stem}`, {verdict} ({trace['seconds']} s)"))
    for a in trace["attempts"]:
        badge = "✅ PASS" if a["passed"] else f"❌ FAIL ({a['error_type']})"
        dep = "  ⚠️ deprecated Qiskit API detected" if a["deprecated"] else ""
        display(Markdown(f"**Turn {a['turn']}: {badge}{dep}**"))
        if show_all_code or a["passed"] or a["turn"] == n:
            show_code(a["code"] or "# (empty answer)")
        if a["error"]:
            show_text(a["error"], "What the unit tests reported", max_chars=1200)
        if a["deprecated_warnings"]:
            show_text(a["deprecated_warnings"], "flake8-qiskit-migration says")


# --------------------------------------------------------------------------------------
# Visualizations
# --------------------------------------------------------------------------------------
def plot_ivr_timeline(traces: list[dict], loop_budget: int):
    """One row per prompt, one cell per attempt: green = passed, red = failed."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    traces = sorted((t for t in traces if t["attempts"]), key=lambda t: (t["solved_at_turn"] or 99, t["problem_idx"]))
    fig, ax = plt.subplots(figsize=(8.5, 0.32 * len(traces) + 1.4))
    for y, t in enumerate(traces):
        for a in t["attempts"]:
            color = "#54A24B" if a["passed"] else ("#E45756" if a["error_type"] != "timeout" else "#B279A2")
            ax.add_patch(plt.Rectangle((a["turn"] - 0.45, y - 0.4), 0.9, 0.8, color=color))
            if a["deprecated"]:
                ax.text(a["turn"], y, "⚠", ha="center", va="center", fontsize=9)
    ax.set_xlim(0.4, loop_budget + 0.6)
    ax.set_ylim(-0.6, len(traces) - 0.4)
    ax.set_xticks(range(1, loop_budget + 1))
    ax.set_xlabel("attempt (turn) inside the IVR loop")
    ax.set_yticks(range(len(traces)))
    ax.set_yticklabels([f"#{t['problem_idx']} {Path(t['template']).stem[:32]}" for t in traces], fontsize=8)
    ax.invert_yaxis()
    ax.set_title("Instruct → Validate → Repair: every attempt the model made")
    ax.legend(
        handles=[Patch(color="#54A24B", label="passed the unit tests"), Patch(color="#E45756", label="failed"),
                 Patch(color="#B279A2", label="timed out")],
        loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=3, frameon=False, fontsize=8,
    )
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def plot_error_types(hub_rows: list[dict]):
    import matplotlib.pyplot as plt

    counts = Counter(r["error_type"] for r in hub_rows if r["attempt_number"] > 0)
    fig, ax = plt.subplots(figsize=(6, 2.6))
    if not counts:
        ax.text(0.5, 0.5, "no failed attempts: every prompt passed on the first try", ha="center", va="center")
        ax.axis("off")
        return fig
    labels, values = zip(*counts.most_common())
    ax.barh(labels, values, color="#E45756")
    for i, v in enumerate(values):
        ax.text(v + 0.05, i, str(v), va="center", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("failed attempts")
    ax.set_title("Why attempts failed (these become correction data)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def plot_dataset_funnel(n_prompts: int, n_solved: int, n_qa: int, n_corr: int, n_fib_by_type: dict):
    """How 20 input prompts turn into three kinds of training data."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    labels = ["input prompts", "solved by the loop", "QA rows", "correction rows", "fill-in-the-blank rows"]
    values = [n_prompts, n_solved, n_qa, n_corr, sum(n_fib_by_type.values())]
    colors = ["#BAB0AC", "#BAB0AC", MODALITY_COLORS["QA"], MODALITY_COLORS["Correction"], MODALITY_COLORS["Fill-in-the-blank"]]
    ax.barh(labels, values, color=colors)
    for i, v in enumerate(values):
        ax.text(v + 0.2, i, str(v), va="center", fontsize=9)
    ax.invert_yaxis()
    ax.set_title("From prompts to training data")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax = axes[1]
    if sum(n_fib_by_type.values()):
        kinds = list(n_fib_by_type)
        ax.bar(kinds, [n_fib_by_type[k] for k in kinds], color=MODALITY_COLORS["Fill-in-the-blank"])
        for i, k in enumerate(kinds):
            ax.text(i, n_fib_by_type[k] + 0.1, str(n_fib_by_type[k]), ha="center", fontsize=9)
    ax.set_title("Fill-in-the-blank: what was deleted")
    ax.set_xlabel("deletion type")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def plot_modality_profile(qa: list[dict], corr: list[dict], fib: list[dict]):
    """Prompt length per modality: each modality hands the model a different amount of context."""
    import matplotlib.pyplot as plt

    groups = {"QA": qa, "Correction": corr, "Fill-in-the-blank": fib}
    fig, ax = plt.subplots(figsize=(7, 2.8))
    data = [[len(r["prompt"]) for r in rows] or [0] for rows in groups.values()]
    bp = ax.boxplot(data, vert=False, patch_artist=True, widths=0.55)
    for patch, name in zip(bp["boxes"], groups):
        patch.set_facecolor(MODALITY_COLORS[name])
        patch.set_alpha(0.75)
    ax.set_yticks(range(1, 4))
    ax.set_yticklabels([f"{k}\n(n={len(v)})" for k, v in groups.items()])
    ax.set_xlabel("prompt length (characters)")
    ax.set_title("How much the model is given to start from")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def plot_base_vs_tuned(results: dict[str, dict[str, list[bool]]]):
    """results[model][modality] = list of pass/fail booleans."""
    import matplotlib.pyplot as plt
    import numpy as np

    modalities = [m for m in MODALITY_COLORS if any(results[k].get(m) for k in results)]
    models = list(results)
    x = np.arange(len(modalities))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    for i, model in enumerate(models):
        rates = [np.mean(results[model][m]) if results[model].get(m) else 0 for m in modalities]
        ns = [len(results[model].get(m, [])) for m in modalities]
        bars = ax.bar(x + (i - 0.5) * w, rates, w, label=model, color=MODEL_COLORS.get(model.split()[0], None))
        for b, r, n in zip(bars, rates, ns):
            ax.text(b.get_x() + b.get_width() / 2, r + 0.02, f"{r:.0%}\n(n={n})", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(modalities)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("pass rate on the unit tests")
    ax.set_title("Base vs. RL-tuned model, one try per prompt", pad=22)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def plot_training_curve(steps: list[int], val: list[float], base: float | None = None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 3))
    ax.plot(steps, val, color=MODEL_COLORS["tuned"], lw=1.8, marker="o", ms=2.5, label="tuned model (validation)")
    if base is not None:
        ax.axhline(base, color=MODEL_COLORS["base"], ls="--", lw=1.4, label="base model")
    ax.set_xlabel("GRPO training step")
    ax.set_ylabel("held-out pass rate")
    ax.set_ylim(0, max(val) * 1.15)
    ax.set_title("Reinforcement learning against unit-test rewards")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig
