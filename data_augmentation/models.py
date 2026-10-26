"""Row and config dataclasses shared across the pipeline stages.

Kept separate from the stage modules so ``correction``, ``gold`` and
``fill_in_blank`` share a single definition of each output schema — and so those
schemas can be imported without pulling in the OpenAI client or pytest machinery.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_SYSTEM_PROMPT = """\
You are a Qiskit coding expert. Write clean, correct Python code using modern \
Qiskit 2.x APIs. Never use deprecated functions like execute(), BasicAer, \
IBMQ, or qiskit.Aer. Use qiskit.circuit.QuantumCircuit, qiskit_aer.AerSimulator, \
qiskit.primitives.StatevectorEstimator/StatevectorSampler, and \
qiskit.quantum_info.SparsePauliOp. Return only the Python code in a \
```python code block, no explanations."""


@dataclass
class CorrectionEntry:
    """One row in the correction hub — a failing attempt or a solved row.

    Failing attempts (``attempt_number >= 1``) are "fix this broken code"
    problems; ``gold`` later keeps the ones from eventually-solved problems.
    The safe solved row (``attempt_number == 0``, ``error_type == "none"``)
    carries ``correct_code`` for ``fill_in_blank`` to mutate.
    """

    prompt: str
    imports: str
    unittest: str
    template: str
    source_problem_idx: int
    attempt_number: int
    error_type: str
    error_message: str
    has_deprecated_apis: bool
    deprecated_api_warnings: str
    solved: bool
    original_prompt: str
    broken_code: str
    correct_code: str
    solved_at_turn: int
    turns_to_fix: int


@dataclass
class FillInBlankEntry:
    """One fill-in-the-blank variant: solved code with a targeted deletion."""

    prompt: str
    imports: str
    unittest: str
    template: str
    source_problem_idx: int
    deletion_type: str
    deleted_line: str
    correct_code: str
    original_prompt: str


@dataclass
class GenConfig:
    """Sampling / loop knobs for the correction Instruct-Validate-Repair loop."""

    model: str
    loop_budget: int = 5
    temperature: float = 0.6
    max_tokens: int = 1024
    num_retries: int = 6
    request_timeout: float = 90.0
    test_python: str | None = None
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
