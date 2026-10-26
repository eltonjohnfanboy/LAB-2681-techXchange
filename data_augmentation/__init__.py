"""Data augmentation for synthetic Qiskit problems: correction + fill-in-blank.

Turns a source problem JSONL (``{prompt, imports, unittest, template}`` — exactly
what ``generate_prompt.py`` emits) into two GRPO training deliverables:

  * ``gold_corrections_<tag>.jsonl`` — failing model attempts from problems that
    were eventually solved (highest-value "fix this broken code" data).
  * ``fill_in_blank_<tag>.jsonl`` — validated import/call/combined deletions of
    solved code (the model must restore the missing line so tests pass).

Both are derived from an intermediate ``correction_problems_<tag>.jsonl`` hub
produced by mellea's Instruct-Validate-Repair loop over its litellm backend — the
exact engine that produced the v5/v6 datasets. mellea is pinned to an exact upstream
commit via the repo ``pyproject.toml`` ``[tool.uv.sources]`` (stock source, no patch —
the endpoint is routed through an ``api_base`` model option); the loop runs
``RepairTemplateStrategy`` with a ``SimpleContext`` so each repair turn re-solves the
original instruction from the raw pytest failure.

Layout:
  * ``models``  — the row/config dataclasses shared by every stage.
  * ``core``    — shared plumbing: the endpoint client and pytest validation.
  * ``stages``  — the three data generators (correction, gold, fill_in_blank).
  * ``cli``     — the command-line interface (``python -m data_augmentation``).

Run as a module: ``python -m data_augmentation all --input problems.jsonl ...``.
"""

from __future__ import annotations

# Endpoint config is read from the environment. LITELLM_* is the primary name
# (specific to this package — prompt_postprocessing has its own .env-based config);
# VLLM_* is accepted as a vLLM-native alias so a self-served vLLM endpoint needs no LiteLLM-flavored vars.
# NB: OPENAI_BASE_URL/OPENAI_API_KEY are deliberately NOT accepted — per the repo's
# AGENTS.md they usually point at a different host and 401 here.
ENV_BASE_URL = "LITELLM_BASE_URL"
ENV_API_KEY = "LITELLM_API_KEY"
ENV_AUTH_HEADER = "LITELLM_AUTH_HEADER"

# Accepted env-var names in priority order (primary first, then aliases).
ENV_BASE_URL_ALIASES = (ENV_BASE_URL, "VLLM_BASE_URL")
ENV_API_KEY_ALIASES = (ENV_API_KEY, "VLLM_API_KEY")
ENV_AUTH_HEADER_ALIASES = (ENV_AUTH_HEADER, "VLLM_AUTH_HEADER")

# vLLM serves an OpenAI-compatible endpoint that ignores auth unless launched with
# --api-key. When no key is set anywhere, fall back to this placeholder (vLLM's own
# convention) so an unauthenticated local server needs no dummy key.
DEFAULT_API_KEY = "EMPTY"

# Leading keys every output row shares, in this order (Harbor-task compatible).
CANONICAL_KEYS = ["prompt", "imports", "unittest", "template"]
