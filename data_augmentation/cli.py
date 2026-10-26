"""Command-line interface for the data-augmentation pipeline.

One command turns a source problem JSONL into the two GRPO deliverables:

    uv run --extra all --with pytest python -m data_augmentation all \
        --input problems.jsonl --output-dir out/ --model <served-model>

Subcommands:
  all             correction (LLM) → gold + fill-in-blank, all three artifacts.
  correction      run only the Instruct-Validate-Repair LLM stage (the hub file).
  gold            derive gold_corrections from an existing correction hub file.
  fill-in-blank   derive validated fill-in-blank data from a correction hub file.

Endpoint auth via env: LITELLM_BASE_URL, LITELLM_API_KEY, optional LITELLM_AUTH_HEADER.

Because gold and fill-in-blank both read the intermediate
correction_problems_<tag>.jsonl, you can re-run them with different options
(e.g. fill-in-blank --types / --max-per-type) without repeating the LLM stage.
"""

from __future__ import annotations

import argparse
import logging
import os

from . import ENV_AUTH_HEADER, ENV_BASE_URL
from .core.llm_client import make_session_factory, resolve_endpoint
from .models import DEFAULT_SYSTEM_PROMPT, GenConfig
from .stages import correction, fill_in_blank, gold


def _default_tag(input_path: str) -> str:
    stem = os.path.splitext(os.path.basename(input_path))[0]
    return stem or "dataset"


def add_correction_args(parser: argparse.ArgumentParser) -> None:
    """Register correction/endpoint args (shared with the `all` subcommand)."""
    parser.add_argument(
        "--model",
        required=True,
        help="litellm-style model id passed to mellea, e.g. hosted_vllm/<served-name> "
        "for a self-served vLLM endpoint",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help=f"OpenAI-compatible base URL, e.g. your vLLM endpoint "
        f"(default: ${ENV_BASE_URL} or $VLLM_BASE_URL)",
    )
    parser.add_argument(
        "--auth-header",
        default=None,
        help=f"Optional extra header name carrying the key (default: ${ENV_AUTH_HEADER} or "
        "$VLLM_AUTH_HEADER). When unset, standard Authorization: Bearer is used.",
    )
    parser.add_argument("--loop-budget", type=int, default=5, help="Max IVR attempts (default: 5)")
    parser.add_argument(
        "--max-workers", type=int, default=4, help="Concurrent problems (default: 4)"
    )
    parser.add_argument(
        "--template-allowlist",
        "-t",
        default=None,
        help="Path to template filter file (one template per line)",
    )
    parser.add_argument(
        "--checkpoint-dir", default=None, help="Directory for incremental checkpoints (resume)"
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=50, help="Checkpoint every N problems (default: 50)"
    )
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT, help="System prompt")
    parser.add_argument(
        "--temperature", type=float, default=0.6, help="Sampling temp (default: 0.6)"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=1024, help="Max new tokens (default: 1024)"
    )
    parser.add_argument(
        "--num-retries",
        type=int,
        default=6,
        help="Per-generation API retries on transient errors (default: 6)",
    )
    parser.add_argument(
        "--request-timeout", type=float, default=90.0, help="Per-request timeout sec (default: 90)"
    )
    parser.add_argument(
        "--test-python",
        default=None,
        help="Interpreter used to run validation pytest (default: this interpreter)",
    )


def config_from_args(args: argparse.Namespace) -> GenConfig:
    return GenConfig(
        model=args.model,
        loop_budget=args.loop_budget,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        num_retries=args.num_retries,
        request_timeout=args.request_timeout,
        test_python=args.test_python,
        system_prompt=args.system_prompt,
    )


def _add_fib_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--types",
        default="import,call,combined",
        help="Comma-separated deletion types (default: import,call,combined)",
    )
    parser.add_argument(
        "--max-per-type",
        type=int,
        default=2,
        help="Max variants per (problem, deletion_type) after dedup (default: 2)",
    )


def _cmd_correction(args: argparse.Namespace) -> None:
    base_url, api_key, auth_header = resolve_endpoint(args.base_url, args.auth_header)
    print(f"Endpoint: {base_url} (auth header: {auth_header or 'Authorization: Bearer'})")
    cfg = config_from_args(args)
    session_factory = make_session_factory(base_url, api_key, auth_header, cfg)
    correction.run_pipeline(
        input_path=args.input,
        output_path=args.output,
        session_factory=session_factory,
        cfg=cfg,
        max_workers=args.max_workers,
        template_allowlist=args.template_allowlist,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
    )


def _cmd_gold(args: argparse.Namespace) -> None:
    gold.extract_gold(args.input, args.output)


def _cmd_fill_in_blank(args: argparse.Namespace) -> None:
    fill_in_blank.run_pipeline(
        input_path=args.input,
        output_path=args.output,
        types=args.types,
        max_per_type=args.max_per_type,
        validate=args.validate,
        max_workers=args.max_workers,
        test_python=args.test_python,
    )


def _cmd_all(args: argparse.Namespace) -> None:
    tag = args.tag or _default_tag(args.input)
    os.makedirs(args.output_dir, exist_ok=True)
    hub = os.path.join(args.output_dir, f"correction_problems_{tag}.jsonl")
    gold_out = os.path.join(args.output_dir, f"gold_corrections_{tag}.jsonl")
    fib_out = os.path.join(args.output_dir, f"fill_in_blank_{tag}.jsonl")

    base_url, api_key, auth_header = resolve_endpoint(args.base_url, args.auth_header)
    print(f"Endpoint: {base_url} (auth header: {auth_header or 'Authorization: Bearer'})")
    cfg = config_from_args(args)
    session_factory = make_session_factory(base_url, api_key, auth_header, cfg)

    print(f"\n=== [1/3] correction → {hub} ===")
    checkpoint_dir = args.checkpoint_dir or os.path.join(args.output_dir, f"ckpt_{tag}")
    correction.run_pipeline(
        input_path=args.input,
        output_path=hub,
        session_factory=session_factory,
        cfg=cfg,
        max_workers=args.max_workers,
        template_allowlist=args.template_allowlist,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
    )

    print(f"\n=== [2/3] gold → {gold_out} ===")
    gold.extract_gold(hub, gold_out)

    print(f"\n=== [3/3] fill-in-blank → {fib_out} ===")
    fill_in_blank.run_pipeline(
        input_path=hub,
        output_path=fib_out,
        types=args.types,
        max_per_type=args.max_per_type,
        validate=not args.no_validate,
        max_workers=args.fib_max_workers,
        test_python=args.test_python,
    )

    print("\nDone. Deliverables:")
    print(f"  hub:            {hub}")
    print(f"  gold:           {gold_out}")
    print(f"  fill-in-blank:  {fib_out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m data_augmentation", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    # all
    p_all = sub.add_parser("all", help="correction → gold + fill-in-blank")
    p_all.add_argument("--input", "-i", required=True, help="Source prompt-unittest JSONL")
    p_all.add_argument("--output-dir", "-o", required=True, help="Directory for all artifacts")
    p_all.add_argument("--tag", default=None, help="Filename tag (default: input stem)")
    add_correction_args(p_all)
    _add_fib_options(p_all)
    p_all.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip pytest validation of fill-in-blank deletions (default: validate)",
    )
    p_all.add_argument(
        "--fib-max-workers",
        type=int,
        default=None,
        help="Threads for fill-in-blank validation (default: same as --max-workers)",
    )
    p_all.set_defaults(func=_cmd_all)

    # correction
    p_corr = sub.add_parser("correction", help="run only the IVR LLM stage")
    p_corr.add_argument("--input", "-i", required=True, help="Source prompt-unittest JSONL")
    p_corr.add_argument("--output", "-o", required=True, help="Output correction JSONL")
    add_correction_args(p_corr)
    p_corr.set_defaults(func=_cmd_correction)

    # gold
    p_gold = sub.add_parser("gold", help="derive gold corrections from a hub file")
    p_gold.add_argument("--input", "-i", required=True, help="correction_problems_<tag>.jsonl")
    p_gold.add_argument("--output", "-o", required=True, help="Output gold JSONL")
    p_gold.set_defaults(func=_cmd_gold)

    # fill-in-blank
    p_fib = sub.add_parser("fill-in-blank", help="derive fill-in-blank data from a hub file")
    p_fib.add_argument("--input", "-i", required=True, help="correction_problems_<tag>.jsonl")
    p_fib.add_argument("--output", "-o", required=True, help="Output fill-in-blank JSONL")
    _add_fib_options(p_fib)
    p_fib.add_argument(
        "--validate",
        action="store_true",
        help="Run pytest to verify each deletion causes test failure (slow)",
    )
    p_fib.add_argument(
        "--max-workers", type=int, default=1, help="Threads for parallel validation (default: 1)"
    )
    p_fib.add_argument(
        "--test-python",
        default=None,
        help="Interpreter used to run validation pytest (default: this interpreter)",
    )
    p_fib.set_defaults(func=_cmd_fill_in_blank)

    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "all" and args.fib_max_workers is None:
        args.fib_max_workers = args.max_workers
    args.func(args)
