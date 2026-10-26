"""Endpoint resolution + mellea session factory for the correction IVR loop.

The correction stage runs mellea's Instruct-Validate-Repair loop over its litellm
backend — the exact engine that produced the v5/v6 datasets. ``resolve_endpoint``
reads the ``LITELLM_*`` env vars (or their ``VLLM_*`` aliases), with optional flag
overrides for the base URL and the extra auth-header name (RITS-style proxies).
The API key defaults to ``"EMPTY"`` when unset so an unauthenticated self-served
vLLM endpoint needs no dummy key.

``make_session_factory`` returns a zero-arg callable that builds a fresh mellea
``MelleaSession`` on each call. The correction stage runs one session per problem
(built via this factory) under a ``ThreadPoolExecutor`` and calls ``session.cleanup()``
when done — mirroring the legacy ``generate_correction_data.py``. A factory (rather
than a shared session) keeps each worker's session independent, since sessions are
not assumed thread-safe.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING

from .. import (
    DEFAULT_API_KEY,
    ENV_API_KEY_ALIASES,
    ENV_AUTH_HEADER_ALIASES,
    ENV_BASE_URL_ALIASES,
)
from ..models import GenConfig

if TYPE_CHECKING:
    from mellea import MelleaSession

log = logging.getLogger(__name__)


def _first_env(names: tuple[str, ...]) -> str | None:
    """Return the first non-empty value among the given env-var names."""
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def resolve_endpoint(
    base_url_arg: str | None = None,
    auth_header_arg: str | None = None,
) -> tuple[str, str, str | None]:
    """Resolve (base_url, api_key, auth_header) from flags + environment.

    Base URL is required (from --base-url or one of the accepted env vars) and the
    call exits with a clear message when it is missing. The API key falls back to
    ``DEFAULT_API_KEY`` ("EMPTY") when unset, so an unauthenticated vLLM server
    works with no dummy key; a real endpoint just sets one of the API-key vars.
    """
    base_url = base_url_arg or _first_env(ENV_BASE_URL_ALIASES)
    api_key = _first_env(ENV_API_KEY_ALIASES) or DEFAULT_API_KEY
    auth_header = auth_header_arg or _first_env(ENV_AUTH_HEADER_ALIASES)
    if not base_url:
        accepted = " / ".join(f"${n}" for n in ENV_BASE_URL_ALIASES)
        sys.exit(f"ERROR: set --base-url or one of: {accepted}")
    return base_url, api_key, auth_header


def make_session_factory(
    base_url: str,
    api_key: str,
    auth_header: str | None,
    cfg: GenConfig,
) -> Callable[[], MelleaSession]:
    """Return a zero-arg factory that builds a fresh mellea litellm session.

    Each call reproduces the legacy ``generate_correction_data.py`` setup exactly:
    the litellm backend, ``SimpleContext`` (no conversation history — every repair
    turn re-solves the original instruction), and the same ``model_options``. The
    plain (non-sentinel) ``num_retries`` / ``timeout`` keys pass straight through to
    ``litellm.acompletion()`` and retry transient endpoint drops
    (ServerDisconnectedError / ReadTimeout) at the HTTP layer *before* the error can
    reach mellea and abort the whole problem — this is what prevents
    disconnect-driven data loss.

    ``--model`` (``cfg.model``) is passed verbatim as the litellm ``model_id`` — the
    caller supplies a litellm-style id such as ``hosted_vllm/<served-name>``. The
    endpoint is routed via ``api_base`` in ``model_options`` (not the backend's
    ``base_url`` kwarg): mellea's litellm backend forwards unknown ``model_options``
    keys straight to ``litellm.acompletion()``, so ``"api_base": base_url`` produces
    the same call the old local ``extra_params["api_base"]`` patch did — which is why
    stock upstream mellea works with no source edit. When ``auth_header`` is set
    (RITS-style proxies), it is forwarded as an ``extra_headers`` entry; this stays
    inert when unset.
    """

    def factory() -> MelleaSession:
        from mellea import start_session
        from mellea.backends import ModelOption
        from mellea.stdlib.context import SimpleContext

        model_options: dict = {
            ModelOption.MAX_NEW_TOKENS: cfg.max_tokens,
            ModelOption.TEMPERATURE: cfg.temperature,
            ModelOption.SYSTEM_PROMPT: cfg.system_prompt,
            # Plain keys pass through to litellm.acompletion(); see docstring.
            # "api_base" routes to the self-served endpoint (replaces the old patch).
            "api_base": base_url,
            "num_retries": cfg.num_retries,
            "timeout": cfg.request_timeout,
            "api_key": api_key,
        }
        if auth_header:
            model_options["extra_headers"] = {auth_header: api_key}

        return start_session(
            backend_name="litellm",
            model_id=cfg.model,
            ctx=SimpleContext(),
            model_options=model_options,
        )

    return factory
