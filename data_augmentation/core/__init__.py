"""Shared plumbing for the pipeline stages.

* ``llm_client`` — the OpenAI-compatible endpoint client + retry/backoff.
* ``validation`` — pytest execution and the ``__solution_source__`` stub fix.
* ``text``       — the assertion-leak filter shared by the correction/gold prompts.
* ``rows``       — canonical output-key ordering shared by every stage.
"""
