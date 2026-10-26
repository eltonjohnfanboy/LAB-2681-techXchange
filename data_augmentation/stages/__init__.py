"""The three data-generation stages.

  * ``correction``    — the LLM Instruct-Validate-Repair loop → correction hub.
  * ``gold``          — hub → gold corrections (failing attempts, later solved).
  * ``fill_in_blank`` — hub → validated import/call/combined deletions.

gold and fill_in_blank both read the correction hub, so they can be re-run with
different options without repeating the (expensive) LLM stage.
"""
