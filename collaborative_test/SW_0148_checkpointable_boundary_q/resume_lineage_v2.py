"""Strict normalization for SW0148's verified source-parity resume marker."""
from __future__ import annotations

import copy


PARITY_KEY = "initial_source_Q_parity"


def current_resume_bindings(current_base: dict) -> dict:
    """Copy fresh bindings and drop only the expected unset parity marker."""
    if not isinstance(current_base, dict):
        raise ValueError("current resume lineage bindings must be a dictionary")
    current = copy.deepcopy(current_base)
    if PARITY_KEY in current:
        if current[PARITY_KEY] is not None:
            raise ValueError("current resume lineage must use the unset parity placeholder")
        current.pop(PARITY_KEY)
    return current


def normalized_resume_bindings(current_base: dict, saved_bindings: dict) -> dict:
    """Validate saved lineage, allowing only the current None parity placeholder.

    Fresh-run bindings contain ``initial_source_Q_parity: None`` because the
    source-Q parity check is performed after construction. A committed resume
    must carry the original successful parity record. This helper compares
    deep copies and never mutates either caller-owned mapping.
    """
    if not isinstance(current_base, dict) or not isinstance(saved_bindings, dict):
        raise ValueError("resume lineage bindings must be dictionaries")

    current = current_resume_bindings(current_base)
    saved = copy.deepcopy(saved_bindings)
    saved_parity = saved.pop(PARITY_KEY, None)

    if not isinstance(saved_parity, dict) or saved_parity.get("exact") is not True:
        raise ValueError("checkpoint lacks an exact successful initial source-Q parity record")
    if saved != current:
        raise ValueError("checkpoint source/science lineage differs from current bindings")
    return {**current, PARITY_KEY: saved_parity}
