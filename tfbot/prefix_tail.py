"""Prefix command tail token counting (aligned with bot._split_command_tokens)."""

from __future__ import annotations

import shlex

__all__ = ["count_prefix_tail_tokens"]


def count_prefix_tail_tokens(raw: str) -> int:
    """Count whitespace-separated tokens using shlex (with split fallback on unbalanced quotes)."""
    text = (raw or "").strip()
    if not text:
        return 0
    try:
        return len([token for token in shlex.split(text) if token.strip()])
    except ValueError:
        return len([token for token in text.split() if token.strip()])
