"""Clone overlay display helpers (frozen-at-clone labels)."""

from __future__ import annotations

from typing import Mapping, Optional


def clone_copied_source_display_name(clone_record: Mapping[str, object]) -> Optional[str]:
    """Name of the character copied at clone time (for VN 'Clone of …'), not the source user's current form."""
    snapshot = clone_record.get("cloned_visual_snapshot")
    if not isinstance(snapshot, Mapping):
        return None
    for key in ("character_name", "identity_display_name"):
        raw = snapshot.get(key)
        if raw is None:
            continue
        label = str(raw).strip()
        if label:
            return label
    return None
