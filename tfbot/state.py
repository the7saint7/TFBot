"""State management for active transformations and persistence."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

from .models import TransformationState, TransformKey
logger = logging.getLogger("tfbot.state")

TF_STATE_FILE: Optional[Path] = None
TF_STATS_FILE: Optional[Path] = None
TF_REROLL_FILE: Optional[Path] = None

active_transformations: Dict[TransformKey, TransformationState] = {}
revert_tasks: Dict[TransformKey, asyncio.Task] = {}
tf_stats: Dict[str, Dict[str, Dict[str, object]]] = {}
reroll_cooldowns: Dict[str, Dict[str, str]] = {}
STATE_RESTORED = False


def configure_state(*, state_file: Path, stats_file: Path, reroll_file: Optional[Path] = None) -> None:
    global TF_STATE_FILE, TF_STATS_FILE, TF_REROLL_FILE
    TF_STATE_FILE = state_file
    TF_STATS_FILE = stats_file
    TF_REROLL_FILE = reroll_file


def state_key(guild_id: int, user_id: int) -> TransformKey:
    return (guild_id, user_id)


def find_active_transformation(
    user_id: int,
    guild_id: Optional[int] = None,
) -> Optional[TransformationState]:
    if guild_id is not None:
        state = active_transformations.get(state_key(guild_id, user_id))
        if state:
            return state
    for state in active_transformations.values():
        if state.user_id == user_id and (guild_id is None or state.guild_id == guild_id):
            return state
    return None


def serialize_state(state: TransformationState) -> Dict[str, object]:
    return {
        "user_id": state.user_id,
        "guild_id": state.guild_id,
        "character_name": state.character_name,
        "character_folder": state.character_folder,
        "character_avatar_path": state.character_avatar_path,
        "character_message": state.character_message,
        "original_nick": state.original_nick,
        "original_display_name": state.original_display_name,
        "started_at": state.started_at.isoformat(),
        "expires_at": state.expires_at.isoformat(),
        "duration_label": state.duration_label,
        "avatar_applied": state.avatar_applied,
        "is_inanimate": state.is_inanimate,
        "inanimate_responses": list(state.inanimate_responses),
        "form_owner_user_id": state.form_owner_user_id,
        "identity_display_name": state.identity_display_name,
        "is_pillow": state.is_pillow,
    }


def deserialize_state(payload: Dict[str, object]) -> TransformationState:
    from datetime import datetime

    return TransformationState(
        user_id=int(payload["user_id"]),
        guild_id=int(payload["guild_id"]),
        character_name=str(payload["character_name"]),
        character_folder=str(payload.get("character_folder") or "") or None,
        character_avatar_path=str(
            payload.get("character_avatar_path") or payload.get("character_avatar_url", "")
        ),
        character_message=str(payload.get("character_message", "")),
        original_nick=payload.get("original_nick"),
        original_display_name=str(payload.get("original_display_name", "") or ""),
        started_at=datetime.fromisoformat(str(payload["started_at"])),
        expires_at=datetime.fromisoformat(str(payload["expires_at"])),
        duration_label=str(payload["duration_label"]),
        avatar_applied=bool(payload.get("avatar_applied", False)),
        is_inanimate=bool(payload.get("is_inanimate", False)),
        inanimate_responses=tuple(payload.get("inanimate_responses", ())),
        form_owner_user_id=payload.get("form_owner_user_id"),
        identity_display_name=payload.get("identity_display_name"),
        is_pillow=bool(payload.get("is_pillow", False)),
    )


def persist_states() -> None:
    if TF_STATE_FILE is None:
        raise RuntimeError("State file not configured. Call configure_state first.")
    TF_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = [serialize_state(state) for state in active_transformations.values()]
    TF_STATE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_states_from_disk() -> Sequence[TransformationState]:
    if TF_STATE_FILE is None or not TF_STATE_FILE.exists():
        return []
    try:
        payload = json.loads(TF_STATE_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse %s: %s", TF_STATE_FILE, exc)
        return []
    states: list[TransformationState] = []
    for entry in payload:
        try:
            states.append(deserialize_state(entry))
        except (KeyError, ValueError) as exc:
            logger.warning("Skipping malformed persisted state: %s", exc)
    return states


def load_stats_from_disk() -> Dict[str, Dict[str, Dict[str, object]]]:
    if TF_STATS_FILE is None or not TF_STATS_FILE.exists():
        return {}
    try:
        data = json.loads(TF_STATS_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse %s: %s", TF_STATS_FILE, exc)
    return {}


def persist_stats() -> None:
    if TF_STATS_FILE is None:
        raise RuntimeError("Stats file not configured. Call configure_state first.")
    TF_STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
    TF_STATS_FILE.write_text(json.dumps(tf_stats, indent=2), encoding="utf-8")


def increment_tf_stats(guild_id: int, user_id: int, character_name: str) -> None:
    guild_stats = tf_stats.setdefault(str(guild_id), {})
    user_stats = guild_stats.setdefault(str(user_id), {"total": 0, "characters": {}})
    total = int(user_stats.get("total", 0)) + 1
    user_stats["total"] = total
    char_stats = user_stats.setdefault("characters", {})
    char_stats[character_name] = int(char_stats.get(character_name, 0)) + 1
    persist_stats()


def load_reroll_cooldowns_from_disk() -> Dict[str, Dict[str, str]]:
    if TF_REROLL_FILE is None or not TF_REROLL_FILE.exists():
        return {}
    try:
        data = json.loads(TF_REROLL_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse %s: %s", TF_REROLL_FILE, exc)
    return {}


def persist_reroll_cooldowns() -> None:
    if TF_REROLL_FILE is None:
        raise RuntimeError("Reroll file not configured. Call configure_state first.")
    TF_REROLL_FILE.parent.mkdir(parents=True, exist_ok=True)
    TF_REROLL_FILE.write_text(json.dumps(reroll_cooldowns, indent=2), encoding="utf-8")


def get_last_reroll_timestamp(guild_id: int, user_id: int) -> Optional[datetime]:
    guild_data = reroll_cooldowns.get(str(guild_id))
    if not isinstance(guild_data, dict):
        return None
    raw_value = guild_data.get(str(user_id))
    if not raw_value:
        return None
    try:
        return datetime.fromisoformat(str(raw_value))
    except ValueError:
        return None


def record_reroll_timestamp(guild_id: int, user_id: int, when: datetime) -> None:
    guild_data = reroll_cooldowns.setdefault(str(guild_id), {})
    guild_data[str(user_id)] = when.isoformat()
    persist_reroll_cooldowns()


__all__ = [
    "STATE_RESTORED",
    "active_transformations",
    "configure_state",
    "deserialize_state",
    "find_active_transformation",
    "increment_tf_stats",
    "load_reroll_cooldowns_from_disk",
    "load_states_from_disk",
    "load_stats_from_disk",
    "persist_states",
    "persist_stats",
    "persist_reroll_cooldowns",
    "record_reroll_timestamp",
    "get_last_reroll_timestamp",
    "revert_tasks",
    "serialize_state",
    "state_key",
    "tf_stats",
    "reroll_cooldowns",
]
