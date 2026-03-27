"""Append-only local action log for command debugging."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Optional

_ACTION_LOG_LOCK = Lock()


def resolve_action_log_path(base_dir: Optional[Path] = None) -> Path:
    if base_dir is None:
        base_dir = Path(__file__).resolve().parents[1]
    override = os.getenv("TFBOT_ACTION_LOG_FILE", "vn_states/command_actions.jsonl").strip()
    path = Path(override) if override else Path("vn_states/command_actions.jsonl")
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _safe_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def record_action_event(
    *,
    status: str,
    source: str,
    command_display: Optional[str] = None,
    invocation_text: Optional[str] = None,
    invocation_id: Optional[str] = None,
    actor: Any = None,
    channel: Any = None,
    guild: Any = None,
    error: Optional[BaseException] = None,
) -> Path:
    payload = {
        "at": datetime.now(timezone.utc).isoformat(),
        "status": _safe_text(status),
        "source": _safe_text(source),
        "command": _safe_text(command_display),
        "invocation_text": _safe_text(invocation_text),
        "invocation_id": _safe_text(invocation_id),
        "actor_id": getattr(actor, "id", None),
        "actor_name": _safe_text(getattr(actor, "name", None)),
        "actor_display_name": _safe_text(getattr(actor, "display_name", None)),
        "channel_id": getattr(channel, "id", None),
        "channel_name": _safe_text(getattr(channel, "name", None)),
        "channel_type": type(channel).__name__ if channel is not None else None,
        "guild_id": getattr(guild, "id", None),
        "guild_name": _safe_text(getattr(guild, "name", None)),
    }
    if error is not None:
        payload["error_type"] = type(error).__name__
        payload["error_message"] = _safe_text(error)

    path = resolve_action_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    with _ACTION_LOG_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.write("\n")
    return path


__all__ = ["record_action_event", "resolve_action_log_path"]
