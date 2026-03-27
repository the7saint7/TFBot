"""Bot-level timeout and ban lists (VN only). Persisted per-guild."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Set, Tuple

logger = logging.getLogger("tfbot.bot_moderation")

# File paths (set by configure())
_BANS_FILE: Optional[Path] = None
_TIMEOUTS_FILE: Optional[Path] = None

# In-memory state: bans as set of (guild_id, user_id)
_bans: Set[Tuple[int, int]] = set()
# Channel where ban was applied: (guild_id, user_id) -> channel_id (for delete+DM only in that channel)
_ban_channel_id: dict[Tuple[int, int], int] = {}

# Timeouts: (guild_id, user_id) -> {"expires_at": datetime, "strike_timestamps": [...], "channel_id": optional int}
_timeouts: dict = {}  # key: (guild_id, user_id), value: dict

# Incremental tiers (minutes). Index = number of strikes in last hour.
TIMEOUT_TIER_MINUTES = [2, 10, 30, 60, 120, 240, 1440]  # 2m, 10m, 30m, 1h, 2h, 4h, 24h
STRIKE_WINDOW_HOURS = 1.0


def configure(bans_file: Path, timeouts_file: Path) -> None:
    global _BANS_FILE, _TIMEOUTS_FILE
    _BANS_FILE = bans_file
    _TIMEOUTS_FILE = timeouts_file
    load_bans()
    load_timeouts()


def _bans_path() -> Path:
    if _BANS_FILE is None:
        raise RuntimeError("bot_moderation not configured. Call configure() first.")
    return _BANS_FILE


def _timeouts_path() -> Path:
    if _TIMEOUTS_FILE is None:
        raise RuntimeError("bot_moderation not configured. Call configure() first.")
    return _TIMEOUTS_FILE


def load_bans() -> None:
    path = _bans_path()
    global _bans, _ban_channel_id
    _bans = set()
    _ban_channel_id = {}
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for item in data:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    _bans.add((int(item[0]), int(item[1])))
        elif isinstance(data, dict) and "bans" in data:
            for item in data["bans"]:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    _bans.add((int(item[0]), int(item[1])))
            for k, cid in (data.get("ban_channel_ids") or {}).items():
                try:
                    gid, uid = k.split(":")
                    key = (int(gid), int(uid))
                    if key in _bans:
                        _ban_channel_id[key] = int(cid)
                except (ValueError, AttributeError):
                    pass
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error("Failed to load bans from %s: %s", path, exc)


def persist_bans() -> None:
    path = _bans_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "bans": [[g, u] for (g, u) in sorted(_bans)],
        "ban_channel_ids": {f"{g}:{u}": cid for (g, u), cid in sorted(_ban_channel_id.items()) if (g, u) in _bans},
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_timeouts() -> None:
    path = _timeouts_path()
    global _timeouts
    _timeouts = {}
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return
        for gk, guild_data in data.items():
            if not isinstance(guild_data, dict):
                continue
            try:
                gid = int(gk)
            except ValueError:
                continue
            for uk, entry in guild_data.items():
                if not isinstance(entry, dict):
                    continue
                try:
                    uid = int(uk)
                except ValueError:
                    continue
                exp = entry.get("expires_at")
                strikes = entry.get("strike_timestamps") or []
                if exp:
                    try:
                        expires_at = datetime.fromisoformat(str(exp).replace("Z", "+00:00"))
                    except ValueError:
                        continue
                else:
                    continue
                strike_list = []
                for s in strikes:
                    try:
                        strike_list.append(datetime.fromisoformat(str(s).replace("Z", "+00:00")))
                    except ValueError:
                        pass
                channel_id = entry.get("channel_id")
                _timeouts[(gid, uid)] = {
                    "expires_at": expires_at,
                    "strike_timestamps": strike_list,
                    "channel_id": int(channel_id) if channel_id is not None else None,
                }
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error("Failed to load timeouts from %s: %s", path, exc)


def persist_timeouts() -> None:
    path = _timeouts_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    for (gid, uid), entry in _timeouts.items():
        sg = str(gid)
        su = str(uid)
        if sg not in data:
            data[sg] = {}
        out = {
            "expires_at": entry["expires_at"].isoformat(),
            "strike_timestamps": [d.isoformat() for d in entry["strike_timestamps"]],
        }
        if entry.get("channel_id") is not None:
            out["channel_id"] = entry["channel_id"]
        data[sg][su] = out
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def is_banned(guild_id: int, user_id: int) -> bool:
    return (guild_id, user_id) in _bans


def get_ban_channel_id(guild_id: int, user_id: int) -> Optional[int]:
    """Channel where ban was applied; delete+DM only in that channel. None if not stored (legacy)."""
    return _ban_channel_id.get((guild_id, user_id))


def add_ban(guild_id: int, user_id: int, channel_id: Optional[int] = None) -> None:
    key = (guild_id, user_id)
    _bans.add(key)
    if channel_id is not None:
        _ban_channel_id[key] = channel_id
    else:
        _ban_channel_id.pop(key, None)
    persist_bans()


def remove_ban(guild_id: int, user_id: int) -> None:
    key = (guild_id, user_id)
    _bans.discard(key)
    _ban_channel_id.pop(key, None)
    persist_bans()


def is_timed_out(guild_id: int, user_id: int, now: Optional[datetime] = None) -> bool:
    if now is None:
        from .utils import utc_now
        now = utc_now()
    key = (guild_id, user_id)
    if key not in _timeouts:
        return False
    return now < _timeouts[key]["expires_at"]


def get_timeout_remaining(guild_id: int, user_id: int, now: Optional[datetime] = None) -> Optional[timedelta]:
    """Return remaining time until timeout expires, or None if not timed out."""
    if now is None:
        from .utils import utc_now
        now = utc_now()
    key = (guild_id, user_id)
    if key not in _timeouts:
        return None
    exp = _timeouts[key]["expires_at"]
    if now >= exp:
        return None
    return exp - now


def format_remaining(remaining: timedelta) -> str:
    """Human-readable remaining time, e.g. '2 mins', '1 hour 5 mins'."""
    total_seconds = max(0, int(remaining.total_seconds()))
    if total_seconds < 60:
        return "1 min" if total_seconds else "0 mins"
    mins, secs = divmod(total_seconds, 60)
    hours, mins = divmod(mins, 60)
    days, hours = divmod(hours, 24)
    parts = []
    if days:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    if hours:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if mins or (not parts and not days):
        parts.append(f"{mins} min{'s' if mins != 1 else ''}")
    return " ".join(parts)


def get_timeout_remaining_str(guild_id: int, user_id: int, now: Optional[datetime] = None) -> Optional[str]:
    """Human-readable remaining timeout, or None if not timed out."""
    rem = get_timeout_remaining(guild_id, user_id, now)
    if rem is None:
        return None
    return format_remaining(rem)


def _count_strikes_in_window(guild_id: int, user_id: int, now: datetime) -> int:
    key = (guild_id, user_id)
    if key not in _timeouts:
        return 0
    window_start = now - timedelta(hours=STRIKE_WINDOW_HOURS)
    return sum(1 for t in _timeouts[key]["strike_timestamps"] if t >= window_start)


def get_timeout_channel_id(guild_id: int, user_id: int) -> Optional[int]:
    """Channel where timeout was applied; delete+DM only in that channel. None if not stored (legacy)."""
    key = (guild_id, user_id)
    if key not in _timeouts:
        return None
    return _timeouts[key].get("channel_id")


def add_timeout(
    guild_id: int,
    user_id: int,
    duration_minutes: Optional[int] = None,
    now: Optional[datetime] = None,
    channel_id: Optional[int] = None,
) -> int:
    """
    Add or extend timeout. If duration_minutes is None, use incremental tier from recent strikes.
    Returns the duration applied in minutes.
    """
    if now is None:
        from .utils import utc_now
        now = utc_now()
    key = (guild_id, user_id)
    if duration_minutes is not None:
        minutes = max(1, duration_minutes)
    else:
        strike_count = _count_strikes_in_window(guild_id, user_id, now)
        tier_index = min(strike_count, len(TIMEOUT_TIER_MINUTES) - 1)
        minutes = TIMEOUT_TIER_MINUTES[tier_index]
    expires_at = now + timedelta(minutes=minutes)
    strike_list = _timeouts.get(key, {}).get("strike_timestamps", [])
    # Prune to last 24h
    cutoff = now - timedelta(hours=24)
    strike_list = [t for t in strike_list if t >= cutoff]
    strike_list.append(now)
    _timeouts[key] = {
        "expires_at": expires_at,
        "strike_timestamps": strike_list,
        "channel_id": channel_id,
    }
    persist_timeouts()
    return minutes


def clear_timeout(guild_id: int, user_id: int) -> None:
    key = (guild_id, user_id)
    _timeouts.pop(key, None)
    persist_timeouts()


def get_timeout_duration_label(minutes: int) -> str:
    """Human-readable duration for announcements, e.g. '2 mins', '1 hour'."""
    if minutes < 60:
        return f"{minutes} min{'s' if minutes != 1 else ''}"
    hours, mins = divmod(minutes, 60)
    if mins == 0:
        return f"{hours} hour{'s' if hours != 1 else ''}"
    return f"{hours} hour{'s' if hours != 1 else ''} {mins} min{'s' if mins != 1 else ''}"


__all__ = [
    "configure",
    "load_bans",
    "load_timeouts",
    "is_banned",
    "get_ban_channel_id",
    "add_ban",
    "remove_ban",
    "is_timed_out",
    "get_timeout_channel_id",
    "get_timeout_remaining",
    "get_timeout_remaining_str",
    "format_remaining",
    "add_timeout",
    "clear_timeout",
    "get_timeout_duration_label",
    "TIMEOUT_TIER_MINUTES",
    "STRIKE_WINDOW_HOURS",
]
