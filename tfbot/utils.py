"""Utility helpers for TFBot."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Set, Tuple

import discord

logger = logging.getLogger("tfbot.utils")


def int_from_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid integer for %s=%s. Falling back to %s.", name, raw, default)
        return default


def float_from_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid float for %s=%s. Falling back to %s.", name, raw, default)
        return default


def path_from_env(name: str) -> Optional[Path]:
    value = os.getenv(name, "").strip()
    if not value:
        return None
    return Path(value).expanduser()


def parse_channel_ids(raw: str) -> Set[int]:
    ids: Set[int] = set()
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            ids.add(int(chunk))
        except ValueError:
            logger.warning("Ignoring invalid channel id %s", chunk)
    return ids


def normalize_pose_name(pose: Optional[str]) -> Optional[str]:
    if pose is None:
        return None
    stripped = pose.strip()
    if not stripped:
        return None
    return stripped.lower()


def is_admin(member: discord.abc.User) -> bool:
    if isinstance(member, discord.Member):
        if member.guild_permissions.administrator:
            return True
        roles: Iterable[discord.Role] = getattr(member, "roles", [])
        return any(role.name.lower() == "admin" for role in roles)
    return False


def member_profile_name(member: discord.Member) -> str:
    """Return the user's profile name, ignoring any server nickname."""
    global_name = getattr(member, "global_name", None)
    if isinstance(global_name, str) and global_name.strip():
        return global_name.strip()
    return member.name


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


__all__ = [
    "float_from_env",
    "int_from_env",
    "is_admin",
    "member_profile_name",
    "normalize_pose_name",
    "parse_channel_ids",
    "path_from_env",
    "utc_now",
]
