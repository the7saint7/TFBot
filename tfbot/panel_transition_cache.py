"""Disk cache helpers for expensive transition renders."""

from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
from typing import Optional

import discord

BASE_DIR = Path(__file__).resolve().parent.parent
_CACHE_DIR_SETTING = os.environ.get("TFBOT_BG_TRANSITION_CACHE_DIR", "vn_cache/bg_transition_cache").strip()
_CACHE_DIR = (BASE_DIR / _CACHE_DIR_SETTING).resolve() if _CACHE_DIR_SETTING else (BASE_DIR / "vn_cache" / "bg_transition_cache")
_CACHE_MAX_BYTES = max(64 * 1024 * 1024, int(os.environ.get("TFBOT_BG_TRANSITION_CACHE_MAX_BYTES", str(512 * 1024 * 1024))))


def _ensure_cache_dir() -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


def build_bg_transition_cache_key(
    *,
    state_user_id: Optional[int],
    state_character_name: str,
    state_avatar_path: str,
    state_folder: str,
    old_background_path: Optional[Path],
    new_background_path: Optional[Path],
    selection_scope: Optional[str],
) -> str:
    def _path_fingerprint(path: Optional[Path]) -> str:
        if path is None:
            return "none"
        p = path.resolve()
        try:
            st = p.stat()
            return f"{p.as_posix()}::{st.st_mtime_ns}::{st.st_size}"
        except OSError:
            return f"{p.as_posix()}::missing"

    raw = "|".join(
        [
            str(state_user_id or 0),
            state_character_name or "",
            state_avatar_path or "",
            state_folder or "",
            _path_fingerprint(old_background_path),
            _path_fingerprint(new_background_path),
            selection_scope or "",
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get_cached_bg_transition_file(cache_key: str, filename: str) -> Optional[discord.File]:
    cache_dir = _ensure_cache_dir()
    suffix = Path(filename).suffix or ".webp"
    cache_path = cache_dir / f"{cache_key}{suffix}"
    if not cache_path.exists():
        return None
    try:
        payload = cache_path.read_bytes()
    except OSError:
        return None
    return discord.File(io.BytesIO(payload), filename=filename)


def put_cached_bg_transition_file(cache_key: str, filename: str, payload: bytes) -> None:
    if not payload:
        return
    cache_dir = _ensure_cache_dir()
    suffix = Path(filename).suffix or ".webp"
    target_path = cache_dir / f"{cache_key}{suffix}"
    temp_path = cache_dir / f"{cache_key}{suffix}.tmp"
    try:
        temp_path.write_bytes(payload)
        temp_path.replace(target_path)
    except OSError:
        return
    _evict_bg_cache(cache_dir, max_bytes=_CACHE_MAX_BYTES)


def _evict_bg_cache(cache_dir: Path, *, max_bytes: int) -> None:
    try:
        entries = [p for p in cache_dir.glob("*") if p.is_file() and not p.name.endswith(".tmp")]
    except OSError:
        return
    total = 0
    sized_entries = []
    for entry in entries:
        try:
            st = entry.stat()
        except OSError:
            continue
        total += st.st_size
        sized_entries.append((entry, st.st_mtime_ns, st.st_size))
    if total <= max_bytes:
        return
    sized_entries.sort(key=lambda item: item[1])  # oldest first
    for entry, _, size in sized_entries:
        try:
            entry.unlink(missing_ok=True)
            total -= size
        except OSError:
            continue
        if total <= max_bytes:
            break
