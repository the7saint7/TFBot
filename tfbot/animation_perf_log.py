"""Dedicated animation performance logging (daily file, low-overhead key/value lines)."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_LOGGER = logging.getLogger("tfbot.animation_perf")
_HANDLER: Optional[logging.FileHandler] = None
_ENABLED = False
_PATH: Optional[Path] = None


def _is_enabled_from_env() -> bool:
    raw = os.getenv("TFBOT_ANIMATION_PERF_LOG", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def resolve_animation_perf_log_path(base_dir: Path) -> Path:
    override = os.getenv("TFBOT_ANIMATION_PERF_LOG_PATH", "").strip()
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if override:
        p = Path(override)
        if not p.is_absolute():
            p = (base_dir / p).resolve()
        if p.suffix.lower() == ".log":
            return (p.parent / f"{p.stem}_{day}{p.suffix}").resolve()
        return (p / f"animation_perf_{day}.log").resolve()
    return (base_dir / "vn_states" / f"animation_perf_{day}.log").resolve()


def install(base_dir: Path) -> None:
    global _HANDLER, _ENABLED, _PATH
    if _HANDLER is not None:
        return
    _ENABLED = _is_enabled_from_env()
    if not _ENABLED:
        return
    path = resolve_animation_perf_log_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))

    _LOGGER.setLevel(logging.INFO)
    _LOGGER.propagate = False
    _LOGGER.addHandler(handler)

    _HANDLER = handler
    _PATH = path


def shutdown() -> None:
    global _HANDLER, _PATH
    if _HANDLER is None:
        return
    try:
        _LOGGER.removeHandler(_HANDLER)
    except ValueError:
        pass
    try:
        _HANDLER.close()
    except Exception:
        pass
    _HANDLER = None
    _PATH = None


def current_path() -> Optional[Path]:
    return _PATH


def log_event(event: str, /, **fields: Any) -> None:
    if not _ENABLED or _HANDLER is None:
        return
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    parts = [f"ts={ts}", f"event={event}"]
    for key, value in fields.items():
        if value is None:
            continue
        text = str(value).replace("\n", "\\n")
        parts.append(f"{key}={text}")
    _LOGGER.info(" ".join(parts))

