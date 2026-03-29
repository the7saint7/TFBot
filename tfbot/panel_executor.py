"""Bounded thread pools for synchronous panel rendering (off asyncio loop).

Two pools reduce head-of-line blocking:
- VN pool for lighter PNG panel work.
- GIF pool for heavier transition/multi-frame work.

Each pool has a bounded worker count and logs queue/render timings for tuning.
"""

from __future__ import annotations

import functools
import logging
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Deque, Dict, TypeVar
from .animation_perf_log import log_event as log_animation_perf_event

R = TypeVar("R")

logger = logging.getLogger("tfbot.panel_executor")

_EXECUTOR_VN: ThreadPoolExecutor | None = None
_EXECUTOR_GIF: ThreadPoolExecutor | None = None
_METRICS_LOCK = threading.Lock()
_QUEUE_MS: Dict[str, Deque[float]] = {
    "vn": deque(maxlen=200),
    "gif": deque(maxlen=200),
}
_RENDER_MS: Dict[str, Deque[float]] = {
    "vn": deque(maxlen=200),
    "gif": deque(maxlen=200),
}
_TOTAL_MS: Dict[str, Deque[float]] = {
    "vn": deque(maxlen=200),
    "gif": deque(maxlen=200),
}
_COUNT: Dict[str, int] = {"vn": 0, "gif": 0}


def _read_workers_env(var_name: str, default: int) -> int:
    raw = os.environ.get(var_name, "").strip()
    if raw.isdigit():
        return max(1, min(32, int(raw)))
    return default


def _default_vn_workers() -> int:
    cpu = os.cpu_count() or 4
    return max(4, min(10, cpu))


def _default_gif_workers() -> int:
    cpu = os.cpu_count() or 4
    return max(2, min(8, max(2, cpu // 2)))


def _pctl(values: Deque[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[idx]


def _record_metrics(kind: str, queue_ms: float, render_ms: float, total_ms: float, label: str) -> None:
    with _METRICS_LOCK:
        _QUEUE_MS[kind].append(queue_ms)
        _RENDER_MS[kind].append(render_ms)
        _TOTAL_MS[kind].append(total_ms)
        _COUNT[kind] += 1
        count = _COUNT[kind]
        if count % 25 == 0:
            logger.info(
                "panel-exec[%s] n=%d q p50=%.0f p95=%.0f | render p50=%.0f p95=%.0f | total p50=%.0f p95=%.0f",
                kind,
                count,
                _pctl(_QUEUE_MS[kind], 0.5),
                _pctl(_QUEUE_MS[kind], 0.95),
                _pctl(_RENDER_MS[kind], 0.5),
                _pctl(_RENDER_MS[kind], 0.95),
                _pctl(_TOTAL_MS[kind], 0.5),
                _pctl(_TOTAL_MS[kind], 0.95),
            )
    if total_ms >= 3000:
        logger.warning(
            "panel-exec slow[%s] label=%s queue=%.0fms render=%.0fms total=%.0fms",
            kind,
            label,
            queue_ms,
            render_ms,
            total_ms,
        )
    log_animation_perf_event(
        "panel_executor",
        kind=kind,
        label=label,
        queue_ms=f"{queue_ms:.0f}",
        render_ms=f"{render_ms:.0f}",
        total_ms=f"{total_ms:.0f}",
    )


def _get_executor_vn() -> ThreadPoolExecutor:
    global _EXECUTOR_VN
    if _EXECUTOR_VN is None:
        default_n = _read_workers_env("TFBOT_PANEL_RENDER_WORKERS", _default_vn_workers())
        n = _read_workers_env("TFBOT_PANEL_RENDER_WORKERS_VN", default_n)
        _EXECUTOR_VN = ThreadPoolExecutor(max_workers=n, thread_name_prefix="panel_vn")
        logger.info("Initialized panel VN executor with %d workers", n)
    return _EXECUTOR_VN


def _get_executor_gif() -> ThreadPoolExecutor:
    global _EXECUTOR_GIF
    if _EXECUTOR_GIF is None:
        default_n = _read_workers_env("TFBOT_PANEL_RENDER_WORKERS", _default_gif_workers())
        n = _read_workers_env("TFBOT_PANEL_RENDER_WORKERS_GIF", default_n)
        _EXECUTOR_GIF = ThreadPoolExecutor(max_workers=n, thread_name_prefix="panel_gif")
        logger.info("Initialized panel GIF executor with %d workers", n)
    return _EXECUTOR_GIF


def _timed_call(
    fn: Callable[..., R],
    kind: str,
    submitted_at: float,
    label: str,
    /,
    *args: Any,
    **kwargs: Any,
) -> R:
    started = time.perf_counter()
    queue_ms = (started - submitted_at) * 1000.0
    result = fn(*args, **kwargs)
    finished = time.perf_counter()
    render_ms = (finished - started) * 1000.0
    total_ms = (finished - submitted_at) * 1000.0
    _record_metrics(kind, queue_ms, render_ms, total_ms, label)
    return result


def shutdown_panel_executor(*, wait: bool = True) -> None:
    """Release worker threads (e.g. on bot shutdown)."""
    global _EXECUTOR_VN, _EXECUTOR_GIF
    if _EXECUTOR_VN is not None:
        _EXECUTOR_VN.shutdown(wait=wait)
        _EXECUTOR_VN = None
    if _EXECUTOR_GIF is not None:
        _EXECUTOR_GIF.shutdown(wait=wait)
        _EXECUTOR_GIF = None


async def _run_panel_render_kind(
    kind: str,
    fn: Callable[..., R],
    /,
    *args: Any,
    label: str | None = None,
    **kwargs: Any,
) -> R:
    import asyncio

    loop = asyncio.get_running_loop()
    submitted = time.perf_counter()
    call_label = label or getattr(fn, "__name__", "panel_render")
    call = functools.partial(_timed_call, fn, kind, submitted, call_label, *args, **kwargs)
    executor = _get_executor_gif() if kind == "gif" else _get_executor_vn()
    return await loop.run_in_executor(executor, call)


async def run_panel_render_vn(fn: Callable[..., R], /, *args: Any, label: str | None = None, **kwargs: Any) -> R:
    """Run lighter VN panel work on VN executor."""
    return await _run_panel_render_kind("vn", fn, *args, label=label, **kwargs)


async def run_panel_render_gif(fn: Callable[..., R], /, *args: Any, label: str | None = None, **kwargs: Any) -> R:
    """Run heavier GIF/transition work on GIF executor."""
    return await _run_panel_render_kind("gif", fn, *args, label=label, **kwargs)


async def run_panel_render_transition(
    fn: Callable[..., R], /, *args: Any, label: str | None = None, **kwargs: Any
) -> R:
    """Alias for `run_panel_render_gif` (output may be WebP or GIF per encoder)."""
    return await run_panel_render_gif(fn, *args, label=label, **kwargs)


async def run_panel_render(fn: Callable[..., R], /, *args: Any, label: str | None = None, **kwargs: Any) -> R:
    """Backward-compatible alias routed to VN executor."""
    return await run_panel_render_vn(fn, *args, label=label, **kwargs)
