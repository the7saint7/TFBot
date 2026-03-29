"""VN animation-first message buffering for command flows.

This module is intentionally isolated so buffering can be toggled on/off
without removing integration code from command handlers.
"""

from __future__ import annotations

import asyncio
import contextvars
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Deque, Dict, Optional, Tuple


# VN send-buffer toggle (not .env). Set False to disable buffering; True matches old default when env was unset.
VN_ANIMATION_BUFFER_ENABLED = False


SendCall = Tuple[Callable[..., Awaitable[Any]], tuple, dict]


@dataclass
class _BufferState:
    animation_sent: bool = False
    queue: Deque[SendCall] = field(default_factory=deque)
    flushing: bool = False


_locks_by_key: Dict[int, asyncio.Lock] = {}
_current_state: contextvars.ContextVar[Optional[_BufferState]] = contextvars.ContextVar(
    "vn_animation_buffer_state",
    default=None,
)


def is_enabled() -> bool:
    return VN_ANIMATION_BUFFER_ENABLED


def _has_media_payload(kwargs: dict) -> bool:
    return kwargs.get("file") is not None or bool(kwargs.get("files"))


async def _flush_state(state: _BufferState) -> None:
    if state.flushing:
        return
    state.flushing = True
    try:
        while state.queue:
            sender, args, kwargs = state.queue.popleft()
            await sender(*args, **kwargs)
    finally:
        state.flushing = False


async def _dispatch_or_queue(
    state: _BufferState,
    sender: Callable[..., Awaitable[Any]],
    args: tuple,
    kwargs: dict,
) -> Any:
    has_media = _has_media_payload(kwargs)
    if has_media:
        state.animation_sent = True
        result = await sender(*args, **kwargs)
        await _flush_state(state)
        return result
    if not state.animation_sent and not state.flushing:
        state.queue.append((sender, args, kwargs))
        return None
    return await sender(*args, **kwargs)


def get_buffered_channel(channel: Any) -> Any:
    state = _current_state.get()
    if state is None:
        return channel

    class _BufferedChannelProxy:
        __slots__ = ("_channel", "_state")

        def __init__(self, base_channel: Any, buffer_state: _BufferState):
            self._channel = base_channel
            self._state = buffer_state

        async def send(self, *args, **kwargs):
            return await _dispatch_or_queue(self._state, self._channel.send, args, kwargs)

        def __getattr__(self, item: str) -> Any:
            return getattr(self._channel, item)

    return _BufferedChannelProxy(channel, state)


async def dispatch_with_active_buffer(sender: Callable[..., Awaitable[Any]], *args, **kwargs) -> Any:
    state = _current_state.get()
    if state is None or not VN_ANIMATION_BUFFER_ENABLED:
        return await sender(*args, **kwargs)
    return await _dispatch_or_queue(state, sender, args, kwargs)


@asynccontextmanager
async def command_buffer_scope(ctx: Any, *, lock_key: Optional[int], enabled: bool):
    if not enabled or not VN_ANIMATION_BUFFER_ENABLED:
        yield
        return

    key = int(lock_key or 0)
    lock = _locks_by_key.setdefault(key, asyncio.Lock())
    await lock.acquire()

    state = _BufferState()
    token = _current_state.set(state)

    original_reply = getattr(ctx, "reply")
    original_send = getattr(ctx, "send")

    async def _buffered_reply(*args, **kwargs):
        return await _dispatch_or_queue(state, original_reply, args, kwargs)

    async def _buffered_send(*args, **kwargs):
        return await _dispatch_or_queue(state, original_send, args, kwargs)

    setattr(ctx, "reply", _buffered_reply)
    setattr(ctx, "send", _buffered_send)
    try:
        yield
        await _flush_state(state)
    finally:
        setattr(ctx, "reply", original_reply)
        setattr(ctx, "send", original_send)
        _current_state.reset(token)
        lock.release()
