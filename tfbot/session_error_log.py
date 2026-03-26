"""
Per-process session logs for the ``tfbot`` logger tree (queued, non-blocking).

- **Errors:** ``<BASE_DIR>/vn_states/error_log_YYYY-MM-DD.log`` (UTC date at install).
  Override: ``TFBOT_SESSION_ERROR_LOG`` (absolute or relative to BASE_DIR).
- **Warnings:** ``<BASE_DIR>/vn_states/warning_log_YYYY-MM-DD.log`` by default.
  Override: ``TFBOT_SESSION_WARNING_LOG``. Set to empty to disable the warning file only.

Both files use the same block formatter and command trigger context. The error file receives
**ERROR** and above; the warning file receives **WARNING** only (no duplicate ERROR lines).

Opens with mode ``w`` each install (truncate). QueueHandler + QueueListener avoids blocking
the event loop on disk I/O.

Trigger context (prefix/slash) is copied onto each LogRecord in the main thread via a Filter
before the record is queued, because the listener thread does not see asyncio ContextVar state.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import logging.handlers
import os
import queue
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import discord
    from discord.ext import commands

_trigger_ctx: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar(
    "session_error_trigger", default=None
)

_listener: Optional[logging.handlers.QueueListener] = None
_queue_handler: Optional[logging.handlers.QueueHandler] = None
_SESSION_ERR_HOOKS_ATTR = "_tfbot_session_err_hooks_installed"

_ON_ERROR_CONTEXT_MAX_LEN = 380


def resolve_session_error_log_path(base_dir: Path) -> Path:
    override = os.getenv("TFBOT_SESSION_ERROR_LOG", "").strip()
    if override:
        p = Path(override)
        if not p.is_absolute():
            p = (base_dir / p).resolve()
        return p
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return (base_dir / "vn_states" / f"error_log_{day}.log").resolve()


def resolve_session_warning_log_path(base_dir: Path) -> Optional[Path]:
    """None if ``TFBOT_SESSION_WARNING_LOG`` is set to empty string (disable warning file)."""
    if "TFBOT_SESSION_WARNING_LOG" in os.environ and not os.environ["TFBOT_SESSION_WARNING_LOG"].strip():
        return None
    override = os.getenv("TFBOT_SESSION_WARNING_LOG", "").strip()
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if not override:
        return (base_dir / "vn_states" / f"warning_log_{day}.log").resolve()
    p = Path(override)
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    if p.suffix.lower() == ".log":
        return (p.parent / f"{p.stem}_{day}{p.suffix}").resolve()
    return (p / f"warning_log_{day}.log").resolve()


class _WarningOnlyFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno == logging.WARNING


class _InjectTriggerFilter(logging.Filter):
    """Copy ContextVar trigger onto the record in the emitting thread (before queue)."""

    def filter(self, record: logging.LogRecord) -> bool:
        t = _trigger_ctx.get()
        if t:
            record.session_trigger = dict(t)
        else:
            record.session_trigger = None
        return True


class SessionErrorFormatter(logging.Formatter):
    """Human- and paste-friendly blocks; full traceback when exc_info is set."""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
        lines = [
            "=" * 80,
            f"time_utc: {ts}",
            f"level: {record.levelname}",
            f"logger: {record.name}",
            f"message: {record.getMessage()}",
        ]
        trig = getattr(record, "session_trigger", None)
        if isinstance(trig, dict) and trig:
            lines.append(
                "trigger: "
                f"kind={trig.get('kind', 'n/a')} "
                f"command={trig.get('command', 'n/a')} "
                f"user_id={trig.get('user_id', 'n/a')} "
                f"guild_id={trig.get('guild_id', 'n/a')} "
                f"channel_id={trig.get('channel_id', 'n/a')}"
            )
        else:
            lines.append("trigger: system (no command context)")
        if record.exc_info and record.exc_info != (None, None, None):
            lines.append("traceback:")
            lines.append(
                "".join(traceback.format_exception(*record.exc_info)).rstrip()
            )
        lines.append("=" * 80)
        return "\n".join(lines) + "\n"


def _prefix_trigger_dict(ctx: Any) -> Dict[str, Any]:
    cmd = getattr(ctx, "command", None)
    name = getattr(cmd, "qualified_name", None) or getattr(cmd, "name", None) or "n/a"
    author = getattr(ctx, "author", None)
    guild = getattr(ctx, "guild", None)
    channel = getattr(ctx, "channel", None)
    return {
        "kind": "prefix",
        "command": name,
        "user_id": getattr(author, "id", None),
        "guild_id": getattr(guild, "id", None),
        "channel_id": getattr(channel, "id", None),
    }


def _slash_trigger_dict(interaction: Any) -> Dict[str, Any]:
    cmd = getattr(interaction, "command", None)
    name = getattr(cmd, "qualified_name", None) or getattr(cmd, "name", None) or "n/a"
    user = getattr(interaction, "user", None)
    guild = getattr(interaction, "guild", None)
    channel = getattr(interaction, "channel", None)
    return {
        "kind": "slash",
        "command": name,
        "user_id": getattr(user, "id", None),
        "guild_id": getattr(guild, "id", None),
        "channel_id": getattr(channel, "id", None),
    }


def _arm_prefix(ctx: Any) -> None:
    token = _trigger_ctx.set(_prefix_trigger_dict(ctx))
    setattr(ctx, "_session_err_log_reset_token", token)


def _disarm_prefix(ctx: Any) -> None:
    tok = getattr(ctx, "_session_err_log_reset_token", None)
    if tok is not None:
        try:
            _trigger_ctx.reset(tok)
        except ValueError:
            pass
        try:
            delattr(ctx, "_session_err_log_reset_token")
        except AttributeError:
            pass


def _arm_slash(interaction: Any) -> None:
    task = asyncio.current_task()
    if task is None:
        return
    token = _trigger_ctx.set(_slash_trigger_dict(interaction))

    def _done(_fut: Any) -> None:
        try:
            _trigger_ctx.reset(token)
        except ValueError:
            pass

    task.add_done_callback(_done)


def _summarize_on_error_args(args: tuple[Any, ...], kwargs: Dict[str, Any]) -> str:
    """Short, bounded context for Client.on_error (ids/types only, no message body)."""
    parts: list[str] = []
    for i, arg in enumerate(args[:6]):
        if arg is None:
            parts.append(f"arg{i}=None")
            continue
        oid = getattr(arg, "id", None)
        if oid is not None:
            bits = [f"{type(arg).__name__}", f"id={oid}"]
            g = getattr(arg, "guild", None)
            if g is not None and getattr(g, "id", None) is not None:
                bits.append(f"guild_id={g.id}")
            ch = getattr(arg, "channel", None)
            if ch is not None and getattr(ch, "id", None) is not None:
                bits.append(f"channel_id={ch.id}")
            parts.append(f"arg{i}=" + " ".join(bits))
        else:
            parts.append(f"arg{i}={type(arg).__name__}")
    if kwargs:
        keys = list(kwargs.keys())[:12]
        parts.append(f"kwargs_keys={keys}")
    out = "; ".join(parts) if parts else "(no args)"
    if len(out) > _ON_ERROR_CONTEXT_MAX_LEN:
        return out[: _ON_ERROR_CONTEXT_MAX_LEN - 3] + "..."
    return out


def install(base_dir: Path) -> None:
    """Truncate/create session log(s), start queue listener, attach QueueHandler to ``tfbot``."""
    global _listener, _queue_handler
    if _listener is not None:
        return

    err_path = resolve_session_error_log_path(base_dir)
    err_path.parent.mkdir(parents=True, exist_ok=True)

    error_handler = logging.FileHandler(err_path, mode="w", encoding="utf-8")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(SessionErrorFormatter())

    handlers: List[logging.Handler] = [error_handler]

    warn_path = resolve_session_warning_log_path(base_dir)
    if warn_path is not None:
        warn_path.parent.mkdir(parents=True, exist_ok=True)
        warning_handler = logging.FileHandler(warn_path, mode="w", encoding="utf-8")
        warning_handler.setLevel(logging.WARNING)
        warning_handler.addFilter(_WarningOnlyFilter())
        warning_handler.setFormatter(SessionErrorFormatter())
        handlers.append(warning_handler)

    log_queue: queue.SimpleQueue = queue.SimpleQueue()

    qh = logging.handlers.QueueHandler(log_queue)
    qh.setLevel(logging.WARNING)
    qh.addFilter(_InjectTriggerFilter())

    listener = logging.handlers.QueueListener(
        log_queue, *handlers, respect_handler_level=True
    )
    listener.start()

    logging.getLogger("tfbot").addHandler(qh)

    _listener = listener
    _queue_handler = qh


def shutdown() -> None:
    """Stop listener (drains queue), remove queue handler from ``tfbot``."""
    global _listener, _queue_handler
    if _queue_handler is not None:
        try:
            logging.getLogger("tfbot").removeHandler(_queue_handler)
        except ValueError:
            pass
        _queue_handler = None
    if _listener is not None:
        _listener.stop()
        _listener = None


def register_bot_hooks(bot: "commands.Bot") -> None:
    """Prefix + slash context; command error logging; client on_error.

    Idempotent per bot instance; a new Bot instance registers hooks normally.
    """
    if getattr(bot, _SESSION_ERR_HOOKS_ATTR, False):
        return

    import discord
    from discord import app_commands
    from discord.ext import commands

    @bot.before_invoke
    async def _session_err_before_invoke(ctx: commands.Context) -> None:
        _arm_prefix(ctx)

    @bot.after_invoke
    async def _session_err_after_invoke(ctx: commands.Context) -> None:
        _disarm_prefix(ctx)

    @bot.event
    async def on_command_error(ctx: commands.Context, error: commands.CommandError) -> None:
        try:
            if isinstance(error, commands.CommandNotFound):
                return
            if isinstance(error, commands.CheckFailure):
                return
            log = logging.getLogger("tfbot")
            cmd_name = getattr(ctx.command, "qualified_name", None) or getattr(
                ctx.command, "name", None
            )
            if isinstance(error, commands.CommandInvokeError):
                orig = error.original
                log.error(
                    "Prefix command error: %s",
                    cmd_name,
                    exc_info=(type(orig), orig, orig.__traceback__),
                )
            else:
                log.error(
                    "Prefix command error: %s: %s",
                    cmd_name,
                    error,
                    exc_info=(type(error), error, error.__traceback__),
                )
        finally:
            _disarm_prefix(ctx)

    @bot.tree.interaction_check
    async def _session_err_interaction_check(interaction: discord.Interaction) -> bool:
        _arm_slash(interaction)
        return True

    @bot.tree.error
    async def _session_err_app_command_error(
        interaction: discord.Interaction, error: app_commands.AppCommandError
    ) -> None:
        if isinstance(error, app_commands.CheckFailure):
            return
        log = logging.getLogger("tfbot")
        cmd = interaction.command
        cmd_name = getattr(cmd, "qualified_name", None) or getattr(cmd, "name", None)
        if isinstance(error, app_commands.CommandInvokeError):
            orig = error.original
            log.error(
                "Slash command error: %s",
                cmd_name,
                exc_info=(type(orig), orig, orig.__traceback__),
            )
        else:
            log.error(
                "Slash command error: %s: %s",
                cmd_name,
                error,
                exc_info=(type(error), error, error.__traceback__),
            )

    @bot.event
    async def on_error(event_method: str, *args: Any, **kwargs: Any) -> None:
        ctx_summary = _summarize_on_error_args(args, kwargs)
        logging.getLogger("tfbot").exception(
            "Unhandled event error in %s context=%s",
            event_method,
            ctx_summary,
        )

    setattr(bot, _SESSION_ERR_HOOKS_ATTR, True)
