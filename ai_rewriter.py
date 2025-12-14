import asyncio
import logging
import os
import time
from typing import Optional

import httpx

try:
    from openai import APIStatusError, RateLimitError, AsyncOpenAI  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    APIStatusError = RateLimitError = None  # type: ignore[misc]
    AsyncOpenAI = None  # type: ignore[assignment]

AI_LOGGER = logging.getLogger("tfbot.ai")

_truthy = {"1", "true", "yes", "on"}
AI_REWRITE_ENABLED = os.getenv("TFBOT_AI_REWRITE", "").lower() in _truthy
AI_API_KEY = os.getenv("TFBOT_AI_API_KEY") or os.getenv("OPENAI_API_KEY")
AI_MODEL = os.getenv("TFBOT_AI_MODEL", "gpt-3.5-turbo-1106")
AI_MAX_TOKENS = int(os.getenv("TFBOT_AI_MAX_TOKENS", "80"))
AI_TEMPERATURE = float(os.getenv("TFBOT_AI_TEMPERATURE", "0.5"))
AI_TIMEOUT = float(os.getenv("TFBOT_AI_TIMEOUT", "2.0"))
AI_CONCURRENCY = max(1, int(os.getenv("TFBOT_AI_CONCURRENCY", "1")))
_DEFAULT_SYSTEM_PROMPT = (
    "You are a paraphrasing assistant. Rewrite the user's message so it matches the provided "
    "character voice while preserving intent, meaning, Discord markdown, mentions (<@...> / <#...> / <@&...>), "
    "URLs, emoji tokens, and tone intensity. Keep the length similar, avoid adding new story beats or actions, "
    "and never include narration outside the spoken dialogue."
)
AI_SYSTEM_PROMPT = os.getenv("TFBOT_AI_SYSTEM_PROMPT") or _DEFAULT_SYSTEM_PROMPT
AI_RATE_LIMIT_BACKOFF = float(os.getenv("TFBOT_AI_BACKOFF", "1.5"))
AI_MIN_INTERVAL = float(os.getenv("TFBOT_AI_MIN_INTERVAL", "0.75"))

_client = None
_client_lock = asyncio.Lock()
_semaphore: Optional[asyncio.Semaphore] = None
_rate_lock = asyncio.Lock()
_last_request = 0.0

if not AI_REWRITE_ENABLED:
    AI_LOGGER.info("AI rewrite disabled (TFBOT_AI_REWRITE flag not set).")
elif not AI_API_KEY:
    AI_LOGGER.warning("AI rewrite disabled (missing TFBOT_AI_API_KEY / OPENAI_API_KEY).")
    AI_REWRITE_ENABLED = False
else:
    _semaphore = asyncio.Semaphore(AI_CONCURRENCY)
    AI_LOGGER.info(
        "AI rewrite enabled: model=%s, max_tokens=%s, temperature=%.2f, timeout=%.1fs, concurrency=%s",
        AI_MODEL,
        AI_MAX_TOKENS,
        AI_TEMPERATURE,
        AI_TIMEOUT,
        AI_CONCURRENCY,
    )


async def _get_client():
    global _client, AI_REWRITE_ENABLED, _semaphore  # pylint: disable=global-statement
    if _client is not None:
        return _client
    async with _client_lock:
        if _client is None:
            if AsyncOpenAI is None:
                AI_LOGGER.warning("OpenAI library not installed; disabling AI rewrites.")
                AI_REWRITE_ENABLED = False
                _semaphore = None
                return None
            _client = AsyncOpenAI(api_key=AI_API_KEY)
    return _client


def _build_user_prompt(
    original_text: str,
    character_name: str,
    character_context: Optional[str],
    user_name: Optional[str],
) -> str:
    lines = [
        f"Original: {original_text}",
        f"Character: {character_name}",
    ]
    if user_name:
        lines.append(f"Speaker: {user_name}")
    if character_context:
        lines.append(f"Character traits: {character_context}")
    lines.append(
        "Constraints: Keep the meaning intact, keep message concise, preserve Discord-specific tokens, "
        "and respond with dialogue only."
    )
    return "\n".join(lines)


def _truncate(value: str, limit: int = 200) -> str:
    value = value.replace("\n", " ").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


async def rewrite_message_for_character(
    original_text: str,
    character_name: str,
    character_context: Optional[str] = None,
    user_name: Optional[str] = None,
) -> Optional[str]:
    """Return an AI-rewritten message or None if disabled/failed."""
    if not AI_REWRITE_ENABLED or not original_text or not original_text.strip():
        return None

    client = await _get_client()
    if client is None:
        return None

    payload = {
        "model": AI_MODEL,
        "messages": [
            {"role": "system", "content": AI_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _build_user_prompt(original_text.strip(), character_name, character_context, user_name),
            },
        ],
        "max_tokens": AI_MAX_TOKENS,
        "temperature": AI_TEMPERATURE,
    }

    user_prompt = payload["messages"][1]["content"]
    AI_LOGGER.debug(
        "AI rewrite request -> character=%s speaker=%s prompt=\"%s\"",
        character_name,
        user_name or "unknown",
        _truncate(user_prompt, 240),
    )

    try:
        async with (_semaphore or asyncio.Semaphore(1)):
            await _respect_min_interval()
            response = await client.chat.completions.create(timeout=AI_TIMEOUT, **payload)
            _mark_request()
    except httpx.TimeoutException:
        AI_LOGGER.warning("AI rewrite timed out for %s (%.1fs)", character_name, AI_TIMEOUT)
        return None
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code if exc.response else None
        if status == 429:
            AI_LOGGER.warning(
                "AI rewrite rate limited for %s (429). Backing off %.1fs.",
                character_name,
                AI_RATE_LIMIT_BACKOFF,
            )
            await asyncio.sleep(AI_RATE_LIMIT_BACKOFF)
            _mark_request()
        else:
            AI_LOGGER.warning("AI rewrite HTTP error for %s: %s", character_name, exc)
        return None
    except Exception as exc:  # pylint: disable=broad-except
        if RateLimitError and isinstance(exc, RateLimitError):
            AI_LOGGER.warning(
                "AI rewrite rate limited (exception) for %s; backing off %.1fs.",
                character_name,
                AI_RATE_LIMIT_BACKOFF,
            )
            await asyncio.sleep(AI_RATE_LIMIT_BACKOFF)
            _mark_request()
            return None
        if APIStatusError and isinstance(exc, APIStatusError):
            AI_LOGGER.warning("AI rewrite API error for %s: %s", character_name, exc)
            return None
        AI_LOGGER.warning("AI rewrite failed for %s: %s", character_name, exc)
        return None

    for choice in getattr(response, "choices", []):
        message = getattr(choice, "message", None)
        if not message:
            continue
        content = getattr(message, "content", None)
        if content and content.strip():
            cleaned = content.strip()
            AI_LOGGER.debug(
                "AI rewrite response <- character=%s text=\"%s\"",
                character_name,
                _truncate(cleaned, 240),
            )
            return cleaned
    AI_LOGGER.debug("AI rewrite yielded no usable content for %s", character_name)
    return None


__all__ = ["AI_REWRITE_ENABLED", "rewrite_message_for_character"]


async def _respect_min_interval() -> None:
    if AI_MIN_INTERVAL <= 0:
        return
    async with _rate_lock:
        global _last_request  # pylint: disable=global-statement
        now = time.monotonic()
        remaining = _last_request + AI_MIN_INTERVAL - now
        if remaining > 0:
            await asyncio.sleep(remaining)
        _last_request = time.monotonic()


def _mark_request() -> None:
    if AI_MIN_INTERVAL <= 0:
        return
    global _last_request  # pylint: disable=global-statement
    _last_request = time.monotonic()
