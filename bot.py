import argparse
import asyncio
import io
import json
import logging
import os
import random
import re
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Set, Tuple

import aiohttp
import discord
from discord.ext import commands
from dotenv import load_dotenv

try:
    import yaml
except ImportError:
    yaml = None

from tf_characters import TF_CHARACTERS as CHARACTER_DATA


load_dotenv()

logging.basicConfig(
    level=os.getenv("TFBOT_LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tfbot")


def _int_from_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid integer for %s=%s. Falling back to %s.", name, raw, default)
        return default


def _float_from_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid float for %s=%s. Falling back to %s.", name, raw, default)
        return default


DEV_CHANNEL_ID = 1432191400983662766
DEV_TF_CHANCE = 0.75
TF_HISTORY_CHANNEL_ID = _int_from_env("TFBOT_HISTORY_CHANNEL_ID", 1432196317722972262)
TF_STATE_FILE = Path(os.getenv("TFBOT_STATE_FILE", "tf_state.json"))
TF_STATS_FILE = Path(os.getenv("TFBOT_STATS_FILE", "tf_stats.json"))
MESSAGE_STYLE = os.getenv("TFBOT_MESSAGE_STYLE", "classic").lower()
VN_BASE_IMAGE = Path(os.getenv("TFBOT_VN_BASE", "vn_assets/vn_base.png"))
VN_FONT_PATH = os.getenv("TFBOT_VN_FONT", "").strip()
VN_EMOJI_FONT_PATH = os.getenv('TFBOT_VN_EMOJI_FONT', 'fonts/NotoEmoji-VariableFont_wght.ttf').strip()
VN_NAME_FONT_SIZE = int(os.getenv("TFBOT_VN_NAME_SIZE", "34"))
VN_TEXT_FONT_SIZE = int(os.getenv("TFBOT_VN_TEXT_SIZE", "26"))
VN_GAME_ROOT = Path(os.getenv("TFBOT_VN_GAME_ROOT", "")).expanduser().resolve() if os.getenv("TFBOT_VN_GAME_ROOT") else None
VN_ASSET_ROOT = VN_GAME_ROOT / "game" / "images" / "characters" if VN_GAME_ROOT else None
VN_DEFAULT_OUTFIT = os.getenv("TFBOT_VN_OUTFIT", "casual.png")
VN_DEFAULT_FACE = os.getenv("TFBOT_VN_FACE", "0.png")
VN_AVATAR_MODE = os.getenv("TFBOT_VN_AVATAR_MODE", "game").lower()
VN_AVATAR_SCALE = max(0.1, _float_from_env("TFBOT_VN_AVATAR_SCALE", 1.0))
_VN_CACHE_DIR_SETTING = os.getenv("TFBOT_VN_CACHE_DIR", "vn_cache").strip()
TRANSFORM_DURATION_CHOICES: Sequence[Tuple[str, timedelta]] = [
    ("10 minutes", timedelta(minutes=10)),
    ("1 hour", timedelta(hours=1)),
    ("10 hours", timedelta(hours=10)),
    ("24 hours", timedelta(hours=24)),
]
DEV_TRANSFORM_DURATION = ("2 minutes", timedelta(minutes=2))
REQUIRED_GUILD_PERMISSIONS = {
    "send_messages": "Send Messages (needed to respond in channels)",
    "embed_links": "Embed Links (history channel logging)",
    "manage_nicknames": "Manage Nicknames (apply/revert TF names and avatars)",
}
MAGIC_EMOJI_NAME = os.getenv("TFBOT_MAGIC_EMOJI_NAME", "magic_emoji")
MAGIC_EMOJI_CACHE: Dict[int, str] = {}
tf_stats: Dict[str, Dict[str, Dict[str, object]]] = {}
_vn_config_cache: Dict[str, Dict] = {}


@dataclass(frozen=True)
class TFCharacter:
    name: str
    avatar_path: str
    message: str


@dataclass
class TransformationState:
    user_id: int
    guild_id: int
    character_name: str
    character_avatar_path: str
    character_message: str
    original_nick: Optional[str]
    started_at: datetime
    expires_at: datetime
    duration_label: str
    avatar_applied: bool = False


def _parse_channel_ids(raw: str) -> Set[int]:
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


def _get_magic_emoji(guild: Optional[discord.Guild]) -> str:
    if guild is None or guild.id is None:
        return f":{MAGIC_EMOJI_NAME}:"
    cached = MAGIC_EMOJI_CACHE.get(guild.id)
    if cached:
        return cached
    emoji = discord.utils.get(guild.emojis, name=MAGIC_EMOJI_NAME)
    if emoji:
        MAGIC_EMOJI_CACHE[guild.id] = str(emoji)
    else:
        MAGIC_EMOJI_CACHE[guild.id] = f":{MAGIC_EMOJI_NAME}:"
    return MAGIC_EMOJI_CACHE[guild.id]


def _is_admin(member: discord.abc.User) -> bool:
    if isinstance(member, discord.Member):
        if member.guild_permissions.administrator:
            return True
        roles: Iterable[discord.Role] = getattr(member, "roles", [])
        return any(role.name.lower() == "admin" for role in roles)
    return False


def _build_character_pool(source: Sequence[Dict[str, str]]) -> Sequence[TFCharacter]:
    pool: list[TFCharacter] = []
    for entry in source:
        try:
            pool.append(
                TFCharacter(
                    name=entry["name"],
                    avatar_path=entry.get("avatar_path", ""),
                    message=entry.get("message", ""),
                )
            )
        except KeyError as exc:
            logger.warning("Skipping character entry missing %s", exc)
    if not pool:
        raise RuntimeError("TF character dataset is empty. Populate tf_characters.py.")
    return pool


CHARACTER_POOL = _build_character_pool(CHARACTER_DATA)

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
if not DISCORD_TOKEN:
    raise RuntimeError("Missing DISCORD_TOKEN. Set it in your environment or .env file.")

TF_CHANCE = float(os.getenv("TFBOT_CHANCE", "0.10"))
TF_CHANCE = max(0.0, min(1.0, TF_CHANCE))

IGNORED_CHANNEL_IDS = _parse_channel_ids(os.getenv("TFBOT_IGNORE_CHANNELS", ""))
ALLOWED_CHANNEL_IDS: Optional[Set[int]] = None

intents = discord.Intents.default()
intents.message_content = True
intents.members = True

bot = commands.Bot(command_prefix=os.getenv("TFBOT_PREFIX", "!"), intents=intents)


TransformKey = Tuple[int, int]
active_transformations: Dict[TransformKey, TransformationState] = {}
revert_tasks: Dict[TransformKey, asyncio.Task] = {}

def find_active_transformation(user_id: int, guild_id: Optional[int] = None) -> Optional[TransformationState]:
    if guild_id is not None:
        state = active_transformations.get(_state_key(guild_id, user_id))
        if state:
            return state
    for state in active_transformations.values():
        if state.user_id == user_id and (guild_id is None or state.guild_id == guild_id):
            return state
    return None

STATE_RESTORED = False


def _state_key(guild_id: int, user_id: int) -> TransformKey:
    return (guild_id, user_id)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _serialize_state(state: TransformationState) -> Dict[str, object]:
    return {
        "user_id": state.user_id,
        "guild_id": state.guild_id,
        "character_name": state.character_name,
        "character_avatar_path": state.character_avatar_path,
        "character_message": state.character_message,
        "original_nick": state.original_nick,
        "started_at": state.started_at.isoformat(),
        "expires_at": state.expires_at.isoformat(),
        "duration_label": state.duration_label,
        "avatar_applied": state.avatar_applied,
    }


def _deserialize_state(payload: Dict[str, object]) -> TransformationState:
    return TransformationState(
        user_id=int(payload["user_id"]),
        guild_id=int(payload["guild_id"]),
        character_name=str(payload["character_name"]),
        character_avatar_path=str(
            payload.get("character_avatar_path") or payload.get("character_avatar_url", "")
        ),
        character_message=str(payload.get("character_message", "")),
        original_nick=payload.get("original_nick"),
        started_at=datetime.fromisoformat(str(payload["started_at"])),
        expires_at=datetime.fromisoformat(str(payload["expires_at"])),
        duration_label=str(payload["duration_label"]),
        avatar_applied=bool(payload.get("avatar_applied", False)),
    )


def persist_states() -> None:
    TF_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = [_serialize_state(state) for state in active_transformations.values()]
    TF_STATE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def persist_stats() -> None:
    TF_STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
    TF_STATS_FILE.write_text(json.dumps(tf_stats, indent=2), encoding="utf-8")



def load_outfit_selections() -> Dict[str, str]:
    if not VN_SELECTION_FILE.exists():
        return {}
    try:
        data = json.loads(VN_SELECTION_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k).lower(): str(v) for k, v in data.items()}
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse %s: %s", VN_SELECTION_FILE, exc)
    return {}

def persist_outfit_selections() -> None:
    try:
        VN_SELECTION_FILE.parent.mkdir(parents=True, exist_ok=True)
        VN_SELECTION_FILE.write_text(json.dumps(vn_outfit_selection, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning("Failed to persist outfit selections: %s", exc)

def load_states_from_disk() -> Sequence[TransformationState]:
    if not TF_STATE_FILE.exists():
        return []
    try:
        payload = json.loads(TF_STATE_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse %s: %s", TF_STATE_FILE, exc)
        return []
    states: list[TransformationState] = []
    for entry in payload:
        try:
            states.append(_deserialize_state(entry))
        except (KeyError, ValueError) as exc:
            logger.warning("Skipping malformed persisted state: %s", exc)
    return states


def load_stats_from_disk() -> Dict[str, Dict[str, Dict[str, object]]]:
    if not TF_STATS_FILE.exists():
        return {}
    try:
        data = json.loads(TF_STATS_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse %s: %s", TF_STATS_FILE, exc)
    return {}


tf_stats = load_stats_from_disk()
VN_SELECTION_FILE = Path(os.getenv("TFBOT_VN_SELECTIONS", "tf_outfits.json"))
vn_outfit_selection = load_outfit_selections()


def increment_tf_stats(guild_id: int, user_id: int, character_name: str) -> None:
    guild_stats = tf_stats.setdefault(str(guild_id), {})
    user_stats = guild_stats.setdefault(
        str(user_id), {"total": 0, "characters": {}}
    )
    total = int(user_stats.get("total", 0)) + 1
    user_stats["total"] = total
    char_stats = user_stats.setdefault("characters", {})
    char_stats[character_name] = int(char_stats.get(character_name, 0)) + 1
    persist_stats()


async def ensure_state_restored() -> None:
    global STATE_RESTORED
    if STATE_RESTORED:
        return
    states = load_states_from_disk()
    now = _now()
    for state in states:
        key = _state_key(state.guild_id, state.user_id)
        active_transformations[key] = state
        remaining = max((state.expires_at - now).total_seconds(), 0)
        if remaining <= 0:
            await revert_transformation(state, expired=True)
        else:
            revert_tasks[key] = asyncio.create_task(_schedule_revert(state, remaining))
            logger.info(
                "Restored TF for user %s in guild %s (expires in %.0fs)",
                state.user_id,
                state.guild_id,
                remaining,
            )
    STATE_RESTORED = True


async def _schedule_revert(state: TransformationState, delay: float) -> None:
    try:
        await asyncio.sleep(delay)
        await revert_transformation(state, expired=True)
    except asyncio.CancelledError:
        logger.debug("Revert task for user %s cancelled", state.user_id)
    except Exception:
        logger.exception("Unexpected error while reverting TF for user %s", state.user_id)


async def revert_transformation(state: TransformationState, *, expired: bool) -> None:
    key = _state_key(state.guild_id, state.user_id)
    current = active_transformations.get(key)
    if current is None or current.expires_at != state.expires_at:
        return

    guild, member = await fetch_member(state.guild_id, state.user_id)
    reason = "TF expired" if expired else "TF reverted"
    if member:
        try:
            await member.edit(nick=state.original_nick, reason=reason)
        except (discord.Forbidden, discord.HTTPException) as exc:
            logger.warning("Failed to restore nickname for %s: %s", member.id, exc)
    else:
        logger.warning("Could not locate member %s in guild %s to revert TF", state.user_id, state.guild_id)

    task = revert_tasks.pop(key, None)
    if task:
        task.cancel()
    active_transformations.pop(key, None)
    persist_states()

    username = member.name if member else state.original_nick or "Unknown"
    await send_history_message(
        "TF Reverted",
        f"Original Name: **{username}**\nCharacter: **{state.character_name}**\nReason: {reason}.",
    )


async def fetch_member(guild_id: int, user_id: int) -> Tuple[Optional[discord.Guild], Optional[discord.Member]]:
    guild = bot.get_guild(guild_id)
    if guild is None:
        try:
            guild = await bot.fetch_guild(guild_id)
        except discord.HTTPException as exc:
            logger.warning("Unable to fetch guild %s: %s", guild_id, exc)
            return None, None
    member = guild.get_member(user_id)
    if member is None:
        try:
            member = await guild.fetch_member(user_id)
        except discord.HTTPException as exc:
            logger.warning("Unable to fetch member %s in guild %s: %s", user_id, guild_id, exc)
            return guild, None
    return guild, member


BASE_DIR = Path(__file__).resolve().parent

if _VN_CACHE_DIR_SETTING:
    _vn_cache_path = Path(_VN_CACHE_DIR_SETTING)
    if not _vn_cache_path.is_absolute():
        VN_CACHE_DIR = (BASE_DIR / _vn_cache_path).resolve()
    else:
        VN_CACHE_DIR = _vn_cache_path.resolve()
else:
    VN_CACHE_DIR = None


@lru_cache(maxsize=64)
def _load_vn_font(size: int) -> "ImageFont.ImageFont":
    try:
        from PIL import ImageFont
    except ImportError:
        return ImageFont.load_default()

    font_candidates = []
    if VN_FONT_PATH:
        font_candidates.append(Path(VN_FONT_PATH))
    font_candidates.append(BASE_DIR / 'fonts' / 'Ubuntu-B.ttf')

    attempted = set()
    for candidate in font_candidates:
        if not candidate or candidate in attempted:
            continue
        attempted.add(candidate)
        if candidate.exists():
            try:
                logger.debug('VN sprite: loading font %s (size=%s)', candidate, size)
                return ImageFont.truetype(str(candidate), size=size)
            except OSError as exc:
                logger.warning('Failed to load VN font %s: %s', candidate, exc)
        else:
            logger.debug('VN sprite: font candidate missing -> %s', candidate)
    logger.warning('VN sprite: falling back to default font (size=%s)', size)
    return ImageFont.load_default()


@lru_cache(maxsize=64)
def _load_emoji_font(size: int):
    try:
        from PIL import ImageFont
    except ImportError:
        return None
    if not VN_EMOJI_FONT_PATH:
        return None
    emoji_path = Path(VN_EMOJI_FONT_PATH)
    if not emoji_path.exists():
        logger.debug('VN sprite: emoji font missing -> %s', emoji_path)
        return None
    try:
        return ImageFont.truetype(str(emoji_path), size=size)
    except OSError as exc:
        logger.warning('Failed to load emoji font %s: %s', emoji_path, exc)
        return None


def _is_emoji_char(ch: str) -> bool:
    code = ord(ch)
    return code >= 0x1F000 or 0x2600 <= code <= 0x27BF or 0x1F300 <= code <= 0x1FAFF


def _select_font_for_segment(segment: Dict, base_font: "ImageFont.ImageFont") -> "ImageFont.ImageFont":
    """Return an appropriate font for this segment, handling bold and emoji fallbacks."""
    size = getattr(base_font, "size", VN_TEXT_FONT_SIZE)
    font = base_font
    if segment.get("bold"):
        font = _load_vn_font(size + 2)
    if segment.get("emoji"):
        emoji_font = _load_emoji_font(getattr(font, "size", size))
        if emoji_font:
            font = emoji_font
    return font


CUSTOM_EMOJI_RE = re.compile(r"<(a?):([a-zA-Z0-9_]{2,}):(\d+)>")


def parse_discord_formatting(text: str) -> Sequence[dict]:
    """Parse a subset of Discord markdown into renderable text segments."""
    segments: list[dict] = []
    bold = italic = strike = False
    buffer: list[str] = []

    def emit_buffer() -> None:
        if buffer:
            segments.append(
                {
                    "text": "".join(buffer),
                    "bold": bold,
                    "italic": italic,
                    "strike": strike,
                    "emoji": False,
                }
            )
            buffer.clear()

    length = len(text)
    i = 0
    while i < length:
        ch = text[i]
        if text.startswith("**", i):
            emit_buffer()
            bold = not bold
            i += 2
            continue
        if text.startswith("~~", i):
            emit_buffer()
            strike = not strike
            i += 2
            continue
        if text.startswith("__", i):
            emit_buffer()
            italic = not italic
            i += 2
            continue
        if ch in ("*", "_"):
            emit_buffer()
            italic = not italic
            i += 1
            continue
        if ch == "\r":
            i += 1
            continue
        if ch == "\n":
            emit_buffer()
            segments.append(
                {
                    "text": "\n",
                    "bold": bold,
                    "italic": italic,
                    "strike": strike,
                    "emoji": False,
                    "newline": True,
                }
            )
            i += 1
            continue
        if ch == "<":
            match = CUSTOM_EMOJI_RE.match(text, i)
            if match:
                emit_buffer()
                animated = match.group(1) == "a"
                name = match.group(2)
                emoji_id = int(match.group(3))
                key = f"{emoji_id}{'a' if animated else ''}"
                segments.append(
                    {
                        "text": f":{name}:",
                        "bold": bold,
                        "italic": italic,
                        "strike": strike,
                        "emoji": False,
                        "custom_emoji": {
                            "name": name,
                            "id": emoji_id,
                            "animated": animated,
                            "key": key,
                        },
                    }
                )
                i = match.end()
                continue
        if _is_emoji_char(ch):
            emit_buffer()
            cluster_chars = [ch]
            i += 1
            while i < length:
                nxt = text[i]
                if nxt == "\r":
                    i += 1
                    continue
                if nxt == "\n":
                    break
                code = ord(nxt)
                if nxt == "\u200d":
                    cluster_chars.append(nxt)
                    i += 1
                    if i < length:
                        cluster_chars.append(text[i])
                        i += 1
                    continue
                if code in (0xFE0F, 0xFE0E) or 0x1F3FB <= code <= 0x1F3FF:
                    cluster_chars.append(nxt)
                    i += 1
                    continue
                if _is_emoji_char(nxt):
                    cluster_chars.append(nxt)
                    i += 1
                    continue
                break
            segments.append(
                {
                    "text": "".join(cluster_chars),
                    "bold": bold,
                    "italic": italic,
                    "strike": strike,
                    "emoji": True,
                }
            )
            continue
        buffer.append(ch)
        i += 1

    emit_buffer()
    if not segments:
        return [{"text": text, "bold": False, "italic": False, "strike": False, "emoji": False}]
    return segments


def layout_formatted_text(draw, segments: Sequence[dict], base_font, max_width: int) -> Sequence[Sequence[dict]]:
    """Lay out formatted segments into lines that fit the VN text box."""
    result: list[list[dict]] = []
    line: list[dict] = []
    current_x = 0
    baseline_height = getattr(base_font, "size", VN_TEXT_FONT_SIZE)

    def push_line(force_blank: bool = False) -> None:
        nonlocal line, current_x
        if line:
            result.append(line)
        elif force_blank:
            result.append([])
        line = []
        current_x = 0

    for segment in segments:
        if segment.get("newline"):
            push_line(force_blank=True)
            continue

        custom_meta = segment.get("custom_emoji")
        if custom_meta:
            emoji_size = getattr(base_font, "size", baseline_height)
            if current_x > 0 and current_x + emoji_size > max_width:
                push_line()
            line.append(
                {
                    "text": segment.get("text", ""),
                    "font": base_font,
                    "strike": False,
                    "bold": segment.get("bold"),
                    "italic": segment.get("italic"),
                    "emoji": False,
                    "custom_emoji": custom_meta,
                    "color": (240, 240, 240, 255),
                    "width": emoji_size,
                    "height": emoji_size,
                    "fallback_text": segment.get("text", ""),
                }
            )
            current_x += emoji_size
            continue

        text = segment.get("text", "")
        if not text:
            continue

        segment_font = _select_font_for_segment(segment, base_font)
        tokens = deque(_tokenize_for_layout(text, segment.get("emoji")))

        while tokens:
            token = tokens.popleft()
            if token == "":
                continue

            if token.isspace() and not line:
                continue

            width = draw.textlength(token, font=segment_font)
            if width > max_width and len(token) > 1 and not segment.get("emoji"):
                splits = _split_token_to_width(token, segment_font, max_width, draw)
                if len(splits) > 1:
                    for part in reversed(splits):
                        tokens.appendleft(part)
                    continue

            if current_x > 0 and current_x + width > max_width:
                push_line()
                if token.isspace():
                    continue
                width = draw.textlength(token, font=segment_font)

            if current_x == 0 and token.isspace():
                continue

            height = getattr(segment_font, "size", baseline_height)
            bbox = None
            try:
                if hasattr(segment_font, "getbbox"):
                    bbox = segment_font.getbbox(token)
                else:
                    bbox = draw.textbbox((0, 0), token, font=segment_font)
            except Exception:  # pylint: disable=broad-except
                bbox = None
            if bbox:
                height = max(height, bbox[3] - bbox[1])

            line.append(
                {
                    "text": token,
                    "font": segment_font,
                    "strike": segment.get("strike"),
                    "bold": segment.get("bold"),
                    "italic": segment.get("italic"),
                    "emoji": segment.get("emoji"),
                    "color": (240, 240, 240, 255),
                    "width": width,
                    "height": height,
                }
            )
            current_x += width

    push_line()
    return result


def _measure_layout_height(lines: Sequence[Sequence[dict]], base_line_height: int) -> int:
    total = 0
    line_spacing = 6
    for line in lines:
        if not line:
            total += base_line_height + line_spacing
            continue
        line_height = base_line_height
        for segment in line:
            seg_height = segment.get("height")
            if seg_height:
                line_height = max(line_height, seg_height)
        total += line_height + line_spacing
    if total > 0:
        total -= line_spacing
    return total


def _fit_text_segments(
    draw,
    segments: Sequence[dict],
    starting_font,
    max_width: int,
    max_height: int,
) -> Tuple[Sequence[Sequence[dict]], "ImageFont.ImageFont"]:
    """Shrink text until it fits within the text box height."""
    start_size = int(getattr(starting_font, "size", VN_TEXT_FONT_SIZE))
    min_size = max(8, int(start_size * 0.5))
    min_size = min(start_size, min_size)

    chosen_lines: Sequence[Sequence[dict]] = []
    chosen_font = starting_font

    for size in range(start_size, min_size - 1, -1):
        base_font = _load_vn_font(size)
        lines = layout_formatted_text(draw, segments, base_font, max_width)
        total_height = _measure_layout_height(lines, getattr(base_font, "size", size))
        chosen_lines = lines
        chosen_font = base_font
        if total_height <= max_height:
            break
    return chosen_lines, chosen_font


def _tokenize_for_layout(text: str, is_emoji: bool) -> Sequence[str]:
    if not text:
        return []
    if is_emoji:
        return [text]
    tokens: list[str] = []
    idx = 0
    length = len(text)
    while idx < length:
        start = idx
        if text[idx].isspace():
            while idx < length and text[idx].isspace():
                idx += 1
        else:
            while idx < length and not text[idx].isspace():
                idx += 1
        tokens.append(text[start:idx])
    return tokens


def _split_token_to_width(token: str, font, max_width: int, draw) -> Sequence[str]:
    if not token:
        return []
    if max_width <= 0:
        return [token]
    pieces: list[str] = []
    current = ""
    for ch in token:
        trial = current + ch
        width = draw.textlength(trial, font=font)
        if current and width > max_width:
            pieces.append(current)
            current = ch
        elif not current and width > max_width:
            pieces.append(ch)
            current = ""
        else:
            current = trial
    if current:
        pieces.append(current)
    return pieces


def _crop_transparent_top(image: "Image.Image") -> "Image.Image":
    """Remove transparent rows from the top of an RGBA sprite while keeping width."""
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    alpha = image.getchannel("A")
    bbox = alpha.getbbox()
    if not bbox:
        return image
    _, top, _, bottom = bbox
    height = bottom - top
    if top <= 0 or height <= 0:
        return image
    bottom = min(image.height, max(top + height, top + 1))
    cropped = image.crop((0, top, image.width, bottom))
    logger.debug("VN sprite: trimmed transparent top (%s -> %s)", image.size, cropped.size)
    return cropped


def _load_character_config(character_dir: Path) -> Dict:
    if yaml is None:
        return {}
    cache_key = str(character_dir)
    if cache_key in _vn_config_cache:
        return _vn_config_cache[cache_key]
    config_path = character_dir / "character.yml"
    config: Dict = {}
    if config_path.exists():
        try:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Failed to read character config %s: %s", config_path, exc)
            config = {}
    _vn_config_cache[cache_key] = config
    return config


def _select_variant_dir(character_dir: Path, config: Dict) -> Optional[Path]:
    variants = {child.name.lower(): child for child in character_dir.iterdir() if child.is_dir()}
    if not variants:
        logger.warning("VN sprite: no variants found under %s", character_dir.name)
        return None
    preferred_names = []
    poses = config.get("poses")
    if isinstance(poses, dict):
        preferred_names.extend(name.lower() for name in poses.keys())
    preferred_names.append("a")
    for name in preferred_names:
        if name in variants:
            return variants[name]
    return variants[sorted(variants.keys())[0]]


def _select_outfit_path(variant_dir: Path, config: Dict, preferred: Optional[str] = None) -> Optional[Path]:
    outfits_dir = variant_dir / "outfits"
    if not outfits_dir.exists():
        logger.warning("VN sprite: outfit directory missing at %s", outfits_dir)
        return None
    outfits = sorted(outfits_dir.glob("*.png"))
    if not outfits:
        logger.warning("VN sprite: no outfit PNGs in %s", outfits_dir)
        return None

    candidates: list[str] = []
    if preferred:
        candidates.append(preferred)
    default_outfit = config.get("default_outfit")
    if isinstance(default_outfit, str):
        candidates.append(default_outfit)
    if VN_DEFAULT_OUTFIT:
        candidates.append(VN_DEFAULT_OUTFIT)

    normalized_candidates = []
    for name in candidates:
        name = name.strip()
        if not name:
            continue
        lower = name.lower()
        if not lower.endswith(".png"):
            normalized_candidates.append(lower)
            normalized_candidates.append(f"{lower}.png")
        else:
            normalized_candidates.append(lower)
            normalized_candidates.append(lower.rstrip(".png"))

    for target in normalized_candidates:
        for outfit in outfits:
            if outfit.name.lower() == target or outfit.stem.lower() == target:
                logger.debug("VN sprite: using outfit %s for variant %s", outfit.name, variant_dir.name)
                return outfit

    logger.debug("VN sprite: defaulting to first outfit %s for variant %s", outfits[0].name, variant_dir.name)
    return outfits[0]

def _select_face_path(variant_dir: Path) -> Optional[Path]:
    faces_dir = variant_dir / "faces"
    if not faces_dir.exists():
        logger.warning("VN sprite: faces directory missing at %s", faces_dir)
        return None
    groups = [d for d in faces_dir.iterdir() if d.is_dir()]
    if not groups:
        logger.warning("VN sprite: no face groups found in %s", faces_dir)
        return None

    group_preference = ["face", "neutral", "default"]
    selected_group = None
    for candidate in group_preference:
        for group in groups:
            if group.name.lower() == candidate:
                selected_group = group
                break
        if selected_group:
            break
    if selected_group is None:
        selected_group = sorted(groups, key=lambda p: p.name.lower())[0]

    images = sorted(selected_group.glob("*.png"))
    if not images:
        return None

    face_candidates = []
    if VN_DEFAULT_FACE:
        if VN_DEFAULT_FACE.lower().endswith(".png"):
            face_candidates.append(VN_DEFAULT_FACE.lower())
            face_candidates.append(VN_DEFAULT_FACE.lower().rstrip(".png"))
        else:
            face_candidates.append(VN_DEFAULT_FACE.lower())
            face_candidates.append(f"{VN_DEFAULT_FACE}.png".lower())
    for target in face_candidates:
        for image_path in images:
            if image_path.name.lower() == target or image_path.stem.lower() == target:
                logger.debug("VN sprite: using face %s from group %s", image_path.name, selected_group.name)
                return image_path
    logger.debug(
        "VN sprite: defaulting to face %s from group %s", images[0].name, selected_group.name
    )
    return images[0]


def _candidate_character_keys(raw_name: str) -> Sequence[str]:
    name = raw_name.lower().strip()
    if not name:
        return []
    candidates = [name]
    if " " in name:
        first_word = name.split(" ", 1)[0]
        candidates.append(first_word)
        candidates.append(name.replace(" ", ""))
        candidates.append(name.replace(" ", "_"))
    if "-" in name:
        candidates.append(name.split("-", 1)[0])
        candidates.append(name.replace("-", ""))
    # Ensure unique order
    seen = set()
    ordered = []
    for cand in candidates:
        if cand not in seen:
            seen.add(cand)
            ordered.append(cand)
    return ordered


def resolve_character_directory(character_name: str) -> tuple[Optional[Path], Sequence[str]]:
    if VN_ASSET_ROOT is None:
        return None, []
    attempted: list[str] = []
    for key in _candidate_character_keys(character_name):
        candidate = VN_ASSET_ROOT / key
        attempted.append(candidate.name)
        if candidate.exists():
            return candidate, attempted
    return None, attempted

def list_available_outfits(character_name: str) -> Sequence[str]:
    directory, attempted = resolve_character_directory(character_name)
    if directory is None:
        logger.debug("VN sprite: cannot list outfits for %s (tried %s)", character_name, attempted)
        return []
    config = _load_character_config(directory)
    variant_dir = _select_variant_dir(directory, config)
    if not variant_dir:
        return []
    outfits_dir = variant_dir / "outfits"
    if not outfits_dir.exists():
        return []
    outfits = sorted({p.stem for p in outfits_dir.glob("*.png")})
    return outfits

def get_selected_outfit_name(character_name: str) -> Optional[str]:
    directory, _ = resolve_character_directory(character_name)
    if directory is None:
        return None
    return vn_outfit_selection.get(directory.name.lower())

def set_selected_outfit_name(character_name: str, outfit_name: str) -> bool:
    directory, attempted = resolve_character_directory(character_name)
    if directory is None:
        logger.debug("VN sprite: cannot set outfit for %s (tried %s)", character_name, attempted)
        return False
    outfits = list_available_outfits(character_name)
    if outfit_name.lower() not in [o.lower() for o in outfits]:
        return False
    vn_outfit_selection[directory.name.lower()] = outfit_name
    persist_outfit_selections()
    compose_game_avatar.cache_clear()
    logger.info("VN sprite: outfit override for %s set to %s", directory.name, outfit_name)
    return True

def get_selected_outfit_for_dir(directory: Path) -> Optional[str]:
    return vn_outfit_selection.get(directory.name.lower())



@lru_cache(maxsize=128)
def compose_game_avatar(character_name: str) -> Optional["Image.Image"]:
    if VN_ASSET_ROOT is None:
        return None
    try:
        from PIL import Image
    except ImportError:
        return None

    character_dir, attempted = resolve_character_directory(character_name)
    if character_dir is None:
        logger.debug("VN sprite: missing character directory for %s (tried %s)", character_name, attempted)
        return None

    config = _load_character_config(character_dir)
    variant_dir = _select_variant_dir(character_dir, config)
    if not variant_dir:
        logger.debug("VN sprite: no variant directory found for %s", character_dir.name)
        return None

    preferred_outfit = get_selected_outfit_for_dir(character_dir)
    if preferred_outfit:
        logger.debug("VN sprite: preferred outfit override for %s is %s", character_dir.name, preferred_outfit)
    outfit_path = _select_outfit_path(variant_dir, config, preferred_outfit)
    if not outfit_path or not outfit_path.exists():
        logger.warning(
            "VN sprite: outfit not found for %s (variant %s, tried %s)",
            character_dir.name,
            variant_dir.name,
            outfit_path,
        )
        return None

    face_path = _select_face_path(variant_dir)

    cache_file: Optional[Path] = None
    if VN_CACHE_DIR:
        face_token = "noface"
        if face_path and face_path.exists():
            face_token = face_path.stem.lower()
        cache_dir = VN_CACHE_DIR / character_dir.name.lower() / variant_dir.name.lower()
        cache_file = cache_dir / f"{outfit_path.stem.lower()}__{face_token}.png"
        if cache_file.exists():
            try:
                cached = Image.open(cache_file).convert("RGBA")
                logger.debug("VN sprite: loaded cached avatar %s", cache_file)
                return cached
            except OSError as exc:
                logger.warning("VN sprite: failed to load cached avatar %s: %s (rebuilding)", cache_file, exc)

    try:
        outfit_image = Image.open(outfit_path).convert("RGBA")
    except OSError as exc:
        logger.warning("Failed to load outfit %s: %s", outfit_path, exc)
        return None

    if face_path and face_path.exists():
        try:
            face_image = Image.open(face_path).convert("RGBA")
            outfit_image.paste(face_image, (0, 0), face_image)
        except OSError as exc:
            logger.warning("Failed to load face %s: %s", face_path, exc)
    else:
        logger.warning(
            "VN sprite: face image missing for %s (variant %s, searched %s)",
            character_dir.name,
            variant_dir.name,
            face_path,
        )

    final_image = _crop_transparent_top(outfit_image)
    if cache_file:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            final_image.save(cache_file)
            logger.debug("VN sprite: cached avatar saved to %s", cache_file)
        except OSError as exc:
            logger.warning("VN sprite: unable to save cached avatar %s: %s", cache_file, exc)
    return final_image

def render_vn_panel(
    state: TransformationState,
    message_content: str,
    character_display_name: str,
    original_name: str,
    attachment_id: Optional[str] = None,
    formatted_segments: Optional[Sequence[dict]] = None,
    custom_emoji_images: Optional[Dict[str, "Image.Image"]] = None,
) -> Optional[discord.File]:
    if MESSAGE_STYLE != "vn":
        return None
    if not VN_BASE_IMAGE.exists():
        logger.warning("VN base image not found at %s; falling back to classic style.", VN_BASE_IMAGE)
        return None

    try:
        from PIL import Image, ImageDraw
    except ImportError:
        logger.warning("Pillow is not installed; cannot render VN message style.")
        return None

    base = Image.open(VN_BASE_IMAGE).convert("RGBA")
    draw = ImageDraw.Draw(base)

    from PIL import Image
    name_font = _load_vn_font(VN_NAME_FONT_SIZE)
    base_text_font = _load_vn_font(VN_TEXT_FONT_SIZE)

    # Layout constants derived from vn_base.png geometry.
    name_box = (170, 10, 303, 26)
    text_box = (178, 80, 775, 250)
    name_padding = 10
    text_padding = 12
    avatar_box = (4, 4, 160, 250)
    avatar_width = avatar_box[2] - avatar_box[0]
    avatar_height = avatar_box[3] - avatar_box[1]

    avatar_image = None
    local_avatar_path: Optional[Path] = None
    if state.character_avatar_path:
        local_avatar_path = Path(state.character_avatar_path)
        if not local_avatar_path.is_absolute():
            local_avatar_path = (BASE_DIR / local_avatar_path).resolve()
        if not local_avatar_path.exists():
            local_avatar_path = None

    if VN_AVATAR_MODE == "user" and local_avatar_path:
        try:
            avatar_image = Image.open(local_avatar_path).convert("RGBA")
        except OSError as exc:
            logger.warning("Failed to load avatar %s for VN panel: %s", local_avatar_path, exc)
            avatar_image = None

    if avatar_image is None and VN_AVATAR_MODE == "game" and state.character_name:
        composed = compose_game_avatar(state.character_name)
        if composed is not None:
            logger.debug("VN sprite: composed avatar ready for %s (%s)", state.character_name, composed.size)
            avatar_image = composed.copy()

    if avatar_image is None and local_avatar_path:
        try:
            avatar_image = Image.open(local_avatar_path).convert("RGBA")
        except OSError as exc:
            logger.warning("Failed to load avatar %s for VN panel: %s", local_avatar_path, exc)
            avatar_image = None

    if avatar_image is not None:
        avatar_image = avatar_image.copy().convert("RGBA")
        orig_w, orig_h = avatar_image.size
        scale = max(0.1, VN_AVATAR_SCALE or 1.0)
        scaled_w = max(1, int(orig_w * scale))
        scaled_h = max(1, int(orig_h * scale))
        if (scaled_w, scaled_h) != avatar_image.size:
            avatar_image = avatar_image.resize((scaled_w, scaled_h), Image.LANCZOS)

        target_w = avatar_width
        target_h = avatar_height

        if avatar_image.width < target_w or avatar_image.height < target_h:
            adjust = max(target_w / avatar_image.width, target_h / avatar_image.height)
            new_size = (
                max(1, int(avatar_image.width * adjust)),
                max(1, int(avatar_image.height * adjust)),
            )
            avatar_image = avatar_image.resize(new_size, Image.LANCZOS)

        crop_left = max(0, (avatar_image.width - target_w) // 2)
        crop_upper = 0
        crop_right = crop_left + target_w
        crop_lower = crop_upper + target_h
        crop_box = (
            crop_left,
            crop_upper,
            min(crop_right, avatar_image.width),
            min(crop_lower, avatar_image.height),
        )
        cropped = avatar_image.crop(crop_box)

        if cropped.size != (target_w, target_h):
            canvas = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
            offset_x = max(0, (target_w - cropped.width) // 2)
            offset_y = max(0, (target_h - cropped.height) // 2)
            canvas.paste(cropped, (offset_x, offset_y), cropped)
            cropped = canvas

        pos_x = avatar_box[0]
        pos_y = avatar_box[1]
        base.paste(cropped, (pos_x, pos_y), cropped)
        logger.debug(
            "VN sprite: pasted avatar for %s at (%s, %s) size %s (scale=%s)",
            state.character_name,
            pos_x,
            pos_y,
            cropped.size,
            scale,
        )
    if avatar_image is None:
        logger.warning(
            "VN sprite: no avatar rendered for %s (mode=%s)",
            state.character_name,
            VN_AVATAR_MODE,
        )

    name_text = character_display_name
    name_x = name_box[0] + name_padding
    name_y = name_box[1] + name_padding
    draw.text((name_x, name_y), name_text, fill=(255, 220, 180, 255), font=name_font)

    working_content = message_content.strip()
    if not working_content:
        working_content = f"{original_name} remains quietly transformed..."

    max_width = text_box[2] - text_box[0] - text_padding * 2
    max_height = text_box[3] - text_box[1] - text_padding * 2
    segments = list(formatted_segments) if formatted_segments else []
    if not segments:
        segments = list(parse_discord_formatting(working_content))
    lines, text_font = _fit_text_segments(draw, segments, base_text_font, max_width, max_height)
    text_y = text_box[1] + text_padding
    base_line_height = getattr(text_font, "size", VN_TEXT_FONT_SIZE)
    for line in lines:
        if not line:
            text_y += base_line_height + 6
            continue
        text_x = text_box[0] + text_padding
        max_height = 0
        for segment in line:
            fill = segment.get("color", (240, 240, 240, 255))
            font_segment = segment["font"]
            text_segment = segment.get("text", "")
            width = segment.get("width", 0)
            height = segment.get("height") or getattr(font_segment, "size", base_line_height)
            custom_meta = segment.get("custom_emoji")
            if custom_meta:
                key = custom_meta.get("key")
                emoji_img = None
                if custom_emoji_images and key:
                    emoji_img = custom_emoji_images.get(key)
                if emoji_img is not None:
                    emoji_render = emoji_img.copy()
                    target_w = int(width) or base_line_height
                    target_h = int(height) or base_line_height
                    emoji_render.thumbnail((target_w, target_h), Image.LANCZOS)
                    offset_y = text_y + max(0, base_line_height - emoji_render.height)
                    base.paste(emoji_render, (int(text_x), int(offset_y)), emoji_render)
                else:
                    fallback = segment.get("fallback_text") or text_segment or custom_meta.get("name") or ""
                    if fallback:
                        draw.text((text_x, text_y), fallback, fill=fill, font=text_font)
                max_height = max(max_height, height)
                text_x += width
                continue
            if text_segment:
                draw.text((text_x, text_y), text_segment, fill=fill, font=font_segment)
                if segment.get("strike"):
                    strike_y = text_y + height / 2
                    draw.line(
                        (text_x, strike_y, text_x + width, strike_y),
                        fill=fill,
                        width=max(1, int(height / 10)),
                    )
            max_height = max(max_height, height)
            text_x += width
        text_y += max_height + 6
    output = io.BytesIO()
    base.save(output, format="PNG")
    output.seek(0)
    unique_fragment = attachment_id or str(int(_now().timestamp() * 1000))
    filename = f"tf-panel-{state.user_id}-{unique_fragment}.png"
    return discord.File(fp=output, filename=filename)


async def fetch_avatar_bytes(path_or_url: str) -> Optional[bytes]:
    if not path_or_url:
        return None

    if path_or_url.startswith(("http://", "https://")):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(path_or_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        logger.warning("Avatar fetch failed (%s): %s", resp.status, path_or_url)
                        return None
                    return await resp.read()
        except aiohttp.ClientError as exc:
            logger.warning("Avatar fetch error for %s: %s", path_or_url, exc)
            return None

    local_path = Path(path_or_url)
    if not local_path.is_absolute():
        local_path = BASE_DIR / local_path
    local_path = local_path.resolve()
    if not local_path.exists():
        logger.warning("Avatar file not found: %s", local_path)
        return None
    try:
        return local_path.read_bytes()
    except OSError as exc:
        logger.warning("Failed to read avatar file %s: %s", local_path, exc)
        return None


async def prepare_custom_emoji_images(
    message: discord.Message,
    segments: Sequence[dict],
) -> Dict[str, "Image.Image"]:
    """Load custom Discord emojis referenced in the message for VN rendering."""
    try:
        from PIL import Image
    except ImportError:
        return {}

    needed: Dict[str, dict] = {}
    for segment in segments:
        meta = segment.get("custom_emoji")
        if not meta:
            continue
        key = meta.get("key")
        if key and key not in needed:
            needed[key] = meta
    if not needed:
        return {}

    cache_dir: Optional[Path] = None
    if VN_CACHE_DIR:
        cache_dir = VN_CACHE_DIR / "__emojis__"
        cache_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, "Image.Image"] = {}
    for key, meta in needed.items():
        emoji_id = meta.get("id")
        if emoji_id is None:
            continue
        cache_path = cache_dir / f"{emoji_id}.png" if cache_dir else None
        image_obj = None
        if cache_path and cache_path.exists():
            try:
                image_obj = Image.open(cache_path).convert("RGBA")
            except OSError as exc:
                logger.warning("VN emoji cache read failed (%s): %s", cache_path, exc)
                image_obj = None
        if image_obj is None:
            emoji_url = None
            if message.guild:
                emoji_obj = discord.utils.get(message.guild.emojis, id=int(emoji_id))
                if emoji_obj:
                    emoji_url = str(emoji_obj.url)
            if emoji_url is None:
                ext = "gif" if meta.get("animated") else "png"
                emoji_url = f"https://cdn.discordapp.com/emojis/{emoji_id}.{ext}?quality=lossless"
            data = await fetch_avatar_bytes(emoji_url)
            if not data:
                continue
            try:
                image_obj = Image.open(io.BytesIO(data))
                if getattr(image_obj, "is_animated", False):
                    image_obj.seek(0)
                image_obj = image_obj.convert("RGBA")
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("VN sprite: failed to decode emoji %s: %s", emoji_id, exc)
                continue
            if cache_path:
                try:
                    image_obj.save(cache_path, format="PNG")
                except OSError as exc:
                    logger.warning("VN sprite: unable to store emoji cache %s: %s", cache_path, exc)
        if image_obj:
            results[key] = image_obj
    return results


async def relay_transformed_message(
    message: discord.Message,
    state: TransformationState,
    *,
    reference: Optional[discord.MessageReference] = None,
) -> bool:
    guild = message.guild
    if guild is None:
        return False

    cleaned_content = message.content.strip()
    description = cleaned_content if cleaned_content else "*no message content*"
    formatted_segments = parse_discord_formatting(cleaned_content) if cleaned_content else None
    custom_emoji_images: Dict[str, "Image.Image"] = {}

    files: list[discord.File] = []
    payload: dict = {}

    if MESSAGE_STYLE == "vn":
        if formatted_segments is None:
            formatted_segments = parse_discord_formatting(cleaned_content)
        custom_emoji_images = await prepare_custom_emoji_images(message, formatted_segments)
        vn_file = render_vn_panel(
            state=state,
            message_content=cleaned_content,
            character_display_name=state.character_name,
            original_name=message.author.display_name,
            attachment_id=str(message.id),
            formatted_segments=formatted_segments,
            custom_emoji_images=custom_emoji_images,
        )
        if vn_file:
            files.append(vn_file)
        else:
            logger.debug("VN panel rendering unavailable; using classic embed.")

    if not files:
        embed = discord.Embed(
            description=description,
            color=0x9B59B6,
            timestamp=_now(),
        )
        embed.set_author(name=state.character_name)

        avatar_bytes = await fetch_avatar_bytes(state.character_avatar_path)
        if avatar_bytes:
            suffix = Path(state.character_avatar_path).suffix or ".png"
            avatar_filename = f"tf-avatar-{state.user_id}{suffix}"
            files.append(
                discord.File(io.BytesIO(avatar_bytes), filename=avatar_filename)
            )
            embed.set_thumbnail(url=f"attachment://{avatar_filename}")

        payload["embed"] = embed

    for attachment in message.attachments:
        try:
            files.append(await attachment.to_file())
        except discord.HTTPException as exc:
            logger.warning(
                "Failed to mirror attachment %s from message %s: %s",
                attachment.id,
                message.id,
                exc,
            )

    deleted = True
    try:
        await message.delete()
    except discord.Forbidden:
        deleted = False
        logger.debug(
            "Missing permission to delete message %s for TF relay in channel %s",
            message.id,
            message.channel.id,
        )
    except discord.HTTPException as exc:
        deleted = False
        logger.warning("Failed to delete message %s: %s", message.id, exc)

    if not deleted and "embed" in payload:
        payload["embed"].set_footer(text="Grant Manage Messages so TF relay can replace posts.")

    send_kwargs: Dict[str, object] = {}
    send_kwargs.update(payload)
    if reference:
        if isinstance(reference, discord.Message):
            reference = reference.to_reference(fail_if_not_exists=False)
        send_kwargs["reference"] = reference
        send_kwargs["mention_author"] = False
    if files:
        send_kwargs["files"] = files

    try:
        await message.channel.send(**send_kwargs)
    except discord.HTTPException as exc:
        logger.warning("Failed to relay TF message %s: %s", message.id, exc)
        return False

    return True


async def send_history_message(title: str, description: str) -> None:
    channel = bot.get_channel(TF_HISTORY_CHANNEL_ID)
    if channel is None:
        try:
            channel = await bot.fetch_channel(TF_HISTORY_CHANNEL_ID)
        except discord.HTTPException as exc:
            logger.warning("Cannot send history message, channel lookup failed: %s", exc)
            return
    embed = discord.Embed(
        title=title,
        description=description,
        color=0x9B59B6 if title == "TF Applied" else 0x546E7A,
        timestamp=_now(),
    )
    try:
        await channel.send(embed=embed, allowed_mentions=discord.AllowedMentions.none())
    except discord.HTTPException as exc:
        logger.warning("Failed to send history message: %s", exc)


def _format_character_message(
    template: str,
    original_name: str,
    mention: str,
    duration: str,
    character_name: str,
) -> str:
    context = {
        "member": original_name,
        "original_name": original_name,
        "mention": mention,
        "character": character_name,
        "duration": duration,
    }

    try:
        unique_segment = template.format(**context).strip() if template else ""
    except KeyError:
        unique_segment = template.strip() if template else ""

    if unique_segment:
        lead_line = f"{original_name} {unique_segment}".strip()
    else:
        lead_line = f"{original_name} feels a strange energy swirling..."

    summary_line = f"{original_name} becomes **{character_name}** for {duration}!"
    return f"{lead_line}\n{summary_line}"


async def handle_transformation(message: discord.Message) -> Optional[TransformationState]:
    if not message.guild:
        logger.debug("Skipping TF outside of guild context.")
        return None

    await ensure_state_restored()

    member = message.guild.get_member(message.author.id)
    if member is None:
        try:
            member = await message.guild.fetch_member(message.author.id)
        except discord.HTTPException as exc:
            logger.warning("Failed to fetch member %s: %s", message.author.id, exc)
            return

    key = _state_key(message.guild.id, member.id)
    if key in active_transformations:
        logger.debug("User %s already transformed; skipping.", member.id)
        return None

    used_characters = {state.character_name for state in active_transformations.values()}
    available_characters = [character for character in CHARACTER_POOL if character.name not in used_characters]
    if not available_characters:
        logger.info("No available TF characters; skipping message %s", message.id)
        return None

    character = random.choice(available_characters)
    if DEV_MODE:
        duration_label, duration_delta = DEV_TRANSFORM_DURATION
    else:
        duration_label, duration_delta = random.choice(TRANSFORM_DURATION_CHOICES)
    now = _now()
    expires_at = now + duration_delta
    original_nick = member.nick

    if member.guild and member.guild.owner_id == member.id:
        logger.warning(
            "Cannot apply TF nickname to server owner %s in guild %s",
            member.id,
            member.guild.id,
        )
        return None

    try:
        await member.edit(nick=character.name, reason="TF event")
    except discord.Forbidden:
        logger.warning("Missing permissions to edit nickname for %s", member.id)
        return None
    except discord.HTTPException as exc:
        logger.warning("Failed to edit nickname for %s: %s", member.id, exc)
        return None

    state = TransformationState(
        user_id=member.id,
        guild_id=message.guild.id,
        character_name=character.name,
        character_avatar_path=character.avatar_path,
        character_message=character.message,
        original_nick=original_nick,
        started_at=now,
        expires_at=expires_at,
        duration_label=duration_label,
        avatar_applied=False,
    )
    active_transformations[key] = state
    persist_states()

    delay = max((expires_at - now).total_seconds(), 0)
    revert_tasks[key] = asyncio.create_task(_schedule_revert(state, delay))

    logger.info(
        "TF applied to user %s (%s) for %s (expires at %s)",
        member.id,
        character.name,
        duration_label,
        expires_at.isoformat(),
    )

    increment_tf_stats(message.guild.id, member.id, character.name)

    await send_history_message(
        "TF Applied",
        f"Original Name: **{member.name}**\nCharacter: **{character.name}**\nDuration: {duration_label}.",
    )

    original_name = original_nick or member.name
    response_text = _format_character_message(
        character.message,
        original_name,
        member.mention,
        duration_label,
        character.name,
    )
    emoji_prefix = _get_magic_emoji(message.guild)
    response_text = f"{emoji_prefix} {response_text}"
    await message.reply(response_text, mention_author=False)

    reply_reference: Optional[discord.MessageReference] = (
        message.to_reference(fail_if_not_exists=False) if message.reference else None
    )
    await relay_transformed_message(message, state, reference=reply_reference)

    return state


async def log_guild_permissions() -> None:
    me = bot.user
    for guild in bot.guilds:
        member = guild.me
        if member is None:
            try:
                member = await guild.fetch_member(me.id) if me else None
            except discord.HTTPException as exc:
                logger.warning("Could not fetch self member in guild %s: %s", guild.id, exc)
                continue
        perms = member.guild_permissions
        missing = []
        for attr, reason in REQUIRED_GUILD_PERMISSIONS.items():
            if not getattr(perms, attr, False):
                missing.append(f"{attr.replace('_', ' ')} ({reason})")
        if missing:
            logger.warning(
                "Guild '%s' (%s) missing permissions: %s",
                guild.name,
                guild.id,
                "; ".join(missing),
            )
        else:
            logger.info("Guild '%s' (%s) has all required permissions.", guild.name, guild.id)


async def log_channel_access() -> None:
    for guild in bot.guilds:
        me = guild.me
        if me is None:
            continue
        channel_ids = set()
        if DEV_MODE:
            channel_ids |= ALLOWED_CHANNEL_IDS or set()
        else:
            channel_ids |= IGNORED_CHANNEL_IDS
        channel_ids.add(DEV_CHANNEL_ID)
        for channel_id in channel_ids:
            channel = guild.get_channel(channel_id)
            if channel is None:
                continue
            perms = channel.permissions_for(me)
            if not perms.view_channel or not perms.read_message_history:
                logger.warning(
                    "Channel %s in guild '%s' missing read permissions (view=%s, history=%s)",
                    channel_id,
                    guild.name,
                    perms.view_channel,
                    perms.read_message_history,
                )
            else:
                logger.info(
                    "Channel %s in guild '%s' readable; send=%s, mentionable=%s",
                    channel_id,
                    guild.name,
                    perms.send_messages,
                    perms.mention_everyone,
                )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TF Discord Bot launcher")
    parser.add_argument(
        "-dev",
        "--dev",
        dest="dev_mode",
        action="store_true",
        help="Run the bot in dev mode (restricted channel, 75%% TF chance).",
    )
    return parser.parse_args()


DEV_MODE = False


def enable_dev_mode() -> None:
    global DEV_MODE, TF_CHANCE, ALLOWED_CHANNEL_IDS
    DEV_MODE = True
    TF_CHANCE = DEV_TF_CHANCE
    ALLOWED_CHANNEL_IDS = {DEV_CHANNEL_ID}
    logger.warning(
        "Dev mode ENABLED: restricting activity to channel %s with TF chance %.0f%%",
        DEV_CHANNEL_ID,
        TF_CHANCE * 100,
    )


@bot.event
async def on_ready():
    await ensure_state_restored()
    logger.info("Logged in as %s (id=%s)", bot.user, bot.user.id if bot.user else "unknown")
    logger.info("TF chance set to %.0f%%", TF_CHANCE * 100)
    logger.info("Message style: %s", MESSAGE_STYLE.upper())
    logger.info("Dev mode: %s", "ON" if DEV_MODE else "OFF")
    if DEV_MODE:
        logger.info("Allowed channel: %s", DEV_CHANNEL_ID)
    if IGNORED_CHANNEL_IDS and not DEV_MODE:
        logger.info("Ignoring channels: %s", ", ".join(str(cid) for cid in IGNORED_CHANNEL_IDS))
    await log_guild_permissions()
    await log_channel_access()


@bot.command(name="synreset", hidden=True)
@commands.guild_only()
async def secret_reset_command(ctx: commands.Context):
    author = ctx.author
    if not isinstance(author, discord.Member):
        await ctx.reply("This command can only be used inside a server.", mention_author=False)
        return None
    if not _is_admin(author):
        await ctx.reply("You lack permission to run this command.", mention_author=False)
        return None

    await ensure_state_restored()

    try:
        await ctx.message.delete()
    except discord.HTTPException:
        pass

    guild = ctx.guild
    await ctx.channel.send("Initiating full TF reset...", delete_after=5)

    states = [
        state for state in list(active_transformations.values()) if state.guild_id == guild.id
    ]
    restored = 0
    for state in states:
        await revert_transformation(state, expired=False)
        restored += 1

    await send_history_message(
        "TF Reset",
        f"Triggered by: **{author.name}**\nRestored TFs: {restored}",
    )
    await ctx.channel.send(f"TF reset completed. Restored {restored} transformations.", delete_after=10)


@bot.command(name="tf", aliases=["TF"])
async def tf_stats_command(ctx: commands.Context):
    guild_id = ctx.guild.id if ctx.guild else None
    if guild_id is None:
        await ctx.reply(
            "Run this command from a server so I know which TF roster to check.",
            mention_author=False,
        )
        return None

    guild_data = tf_stats.get(str(guild_id), {})
    user_data = guild_data.get(str(ctx.author.id))

    if not user_data:
        try:
            await ctx.message.delete()
        except discord.HTTPException:
            pass
        try:
            await ctx.author.send(
                "You haven't experienced any transformations yet."
            )
        except discord.Forbidden:
            await ctx.reply(
                "I couldn't DM you. Please enable direct messages from server members.",
                mention_author=False,
                delete_after=10,
            )
        return None

    embed = discord.Embed(
        title="Transformation Stats",
        color=0x9B59B6,
        timestamp=_now(),
    )
    avatar_url = (
        ctx.author.display_avatar.url if ctx.author.display_avatar else None
    )
    embed.set_author(
        name=ctx.author.display_name,
        icon_url=avatar_url,
    )
    embed.add_field(
        name="Total Transformations",
        value=str(user_data.get("total", 0)),
        inline=False,
    )

    characters = user_data.get("characters", {})
    if characters:
        sorted_chars = sorted(characters.items(), key=lambda item: item[1], reverse=True)
        lines = [f"- {name}: **{count}**" for name, count in sorted_chars]
        embed.add_field(name="By Character", value="\n".join(lines), inline=False)

    key = _state_key(guild_id, ctx.author.id)
    current_state = active_transformations.get(key)
    if current_state:
        remaining = max(
            (current_state.expires_at - _now()).total_seconds(),
            0,
        )
        minutes, seconds = divmod(int(remaining), 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            remaining_str = f"{hours}h {minutes}m {seconds}s"
        elif minutes:
            remaining_str = f"{minutes}m {seconds}s"
        else:
            remaining_str = f"{seconds}s"

        embed.add_field(
            name="Current Transformation",
            value=f"Character: **{current_state.character_name}**\nTime left: `{remaining_str}`",
            inline=False,
        )

    try:
        await ctx.author.send(embed=embed)
        if current_state:
            outfits = list_available_outfits(current_state.character_name)
            if outfits:
                selected = get_selected_outfit_name(current_state.character_name)
                formatted = []
                for option in outfits:
                    label = option
                    if selected and option.lower() == selected.lower():
                        label = f"{option} (current)"
                    formatted.append(label)
                outfit_note = (
                    f"Outfits available for {current_state.character_name}: {', '.join(formatted)}.\n"
                    "Use `!outfit <name>` while you are transformed to switch outfits."
                )
                try:
                    await ctx.author.send(outfit_note)
                except discord.Forbidden:
                    pass
    except discord.Forbidden:
        await ctx.reply(
            "I couldn't DM you. Please enable direct messages from server members.",
            mention_author=False,
            delete_after=10,
        )
    finally:
        try:
            await ctx.message.delete()
        except discord.HTTPException:
            pass



@bot.command(name="outfit")
async def outfit_command(ctx: commands.Context, *, outfit_name: str = ""):
    outfit_name = outfit_name.strip()
    if not outfit_name:
        message = "Usage: !outfit <outfit name>"
        if ctx.guild:
            await ctx.reply(message, mention_author=False)
        else:
            await ctx.send(message)
        return

    guild_id = ctx.guild.id if ctx.guild else None
    state = find_active_transformation(ctx.author.id, guild_id)
    if not state:
        message = "You need to be transformed to change outfits."
        if ctx.guild:
            await ctx.reply(message, mention_author=False)
        else:
            await ctx.send(message)
        return

    outfits = list_available_outfits(state.character_name)
    if not outfits:
        message = f"No outfits are available for {state.character_name}."
        if ctx.guild:
            await ctx.reply(message, mention_author=False)
        else:
            await ctx.send(message)
        return

    normalized = outfit_name.lower()
    match = next((o for o in outfits if o.lower() == normalized), None)
    if match is None:
        message = f"Unknown outfit `{outfit_name}`. Available: {', '.join(outfits)}"
        if ctx.guild:
            await ctx.reply(message, mention_author=False)
        else:
            await ctx.send(message)
        return

    if not set_selected_outfit_name(state.character_name, match):
        message = "Unable to update outfit at this time."
        if ctx.guild:
            await ctx.reply(message, mention_author=False)
        else:
            await ctx.send(message)
        return

    confirmation = f"Outfit for {state.character_name} set to `{match}`. Future messages will use this outfit."
    if ctx.guild:
        await ctx.reply(confirmation, mention_author=False)
    else:
        await ctx.send(confirmation)

@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return None

    logger.info(
        "Message %s from %s in channel %s (dev=%s)",
        message.id,
        message.author.id,
        getattr(message.channel, "id", "dm"),
        DEV_MODE,
    )

    command_invoked = False
    ctx = await bot.get_context(message)
    if ctx.command:
        command_invoked = True
        await bot.invoke(ctx)

    if command_invoked:
        return None

    is_admin_user = _is_admin(message.author)
    if message.guild and message.guild.owner_id == message.author.id:
        logger.debug("Ignoring message %s from server owner %s", message.id, message.author.id)
        return None
    channel_id = getattr(message.channel, "id", None)
    if DEV_MODE and ALLOWED_CHANNEL_IDS and channel_id not in ALLOWED_CHANNEL_IDS:
        logger.info(
            "Skipping message %s: channel %s not in dev allow list (%s)",
            message.id,
            channel_id,
            ", ".join(str(c) for c in ALLOWED_CHANNEL_IDS),
        )
        return None

    if not DEV_MODE and channel_id in IGNORED_CHANNEL_IDS:
        logger.info(
            "Skipping message %s: channel %s is in the ignore list (%s)",
            message.id,
            channel_id,
            ", ".join(str(c) for c in IGNORED_CHANNEL_IDS),
        )
        return None

    if message.guild:
        key = _state_key(message.guild.id, message.author.id)
        state = active_transformations.get(key)
        if state:
            reply_reference: Optional[discord.MessageReference] = (
                message.to_reference(fail_if_not_exists=False) if message.reference else None
            )
            await relay_transformed_message(message, state, reference=reply_reference)
            return None

    logger.info(
        "Message intercepted (dev=%s, admin=%s): user %s in channel %s",
        DEV_MODE,
        is_admin_user,
        message.author.id,
        channel_id,
    )

    roll = random.random()
    logger.debug("Roll for message %s is %.4f vs threshold %.4f", message.id, roll, TF_CHANCE)
    if roll <= TF_CHANCE:
        state = await handle_transformation(message)
        if state:
            logger.debug("TF triggered for message %s in channel %s", message.id, channel_id)


def main():
    args = parse_args()
    if args.dev_mode:
        enable_dev_mode()
    bot.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()
