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

load_dotenv()

try:
    import yaml
except ImportError:
    yaml = None

from tf_characters import TF_CHARACTERS as CHARACTER_DATA
from ai_rewriter import AI_REWRITE_ENABLED, rewrite_message_for_character


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
_VN_BG_ROOT_SETTING = os.getenv("TFBOT_VN_BG_ROOT", "").strip()
_VN_BG_DEFAULT_SETTING = os.getenv("TFBOT_VN_BG_DEFAULT", "school/cafeteria.png").strip()
VN_BACKGROUND_DEFAULT_RELATIVE = Path(_VN_BG_DEFAULT_SETTING) if _VN_BG_DEFAULT_SETTING else None
if _VN_BG_ROOT_SETTING:
    candidate_bg_root = Path(_VN_BG_ROOT_SETTING).expanduser()
    VN_BACKGROUND_ROOT = candidate_bg_root if candidate_bg_root.exists() else None
elif VN_GAME_ROOT:
    candidate_bg_root = VN_GAME_ROOT / "game" / "images" / "bg"
    VN_BACKGROUND_ROOT = candidate_bg_root if candidate_bg_root.exists() else None
else:
    VN_BACKGROUND_ROOT = None
if VN_BACKGROUND_ROOT and VN_BACKGROUND_DEFAULT_RELATIVE:
    VN_BACKGROUND_DEFAULT_PATH = (VN_BACKGROUND_ROOT / VN_BACKGROUND_DEFAULT_RELATIVE).resolve()
    if not VN_BACKGROUND_DEFAULT_PATH.exists():
        logger.warning(
            "VN background: default background %s does not exist under %s",
            VN_BACKGROUND_DEFAULT_RELATIVE,
            VN_BACKGROUND_ROOT,
        )
        VN_BACKGROUND_DEFAULT_PATH = None
else:
    VN_BACKGROUND_DEFAULT_PATH = None
_BG_SELECTION_FILE_SETTING = os.getenv("TFBOT_VN_BG_SELECTIONS", "tf_backgrounds.json").strip()
VN_BACKGROUND_SELECTION_FILE = Path(_BG_SELECTION_FILE_SETTING) if _BG_SELECTION_FILE_SETTING else None
VN_NAME_DEFAULT_COLOR: Tuple[int, int, int, int] = (255, 220, 180, 255)
_VN_CACHE_DIR_SETTING = os.getenv("TFBOT_VN_CACHE_DIR", "vn_cache").strip()
TRANSFORM_DURATION_CHOICES: Sequence[Tuple[str, timedelta]] = [
    ("10 minutes", timedelta(minutes=10)),
    ("1 hour", timedelta(hours=1)),
    ("10 hours", timedelta(hours=10)),
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
_VN_BACKGROUND_IMAGES: list[Path] = []
background_selections: Dict[str, str] = {}

@dataclass(frozen=True)
class TFCharacter:
    name: str
    avatar_path: str
    message: str


@dataclass(frozen=True)
class OutfitAsset:
    name: str
    base_path: Path
    accessory_layers: Sequence[Path] = ()


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
    original_display_name: str = ""


@dataclass
class ReplyContext:
    author: str
    text: str


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


def _member_profile_name(member: discord.Member) -> str:
    """Return the user's profile name, ignoring any server nickname."""
    global_name = getattr(member, "global_name", None)
    if isinstance(global_name, str) and global_name.strip():
        return global_name.strip()
    return member.name


def _normalize_pose_name(pose: Optional[str]) -> Optional[str]:
    if pose is None:
        return None
    stripped = pose.strip()
    if not stripped:
        return None
    return stripped.lower()


def _load_background_images() -> Sequence[Path]:
    global _VN_BACKGROUND_IMAGES
    if _VN_BACKGROUND_IMAGES:
        return _VN_BACKGROUND_IMAGES
    if VN_BACKGROUND_ROOT and VN_BACKGROUND_ROOT.exists():
        try:
            _VN_BACKGROUND_IMAGES = sorted(
                path
                for path in VN_BACKGROUND_ROOT.rglob("*.png")
                if path.is_file()
            )
        except OSError as exc:
            logger.warning("VN background: failed to scan directory %s: %s", VN_BACKGROUND_ROOT, exc)
            _VN_BACKGROUND_IMAGES = []
    else:
        _VN_BACKGROUND_IMAGES = []
    return _VN_BACKGROUND_IMAGES


def _compose_background_layer(panel_size: Tuple[int, int], background_path: Optional[Path]) -> Optional["Image.Image"]:
    backgrounds = _load_background_images()
    if background_path is None or not background_path.exists():
        if not backgrounds:
            return None
        background_path = random.choice(backgrounds)
    try:
        from PIL import Image, ImageOps
        with Image.open(background_path) as background_image:
            fitted = ImageOps.fit(
                background_image.convert("RGBA"),
                panel_size,
                Image.LANCZOS,
                centering=(0.5, 0.5),
            )
    except OSError as exc:
        logger.warning("VN background: failed to load %s: %s", background_path, exc)
        try:
            _VN_BACKGROUND_IMAGES.remove(background_path)
        except ValueError:
            pass
        return None
    layer = Image.new("RGBA", panel_size, (0, 0, 0, 0))
    layer.paste(fitted, (0, 0), fitted)
    return layer


def list_background_choices() -> Sequence[Path]:
    return list(_load_background_images())


def get_selected_background_path(user_id: int) -> Optional[Path]:
    if VN_BACKGROUND_ROOT is None:
        return None
    key = str(user_id)
    selected = background_selections.get(key)
    if selected:
        candidate = (VN_BACKGROUND_ROOT / selected).resolve()
        if candidate.exists():
            return candidate
        logger.warning("VN background: stored selection %s missing for user %s", selected, user_id)
        background_selections.pop(key, None)
        persist_background_selections()
    if VN_BACKGROUND_DEFAULT_PATH and VN_BACKGROUND_DEFAULT_PATH.exists():
        return VN_BACKGROUND_DEFAULT_PATH
    backgrounds = _load_background_images()
    if backgrounds:
        return backgrounds[0]
    return None


def set_selected_background(user_id: int, background_path: Path) -> bool:
    if VN_BACKGROUND_ROOT is None:
        return False
    relative = _relative_background_path(background_path)
    if not relative:
        return False
    background_selections[str(user_id)] = relative
    persist_background_selections()
    return True


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
        "original_display_name": state.original_display_name,
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
        original_display_name=str(payload.get("original_display_name", "") or ""),
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



def load_outfit_selections() -> Dict[str, Dict[str, str]]:
    if not VN_SELECTION_FILE.exists():
        return {}
    try:
        data = json.loads(VN_SELECTION_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            normalized: Dict[str, Dict[str, str]] = {}
            for key, value in data.items():
                entry: Dict[str, str] = {}
                pose_value: Optional[str] = None
                outfit_value: Optional[str] = None
                if isinstance(value, dict):
                    pose_raw = value.get("pose")
                    outfit_raw = value.get("outfit") or value.get("name")
                    if isinstance(pose_raw, str):
                        pose_value = pose_raw.strip()
                    elif pose_raw is not None:
                        pose_value = str(pose_raw).strip()
                    if isinstance(outfit_raw, str):
                        outfit_value = outfit_raw.strip()
                    elif outfit_raw is not None:
                        outfit_value = str(outfit_raw).strip()
                elif isinstance(value, str):
                    outfit_value = value.strip()
                elif value is not None:
                    outfit_value = str(value).strip()

                if outfit_value:
                    if pose_value:
                        entry["pose"] = pose_value
                    entry["outfit"] = outfit_value
                    normalized[str(key).lower()] = entry
            return normalized
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse %s: %s", VN_SELECTION_FILE, exc)
    return {}

def persist_outfit_selections() -> None:
    try:
        VN_SELECTION_FILE.parent.mkdir(parents=True, exist_ok=True)
        VN_SELECTION_FILE.write_text(json.dumps(vn_outfit_selection, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning("Failed to persist outfit selections: %s", exc)


def _relative_background_path(path: Path) -> Optional[str]:
    if VN_BACKGROUND_ROOT is None:
        return None
    try:
        relative = path.resolve().relative_to(VN_BACKGROUND_ROOT.resolve())
    except ValueError:
        return None
    return relative.as_posix()


def load_background_selections() -> Dict[str, str]:
    if VN_BACKGROUND_SELECTION_FILE is None or not VN_BACKGROUND_SELECTION_FILE.exists():
        return {}
    try:
        data = json.loads(VN_BACKGROUND_SELECTION_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            normalized: Dict[str, str] = {}
            for key, value in data.items():
                if not isinstance(value, str):
                    value = str(value)
                normalized[str(key)] = value.strip()
            return normalized
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse %s: %s", VN_BACKGROUND_SELECTION_FILE, exc)
    return {}


def persist_background_selections() -> None:
    if VN_BACKGROUND_SELECTION_FILE is None:
        return
    try:
        VN_BACKGROUND_SELECTION_FILE.parent.mkdir(parents=True, exist_ok=True)
        VN_BACKGROUND_SELECTION_FILE.write_text(json.dumps(background_selections, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning("Failed to persist background selections: %s", exc)

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
background_selections = load_background_selections()


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
        if not state.original_display_name:
            state.original_display_name = _member_profile_name(member)
        desired_nick = state.original_nick if state.original_nick else None
        attempts: list[tuple[Optional[str], str]] = []
        attempts.append((desired_nick, reason))
        if desired_nick is None and state.original_display_name:
            attempts.append((state.original_display_name, f"{reason} (profile fallback)"))
        restored = False
        for nick_value, attempt_reason in attempts:
            try:
                await member.edit(nick=nick_value, reason=attempt_reason)
            except (discord.Forbidden, discord.HTTPException) as exc:
                logger.warning(
                    "Failed to restore nickname for %s (value=%s): %s",
                    member.id,
                    nick_value,
                    exc,
                )
                continue
            restored = True
            break
        if not restored:
            logger.warning(
                "Unable to restore nickname for %s in guild %s after TF revert.",
                member.id,
                state.guild_id,
            )
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


def _load_character_context() -> Dict[str, str]:
    context_path = BASE_DIR / "data" / "character_context.json"
    if not context_path.exists():
        return {}
    try:
        return json.loads(context_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load character context dataset: %s", exc)
        return {}


CHARACTER_CONTEXT = _load_character_context()

_REPLY_LOG_SETTING = os.getenv("TFBOT_REPLY_LOG", "transform_replies.json").strip()
if _REPLY_LOG_SETTING:
    _reply_path = Path(_REPLY_LOG_SETTING)
    if not _reply_path.is_absolute():
        REPLY_LOG_FILE = (BASE_DIR / _reply_path).resolve()
    else:
        REPLY_LOG_FILE = _reply_path.resolve()
else:
    REPLY_LOG_FILE = (BASE_DIR / "transform_replies.json").resolve()


def _load_reply_log() -> Dict[int, ReplyContext]:
    if not REPLY_LOG_FILE.exists():
        return {}
    try:
        raw = json.loads(REPLY_LOG_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read reply log %s: %s", REPLY_LOG_FILE, exc)
        return {}
    result: Dict[int, ReplyContext] = {}
    for key, value in raw.items():
        try:
            message_id = int(key)
            author = value.get("author", "Unknown")
            text = value.get("text", "")
        except Exception:  # pylint: disable=broad-except
            continue
        if text:
            result[message_id] = ReplyContext(author=author, text=text)
    return result


def _persist_reply_log() -> None:
    try:
        REPLY_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            str(message_id): {"author": ctx.author, "text": ctx.text}
            for message_id, ctx in TRANSFORM_MESSAGE_LOG.items()
        }
        REPLY_LOG_FILE.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("Failed to persist reply log %s: %s", REPLY_LOG_FILE, exc)


TRANSFORM_MESSAGE_LOG: Dict[int, ReplyContext] = _load_reply_log()


def _register_relay_message(message_id: int, author: str, text: str) -> None:
    if not text:
        return
    TRANSFORM_MESSAGE_LOG[message_id] = ReplyContext(author=author, text=text)
    if len(TRANSFORM_MESSAGE_LOG) > 500:
        for key in list(TRANSFORM_MESSAGE_LOG.keys())[:100]:
            TRANSFORM_MESSAGE_LOG.pop(key, None)
    _persist_reply_log()
    logger.debug("Reply log registered id=%s author=%s text=%s", message_id, author, text[:120])


async def _resolve_reply_context(message: discord.Message) -> Optional[ReplyContext]:
    reference = message.reference
    if not reference or not reference.message_id:
        return None

    cached = TRANSFORM_MESSAGE_LOG.get(reference.message_id)
    resolved_msg = reference.resolved
    target_msg: Optional[discord.Message] = None

    if isinstance(resolved_msg, discord.Message):
        target_msg = resolved_msg
    else:
        try:
            target_msg = await message.channel.fetch_message(reference.message_id)  # type: ignore[arg-type]
        except discord.HTTPException as exc:
            logger.debug("Unable to fetch referenced message %s: %s", reference.message_id, exc)
            target_msg = None

    if target_msg is None:
        if cached:
            logger.debug("Reply context resolved from cache for %s", reference.message_id)
        return cached

    author = getattr(target_msg.author, "display_name", None) or getattr(
        target_msg.author, "name", "Unknown"
    )

    content = (target_msg.content or "").strip()
    if content:
        context = ReplyContext(author=author, text=content)
        TRANSFORM_MESSAGE_LOG[reference.message_id] = context
        _persist_reply_log()
        logger.debug("Reply context resolved from message %s", reference.message_id)
        return context

    if cached:
        return cached

    if target_msg.embeds:
        embed = target_msg.embeds[0]
        if embed.description:
            context = ReplyContext(author=author, text=embed.description.strip())
            TRANSFORM_MESSAGE_LOG[reference.message_id] = context
            _persist_reply_log()
            logger.debug("Reply context resolved from embed %s", reference.message_id)
            return context

    return None


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


def _font_line_height(font) -> int:
    try:
        bbox = font.getbbox("Ag")
        return max(1, bbox[3] - bbox[1])
    except Exception:  # pylint: disable=broad-except
        return getattr(font, "size", VN_TEXT_FONT_SIZE)


def _prepare_reply_snippet(text: str, limit: int = 180) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def _wrap_plain_text(draw, text: str, font, max_width: int) -> Sequence[str]:
    if not text:
        return []
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        width = draw.textlength(candidate, font=font)
        if current and width > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _truncate_text_to_width(draw, text: str, font, max_width: int) -> str:
    if not text:
        return "..."
    if max_width <= 0:
        return "..."
    ellipsis = "..."
    width = draw.textlength(text, font=font)
    if width <= max_width:
        return text
    truncated = text
    while truncated and draw.textlength(truncated + ellipsis, font=font) > max_width:
        truncated = truncated[:-1]
    if not truncated:
        return ellipsis
    return truncated.rstrip() + ellipsis


def _default_accessory_layer(accessory_dir: Path) -> Optional[Path]:
    if not accessory_dir.is_dir():
        return None
    pngs = sorted(p for p in accessory_dir.rglob("*.png") if p.is_file())
    if not pngs:
        return None
    for candidate in pngs:
        if candidate.stem.lower() == "on":
            return candidate
    for candidate in pngs:
        if "on" in candidate.stem.lower():
            return candidate
    for candidate in pngs:
        parents = [parent.name.lower() for parent in candidate.parents]
        if "on" in parents:
            return candidate
    return pngs[0]


def _discover_outfit_assets(variant_dir: Path) -> Dict[str, OutfitAsset]:
    assets: Dict[str, OutfitAsset] = {}
    outfits_dir = variant_dir / "outfits"
    if not outfits_dir.exists():
        return assets
    for entry in sorted(outfits_dir.iterdir(), key=lambda p: p.name.lower()):
        if entry.is_file() and entry.suffix.lower() == ".png":
            name = entry.stem
            assets[name.lower()] = OutfitAsset(name=name, base_path=entry, accessory_layers=())
        elif entry.is_dir():
            primary = entry / f"{entry.name}.png"
            if not primary.exists():
                primary = next((p for p in entry.glob("*.png")), None)
            if not primary:
                primary = next((p for p in entry.rglob("*.png")), None)
            if not primary:
                continue
            accessories: list[Path] = []
            for accessory_dir in sorted(
                (child for child in entry.iterdir() if child.is_dir()),
                key=lambda p: p.name.lower(),
            ):
                layer = _default_accessory_layer(accessory_dir)
                if layer:
                    accessories.append(layer)
            assets[entry.name.lower()] = OutfitAsset(
                name=entry.name,
                base_path=primary,
                accessory_layers=tuple(accessories),
            )
    return assets


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


def _crop_transparent_left(image: "Image.Image") -> "Image.Image":
    """Remove transparent columns from the left of an RGBA sprite while keeping height."""
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    if image.width <= 1:
        return image
    alpha = image.getchannel("A")
    pixels = alpha.load()
    left = 0
    right = image.width
    # move left bound until any non-transparent pixel is found
    while left < image.width:
        column = [pixels[left, y] for y in range(image.height)]
        if any(column):
            break
        left += 1
    if left >= right - 1:
        return image
    cropped = image.crop((left, 0, right, image.height))
    logger.debug("VN sprite: trimmed transparent left (%s -> %s)", image.size, cropped.size)
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


def _parse_hex_color(raw_color: str) -> Optional[Tuple[int, int, int, int]]:
    if not raw_color:
        return None
    value = raw_color.strip()
    if not value:
        return None
    if value.lower().startswith("0x"):
        value = value[2:]
    if value.startswith("#"):
        value = value[1:]
    if len(value) not in {6, 8}:
        return None
    try:
        components = [int(value[i : i + 2], 16) for i in range(0, len(value), 2)]
    except ValueError:
        return None
    if len(components) == 3:
        components.append(255)
    if len(components) != 4:
        return None
    r, g, b, a = components[:4]
    return r, g, b, a


def _resolve_character_name_color(character_name: str) -> Tuple[int, int, int, int]:
    if not character_name:
        return VN_NAME_DEFAULT_COLOR
    directory, _ = resolve_character_directory(character_name)
    if directory is None:
        return VN_NAME_DEFAULT_COLOR
    config = _load_character_config(directory)
    raw_color = config.get("name_color") or config.get("text_color")
    if isinstance(raw_color, str):
        parsed = _parse_hex_color(raw_color)
        if parsed:
            return parsed
    if isinstance(raw_color, Sequence) and not isinstance(raw_color, (str, bytes, bytearray)):
        components = list(raw_color)
        if 3 <= len(components) <= 4 and all(isinstance(c, (int, float)) for c in components):
            channel_values = [max(0, min(255, int(c))) for c in components[:4]]
            if len(channel_values) == 3:
                channel_values.append(255)
            if len(channel_values) == 4:
                r, g, b, a = channel_values
                return r, g, b, a
    return VN_NAME_DEFAULT_COLOR


def _select_variant_dir(character_dir: Path, config: Dict) -> Optional[Path]:
    ordered_variants = _ordered_variant_dirs(character_dir, config)
    if not ordered_variants:
        logger.warning("VN sprite: no variants found under %s", character_dir.name)
        return None
    return ordered_variants[0]


def _ordered_variant_dirs(character_dir: Path, config: Dict) -> Sequence[Path]:
    variants = {child.name.lower(): child for child in character_dir.iterdir() if child.is_dir()}
    if not variants:
        return []
    preferred_names: list[str] = []
    poses = config.get("poses")
    if isinstance(poses, dict):
        preferred_names.extend(name.lower() for name in poses.keys())
    preferred_names.append("a")
    preferred_names.extend(sorted(variants.keys()))

    ordered: list[Path] = []
    seen: set[str] = set()
    for name in preferred_names:
        if name in variants and name not in seen:
            ordered.append(variants[name])
            seen.add(name)
    return ordered


def _select_outfit_path(
    variant_dirs: Sequence[Path],
    config: Dict,
    preferred: Optional[str] = None,
    preferred_pose: Optional[str] = None,
) -> tuple[Optional[Path], Optional[OutfitAsset]]:
    if not variant_dirs:
        return None, None

    variant_outfits: dict[Path, Dict[str, OutfitAsset]] = {}
    for variant_dir in variant_dirs:
        assets = _discover_outfit_assets(variant_dir)
        if not assets:
            continue
        variant_outfits[variant_dir] = assets

    if not variant_outfits:
        logger.warning(
            "VN sprite: no outfits discovered for variants %s",
            ", ".join(v.name for v in variant_dirs),
        )
        return None, None

    search_variants = [variant for variant in variant_dirs if variant in variant_outfits]
    if not search_variants:
        return None, None

    normalized_pose = _normalize_pose_name(preferred_pose)
    if normalized_pose:
        search_variants.sort(key=lambda var: 0 if var.name.lower() == normalized_pose else 1)

    candidates: list[str] = []
    if preferred:
        candidates.append(preferred)
    default_outfit = config.get("default_outfit")
    if isinstance(default_outfit, str):
        candidates.append(default_outfit)
    if VN_DEFAULT_OUTFIT:
        candidates.append(VN_DEFAULT_OUTFIT)

    normalized_targets: list[str] = []
    for name in candidates:
        name = name.strip()
        if not name:
            continue
        lower = name.lower()
        if lower.endswith(".png"):
            normalized_targets.append(lower)
            normalized_targets.append(lower.rstrip(".png"))
        else:
            normalized_targets.append(lower)

    for target in normalized_targets:
        for variant_dir in search_variants:
            assets = variant_outfits.get(variant_dir, {})
            asset = assets.get(target)
            if asset is None:
                asset = assets.get(target.rstrip(".png"))
            if asset:
                logger.debug(
                    "VN sprite: using outfit %s for variant %s",
                    asset.base_path.name,
                    variant_dir.name,
                )
                return variant_dir, asset

    for variant_dir in search_variants:
        assets = variant_outfits.get(variant_dir, {})
        if assets:
            first_asset = next(iter(sorted(assets.values(), key=lambda a: a.name.lower())))
            logger.debug(
                "VN sprite: defaulting to first outfit %s for variant %s",
                first_asset.base_path.name,
                variant_dir.name,
            )
            return variant_dir, first_asset

    return None, None

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
    name = raw_name.strip()
    if not name:
        return []
    first_word = name.split(" ", 1)[0].lower()
    if not first_word:
        return []
    candidates: list[str] = [first_word]
    stripped = first_word.replace('"', "").replace("'", "")
    if stripped and stripped not in candidates:
        candidates.append(stripped)
    hyphen_removed = stripped.replace("-", "")
    if hyphen_removed and hyphen_removed not in candidates:
        candidates.append(hyphen_removed)
    underscore_variant = stripped.replace("-", "_")
    if underscore_variant and underscore_variant not in candidates:
        candidates.append(underscore_variant)
    return candidates


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

def list_pose_outfits(character_name: str) -> Dict[str, list[str]]:
    directory, attempted = resolve_character_directory(character_name)
    if directory is None:
        logger.debug("VN sprite: cannot list outfits for %s (tried %s)", character_name, attempted)
        return {}
    config = _load_character_config(directory)
    variant_dirs = _ordered_variant_dirs(directory, config)
    if not variant_dirs:
        return {}
    pose_map: Dict[str, list[str]] = {}
    for variant_dir in variant_dirs:
        assets = _discover_outfit_assets(variant_dir)
        if assets:
            pose_map[variant_dir.name] = sorted(asset.name for asset in assets.values())
    return pose_map

def list_available_outfits(character_name: str) -> Sequence[str]:
    pose_map = list_pose_outfits(character_name)
    outfits: set[str] = set()
    for options in pose_map.values():
        outfits.update(options)
    return sorted(outfits)

def get_selected_outfit_name(character_name: str) -> Optional[str]:
    _, outfit = get_selected_pose_outfit(character_name)
    return outfit

def get_selected_pose_outfit(character_name: str) -> tuple[Optional[str], Optional[str]]:
    directory, _ = resolve_character_directory(character_name)
    if directory is None:
        return None, None
    return get_selected_pose_outfit_for_dir(directory)

def set_selected_outfit_name(character_name: str, outfit_name: str) -> bool:
    return set_selected_pose_outfit(character_name, None, outfit_name)

def set_selected_pose_outfit(character_name: str, pose_name: Optional[str], outfit_name: str) -> bool:
    directory, attempted = resolve_character_directory(character_name)
    if directory is None:
        logger.debug("VN sprite: cannot set outfit for %s (tried %s)", character_name, attempted)
        return False
    pose_outfits = list_pose_outfits(character_name)
    if not pose_outfits:
        return False

    normalized_outfit = outfit_name.strip()
    if not normalized_outfit:
        return False
    normalized_pose = _normalize_pose_name(pose_name)

    matched_pose: Optional[str] = None
    matched_outfit: Optional[str] = None

    for pose, outfits in pose_outfits.items():
        outfit_lookup = {option.lower(): option for option in outfits}
        option = outfit_lookup.get(normalized_outfit.lower())
        if not option:
            continue
        if normalized_pose is None or pose.lower() == normalized_pose:
            matched_pose = pose
            matched_outfit = option
            break

    if matched_outfit is None:
        # No direct match; allow fallback to any pose if specific pose requested but not found.
        if normalized_pose:
            logger.debug(
                "VN sprite: pose %s not available for outfit %s (character %s)",
                normalized_pose,
                outfit_name,
                character_name,
            )
            return False
        # Attempt to select first pose containing the outfit.
        for pose, outfits in pose_outfits.items():
            outfit_lookup = {option.lower(): option for option in outfits}
            option = outfit_lookup.get(normalized_outfit.lower())
            if option:
                matched_pose = pose
                matched_outfit = option
                break

    if matched_outfit is None or matched_pose is None:
        return False

    key = directory.name.lower()
    vn_outfit_selection[key] = {"pose": matched_pose, "outfit": matched_outfit}
    persist_outfit_selections()
    compose_game_avatar.cache_clear()
    logger.info(
        "VN sprite: outfit override for %s set to pose %s outfit %s",
        directory.name,
        matched_pose,
        matched_outfit,
    )
    return True

def get_selected_outfit_for_dir(directory: Path) -> Optional[str]:
    _, outfit = get_selected_pose_outfit_for_dir(directory)
    return outfit

def get_selected_pose_outfit_for_dir(directory: Path) -> tuple[Optional[str], Optional[str]]:
    entry = vn_outfit_selection.get(directory.name.lower())
    if not entry:
        return None, None
    pose: Optional[str] = None
    outfit: Optional[str] = None
    if isinstance(entry, dict):
        pose_raw = entry.get("pose")
        outfit_raw = entry.get("outfit") or entry.get("name")
        if isinstance(pose_raw, str):
            pose = pose_raw.strip() or None
        elif pose_raw is not None:
            pose = str(pose_raw).strip() or None
        if isinstance(outfit_raw, str):
            outfit = outfit_raw.strip() or None
        elif outfit_raw is not None:
            outfit = str(outfit_raw).strip() or None
    elif isinstance(entry, str):
        outfit = entry.strip() or None
    else:
        outfit = str(entry).strip() or None
    return pose, outfit



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
    variant_dirs = _ordered_variant_dirs(character_dir, config)
    if not variant_dirs:
        logger.debug("VN sprite: no variant directory found for %s", character_dir.name)
        return None

    preferred_pose, preferred_outfit = get_selected_pose_outfit_for_dir(character_dir)
    if preferred_outfit:
        logger.debug(
            "VN sprite: preferred outfit override for %s is pose %s outfit %s",
            character_dir.name,
            preferred_pose or "auto",
            preferred_outfit,
        )
    variant_dir, outfit_asset = _select_outfit_path(
        variant_dirs,
        config,
        preferred_outfit,
        preferred_pose,
    )
    if not variant_dir or not outfit_asset:
        logger.warning(
            "VN sprite: outfit not found for %s (variant %s)",
            character_dir.name,
            variant_dir.name if variant_dir else "unknown",
        )
        return None

    outfit_path = outfit_asset.base_path
    face_path = _select_face_path(variant_dir)

    cache_file: Optional[Path] = None
    if VN_CACHE_DIR:
        face_token = "noface"
        if face_path and face_path.exists():
            face_token = face_path.stem.lower()
        accessory_token = "noacc"
        if outfit_asset.accessory_layers:
            accessory_token = "-".join(layer.stem.lower() for layer in outfit_asset.accessory_layers)
        cache_dir = VN_CACHE_DIR / character_dir.name.lower() / variant_dir.name.lower()
        cache_file = cache_dir / f"{outfit_path.stem.lower()}__{face_token}__{accessory_token}.png"
        if cache_file.exists():
            try:
                cached = Image.open(cache_file).convert("RGBA")
                trimmed_cached = _crop_transparent_top(cached)
                trimmed_cached = _crop_transparent_left(trimmed_cached)
                if trimmed_cached.size != cached.size:
                    try:
                        trimmed_cached.save(cache_file)
                        logger.debug(
                            "VN sprite: refreshed cached avatar %s (%s -> %s)",
                            cache_file,
                            cached.size,
                            trimmed_cached.size,
                        )
                    except OSError as exc:
                        logger.warning("VN sprite: unable to refresh cached avatar %s: %s", cache_file, exc)
                else:
                    logger.debug("VN sprite: loaded cached avatar %s", cache_file)
                return trimmed_cached
            except OSError as exc:
                logger.warning("VN sprite: failed to load cached avatar %s: %s (rebuilding)", cache_file, exc)

    try:
        outfit_image = Image.open(outfit_path).convert("RGBA")
    except OSError as exc:
        logger.warning("Failed to load outfit %s: %s", outfit_path, exc)
        return None

    for layer_path in outfit_asset.accessory_layers:
        if not layer_path.exists():
            continue
        try:
            layer_image = Image.open(layer_path).convert("RGBA")
            outfit_image.paste(layer_image, (0, 0), layer_image)
        except OSError as exc:
            logger.warning("Failed to load accessory %s: %s", layer_path, exc)

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
    final_image = _crop_transparent_left(final_image)
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
    reply_context: Optional[ReplyContext] = None,
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
    background_path = get_selected_background_path(state.user_id)
    background_layer = _compose_background_layer(base.size, background_path)
    if background_layer:
        base = Image.alpha_composite(background_layer, base)
    draw = ImageDraw.Draw(base)

    from PIL import Image
    name_font = _load_vn_font(VN_NAME_FONT_SIZE)
    base_text_font = _load_vn_font(VN_TEXT_FONT_SIZE)

    # Layout constants derived from vn_base.png geometry.
    name_box = (170, 10, 303, 26)
    text_box = (178, 80, 775, 250)
    name_padding = 10
    text_padding = 12
    avatar_box = (4, 4, 250, 250)
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
    name_color = _resolve_character_name_color(state.character_name)
    draw.text((name_x, name_y), name_text, fill=name_color, font=name_font)

    working_content = message_content.strip()
    if not working_content:
        working_content = f"{original_name} remains quietly transformed..."

    max_width = text_box[2] - text_box[0] - text_padding * 2
    total_height = text_box[3] - text_box[1] - text_padding * 2

    reply_font = None
    reply_line: Optional[str] = None
    reply_block_height = 0

    if reply_context and reply_context.text:
        reply_font = _load_vn_font(max(10, VN_TEXT_FONT_SIZE - 10))
        label_text = f"Replying to {reply_context.author}: "
        snippet_text = _prepare_reply_snippet(reply_context.text).replace("\n", " ").strip()
        available_width = max_width - draw.textlength(label_text, font=reply_font)
        truncated_snippet = _truncate_text_to_width(draw, snippet_text, reply_font, available_width)
        reply_line = f"{label_text}{truncated_snippet}"
        reply_block_height = _font_line_height(reply_font) + 6
        logger.debug(
            "Rendering reply context for %s -> %s: %s",
            state.character_name,
            reply_context.author,
            snippet_text,
        )

    available_height = max(total_height - reply_block_height, _font_line_height(base_text_font) * 2)

    segments = list(formatted_segments) if formatted_segments else []
    if not segments:
        segments = list(parse_discord_formatting(working_content))
    lines, text_font = _fit_text_segments(draw, segments, base_text_font, max_width, available_height)
    text_y = text_box[1] + text_padding

    if reply_font and reply_line:
        reply_fill = (190, 190, 190, 255)
        text_x = text_box[0] + text_padding
        draw.text((text_x, text_y), reply_line, fill=reply_fill, font=reply_font)
        text_y += _font_line_height(reply_font) + 6

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
    reply_context = await _resolve_reply_context(message)
    if (
        AI_REWRITE_ENABLED
        and cleaned_content
        and not cleaned_content.startswith(str(bot.command_prefix))
    ):
        context_snippet = CHARACTER_CONTEXT.get(state.character_name) or state.character_message
        rewritten = await rewrite_message_for_character(
            original_text=cleaned_content,
            character_name=state.character_name,
            character_context=context_snippet,
            user_name=message.author.display_name,
        )
        if rewritten and rewritten.strip():
            logger.debug(
                "AI rewrite applied for %s: %s -> %s",
                state.character_name,
                cleaned_content[:120],
                rewritten[:120],
            )
            cleaned_content = rewritten.strip()

    description = cleaned_content if cleaned_content else "*no message content*"
    formatted_segments = parse_discord_formatting(cleaned_content) if cleaned_content else None
    custom_emoji_images: Dict[str, "Image.Image"] = {}

    files: list[discord.File] = []
    payload: dict = {}

    if MESSAGE_STYLE == "vn":
        if formatted_segments is None:
            formatted_segments = parse_discord_formatting(cleaned_content)
        custom_emoji_images = await prepare_custom_emoji_images(message, formatted_segments)
        if reply_context:
            logger.debug(
                "Replying panel: %s -> %s snippet=%s",
                state.character_name,
                reply_context.author,
                reply_context.text[:120],
            )
        vn_file = render_vn_panel(
            state=state,
            message_content=cleaned_content,
            character_display_name=state.character_name,
            original_name=message.author.display_name,
            attachment_id=str(message.id),
            formatted_segments=formatted_segments,
            custom_emoji_images=custom_emoji_images,
            reply_context=reply_context,
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

    sent_message: Optional[discord.Message] = None
    try:
        sent_message = await message.channel.send(**send_kwargs)
    except discord.HTTPException as exc:
        logger.warning("Failed to relay TF message %s: %s", message.id, exc)
        return False

    if sent_message and cleaned_content:
        _register_relay_message(sent_message.id, state.character_name, cleaned_content)

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
    profile_name = _member_profile_name(member)

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
        original_display_name=profile_name,
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

    original_name = profile_name
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
    else:
        logger.info("Dev channel %s is excluded while dev mode is OFF.", DEV_CHANNEL_ID)
        if IGNORED_CHANNEL_IDS:
            logger.info(
                "Ignoring additional channels: %s",
                ", ".join(str(cid) for cid in IGNORED_CHANNEL_IDS),
            )
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


@bot.command(name="reroll")
@commands.guild_only()
async def reroll_command(ctx: commands.Context, *, args: str = ""):
    author = ctx.author
    if not isinstance(author, discord.Member):
        await ctx.reply("This command can only be used inside a server.", mention_author=False)
        return None
    if not _is_admin(author):
        await ctx.reply("You lack permission to run this command.", mention_author=False)
        return None

    guild = ctx.guild
    target_member: Optional[discord.Member] = None
    state: Optional[TransformationState] = None

    forced_character: Optional[TFCharacter] = None
    query = args.strip().lower()
    if query:
        parts = query.split()
        first_token = parts[0]
        if len(parts) > 1:
            forced_name = parts[1]
            forced_match = next((c for c in CHARACTER_POOL if c.name.split(" ", 1)[0].lower() == forced_name), None)
            if forced_match is None:
                await ctx.reply(
                    f"Unknown target character `{forced_name}`. Provide a valid character first name.",
                    mention_author=False,
                )
                return None
            forced_character = forced_match
        matching_states = [
            s
            for s in active_transformations.values()
            if s.guild_id == guild.id and s.character_name.split(" ", 1)[0].lower() == first_token
        ]
        if not matching_states:
            await ctx.reply(
                f"No active transformation found for character `{first_token}`.",
                mention_author=False,
            )
            return None
        state = matching_states[0]
        _, target_member = await fetch_member(state.guild_id, state.user_id)
        if target_member is None:
            await ctx.reply(
                f"Unable to locate the member transformed into {state.character_name}.",
                mention_author=False,
            )
            return None
    else:
        target_member = author
        key = _state_key(guild.id, target_member.id)
        state = active_transformations.get(key)
        if state is None:
            await ctx.reply(
                "You are not currently transformed; nothing to reroll.",
                mention_author=False,
            )
            return None

    key = _state_key(guild.id, target_member.id)
    current_state = active_transformations.get(key)
    if current_state is None or current_state != state:
        await ctx.reply(
            "Unable to locate the transformation for this member.",
            mention_author=False,
        )
        return None

    used_characters = {
        current_state.character_name
        for current_key, current_state in active_transformations.items()
        if current_key != key
    }
    available_characters = [
        character
        for character in CHARACTER_POOL
        if character.name not in used_characters and character.name != state.character_name
    ]
    forced_mode = forced_character is not None

    if forced_mode:
        if forced_character.name == state.character_name:
            await ctx.reply(
                f"They are already transformed into {forced_character.name}.",
                mention_author=False,
            )
            return None
        if forced_character.name in used_characters:
            await ctx.reply(
                f"{forced_character.name} is already in use by another transformation.",
                mention_author=False,
            )
            return None
        new_character = forced_character
    else:
        if not available_characters:
            await ctx.reply(
                "No alternative characters are available to reroll right now.",
                mention_author=False,
            )
            return None
        new_character = random.choice(available_characters)
    try:
        await target_member.edit(nick=new_character.name, reason="TF reroll")
    except (discord.Forbidden, discord.HTTPException) as exc:
        logger.warning("Failed to update nickname for reroll on %s: %s", target_member.id, exc)
        await ctx.reply(
            "I couldn't update their nickname, so the reroll was cancelled.",
            mention_author=False,
        )
        return None

    previous_character = state.character_name
    state.character_name = new_character.name
    state.character_avatar_path = new_character.avatar_path
    state.character_message = new_character.message
    state.avatar_applied = False
    persist_states()

    increment_tf_stats(guild.id, target_member.id, new_character.name)

    history_details = (
        f"Triggered by: **{author.display_name}**\n"
        f"Member: **{target_member.display_name}**\n"
        f"Previous Character: **{previous_character}**\n"
        f"New Character: **{new_character.name}**"
    )
    if forced_mode:
        history_details += "\nReason: Forced reroll override."
    await send_history_message(
        "TF Rerolled",
        history_details,
    )

    original_name = _member_profile_name(target_member)
    if forced_mode:
        custom_template = (
            "barely has time to react before Syn swoops in with a grin and swaps them straight into {character}. Syn just had to spice things up."
        )
        response_text = _format_character_message(
            custom_template,
            original_name,
            target_member.mention,
            state.duration_label,
            new_character.name,
        )
    else:
        response_text = _format_character_message(
            new_character.message,
            original_name,
            target_member.mention,
            state.duration_label,
            new_character.name,
        )
    emoji_prefix = _get_magic_emoji(guild)
    try:
        await ctx.channel.send(
            f"{emoji_prefix} {response_text}",
            allowed_mentions=discord.AllowedMentions(users=[target_member]),
        )
    except discord.HTTPException as exc:
        logger.warning("Failed to announce reroll in channel %s: %s", ctx.channel.id, exc)

    try:
        await ctx.message.delete()
    except discord.HTTPException:
        pass

    summary_message = f"{target_member.display_name} has been rerolled into **{new_character.name}**."
    if forced_mode:
        summary_message += " (Syn insisted on this one.)"
    await ctx.send(
        summary_message,
        delete_after=10,
    )


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
            pose_outfits = list_pose_outfits(current_state.character_name)
            if pose_outfits:
                selected_pose, selected_outfit = get_selected_pose_outfit(current_state.character_name)
                selected_pose_normalized = _normalize_pose_name(selected_pose)
                selected_outfit_normalized = (
                    selected_outfit.lower() if selected_outfit else None
                )
                pose_lines: list[str] = []
                for pose, options in pose_outfits.items():
                    entries: list[str] = []
                    for option in options:
                        display = option
                        if (
                            selected_outfit_normalized
                            and option.lower() == selected_outfit_normalized
                            and (
                                selected_pose_normalized is None
                                or pose.lower() == selected_pose_normalized
                            )
                        ):
                            display = f"{option} (current)"
                        entries.append(display)
                    pose_lines.append(f"{pose}: {', '.join(entries)}")
                outfit_note = (
                    f"Outfits available for {current_state.character_name}:\n"
                    + "\n".join(f"- {line}" for line in pose_lines)
                    + "\nUse `!outfit <outfit>` to pick by name or "
                    + "`!outfit <pose> <outfit>` (you can also separate with ':' or '/')."
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


@bot.command(name="bg")
async def background_command(ctx: commands.Context, *, selection: str = ""):
    try:
        await ctx.message.delete()
    except discord.HTTPException:
        pass

    if VN_BACKGROUND_ROOT is None:
        try:
            await ctx.author.send("Backgrounds are not configured on this bot.")
        except discord.Forbidden:
            if ctx.guild:
                await ctx.send("I couldn't DM you. Please enable direct messages.", delete_after=10)
        return

    choices = list_background_choices()
    if not choices:
        try:
            await ctx.author.send("No background images were found in the configured directory.")
        except discord.Forbidden:
            if ctx.guild:
                await ctx.send("I couldn't DM you. Please enable direct messages.", delete_after=10)
        return

    selection = selection.strip()
    if not selection:
        lines: list[str] = []
        for idx, path in enumerate(choices, start=1):
            try:
                relative = path.resolve().relative_to(VN_BACKGROUND_ROOT.resolve())
                display = relative.as_posix()
            except ValueError:
                display = str(path)
            lines.append(f"{idx}: {display}")

        chunks: list[str] = []
        current: list[str] = []
        length = 0
        for line in lines:
            if length + len(line) + 1 > 1900 and current:
                chunks.append("\n".join(current))
                current = []
                length = 0
            current.append(line)
            length += len(line) + 1
        if current:
            chunks.append("\n".join(current))

        default_display = (
            VN_BACKGROUND_DEFAULT_RELATIVE.as_posix()
            if VN_BACKGROUND_DEFAULT_RELATIVE
            else "system default"
        )
        instructions = (
            "Use `!bg <number>` to apply that background to your VN panel.\n"
            "Example: `!bg 45` selects option 45 from the list.\n"
            f"The default background is `{default_display}`."
        )

        try:
            for chunk in chunks:
                await ctx.author.send(f"```\n{chunk}\n```")
            await ctx.author.send(instructions)
        except discord.Forbidden:
            if ctx.guild:
                await ctx.send("I couldn't DM you. Please enable direct messages, then rerun `!bg`.", delete_after=10)
            return

        return

    try:
        index = int(selection)
    except ValueError:
        try:
            await ctx.author.send(f"`{selection}` isn't a valid background number. Use `!bg` with no arguments to see the list.")
        except discord.Forbidden:
            if ctx.guild:
                await ctx.send("I couldn't DM you. Please enable direct messages.", delete_after=10)
        return

    if index < 1 or index > len(choices):
        try:
            await ctx.author.send(f"Background number must be between 1 and {len(choices)}.")
        except discord.Forbidden:
            if ctx.guild:
                await ctx.send("I couldn't DM you. Please enable direct messages.", delete_after=10)
        return

    selected_path = choices[index - 1]
    if not set_selected_background(ctx.author.id, selected_path):
        try:
            await ctx.author.send("Unable to update your background at this time.")
        except discord.Forbidden:
            if ctx.guild:
                await ctx.send("I couldn't DM you. Please enable direct messages.", delete_after=10)
        return

    try:
        relative = selected_path.resolve().relative_to(VN_BACKGROUND_ROOT.resolve())
        display = relative.as_posix()
    except ValueError:
        display = str(selected_path)

    try:
        await ctx.author.send(f"Background set to `{display}`.")
    except discord.Forbidden:
        if ctx.guild:
            await ctx.send("I couldn't DM you. Please enable direct messages.", delete_after=10)



@bot.command(name="outfit")
async def outfit_command(ctx: commands.Context, *, outfit_name: str = ""):
    outfit_name = outfit_name.strip()
    if not outfit_name:
        message = "Usage: !outfit <outfit>` or `!outfit <pose> <outfit>`"
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

    pose_outfits = list_pose_outfits(state.character_name)
    if not pose_outfits:
        message = f"No outfits are available for {state.character_name}."
        if ctx.guild:
            await ctx.reply(message, mention_author=False)
        else:
            await ctx.send(message)
        return

    parsed_pose: Optional[str] = None
    parsed_outfit: Optional[str] = None

    for separator in (":", "/"):
        if separator in outfit_name:
            left, right = outfit_name.split(separator, 1)
            parsed_pose = left.strip()
            parsed_outfit = right.strip()
            break

    if parsed_outfit is None:
        parts = outfit_name.split()
        if len(parts) >= 2:
            parsed_pose = parts[0].strip()
            parsed_outfit = " ".join(parts[1:]).strip()
        else:
            parsed_outfit = outfit_name

    if not parsed_outfit:
        message = "Please provide the outfit to select. Example: `!outfit cheer` or `!outfit b cheer`."
        if ctx.guild:
            await ctx.reply(message, mention_author=False)
        else:
            await ctx.send(message)
        return

    if parsed_pose:
        normalized_pose = _normalize_pose_name(parsed_pose)
        known_poses = {pose.lower() for pose in pose_outfits.keys()}
        if normalized_pose not in known_poses:
            message = (
                f"Unknown pose `{parsed_pose}`. Available poses: {', '.join(pose_outfits.keys())}."
            )
            if ctx.guild:
                await ctx.reply(message, mention_author=False)
            else:
                await ctx.send(message)
            return
    else:
        normalized_pose = None

    if not set_selected_pose_outfit(state.character_name, parsed_pose if normalized_pose else None, parsed_outfit):
        pose_lines = []
        for pose, options in pose_outfits.items():
            pose_lines.append(f"{pose}: {', '.join(options)}")
        message = (
            f"Unable to update outfit. Available options:\n"
            + "\n".join(f"- {line}" for line in pose_lines)
        )
        if ctx.guild:
            await ctx.reply(message, mention_author=False)
        else:
            await ctx.send(message)
        return

    selected_pose, selected_outfit = get_selected_pose_outfit(state.character_name)
    pose_label = selected_pose or "auto"
    outfit_label = selected_outfit or parsed_outfit
    confirmation = (
        f"Outfit for {state.character_name} set to `{outfit_label}` (pose `{pose_label}`). "
        "Future messages will use this combination."
    )
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

    if not DEV_MODE and channel_id == DEV_CHANNEL_ID:
        logger.info(
            "Skipping message %s: dev channel %s is disabled while dev mode is off.",
            message.id,
            channel_id,
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
