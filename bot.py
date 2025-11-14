import asyncio
import importlib.util
import io
import json
import logging
import os
import random
import re
from collections import deque
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

import aiohttp
import discord
from discord.ext import commands
from dotenv import load_dotenv

load_dotenv()

try:
    import yaml
except ImportError:
    yaml = None

BASE_DIR = Path(__file__).resolve().parent

from tf_characters import TF_CHARACTERS as _DEFAULT_CHARACTER_DATA
from ai_rewriter import AI_REWRITE_ENABLED, rewrite_message_for_character
from tfbot.models import (
    OutfitAsset,
    ReplyContext,
    TFCharacter,
    TransformationState,
    TransformKey,
)
import tfbot.state as tf_state
from tfbot.state import (
    active_transformations,
    configure_state,
    find_active_transformation,
    get_last_reroll_timestamp,
    increment_tf_stats,
    load_reroll_cooldowns_from_disk,
    load_states_from_disk,
    load_stats_from_disk,
    persist_states,
    persist_stats,
    record_reroll_timestamp,
    reroll_cooldowns,
    tf_stats,
    revert_tasks,
    state_key,
)
from tfbot.legacy_embed import build_legacy_embed
from tfbot.history import publish_history_snapshot
from tfbot.panels import (
    VN_BACKGROUND_ROOT,
    VN_BACKGROUND_DEFAULT_RELATIVE,
    VN_AVATAR_MODE,
    VN_AVATAR_SCALE,
    apply_mention_placeholders,
    compose_game_avatar,
    fetch_avatar_bytes,
    get_selected_background_path,
    get_selected_outfit_name,
    get_selected_pose_outfit,
    list_available_outfits,
    list_background_choices,
    list_pose_outfits,
    parse_discord_formatting,
    prepare_custom_emoji_images,
    prepare_panel_mentions,
    prepare_reply_snippet,
    render_vn_panel,
    set_character_directory_overrides,
    set_selected_background,
    set_selected_outfit_name,
    set_selected_pose_outfit,
    strip_urls,
    vn_outfit_selection,
    persist_outfit_selections,
)
from tfbot.utils import (
    float_from_env,
    int_from_env,
    is_admin,
    member_profile_name,
    normalize_pose_name,
    path_from_env,
    utc_now,
)

if TYPE_CHECKING:
    from tfbot.gacha import GachaProfile


logging.basicConfig(
    level=os.getenv("TFBOT_LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tfbot")


BOT_MODE = os.getenv("TFBOT_MODE", "classic").lower()
TF_CHANNEL_ID = int_from_env("TFBOT_CHANNEL_ID", 0)
GACHA_CHANNEL_ID = int_from_env("TFBOT_GACHA_CHANNEL_ID", 0)
GACHA_ENABLED = GACHA_CHANNEL_ID > 0
CLASSIC_ENABLED = BOT_MODE != "gacha" and TF_CHANNEL_ID > 0

if BOT_MODE == "gacha" and not GACHA_ENABLED:
    raise RuntimeError("TFBOT_GACHA_CHANNEL_ID is required when running in gacha mode.")
if not CLASSIC_ENABLED and not GACHA_ENABLED:
    raise RuntimeError("Configure at least TFBOT_CHANNEL_ID or TFBOT_GACHA_CHANNEL_ID.")
TF_HISTORY_CHANNEL_ID = int_from_env("TFBOT_HISTORY_CHANNEL_ID", 1432196317722972262)
TF_ARCHIVE_CHANNEL_ID = int_from_env("TFBOT_ARCHIVE_CHANNEL_ID", 0)
TF_STATE_FILE = Path(os.getenv("TFBOT_STATE_FILE", "tf_state.json"))
TF_STATS_FILE = Path(os.getenv("TFBOT_STATS_FILE", "tf_stats.json"))
TF_REROLL_FILE = Path(os.getenv("TFBOT_REROLL_FILE", "tf_reroll.json"))
MESSAGE_STYLE = os.getenv(
    "TFBOT_MESSAGE_STYLE",
    "vn" if GACHA_ENABLED else "classic",
).lower()
TRANSFORM_DURATION_CHOICES: Sequence[Tuple[str, timedelta]] = [
    ("10 minutes", timedelta(minutes=10)),
    ("1 hour", timedelta(hours=1)),
    ("10 hours", timedelta(hours=10)),
]
INANIMATE_DURATION = timedelta(minutes=10)
REQUIRED_GUILD_PERMISSIONS = {
    "send_messages": "Send Messages (needed to respond in channels)",
    "embed_links": "Embed Links (history channel logging)",
}
MAGIC_EMOJI_NAME = os.getenv("TFBOT_MAGIC_EMOJI_NAME", "magic_emoji")
MAGIC_EMOJI_CACHE: Dict[int, str] = {}
SPECIAL_REROLL_FORMS = ("ball", "narrator")
ADMIN_ONLY_RANDOM_FORMS = ("syn", "circe")

configure_state(state_file=TF_STATE_FILE, stats_file=TF_STATS_FILE, reroll_file=TF_REROLL_FILE)
tf_stats.update(load_stats_from_disk())
reroll_cooldowns.update(load_reroll_cooldowns_from_disk())

INANIMATE_DATA_FILE = Path(os.getenv("TFBOT_INANIMATE_FILE", "tf_inanimate.json")).expanduser()
INANIMATE_TF_CHANCE = float(os.getenv("TFBOT_INANIMATE_CHANCE", "0"))


def _is_special_reroll_name(name: str) -> bool:
    normalized = (name or "").strip().lower()
    if not normalized:
        return False
    for token in SPECIAL_REROLL_FORMS:
        if normalized == token:
            return True
        if normalized.startswith(f"{token} "):
            return True
        if normalized.endswith(f" ({token})"):
            return True
    return False


def _has_special_reroll_access(state: Optional[TransformationState]) -> bool:
    if state is None:
        return False
    name = (state.character_name or "").strip()
    if not name:
        return False
    return _is_special_reroll_name(name)


def _token_variants(token: str) -> set[str]:
    normalized_token = (token or "").strip().lower()
    if not normalized_token:
        return set()
    variants = {normalized_token}

    def _add(value: str) -> None:
        cleaned = value.strip()
        if cleaned:
            variants.add(cleaned)

    if "_" in normalized_token:
        _add(normalized_token.replace("_", " "))
        _add(normalized_token.replace("_", ""))
        _add(normalized_token.split("_", 1)[0])
    if "-" in normalized_token:
        _add(normalized_token.replace("-", " "))
        _add(normalized_token.replace("-", ""))
        _add(normalized_token.split("-", 1)[0])
    return variants


def _name_matches_token(name: str, token: str) -> bool:
    name_normalized = (name or "").strip().lower()
    if not name_normalized:
        return False
    first_token = name_normalized.split(" ", 1)[0]
    variants = _token_variants(token)
    if not variants:
        return False
    for variant in variants:
        if variant == name_normalized or variant == first_token:
            return True
    return False


def _character_matches_token(character: TFCharacter, token: str) -> bool:
    variants = _token_variants(token)
    if not variants:
        return False
    name_normalized = character.name.lower()
    first_token = name_normalized.split(" ", 1)[0]
    for variant in variants:
        if variant == name_normalized or variant == first_token:
            return True
    folder_name_raw = (character.folder or "").strip()
    if folder_name_raw:
        folder_normalized = folder_name_raw.replace("\\", "/").strip("/").lower()
        folder_candidates = {folder_normalized}
        last_segment = folder_normalized.rsplit("/", 1)[-1]
        folder_candidates.add(last_segment)
        for folder_candidate in folder_candidates:
            if folder_candidate in variants:
                return True
    return False


def _is_admin_only_random_name(name: str) -> bool:
    normalized = (name or "").strip().lower()
    if not normalized:
        return False
    for token in ADMIN_ONLY_RANDOM_FORMS:
        if normalized == token:
            return True
        if normalized.startswith(f"{token} "):
            return True
        if normalized.endswith(f" ({token})"):
            return True
    return False


def _format_special_reroll_hint(character_name: str) -> Optional[str]:
    if not _is_special_reroll_name(character_name):
        return None
    return (
        "```diff\n"
        f"- {character_name} perk unlocked! Use `!reroll character` for a random swap or `!reroll character character` to force a form.\n"
        "```"
    )


def _default_inanimate_forms() -> Tuple[Dict[str, object], ...]:
    return (
        {
            "name": "Bewitched Pumpkin",
            "avatar_path": "avatars/inanimate/pumpkin.png",
            "message": "A carved grin flickers to life as candlelight dances from within.",
            "responses": [
                "*Your carved grin flickers with eerie candlelight.*",
                "*Seeds tumble out as you wobble helplessly on the table.*",
                "*The wind whistles through your hollow interior.*",
            ],
        },
        {
            "name": "Haunted Locker",
            "avatar_path": "avatars/inanimate/locker.png",
            "message": "Metal hinges groan, and a chill seeps through with every creak.",
            "responses": [
                "*The locker door creaks open with a metallic groan.*",
                "*A stack of dusty textbooks rattles inside.*",
                "*Someone scribbled 'boo' across your dented surface.*",
            ],
        },
        {
            "name": "Sentient Broom",
            "avatar_path": "avatars/inanimate/broom.png",
            "message": "Bristles rustle to life, eager to sweep the nearest floor.",
            "responses": [
                "*Bristles rustle as you sweep across the floor on your own.*",
                "*You lean dramatically against the wall, awaiting orders.*",
                "*A chill runs down your handleâ€”if you still had a spine.*",
            ],
        },
    )


def _load_inanimate_forms_from_gacha() -> Tuple[Dict[str, object], ...]:
    config_path = path_from_env("TFBOT_GACHA_CONFIG") or Path("gacha_config.json")
    if not config_path.exists():
        return ()
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse gacha config %s (%s); skipping inanimate import.", config_path, exc)
        return ()
    characters = payload.get("characters")
    if not isinstance(characters, dict):
        return ()
    forms: list[Dict[str, object]] = []
    for entry in characters.values():
        if not isinstance(entry, dict) or not entry.get("inanimate"):
            continue
        name = str(entry.get("display_name") or entry.get("name") or "").strip()
        message = str(entry.get("message") or "").strip()
        avatar_path = str(entry.get("avatar_path") or "").strip()
        if not name or not message:
            continue
        responses_field = entry.get("responses") or []
        if isinstance(responses_field, list):
            responses = [str(item).strip() for item in responses_field if str(item).strip()]
        else:
            responses = []
        if not responses:
            responses = [message]
        forms.append(
            {
                "name": name,
                "avatar_path": avatar_path,
                "message": message,
                "responses": responses,
            }
        )
    return tuple(forms)


def _load_inanimate_forms() -> Tuple[Dict[str, object], ...]:
    from_gacha = _load_inanimate_forms_from_gacha()
    if from_gacha:
        return from_gacha
    if not INANIMATE_DATA_FILE.exists():
        logger.info("Inanimate TF file %s not found; using defaults.", INANIMATE_DATA_FILE)
        return _default_inanimate_forms()
    try:
        payload = json.loads(INANIMATE_DATA_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse %s (%s); using defaults.", INANIMATE_DATA_FILE, exc)
        return _default_inanimate_forms()
    if not isinstance(payload, list):
        logger.warning("Inanimate TF file %s did not contain a list; using defaults.", INANIMATE_DATA_FILE)
        return _default_inanimate_forms()

    forms: list[Dict[str, object]] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        avatar_path = str(entry.get("avatar_path") or "").strip()
        message = str(entry.get("message") or "").strip()
        responses_field = entry.get("responses") or []
        if not name or not message:
            logger.debug("Skipping inanimate entry missing required fields: %s", entry)
            continue
        if isinstance(responses_field, list):
            responses = [str(item).strip() for item in responses_field if str(item).strip()]
        else:
            responses = []
        if not responses:
            responses = [message]
        forms.append(
            {
                "name": name,
                "avatar_path": avatar_path,
                "message": message,
                "responses": responses,
            }
        )
    if not forms:
        logger.warning("No valid inanimate forms loaded from %s; using defaults.", INANIMATE_DATA_FILE)
        return _default_inanimate_forms()
    return tuple(forms)


INANIMATE_FORMS = _load_inanimate_forms()

CHARACTER_DATA_FILE_SETTING = os.getenv("TFBOT_CHARACTERS_FILE", "").strip()
_CHARACTER_AVATAR_ROOT_SETTING = os.getenv("TFBOT_AVATAR_ROOT", "").strip()

_history_refresh_task: Optional[asyncio.Task] = None
_history_refresh_lock = asyncio.Lock()
_history_refresh_seq = 0


def schedule_history_refresh(delay: float = 0.2) -> None:
    """Debounce history snapshot updates."""
    if not CLASSIC_ENABLED:
        return
    global _history_refresh_task, _history_refresh_seq  # pylint: disable=global-statement

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.get_event_loop()

    _history_refresh_seq += 1
    sequence_id = _history_refresh_seq

    if _history_refresh_task and not _history_refresh_task.done():
        _history_refresh_task.cancel()

    async def runner(expected_seq: int) -> None:
        try:
            if delay:
                await asyncio.sleep(delay)
            async with _history_refresh_lock:
                if expected_seq != _history_refresh_seq:
                    return
                await publish_history_snapshot(
                    bot,
                    active_transformations,
                    tf_stats,
                    CHARACTER_POOL,
                    current_history_channel_id(),
                )
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Failed to refresh history snapshot: %s", exc)

    _history_refresh_task = loop.create_task(runner(sequence_id))


def _resolve_avatar_root() -> Optional[Path]:
    if not _CHARACTER_AVATAR_ROOT_SETTING:
        return None
    root = Path(_CHARACTER_AVATAR_ROOT_SETTING)
    if not root.is_absolute():
        root = (BASE_DIR / root).resolve()
    return root


def _load_character_dataset() -> Sequence[Dict[str, str]]:
    if CHARACTER_DATA_FILE_SETTING:
        dataset_path = Path(CHARACTER_DATA_FILE_SETTING).expanduser()
        if not dataset_path.is_absolute():
            dataset_path = (BASE_DIR / dataset_path).resolve()
        if dataset_path.exists():
            suffix = dataset_path.suffix.lower()
            if suffix == ".py":
                try:
                    spec = importlib.util.spec_from_file_location(
                        "_tf_character_override",
                        dataset_path,
                    )
                    module = importlib.util.module_from_spec(spec) if spec and spec.loader else None
                    if module and spec and spec.loader:
                        spec.loader.exec_module(module)  # type: ignore[attr-defined]
                        data = getattr(module, "TF_CHARACTERS", None)
                        if isinstance(data, list):
                            return data
                        logger.warning("TF_CHARACTERS missing or invalid in %s; using default.", dataset_path)
                    else:
                        logger.warning("Unable to load character module from %s; using default.", dataset_path)
                except Exception as exc:  # pylint: disable=broad-except
                    logger.warning("Failed to import character dataset %s: %s. Using default.", dataset_path, exc)
            else:
                try:
                    data = json.loads(dataset_path.read_text(encoding="utf-8"))
                    if isinstance(data, list):
                        return data
                    logger.warning("Character dataset %s is not a list; using default.", dataset_path)
                except json.JSONDecodeError as exc:
                    logger.warning("Failed to parse character dataset %s: %s. Using default.", dataset_path, exc)
        else:
            logger.warning("Character dataset %s not found; falling back to default.", dataset_path)
    return _DEFAULT_CHARACTER_DATA


CHARACTER_AVATAR_ROOT = _resolve_avatar_root()

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


def _build_character_pool(
    source: Sequence[Dict[str, str]],
    avatar_root: Optional[Path] = None,
) -> Sequence[TFCharacter]:
    pool: list[TFCharacter] = []
    for entry in source:
        try:
            avatar_path = str(entry.get("avatar_path", "")).strip()
            if (
                avatar_root
                and avatar_path
                and not avatar_path.startswith(("http://", "https://"))
            ):
                candidate = Path(avatar_path)
                if not candidate.is_absolute():
                    avatar_path = str((avatar_root / candidate).resolve())
                else:
                    avatar_path = str(candidate)
            folder_name = str(entry.get("folder", "")).strip() or None
            pool.append(
                TFCharacter(
                    name=entry["name"],
                    avatar_path=avatar_path,
                    message=str(entry.get("message", "")),
                    folder=folder_name,
                )
            )
        except KeyError as exc:
            logger.warning("Skipping character entry missing %s", exc)
    if not pool:
        raise RuntimeError("TF character dataset is empty. Populate tf_characters.py.")
    return pool


_CHARACTER_DATASET = _load_character_dataset()
_CHARACTER_FOLDER_OVERRIDES = {
    str(entry.get("name", "")).strip().lower(): str(entry.get("folder", "")).strip()
    for entry in _CHARACTER_DATASET
    if isinstance(entry, dict)
    and str(entry.get("name", "")).strip()
    and str(entry.get("folder", "")).strip()
}
CHARACTER_POOL = _build_character_pool(_CHARACTER_DATASET, CHARACTER_AVATAR_ROOT)
CHARACTER_BY_NAME: Dict[str, TFCharacter] = {
    character.name.strip().lower(): character for character in CHARACTER_POOL
}
set_character_directory_overrides(_CHARACTER_FOLDER_OVERRIDES)
GACHA_MANAGER = None

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
if not DISCORD_TOKEN:
    raise RuntimeError("Missing DISCORD_TOKEN. Set it in your environment or .env file.")

TF_CHANCE = float(os.getenv("TFBOT_CHANCE", "0.10"))
TF_CHANCE = max(0.0, min(1.0, TF_CHANCE))

intents = discord.Intents.default()
intents.message_content = True
intents.members = True

bot = commands.Bot(command_prefix=os.getenv("TFBOT_PREFIX", "!"), intents=intents)


class GuardedHelpCommand(commands.DefaultHelpCommand):
    """Custom help command that can ignore specific channels."""

    def __init__(self, blocked_channel_ids: set[int], **options):
        super().__init__(**options)
        self.blocked_channel_ids = blocked_channel_ids

    async def _should_block(self) -> bool:
        ctx = self.context
        if ctx is None or not self.blocked_channel_ids:
            return False
        channel_id = getattr(ctx.channel, "id", None)
        if channel_id in self.blocked_channel_ids:
            try:
                await ctx.message.delete()
            except discord.HTTPException:
                pass
            return True
        return False

    async def send_bot_help(self, mapping):
        if await self._should_block():
            return
        await super().send_bot_help(mapping)

    async def send_cog_help(self, cog):
        if await self._should_block():
            return
        await super().send_cog_help(cog)

    async def send_group_help(self, group):
        if await self._should_block():
            return
        await super().send_group_help(group)

    async def send_command_help(self, command):
        if await self._should_block():
            return
        await super().send_command_help(command)

    async def send_error_message(self, error):
        if await self._should_block():
            return
        await super().send_error_message(error)


blocked_help_channels: set[int] = set()
if GACHA_ENABLED:
    blocked_help_channels.add(GACHA_CHANNEL_ID)
if blocked_help_channels:
    bot.help_command = GuardedHelpCommand(blocked_help_channels, verify_checks=False)
else:
    bot.help_command = commands.DefaultHelpCommand()


async def ensure_state_restored() -> None:
    if tf_state.STATE_RESTORED:
        return
    states = load_states_from_disk()
    now = utc_now()
    for state in states:
        key = state_key(state.guild_id, state.user_id)
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
    tf_state.STATE_RESTORED = True


async def _schedule_revert(state: TransformationState, delay: float) -> None:
    try:
        await asyncio.sleep(delay)
        await revert_transformation(state, expired=True)
    except asyncio.CancelledError:
        logger.debug("Revert task for user %s cancelled", state.user_id)
    except Exception:
        logger.exception("Unexpected error while reverting TF for user %s", state.user_id)


async def revert_transformation(state: TransformationState, *, expired: bool) -> None:
    key = state_key(state.guild_id, state.user_id)
    current = active_transformations.get(key)
    if current is None or current.expires_at != state.expires_at:
        return

    guild, member = await fetch_member(state.guild_id, state.user_id)
    reason = "TF expired" if expired else "TF reverted"
    if member:
        if not state.original_display_name:
            state.original_display_name = member_profile_name(member)
    else:
        logger.warning("Could not locate member %s in guild %s to revert TF", state.user_id, state.guild_id)

    task = revert_tasks.pop(key, None)
    if task:
        task.cancel()
    active_transformations.pop(key, None)
    persist_states()

    schedule_history_refresh()


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


async def relay_transformed_message(
    message: discord.Message,
    state: TransformationState,
    *,
    reference: Optional[discord.MessageReference] = None,
    gacha_stars: Optional[int] = None,
    gacha_outfit: Optional[str] = None,
    gacha_pose: Optional[str] = None,
    gacha_rudy: Optional[int] = None,
    gacha_frog: Optional[int] = None,
    gacha_border: Optional[str] = None,
) -> bool:
    guild = message.guild
    if guild is None:
        return False

    cleaned_content = message.content.strip()
    original_content = cleaned_content
    generated_inanimate_response = False
    has_links = False
    character_name_normalized = (state.character_name or "").strip().lower()
    is_ball_override = state.is_inanimate and character_name_normalized == "ball"
    behaves_like_character = not state.is_inanimate or is_ball_override
    if state.is_inanimate and not is_ball_override:
        options = state.inanimate_responses or (
            "You emit a faint, spooky rattle.",
        )
        base_response = random.choice(options)
        spoiler_line = ""
        if original_content:
            sanitized_original, has_links = strip_urls(original_content)
            if sanitized_original:
                sanitized_original = discord.utils.escape_mentions(sanitized_original)
                sanitized_original = discord.utils.escape_markdown(sanitized_original)
            if sanitized_original:
                spoiler_line = f"\n||*{sanitized_original}*||"
        cleaned_content = f"{base_response}{spoiler_line}"
        generated_inanimate_response = True
    reply_context = await _resolve_reply_context(message)
    if (
        AI_REWRITE_ENABLED
        and cleaned_content
        and not cleaned_content.startswith(str(bot.command_prefix))
        and behaves_like_character
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
    if cleaned_content and behaves_like_character:
        cleaned_content, has_links = strip_urls(cleaned_content)
        cleaned_content = cleaned_content.strip()

    description = cleaned_content if cleaned_content else "*no message content*"
    formatted_segments = parse_discord_formatting(cleaned_content) if cleaned_content else None
    custom_emoji_images: Dict[str, "Image.Image"] = {}

    files: list[discord.File] = []
    payload: dict = {}

    if MESSAGE_STYLE == "vn" and not state.is_inanimate:
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
            gacha_star_count=gacha_stars,
            gacha_outfit_override=gacha_outfit,
            gacha_pose_override=gacha_pose,
            gacha_rudy=gacha_rudy,
            gacha_frog=gacha_frog,
            gacha_border=gacha_border,
        )
        if vn_file:
            files.append(vn_file)
        else:
            logger.debug("VN panel rendering unavailable; using classic embed.")

    if not files:
        embed, avatar_file = await build_legacy_embed(state, description)
        if avatar_file:
            files.append(avatar_file)
        payload["embed"] = embed


    has_attachments = bool(message.attachments)
    preserve_original = has_attachments or has_links
    deleted = False
    if not preserve_original:
        deleted = True
        try:
            await archive_original_message(message)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to archive message %s: %s", message.id, exc)
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

    if has_attachments and not has_links:
        placeholder = "\u200b"
        if message.content != placeholder:
            try:
                await message.edit(content=placeholder, attachments=message.attachments, suppress=True)
            except discord.HTTPException as exc:
                logger.debug("Unable to clear attachment message %s: %s", message.id, exc)

    return True


if GACHA_ENABLED:
    from tfbot.gacha import setup_gacha_mode

    GACHA_MANAGER = setup_gacha_mode(
        bot,
        character_pool=CHARACTER_POOL,
        relay_fn=relay_transformed_message,
    )
async def send_history_message(title: str, description: str) -> None:
    channel = bot.get_channel(current_history_channel_id())
    if channel is None:
        try:
            channel = await bot.fetch_channel(current_history_channel_id())
        except discord.HTTPException as exc:
            logger.warning("Cannot send history message, channel lookup failed: %s", exc)
            return
    embed = discord.Embed(
        title=title,
        description=description,
        color=0x9B59B6 if title == "TF Applied" else 0x546E7A,
        timestamp=utc_now(),
    )
    try:
        await channel.send(embed=embed, allowed_mentions=discord.AllowedMentions.none())
    except discord.HTTPException as exc:
        logger.warning("Failed to send history message: %s", exc)
    schedule_history_refresh()


async def archive_original_message(message: discord.Message) -> None:
    if TF_ARCHIVE_CHANNEL_ID <= 0:
        return

    archive_channel = None
    if message.guild is not None:
        archive_channel = message.guild.get_channel(TF_ARCHIVE_CHANNEL_ID)
    if archive_channel is None:
        archive_channel = bot.get_channel(TF_ARCHIVE_CHANNEL_ID)
    if archive_channel is None:
        try:
            archive_channel = await bot.fetch_channel(TF_ARCHIVE_CHANNEL_ID)
        except (discord.Forbidden, discord.HTTPException) as exc:
            logger.warning("Cannot access archive channel %s: %s", TF_ARCHIVE_CHANNEL_ID, exc)
            return

    if archive_channel is None or not hasattr(archive_channel, "send"):
        logger.debug("Archive channel %s unavailable or not messageable.", TF_ARCHIVE_CHANNEL_ID)
        return

    created_at = message.created_at
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)

    author_name = getattr(message.author, "display_name", str(message.author))
    author_id = getattr(message.author, "id", "unknown")
    channel_value = getattr(message.channel, "mention", f"#{getattr(message.channel, 'id', 'unknown')}")
    jump_link = getattr(message, "jump_url", None)

    embed = discord.Embed(
        title="Archived TF Message",
        description=message.content or "*no message content*",
        color=0x546E7A,
        timestamp=created_at,
    )
    embed.add_field(name="Author", value=f"{author_name} (`{author_id}`)", inline=False)
    embed.add_field(name="Channel", value=str(channel_value), inline=False)
    embed.add_field(name="Message ID", value=str(message.id), inline=False)
    if jump_link:
        embed.add_field(name="Jump Link", value=jump_link, inline=False)

    avatar_asset = getattr(message.author, "display_avatar", None)
    avatar_url = getattr(avatar_asset, "url", None)
    if avatar_url:
        embed.set_thumbnail(url=avatar_url)

    attachments = [attachment.url for attachment in message.attachments]
    if attachments:
        embed.add_field(name="Attachments", value="\n".join(attachments), inline=False)

    try:
        await archive_channel.send(embed=embed, allowed_mentions=discord.AllowedMentions.none())
    except discord.Forbidden as exc:
        logger.warning("Forbidden to send archive message for %s: %s", message.id, exc)
    except discord.HTTPException as exc:
        logger.warning("Failed to send archive message for %s: %s", message.id, exc)


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

    key = state_key(message.guild.id, member.id)
    if key in active_transformations:
        logger.debug("User %s already transformed; skipping.", member.id)
        return None

    used_characters = {state.character_name for state in active_transformations.values()}
    available_characters = [
        character for character in CHARACTER_POOL if character.name not in used_characters
    ]
    if not is_admin(member):
        available_characters = [
            character
            for character in available_characters
            if not _is_admin_only_random_name(character.name)
        ]
    character: Optional[TFCharacter] = None

    inanimate_form = None
    if INANIMATE_FORMS and random.random() <= INANIMATE_TF_CHANCE:
        inanimate_form = random.choice(INANIMATE_FORMS)

    if inanimate_form is None and not available_characters:
        logger.info("No available TF characters; skipping message %s", message.id)
        return None

    if inanimate_form is not None:
        selected_name = str(inanimate_form.get("name") or "").strip() or "Mystery Relic"
        character_avatar_path = str(inanimate_form.get("avatar_path") or "").strip()
        character_message = str(inanimate_form.get("message") or "").strip()
        responses_raw = inanimate_form.get("responses") or []
        if isinstance(responses_raw, (list, tuple)):
            inanimate_responses = tuple(
                str(item).strip() for item in responses_raw if str(item).strip()
            )
        else:
            inanimate_responses = tuple()
        if not character_message:
            character_message = "You feel unsettlingly still."
        if not inanimate_responses:
            inanimate_responses = (character_message,)
        duration_label = "10 minutes"
        duration_delta = INANIMATE_DURATION
    else:
        character = random.choice(available_characters)
        selected_name = character.name
        character_avatar_path = character.avatar_path
        character_message = character.message
        inanimate_responses = tuple()
        duration_label, duration_delta = random.choice(TRANSFORM_DURATION_CHOICES)
    now = utc_now()
    expires_at = now + duration_delta
    original_nick = member.nick
    profile_name = member_profile_name(member)

    state = TransformationState(
        user_id=member.id,
        guild_id=message.guild.id,
        character_name=selected_name,
        character_avatar_path=character_avatar_path,
        character_message=character_message,
        original_nick=original_nick,
        started_at=now,
        expires_at=expires_at,
        duration_label=duration_label,
        avatar_applied=False,
        original_display_name=profile_name,
        is_inanimate=inanimate_form is not None,
        inanimate_responses=inanimate_responses,
    )
    active_transformations[key] = state
    persist_states()

    delay = max((expires_at - now).total_seconds(), 0)
    revert_tasks[key] = asyncio.create_task(_schedule_revert(state, delay))

    logger.info(
        "TF applied to user %s (%s) for %s (expires at %s)",
        member.id,
        selected_name,
        duration_label,
        expires_at.isoformat(),
    )

    if character is not None:
        increment_tf_stats(message.guild.id, member.id, character.name)

    await send_history_message(
        "TF Applied",
        f"Original Name: **{member.name}**\nCharacter: **{selected_name}**\nDuration: {duration_label}.",
    )

    original_name = profile_name
    response_text = _format_character_message(
        character_message,
        original_name,
        member.mention,
        duration_label,
        selected_name,
    )
    special_hint = _format_special_reroll_hint(selected_name)
    if special_hint:
        response_text = f"{response_text}\n{special_hint}"
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
        if CLASSIC_ENABLED:
            channel_ids.add(TF_CHANNEL_ID)
        history_channel_id = current_history_channel_id()
        if history_channel_id:
            channel_ids.add(history_channel_id)
        if GACHA_MANAGER is not None:
            channel_ids.add(GACHA_MANAGER.channel_id)
        channel_ids.discard(0)
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

def current_history_channel_id() -> int:
    return TF_HISTORY_CHANNEL_ID


@bot.event
async def on_ready():
    await ensure_state_restored()
    logger.info("Logged in as %s (id=%s)", bot.user, bot.user.id if bot.user else "unknown")
    logger.info("TF chance set to %.0f%%", TF_CHANCE * 100)
    logger.info("Message style: %s", MESSAGE_STYLE.upper())
    if TF_CHANNEL_ID > 0:
        status = "enabled" if CLASSIC_ENABLED else "disabled"
        logger.info("Primary channel (%s): %s", status, TF_CHANNEL_ID)
    elif CLASSIC_ENABLED:
        logger.warning("No primary channel configured.")
    if GACHA_MANAGER is not None:
        logger.info("Gacha channel: %s", GACHA_MANAGER.channel_id)
    await log_guild_permissions()
    await log_channel_access()
    schedule_history_refresh()


@bot.command(name="synreset", hidden=True)
@commands.guild_only()
async def secret_reset_command(ctx: commands.Context):
    author = ctx.author
    if not isinstance(author, discord.Member):
        await ctx.reply("This command can only be used inside a server.", mention_author=False)
        return None
    if not is_admin(author):
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
    author_is_admin = is_admin(author)

    guild = ctx.guild
    if guild is None:
        await ctx.reply("This command can only be used inside a server.", mention_author=False)
        return None
    now = utc_now()
    author_state = find_active_transformation(author.id, guild.id)
    author_has_special_reroll = _has_special_reroll_access(author_state)
    can_force_reroll = author_is_admin or author_has_special_reroll
    target_member: Optional[discord.Member] = None
    target_is_admin = False
    state: Optional[TransformationState] = None

    forced_character: Optional[TFCharacter] = None
    forced_inanimate: Optional[Dict[str, object]] = None

    query = args.strip()
    forced_token: Optional[str] = None
    if query:
        parts = query.split()
        first_token_input = parts[0]
        first_token = first_token_input.lower()
        first_token_variants = _token_variants(first_token_input)
        if len(parts) > 1:
            if not can_force_reroll:
                await ctx.reply(
                    "Only admins or Syn's Ball/Narrator TFs can force a reroll into a specific form.",
                    mention_author=False,
                )
                return None
            forced_token = parts[1].lower()

        def _state_matches(state: TransformationState) -> bool:
            if state.guild_id != guild.id:
                return False
            if _name_matches_token(state.character_name, first_token_input):
                return True
            character_entry = CHARACTER_BY_NAME.get(state.character_name.strip().lower())
            if character_entry and _character_matches_token(character_entry, first_token_input):
                return True
            member_obj = guild.get_member(state.user_id)
            if member_obj:
                profile = member_profile_name(member_obj).lower()
                profile_first = profile.split(" ", 1)[0]
                if profile in first_token_variants or profile_first in first_token_variants:
                    return True
            return False

        matching_states = [s for s in active_transformations.values() if _state_matches(s)]
        target_member = None
        if matching_states:
            state = matching_states[0]
            _, target_member = await fetch_member(state.guild_id, state.user_id)
        else:
            # Fallback: caller's own transformation
            caller_state = author_state
            if caller_state:
                if _name_matches_token(caller_state.character_name, first_token_input):
                    state = caller_state
                    target_member = author
                else:
                    entry = CHARACTER_BY_NAME.get(caller_state.character_name.strip().lower())
                    if entry and _character_matches_token(entry, first_token_input):
                        state = caller_state
                        target_member = author
            if state is None:
                member_candidate = discord.utils.find(
                    lambda m: (
                        m.display_name.lower() in first_token_variants
                        or m.display_name.split(" ", 1)[0].lower() in first_token_variants
                        or m.name.lower() in first_token_variants
                        or m.name.split(" ", 1)[0].lower() in first_token_variants
                    ),
                    guild.members if guild else [],
                )
                if member_candidate:
                    state = find_active_transformation(member_candidate.id, guild.id)
                    target_member = member_candidate
                else:
                    state = None
            if state is None:
                await ctx.reply(
                    f"No active transformation found for `{first_token_input}`.",
                    mention_author=False,
                )
                return None
            if target_member is None:
                _, target_member = await fetch_member(state.guild_id, state.user_id)
        if target_member is None:
            await ctx.reply(
                f"Unable to locate the member transformed into {state.character_name}.",
                mention_author=False,
            )
            return None

        target_is_admin = isinstance(target_member, discord.Member) and is_admin(target_member)

        if (
            not author_is_admin
            and target_member.id == author.id
            and not author_has_special_reroll
        ):
            await ctx.reply(
                "You can't use your reroll on yourself. Ask another player to reroll you instead.",
                mention_author=False,
            )
            return None

        if forced_token:
            forced_character = next(
                (character for character in CHARACTER_POOL if _character_matches_token(character, forced_token)),
                None,
            )
            if forced_character is None:
                forced_inanimate = next(
                    (
                        entry
                        for entry in INANIMATE_FORMS
                        if str(entry.get("name", "")).split(" ", 1)[0].lower() == forced_token
                        or str(entry.get("name", "")).lower() == forced_token
                    ),
                    None,
                )
            if forced_character is None and forced_inanimate is None:
                await ctx.reply(
                    f"Unknown target `{forced_token}`. Provide a valid first name.",
                    mention_author=False,
                )
                return None
            forced_special_name = None
            if forced_character is not None and _is_special_reroll_name(forced_character.name):
                forced_special_name = forced_character.name
            elif forced_inanimate is not None and _is_special_reroll_name(
                str(forced_inanimate.get("name", ""))
            ):
                forced_special_name = str(forced_inanimate.get("name", ""))
            if (
                forced_character is not None
                and _is_admin_only_random_name(forced_character.name)
                and not author_is_admin
                and not target_is_admin
            ):
                await ctx.reply(
                    "You can only force Syn or Circe onto admins unless you're an admin yourself.",
                    mention_author=False,
                )
                return None
            if forced_special_name and author_has_special_reroll and not author_is_admin:
                await ctx.reply(
                    "Ball/Narrator TFs can't force others into Ball or Narrator. Ask an admin or owner.",
                    mention_author=False,
                )
                return None
    else:
        if not author_is_admin:
            await ctx.reply(
                "You need to specify someone to reroll, e.g. `!reroll charname`.",
                mention_author=False,
            )
            return None
        target_member = author
        key = state_key(guild.id, target_member.id)
        state = author_state
        if state is None:
            await ctx.reply(
                "You are not currently transformed; nothing to reroll.",
                mention_author=False,
            )
            return None
        target_is_admin = isinstance(target_member, discord.Member) and is_admin(target_member)

    key = state_key(guild.id, target_member.id)
    current_state = active_transformations.get(key)
    if current_state is None or current_state != state:
        await ctx.reply(
            "Unable to locate the transformation for this member.",
            mention_author=False,
        )
        return None

    used_names = {
        current_state.character_name
        for current_key, current_state in active_transformations.items()
        if current_key != key
    }

    if not author_is_admin and not author_has_special_reroll:
        last_reroll_at = get_last_reroll_timestamp(guild.id, author.id)
        if last_reroll_at is not None:
            cooldown_end = last_reroll_at + timedelta(hours=24)
            if cooldown_end > now:
                remaining = cooldown_end - now
                remaining_seconds = max(int(remaining.total_seconds()), 0)
                hours, remainder = divmod(remaining_seconds, 3600)
                minutes = remainder // 60
                if hours and minutes:
                    when_text = f"{hours} hour{'s' if hours != 1 else ''} and {minutes} minute{'s' if minutes != 1 else ''}"
                elif hours:
                    when_text = f"{hours} hour{'s' if hours != 1 else ''}"
                elif minutes:
                    when_text = f"{minutes} minute{'s' if minutes != 1 else ''}"
                else:
                    when_text = "less than a minute"
                await ctx.reply(
                    f"You've already used your reroll. You can reroll again in {when_text}.",
                    mention_author=False,
                )
                return None

    forced_mode = forced_character is not None or forced_inanimate is not None

    new_name: str
    new_avatar_path: str
    new_message: str
    new_is_inanimate: bool
    new_responses: Tuple[str, ...]

    if forced_inanimate is not None:
        new_name = str(forced_inanimate.get("name") or "Mystery Relic")
        new_avatar_path = str(forced_inanimate.get("avatar_path") or "")
        new_message = str(forced_inanimate.get("message") or "You feel unsettlingly still.")
        responses_raw = forced_inanimate.get("responses") or []
        if isinstance(responses_raw, (list, tuple)):
            new_responses = tuple(str(item).strip() for item in responses_raw if str(item).strip())
        else:
            new_responses = tuple()
        if not new_responses:
            new_responses = (new_message,)
        new_is_inanimate = True
    elif forced_character is not None:
        new_name = forced_character.name
        new_avatar_path = forced_character.avatar_path
        new_message = forced_character.message
        new_responses = tuple()
        new_is_inanimate = False
    else:
        available_characters = [
            character
            for character in CHARACTER_POOL
            if character.name not in used_names and character.name != state.character_name
        ]
        if not target_is_admin:
            available_characters = [
                character
                for character in available_characters
                if not _is_admin_only_random_name(character.name)
            ]
        if not available_characters:
            await ctx.reply(
                "No alternative characters are available to reroll right now.",
                mention_author=False,
            )
            return None
        chosen = random.choice(available_characters)
        new_name = chosen.name
        new_avatar_path = chosen.avatar_path
        new_message = chosen.message
        new_responses = tuple()
        new_is_inanimate = False

    if new_name == state.character_name:
        await ctx.reply(
            f"They are already transformed into {new_name}.",
            mention_author=False,
        )
        return None
    if new_name in used_names:
        await ctx.reply(
            f"{new_name} is already in use by another transformation.",
            mention_author=False,
        )
        return None

    previous_character = state.character_name
    state.character_name = new_name
    state.character_avatar_path = new_avatar_path
    state.character_message = new_message
    state.avatar_applied = False
    state.is_inanimate = new_is_inanimate
    state.inanimate_responses = new_responses

    if not author_is_admin:
        guaranteed_duration = timedelta(hours=10)
        state.started_at = now
        state.expires_at = now + guaranteed_duration
        state.duration_label = "10 hours"
        existing_task = revert_tasks.get(key)
        if existing_task:
            existing_task.cancel()
        revert_tasks[key] = asyncio.create_task(
            _schedule_revert(state, guaranteed_duration.total_seconds())
        )

    persist_states()

    if not new_is_inanimate:
        increment_tf_stats(guild.id, target_member.id, new_name)
    if not author_is_admin and not author_has_special_reroll:
        record_reroll_timestamp(guild.id, author.id, now)

    history_details = (
        f"Triggered by: **{author.display_name}**\n"
        f"Member: **{target_member.display_name}**\n"
        f"Previous Character: **{previous_character}**\n"
        f"New Character: **{new_name}**"
    )
    if forced_mode:
        history_details += "\nReason: Forced reroll override."
    await send_history_message(
        "TF Rerolled",
        history_details,
    )

    original_name = member_profile_name(target_member)
    if forced_mode:
        custom_template = (
            "barely has time to react before Syn swoops in with a grin and swaps them straight into {character}. Syn just had to spice things up."
        )
        response_text = _format_character_message(
            custom_template,
            original_name,
            target_member.mention,
            state.duration_label,
            new_name,
        )
    else:
        base_message = _format_character_message(
            new_message,
            original_name,
            target_member.mention,
            state.duration_label,
            new_name,
        )
        if author_is_admin:
            response_text = base_message
        else:
            response_text = (
                f"{author.display_name} cashes in their reroll on {target_member.mention}! {base_message}"
            )
    special_hint = _format_special_reroll_hint(new_name)
    if special_hint:
        response_text = f"{response_text}\n{special_hint}"
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

    summary_message = f"{target_member.display_name} has been rerolled into **{new_name}**."
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
        timestamp=utc_now(),
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
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        for line in lines:
            line_len = len(line) + 1
            if current and current_len + line_len > 1000:
                chunks.append("\n".join(current))
                current = []
                current_len = 0
            current.append(line)
            current_len += line_len
        if current:
            chunks.append("\n".join(current))
        for idx, chunk in enumerate(chunks):
            name = "By Character" if idx == 0 else "\u200b"
            embed.add_field(name=name, value=chunk or "\u200b", inline=False)

    key = state_key(guild_id, ctx.author.id)
    current_state = active_transformations.get(key)
    if current_state:
        remaining = max(
            (current_state.expires_at - utc_now()).total_seconds(),
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
                selected_pose_normalized = normalize_pose_name(selected_pose)
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
    schedule_history_refresh()



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

    await ensure_state_restored()

    guild_id = ctx.guild.id if ctx.guild else None
    state = find_active_transformation(ctx.author.id, guild_id)
    if not state:
        fallback_state = find_active_transformation(ctx.author.id)
        if fallback_state and ctx.guild and fallback_state.guild_id != guild_id:
            target_guild = bot.get_guild(fallback_state.guild_id)
            guild_name = target_guild.name if target_guild else f"server {fallback_state.guild_id}"
            message = (
                "You're transformed right now, but in a different server. "
                f"Use this command in **{guild_name}** to change that outfit."
            )
        else:
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
        normalized_pose = normalize_pose_name(parsed_pose)
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
    schedule_history_refresh()
    if ctx.guild:
        await ctx.reply(confirmation, mention_author=False)
    else:
        await ctx.send(confirmation)

@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return None

    logger.info(
        "Message %s from %s in channel %s",
        message.id,
        message.author.id,
        getattr(message.channel, "id", "dm"),
    )

    command_invoked = False
    ctx = await bot.get_context(message)
    if ctx.command:
        command_invoked = True
        await bot.invoke(ctx)

    if command_invoked:
        return None

    is_gacha_channel = (
        GACHA_MANAGER is not None
        and isinstance(message.channel, discord.TextChannel)
        and message.guild is not None
        and message.channel.id == GACHA_MANAGER.channel_id
    )
    if is_gacha_channel:
        await GACHA_MANAGER.handle_message(message, command_invoked=command_invoked)
        return None

    if not CLASSIC_ENABLED:
        return None

    is_admin_user = is_admin(message.author)
    if message.guild and message.guild.owner_id == message.author.id:
        logger.debug("Ignoring message %s from server owner %s", message.id, message.author.id)
        return None
    channel_id = getattr(message.channel, "id", None)
    if message.guild and TF_CHANNEL_ID > 0 and channel_id != TF_CHANNEL_ID:
        logger.info(
            "Skipping message %s: channel %s is not the configured TF channel (%s).",
            message.id,
            channel_id,
            TF_CHANNEL_ID,
        )
        return None

    profile: Optional["GachaProfile"] = None
    gacha_equipped = False
    gacha_handled = False
    if GACHA_MANAGER is not None and message.guild:
        profile = await GACHA_MANAGER.ensure_profile(message.guild.id, message.author.id)
        allowed = await GACHA_MANAGER.enforce_spam_policy(message, profile=profile)
        if not allowed:
            return None
        await GACHA_MANAGER.award_message_reward(message, profile=profile)
        gacha_equipped = bool(profile.equipped_character)
        if gacha_equipped:
            gacha_handled = await GACHA_MANAGER.relay_classic_message(
                message,
                profile=profile,
            )
            if gacha_handled:
                return None

    if message.guild and not gacha_equipped:
        key = state_key(message.guild.id, message.author.id)
        state = active_transformations.get(key)
        if state:
            reply_reference: Optional[discord.MessageReference] = (
                message.to_reference(fail_if_not_exists=False) if message.reference else None
            )
            await relay_transformed_message(message, state, reference=reply_reference)
            return None

    if message.guild and GACHA_MANAGER is not None and gacha_equipped:
        logger.debug(
            "Skipping TF roll for user %s in guild %s: gacha character equipped.",
            message.author.id,
            message.guild.id,
        )
        return None

    logger.info(
        "Message intercepted (admin=%s): user %s in channel %s",
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
    bot.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()










