import argparse
import asyncio
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
from typing import Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple

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
    increment_tf_stats,
    load_states_from_disk,
    load_stats_from_disk,
    persist_states,
    persist_stats,
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
    parse_channel_ids,
    path_from_env,
    utc_now,
)


logging.basicConfig(
    level=os.getenv("TFBOT_LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tfbot")


DEV_CHANNEL_ID = 1432191400983662766
DEV_TF_CHANCE = 0.75
TF_HISTORY_CHANNEL_ID = int_from_env("TFBOT_HISTORY_CHANNEL_ID", 1432196317722972262)
TF_HISTORY_DEV_CHANNEL_ID = int_from_env("TFBOT_HISTORY_DEV_CHANNEL_ID", 1433105932392595609)
TF_STATE_FILE = Path(os.getenv("TFBOT_STATE_FILE", "tf_state.json"))
TF_STATS_FILE = Path(os.getenv("TFBOT_STATS_FILE", "tf_stats.json"))
MESSAGE_STYLE = os.getenv("TFBOT_MESSAGE_STYLE", "classic").lower()
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

configure_state(state_file=TF_STATE_FILE, stats_file=TF_STATS_FILE)
tf_stats.update(load_stats_from_disk())

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

IGNORED_CHANNEL_IDS = parse_channel_ids(os.getenv("TFBOT_IGNORE_CHANNELS", ""))
ALLOWED_CHANNEL_IDS: Optional[Set[int]] = None

intents = discord.Intents.default()
intents.message_content = True
intents.members = True

bot = commands.Bot(command_prefix=os.getenv("TFBOT_PREFIX", "!"), intents=intents)

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
    try:
        await publish_history_snapshot(
            bot,
            active_transformations,
            tf_stats,
            CHARACTER_POOL,
            current_history_channel_id(),
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Failed to refresh history snapshot after TF revert: %s", exc)


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
    has_links = False
    if cleaned_content:
        cleaned_content, has_links = strip_urls(cleaned_content)
        cleaned_content = cleaned_content.strip()

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
    try:
        await publish_history_snapshot(
            bot,
            active_transformations,
            tf_stats,
            CHARACTER_POOL,
            current_history_channel_id(),
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Failed to refresh history snapshot: %s", exc)


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
    available_characters = [character for character in CHARACTER_POOL if character.name not in used_characters]
    if not available_characters:
        logger.info("No available TF characters; skipping message %s", message.id)
        return None

    character = random.choice(available_characters)
    if DEV_MODE:
        duration_label, duration_delta = DEV_TRANSFORM_DURATION
    else:
        duration_label, duration_delta = random.choice(TRANSFORM_DURATION_CHOICES)
    now = utc_now()
    expires_at = now + duration_delta
    original_nick = member.nick
    profile_name = member_profile_name(member)

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


def current_history_channel_id() -> int:
    return TF_HISTORY_DEV_CHANNEL_ID if DEV_MODE else TF_HISTORY_CHANNEL_ID


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
    try:
        await publish_history_snapshot(
            bot,
            active_transformations,
            tf_stats,
            CHARACTER_POOL,
            current_history_channel_id(),
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Failed to refresh history snapshot on startup: %s", exc)


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
    if not is_admin(author):
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
        key = state_key(guild.id, target_member.id)
        state = active_transformations.get(key)
        if state is None:
            await ctx.reply(
                "You are not currently transformed; nothing to reroll.",
                mention_author=False,
            )
            return None

    key = state_key(guild.id, target_member.id)
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
        embed.add_field(name="By Character", value="\n".join(lines), inline=False)

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
    try:
        await publish_history_snapshot(
            bot,
            active_transformations,
            tf_stats,
            CHARACTER_POOL,
            current_history_channel_id(),
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Failed to refresh history snapshot after background change: %s", exc)



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
    try:
        await publish_history_snapshot(
            bot,
            active_transformations,
            tf_stats,
            CHARACTER_POOL,
            current_history_channel_id(),
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Failed to refresh history snapshot after outfit change: %s", exc)
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

    is_admin_user = is_admin(message.author)
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
        key = state_key(message.guild.id, message.author.id)
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











