import argparse
import asyncio
import io
import json
import logging
import os
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Set, Tuple

import aiohttp
import discord
from discord.ext import commands
from dotenv import load_dotenv

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


DEV_CHANNEL_ID = 1432191400983662766
DEV_TF_CHANCE = 0.75
TF_HISTORY_CHANNEL_ID = _int_from_env("TFBOT_HISTORY_CHANNEL_ID", 1432196317722972262)
TF_STATE_FILE = Path(os.getenv("TFBOT_STATE_FILE", "tf_state.json"))
TF_STATS_FILE = Path(os.getenv("TFBOT_STATS_FILE", "tf_stats.json"))
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

    embed = discord.Embed(
        description=description,
        color=0x9B59B6,
        timestamp=_now(),
    )
    embed.set_author(name=state.character_name)

    files: list[discord.File] = []
    avatar_bytes = await fetch_avatar_bytes(state.character_avatar_path)
    avatar_filename = None
    if avatar_bytes:
        suffix = Path(state.character_avatar_path).suffix or ".png"
        avatar_filename = f"tf-avatar-{state.user_id}{suffix}"
        files.append(
            discord.File(io.BytesIO(avatar_bytes), filename=avatar_filename)
        )
        embed.set_thumbnail(url=f"attachment://{avatar_filename}")

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

    if not deleted:
        embed.set_footer(text="Grant Manage Messages so TF relay can replace posts.")

    send_kwargs = {"embed": embed}
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
@commands.guild_only()
async def tf_stats_command(ctx: commands.Context):
    guild_data = tf_stats.get(str(ctx.guild.id), {})
    user_data = guild_data.get(str(ctx.author.id))

    if not user_data:
        await ctx.reply(
            "You haven't experienced any transformations yet.", mention_author=False
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

    await ctx.reply(embed=embed, mention_author=False)


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
    if not DEV_MODE and is_admin_user:
        logger.debug("Ignoring message %s from admin user %s (normal mode)", message.id, message.author.id)
        return None
    if DEV_MODE and is_admin_user:
        logger.debug("Admin user %s message allowed (dev mode)", message.author.id)

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
