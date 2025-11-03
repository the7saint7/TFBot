"""History channel snapshot management."""

from __future__ import annotations

import asyncio
import io
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import discord

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - Pillow is an optional runtime dependency
    Image = ImageDraw = ImageFont = None  # type: ignore[misc,assignment]

from .models import TFCharacter, TransformationState
from .state import load_states_from_disk
from .panels import (
    BASE_DIR,
    VN_BACKGROUND_DEFAULT_RELATIVE,
    VN_BACKGROUND_ROOT,
    VN_FONT_BOLD_PATH,
    VN_FONT_REGULAR_PATH,
    compose_game_avatar,
    get_selected_background_path,
    get_selected_pose_outfit,
    list_background_choices,
)

logger = logging.getLogger("tfbot.history")

# Default testing channel; override with TFBOT_HISTORY_SNAPSHOT_CHANNEL
DEFAULT_HISTORY_CHANNEL_ID = 1433105932392595609

HISTORY_CANVAS_WIDTH = 760
HISTORY_CARD_PADDING = 24
HISTORY_CARD_SPACING = 18
HISTORY_AVATAR_MAX_WIDTH = 220
HISTORY_AVATAR_MAX_HEIGHT = 260
HISTORY_BACKGROUND_COLOR = (18, 20, 24, 255)
HISTORY_CARD_COLOR = (32, 35, 41, 255)
HISTORY_TEXT_COLOR = (230, 232, 236, 255)
HISTORY_SUBTEXT_COLOR = (180, 184, 190, 255)
HISTORY_SEPARATOR_COLOR = (54, 57, 63, 255)
HISTORY_TITLE_FONT_SIZE = 32
HISTORY_BODY_FONT_SIZE = 24
HISTORY_ACTIVE_EMBED_COLOR = (152, 118, 255)

_FONT_REGULAR_CACHE: Optional["ImageFont.FreeTypeFont"] = None
_FONT_BOLD_CACHE: Optional["ImageFont.FreeTypeFont"] = None


@dataclass(frozen=True)
class HistoryCardData:
    character_name: str
    title: str
    lines: Sequence[str]
    accent_color: Optional[Tuple[int, int, int]] = None
    user_id: Optional[int] = None
    avatar_path: Optional[str] = None


def _ensure_history_fonts() -> Tuple[Optional["ImageFont.FreeTypeFont"], Optional["ImageFont.FreeTypeFont"]]:
    global _FONT_REGULAR_CACHE, _FONT_BOLD_CACHE  # pylint: disable=global-statement
    if ImageFont is None:
        return None, None

    if _FONT_REGULAR_CACHE is None:
        if VN_FONT_REGULAR_PATH:
            try:
                _FONT_REGULAR_CACHE = ImageFont.truetype(
                    str(VN_FONT_REGULAR_PATH),
                    HISTORY_BODY_FONT_SIZE,
                )
            except OSError:
                _FONT_REGULAR_CACHE = None
        if _FONT_REGULAR_CACHE is None:
            _FONT_REGULAR_CACHE = ImageFont.load_default()

    if _FONT_BOLD_CACHE is None:
        bold_source = VN_FONT_BOLD_PATH or VN_FONT_REGULAR_PATH
        if bold_source:
            try:
                _FONT_BOLD_CACHE = ImageFont.truetype(
                    str(bold_source),
                    HISTORY_TITLE_FONT_SIZE,
                )
            except OSError:
                _FONT_BOLD_CACHE = None
        if _FONT_BOLD_CACHE is None:
            _FONT_BOLD_CACHE = ImageFont.load_default()

    return _FONT_REGULAR_CACHE, _FONT_BOLD_CACHE


def _parse_hex_color(value: Optional[str]) -> Optional[Tuple[int, int, int]]:
    if not value:
        return None
    raw = value.strip()
    if not raw:
        return None
    if raw.startswith("#"):
        raw = raw[1:]
    if len(raw) == 3:
        raw = "".join(ch * 2 for ch in raw)
    if len(raw) != 6:
        return None
    try:
        r = int(raw[0:2], 16)
        g = int(raw[2:4], 16)
        b = int(raw[4:6], 16)
        return (r, g, b)
    except ValueError:
        return None


def _first_given_name(full_name: str) -> str:
    if not full_name:
        return ""
    token = full_name.strip().split()[0]
    if "-" in token:
        token = token.split("-")[0]
    return token


def _human_remaining(expires_at: datetime) -> str:
    now = discord.utils.utcnow()
    total_seconds = int((expires_at - now).total_seconds())
    if total_seconds <= 0:
        return "Expired"
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    parts: List[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if not parts and seconds:
        parts.append(f"{seconds}s")
    return " ".join(parts) or "Expired"


def _font_height(font: "ImageFont.FreeTypeFont") -> int:
    try:
        ascent, descent = font.getmetrics()
        return ascent + descent
    except (AttributeError, TypeError):
        return getattr(font, "size", HISTORY_BODY_FONT_SIZE)


def _wrap_text(text: str, font: "ImageFont.FreeTypeFont", max_width: int) -> List[str]:
    if max_width <= 0 or not text:
        return [text]
    dummy = Image.new("RGB", (1, 1)) if Image else None
    draw = ImageDraw.Draw(dummy) if dummy else None
    words = text.split()
    if not words:
        return [text]
    lines: List[str] = []
    current = words[0]
    font_size = getattr(font, "size", HISTORY_BODY_FONT_SIZE)
    for word in words[1:]:
        tentative = f"{current} {word}"
        width = draw.textlength(tentative, font=font) if draw else len(tentative) * font_size
        if width <= max_width:
            current = tentative
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _render_history_card(
    entry: HistoryCardData,
    regular_font: "ImageFont.FreeTypeFont",
    bold_font: "ImageFont.FreeTypeFont",
) -> Optional["Image.Image"]:
    if Image is None:
        return None

    avatar_image = compose_game_avatar(entry.character_name)
    if avatar_image is None and entry.avatar_path:
        candidate = Path(entry.avatar_path)
        if not candidate.is_absolute():
            candidate = (BASE_DIR / candidate).resolve()
        if candidate.exists():
            try:
                avatar_image = Image.open(candidate).convert("RGBA")
            except OSError as exc:
                logger.debug("History card: failed to load fallback avatar %s: %s", candidate, exc)
    if avatar_image is not None:
        avatar = avatar_image.copy().convert("RGBA")
        avatar.thumbnail((HISTORY_AVATAR_MAX_WIDTH, HISTORY_AVATAR_MAX_HEIGHT), Image.LANCZOS)
    else:
        avatar = Image.new(
            "RGBA",
            (HISTORY_AVATAR_MAX_WIDTH, HISTORY_AVATAR_MAX_HEIGHT),
            HISTORY_SEPARATOR_COLOR,
        )

    text_width = HISTORY_CANVAS_WIDTH - (avatar.width + (HISTORY_CARD_PADDING * 4))
    title_lines = _wrap_text(entry.title, bold_font, text_width)
    body_lines: List[str] = []
    for line in entry.lines:
        body_lines.extend(_wrap_text(line, regular_font, text_width))

    title_height = sum(_font_height(bold_font) for _ in title_lines)
    if title_lines:
        title_height += max(0, (len(title_lines) - 1) * 4)
    body_height = sum(_font_height(regular_font) for _ in body_lines)
    if body_lines:
        body_height += max(0, (len(body_lines) - 1) * 3)
    text_height = title_height + (10 if body_lines else 0) + body_height

    card_width = HISTORY_CANVAS_WIDTH - (HISTORY_CARD_PADDING * 2)
    card_height = max(avatar.height + (HISTORY_CARD_PADDING * 2), text_height + (HISTORY_CARD_PADDING * 2))

    card = Image.new("RGBA", (card_width, card_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(card)
    radius = 28
    draw.rounded_rectangle(
        (0, 0, card_width, card_height),
        radius=radius,
        fill=HISTORY_CARD_COLOR,
    )

    avatar_x = HISTORY_CARD_PADDING
    avatar_y = (card_height - avatar.height) // 2
    card.paste(avatar, (avatar_x, avatar_y), avatar)

    text_x = avatar_x + avatar.width + HISTORY_CARD_PADDING
    text_y = HISTORY_CARD_PADDING
    accent = entry.accent_color or (HISTORY_TEXT_COLOR[0], HISTORY_TEXT_COLOR[1], HISTORY_TEXT_COLOR[2])
    title_color = (accent[0], accent[1], accent[2], 255)

    for idx, line in enumerate(title_lines):
        draw.text((text_x, text_y), line, fill=title_color, font=bold_font)
        text_y += _font_height(bold_font)
        if idx < len(title_lines) - 1:
            text_y += 4

    if body_lines:
        text_y += 10
    for idx, line in enumerate(body_lines):
        draw.text((text_x, text_y), line, fill=HISTORY_SUBTEXT_COLOR, font=regular_font)
        text_y += _font_height(regular_font)
        if idx < len(body_lines) - 1:
            text_y += 3

    return card


def _render_card_file(
    entry: HistoryCardData,
    filename_prefix: str,
    index: int,
    regular_font: Optional["ImageFont.FreeTypeFont"],
    bold_font: Optional["ImageFont.FreeTypeFont"],
) -> Optional[discord.File]:
    if Image is None or regular_font is None or bold_font is None:
        return None

    card = _render_history_card(entry, regular_font, bold_font)
    if card is None:
        return None

    buffer = io.BytesIO()
    card.save(buffer, format="PNG")
    buffer.seek(0)
    filename = f"{filename_prefix}-{index}.png"
    return discord.File(fp=buffer, filename=filename)


def _embed_color(
    accent: Optional[Tuple[int, int, int]],
    fallback: Tuple[int, int, int],
) -> discord.Color:
    base = accent or fallback
    r, g, b = base
    return discord.Color.from_rgb(r, g, b)


def _chunk_sections(sections: Sequence[str], max_length: int = 3600, max_chunks: int = 10) -> List[str]:
    if not sections:
        return []
    chunks: List[str] = []
    current: List[str] = []
    length = 0
    for section in sections:
        seg_len = len(section)
        if seg_len > max_length:
            lines = section.splitlines()
            buf: List[str] = []
            buf_len = 0
            for line in lines:
                line_len = len(line) + 1
                if buf and buf_len + line_len > max_length:
                    chunks.append("\n".join(buf))
                    if len(chunks) >= max_chunks:
                        return chunks
                    buf = []
                    buf_len = 0
                buf.append(line)
                buf_len += line_len
            if buf:
                chunks.append("\n".join(buf))
                if len(chunks) >= max_chunks:
                    return chunks
            continue
        if current and length + seg_len + 2 > max_length:
            chunks.append("\n\n".join(current))
            if len(chunks) >= max_chunks:
                return chunks
            current = []
            length = 0
        current.append(section)
        length += seg_len + 2
    if current and len(chunks) < max_chunks:
        chunks.append("\n\n".join(current))
    return chunks


async def _fetch_history_channel(bot: discord.Client, channel_id: Optional[int]) -> discord.TextChannel:
    if channel_id is None:
        raw = os.getenv("TFBOT_HISTORY_SNAPSHOT_CHANNEL")
        if raw and raw.isdigit():
            channel_id = int(raw)
        else:
            channel_id = DEFAULT_HISTORY_CHANNEL_ID
    channel = bot.get_channel(channel_id)
    if channel is None:
        try:
            channel = await bot.fetch_channel(channel_id)
        except discord.HTTPException as exc:
            raise RuntimeError(f"Unable to fetch history channel {channel_id}: {exc}") from exc
    if not isinstance(channel, discord.TextChannel):
        raise RuntimeError(f"History channel {channel_id} is not a text channel.")
    return channel


async def _clear_history_channel(channel: discord.TextChannel) -> None:
    async def purge_once() -> bool:
        removed = False
        try:
            messages = [
                message async for message in channel.history(limit=100)
                if (discord.utils.utcnow() - message.created_at).days < 14
            ]
        except discord.HTTPException as exc:
            logger.debug("Unable to iterate history channel %s: %s", channel.id, exc)
            return False

        if not messages:
            return False

        bot_messages = [m for m in messages if m.author == channel.guild.me]
        other_messages = [m for m in messages if m.author != channel.guild.me]

        if bot_messages:
            try:
                await channel.delete_messages(bot_messages)
            except discord.HTTPException:
                for msg in bot_messages:
                    try:
                        await msg.delete()
                    except discord.HTTPException:
                        continue
            removed = True

        for msg in other_messages:
            try:
                await msg.delete()
                removed = True
            except discord.HTTPException:
                continue
        return removed

    for _ in range(5):
        changed = await purge_once()
        if not changed:
            break
        await asyncio.sleep(0.5)


def _aggregate_usage(tf_stats: Mapping[str, Mapping[str, Mapping[str, object]]]) -> Dict[str, int]:
    usage: Dict[str, int] = {}
    for guild_data in tf_stats.values():
        for entry in guild_data.values():
            characters = entry.get("characters")
            if not isinstance(characters, Mapping):
                continue
            for name, count in characters.items():
                try:
                    usage[name] = usage.get(name, 0) + int(count)
                except (TypeError, ValueError):
                    continue
    return usage


def _relative_background_label(path: Path) -> str:
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    if VN_BACKGROUND_ROOT:
        try:
            return resolved.relative_to(VN_BACKGROUND_ROOT.resolve()).as_posix()
        except ValueError:
            pass
    return resolved.as_posix()


def _background_index_map() -> Dict[str, int]:
    index_map: Dict[str, int] = {}
    for idx, path in enumerate(list_background_choices(), start=1):
        label = _relative_background_label(path).lower()
        index_map[label] = idx
    return index_map


def _format_background_label(
    user_id: Optional[int],
    index_lookup: Mapping[str, int],
) -> Optional[str]:
    target: Optional[Path]
    if user_id is not None:
        target = get_selected_background_path(user_id)
    else:
        target = None

    if target is None:
        if VN_BACKGROUND_DEFAULT_RELATIVE is None:
            return None
        if VN_BACKGROUND_ROOT:
            target = VN_BACKGROUND_ROOT / VN_BACKGROUND_DEFAULT_RELATIVE
        else:
            target = VN_BACKGROUND_DEFAULT_RELATIVE

    label = _relative_background_label(target)
    index = index_lookup.get(label.lower())
    if index:
        return f"#{index} {label}"
    return label


def _build_active_entries(
    active_states: Sequence[TransformationState],
    usage_counts: Mapping[str, int],
    outfit_lookup: Mapping[str, Optional[str]],
    background_lookup: Mapping[int, str],
    member_lookup: Mapping[int, str],
    color_lookup: Mapping[str, Tuple[int, int, int]],
) -> Tuple[List[str], List[HistoryCardData]]:
    if not active_states:
        return ["*No active transformations.*"], []

    sections: List[str] = []
    cards: List[HistoryCardData] = []
    for state in active_states:
        usage = usage_counts.get(state.character_name, 0)
        outfit = outfit_lookup.get(state.character_name)
        background = background_lookup.get(state.user_id)
        countdown = int(state.expires_at.timestamp())

        member_name = member_lookup.get(state.user_id) or state.original_display_name or "Unknown"
        first_name = _first_given_name(state.character_name)
        first_name_display = first_name or state.character_name or "this character"

        section_lines = [
            f"**{state.character_name}**",
            f"- User: ||**{member_name}** (<@{state.user_id}>)||",
            f"- Time remaining: <t:{countdown}:R>",
            f"- Times someone TFed into {first_name_display}: `{usage}`",
        ]
        card_lines = [
            f"Time remaining: {_human_remaining(state.expires_at)}",
            f"Times someone TFed into {first_name_display}: {usage}",
        ]
        if outfit:
            section_lines.append(f"- Outfit: `{outfit}`")
            card_lines.append(f"Outfit: {outfit}")
        if background:
            section_lines.append(f"- Background: `{background}`")
            card_lines.append(f"Background: {background}")

        sections.append("\n".join(section_lines))
        cards.append(
            HistoryCardData(
                character_name=state.character_name,
                title=state.character_name,
                lines=card_lines,
                accent_color=color_lookup.get(state.character_name.lower()),
                user_id=state.user_id,
                avatar_path=state.character_avatar_path,
            )
        )
    return sections, cards


async def publish_history_snapshot(
    bot: discord.Client,
    active_states: Mapping[Tuple[int, int], TransformationState],
    tf_stats: Mapping[str, Mapping[str, Mapping[str, object]]],
    character_pool: Sequence[TFCharacter],
    channel_id: Optional[int] = None,
) -> None:
    try:
        channel = await _fetch_history_channel(bot, channel_id)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Unable to prepare history channel: %s", exc)
        return

    await _clear_history_channel(channel)

    usage_counts = _aggregate_usage(tf_stats)
    guild_id = channel.guild.id

    combined: Dict[Tuple[int, int], TransformationState] = {}
    for persisted in load_states_from_disk():
        if persisted.guild_id == guild_id:
            combined[(persisted.guild_id, persisted.user_id)] = persisted
    for state in active_states.values():
        if state.guild_id == guild_id:
            combined[(state.guild_id, state.user_id)] = state

    active_list = list(combined.values())

    outfits = {
        character.name: get_selected_pose_outfit(character.name)[1]
        for character in character_pool
    }

    background_index_lookup = _background_index_map()
    backgrounds = {
        state.user_id: _format_background_label(state.user_id, background_index_lookup)
        for state in active_list
    }
    color_lookup = {
        character.name.lower(): color
        for character in character_pool
        if (color := _parse_hex_color(character.speech_color)) is not None
    }

    member_lookup: Dict[int, str] = {}
    missing_members: List[int] = []
    for state in active_list:
        member = channel.guild.get_member(state.user_id)
        if member is not None:
            member_lookup[state.user_id] = member.name
        else:
            missing_members.append(state.user_id)
    for user_id in missing_members:
        try:
            member = await channel.guild.fetch_member(user_id)
        except discord.HTTPException:
            continue
        if member:
            member_lookup[user_id] = member.name

    active_sorted = sorted(active_list, key=lambda s: s.expires_at)

    active_sections, active_cards = _build_active_entries(
        active_sorted,
        usage_counts,
        outfits,
        backgrounds,
        member_lookup,
        color_lookup,
    )
    regular_font, bold_font = _ensure_history_fonts()
    can_render_cards = (
        Image is not None
        and regular_font is not None
        and bold_font is not None
    )

    active_allowed_mentions = discord.AllowedMentions(users=True, roles=True, everyone=False)

    active_embeds: List[discord.Embed] = []
    active_files: List[discord.File] = []
    if active_cards and can_render_cards and len(active_cards) <= 10:
        success = True
        for index, card in enumerate(active_cards, start=1):
            file = _render_card_file(
                card,
                "history-active",
                index,
                regular_font,
                bold_font,
            )
            if file is None:
                success = False
                break
            embed = discord.Embed(
                color=_embed_color(card.accent_color, HISTORY_ACTIVE_EMBED_COLOR),
            )
            embed.set_image(url=f"attachment://{file.filename}")
            if card.user_id is not None:
                embed.description = f"||<@{card.user_id}>||"
            active_embeds.append(embed)
            active_files.append(file)
        if not success:
            active_embeds.clear()
            active_files.clear()

    if not active_embeds:
        fallback_chunks = _chunk_sections(active_sections) or ["*No active transformations.*"]
        for chunk in fallback_chunks:
            active_embeds.append(
                discord.Embed(
                    description=chunk,
                    color=discord.Color.from_rgb(*HISTORY_ACTIVE_EMBED_COLOR),
                )
            )

    try:
        active_kwargs: Dict[str, object] = {
            "embeds": active_embeds,
            "allowed_mentions": active_allowed_mentions,
        }
        if active_files:
            active_kwargs["files"] = active_files
        await channel.send(**active_kwargs)
    except discord.HTTPException as exc:
        logger.warning("Failed to update history channel: %s", exc)


__all__ = ["publish_history_snapshot"]
