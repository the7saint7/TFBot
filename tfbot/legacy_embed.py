"""Legacy embed-based avatar rendering helpers."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional, Tuple

import discord

from tfbot.models import TransformationState
from tfbot.panels import fetch_avatar_bytes
from tfbot.utils import utc_now


async def build_legacy_embed(
    state: TransformationState,
    description: str,
) -> Tuple[discord.Embed, Optional[discord.File]]:
    """Return the classic embed representation and optional avatar attachment."""
    embed = discord.Embed(
        description=description,
        color=0x9B59B6,
        timestamp=utc_now(),
    )
    embed.set_author(name=state.character_name)

    avatar_file: Optional[discord.File] = None
    avatar_bytes = await fetch_avatar_bytes(state.character_avatar_path)
    if avatar_bytes:
        suffix = Path(state.character_avatar_path).suffix or ".png"
        filename = f"tf-avatar-{state.user_id}{suffix}"
        avatar_file = discord.File(io.BytesIO(avatar_bytes), filename=filename)
        embed.set_thumbnail(url=f"attachment://{filename}")

    return embed, avatar_file


__all__ = ["build_legacy_embed"]

