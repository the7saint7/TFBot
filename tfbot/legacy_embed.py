"""Legacy embed-based avatar rendering helpers."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional, Tuple

import discord

from tfbot.models import TransformationState
from tfbot.panels import compose_state_avatar_image, fetch_avatar_bytes
from tfbot.utils import utc_now


async def build_legacy_embed(
    state: TransformationState,
    description: str,
) -> Tuple[discord.Embed, Optional[discord.File]]:
    """Return the classic embed representation and optional avatar attachment."""
    embed_color = 0x9B59B6
    character_name_normalized = (state.character_name or "").strip().lower()
    if state.is_inanimate and character_name_normalized == "ball":
        embed_color = 0x3498DB  # Match the ball's blue hue.

    embed = discord.Embed(
        description=description,
        color=embed_color,
        timestamp=utc_now(),
    )
    embed.set_author(name=state.character_name)

    avatar_file: Optional[discord.File] = None
    avatar_bytes: Optional[bytes] = None
    if getattr(state, "is_pillow", False):
        avatar_image = compose_state_avatar_image(state)
        if avatar_image is not None:
            buffer = io.BytesIO()
            avatar_image.save(buffer, format="PNG")
            avatar_bytes = buffer.getvalue()
    if avatar_bytes is None:
        avatar_bytes = await fetch_avatar_bytes(state.character_avatar_path)
    if avatar_bytes:
        suffix = Path(state.character_avatar_path).suffix or ".png"
        filename = f"tf-avatar-{state.user_id}{suffix}"
        avatar_file = discord.File(io.BytesIO(avatar_bytes), filename=filename)
        embed.set_thumbnail(url=f"attachment://{filename}")

    return embed, avatar_file


__all__ = ["build_legacy_embed"]
