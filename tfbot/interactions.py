"""Adapters for reusing command logic with Discord interactions."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import discord


class InteractionContextAdapter:
    """Minimal commands.Context-like adapter for slash interactions."""

    def __init__(
        self,
        interaction: discord.Interaction,
        *,
        default_ephemeral: bool = False,
        bot: Optional[discord.Client] = None,
    ):
        self.interaction = interaction
        self.guild = interaction.guild
        self.channel = interaction.channel
        self.author = interaction.user
        self.bot = bot or interaction.client
        self.default_ephemeral = default_ephemeral
        self._responded = False
        self.message = _InteractionMessageProxy(interaction)
        self._responded_flag = False

    @property
    def responded(self) -> bool:
        return self._responded_flag or self._responded

    async def reply(self, *args, **kwargs):
        kwargs.pop("mention_author", None)
        reference = kwargs.pop("reference", None)
        delete_after = kwargs.pop("delete_after", None)
        ephemeral = kwargs.pop("ephemeral", self.default_ephemeral)

        destination = self.channel if isinstance(self.channel, discord.abc.Messageable) else None
        if reference is not None or delete_after:
            if destination is not None:
                await destination.send(*args, reference=reference, delete_after=delete_after, **kwargs)
            else:
                if not self.interaction.response.is_done():
                    await self.interaction.response.defer(ephemeral=ephemeral)
                await self.interaction.followup.send(*args, ephemeral=ephemeral, **kwargs)
            self._responded = True
            self._responded_flag = True
            return

        if not self.interaction.response.is_done():
            await self.interaction.response.defer(ephemeral=ephemeral)
        await self.interaction.followup.send(*args, ephemeral=ephemeral, **kwargs)
        self._responded = True
        self._responded_flag = True

    async def send(self, *args, **kwargs):
        await self.reply(*args, **kwargs)

    def __repr__(self) -> str:
        return f"<InteractionContextAdapter guild={getattr(self.guild, 'id', None)} user={getattr(self.author, 'id', None)}>"


class _InteractionMessageProxy:
    """Provides minimal ctx.message attributes for adapter compatibility."""

    def __init__(self, interaction: discord.Interaction):
        self.id = interaction.id
        self.guild = interaction.guild
        self.channel = interaction.channel
        self.author = interaction.user

    async def delete(self, *_args, **_kwargs):
        return None

    def to_reference(self, **_kwargs):
        return None
