"""Roleplay thread helper commands/state."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import timedelta
from pathlib import Path
from typing import Dict, Optional

import discord
from discord.ext import commands

from .models import TransformationState
from .interactions import InteractionContextAdapter
from .state import active_transformations, persist_states, revert_tasks, state_key
from .swaps import unswap_chain
from .utils import is_admin, member_profile_name, utc_now

logger = logging.getLogger("tfbot.roleplay")


AssignmentPayload = Dict[str, object]


class RoleplayCog(commands.Cog):
    """Encapsulates RP forum post commands and identity handling."""

    def __init__(self, bot: commands.Bot, *, forum_post_id: int, state_file: Path):
        self.bot = bot
        self.forum_post_id = forum_post_id
        self.state_file = state_file
        self.dm_user_id: Optional[int] = None
        self.assignments: Dict[str, AssignmentPayload] = {}
        self._lock = asyncio.Lock()
        self._load_state()

    #
    # Persistent state helpers
    #
    def _assignment_key(self, guild_id: int, user_id: int) -> str:
        return f"{guild_id}:{user_id}"

    def _load_state(self) -> None:
        if not self.state_file.exists():
            return
        try:
            payload = json.loads(self.state_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse RP state %s: %s", self.state_file, exc)
            return
        dm_id = payload.get("dm_user_id")
        if isinstance(dm_id, int):
            self.dm_user_id = dm_id
        assignments = payload.get("assignments", {})
        if isinstance(assignments, dict):
            for key, entry in assignments.items():
                if not isinstance(entry, dict):
                    continue
                original = str(entry.get("original_character") or "").strip()
                if not original:
                    continue
                guild_id = int(entry.get("guild_id", 0))
                user_id = int(entry.get("user_id", 0))
                if guild_id <= 0 or user_id <= 0:
                    continue
                rename = str(entry.get("rename_override") or "").strip() or None
                self.assignments[key] = {
                    "guild_id": guild_id,
                    "user_id": user_id,
                    "original_character": original,
                    "rename_override": rename,
                }

    async def _save_state(self) -> None:
        async with self._lock:
            data = {
                "dm_user_id": self.dm_user_id,
                "forum_post_id": self.forum_post_id,
                "assignments": self.assignments,
            }
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    #
    # Query helpers
    #
    def is_roleplay_post(self, channel: Optional[discord.abc.GuildChannel]) -> bool:
        channel_id = getattr(channel, "id", None)
        logger.debug("RP is_roleplay_post? channel=%s target=%s", channel_id, self.forum_post_id)
        if not (self.forum_post_id and channel_id == self.forum_post_id):
            return False
        if isinstance(channel, discord.Thread):
            parent = getattr(channel, "parent", None)
            if isinstance(parent, discord.ForumChannel):
                return True
            # Allow legacy compatibility if they accidentally pointed at a text-thread.
            thread_type = getattr(channel, "type", None)
            return thread_type in {
                discord.ChannelType.public_thread,
                discord.ChannelType.private_thread,
                discord.ChannelType.news_thread,
            }
        return False

    def is_dm(self, member: Optional[discord.Member]) -> bool:
        if member is None:
            return False
        return self.dm_user_id is not None and member.id == self.dm_user_id

    def has_control(self, member: Optional[discord.Member]) -> bool:
        if member is None:
            return False
        if self.is_dm(member):
            return True
        guild = getattr(member, "guild", None)
        return guild is not None and guild.owner_id == member.id

    def resolve_display_name(self, guild_id: int, user_id: int) -> Optional[str]:
        entry = self.assignments.get(self._assignment_key(guild_id, user_id))
        if not entry:
            return None
        rename = entry.get("rename_override")
        if isinstance(rename, str) and rename.strip():
            return rename.strip()
        original = entry.get("original_character")
        return str(original).strip() or None

    async def _record_original_character(self, guild_id: int, user_id: int, character_name: str) -> None:
        key = self._assignment_key(guild_id, user_id)
        entry = self.assignments.get(key)
        if entry and str(entry.get("original_character")).strip():
            return
        self.assignments[key] = {
            "guild_id": guild_id,
            "user_id": user_id,
            "original_character": character_name,
            "rename_override": entry.get("rename_override") if entry else None,
        }
        await self._save_state()

    async def _set_rename(self, guild_id: int, user_id: int, *, new_name: str) -> None:
        key = self._assignment_key(guild_id, user_id)
        entry = self.assignments.setdefault(
            key,
            {
                "guild_id": guild_id,
                "user_id": user_id,
                "original_character": "",
                "rename_override": None,
            },
        )
        entry["rename_override"] = new_name
        await self._save_state()

    async def _unload_assignment(self, guild_id: int, user_id: int) -> bool:
        key = self._assignment_key(guild_id, user_id)
        if key not in self.assignments:
            return False
        self.assignments.pop(key, None)
        await self._save_state()
        return True

    async def _clear_active_transformation(self, guild_id: int, user_id: int) -> bool:
        key = state_key(guild_id, user_id)
        state = active_transformations.pop(key, None)
        if state is None:
            return False
        task = revert_tasks.pop(key, None)
        if task:
            task.cancel()
        persist_states()
        logger.debug("RP unload removed active transformation for user %s in guild %s", user_id, guild_id)
        return True

    async def _set_dm(self, member: discord.Member) -> None:
        self.dm_user_id = member.id
        await self._save_state()

    async def _announce_swap_reset(
        self,
        guild: discord.Guild,
        channel: Optional[discord.abc.Messageable],
        *,
        user_id: int,
        reason: str,
    ) -> bool:
        if channel is None:
            return False
        transitions = unswap_chain(guild.id, user_id)
        if not transitions:
            return False
        lines: list[str] = []
        for transition in transitions:
            member = guild.get_member(transition.user_id)
            display = member_profile_name(member) if member else f"User {transition.user_id}"
            lines.append(f"- {display}: {transition.before_form} -> {transition.after_form}")
        summary = "\n".join(lines) if lines else "No participants."
        try:
            await channel.send(
                f"Swap chain reset: {reason}\n{summary}",
                allowed_mentions=discord.AllowedMentions.none(),
            )
        except discord.HTTPException:
            logger.warning("Failed to announce swap reset in RP channel %s", getattr(channel, "id", "unknown"))
        return True

    #
    # Reroll helpers
    #
    def _snapshot_names(self, guild_id: int) -> Dict[str, str]:
        snapshot: Dict[str, str] = {}
        for key, state in active_transformations.items():
            g_id, _ = key
            if g_id != guild_id:
                continue
            snapshot[f"{key[0]}:{key[1]}"] = state.character_name
        return snapshot

    async def _finalize_reroll(self, guild: discord.Guild, before: Dict[str, str]) -> None:
        changed_states: Dict[str, TransformationState] = {}
        for key, state in active_transformations.items():
            if key[0] != guild.id:
                continue
            serialized = f"{key[0]}:{key[1]}"
            previous = before.get(serialized)
            if previous != state.character_name:
                changed_states[serialized] = state
        if not changed_states:
            return

        persist_needed = False
        now = utc_now()
        expires_at = now + timedelta(days=3650)
        for serialized, state in changed_states.items():
            key_tuple = (state.guild_id, state.user_id)
            task = revert_tasks.pop(key_tuple, None)
            if task:
                task.cancel()
            state.started_at = now
            state.expires_at = expires_at
            state.duration_label = "RP Session"
            state.avatar_applied = False
            await self._record_original_character(state.guild_id, state.user_id, state.character_name)
            persist_needed = True

        if persist_needed:
            persist_states()

    #
    # Command guards
    #
    async def _ensure_dm_actor(self, ctx: commands.Context) -> bool:
        if self.dm_user_id is None:
            await ctx.reply("No DM is configured yet. Ask an admin to run `/dm <user>` first.", mention_author=False)
            return False
        if not isinstance(ctx.author, discord.Member):
            await ctx.reply("These commands can only be used within a server.", mention_author=False)
            return False
        if self.has_control(ctx.author):
            return True
        if ctx.author.id != self.dm_user_id:
            await ctx.reply("Only the assigned DM can use that command in this thread.", mention_author=False)
            return False
        return True

    #
    # Commands
    #
    async def assign_dm_command(self, ctx: commands.Context, *, target: str = ""):
        if not self.forum_post_id:
            await ctx.reply("The RP forum post isn't configured on this bot.", mention_author=False)
            return
        if not self.is_roleplay_post(ctx.channel):
            logger.debug(
                "RP !dm ignored: channel %s is not the configured RP forum post (%s).",
                getattr(ctx.channel, "id", None),
                self.forum_post_id,
            )
            await ctx.reply("You can only assign the RP DM from inside the RP forum post.", mention_author=False)
            return
        guild = ctx.guild
        cleaned = target.strip()
        logger.debug(
            "RP !dm called in channel %s by %s (%s) with target='%s'",
            getattr(ctx.channel, "id", None),
            ctx.author,
            getattr(ctx.author, "id", "unknown"),
            cleaned,
        )
        if not cleaned:
            logger.debug(
                "RP !dm status invoked by %s (%s) in guild %s",
                ctx.author,
                getattr(ctx.author, "id", "unknown"),
                guild.id if guild else "dm",
            )
            dm_member = None
            if guild and self.dm_user_id:
                dm_member = guild.get_member(self.dm_user_id)
            if self.dm_user_id and dm_member:
                description = f"The current DM is {dm_member.mention}."
                footer = f"User ID: {dm_member.id}"
            elif self.dm_user_id:
                description = f"The current DM is <@{self.dm_user_id}>."
                footer = f"User ID: {self.dm_user_id}"
            else:
                description = "No DM has been assigned yet."
                footer = None
            embed = discord.Embed(
                title="RP Dungeon Master",
                description=description,
                color=0x9B59B6,
            )
            if footer:
                embed.set_footer(text=footer)
            logger.debug(
                "RP !dm sending embed response in channel %s: %s",
                getattr(ctx.channel, "id", None),
                description,
            )
            await ctx.send(embed=embed)
            return

        if not isinstance(ctx.author, discord.Member) or not (is_admin(ctx.author) or guild.owner_id == ctx.author.id):
            logger.debug("RP !dm denied: %s not authorized to assign DM.", ctx.author)
            await ctx.send("Only admins or the server owner can assign the DM.")
            return
        converter = commands.MemberConverter()
        try:
            member = await converter.convert(ctx, cleaned)
        except commands.BadArgument:
            logger.debug("RP !dm failed to resolve member from input '%s'", cleaned)
            await ctx.send(f"I couldn't find `{cleaned}` in this server.")
            return
        logger.debug(
            "RP !dm invoked by %s (%s) assigning %s (%s) in guild %s",
            ctx.author,
            ctx.author.id,
            member,
            member.id,
            guild.id if guild else "dm",
        )
        await self._set_dm(member)
        logger.debug("RP !dm assigned %s (%s) as DM.", member, member.id)
        await ctx.send(f"{member.mention} is now the RP DM.")

    async def narrator_shortcut_command(self, ctx: commands.Context, *, text: str = ""):
        if not self.is_roleplay_post(ctx.channel):
            await ctx.reply("`/n` can only be used inside the RP forum post.", mention_author=False)
            return
        if not await self._ensure_dm_actor(ctx):
            return
        cleaned = text.strip()
        if not cleaned:
            await ctx.reply("Usage: `/n <text>`", mention_author=False)
            return
        logger.debug(
            "RP !n invoked by %s (%s) with %s chars",
            ctx.author,
            ctx.author.id,
            len(cleaned),
        )
        say_command = self.bot.get_command("say")
        if say_command is None:
            await ctx.reply("Narrator command is currently unavailable.", mention_author=False)
            return
        await ctx.invoke(say_command, args=f"narrator {cleaned}")

    async def ball_shortcut_command(self, ctx: commands.Context, *, text: str = ""):
        if not self.is_roleplay_post(ctx.channel):
            await ctx.reply("`/b` can only be used inside the RP forum post.", mention_author=False)
            return
        if not await self._ensure_dm_actor(ctx):
            return
        cleaned = text.strip()
        if not cleaned:
            await ctx.reply("Usage: `/b <text>`", mention_author=False)
            return
        logger.debug(
            "RP !b invoked by %s (%s) with %s chars",
            ctx.author,
            ctx.author.id,
            len(cleaned),
        )
        say_command = self.bot.get_command("say")
        if say_command is None:
            await ctx.reply("Narrator command is currently unavailable.", mention_author=False)
            return
        await ctx.invoke(say_command, args=f"ball {cleaned}")

    async def reroll_shortcut_command(self, ctx: commands.Context, *, args: str = ""):
        if not self.is_roleplay_post(ctx.channel):
            await ctx.reply("`!r` can only be used inside the RP forum post.", mention_author=False)
            return
        if not await self._ensure_dm_actor(ctx):
            return
        reroll_command = self.bot.get_command("reroll")
        if reroll_command is None:
            await ctx.reply("Rerolling is currently unavailable.", mention_author=False)
            return
        logger.debug(
            "RP !r invoked by %s (%s) with args=%s",
            ctx.author,
            ctx.author.id,
            args,
        )
        before = self._snapshot_names(ctx.guild.id)
        await ctx.invoke(reroll_command, args=args)
        await self._finalize_reroll(ctx.guild, before)

    async def rename_identity_command(self, ctx: commands.Context, member: str, *, new_name: str):
        if not self.is_roleplay_post(ctx.channel):
            await ctx.reply("`/rename` can only be used inside the RP forum post.", mention_author=False)
            return
        if not await self._ensure_dm_actor(ctx):
            return
        member_arg = member.strip()
        if not member_arg:
            await ctx.send("Please specify which player to rename, e.g. `/rename @player New Name`.")
            return
        cleaned = new_name.strip()
        if not cleaned:
            await ctx.send("Provide the new display name, e.g. `/rename @player New Name`.")
            return
        converter = commands.MemberConverter()
        try:
            resolved_member = await converter.convert(ctx, member_arg)
        except commands.BadArgument:
            await ctx.send(f"I couldn't find `{member_arg}` in this server.")
            return
        logger.debug(
            "RP !rename invoked by %s (%s) target=%s (%s) new_name=%s",
            ctx.author,
            ctx.author.id,
            resolved_member,
            resolved_member.id,
            cleaned,
        )
        await self._set_rename(ctx.guild.id, resolved_member.id, new_name=cleaned)
        await ctx.reply(
            f"{member_profile_name(resolved_member)} will now appear as **{cleaned}** in VN panels.",
            mention_author=False,
        )

    async def unload_identity_command(self, ctx: commands.Context, *, member: str):
        if not self.is_roleplay_post(ctx.channel):
            await ctx.reply("`/unload` can only be used inside the RP forum post.", mention_author=False)
            return
        if not await self._ensure_dm_actor(ctx):
            return
        target_arg = member.strip()
        if not target_arg:
            await ctx.send("Specify which player to unload, e.g. `/unload @player` or `/unload all`.")
            return
        if target_arg.lower() == "all":
            guild_id = ctx.guild.id
            assignment_removed = 0
            tf_removed = 0
            keys_to_clear = [
                (entry.get("guild_id"), entry.get("user_id"))
                for entry in self.assignments.values()
                if entry.get("guild_id") == guild_id
            ]
            for g_id, user_id in keys_to_clear:
                if g_id is None or user_id is None:
                    continue
                if await self._unload_assignment(guild_id, int(user_id)):
                    assignment_removed += 1
            for key, state in list(active_transformations.items()):
                g_id, u_id = key
                if g_id != guild_id:
                    continue
                await self._announce_swap_reset(
                    ctx.guild,
                    ctx.channel,
                    user_id=u_id,
                    reason="RP unload (all)",
                )
                if await self._clear_active_transformation(g_id, u_id):
                    tf_removed += 1
            await ctx.reply(
                f"Unloaded {assignment_removed} RP entries and removed {tf_removed} active TFs.",
                mention_author=False,
            )
            return

        converter = commands.MemberConverter()
        try:
            resolved_member = await converter.convert(ctx, target_arg)
        except commands.BadArgument:
            await ctx.send(f"I couldn't find `{target_arg}` in this server.")
            return
        await self._announce_swap_reset(
            ctx.guild,
            ctx.channel,
            user_id=resolved_member.id,
            reason=f"RP unload for {member_profile_name(resolved_member)}",
        )
        assignment_removed = await self._unload_assignment(ctx.guild.id, resolved_member.id)
        tf_removed = await self._clear_active_transformation(ctx.guild.id, resolved_member.id)
        if assignment_removed or tf_removed:
            details: list[str] = []
            if assignment_removed:
                details.append("cleared RP roster entry")
            if tf_removed:
                details.append("stopped TF relay")
            note = " and ".join(details) if details else "reset"
            await ctx.reply(
                f"{member_profile_name(resolved_member)} has been unloaded ({note}).",
                mention_author=False,
            )
        else:
            await ctx.reply(
                f"No RP assignment or active TF was found for {member_profile_name(resolved_member)}.",
                mention_author=False,
            )


async def add_roleplay_cog(bot: commands.Bot, *, forum_post_id: int, state_file: Path) -> Optional[RoleplayCog]:
    if forum_post_id <= 0:
        return None
    cog = RoleplayCog(bot, forum_post_id=forum_post_id, state_file=state_file)
    await bot.add_cog(cog)
    logger.info("Roleplay forum post enabled for channel id %s", forum_post_id)
    return cog


__all__ = ["RoleplayCog", "add_roleplay_cog"]
