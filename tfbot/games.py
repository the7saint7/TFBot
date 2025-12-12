"""Game board framework for managing board games with visual boards and VN chat."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Sequence

import discord
from discord.ext import commands

from .game_models import GameConfig, GamePlayer, GameState
from .game_board import render_game_board, validate_coordinate
from .utils import is_admin, int_from_env, path_from_env
from .models import TransformationState

logger = logging.getLogger("tfbot.games")


def _load_game_config(config_path: Path) -> Optional[GameConfig]:
    """Load a game configuration from a JSON file."""
    if not config_path.exists():
        return None
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse game config %s: %s", config_path, exc)
        return None
    
    name = str(data.get("name", config_path.stem))
    board_image = str(data.get("board_image", ""))
    grid = data.get("grid", {})
    dice = data.get("dice", {})
    rules = data.get("rules")
    tokens = data.get("tokens")
    
    return GameConfig(
        name=name,
        board_image=board_image,
        grid=grid if isinstance(grid, dict) else {},
        dice=dice if isinstance(dice, dict) else {},
        rules=rules if isinstance(rules, dict) else None,
        tokens=tokens if isinstance(tokens, dict) else None,
    )


def _scan_game_configs(configs_dir: Path) -> Dict[str, GameConfig]:
    """Scan games/configs/ directory for JSON files and load them."""
    if not configs_dir.exists():
        return {}
    
    configs: Dict[str, GameConfig] = {}
    for config_file in configs_dir.glob("*.json"):
        config = _load_game_config(config_file)
        if config:
            game_key = config_file.stem  # filename without .json extension
            configs[game_key] = config
            logger.info("Loaded game config: %s (%s)", game_key, config.name)
    
    return configs


class GameBoardManager:
    """Manages game board framework - forum threads, game state, and commands."""

    def __init__(
        self,
        *,
        bot: commands.Bot,
        config_path: Optional[Path] = None,
        assets_dir: Optional[Path] = None,
    ) -> None:
        self.bot = bot
        self.config_path = config_path or path_from_env("TFBOT_GAME_CONFIG_FILE") or Path("games/game_config.json")
        self.assets_dir = assets_dir or path_from_env("TFBOT_GAME_ASSETS_DIR") or Path("games/assets")
        
        # Load main config
        self._config = self._load_main_config()
        # Use environment variable if available, otherwise use config file
        env_forum_id = int_from_env("TFBOT_GAME_FORUM_CHANNEL_ID", 0)
        self.forum_channel_id = env_forum_id if env_forum_id > 0 else int(self._config.get("forum_channel_id", 0))
        env_dm_id = int_from_env("TFBOT_GAME_DM_CHANNEL_ID", 0)
        self.dm_channel_id = env_dm_id if env_dm_id > 0 else int(self._config.get("dm_channel_id", 0))
        
        # Automatic game detection - scan configs directory
        configs_dir = self.config_path.parent / "configs"
        self._game_configs: Dict[str, GameConfig] = _scan_game_configs(configs_dir)
        
        # Active game state (one game per thread)
        self._active_games: Dict[int, GameState] = {}  # thread_id -> GameState
        self._lock = asyncio.Lock()
        
        # States directory
        self.states_dir = self.config_path.parent / "states"
        self.states_dir.mkdir(parents=True, exist_ok=True)
        
        # Load any active games from disk
        self._load_active_games()
        
        logger.info(
            "GameBoardManager initialized: forum_channel_id=%s, detected %d games",
            self.forum_channel_id,
            len(self._game_configs),
        )

    def _load_main_config(self) -> Dict[str, object]:
        """Load the main game_config.json file."""
        if not self.config_path.exists():
            logger.warning("Game config file not found at %s, using defaults", self.config_path)
            return {}
        try:
            return json.loads(self.config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse game config %s: %s", self.config_path, exc)
            return {}

    def _load_active_games(self) -> None:
        """Load active game states from disk."""
        if not self.states_dir.exists():
            return
        
        for state_file in self.states_dir.glob("*.json"):
            try:
                data = json.loads(state_file.read_text(encoding="utf-8"))
                thread_id = int(data.get("game_thread_id", 0))
                if thread_id <= 0:
                    continue
                
                # Deserialize GameState from JSON
                players_dict = data.get("players", {})
                players = {}
                for user_id_str, player_data in players_dict.items():
                    try:
                        user_id = int(user_id_str)
                        players[user_id] = GamePlayer(
                            user_id=user_id,
                            character_name=player_data.get("character_name"),
                            grid_position=player_data.get("grid_position", "A1"),
                            background_id=player_data.get("background_id"),
                            outfit_name=player_data.get("outfit_name"),
                            token_image=player_data.get("token_image", "default.png"),
                        )
                    except (ValueError, KeyError) as exc:
                        logger.warning("Failed to load player data in state %s: %s", state_file.name, exc)
                        continue
                
                game_state = GameState(
                    game_thread_id=thread_id,
                    forum_channel_id=int(data.get("forum_channel_id", 0)),
                    dm_channel_id=int(data.get("dm_channel_id", 0)),
                    gm_user_id=int(data.get("gm_user_id", 0)),
                    game_type=str(data.get("game_type", "")),
                    players=players,
                    current_turn=data.get("current_turn"),
                    board_message_id=data.get("board_message_id"),
                    is_locked=bool(data.get("is_locked", False)),
                )
                
                self._active_games[thread_id] = game_state
                logger.info("Loaded game state: thread_id=%s, game_type=%s", thread_id, game_state.game_type)
            except (json.JSONDecodeError, ValueError, KeyError) as exc:
                logger.warning("Failed to load game state %s: %s", state_file, exc)

    def is_game_thread(self, channel: Optional[discord.abc.GuildChannel]) -> bool:
        """Check if a channel is an active game thread."""
        if not isinstance(channel, discord.Thread):
            return False
        
        thread_id = channel.id
        return thread_id in self._active_games

    def _get_character_by_name(self, character_name: str):
        """Get character by name, using lazy import to avoid circular dependency."""
        try:
            # Lazy import to avoid circular dependency
            import sys
            bot_module = sys.modules.get('bot')
            if bot_module:
                CHARACTER_BY_NAME = getattr(bot_module, 'CHARACTER_BY_NAME', None)
                if CHARACTER_BY_NAME:
                    return CHARACTER_BY_NAME.get(character_name.strip().lower())
        except Exception as exc:
            logger.warning("Failed to lookup character %s: %s", character_name, exc)
        return None

    def _create_game_state_for_player(
        self,
        player: GamePlayer,
        user_id: int,
        guild_id: int,
        character_name: str,
    ) -> Optional[TransformationState]:
        """Create a TransformationState from game player data."""
        from tfbot.models import TFCharacter
        
        character = self._get_character_by_name(character_name)
        if not character or not isinstance(character, TFCharacter):
            logger.warning("Character not found for game player: %s", character_name)
            return None
        
        now = datetime.utcnow()
        return TransformationState(
            user_id=user_id,
            guild_id=guild_id,
            character_name=character_name,
            character_avatar_path=str(character.avatar_path) if character.avatar_path else "",
            character_message=character.message or "",
            original_nick=None,
            started_at=now,
            expires_at=now + timedelta(days=365),  # Long duration for games
            duration_label="Game",
            character_folder=character.folder,
            avatar_applied=False,
            original_display_name="",
            is_inanimate=False,
            inanimate_responses=tuple(),
        )

    def _get_game_background_path(self, background_id: Optional[int]) -> Optional[Path]:
        """Get background path from background_id index."""
        if background_id is None:
            return None
        
        # Import here to avoid circular dependency
        from tfbot.panels import list_background_choices, VN_BACKGROUND_DEFAULT_PATH
        
        backgrounds = list_background_choices()
        if 1 <= background_id <= len(backgrounds):
            return backgrounds[background_id - 1]  # 1-indexed
        
        # Fall back to default
        if VN_BACKGROUND_DEFAULT_PATH and VN_BACKGROUND_DEFAULT_PATH.exists():
            return VN_BACKGROUND_DEFAULT_PATH
        
        return backgrounds[0] if backgrounds else None

    async def handle_message(self, message: discord.Message, *, command_invoked: bool) -> bool:
        """Handle a message in a game thread. Returns True if handled."""
        if message.author.bot or command_invoked:
            return False
        
        if not isinstance(message.channel, discord.Thread):
            return False
        
        if not message.guild:
            return False
        
        thread_id = message.channel.id
        game_state = self._active_games.get(thread_id)
        if not game_state:
            return False
        
        if game_state.is_locked:
            return False
        
        # Check if player is in the game and has a character assigned
        player = game_state.players.get(message.author.id)
        if not player or not player.character_name:
            # Player not in game or no character assigned - ignore message
            return True
        
        # Get character data and create state
        state = self._create_game_state_for_player(
            player,
            message.author.id,
            message.guild.id,
            player.character_name,
        )
        if not state:
            return True
        
        # Render VN panel
        try:
            from tfbot.panels import (
                render_vn_panel,
                parse_discord_formatting,
                prepare_custom_emoji_images,
            )
            from .models import ReplyContext
            
            # Get MESSAGE_STYLE and strip_urls via lazy import
            import sys
            bot_module = sys.modules.get('bot')
            MESSAGE_STYLE = getattr(bot_module, 'MESSAGE_STYLE', 'classic') if bot_module else 'classic'
            strip_urls = getattr(bot_module, 'strip_urls', lambda x, y: (x, [])) if bot_module else lambda x, y: (x, [])
            
            # Clean message content
            cleaned_content = message.content.strip()
            cleaned_content, _ = strip_urls(cleaned_content)
            cleaned_content = cleaned_content.strip()
            
            if not cleaned_content:
                # No content after filtering - ignore
                return True
            
            # Parse formatting
            formatted_segments = parse_discord_formatting(cleaned_content)
            custom_emoji_images = await prepare_custom_emoji_images(message, formatted_segments)
            
            # Get reply context if message is a reply
            reply_context = None
            if message.reference and message.reference.resolved:
                ref_msg = message.reference.resolved
                if isinstance(ref_msg, discord.Message):
                    reply_context = ReplyContext(
                        author=ref_msg.author.display_name,
                        text=ref_msg.content[:200] if ref_msg.content else "",
                    )
            
            # Get background path from game player (completely isolated from global state)
            background_path = self._get_game_background_path(player.background_id)
            
            # Render VN panel (only if VN style is enabled)
            files = []
            if MESSAGE_STYLE == "vn":
                # Create a wrapper to use game-specific background without touching global state
                # We'll temporarily monkey-patch get_selected_background_path for this render
                from tfbot.panels import get_selected_background_path as global_get_bg
                
                def game_background_getter(user_id: int) -> Optional[Path]:
                    """Override background lookup to use game-specific background."""
                    if user_id == message.author.id and background_path:
                        return background_path
                    # Fall back to default behavior for other users (shouldn't happen in games)
                    return global_get_bg(user_id)
                
                # Temporarily replace the function
                import tfbot.panels as panels_module
                original_func = panels_module.get_selected_background_path
                panels_module.get_selected_background_path = game_background_getter
                
                try:
                    # Render with game-specific outfit (already isolated via override parameter)
                    vn_file = render_vn_panel(
                        state=state,
                        message_content=cleaned_content,
                        character_display_name=player.character_name,
                        original_name=message.author.display_name,
                        attachment_id=str(message.id),
                        formatted_segments=formatted_segments,
                        custom_emoji_images=custom_emoji_images,
                        reply_context=reply_context,
                        gacha_outfit_override=player.outfit_name if player.outfit_name else None,
                    )
                    if vn_file:
                        files.append(vn_file)
                finally:
                    # Restore original function (critical for isolation)
                    panels_module.get_selected_background_path = original_func
            
            # If we have files to send, send them and delete original
            if files:
                send_kwargs: Dict[str, object] = {
                    "files": files,
                    "allowed_mentions": discord.AllowedMentions.none(),
                }
                
                # Preserve reply reference if present
                if message.reference:
                    send_kwargs["reference"] = message.reference
                
                try:
                    await message.channel.send(**send_kwargs)
                    # Delete original message
                    try:
                        await message.delete()
                    except discord.HTTPException:
                        pass  # Message might already be deleted
                except discord.HTTPException as exc:
                    logger.warning("Failed to send game VN panel: %s", exc)
            
        except Exception as exc:
            logger.exception("Error rendering game VN panel: %s", exc)
        
        return True

    def get_game_config(self, game_type: str) -> Optional[GameConfig]:
        """Get game configuration by type name."""
        return self._game_configs.get(game_type)

    def list_available_games(self) -> Sequence[str]:
        """Get list of available game types."""
        return list(self._game_configs.keys())

    def _is_gm(self, member: Optional[discord.Member], game_state: Optional[GameState] = None) -> bool:
        """Check if member is GM for a game or is admin."""
        if not isinstance(member, discord.Member):
            return False
        if is_admin(member):
            return True
        if game_state and game_state.gm_user_id == member.id:
            return True
        return False

    def _get_game_state_for_context(self, ctx: commands.Context) -> Optional[GameState]:
        """Get game state for a command context (thread or DM channel)."""
        if isinstance(ctx.channel, discord.Thread):
            return self._active_games.get(ctx.channel.id)
        # Could also check DM channel
        return None

    async def _save_game_state(self, game_state: GameState) -> None:
        """Save game state to disk."""
        async with self._lock:
            state_file = self.states_dir / f"{game_state.game_thread_id}.json"
            data = {
                "game_thread_id": game_state.game_thread_id,
                "forum_channel_id": game_state.forum_channel_id,
                "dm_channel_id": game_state.dm_channel_id,
                "gm_user_id": game_state.gm_user_id,
                "game_type": game_state.game_type,
                "current_turn": game_state.current_turn,
                "board_message_id": game_state.board_message_id,
                "is_locked": game_state.is_locked,
                "players": {
                    str(user_id): {
                        "user_id": player.user_id,
                        "character_name": player.character_name,
                        "grid_position": player.grid_position,
                        "background_id": player.background_id,
                        "outfit_name": player.outfit_name,
                        "token_image": player.token_image,
                    }
                    for user_id, player in game_state.players.items()
                },
            }
            state_file.parent.mkdir(parents=True, exist_ok=True)
            state_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    async def _update_board(self, game_state: GameState) -> None:
        """Update board image (post new message + update pinned)."""
        game_config = self.get_game_config(game_state.game_type)
        if not game_config:
            return
        
        thread = self.bot.get_channel(game_state.game_thread_id)
        if not isinstance(thread, discord.Thread):
            return
        
        board_file = render_game_board(game_state, game_config, self.assets_dir)
        if not board_file:
            return
        
        # Post new board message
        try:
            await thread.send(file=board_file)
        except discord.HTTPException as exc:
            logger.warning("Failed to post board update: %s", exc)
        
        # Update pinned message (delete old and create new, since Discord doesn't allow editing attachments)
        if game_state.board_message_id:
            try:
                old_pinned = await thread.fetch_message(game_state.board_message_id)
                await old_pinned.delete()
            except discord.HTTPException:
                pass  # Message might not exist anymore
        
        # Create and pin new board message
        try:
            pinned_msg = await thread.send(file=board_file)
            await pinned_msg.pin()
            game_state.board_message_id = pinned_msg.id
            await self._save_game_state(game_state)
        except discord.HTTPException as exc:
            logger.warning("Failed to create pinned board: %s", exc)

    async def _log_action(self, game_state: GameState, action: str) -> None:
        """Log a game action."""
        # TODO: Implement logging to file or channel
        logger.info("Game action [thread %s]: %s", game_state.game_thread_id, action)

    # GM Command Methods
    async def command_startgame(self, ctx: commands.Context, game_type: str = "") -> None:
        """Start a new game (GM only)."""
        if not isinstance(ctx.author, discord.Member):
            await ctx.reply("This command can only be used inside a server.", mention_author=False)
            return
        
        if not is_admin(ctx.author):
            await ctx.reply("Only admins can start games.", mention_author=False)
            return
        
        if not game_type:
            await ctx.reply("Usage: `!startgame <game_type>`\nAvailable games: " + ", ".join(self.list_available_games()), mention_author=False)
            return
        
        game_config = self.get_game_config(game_type)
        if not game_config:
            await ctx.reply(f"Unknown game type: `{game_type}`. Available: " + ", ".join(self.list_available_games()), mention_author=False)
            return
        
        # Check if forum channel exists
        forum_channel = self.bot.get_channel(self.forum_channel_id)
        if not isinstance(forum_channel, discord.ForumChannel):
            await ctx.reply("Forum channel not configured or not found.", mention_author=False)
            return
        
        # Create forum thread
        thread_name = f"{game_config.name} - {ctx.author.display_name}"
        try:
            thread = await forum_channel.create_thread(name=thread_name, auto_archive_duration=1440)
        except discord.HTTPException as exc:
            await ctx.reply(f"Failed to create game thread: {exc}", mention_author=False)
            return
        
        # Create game state
        game_state = GameState(
            game_thread_id=thread.id,
            forum_channel_id=forum_channel.id,
            dm_channel_id=self.dm_channel_id,
            gm_user_id=ctx.author.id,
            game_type=game_type,
        )
        
        self._active_games[thread.id] = game_state
        await self._save_game_state(game_state)
        
        # Post initial board
        await self._update_board(game_state)
        
        await ctx.reply(f"Game started! Thread: {thread.mention}", mention_author=False)
        await self._log_action(game_state, f"Game started by {ctx.author.display_name}")

    async def command_listgames(self, ctx: commands.Context) -> None:
        """List available games."""
        games = self.list_available_games()
        if not games:
            await ctx.reply("No games configured.", mention_author=False)
            return
        
        lines = ["**Available Games:**"]
        for game_key in games:
            config = self._game_configs.get(game_key)
            name = config.name if config else game_key
            lines.append(f"- `{game_key}` - {name}")
        
        await ctx.reply("\n".join(lines), mention_author=False)

    async def command_addplayer(self, ctx: commands.Context, member: Optional[discord.Member] = None) -> None:
        """Add a player to the game (GM only)."""
        if not isinstance(ctx.author, discord.Member):
            await ctx.reply("This command can only be used inside a server.", mention_author=False)
            return
        
        game_state = self._get_game_state_for_context(ctx)
        if not game_state:
            await ctx.reply("No active game in this thread.", mention_author=False)
            return
        
        if not self._is_gm(ctx.author, game_state):
            await ctx.reply("Only the GM can add players.", mention_author=False)
            return
        
        if not member:
            await ctx.reply("Usage: `!addplayer @user`", mention_author=False)
            return
        
        if member.id in game_state.players:
            await ctx.reply(f"{member.mention} is already in the game.", mention_author=False)
            return
        
        game_state.players[member.id] = GamePlayer(user_id=member.id, grid_position="A1")
        await self._save_game_state(game_state)
        await ctx.reply(f"Added {member.mention} to the game.", mention_author=False)
        await self._log_action(game_state, f"Player {member.display_name} added")

    async def command_assign(self, ctx: commands.Context, member: Optional[discord.Member] = None, *, character_name: str = "") -> None:
        """Assign a character to a player (GM only)."""
        if not isinstance(ctx.author, discord.Member):
            await ctx.reply("This command can only be used inside a server.", mention_author=False)
            return
        
        game_state = self._get_game_state_for_context(ctx)
        if not game_state:
            await ctx.reply("No active game in this thread.", mention_author=False)
            return
        
        if not self._is_gm(ctx.author, game_state):
            await ctx.reply("Only the GM can assign characters.", mention_author=False)
            return
        
        if not member or not character_name:
            await ctx.reply("Usage: `!assign @user <character>`", mention_author=False)
            return
        
        if member.id not in game_state.players:
            await ctx.reply(f"{member.mention} is not in the game. Add them first with `!addplayer`.", mention_author=False)
            return
        
        game_state.players[member.id].character_name = character_name.strip()
        await self._save_game_state(game_state)
        await ctx.reply(f"Assigned {character_name} to {member.mention}.", mention_author=False)
        await self._log_action(game_state, f"{member.display_name} assigned character: {character_name}")

    async def command_reroll(self, ctx: commands.Context, member: Optional[discord.Member] = None) -> None:
        """Randomly reroll a player's character (GM only)."""
        if not isinstance(ctx.author, discord.Member):
            await ctx.reply("This command can only be used inside a server.", mention_author=False)
            return
        
        game_state = self._get_game_state_for_context(ctx)
        if not game_state:
            await ctx.reply("No active game in this thread.", mention_author=False)
            return
        
        if not self._is_gm(ctx.author, game_state):
            await ctx.reply("Only the GM can reroll characters.", mention_author=False)
            return
        
        if not member:
            await ctx.reply("Usage: `!reroll @user`", mention_author=False)
            return
        
        if member.id not in game_state.players:
            await ctx.reply(f"{member.mention} is not in the game.", mention_author=False)
            return
        
        # TODO: Get character pool and randomly select
        # For now, just clear the character
        old_char = game_state.players[member.id].character_name
        game_state.players[member.id].character_name = None
        await self._save_game_state(game_state)
        await ctx.reply(f"Rerolled {member.mention}'s character (was: {old_char}).", mention_author=False)
        await self._log_action(game_state, f"{member.display_name} character rerolled")

    async def command_swap(self, ctx: commands.Context, member1: Optional[discord.Member] = None, member2: Optional[discord.Member] = None) -> None:
        """Swap characters between two players (GM only)."""
        if not isinstance(ctx.author, discord.Member):
            await ctx.reply("This command can only be used inside a server.", mention_author=False)
            return
        
        game_state = self._get_game_state_for_context(ctx)
        if not game_state:
            await ctx.reply("No active game in this thread.", mention_author=False)
            return
        
        if not self._is_gm(ctx.author, game_state):
            await ctx.reply("Only the GM can swap characters.", mention_author=False)
            return
        
        if not member1 or not member2:
            await ctx.reply("Usage: `!swap @user1 @user2`", mention_author=False)
            return
        
        if member1.id not in game_state.players or member2.id not in game_state.players:
            await ctx.reply("Both players must be in the game.", mention_author=False)
            return
        
        # Swap characters
        char1 = game_state.players[member1.id].character_name
        char2 = game_state.players[member2.id].character_name
        game_state.players[member1.id].character_name = char2
        game_state.players[member2.id].character_name = char1
        
        await self._save_game_state(game_state)
        await ctx.reply(f"Swapped characters: {member1.mention} ↔ {member2.mention}", mention_author=False)
        await self._log_action(game_state, f"{member1.display_name} and {member2.display_name} swapped characters")

    async def command_movetoken(self, ctx: commands.Context, member: Optional[discord.Member] = None, *, position: str = "") -> None:
        """Move a player's token (GM only)."""
        if not isinstance(ctx.author, discord.Member):
            await ctx.reply("This command can only be used inside a server.", mention_author=False)
            return
        
        game_state = self._get_game_state_for_context(ctx)
        if not game_state:
            await ctx.reply("No active game in this thread.", mention_author=False)
            return
        
        if not self._is_gm(ctx.author, game_state):
            await ctx.reply("Only the GM can move tokens.", mention_author=False)
            return
        
        if not member or not position:
            await ctx.reply("Usage: `!movetoken @user <coord>` (e.g., `!movetoken @user A1`)", mention_author=False)
            return
        
        if member.id not in game_state.players:
            await ctx.reply(f"{member.mention} is not in the game.", mention_author=False)
            return
        
        position = position.strip().upper()
        game_config = self.get_game_config(game_state.game_type)
        if not game_config:
            await ctx.reply("Game configuration not found.", mention_author=False)
            return
        
        if not validate_coordinate(position, game_config):
            await ctx.reply(f"Invalid coordinate: `{position}`. Check bounds and blocked cells.", mention_author=False)
            return
        
        old_pos = game_state.players[member.id].grid_position
        game_state.players[member.id].grid_position = position
        await self._save_game_state(game_state)
        await self._update_board(game_state)
        await ctx.reply(f"Moved {member.mention}'s token from {old_pos} to {position}.", mention_author=False)
        await self._log_action(game_state, f"{member.display_name} token moved to {position}")

    async def command_dice(self, ctx: commands.Context) -> None:
        """Roll dice (player command)."""
        game_state = self._get_game_state_for_context(ctx)
        if not game_state:
            await ctx.reply("No active game in this thread.", mention_author=False)
            return
        
        if game_state.is_locked:
            await ctx.reply("Game is locked.", mention_author=False)
            return
        
        game_config = self.get_game_config(game_state.game_type)
        if not game_config:
            await ctx.reply("Game configuration not found.", mention_author=False)
            return
        
        dice_count = int(game_config.dice.get("count", 1))
        dice_faces = int(game_config.dice.get("faces", 6))
        
        rolls = [random.randint(1, dice_faces) for _ in range(dice_count)]
        total = sum(rolls)
        
        if dice_count == 1:
            result = f"Rolled: **{rolls[0]}**"
        else:
            rolls_str = ", ".join(str(r) for r in rolls)
            result = f"Rolled: {rolls_str} = **{total}**"
        
        await ctx.reply(result, mention_author=False)
        await self._log_action(game_state, f"{ctx.author.display_name} rolled {result}")

    async def command_endgame(self, ctx: commands.Context) -> None:
        """End the current game and lock the thread (GM only)."""
        if not isinstance(ctx.author, discord.Member):
            await ctx.reply("This command can only be used inside a server.", mention_author=False)
            return
        
        game_state = self._get_game_state_for_context(ctx)
        if not game_state:
            await ctx.reply("No active game in this thread.", mention_author=False)
            return
        
        if not self._is_gm(ctx.author, game_state):
            await ctx.reply("Only the GM can end games.", mention_author=False)
            return
        
        game_state.is_locked = True
        await self._save_game_state(game_state)
        
        # Lock the thread
        if isinstance(ctx.channel, discord.Thread):
            try:
                await ctx.channel.edit(locked=True)
            except discord.HTTPException as exc:
                logger.warning("Failed to lock thread: %s", exc)
        
        await ctx.reply("Game ended. Thread locked.", mention_author=False)
        await self._log_action(game_state, f"Game ended by {ctx.author.display_name}")

    async def command_removeplayer(self, ctx: commands.Context, member: Optional[discord.Member] = None) -> None:
        """Remove a player from the game (GM only)."""
        if not isinstance(ctx.author, discord.Member):
            await ctx.reply("This command can only be used inside a server.", mention_author=False)
            return
        
        game_state = self._get_game_state_for_context(ctx)
        if not game_state:
            await ctx.reply("No active game in this thread.", mention_author=False)
            return
        
        if not self._is_gm(ctx.author, game_state):
            await ctx.reply("Only the GM can remove players.", mention_author=False)
            return
        
        if not member:
            await ctx.reply("Usage: `!removeplayer @user`", mention_author=False)
            return
        
        if member.id not in game_state.players:
            await ctx.reply(f"{member.mention} is not in the game.", mention_author=False)
            return
        
        del game_state.players[member.id]
        await self._save_game_state(game_state)
        await ctx.reply(f"Removed {member.mention} from the game.", mention_author=False)
        await self._log_action(game_state, f"Player {member.display_name} removed")

    async def command_bg_list(self, ctx: commands.Context) -> None:
        """List available backgrounds (game-specific, isolated from global VN)."""
        from tfbot.panels import list_background_choices
        
        choices = list_background_choices()
        if not choices:
            await ctx.reply("No background images found.", mention_author=False)
            return
        
        lines: list[str] = []
        for idx, path in enumerate(choices, start=1):
            try:
                from tfbot.panels import VN_BACKGROUND_ROOT
                if VN_BACKGROUND_ROOT:
                    relative = path.resolve().relative_to(VN_BACKGROUND_ROOT.resolve())
                    display = relative.as_posix()
                else:
                    display = path.name
            except ValueError:
                display = path.name
            lines.append(f"{idx}: {display}")
        
        # Split into chunks if too long
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
        
        header = "**Available Backgrounds (Game VN):**\n"
        header += "Use `!bg @user <number>` or `!bg all <number>` to set.\n\n"
        
        for chunk in chunks:
            await ctx.reply(header + chunk, mention_author=False)
            header = ""  # Only show header once

    async def command_bg(self, ctx: commands.Context, target: Optional[discord.Member] = None, *, bg_id: str = "") -> None:
        """Set background for a player or all players (GM only) - game-specific, isolated."""
        if not isinstance(ctx.author, discord.Member):
            await ctx.reply("This command can only be used inside a server.", mention_author=False)
            return
        
        game_state = self._get_game_state_for_context(ctx)
        if not game_state:
            await ctx.reply("No active game in this thread.", mention_author=False)
            return
        
        if not self._is_gm(ctx.author, game_state):
            await ctx.reply("Only the GM can set backgrounds.", mention_author=False)
            return
        
        bg_id = bg_id.strip()
        if not bg_id:
            await self.command_bg_list(ctx)
            return
        
        try:
            bg_id_int = int(bg_id)
        except ValueError:
            await ctx.reply("Background ID must be a number.", mention_author=False)
            return
        
        # Validate background ID exists
        from tfbot.panels import list_background_choices
        choices = list_background_choices()
        if bg_id_int < 1 or bg_id_int > len(choices):
            await ctx.reply(f"Background ID must be between 1 and {len(choices)}.", mention_author=False)
            return
        
        # Check if target is None (meaning "all" was passed as string)
        if target is None:
            # Set for all players (game-specific only, doesn't touch global state)
            for player in game_state.players.values():
                player.background_id = bg_id_int
            await self._save_game_state(game_state)
            await ctx.reply(f"Set background {bg_id_int} for all players (game VN only).", mention_author=False)
            await self._log_action(game_state, f"All players background set to {bg_id_int}")
        elif target.id in game_state.players:
            game_state.players[target.id].background_id = bg_id_int
            await self._save_game_state(game_state)
            await ctx.reply(f"Set background {bg_id_int} for {target.mention} (game VN only).", mention_author=False)
            await self._log_action(game_state, f"{target.display_name} background set to {bg_id_int}")
        else:
            await ctx.reply(f"{target.mention} is not in the game.", mention_author=False)

    async def command_outfit_list(self, ctx: commands.Context, character_name: Optional[str] = None) -> None:
        """List available outfits for a character (game-specific, isolated from global VN)."""
        game_state = self._get_game_state_for_context(ctx)
        if not game_state:
            await ctx.reply("No active game in this thread.", mention_author=False)
            return
        
        if not character_name:
            # Show current game players and their characters
            lines = ["**Current Game Players:**"]
            for player in game_state.players.values():
                if player.character_name:
                    char_info = f"- {player.character_name}"
                    if player.outfit_name:
                        char_info += f" (current outfit: {player.outfit_name})"
                    lines.append(char_info)
            await ctx.reply("\n".join(lines) + "\n\nUse `!outfit @user <outfit>` to set.", mention_author=False)
            return
        
        # List outfits for specific character
        from tfbot.panels import list_available_outfits
        
        outfits = list_available_outfits(character_name)
        if not outfits:
            await ctx.reply(f"No outfits found for character: {character_name}", mention_author=False)
            return
        
        lines = [f"**Available Outfits for {character_name} (Game VN):**"]
        for outfit in outfits:
            lines.append(f"- {outfit}")
        
        await ctx.reply("\n".join(lines) + "\n\nUse `!outfit @user <outfit>` to set.", mention_author=False)

    async def command_outfit(self, ctx: commands.Context, member: Optional[discord.Member] = None, *, outfit_name: str = "") -> None:
        """Set outfit for a player (GM only) - game-specific, isolated from global VN."""
        if not isinstance(ctx.author, discord.Member):
            await ctx.reply("This command can only be used inside a server.", mention_author=False)
            return
        
        game_state = self._get_game_state_for_context(ctx)
        if not game_state:
            await ctx.reply("No active game in this thread.", mention_author=False)
            return
        
        if not self._is_gm(ctx.author, game_state):
            await ctx.reply("Only the GM can set outfits.", mention_author=False)
            return
        
        if not member:
            # Show list of current players
            await self.command_outfit_list(ctx)
            return
        
        if not outfit_name:
            # Show outfits for this player's character
            if member.id not in game_state.players:
                await ctx.reply(f"{member.mention} is not in the game.", mention_author=False)
                return
            player = game_state.players[member.id]
            if not player.character_name:
                await ctx.reply(f"{member.mention} doesn't have a character assigned yet.", mention_author=False)
                return
            await self.command_outfit_list(ctx, character_name=player.character_name)
            return
        
        if member.id not in game_state.players:
            await ctx.reply(f"{member.mention} is not in the game.", mention_author=False)
            return
        
        player = game_state.players[member.id]
        if not player.character_name:
            await ctx.reply(f"{member.mention} doesn't have a character assigned yet. Use `!assign` first.", mention_author=False)
            return
        
        # Set outfit (game-specific only, doesn't touch global vn_outfit_selection)
        game_state.players[member.id].outfit_name = outfit_name.strip()
        await self._save_game_state(game_state)
        await ctx.reply(f"Set outfit '{outfit_name}' for {member.mention} (game VN only).", mention_author=False)
        await self._log_action(game_state, f"{member.display_name} outfit set to {outfit_name}")

    async def command_savegame(self, ctx: commands.Context) -> None:
        """Save the current game state (GM only)."""
        if not isinstance(ctx.author, discord.Member):
            await ctx.reply("This command can only be used inside a server.", mention_author=False)
            return
        
        game_state = self._get_game_state_for_context(ctx)
        if not game_state:
            await ctx.reply("No active game in this thread.", mention_author=False)
            return
        
        if not self._is_gm(ctx.author, game_state):
            await ctx.reply("Only the GM can save games.", mention_author=False)
            return
        
        await self._save_game_state(game_state)
        await ctx.reply("Game state saved.", mention_author=False)

    async def command_loadgame(self, ctx: commands.Context, state_file: str = "") -> None:
        """Load a saved game state (GM only)."""
        if not isinstance(ctx.author, discord.Member):
            await ctx.reply("This command can only be used inside a server.", mention_author=False)
            return
        
        if not is_admin(ctx.author):
            await ctx.reply("Only admins can load games.", mention_author=False)
            return
        
        if not state_file:
            await ctx.reply("Usage: `!loadgame <state_file>`", mention_author=False)
            return
        
        # TODO: Implement game state loading from file
        await ctx.reply("Game state loading not yet implemented.", mention_author=False)

    async def command_rules(self, ctx: commands.Context) -> None:
        """Show game rules (player command)."""
        game_state = self._get_game_state_for_context(ctx)
        if not game_state:
            await ctx.reply("No active game in this thread.", mention_author=False)
            return
        
        game_config = self.get_game_config(game_state.game_type)
        if not game_config:
            await ctx.reply("Game configuration not found.", mention_author=False)
            return
        
        rules = game_config.rules
        if not rules:
            await ctx.reply("No rules configured for this game.", mention_author=False)
            return
        
        # Format rules as text
        rules_text = "**Game Rules:**\n"
        if isinstance(rules, dict):
            for key, value in rules.items():
                rules_text += f"**{key}**: {value}\n"
        else:
            rules_text += str(rules)
        
        await ctx.reply(rules_text, mention_author=False)

    async def command_help(self, ctx: commands.Context) -> None:
        """Show available player commands."""
        game_state = self._get_game_state_for_context(ctx)
        if not game_state:
            await ctx.reply("No active game in this thread.", mention_author=False)
            return
        
        help_text = """**Player Commands:**
`!dice` or `!roll` - Roll dice
`!rules` - Show game rules
`!help` - Show this help

**GM Commands:**
`!startgame <game_type>` - Start a new game
`!endgame` - End the current game
`!listgames` - List available games
`!addplayer @user` - Add a player
`!removeplayer @user` - Remove a player
`!assign @user <character>` - Assign character
`!reroll @user` - Reroll a player's character
`!swap @user1 @user2` - Swap characters
`!movetoken @user <coord>` - Move token (e.g., A1)
`!bg @user <id>` or `!bg all <id>` - Set background
`!outfit @user <outfit>` - Set outfit
`!savegame` - Save game state
`!loadgame <file>` - Load game state"""
        
        await ctx.reply(help_text, mention_author=False)

    # Commands will be registered here
    def register_commands(self) -> None:
        """Register all game commands."""
        # TODO: Implement command registration
        pass

