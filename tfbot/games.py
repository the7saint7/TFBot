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
from .game_pack_loader import get_game_pack
from .utils import is_admin, int_from_env, path_from_env
from .models import TransformationState
from .swaps import ensure_form_owner

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
        
        # Packs directory for game-specific logic
        self.packs_dir = self.config_path.parent / "packs"
        
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
                    narrator_user_id=data.get("narrator_user_id"),
                )
                
                # Recreate player_states for players with assigned characters (like RP mode)
                # Get guild_id from forum channel if available
                guild_id = 0
                if game_state.forum_channel_id:
                    try:
                        forum_channel = self.bot.get_channel(game_state.forum_channel_id)
                        if forum_channel and hasattr(forum_channel, 'guild') and forum_channel.guild:
                            guild_id = forum_channel.guild.id
                    except Exception:
                        pass
                
                # Recreate states for all players with character assignments
                # NOTE: Cannot use await in sync function - states will be recreated on first message
                # This is acceptable as states are lazy-loaded when needed
                logger.debug("Skipping state recreation in _load_active_games (sync context) - will be created on first message")
                
                self._active_games[thread_id] = game_state
                logger.info("Loaded game state: thread_id=%s, game_type=%s, players=%d", thread_id, game_state.game_type, len(players))
            except (json.JSONDecodeError, ValueError, KeyError) as exc:
                logger.warning("Failed to load game state %s: %s", state_file, exc)

    def is_game_thread(self, channel: Optional[discord.abc.GuildChannel]) -> bool:
        """Check if a channel is a game thread (active or by name pattern)."""
        if not isinstance(channel, discord.Thread):
            return False
        
        thread_id = channel.id
        # Check if already in active games
        if thread_id in self._active_games:
            return True
        
        # Check if thread name matches game pattern
        if channel.name:
            for game_key, game_config in self._game_configs.items():
                game_prefix = f"{game_config.name} #"
                if channel.name.startswith(game_prefix):
                    return True
        
        return False
    
    async def _detect_and_load_game_thread(self, thread: discord.Thread) -> Optional[GameState]:
        """Detect if a thread is a game thread and load/create its state."""
        if not isinstance(thread, discord.Thread):
            return None
        
        # Check if already loaded
        if thread.id in self._active_games:
            return self._active_games[thread.id]
        
        # Check if thread name matches game pattern
        if not thread.name:
            return None
        
        matched_game_type = None
        for game_key, game_config in self._game_configs.items():
            game_prefix = f"{game_config.name} #"
            if thread.name.startswith(game_prefix):
                matched_game_type = game_key
                break
        
        if not matched_game_type:
            return None
        
        # Try to load from disk first
        state_file = self.states_dir / f"{thread.id}.json"
        if state_file.exists():
            try:
                data = json.loads(state_file.read_text(encoding="utf-8"))
                # Verify it matches this thread
                if int(data.get("game_thread_id", 0)) == thread.id:
                    # Load the state (reuse existing load logic)
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
                        except (ValueError, KeyError):
                            continue
                    
                    narrator_user_id = data.get("narrator_user_id")
                    if narrator_user_id is not None:
                        narrator_user_id = int(narrator_user_id)
                    
                    game_state = GameState(
                        game_thread_id=thread.id,
                        forum_channel_id=int(data.get("forum_channel_id", thread.parent_id if isinstance(thread.parent, discord.ForumChannel) else 0)),
                        dm_channel_id=int(data.get("dm_channel_id", 0)),
                        gm_user_id=int(data.get("gm_user_id", 0)),
                        game_type=str(data.get("game_type", matched_game_type)),
                        map_thread_id=data.get("map_thread_id"),
                        players=players,
                        current_turn=data.get("current_turn"),
                        board_message_id=data.get("board_message_id"),
                        is_locked=bool(data.get("is_locked", False)),
                        narrator_user_id=narrator_user_id,
                        player_states={},  # Will be recreated below
                    )
                    
                    # Recreate player_states for players with assigned characters (like RP mode)
                    guild_id = 0
                    if thread.guild:
                        guild_id = thread.guild.id
                    elif game_state.forum_channel_id:
                        try:
                            forum_channel = self.bot.get_channel(game_state.forum_channel_id)
                            if forum_channel and hasattr(forum_channel, 'guild') and forum_channel.guild:
                                guild_id = forum_channel.guild.id
                        except Exception:
                            pass
                    
                    # Recreate states for all players with character assignments
                    if guild_id > 0:
                        for user_id, player in players.items():
                            if player.character_name:
                                state = await self._create_game_state_for_player(
                                    player,
                                    user_id,
                                    guild_id,
                                    player.character_name,
                                )
                                if state:
                                    game_state.player_states[user_id] = state
                                    logger.debug("Restored player state for user %s as %s", user_id, player.character_name)
                    
                    self._active_games[thread.id] = game_state
                    logger.info("Loaded existing game state from disk: thread_id=%s, game_type=%s, players=%d", thread.id, game_state.game_type, len(players))
                    return game_state
            except (json.JSONDecodeError, ValueError, KeyError) as exc:
                logger.warning("Failed to load game state from disk for thread %s: %s", thread.id, exc)
        
        # No saved state found - create a new one (thread exists but game wasn't properly initialized)
        # Try to determine GM from thread owner or first message
        gm_user_id = 0
        try:
            # Try to get thread owner (forum thread starter)
            if hasattr(thread, 'owner_id') and thread.owner_id:
                gm_user_id = thread.owner_id
            else:
                # Fallback: try to get from first message
                async for message in thread.history(limit=1, oldest_first=True):
                    if message.author and not message.author.bot:
                        gm_user_id = message.author.id
                        break
        except discord.HTTPException:
            pass
        
        if gm_user_id == 0:
            logger.warning("Could not determine GM for thread %s, cannot auto-create game state", thread.id)
            return None
        
        # Create new game state for existing thread
        game_state = GameState(
            game_thread_id=thread.id,
            forum_channel_id=thread.parent_id if isinstance(thread.parent, discord.ForumChannel) else 0,
            dm_channel_id=self.dm_channel_id,
            gm_user_id=gm_user_id,
            game_type=matched_game_type,
            narrator_user_id=gm_user_id,  # GM becomes narrator by default
        )
        
        self._active_games[thread.id] = game_state
        await self._save_game_state(game_state)
        logger.info("Created new game state for existing thread: thread_id=%s, game_type=%s, gm=%s", thread.id, matched_game_type, gm_user_id)
        return game_state

    def _get_character_by_name(self, character_name: str):
        """
        Get character by name using the SAME function that !reroll uses.
        Uses _find_character_by_token from bot.py - no reimplementation!
        """
        try:
            # Use the EXACT same function that !reroll uses
            import sys
            bot_module = None
            
            # Try 'bot' first (when imported)
            bot_module = sys.modules.get('bot')
            
            # If that fails, try '__main__' (when run directly as python bot.py)
            if not bot_module:
                bot_module = sys.modules.get('__main__')
            
            # If still not found, try to import it
            if not bot_module:
                try:
                    import bot as bot_module
                except ImportError:
                    pass
            
            if not bot_module:
                logger.warning("Bot module not found in sys.modules (tried 'bot' and '__main__')")
                return None
            
            # Use the EXACT same function that !reroll uses
            _find_character_by_token = getattr(bot_module, '_find_character_by_token', None)
            if not _find_character_by_token:
                logger.warning("_find_character_by_token not found in bot module")
                return None
            
            # Call it exactly like !reroll does
            result = _find_character_by_token(character_name)
            if result:
                logger.info("Found character for '%s': %s", character_name, result.name)
            else:
                logger.debug("No character found for '%s'", character_name)
            return result
            
        except Exception as exc:
            logger.warning("Failed to lookup character %s: %s", character_name, exc, exc_info=True)
        return None

    async def _create_game_state_for_player(
        self,
        player: GamePlayer,
        user_id: int,
        guild_id: int,
        character_name: str,
    ) -> Optional[TransformationState]:
        """
        Create a TransformationState using the EXACT same function VN roll uses.
        
        CRITICAL: This state is ONLY stored in game_state.player_states.
        It is NEVER added to global active_transformations.
        This allows player to be one character in game, another in VN simultaneously!
        """
        character = self._get_character_by_name(character_name)
        if not character:
            logger.warning("Character not found for game player: %s", character_name)
            return None
        
        # Get member object (needed for _build_roleplay_state)
        import sys
        bot_module = sys.modules.get('bot') or sys.modules.get('__main__')
        if not bot_module:
            try:
                import bot as bot_module
            except ImportError:
                logger.warning("Cannot import bot module to get member")
                return None
        
        if not hasattr(bot_module, 'bot'):
            logger.warning("Bot instance not found in module")
            return None
        
        bot_instance = getattr(bot_module, 'bot')
        member = None
        guild = None
        if guild_id > 0:
            guild = bot_instance.get_guild(guild_id)
            if guild:
                # Try get_member first (cached)
                member = guild.get_member(user_id)
                # If not found, try fetching (might not be cached)
                if not member:
                    try:
                        member = await guild.fetch_member(user_id)
                        logger.debug("Fetched member %s from guild %s", user_id, guild_id)
                    except discord.HTTPException as exc:
                        logger.warning("Failed to fetch member %s from guild %s: %s", user_id, guild_id, exc)
        
        if not member:
            logger.error("Member %s not found in guild %s - cannot create state. Tried get_member and fetch_member.", user_id, guild_id)
            return None
        
        # Use the EXACT same function that VN roll uses!
        _build_roleplay_state = getattr(bot_module, '_build_roleplay_state', None)
        if not _build_roleplay_state:
            logger.error("_build_roleplay_state not found in bot module - gameboard assignment will fail!")
            return None
        
        # Call it EXACTLY like VN roll does
        try:
            state = _build_roleplay_state(character, member, guild)
            if not state:
                logger.error("_build_roleplay_state returned None for character %s (folder: %s), member %s (display: %s)", 
                           character.name, character.folder, member.id, member.display_name)
                return None
            logger.debug("_build_roleplay_state succeeded: character=%s, user_id=%s, guild_id=%s", 
                        character.name, user_id, guild_id)
        except Exception as exc:
            logger.error("Failed to call _build_roleplay_state for character %s, member %s: %s", 
                       character.name, member.id, exc, exc_info=True)
            return None
        
        # Modify duration for games (longer than roleplay)
        state.expires_at = state.started_at + timedelta(days=365)
        state.duration_label = "Game"
        
        # Verify state was created correctly
        if state.character_name != character.name:
            logger.error("State character_name mismatch! Expected '%s', got '%s'", character.name, state.character_name)
            state.character_name = character.name  # Fix it
        
        # CRITICAL: Verify all required fields are present for VN rendering
        if not state.character_avatar_path:
            logger.error("State missing character_avatar_path! Character: %s", character.name)
            state.character_avatar_path = character.avatar_path or ""
        if not state.character_folder:
            logger.warning("State missing character_folder! Character: %s", character.name)
            state.character_folder = character.folder
        
        # Log state details for debugging
        logger.info("Created game TransformationState: user_id=%s, character_name='%s', character_folder='%s', avatar_path='%s'", 
                   user_id, state.character_name, state.character_folder, state.character_avatar_path)
        
        # CRITICAL: Do NOT add to active_transformations - game state is isolated
        # This allows player to be one character in game, another in VN simultaneously!
        logger.debug("Created game TransformationState for user %s as %s using _build_roleplay_state (NOT added to global active_transformations)", user_id, character_name)
        return state

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
            # Check if this is a map thread
            for state in self._active_games.values():
                if state.map_thread_id == thread_id:
                    game_state = state
                    break
            
            # Try to detect and load existing game thread
            if not game_state:
                if isinstance(message.channel, discord.Thread):
                    game_state = await self._detect_and_load_game_thread(message.channel)
                    if not game_state:
                        return False
                else:
                    return False
        
        # Handle map thread: auto-delete all messages except admin commands
        if game_state.map_thread_id and thread_id == game_state.map_thread_id:
            is_admin_user = is_admin(message.author)
            # Allow admin commands, delete everything else
            if not is_admin_user or not (message.content and message.content.strip().startswith('!')):
                try:
                    await message.delete()
                except discord.HTTPException:
                    pass
                return True
            # Admin command in map thread - let it through
            return False
        
        if game_state.is_locked:
            return False
        
        # Check if author is GM or admin
        is_gm = self._is_gm(message.author, game_state)
        is_admin_user = is_admin(message.author)
        is_narrator = game_state.narrator_user_id == message.author.id
        
        # GM and admins can always send messages (commands or narrator) - never block them
        if is_gm or is_admin_user:
            # If GM is narrator and not a player, handle as narrator
            player = game_state.players.get(message.author.id)
            if is_narrator and not (player and player.character_name):
                # Check if message looks like a command - if so, let it through
                # CRITICAL: If command doesn't exist, we still return False to let command handler process it
                # But we need to prevent it from falling through to VN mode
                if message.content and message.content.strip().startswith('!'):
                    # Command attempt - let command handler process it (even if command doesn't exist)
                    # This prevents narrator from reverting to VN character on typos
                    return False  # Let command handler process it (will show error if command doesn't exist)
                # Otherwise handle as narrator message
                await self._handle_narrator_message(message, game_state)
                return True
            # GM/admin but not narrator or has character - let through (for commands)
            return False
        
        # Check if player is in the game and has a character assigned
        player = game_state.players.get(message.author.id)
        has_character = player and player.character_name
        
        # Block messages from non-GM, non-admin players without assigned character
        if not has_character:
            try:
                await message.delete()
            except discord.HTTPException:
                pass
            return True
        
        # Player has character - proceed with normal VN panel rendering
        
        # Get character state from GAME STATE ONLY (completely isolated from global active_transformations)
        # This ensures game characters don't affect VN mode and vice versa
        state = game_state.player_states.get(message.author.id)
        if not state:
            # Fallback: create state if missing (still isolated - only in game_state.player_states)
            logger.debug("Creating missing game state for player %s as %s", message.author.id, player.character_name)
            state = await self._create_game_state_for_player(
                player,
                message.author.id,
                message.guild.id,
                player.character_name,
            )
            if state:
                game_state.player_states[message.author.id] = state
                logger.debug("Stored game state in game_state.player_states (NOT in global active_transformations)")
            else:
                return True
        
        # CRITICAL: Verify state.character_name matches player.character_name
        # If they don't match, the state is stale and needs to be recreated
        if state.character_name != player.character_name:
            logger.warning("State character_name mismatch for player %s! State has '%s', player has '%s'. Recreating state.",
                        message.author.id, state.character_name, player.character_name)
            # Recreate state with correct character name
            state = await self._create_game_state_for_player(
                player,
                message.author.id,
                message.guild.id,
                player.character_name,
            )
            if state:
                game_state.player_states[message.author.id] = state
                logger.info("Recreated game state for player %s with correct character '%s'", 
                           message.author.id, state.character_name)
            else:
                logger.error("Failed to recreate state for player %s", message.author.id)
                return True
        
        # Verify state is from game, not global (safety check)
        logger.debug("Using game state for player %s: character=%s (guild_id=%s, isolated from VN mode)", 
                     message.author.id, state.character_name, state.guild_id)
        
        # Render VN panel
        try:
            from tfbot.panels import (
                render_vn_panel,
                parse_discord_formatting,
                prepare_custom_emoji_images,
                strip_urls,
            )
            from .models import ReplyContext
            
            # Get MESSAGE_STYLE via lazy import
            import sys
            bot_module = sys.modules.get('bot') or sys.modules.get('__main__')
            MESSAGE_STYLE = getattr(bot_module, 'MESSAGE_STYLE', 'classic') if bot_module else 'classic'
            
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
                    # Render with game-specific character (use game-assigned character, not VN mode)
                    # CRITICAL: Always use player.character_name (the source of truth) not state.character_name
                    # The state might be stale or incorrect, but player.character_name is always correct
                    character_display_name = player.character_name  # Use player's character name (source of truth)
                    
                    # CRITICAL: If state.character_name doesn't match, fix the state immediately
                    if state.character_name != player.character_name:
                        logger.error("CRITICAL MISMATCH before render: state.character_name='%s' != player.character_name='%s'. Fixing state...", 
                                    state.character_name, player.character_name)
                        # Fix the state immediately
                        state.character_name = player.character_name
                        # Also update the stored state
                        game_state.player_states[message.author.id] = state
                        await self._save_game_state(game_state)
                        logger.info("Fixed state: now state.character_name='%s'", state.character_name)
                    
                    logger.info("Rendering VN panel for game player: user_id=%s, player.character_name='%s', state.character_name='%s', character_display_name='%s'", 
                               message.author.id, player.character_name, state.character_name, character_display_name)
                    
                    # CRITICAL: Verify state has all required fields before rendering
                    if not state.character_name:
                        logger.error("State missing character_name! Cannot render VN panel.")
                    elif not state.character_avatar_path:
                        logger.warning("State missing character_avatar_path for %s", state.character_name)
                    
                    vn_file = render_vn_panel(
                        state=state,
                        message_content=cleaned_content,
                        character_display_name=character_display_name,
                        original_name=message.author.display_name,
                        attachment_id=str(message.id),
                        formatted_segments=formatted_segments,
                        custom_emoji_images=custom_emoji_images,
                        reply_context=reply_context,
                        gacha_outfit_override=player.outfit_name if player.outfit_name else None,
                    )
                    if vn_file:
                        logger.info("VN panel rendered successfully for %s", character_display_name)
                        files.append(vn_file)
                    else:
                        logger.warning("render_vn_panel returned None for %s (character_name='%s', state.character_name='%s')", 
                                     character_display_name, player.character_name, state.character_name)
                finally:
                    # Restore original function (critical for isolation)
                    panels_module.get_selected_background_path = original_func
            
            # If we have files to send, send them and delete original
            if files:
                logger.info("Sending VN panel file for game player %s as %s", message.author.id, character_display_name)
                send_kwargs: Dict[str, object] = {
                    "files": files,
                    "allowed_mentions": discord.AllowedMentions.none(),
                }
                
                # Preserve reply reference if present
                if message.reference:
                    send_kwargs["reference"] = message.reference
                
                try:
                    await message.channel.send(**send_kwargs)
                    logger.info("Successfully sent VN panel for game player %s", message.author.id)
                    # Delete original message
                    try:
                        await message.delete()
                    except discord.HTTPException:
                        pass  # Message might already be deleted
                except discord.HTTPException as exc:
                    logger.error("Failed to send game VN panel: %s", exc, exc_info=True)
            else:
                logger.warning("No VN panel file created for game player %s as %s (MESSAGE_STYLE=%s, files=%s)", 
                             message.author.id, character_display_name, MESSAGE_STYLE, len(files) if 'files' in locals() else 0)
            
        except Exception as exc:
            logger.exception("Error rendering game VN panel: %s", exc)
        
        return True
    
    async def _handle_narrator_message(self, message: discord.Message, game_state: GameState) -> None:
        """Handle narrator message (GM speaking as narrator)."""
        # Use the slash say command handler directly
        import sys
        bot_module = sys.modules.get('bot') or sys.modules.get('__main__')
        if not bot_module:
            logger.warning("Cannot get bot module for narrator message")
            return
        
        _handle_slash_say = getattr(bot_module, '_handle_slash_say', None)
        if not _handle_slash_say:
            logger.warning("_handle_slash_say not found for narrator message")
            return
        
        # Create a fake interaction for the say command
        # We'll use the message directly instead
        try:
            # Get narrator character
            CHARACTER_BY_NAME = getattr(bot_module, 'CHARACTER_BY_NAME', None)
            if not CHARACTER_BY_NAME:
                logger.warning("CHARACTER_BY_NAME not found")
                return
            
            narrator_char = CHARACTER_BY_NAME.get("narrator")
            if not narrator_char:
                # Try case-insensitive
                for name, char in CHARACTER_BY_NAME.items():
                    if name.lower() == "narrator":
                        narrator_char = char
                        break
            
            if not narrator_char:
                logger.warning("Narrator character not found")
                return
            
            # Create a TransformationState for narrator
            from tfbot.models import TransformationState
            from tfbot.utils import member_profile_name, utc_now
            from datetime import timedelta
            
            now = utc_now()
            narrator_state = TransformationState(
                user_id=message.author.id,
                guild_id=message.guild.id if message.guild else 0,
                character_name=narrator_char.name,
                character_folder=narrator_char.folder,
                character_avatar_path=narrator_char.avatar_path,
                character_message=narrator_char.message or "",
                original_nick=message.author.nick,
                started_at=now,
                expires_at=now + timedelta(hours=1),
                duration_label="narrator",
                avatar_applied=False,
                original_display_name=member_profile_name(message.author),
                is_inanimate=False,
                inanimate_responses=tuple(),
            )
            
            # Render and send VN panel
            from tfbot.panels import render_vn_panel, parse_discord_formatting, prepare_custom_emoji_images
            
            cleaned_content = message.content.strip()
            formatted_segments = parse_discord_formatting(cleaned_content)
            custom_emoji_images = await prepare_custom_emoji_images(message, formatted_segments)
            
            vn_file = render_vn_panel(
                state=narrator_state,
                message_content=cleaned_content,
                character_display_name=narrator_char.name,
                original_name=message.author.display_name,
                attachment_id=str(message.id),
                formatted_segments=formatted_segments,
                custom_emoji_images=custom_emoji_images,
                reply_context=None,
            )
            
            if vn_file:
                await message.channel.send(files=[vn_file], allowed_mentions=discord.AllowedMentions.none())
            else:
                # Fallback to text
                await message.channel.send(f"**{narrator_char.name}**: {cleaned_content}", allowed_mentions=discord.AllowedMentions.none())
            
            # Delete original message
            try:
                await message.delete()
            except discord.HTTPException:
                pass
        except Exception as exc:
            logger.exception("Error handling narrator message: %s", exc)

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

    async def _get_game_state_for_context(self, ctx: commands.Context) -> Optional[GameState]:
        """Get game state for a command context (thread or DM channel)."""
        if isinstance(ctx.channel, discord.Thread):
            thread_id = ctx.channel.id
            # Check if already loaded
            if thread_id in self._active_games:
                return self._active_games[thread_id]
            
            # Try to detect and load existing game thread
            game_state = await self._detect_and_load_game_thread(ctx.channel)
            return game_state
        
        # Could also check DM channel
        return None

    async def _save_game_state(self, game_state: GameState) -> None:
        """Save game state to disk."""
        async with self._lock:
            # Generate filename with game type, date, and time for better organization
            from datetime import datetime
            now = datetime.now()
            date_str = now.strftime("%d-%m-%Y")
            time_str = now.strftime("%H:%M")
            game_type = game_state.game_type or "unknown"
            # Sanitize game_type for filename (remove invalid chars)
            safe_game_type = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in game_type)
            filename = f"{safe_game_type}-{date_str}-{time_str}.json"
            state_file = self.states_dir / filename
            
            data = {
                "game_thread_id": game_state.game_thread_id,
                "forum_channel_id": game_state.forum_channel_id,
                "dm_channel_id": game_state.dm_channel_id,
                "gm_user_id": game_state.gm_user_id,
                "game_type": game_state.game_type,
                "map_thread_id": game_state.map_thread_id,
                "current_turn": game_state.current_turn,
                "board_message_id": game_state.board_message_id,
                "is_locked": game_state.is_locked,
                "narrator_user_id": game_state.narrator_user_id,
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

    async def _update_board(
        self,
        game_state: GameState,
        error_channel: Optional[discord.abc.Messageable] = None,
        target_thread: str = "map"
    ) -> None:
        """
        Update board image.
        
        Args:
            game_state: The game state
            error_channel: Channel to send error messages to
            target_thread: "game" to send to game thread, "map" to send to map thread (default)
        """
        logger.info("Updating board for game thread %s (map thread %s), target=%s", 
                   game_state.game_thread_id, game_state.map_thread_id, target_thread)
        game_config = self.get_game_config(game_state.game_type)
        if not game_config:
            error_msg = f"❌ No game config found for game type: {game_state.game_type}"
            logger.warning(error_msg)
            if error_channel:
                try:
                    await error_channel.send(error_msg)
                except Exception:
                    pass
            return
        
        # Determine target thread
        if target_thread == "game":
            target_thread_id = game_state.game_thread_id
        else:
            # Default to map thread
            target_thread_id = game_state.map_thread_id if game_state.map_thread_id else game_state.game_thread_id
        
        thread = self.bot.get_channel(target_thread_id)
        if not isinstance(thread, discord.Thread):
            error_msg = f"❌ Thread not found: {target_thread_id}"
            logger.warning(error_msg)
            if error_channel:
                try:
                    await error_channel.send(error_msg)
                except Exception:
                    pass
            return
        
        # Log player positions for debugging
        logger.info("Players on board: %s", {pid: (p.grid_position, p.character_name) for pid, p in game_state.players.items()})
        
        # Regenerate board image with current token positions (this creates a NEW image)
        board_file = render_game_board(game_state, game_config, self.assets_dir)
        if not board_file:
            error_msg = "❌ Failed to render board image"
            logger.warning(error_msg)
            if error_channel:
                try:
                    await error_channel.send(error_msg)
                except Exception:
                    pass
            return
        
        logger.info("Board image regenerated, posting to map thread")
        
        # Post new board image (don't delete old ones - they persist for history)
        try:
            # Send the board image file directly (no embed, no URL - just the image)
            # This is a NEW image with updated token positions
            # Old images remain visible for history
            # Use allowed_mentions to prevent pings
            board_msg = await thread.send(
                file=board_file,
                allowed_mentions=discord.AllowedMentions.none()
            )
            game_state.board_message_id = board_msg.id  # Store latest for reference
            await self._save_game_state(game_state)
            logger.info("Board updated successfully, new message ID: %s", board_msg.id)
        except discord.HTTPException as exc:
            error_msg = f"❌ Failed to post board image: {exc}"
            logger.warning(error_msg)
            if error_channel:
                try:
                    await error_channel.send(error_msg)
                except Exception:
                    pass

    async def _log_action(self, game_state: GameState, action: str) -> None:
        """Log a game action."""
        # TODO: Implement logging to file or channel
        logger.info("Game action [thread %s]: %s", game_state.game_thread_id, action)

    # GM Command Methods
    async def command_startgame(self, ctx: commands.Context, game_type: str = "") -> None:
        """Start a new game or resume an existing game. Only admins can start games, and they become the GM."""
        if not isinstance(ctx.author, discord.Member):
            await ctx.reply("This command can only be used inside a server.", mention_author=False)
            return
        
        # Only admins can start new games (they become the GM)
        if not is_admin(ctx.author):
            await ctx.reply("Only admins can start new games. The admin who starts the game becomes the GM.", mention_author=False)
            return
        
        if not game_type:
            await ctx.reply("Usage: `!startgame <game_type>`\nAvailable games: " + ", ".join(self.list_available_games()), mention_author=False)
            return
        
        game_config = self.get_game_config(game_type)
        if not game_config:
            await ctx.reply(f"Unknown game type: `{game_type}`. Available: " + ", ".join(self.list_available_games()), mention_author=False)
            return
        
        # Check if we're in an existing game thread - if so, just resume it
        if isinstance(ctx.channel, discord.Thread):
            existing_state = await self._detect_and_load_game_thread(ctx.channel)
            if existing_state:
                await ctx.reply(f"✅ Game already exists in this thread! Resuming game. Use `!addplayer` and `!assign` to continue.", mention_author=False)
                # Update board if needed
                if not existing_state.board_message_id:
                    await self._update_board(existing_state, error_channel=ctx.channel)
                return
        
        # Check if forum channel exists
        forum_channel = self.bot.get_channel(self.forum_channel_id)
        if not isinstance(forum_channel, discord.ForumChannel):
            await ctx.reply("Forum channel not configured or not found.", mention_author=False)
            return
        
        # Generate unique game number by checking existing threads
        game_number = 1
        game_prefix = f"{game_config.name} #"
        try:
            # Fetch recent threads to find highest number
            async for thread in forum_channel.archived_threads(limit=100):
                if thread.name.startswith(game_prefix):
                    try:
                        # Extract number from thread name like "Snakes and Ladders #123 - Username"
                        parts = thread.name.split("#", 1)
                        if len(parts) > 1:
                            num_part = parts[1].split(" - ", 1)[0]
                            thread_num = int(num_part)
                            if thread_num >= game_number:
                                game_number = thread_num + 1
                    except (ValueError, IndexError):
                        pass
            
            # Also check active threads
            for thread in forum_channel.threads:
                if thread.name.startswith(game_prefix):
                    try:
                        parts = thread.name.split("#", 1)
                        if len(parts) > 1:
                            num_part = parts[1].split(" - ", 1)[0]
                            thread_num = int(num_part)
                            if thread_num >= game_number:
                                game_number = thread_num + 1
                    except (ValueError, IndexError):
                        pass
        except Exception as exc:
            logger.warning("Failed to check existing threads for game number: %s", exc)
            # Fallback to timestamp-based number
            import time
            game_number = int(time.time()) % 100000
        
        # Create TWO separate forum posts in the same channel: one for chat, one for board images
        thread_name = f"{game_config.name} #{game_number} - {ctx.author.display_name}"
        map_thread_name = f"{game_config.name} #{game_number} Map - {ctx.author.display_name}"
        initial_message = f"🎲 **{game_config.name}** game started by {ctx.author.mention}\n\nUse `!addplayer @user` to add players, then `!assign @user character_name` to assign characters."
        map_initial_message = f"🗺️ **{game_config.name} Map** - Board updates will appear here.\n\nThis post is read-only. All messages except admin commands will be automatically deleted."
        
        try:
            # Create game thread (for chat/commands)
            thread, message = await forum_channel.create_thread(
                name=thread_name,
                auto_archive_duration=1440,
                content=initial_message
            )
            
            # Create map thread (for board images only, same channel, different post)
            map_thread, map_message = await forum_channel.create_thread(
                name=map_thread_name,
                auto_archive_duration=1440,
                content=map_initial_message
            )
        except discord.HTTPException as exc:
            await ctx.reply(f"Failed to create game thread: {exc}", mention_author=False)
            return
        
        # Create game state with both threads (same channel, different posts)
        game_state = GameState(
            game_thread_id=thread.id,
            forum_channel_id=forum_channel.id,
            dm_channel_id=self.dm_channel_id,
            gm_user_id=ctx.author.id,
            game_type=game_type,
            map_thread_id=map_thread.id,
            narrator_user_id=ctx.author.id,  # GM becomes narrator by default
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
        
        game_state = await self._get_game_state_for_context(ctx)
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
        
        player = GamePlayer(user_id=member.id, grid_position="A1")
        game_state.players[member.id] = player
        
        # Call pack's on_player_added if it exists
        game_config = self.get_game_config(game_state.game_type)
        pack = get_game_pack(game_state.game_type, self.packs_dir)
        if pack and pack.has_function("on_player_added") and game_config:
            pack.call("on_player_added", game_state, player, game_config)
        
        await self._save_game_state(game_state)
        
        # Update board to show the new player token at starting position
        await self._update_board(game_state, error_channel=ctx.channel)
        
        await ctx.reply(f"Added {member.mention} to the game.", mention_author=False)
        await self._log_action(game_state, f"Player {member.display_name} added")

    async def command_assign(self, ctx: commands.Context, member: Optional[discord.Member] = None, *, character_name: str = "") -> None:
        """Assign a character to a player (GM only)."""
        try:
            logger.info("command_assign called by %s, member=%s, character_name=%s", ctx.author.id, member, character_name)
            
            if not isinstance(ctx.author, discord.Member):
                await ctx.reply("This command can only be used inside a server.", mention_author=False)
                return
            
            game_state = await self._get_game_state_for_context(ctx)
            if not game_state:
                logger.warning("No game state found for context in thread %s", ctx.channel.id if isinstance(ctx.channel, discord.Thread) else None)
                await ctx.reply("No active game in this thread.", mention_author=False)
                return
            
            logger.debug("Game state found: game_type=%s, gm_user_id=%s", game_state.game_type, game_state.gm_user_id)
            
            if not self._is_gm(ctx.author, game_state):
                logger.warning("User %s is not GM (GM is %s)", ctx.author.id, game_state.gm_user_id)
                await ctx.reply("Only the GM can assign characters.", mention_author=False)
                return
            
            if not member:
                await ctx.reply("Usage: `!assign @user <character>`", mention_author=False)
                return
            
            if not character_name or not character_name.strip():
                await ctx.reply("Usage: `!assign @user <character>`", mention_author=False)
                return
            
            character_name = character_name.strip()
            logger.debug("Assigning character %s to member %s", character_name, member.id)
            
            if member.id not in game_state.players:
                await ctx.reply(f"{member.mention} is not in the game. Add them first with `!addplayer`.", mention_author=False)
                return
            
            player = game_state.players[member.id]
            
            # Verify character exists before assigning (uses first name matching like !reroll)
            character = self._get_character_by_name(character_name)
            if not character:
                # Character not found - show error (no suggestions, just error)
                await ctx.reply(f"❌ Unable to locate '{character_name}'.", mention_author=False)
                logger.warning("Character assignment failed: '%s' not found", character_name)
                return
            
            # Character found - assign it using the character's actual name (for consistency)
            # This ensures player.character_name matches what's stored in the character object
            actual_character_name = character.name
            player.character_name = actual_character_name
            
            # Create TransformationState for the player to enable VN mode
            # CRITICAL: Always create a new state with the correct character name
            # This ensures state.character_name matches the game-assigned character
            if not ctx.guild:
                await ctx.reply("❌ Error: Cannot assign character outside of a server.", mention_author=False)
                return
            
            state = await self._create_game_state_for_player(
                player,
                member.id,
                ctx.guild.id,
                actual_character_name,  # Use the character's actual name, not the lookup parameter
            )
            if not state:
                # This should never happen if character lookup worked, but handle it anyway
                logger.error("CRITICAL: Character found but state creation failed for %s with character %s", member.id, character_name)
                await ctx.reply(f"❌ Error: Failed to create state for character '{character_name}'. Assignment not completed.", mention_author=False)
                return
            
            # CRITICAL: Force state to use the correct character name BEFORE storing
            # This ensures state.character_name ALWAYS matches player.character_name
            if state.character_name != actual_character_name:
                logger.warning("State character_name mismatch during assignment! Expected '%s', got '%s'. Fixing...", 
                            actual_character_name, state.character_name)
                # Force correct character name
                state.character_name = actual_character_name
                # Also update character_folder and avatar_path to match
                if character:
                    state.character_folder = character.folder
                    state.character_avatar_path = character.avatar_path
                    state.character_message = character.message or ""
            
            # Store the new state (this replaces any old state)
            game_state.player_states[member.id] = state
            logger.info("Assigned character '%s' (lookup: '%s') to player %s (user_id=%s). State stored with character_name='%s'", 
                       actual_character_name, character_name, member.display_name, member.id, state.character_name)
            
            # Final verification - state MUST match player
            if state.character_name != actual_character_name or player.character_name != actual_character_name:
                logger.error("CRITICAL: Final verification failed! player.character_name='%s', state.character_name='%s', actual_character_name='%s'", 
                            player.character_name, state.character_name, actual_character_name)
                # Force everything to match
                player.character_name = actual_character_name
                state.character_name = actual_character_name
                game_state.player_states[member.id] = state
            
            # If GM was assigned as a player, remove narrator role
            if member.id == game_state.gm_user_id and game_state.narrator_user_id == member.id:
                game_state.narrator_user_id = None
                logger.debug("GM assigned as player, removed narrator role")
            
            # Call pack's on_character_assigned if it exists
            pack = get_game_pack(game_state.game_type, self.packs_dir)
            if pack and pack.has_function("on_character_assigned"):
                pack.call("on_character_assigned", game_state, player, character_name)
            
            await self._save_game_state(game_state)
            
            # Update board to show the new token
            await self._update_board(game_state, error_channel=ctx.channel)
            
            # CRITICAL: Send transformation message as TEXT (from bot, NOT from narrator VN panel)
            # This is the "roll" announcement - player becomes the character!
            # The player can then type messages which will show as the character
            # Bot text messages should NOT be deleted!
            
            # Send transformation message as TEXT (like VN roll does) - from bot, not character, not narrator
            try:
                from tfbot.utils import member_profile_name
                
                # Get helper functions via lazy import
                import sys
                bot_module = sys.modules.get('bot') or sys.modules.get('__main__')
                _format_character_message = getattr(bot_module, '_format_character_message', None)
                _get_magic_emoji = getattr(bot_module, '_get_magic_emoji', None)
                _format_special_reroll_hint = getattr(bot_module, '_format_special_reroll_hint', None)
                
                # Format transformation message EXACTLY like VN roll does
                original_name = member_profile_name(member)
                character_message = character.message or ""
                duration_label = "Game"
                
                if _format_character_message:
                    response_text = _format_character_message(
                        character_message,
                        original_name,
                        member.mention,
                        duration_label,
                        actual_character_name,
                    )
                else:
                    # Fallback if function not available
                    response_text = f"{original_name} becomes **{actual_character_name}**!"
                
                # Add special hint if available (like VN roll does)
                if _format_special_reroll_hint:
                    special_hint = _format_special_reroll_hint(actual_character_name, character.folder if character else None)
                    if special_hint:
                        response_text = f"{response_text}\n{special_hint}"
                
                # Add emoji prefix (like VN roll does)
                if _get_magic_emoji and ctx.guild:
                    emoji_prefix = _get_magic_emoji(ctx.guild)
                    response_text = f"{emoji_prefix} {response_text}"
                
                # Send as TEXT message (from bot) - this is the transformation announcement
                # CRITICAL: This message should NOT be deleted!
                await ctx.reply(response_text, mention_author=False)
                logger.info("Assignment transformation message sent as TEXT for %s as %s", member.display_name, actual_character_name)
                
            except Exception as msg_exc:
                logger.exception("Error sending assignment transformation message: %s", msg_exc)
                # Fallback to simple text message
                await ctx.reply(f"✅ {member.display_name} becomes **{actual_character_name}**!", mention_author=False)
            
            await self._log_action(game_state, f"{member.display_name} assigned character: {character_name}")
            logger.info("Successfully assigned character %s to %s", character_name, member.id)
        except Exception as exc:
            logger.exception("Error in command_assign: %s", exc)
            # Get a more useful error message
            error_type = type(exc).__name__
            error_msg = str(exc)
            # If error message is just a number (like a user ID), provide more context
            if error_msg.isdigit() or (error_msg and len(error_msg) > 10 and error_msg.replace('-', '').isdigit()):
                error_msg = f"{error_type}: {error_msg}"
            if not error_msg or error_msg == "None":
                error_msg = f"{error_type} occurred"
            await ctx.reply(f"❌ Error assigning character: {error_msg}", mention_author=False)

    async def command_reroll(self, ctx: commands.Context, member: Optional[discord.Member] = None) -> None:
        """Randomly reroll a player's character (GM only)."""
        if not isinstance(ctx.author, discord.Member):
            await ctx.reply("This command can only be used inside a server.", mention_author=False)
            return
        
        game_state = await self._get_game_state_for_context(ctx)
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
        
        # Get list of available characters from CHARACTER_BY_NAME
        import sys
        bot_module = sys.modules.get('bot')
        if not bot_module:
            await ctx.reply("Failed to access character list.", mention_author=False)
            return
        
        CHARACTER_BY_NAME = getattr(bot_module, 'CHARACTER_BY_NAME', None)
        if not CHARACTER_BY_NAME:
            await ctx.reply("Character list not available.", mention_author=False)
            return
        
        # Get all available character names
        available_characters = list(CHARACTER_BY_NAME.keys())
        if not available_characters:
            await ctx.reply("No characters available for reroll.", mention_author=False)
            return
        
        # Randomly select a character
        old_character = game_state.players[member.id].character_name
        new_character = random.choice(available_characters)
        
        # CRITICAL: Only modify game state, never global state
        game_state.players[member.id].character_name = new_character
        await self._save_game_state(game_state)
        
        await ctx.reply(
            f"Rerolled {member.mention}'s character from {old_character or 'none'} to {new_character}.",
            mention_author=False
        )
        await self._log_action(game_state, f"{member.display_name} rerolled from {old_character} to {new_character}")

    async def command_swap(self, ctx: commands.Context, member1: Optional[discord.Member] = None, member2: Optional[discord.Member] = None) -> None:
        """Swap characters between two players (GM only)."""
        if not isinstance(ctx.author, discord.Member):
            await ctx.reply("This command can only be used inside a server.", mention_author=False)
            return
        
        game_state = await self._get_game_state_for_context(ctx)
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
        
        game_state = await self._get_game_state_for_context(ctx)
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
        await self._update_board(game_state, error_channel=ctx.channel)
        await ctx.reply(f"Moved {member.mention}'s token from {old_pos} to {position}.", mention_author=False)
        await self._log_action(game_state, f"{member.display_name} token moved to {position}")

    async def command_dice(self, ctx: commands.Context, target_player: Optional[discord.Member] = None) -> None:
        """
        Roll dice (player command).
        
        GM can use: !dice @playername to force a roll for that player (skips turn order).
        """
        game_state = await self._get_game_state_for_context(ctx)
        if not game_state:
            await ctx.reply("No active game in this thread.", mention_author=False)
            return
        
        if game_state.is_locked:
            await ctx.reply("Game is locked.", mention_author=False)
            return
        
        # Check if GM is forcing a roll for another player
        is_gm_override = False
        if target_player and self._is_gm(ctx.author, game_state):
            # GM override - roll for the target player
            if target_player.id not in game_state.players:
                await ctx.reply(f"{target_player.display_name} is not in this game.", mention_author=False)
                return
            player = game_state.players[target_player.id]
            is_gm_override = True
        else:
            # Normal player roll
            if ctx.author.id not in game_state.players:
                await ctx.reply("You're not in this game.", mention_author=False)
                return
            player = game_state.players[ctx.author.id]
        
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
        
        # Call pack's on_dice_rolled if it exists
        pack = get_game_pack(game_state.game_type, self.packs_dir)
        if pack and pack.has_function("on_dice_rolled"):
            pack_result = pack.call("on_dice_rolled", game_state, player, total, game_config)
            if pack_result:
                # Unpack new return signature: (message, should_auto_move, transformation_char, is_turn_complete)
                if len(pack_result) == 4:
                    pack_msg, should_auto_move, transformation_char, is_turn_complete = pack_result
                elif len(pack_result) == 2:
                    # Legacy format for backwards compatibility
                    pack_msg, should_auto_move = pack_result
                    transformation_char = None
                    is_turn_complete = False
                else:
                    pack_msg = None
                    should_auto_move = False
                    transformation_char = None
                    is_turn_complete = False
                
                if pack_msg:
                    result = f"{result}\n{pack_msg}"
                
                # Apply transformation if needed
                if transformation_char:
                    # Create transformation state for the player
                    state = await self._create_game_state_for_player(
                        player,
                        player.user_id,
                        ctx.guild.id if ctx.guild else 0,
                        transformation_char
                    )
                    if state:
                        game_state.player_states[player.user_id] = state
                        # Send VN panel for transformation
                        from tfbot.panels import render_vn_panel, parse_discord_formatting, prepare_custom_emoji_images
                        from tfbot.utils import member_profile_name
                        
                        # Format transformation message like VN roll
                        import sys
                        bot_module = sys.modules.get('bot') or sys.modules.get('__main__')
                        _format_character_message = getattr(bot_module, '_format_character_message', None)
                        _get_magic_emoji = getattr(bot_module, '_get_magic_emoji', None)
                        
                        if _format_character_message and state.character_name:
                            transform_msg = _format_character_message(
                                state.character_name,
                                player.character_name if hasattr(player, 'character_name') else None,
                                _get_magic_emoji() if _get_magic_emoji else ""
                            )
                        else:
                            transform_msg = f"Transformed to {transformation_char}!"
                        
                        formatted_segments = parse_discord_formatting(transform_msg)
                        custom_emoji_images = await prepare_custom_emoji_images(ctx.message, formatted_segments)
                        
                        vn_file = render_vn_panel(
                            state=state,
                            message_content=transform_msg,
                            character_display_name=state.character_name,
                            original_name=member_profile_name(ctx.guild.get_member(player.user_id)) if ctx.guild else f"User {player.user_id}",
                            attachment_id=str(ctx.message.id),
                            formatted_segments=formatted_segments,
                            custom_emoji_images=custom_emoji_images,
                            reply_context=None,
                        )
                        
                        if vn_file:
                            await ctx.channel.send(files=[vn_file], allowed_mentions=discord.AllowedMentions.none())
                        else:
                            # Fallback to text
                            await ctx.channel.send(transform_msg, allowed_mentions=discord.AllowedMentions.none())
                
                if should_auto_move:
                    await self._update_board(game_state, error_channel=ctx.channel, target_thread="map")
                    await self._save_game_state(game_state)
                
                # Handle turn completion
                if is_turn_complete:
                    # Get turn summary from pack
                    if pack and pack.has_function("get_turn_summary"):
                        summary_msg = pack.call("get_turn_summary", game_state, game_config)
                        if summary_msg:
                            # Send turn summary to game thread with board image
                            await self._update_board(game_state, error_channel=ctx.channel, target_thread="game")
                            await ctx.channel.send(summary_msg, allowed_mentions=discord.AllowedMentions.none())
                    
                    # Advance turn
                    if pack and pack.has_function("advance_turn"):
                        pack.call("advance_turn", game_state)
                    
                    await self._save_game_state(game_state)
        
        await ctx.reply(result, mention_author=False)
        await self._log_action(game_state, f"{ctx.author.display_name} rolled {result}")

    async def command_endgame(self, ctx: commands.Context) -> None:
        """End the current game and lock the thread (GM only)."""
        if not isinstance(ctx.author, discord.Member):
            await ctx.reply("This command can only be used inside a server.", mention_author=False)
            return
        
        game_state = await self._get_game_state_for_context(ctx)
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
        
        game_state = await self._get_game_state_for_context(ctx)
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
        
        game_state = await self._get_game_state_for_context(ctx)
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
        game_state = await self._get_game_state_for_context(ctx)
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
        
        game_state = await self._get_game_state_for_context(ctx)
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
        
        game_state = await self._get_game_state_for_context(ctx)
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
            # List available state files
            if not self.states_dir.exists():
                await ctx.reply("No saved game states found.", mention_author=False)
                return
            
            state_files = list(self.states_dir.glob("*.json"))
            if not state_files:
                await ctx.reply("No saved game states found.", mention_author=False)
                return
            
            lines = ["**Available game states:**"]
            for state_file_path in sorted(state_files):
                lines.append(f"- `{state_file_path.name}`")
            await ctx.reply("\n".join(lines) + "\n\nUsage: `!loadgame <state_file>`", mention_author=False)
            return
        
        # Load state file (sanitize filename to prevent directory traversal)
        state_file_clean = state_file.strip().replace("..", "").replace("/", "").replace("\\", "")
        state_file_path = self.states_dir / state_file_clean
        if not state_file_path.exists() or not state_file_path.suffix == ".json":
            await ctx.reply(f"State file not found: `{state_file}`", mention_author=False)
            return
        
        try:
            data = json.loads(state_file_path.read_text(encoding="utf-8"))
            thread_id = int(data.get("game_thread_id", 0))
            if thread_id <= 0:
                await ctx.reply("Invalid state file: missing thread ID.", mention_author=False)
                return
            
            # Verify thread still exists
            thread = self.bot.get_channel(thread_id)
            if not isinstance(thread, discord.Thread):
                await ctx.reply(f"Game thread {thread_id} no longer exists.", mention_author=False)
                return
            
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
                    logger.warning("Failed to load player data in state %s: %s", state_file, exc)
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
            
            # Replace active game state in memory
            self._active_games[thread_id] = game_state
            
            # Update board display
            await self._update_board(game_state, error_channel=ctx.channel)
            
            await ctx.reply(f"Loaded game state from `{state_file}`. Board updated.", mention_author=False)
            await self._log_action(game_state, f"Game state loaded from {state_file} by {ctx.author.display_name}")
            
        except json.JSONDecodeError as exc:
            await ctx.reply(f"Invalid JSON in state file: {exc}", mention_author=False)
        except (ValueError, KeyError) as exc:
            await ctx.reply(f"Error loading state file: {exc}", mention_author=False)

    async def command_rules(self, ctx: commands.Context) -> None:
        """Show game rules (player command) - simplified version under 2000 characters."""
        game_state = await self._get_game_state_for_context(ctx)
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
        
        # Build simplified rules (under 2000 chars)
        rules_text = "**Game Rules:**\n\n"
        
        # Objective
        win_tile = rules.get("win_tile", 100)
        rules_text += f"**Objective:** Reach tile {win_tile} to win!\n\n"
        
        # Basic gameplay
        rules_text += "**Gameplay:**\n"
        rules_text += "• Roll dice with `!dice` on your turn\n"
        rules_text += "• Move forward by the number rolled\n"
        rules_text += "• Land on snakes to slide down, ladders to climb up\n"
        rules_text += "• Transform when landing on colored tiles\n\n"
        
        # Snakes and ladders
        snakes = rules.get("snakes", {})
        ladders = rules.get("ladders", {})
        if snakes or ladders:
            rules_text += "**Snakes & Ladders:**\n"
            if snakes:
                rules_text += f"• {len(snakes)} snakes on the board (slide down)\n"
            if ladders:
                rules_text += f"• {len(ladders)} ladders on the board (climb up)\n"
            rules_text += "\n"
        
        # Tile effects (simplified)
        tile_colors = rules.get("tile_colors", {})
        if tile_colors:
            rules_text += "**Tile Effects:**\n"
            effect_descriptions = {
                "yellow": "🟡 Random transform",
                "blue": "🔵 Age change",
                "dark_blue": "🔵 Age change",
                "purple": "🟣 Revert to real body",
                "red": "🔴 Transform other (GM)",
                "green": "🟢 Body swap (GM)",
                "orange": "🟠 Gender swap",
                "pink": "🩷 Mind change"
            }
            shown_effects = set()
            for color, effect in tile_colors.items():
                if effect not in shown_effects:
                    desc = effect_descriptions.get(color, f"{effect}")
                    rules_text += f"• {desc}\n"
                    shown_effects.add(effect)
            rules_text += "\n"
        
        # Starting position
        starting_tile = rules.get("starting_tile", 1)
        starting_pos = rules.get("starting_position", "A1")
        rules_text += f"**Starting Position:** Tile {starting_tile} ({starting_pos})\n"
        
        # Ensure under 2000 characters
        if len(rules_text) > 1950:
            rules_text = rules_text[:1950] + "\n\n*(Rules truncated - ask GM for full details)*"
        
        await ctx.reply(rules_text, mention_author=False)

    async def command_transfergm(self, ctx: commands.Context, member: Optional[discord.Member] = None) -> None:
        """Transfer GM role to another user (current GM only)."""
        if not isinstance(ctx.author, discord.Member):
            await ctx.reply("This command can only be used inside a server.", mention_author=False)
            return
        
        game_state = await self._get_game_state_for_context(ctx)
        if not game_state:
            await ctx.reply("No active game in this thread.", mention_author=False)
            return
        
        if not self._is_gm(ctx.author, game_state):
            await ctx.reply("Only the current GM can transfer the GM role.", mention_author=False)
            return
        
        if not member:
            await ctx.reply("Usage: `!transfergm @user`", mention_author=False)
            return
        
        if member.id == game_state.gm_user_id:
            await ctx.reply("That user is already the GM.", mention_author=False)
            return
        
        # Transfer GM role
        old_gm_id = game_state.gm_user_id
        game_state.gm_user_id = member.id
        
        # If old GM was narrator and not a player, transfer narrator to new GM
        old_gm_player = game_state.players.get(old_gm_id)
        if game_state.narrator_user_id == old_gm_id and not (old_gm_player and old_gm_player.character_name):
            # Old GM was narrator and not a player - transfer narrator role
            new_gm_player = game_state.players.get(member.id)
            if not (new_gm_player and new_gm_player.character_name):
                # New GM is not a player - make them narrator
                game_state.narrator_user_id = member.id
            else:
                # New GM is a player - remove narrator role
                game_state.narrator_user_id = None
        
        await self._save_game_state(game_state)
        await ctx.reply(f"GM role transferred to {member.mention}.", mention_author=False)
        await self._log_action(game_state, f"GM role transferred from {ctx.author.display_name} to {member.display_name}")

    async def command_help(self, ctx: commands.Context) -> None:
        """Show available player commands."""
        game_state = await self._get_game_state_for_context(ctx)
        if not game_state:
            await ctx.reply("No active game in this thread.", mention_author=False)
            return
        
        help_text = """**Player Commands:**
`!dice` or `!roll` - Roll dice
`!rules` - Show game rules
`!help` - Show this help

**GM Commands:**
`!startgame <game_type>` - Start a new game (GM only)
`!transfergm @user` - Transfer GM role to another user
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
