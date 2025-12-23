"""Game board framework for managing board games with visual boards and VN chat."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import discord
from discord.ext import commands

from .game_models import GameConfig, GamePlayer, GameState
from .game_board import render_game_board, validate_coordinate, _resolve_face_cache_path
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
        env_map_forum_id = int_from_env("TFBOT_GAME_MAP_FORUM_CHANNEL_ID", 0)
        self.map_forum_channel_id = env_map_forum_id if env_map_forum_id > 0 else int(self._config.get("map_forum_channel_id", 0))
        # If map forum not configured, fall back to game forum (backwards compatibility)
        if self.map_forum_channel_id == 0:
            self.map_forum_channel_id = self.forum_channel_id
        env_dm_id = int_from_env("TFBOT_GAME_DM_CHANNEL_ID", 0)
        self.dm_channel_id = env_dm_id if env_dm_id > 0 else int(self._config.get("dm_channel_id", 0))
        
        # Automatic game detection - scan configs directory
        configs_dir = self.config_path.parent / "configs"
        self._game_configs: Dict[str, GameConfig] = _scan_game_configs(configs_dir)
        
        # Packs directory for game-specific logic
        self.packs_dir = self.config_path.parent / "packs"
        
        # Check if any packs exist - if not, disable gameboard functionality
        self._has_packs = False
        if self.packs_dir.exists():
            pack_files = list(self.packs_dir.glob("*.py"))
            self._has_packs = len(pack_files) > 0
            if not self._has_packs:
                logger.warning("GameBoardManager: No game packs found in %s - gameboard will be disabled", self.packs_dir)
            else:
                logger.info("GameBoardManager: Found %d game pack(s) in %s", len(pack_files), self.packs_dir)
        else:
            logger.warning("GameBoardManager: Packs directory %s does not exist - gameboard will be disabled", self.packs_dir)
        
        # If no packs found, clear game configs (fallback redundancy)
        if not self._has_packs:
            logger.warning("GameBoardManager: No game packs available - clearing game configs and disabling gameboard")
            self._game_configs = {}
        
        # Active game state (one game per thread)
        self._active_games: Dict[int, GameState] = {}  # thread_id -> GameState
        self._lock = asyncio.Lock()
        self._command_locks: Dict[int, asyncio.Lock] = {}  # Per-game command locks (thread_id -> Lock)
        self._message_queues: Dict[int, List[Dict]] = {}  # Per-game message queues (thread_id -> List[message_data])
        
        # States directory
        self.states_dir = self.config_path.parent / "states"
        self.states_dir.mkdir(parents=True, exist_ok=True)
        
        # Load any active games from disk
        self._load_active_games()
        
        if self._has_packs:
            logger.info(
                "GameBoardManager initialized: forum_channel_id=%s, map_forum_channel_id=%s, detected %d games",
                self.forum_channel_id,
                self.map_forum_channel_id,
                len(self._game_configs),
            )
        else:
            logger.warning(
                "GameBoardManager initialized (DISABLED - no packs): forum_channel_id=%s, map_forum_channel_id=%s, games=0",
                self.forum_channel_id,
                self.map_forum_channel_id,
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
                    debug_mode=bool(data.get("debug_mode", False)),
                    turn_count=int(data.get("turn_count", 0)),
                    game_started=bool(data.get("game_started", False)),  # Default to False for old saves
                    is_paused=bool(data.get("is_paused", False)),  # Default to False for old saves
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
        # If no packs available, gameboard is disabled - no threads are game threads
        if not self._has_packs:
            return False
        
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
                    debug_mode=bool(data.get("debug_mode", False)),
                    game_started=bool(data.get("game_started", False)),  # Default to False for old saves
                    is_paused=bool(data.get("is_paused", False)),  # Default to False for old saves
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
            debug_mode=False,  # Default to off
            game_started=False,  # Game starts as not ready - GM must use !start to begin
            is_paused=False,  # Game starts unpaused
        )
        
        self._active_games[thread.id] = game_state
        # Note: Auto-save removed - use !savegame to save manually
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
        
        # Preserve original identity for VN panels (used during swaps)
        state.identity_display_name = state.identity_display_name or character.name
        
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

    async def handle_message(self, message: discord.Message, *, command_invoked: bool, is_queued: bool = False) -> bool:
        """Handle a message in a game thread. Returns True if handled.
        
        Args:
            message: The message to handle
            command_invoked: Whether this message invoked a command
            is_queued: Whether this is a queued message being reprocessed (skip lock check)
        """
        # If no packs available, gameboard is disabled - don't handle messages
        if not self._has_packs:
            return False
        
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
        
        # CRITICAL: Check if a command is currently processing (lock is held)
        # If so, delete message immediately and queue it for processing after command completes
        # This applies to ALL messages (including GM/admin) to ensure proper ordering
        # EXCEPTION: Skip this check if this is a queued message being reprocessed
        if not is_queued:
            command_lock = self._get_command_lock(thread_id)
            if command_lock.locked():
                # Command is processing - extract message data BEFORE deleting
                logger.debug("Command processing - deleting and queuing message from %s (GM/admin messages also queued)", message.author.id)
                
                # Download attachments before deleting message (attachments become inaccessible after deletion)
                attachment_data = []
                if message.attachments:
                    for attachment in message.attachments:
                        try:
                            attachment_bytes = await attachment.read()
                            attachment_data.append({
                                'filename': attachment.filename,
                                'bytes': attachment_bytes,
                                'content_type': attachment.content_type,
                            })
                            logger.debug("Downloaded attachment %s (%d bytes) for queuing", attachment.filename, len(attachment_bytes))
                        except Exception as exc:
                            logger.warning("Failed to download attachment %s for queuing: %s", attachment.filename, exc)
                
                # Extract all necessary message data before deletion
                message_data = {
                    'content': message.content,
                    'author': message.author,
                    'channel': message.channel,
                    'guild': message.guild,
                    'reference': message.reference,
                    'attachments': attachment_data,  # Store downloaded attachment data
                    'id': message.id,
                }
                
                # Delete message immediately
                try:
                    await message.delete()
                except discord.HTTPException:
                    pass  # Message might already be deleted
                
                # Queue message data for processing after command completes
                # CRITICAL: Queue ALL messages (including GM/admin) to ensure proper ordering
                if thread_id not in self._message_queues:
                    self._message_queues[thread_id] = []
                self._message_queues[thread_id].append(message_data)
                logger.debug("Queued message from %s (queue size: %d, attachments: %d)", 
                            message.author.id, len(self._message_queues[thread_id]), len(attachment_data))
                return True
        
        # Check if author is GM or admin
        is_gm = self._is_gm(message.author, game_state)
        is_admin_user = is_admin(message.author)
        is_narrator = game_state.narrator_user_id == message.author.id
        
        # GM and admins can always send messages (commands or narrator) - never block them
        if is_gm or is_admin_user:
            # Check if GM/admin is a player with a character assigned
            player = game_state.players.get(message.author.id)
            has_character = player and player.character_name
            
            # If GM/admin is a player with a character, they should get VN panel rendering like regular players
            if has_character:
                # Fall through to VN panel rendering below (don't return False)
                pass
            elif is_narrator and not has_character:
                # GM is narrator and not a player - handle as narrator
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
            else:
                # GM/admin but not a player and not narrator - let through (for commands only)
                return False
        else:
            # Not GM/admin - check if player is in the game and has a character assigned
            player = game_state.players.get(message.author.id)
            has_character = player and player.character_name
        
        # Block messages from non-GM, non-admin players without assigned character
        # This includes players not in the game (player is None) or players without character assigned
        if not has_character:
            logger.debug("Deleting message from unassigned player %s (player=%s, has_character=%s)", 
                        message.author.id, player is not None, has_character)
            try:
                await message.delete()
            except discord.HTTPException as exc:
                logger.warning("Failed to delete message from unassigned player %s: %s", message.author.id, exc)
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
            # CRITICAL: Delete the old state first to ensure clean recreation
            if message.author.id in game_state.player_states:
                del game_state.player_states[message.author.id]
            # Recreate state with correct character name
            state = await self._create_game_state_for_player(
                player,
                message.author.id,
                message.guild.id,
                player.character_name,
            )
            if state:
                game_state.player_states[message.author.id] = state
                logger.info("Recreated game state for player %s with correct character '%s' (was '%s')", 
                           message.author.id, state.character_name, player.character_name)
            else:
                logger.error("Failed to recreate state for player %s with character '%s'", message.author.id, player.character_name)
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
            
            # Clean message content - match VN bot exactly
            cleaned_content = message.content.strip() if message.content else ""
            cleaned_content, has_links = strip_urls(cleaned_content)
            cleaned_content = cleaned_content.strip()
            
            # Check if message has attachments (images) - these should be allowed even without text
            has_attachments = bool(message.attachments)
            
            if not cleaned_content and not has_attachments:
                # No content and no attachments - ignore
                return True
            
            # Match VN bot: description = cleaned_content if cleaned_content else "*no message content*"
            description = cleaned_content if cleaned_content else "*no message content*"
            
            # Parse formatting - use cleaned_content directly (like VN bot)
            formatted_segments = parse_discord_formatting(cleaned_content) if cleaned_content else None
            custom_emoji_images = await prepare_custom_emoji_images(message, formatted_segments) if formatted_segments else {}
            
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
            
            # Get character display name (used for logging and display)
            character_display_name = state.identity_display_name or player.character_name
            
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
                    # character_display_name is already defined above
                    
                    # CRITICAL: If state.character_name doesn't match, fix the state immediately
                    if state.character_name != player.character_name:
                        logger.error("CRITICAL MISMATCH before render: state.character_name='%s' != player.character_name='%s'. Fixing state...", 
                                    state.character_name, player.character_name)
                        # Get the character object to update all fields
                        character = self._get_character_by_name(player.character_name)
                        if character:
                            # Update ALL character-related fields, not just the name
                            state.character_name = player.character_name
                            state.character_folder = character.folder
                            state.character_avatar_path = character.avatar_path or ""
                            state.character_message = character.message or ""
                            logger.info("Updated state with character '%s' (folder='%s', avatar='%s')", 
                                       character.name, character.folder, character.avatar_path)
                        else:
                            # Fallback: just update the name if character lookup fails
                            logger.warning("Character lookup failed for '%s', only updating name", player.character_name)
                            state.character_name = player.character_name
                        # Also update the stored state
                        game_state.player_states[message.author.id] = state
                        # Note: Auto-save removed - use !savegame to save manually
                        logger.info("Fixed state: now state.character_name='%s', folder='%s', avatar='%s'", 
                                   state.character_name, state.character_folder, state.character_avatar_path)
                    
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
            
            # If we have files to send, send them and handle original message like VN bot
            if files:
                logger.info("Sending VN panel file for game player %s as %s", message.author.id, character_display_name)
                send_kwargs: Dict[str, object] = {
                    "files": files,
                    "allowed_mentions": discord.AllowedMentions.none(),
                }
                
                # Preserve reply reference if present
                if message.reference:
                    send_kwargs["reference"] = message.reference
                
                # Match VN bot: preserve_original = has_attachments or has_links
                preserve_original = has_attachments or has_links
                deleted = False
                
                try:
                    await message.channel.send(**send_kwargs)
                    logger.info("Successfully sent VN panel for game player %s", message.author.id)
                    
                    # Handle original message - match VN bot behavior exactly
                    if not preserve_original:
                        # No attachments and no links - delete original message
                        deleted = True
                        try:
                            await message.delete()
                        except discord.Forbidden:
                            deleted = False
                            logger.debug(
                                "Missing permission to delete message %s for game relay in channel %s",
                                message.id,
                                message.channel.id,
                            )
                        except discord.HTTPException as exc:
                            deleted = False
                            logger.warning("Failed to delete message %s: %s", message.id, exc)
                    
                    # After sending, if has_attachments and not has_links, edit to placeholder (match VN bot)
                    if has_attachments and not has_links:
                        placeholder = "\u200b"
                        if message.content != placeholder:
                            try:
                                await message.edit(content=placeholder, attachments=message.attachments, suppress=True)
                            except discord.HTTPException as exc:
                                logger.debug("Unable to clear attachment message %s: %s", message.id, exc)
                except discord.HTTPException as exc:
                    logger.error("Failed to send game VN panel: %s", exc, exc_info=True)
            elif has_attachments:
                # No VN panel but message has attachments - preserve original message (match VN bot)
                # VN bot doesn't delete messages with attachments unless they interrupt calculations
                logger.info("Preserving message with attachments (no VN panel) for game player %s", message.author.id)
                # Don't delete - attachments should remain in original message
            else:
                logger.warning("No VN panel file created for game player %s as %s (MESSAGE_STYLE=%s, files=%s)", 
                             message.author.id, character_display_name, MESSAGE_STYLE, len(files) if 'files' in locals() else 0)
            
        except Exception as exc:
            logger.exception("Error rendering game VN panel: %s", exc)
        
        return True
    
    async def _handle_narrator_message(self, message: discord.Message, game_state: GameState) -> None:
        """Handle narrator message (GM speaking as narrator). Uses EXACT same rendering as VN mode."""
        import sys
        bot_module = sys.modules.get('bot') or sys.modules.get('__main__')
        if not bot_module:
            logger.warning("Cannot get bot module for narrator message")
            return
        
        try:
            # Get narrator character - MUST match exactly how VN mode gets it
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
            
            # CRITICAL: Character name MUST be exactly "Narrator" (capitalized) to match vn_layouts.json
            # The layout key is normalized, but the character_name in state must match the original
            narrator_char_name = "Narrator"  # Force exact match for layout lookup
            
            # Create a TransformationState for narrator - EXACT same as VN mode
            from tfbot.models import TransformationState
            from tfbot.utils import member_profile_name, utc_now
            from datetime import timedelta
            from tfbot.panels import render_vn_panel, parse_discord_formatting, prepare_custom_emoji_images
            
            now = utc_now()
            narrator_state = TransformationState(
                user_id=message.author.id,
                guild_id=message.guild.id if message.guild else 0,
                character_name=narrator_char_name,  # MUST be "Narrator" for layout lookup
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
            
            # Use EXACT same rendering path as VN mode - direct render_vn_panel call
            cleaned_content = message.content.strip()
            formatted_segments = parse_discord_formatting(cleaned_content)
            custom_emoji_images = await prepare_custom_emoji_images(message, formatted_segments)
            
            # Get reply context if message is a reply
            reply_context = None
            if message.reference and message.reference.resolved:
                ref_msg = message.reference.resolved
                if isinstance(ref_msg, discord.Message):
                    reply_context = type('ReplyContext', (), {
                        'author': ref_msg.author.display_name,
                        'text': ref_msg.content[:100] if ref_msg.content else "",
                    })()
            
            # Render VN panel - EXACT same call as VN mode uses
            vn_file = render_vn_panel(
                state=narrator_state,
                message_content=cleaned_content,
                character_display_name=narrator_char_name,  # Use "Narrator" for display
                original_name=message.author.display_name,
                attachment_id=str(message.id),
                formatted_segments=formatted_segments,
                custom_emoji_images=custom_emoji_images,
                reply_context=reply_context,
            )
            
            if vn_file:
                # Send with same parameters as VN mode
                send_kwargs = {
                    "files": [vn_file],
                    "allowed_mentions": discord.AllowedMentions.none(),
                }
                if message.reference:
                    send_kwargs["reference"] = message.reference
                await message.channel.send(**send_kwargs)
                
                # Delete original message
                try:
                    await message.delete()
                except discord.HTTPException:
                    pass
            else:
                logger.warning("Failed to render narrator VN panel, falling back to text")
                # Fallback to text
                await message.channel.send(f"**{narrator_char_name}**: {cleaned_content}", allowed_mentions=discord.AllowedMentions.none())
                try:
                    await message.delete()
                except discord.HTTPException:
                    pass
        except Exception as exc:
            logger.exception("Error handling narrator message: %s", exc)

    def get_game_config(self, game_type: str) -> Optional[GameConfig]:
        """Get game configuration by type name."""
        return self._game_configs.get(game_type)
    
    def _get_player_number(self, game_state: GameState, user_id: int) -> Optional[int]:
        """Get player number (1, 2, 3, etc.) from pack."""
        pack = get_game_pack(game_state.game_type, self.packs_dir)
        if pack and pack.has_function("get_player_number"):
            return pack.call("get_player_number", game_state, user_id)
        return None
    
    def _swap_pack_player_metadata(self, game_state: GameState, user_id1: int, user_id2: int) -> None:
        """Swap per-pack metadata (tile numbers, turn order, player numbers)."""
        pack = get_game_pack(game_state.game_type, self.packs_dir)
        if not pack:
            return
        pack_module = getattr(pack, "module", None)
        if not pack_module:
            return
        get_game_data = getattr(pack_module, "get_game_data", None)
        if not callable(get_game_data):
            return
        
        try:
            data = get_game_data(game_state)
        except Exception as exc:
            logger.warning("Failed to fetch pack data during swap: %s", exc)
            return
        if not isinstance(data, dict):
            return
        
        def _swap_dict_entry(mapping: Dict[int, object]) -> None:
            if not isinstance(mapping, dict):
                return
            val1 = mapping.get(user_id1)
            val2 = mapping.get(user_id2)
            if val1 is None and val2 is None:
                return
            mapping[user_id1], mapping[user_id2] = val2, val1
        
        def _swap_list_entries(seq: List[int]) -> None:
            if not isinstance(seq, list):
                return
            for idx, value in enumerate(seq):
                if value == user_id1:
                    seq[idx] = user_id2
                elif value == user_id2:
                    seq[idx] = user_id1
        
        tile_numbers = data.get("tile_numbers")
        if isinstance(tile_numbers, dict):
            _swap_dict_entry(tile_numbers)
        
        player_numbers = data.get("player_numbers")
        if isinstance(player_numbers, dict):
            _swap_dict_entry(player_numbers)
        
        turn_order = data.get("turn_order")
        if isinstance(turn_order, list):
            try:
                idx1 = turn_order.index(user_id1)
                idx2 = turn_order.index(user_id2)
            except ValueError:
                idx1 = idx2 = None
            if idx1 is not None and idx2 is not None:
                turn_order[idx1], turn_order[idx2] = turn_order[idx2], turn_order[idx1]
        
        for list_key in ("players_rolled_this_turn", "winners", "players_reached_end_this_turn"):
            seq = data.get(list_key)
            if isinstance(seq, list):
                _swap_list_entries(seq)
        
        goal_reached_turn = data.get("goal_reached_turn")
        if isinstance(goal_reached_turn, dict):
            _swap_dict_entry(goal_reached_turn)

    def list_available_games(self) -> List[str]:
        """Get list of available game types."""
        if not self._has_packs:
            return []  # No games available if no packs
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
    
    def _get_command_lock(self, thread_id: int) -> asyncio.Lock:
        """Get or create a command lock for a specific game thread."""
        if thread_id not in self._command_locks:
            self._command_locks[thread_id] = asyncio.Lock()
        return self._command_locks[thread_id]
    
    async def _execute_gameboard_command(self, ctx: commands.Context, coro) -> None:
        """Execute a gameboard GM command with per-game locking to ensure message ordering."""
        # Only lock if we're in a game thread
        if not isinstance(ctx.channel, discord.Thread):
            # Not in a thread, execute without lock
            await coro()
            return
        
        thread_id = ctx.channel.id
        command_lock = self._get_command_lock(thread_id)
        
        # Check if lock is already held (another command is processing)
        if command_lock.locked():
            await ctx.reply("⏳ Another command is currently processing for this game. Please wait...", mention_author=False)
            return
        
        # Acquire lock and execute - this ensures all messages from this command appear in order
        async with command_lock:
            await coro()
        
        # CRITICAL: Process queued messages after command completes
        # This ensures messages sent during command processing are re-printed in order
        game_state = await self._get_game_state_for_context(ctx)
        if game_state:
            await self._process_queued_messages(game_state)
    
    def _resolve_target_member(
        self, 
        ctx: commands.Context, 
        game_state: GameState, 
        token: str
    ) -> Optional[discord.Member]:
        """
        Resolve a target token to a Discord member.
        Supports: @user mention, character_name, character_folder, or player display name.
        Returns the member if found, None otherwise.
        """
        if not ctx.guild:
            return None
        
        token = token.strip()
        if not token:
            return None
        
        # Try member mention first
        import re
        mention_match = re.search(r'<@!?(\d+)>', token)
        if mention_match:
            member_id = int(mention_match.group(1))
            member = ctx.guild.get_member(member_id)
            if member and member.id in game_state.players:
                return member
        
        token_lower = token.lower()
        
        # Try character name (exact match)
        for user_id, player in game_state.players.items():
            if player.character_name and player.character_name.lower() == token_lower:
                member = ctx.guild.get_member(user_id)
                if member:
                    return member
        
        # Try character name (partial match)
        for user_id, player in game_state.players.items():
            if player.character_name and token_lower in player.character_name.lower():
                member = ctx.guild.get_member(user_id)
                if member:
                    return member
        
        # Try character folder (via character lookup)
        character = self._get_character_by_name(token)
        if character:
            # Find player with this character
            for user_id, player in game_state.players.items():
                if player.character_name == character.name:
                    member = ctx.guild.get_member(user_id)
                    if member:
                        return member
        
        # Try display name (exact match)
        for user_id, player in game_state.players.items():
            member = ctx.guild.get_member(user_id)
            if member and member.display_name.lower() == token_lower:
                return member
        
        # Try display name (partial match)
        for user_id, player in game_state.players.items():
            member = ctx.guild.get_member(user_id)
            if member and token_lower in member.display_name.lower():
                return member
        
        return None

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
            # CRITICAL: Use Windows-safe time format (no colons)
            from datetime import datetime
            now = datetime.now()
            date_str = now.strftime("%d-%m-%Y")
            time_str = now.strftime("%H-%M")  # Changed from %H:%M to %H-%M (Windows-safe)
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
                    "debug_mode": game_state.debug_mode,
                    "turn_count": game_state.turn_count,
                    "game_started": game_state.game_started,
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
            # Ensure directory exists
            state_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file and verify it was written successfully
            try:
                json_content = json.dumps(data, indent=2)
                state_file.write_text(json_content, encoding="utf-8")
                
                # Verify file was written (check file size)
                if not state_file.exists():
                    logger.error("CRITICAL: Save file was not created: %s", state_file)
                    return
                
                file_size = state_file.stat().st_size
                if file_size == 0:
                    logger.error("CRITICAL: Save file is 0 bytes: %s", state_file)
                    return
                
                logger.info("Game state saved successfully: %s (%d bytes)", filename, file_size)
            except Exception as exc:
                logger.error("CRITICAL: Failed to save game state to %s: %s", state_file, exc, exc_info=True)
                raise

    async def _save_auto_save(self, game_state: GameState, ctx: Optional[commands.Context] = None) -> None:
        """Save auto-save at end of turn. Replaces previous auto-save for this game."""
        async with self._lock:
            try:
                # Get thread name for filename
                thread_name = "game"
                guild = None
                if ctx and ctx.guild:
                    guild = ctx.guild
                elif game_state.forum_channel_id:
                    # Try to get guild from forum channel
                    try:
                        forum_channel = self.bot.get_channel(game_state.forum_channel_id)
                        if forum_channel and hasattr(forum_channel, 'guild') and forum_channel.guild:
                            guild = forum_channel.guild
                    except Exception:
                        pass
                
                if guild:
                    try:
                        thread = guild.get_channel(game_state.game_thread_id)
                        if thread and hasattr(thread, 'name'):
                            thread_name = thread.name
                            # Sanitize thread name for filename
                            thread_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in thread_name)
                            if not thread_name:
                                thread_name = "game"
                    except Exception:
                        pass
                
                # Get GM name for filename
                gm_name = "gm"
                if guild:
                    try:
                        gm_member = guild.get_member(game_state.gm_user_id)
                        if gm_member:
                            gm_name = gm_member.display_name.lower()
                            # Sanitize GM name for filename
                            gm_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in gm_name)
                            if not gm_name:
                                gm_name = "gm"
                    except Exception:
                        pass
                
                # Get game type
                game_type = game_state.game_type or "unknown"
                safe_game_type = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in game_type)
                
                # Get date
                from datetime import datetime
                now = datetime.now()
                date_str = now.strftime("%d-%m-%Y")
                
                # Get turn number
                turn_num = game_state.turn_count
                
                # Generate filename: game1_snakesladders_aireo_date_turn1
                filename = f"{thread_name}_{safe_game_type}_{gm_name}_{date_str}_turn{turn_num}.json"
                state_file = self.states_dir / filename
                
                # Delete previous auto-save for this game (if exists)
                # Pattern: {thread_name}_{safe_game_type}_{gm_name}_{date_str}_turn*.json
                pattern = f"{thread_name}_{safe_game_type}_{gm_name}_{date_str}_turn*.json"
                for old_file in self.states_dir.glob(pattern):
                    if old_file != state_file:
                        try:
                            old_file.unlink()
                            logger.info("Deleted previous auto-save: %s", old_file.name)
                        except Exception as exc:
                            logger.warning("Failed to delete previous auto-save %s: %s", old_file.name, exc)
                
                # Save current state
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
                    "debug_mode": game_state.debug_mode,
                    "turn_count": game_state.turn_count,
                    "game_started": game_state.game_started,
                    "is_paused": game_state.is_paused,
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
                
                # Ensure directory exists
                state_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Write file and verify
                json_content = json.dumps(data, indent=2)
                state_file.write_text(json_content, encoding="utf-8")
                
                # Verify file was written
                if not state_file.exists():
                    logger.error("CRITICAL: Auto-save file was not created: %s", state_file)
                    return
                
                file_size = state_file.stat().st_size
                if file_size == 0:
                    logger.error("CRITICAL: Auto-save file is 0 bytes: %s", state_file)
                    return
                
                logger.info("Auto-save created successfully: %s (%d bytes, turn %d)", filename, file_size, turn_num)
            except Exception as exc:
                logger.error("CRITICAL: Failed to create auto-save: %s", exc, exc_info=True)

    async def _update_board(
        self,
        game_state: GameState,
        error_channel: Optional[discord.abc.Messageable] = None,
        target_thread: str = "map",
        also_post_to_game: bool = False
    ) -> None:
        """
        Update board image.
        
        Args:
            game_state: The game state
            error_channel: Channel to send error messages to
            target_thread: "game" to send to game thread, "map" to send to map thread (default)
            also_post_to_game: If True, also post to game thread in addition to map thread (for game start and turn start)
        """
        logger.info("Updating board for game thread %s (map thread %s), target=%s, also_post_to_game=%s", 
                   game_state.game_thread_id, game_state.map_thread_id, target_thread, also_post_to_game)
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
        
        # Send progress message if we have an error channel (game thread) and it's a Thread or TextChannel
        progress_msg = None
        if error_channel and (isinstance(error_channel, discord.Thread) or isinstance(error_channel, discord.TextChannel)):
            try:
                progress_msg = await error_channel.send("⏳ Generating board...", allowed_mentions=discord.AllowedMentions.none())
            except Exception:
                pass
        
        # CRITICAL: Make board rendering async to avoid blocking
        # Run PIL operations in executor thread to prevent blocking the event loop
        try:
            loop = asyncio.get_event_loop()
            board_file = await loop.run_in_executor(
                None,
                lambda: render_game_board(game_state, game_config, self.assets_dir)
            )
        except Exception as exc:
            logger.error("Failed to render board image (async): %s", exc, exc_info=True)
            board_file = None
        
        if not board_file:
            error_msg = "❌ Failed to render board image"
            logger.warning(error_msg)
            if error_channel:
                try:
                    await error_channel.send(error_msg)
                except Exception:
                    pass
            return
        
        # Post to primary target thread (map forum by default)
        logger.info("Board image regenerated, posting to %s thread", target_thread)
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
            logger.info("Board updated successfully in %s thread, new message ID: %s", target_thread, board_msg.id)
            
            # Delete progress message if it exists
            if progress_msg:
                try:
                    await progress_msg.delete()
                except Exception:
                    pass
        except discord.HTTPException as exc:
            error_msg = f"❌ Failed to post board image to {target_thread} thread: {exc}"
            logger.warning(error_msg)
            if error_channel:
                try:
                    await error_channel.send(error_msg)
                except Exception:
                    pass
            return
        
        # If also_post_to_game is True, also post to game thread (for game start and turn start)
        if also_post_to_game and game_state.game_thread_id and game_state.game_thread_id != target_thread_id:
            game_thread = self.bot.get_channel(game_state.game_thread_id)
            if isinstance(game_thread, discord.Thread):
                try:
                    # Re-read the board file (or use the same file object if it supports multiple sends)
                    # Note: discord.File objects can only be sent once, so we need to recreate it
                    loop = asyncio.get_event_loop()
                    game_board_file = await loop.run_in_executor(
                        None,
                        lambda: render_game_board(game_state, game_config, self.assets_dir)
                    )
                    if game_board_file:
                        await game_thread.send(
                            file=game_board_file,
                            allowed_mentions=discord.AllowedMentions.none()
                        )
                        logger.info("Board also posted to game thread for visibility")
                    else:
                        logger.warning("Failed to generate board file for game thread")
                except Exception as exc:
                    logger.exception("CRITICAL: Failed to post board to game thread: %s", exc)
                    # Try to send error message to game thread
                    try:
                        await game_thread.send("❌ Failed to display board image. Check map thread for board updates.", allowed_mentions=discord.AllowedMentions.none())
                    except Exception:
                        pass
        
    async def _process_queued_messages(self, game_state: GameState) -> None:
        """Process all queued messages for a game thread after command completes."""
        thread_id = game_state.game_thread_id
        if thread_id not in self._message_queues:
            return
        
        queue = self._message_queues[thread_id]
        if not queue:
            return
        
        logger.info("Processing %d queued message(s) for thread %s", len(queue), thread_id)
        
        # Process all queued messages
        messages_to_process = queue.copy()
        queue.clear()  # Clear queue immediately to prevent duplicates
        
        for message_data in messages_to_process:
            try:
                # Recreate a message-like object from stored data
                class QueuedMessage:
                    def __init__(self, data):
                        self.content = data.get('content', '')
                        self.author = data.get('author')
                        self.channel = data.get('channel')
                        self.guild = data.get('guild')
                        self.reference = data.get('reference')
                        # Reconstruct attachment objects from stored data
                        self._attachment_data = data.get('attachments', [])
                        # Create attachment-like objects for compatibility
                        self.attachments = []
                        for att_data in self._attachment_data:
                            # Create a simple object that mimics discord.Attachment
                            class AttachmentProxy:
                                def __init__(self, att_data):
                                    self.filename = att_data.get('filename', 'unknown')
                                    self._bytes = att_data.get('bytes', b'')
                                    self.content_type = att_data.get('content_type')
                                
                                async def read(self):
                                    return self._bytes
                            
                            self.attachments.append(AttachmentProxy(att_data))
                        self.id = data.get('id', 0)
                    
                    async def delete(self):
                        # Message was already deleted, so this is a no-op
                        pass
                
                queued_message = QueuedMessage(message_data)
                has_attachments = len(queued_message.attachments) > 0
                logger.debug("Processing queued message from %s: %s (attachments: %d)", 
                           queued_message.author.id if queued_message.author else "?", 
                           queued_message.content[:50] if queued_message.content else "(no content)",
                           len(queued_message.attachments) if has_attachments else 0)
                
                # CRITICAL: Check if lock is still held - if so, skip processing (shouldn't happen, but safety check)
                command_lock = self._get_command_lock(thread_id)
                if command_lock.locked():
                    logger.warning("Command lock still held when processing queued messages - skipping to prevent recursion")
                    continue
                
                # Re-call handle_message - it will process the message normally
                # The message.delete() call in handle_message will fail silently since message is already deleted
                # CRITICAL: This will process ALL queued messages (including GM/admin) in order
                # Pass is_queued=True to skip the lock check and prevent re-queuing
                await self.handle_message(queued_message, command_invoked=False, is_queued=True)
            except Exception as exc:
                logger.exception("Error processing queued message from %s: %s", message_data.get('author').id if message_data.get('author') else "unknown", exc)

    async def _log_action(self, game_state: GameState, action: str) -> None:
        """Log a game action to the logger."""
        logger.info("Game action [thread %s]: %s", game_state.game_thread_id, action)

    # GM Command Methods
    async def command_startgame(self, ctx: commands.Context, game_type: str = "") -> None:
        """Start a new game or resume an existing game. Only admins can start games, and they become the GM."""
        logger.info("=" * 80)
        logger.info("command_startgame: ENTRY - called by %s (%s) in channel %s with game_type='%s'", 
                   ctx.author.id, ctx.author.display_name, ctx.channel.id if ctx.channel else None, game_type)
        logger.info("command_startgame: Context - guild=%s, channel_type=%s", 
                   ctx.guild.id if ctx.guild else None, type(ctx.channel).__name__ if ctx.channel else None)
        
        logger.info("command_startgame: STEP 1 - Checking if author is guild member")
        if not isinstance(ctx.author, discord.Member):
            logger.warning("command_startgame: STEP 1 FAILED - Not a guild member (author type: %s)", type(ctx.author).__name__)
            await ctx.reply("This command can only be used inside a server.", mention_author=False)
            return
        
        logger.info("command_startgame: STEP 1 PASSED - Author is guild member")
        logger.info("command_startgame: STEP 2 - Checking admin status for user %s", ctx.author.id)
        
        # Only admins can start new games (they become the GM)
        is_admin_result = is_admin(ctx.author)
        logger.info("command_startgame: is_admin(%s) = %s", ctx.author.id, is_admin_result)
        if not is_admin_result:
            logger.warning("command_startgame: STEP 2 FAILED - User %s is not admin", ctx.author.id)
            await ctx.reply("Only admins can start new games. The admin who starts the game becomes the GM.", mention_author=False)
            return
        
        logger.info("command_startgame: STEP 2 PASSED - User %s is admin, proceeding", ctx.author.id)
        
        logger.info("command_startgame: STEP 3 - Checking game_type parameter")
        if not game_type:
            available = ", ".join(self.list_available_games()) if self.list_available_games() else "No games available"
            logger.warning("command_startgame: STEP 3 FAILED - No game_type provided, available: %s", available)
            if not self._has_packs:
                await ctx.reply("❌ **No games available:** No game packs found. Gameboard is disabled.", mention_author=False)
            else:
                await ctx.reply("Usage: `!startgame <game_type>`\nAvailable games: " + available, mention_author=False)
            return
        
        # Check if gameboard is disabled (no packs)
        if not self._has_packs:
            logger.warning("command_startgame: Gameboard disabled - no packs available")
            await ctx.reply("❌ **Gameboard disabled:** No game packs found. Please add game pack files to `games/packs/` directory.", mention_author=False)
            return
        
        logger.info("command_startgame: STEP 3 PASSED - game_type='%s'", game_type)
        logger.info("command_startgame: STEP 4 - Looking up game config for type '%s'", game_type)
        logger.info("command_startgame: Available game configs: %s", list(self._game_configs.keys()))
        game_config = self.get_game_config(game_type)
        if not game_config:
            available = ", ".join(self.list_available_games())
            logger.error("command_startgame: STEP 4 FAILED - Unknown game type '%s', available: %s", game_type, available)
            await ctx.reply(f"Unknown game type: `{game_type}`. Available: " + available, mention_author=False)
            return
        
        logger.info("command_startgame: STEP 4 PASSED - Game config found for '%s' (name: %s)", game_type, game_config.name)
        
        logger.info("command_startgame: STEP 5 - Checking if in existing game thread")
        logger.info("command_startgame: Channel type: %s, channel ID: %s", type(ctx.channel).__name__, ctx.channel.id if ctx.channel else None)
        if isinstance(ctx.channel, discord.Thread):
            logger.info("command_startgame: STEP 5 - In a thread, checking for existing game state")
            existing_state = await self._detect_and_load_game_thread(ctx.channel)
            if existing_state:
                logger.info("command_startgame: STEP 5 - Found existing game state in thread %s, resuming", ctx.channel.id)
                await ctx.reply(f"✅ Game already exists in this thread! Resuming game. Use `!addplayer` and `!assign` to continue.", mention_author=False)
                # Update board if needed
                if not existing_state.board_message_id:
                    logger.info("command_startgame: Updating board for existing game")
                    await self._update_board(existing_state, error_channel=ctx.channel)
                logger.info("command_startgame: EXIT - Resumed existing game")
                return
            logger.info("command_startgame: STEP 5 - No existing game state found in thread, continuing")
        else:
            logger.info("command_startgame: STEP 5 - Not in a thread, will create new game")
        
        logger.info("command_startgame: STEP 6 - Looking up forum channel")
        logger.info("command_startgame: Forum channel ID from config: %s", self.forum_channel_id)
        logger.info("command_startgame: Bot has %d guilds", len(self.bot.guilds))
        
        forum_channel = self.bot.get_channel(self.forum_channel_id)
        logger.info("command_startgame: get_channel(%s) returned: %s", self.forum_channel_id, forum_channel)
        
        if not forum_channel:
            logger.error("command_startgame: STEP 6 FAILED - Forum channel %s not found or not accessible by bot", self.forum_channel_id)
            logger.error("command_startgame: Bot guilds: %s", [g.id for g in self.bot.guilds])
            await ctx.reply(
                f"❌ **Forum channel not found:** The configured forum channel (ID: {self.forum_channel_id}) doesn't exist or the bot can't access it.\n\n"
                f"Please check:\n"
                f"• The channel ID in `games/game_config.json` is correct\n"
                f"• The channel still exists\n"
                f"• The bot has access to the channel",
                mention_author=False
            )
            logger.info("command_startgame: EXIT - Forum channel not found")
            return
        
        logger.info("command_startgame: STEP 6 PASSED - Forum channel found: %s (type: %s, ID: %s)", 
                   forum_channel.name, type(forum_channel).__name__, forum_channel.id)
        
        logger.info("command_startgame: STEP 7 - Validating channel type")
        if not isinstance(forum_channel, discord.ForumChannel):
            logger.error("command_startgame: STEP 7 FAILED - Channel %s is not a forum channel (type: %s)", 
                        self.forum_channel_id, type(forum_channel).__name__)
            await ctx.reply(
                f"❌ **Invalid channel type:** Channel ID {self.forum_channel_id} is not a forum channel.\n\n"
                f"Please configure a forum channel in `games/game_config.json`.",
                mention_author=False
            )
            logger.info("command_startgame: EXIT - Invalid channel type")
            return
        
        logger.info("command_startgame: STEP 7 PASSED - Forum channel validated: %s (%s)", forum_channel.name, forum_channel.id)
        
        # Validate map forum channel (separate from game forum)
        logger.info("command_startgame: STEP 7.5 - Looking up map forum channel")
        logger.info("command_startgame: Map forum channel ID from config: %s", self.map_forum_channel_id)
        map_forum_channel = self.bot.get_channel(self.map_forum_channel_id)
        logger.info("command_startgame: get_channel(%s) returned: %s", self.map_forum_channel_id, map_forum_channel)
        
        if not map_forum_channel:
            logger.error("command_startgame: STEP 7.5 FAILED - Map forum channel %s not found or not accessible by bot", self.map_forum_channel_id)
            await ctx.reply(
                f"❌ **Map forum channel not found:** The configured map forum channel (ID: {self.map_forum_channel_id}) doesn't exist or the bot can't access it.\n\n"
                f"Please check:\n"
                f"• The channel ID in environment variable `TFBOT_GAME_MAP_FORUM_CHANNEL_ID` is correct\n"
                f"• The channel still exists\n"
                f"• The bot has access to the channel",
                mention_author=False
            )
            logger.info("command_startgame: EXIT - Map forum channel not found")
            return
        
        logger.info("command_startgame: STEP 7.5 PASSED - Map forum channel found: %s (type: %s, ID: %s)", 
                   map_forum_channel.name, type(map_forum_channel).__name__, map_forum_channel.id)
        
        logger.info("command_startgame: STEP 7.6 - Validating map forum channel type")
        if not isinstance(map_forum_channel, discord.ForumChannel):
            logger.error("command_startgame: STEP 7.6 FAILED - Channel %s is not a forum channel (type: %s)", 
                        self.map_forum_channel_id, type(map_forum_channel).__name__)
            await ctx.reply(
                f"❌ **Invalid map forum channel type:** Channel ID {self.map_forum_channel_id} is not a forum channel.\n\n"
                f"Please configure a forum channel in environment variable `TFBOT_GAME_MAP_FORUM_CHANNEL_ID`.",
                mention_author=False
            )
            logger.info("command_startgame: EXIT - Invalid map forum channel type")
            return
        
        logger.info("command_startgame: STEP 7.6 PASSED - Map forum channel validated: %s (%s)", map_forum_channel.name, map_forum_channel.id)
        
        logger.info("command_startgame: STEP 8 - Checking bot permissions in forum channels")
        if ctx.guild:
            logger.info("command_startgame: Getting bot member for guild %s (bot.user.id: %s)", 
                       ctx.guild.id, self.bot.user.id if self.bot.user else None)
            bot_member = ctx.guild.get_member(self.bot.user.id) if self.bot.user else None
            if not bot_member:
                logger.error("command_startgame: STEP 8 FAILED - Bot member not found in guild (bot.user.id: %s)", 
                           self.bot.user.id if self.bot.user else None)
                await ctx.reply("❌ **Error:** Bot member not found in guild. This is unusual.", mention_author=False)
                logger.info("command_startgame: EXIT - Bot member not found")
                return
            else:
                logger.info("command_startgame: Bot member found: %s", bot_member.display_name)
                logger.info("command_startgame: Checking permissions for bot in forum channel")
                channel_perms = forum_channel.permissions_for(bot_member)
                missing_perms = []
                
                logger.info("command_startgame: Permission check results:")
                # For forum channels, use create_public_threads (not create_forum_threads)
                has_create_threads = getattr(channel_perms, 'create_public_threads', False)
                logger.info("  - create_public_threads: %s", has_create_threads)
                logger.info("  - send_messages_in_threads: %s", channel_perms.send_messages_in_threads)
                logger.info("  - attach_files: %s", channel_perms.attach_files)
                logger.info("  - view_channel: %s", channel_perms.view_channel)
                
                if not has_create_threads:
                    missing_perms.append("Create Public Threads")
                if not channel_perms.send_messages_in_threads:
                    missing_perms.append("Send Messages in Threads")
                if not channel_perms.attach_files:
                    missing_perms.append("Attach Files")
                if not channel_perms.view_channel:
                    missing_perms.append("View Channel")
                
                if missing_perms:
                    perm_list = ", ".join(missing_perms)
                    logger.error("command_startgame: STEP 8 FAILED - Bot missing permissions: %s", perm_list)
                    await ctx.reply(
                        f"❌ **Permission Error:** The bot is missing required permissions in the forum channel:\n"
                        f"**Missing:** {perm_list}\n\n"
                        f"Please ensure the bot has the following permissions in <#{forum_channel.id}>:\n"
                        f"• Create Public Threads (or Create Forum Threads)\n"
                        f"• Send Messages in Threads\n"
                        f"• Attach Files\n"
                        f"• View Channel",
                        mention_author=False
                    )
                    logger.info("command_startgame: EXIT - Missing permissions")
                    return
                logger.info("command_startgame: STEP 8 PASSED - All required permissions present")
        else:
            logger.error("command_startgame: STEP 8 FAILED - No guild context available for permission check")
            await ctx.reply("❌ **Error:** No guild context available.", mention_author=False)
            logger.info("command_startgame: EXIT - No guild context")
            return
        
        # Generate unique game number by checking existing threads
        logger.info("command_startgame: Generating unique game number")
        game_number = 1
        game_prefix = f"{game_config.name} #"
        logger.debug("command_startgame: Game prefix: '%s'", game_prefix)
        try:
            logger.debug("command_startgame: Checking archived threads")
            # Fetch recent threads to find highest number
            archived_count = 0
            async for thread in forum_channel.archived_threads(limit=100):
                archived_count += 1
                if thread.name.startswith(game_prefix):
                    try:
                        # Extract number from thread name like "Snakes and Ladders #123 - Username"
                        parts = thread.name.split("#", 1)
                        if len(parts) > 1:
                            num_part = parts[1].split(" - ", 1)[0]
                            thread_num = int(num_part)
                            if thread_num >= game_number:
                                game_number = thread_num + 1
                                logger.debug("command_startgame: Found game number %d in archived thread", thread_num)
                    except (ValueError, IndexError) as e:
                        logger.debug("command_startgame: Failed to parse thread name '%s': %s", thread.name, e)
                        pass
            logger.debug("command_startgame: Checked %d archived threads", archived_count)
            
            # Also check active threads
            logger.debug("command_startgame: Checking active threads")
            active_threads = forum_channel.threads
            logger.debug("command_startgame: Found %d active threads", len(active_threads))
            for thread in active_threads:
                if thread.name.startswith(game_prefix):
                    try:
                        parts = thread.name.split("#", 1)
                        if len(parts) > 1:
                            num_part = parts[1].split(" - ", 1)[0]
                            thread_num = int(num_part)
                            if thread_num >= game_number:
                                game_number = thread_num + 1
                                logger.debug("command_startgame: Found game number %d in active thread", thread_num)
                    except (ValueError, IndexError) as e:
                        logger.debug("command_startgame: Failed to parse thread name '%s': %s", thread.name, e)
                        pass
            logger.info("command_startgame: Generated game number: %d", game_number)
        except Exception as exc:
            logger.error("command_startgame: Failed to check existing threads for game number: %s", exc, exc_info=True)
            # Fallback to timestamp-based number
            import time
            game_number = int(time.time()) % 100000
            logger.warning("command_startgame: Using fallback game number: %d", game_number)
        
        # Create TWO separate forum posts in the same channel: one for chat, one for board images
        thread_name = f"{game_config.name} #{game_number} - {ctx.author.display_name}"
        map_thread_name = f"{game_config.name} #{game_number} Map - {ctx.author.display_name}"
        initial_message = (
            f"🎲 **{game_config.name}** game started by {ctx.author.display_name}\n\n"
            f"Use `!addplayer @user` to add players, then `!assign @user character_name` to assign characters."
        )
        map_initial_message = (
            f"🗺️ **{game_config.name} Map** - Board updates will appear here.\n\n"
            f"This post is read-only. All messages except admin commands will be automatically deleted."
        )
        
        logger.info("command_startgame: Creating game thread: '%s'", thread_name)
        logger.debug("command_startgame: Thread name length: %d, Map thread name length: %d", len(thread_name), len(map_thread_name))
        
        try:
            # Create game thread (for chat/commands)
            logger.info("command_startgame: Attempting to create game thread in forum channel %s", forum_channel.id)
            thread, message = await forum_channel.create_thread(
                name=thread_name,
                auto_archive_duration=1440,
                content=initial_message
            )
            logger.info("command_startgame: Successfully created game thread: %s (ID: %s)", thread.name, thread.id)
            # Edit in GM mention after creation to avoid ping
            if isinstance(ctx.author, discord.Member):
                try:
                    mention_content = f"{ctx.author.mention}\n\n{initial_message}"
                    await message.edit(content=mention_content)
                    logger.debug("command_startgame: Added GM mention to game thread via edit")
                except Exception as exc:
                    logger.warning("command_startgame: Failed to edit game thread message with GM mention: %s", exc)
            
            # Create map thread (for board images only, in separate map forum channel)
            logger.info("command_startgame: Attempting to create map thread: '%s' in map forum channel %s", map_thread_name, map_forum_channel.id)
            map_thread, map_message = await map_forum_channel.create_thread(
                name=map_thread_name,
                auto_archive_duration=1440,
                content=map_initial_message
            )
            logger.info("command_startgame: Successfully created map thread: %s (ID: %s) in map forum channel %s", map_thread.name, map_thread.id, map_forum_channel.id)
            if isinstance(ctx.author, discord.Member):
                try:
                    mention_content = f"{ctx.author.mention}\n\n{map_initial_message}"
                    await map_message.edit(content=mention_content)
                    logger.debug("command_startgame: Added GM mention to map thread via edit")
                except Exception as exc:
                    logger.warning("command_startgame: Failed to edit map thread message with GM mention: %s", exc)
        except discord.Forbidden as exc:
            # Determine which channel had the permission error
            error_msg = (
                f"❌ **Permission Denied:** The bot doesn't have permission to create threads.\n\n"
                f"**Required permissions:**\n"
                f"• In game forum <#{forum_channel.id}>: Create Public Threads, Send Messages in Threads, Attach Files\n"
                f"• In map forum <#{map_forum_channel.id}>: Create Public Threads, Send Messages in Threads, Attach Files\n\n"
                f"Please check the bot's role permissions in both forum channels and try again."
            )
            await ctx.reply(error_msg, mention_author=False)
            logger.error("Permission denied creating threads (game forum: %s, map forum: %s): %s", forum_channel.id, map_forum_channel.id, exc)
            return
        except discord.HTTPException as exc:
            # Check for specific error codes
            if exc.status == 403:
                error_msg = (
                    f"❌ **Permission Error (403):** The bot lacks permissions to create threads.\n\n"
                    f"Please ensure the bot has 'Create Public Threads' (or 'Create Forum Threads') permission in:\n"
                    f"• Game forum: <#{forum_channel.id}>\n"
                    f"• Map forum: <#{map_forum_channel.id}>"
                )
            elif exc.status == 404:
                error_msg = (
                    f"❌ **Channel Not Found (404):** One or both forum channels no longer exist.\n\n"
                    f"• Game forum: {forum_channel.id}\n"
                    f"• Map forum: {map_forum_channel.id}\n\n"
                    f"Please update the forum channel configuration."
                )
            else:
                error_msg = (
                    f"❌ **Failed to create game thread:** {exc}\n\n"
                    f"**Error Code:** {exc.status if hasattr(exc, 'status') else 'Unknown'}\n"
                    f"**Forum Channels:**\n"
                    f"• Game forum: <#{forum_channel.id}>\n"
                    f"• Map forum: <#{map_forum_channel.id}>\n\n"
                    f"If this persists, check:\n"
                    f"• Bot permissions in both forum channels\n"
                    f"• Both forum channels still exist\n"
                    f"• Bot has proper role hierarchy"
                )
            await ctx.reply(error_msg, mention_author=False)
            logger.error("HTTPException creating game thread: %s (status: %s)", exc, getattr(exc, 'status', 'unknown'))
            return
        
        # Create game state with both threads (same channel, different posts)
        logger.info("command_startgame: Creating game state with threads: game=%s, map=%s", thread.id, map_thread.id)
        game_state = GameState(
            game_thread_id=thread.id,
            forum_channel_id=forum_channel.id,
            dm_channel_id=self.dm_channel_id,
            gm_user_id=ctx.author.id,
            game_type=game_type,
            map_thread_id=map_thread.id,
            narrator_user_id=ctx.author.id,  # GM becomes narrator by default
            debug_mode=False,  # Default to off
            turn_count=0,  # Start at turn 0, increments to 1 on first turn completion
            is_paused=False,  # Game starts unpaused
            game_started=False,  # Game starts as not ready - GM must use !start to begin
        )
        
        logger.info("command_startgame: Storing game state in active games")
        self._active_games[thread.id] = game_state
        
        # Send progress message
        progress_msg = await ctx.reply("⏳ Creating initial board...", mention_author=False)
        
        # Post initial blank board (no players yet - board will update when characters are assigned)
        logger.info("command_startgame: Creating initial blank board")
        await self._update_board(game_state, error_channel=ctx.channel)
        logger.info("command_startgame: Initial blank board created")
        
        # Delete progress message
        try:
            await progress_msg.delete()
        except Exception:
            pass
        
        logger.info("command_startgame: STEP 12 - Sending success message to user")
        try:
            await ctx.reply(f"Game started! Thread: {thread.mention}", mention_author=False)
            logger.info("command_startgame: Success message sent")
        except Exception as exc:
            logger.error("command_startgame: Failed to send success message: %s", exc, exc_info=True)
        
        await self._log_action(game_state, f"Game started by {ctx.author.display_name}")
        logger.info("=" * 80)
        logger.info("command_startgame: EXIT - Command completed successfully")
        logger.info("=" * 80)

    async def command_listgames(self, ctx: commands.Context) -> None:
        """List available games."""
        games = self.list_available_games()
        if not games:
            if not self._has_packs:
                await ctx.reply("❌ **No games available:** No game packs found. Gameboard is disabled.", mention_author=False)
            else:
                await ctx.reply("No games configured.", mention_author=False)
            return
        
        lines = ["**Available Games:**"]
        for game_key in games:
            config = self._game_configs.get(game_key)
            name = config.name if config else game_key
            lines.append(f"- `{game_key}` - {name}")
        
        await ctx.reply("\n".join(lines), mention_author=False)

    async def command_addplayer(self, ctx: commands.Context, member: Optional[discord.Member] = None, *, character_name: str = "") -> None:
        """Add a player to the game (GM only). Optional: assign character with !addplayer @user character_name"""
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
            await ctx.reply("Usage: `!addplayer @user [character_name]`\nExample: `!addplayer @user kiyoshi`", mention_author=False)
            return
        
        if member.id in game_state.players:
            await ctx.reply(f"{member.mention} is already in the game.", mention_author=False)
            return
        
        # Send immediate progress message
        progress_msg = await ctx.reply("⏳ Adding player...", mention_author=False)
        
        player = GamePlayer(user_id=member.id, grid_position="A1")
        game_state.players[member.id] = player
        
        # Call pack's on_player_added if it exists
        game_config = self.get_game_config(game_state.game_type)
        pack = get_game_pack(game_state.game_type, self.packs_dir)
        if pack and pack.has_function("on_player_added") and game_config:
            pack.call("on_player_added", game_state, player, game_config)
        
        # Check if pack wants board update on player added
        should_update = False
        if pack and pack.has_function("should_update_board"):
            should_update = pack.call("should_update_board", game_state, "player_added")
        
        # Board will be updated when character is assigned (not when player is added)
        # Unless pack specifically requests it
        if should_update:
            await self._update_board(game_state, error_channel=ctx.channel)
        
        # If character_name provided, assign character
        if character_name and character_name.strip():
            character_name = character_name.strip()
            logger.info("command_addplayer: Auto-assigning character '%s' to %s", character_name, member.id)
            
            # Verify character exists
            character = self._get_character_by_name(character_name)
            if not character:
                await ctx.reply(f"Added {member.mention} to the game.\n❌ Unable to locate character '{character_name}'. Use `!assign @{member.display_name} <character>` to assign later.", mention_author=False)
                await self._log_action(game_state, f"Player {member.display_name} added (character '{character_name}' not found)")
                return
            
            # Assign character using the same logic as command_assign
            actual_character_name = character.name
            player.character_name = actual_character_name
            
            # Create TransformationState for the player
            if not ctx.guild:
                await ctx.reply(f"Added {member.mention} to the game.\n❌ Error: Cannot assign character outside of a server.", mention_author=False)
                return
            
            # CRITICAL: Clear any existing game state for this player before creating new one
            # This ensures we don't have stale state interfering
            if member.id in game_state.player_states:
                logger.info("Clearing existing game state for player %s before assignment", member.id)
                del game_state.player_states[member.id]
            
            # CRITICAL: Always create a new state - never reuse old state
            state = await self._create_game_state_for_player(
                player,
                member.id,
                ctx.guild.id,
                actual_character_name,
            )
            if not state:
                logger.error("CRITICAL: Failed to create state for player %s with character %s", member.id, actual_character_name)
                await ctx.reply(f"Added {member.mention} to the game.\n❌ Error: Failed to create state for character '{character_name}'. Use `!assign @{member.display_name} <character>` to assign later.", mention_author=False)
                await self._log_action(game_state, f"Player {member.display_name} added (state creation failed)")
                return
            
            # CRITICAL: Force state to match player character name BEFORE storing
            if state.character_name != actual_character_name:
                logger.warning("State character_name mismatch in addplayer! Expected '%s', got '%s'. Fixing...", 
                            actual_character_name, state.character_name)
                state.character_name = actual_character_name
                if character:
                    state.character_folder = character.folder
                    state.character_avatar_path = character.avatar_path
                    state.character_message = character.message or ""
            
            # CRITICAL: Always store the state - this replaces any old state
            game_state.player_states[member.id] = state
            logger.info("Stored game state for player %s with character '%s'", member.id, state.character_name)
            
            # Final verification - ensure state matches player
            if state.character_name != actual_character_name or player.character_name != actual_character_name:
                logger.error("CRITICAL: Verification failed after storing state! player.character_name='%s', state.character_name='%s', actual='%s'", 
                            player.character_name, state.character_name, actual_character_name)
                # Force correction
                player.character_name = actual_character_name
                state.character_name = actual_character_name
                game_state.player_states[member.id] = state
            
            # If GM was assigned as a player, remove narrator role
            if member.id == game_state.gm_user_id and game_state.narrator_user_id == member.id:
                game_state.narrator_user_id = None
            
            # Call pack's on_character_assigned if it exists
            if pack and pack.has_function("on_character_assigned"):
                pack.call("on_character_assigned", game_state, player, actual_character_name)
            
            # Check if pack wants board update on character assignment
            should_update = True
            if pack and pack.has_function("should_update_board"):
                should_update = pack.call("should_update_board", game_state, "character_assigned")
            
            if should_update:
                await self._update_board(game_state, error_channel=ctx.channel)
            
            # Update progress message
            try:
                await progress_msg.edit(content="⏳ Assigning character...")
            except Exception:
                pass
            
            # Send transformation message
            try:
                from tfbot.utils import member_profile_name
                import sys
                bot_module = sys.modules.get('bot') or sys.modules.get('__main__')
                _format_character_message = getattr(bot_module, '_format_character_message', None)
                _get_magic_emoji = getattr(bot_module, '_get_magic_emoji', None)
                _format_special_reroll_hint = getattr(bot_module, '_format_special_reroll_hint', None)
                
                original_name = member_profile_name(member)
                character_message = character.message or ""
                duration_label = "Game"
                
                # Get player number for display (always get it, regardless of formatting method)
                player_number = self._get_player_number(game_state, member.id)
                if player_number:
                    player_number_text = f" - Player {player_number}"
                    logger.info("Assignment message (addplayer): %s assigned as Player %d", member.display_name, player_number)
                else:
                    player_number_text = ""
                    logger.warning("No player number found for %s (user_id=%s) in addplayer - player may not have been properly added to game", member.display_name, member.id)
                
                if _format_character_message:
                    response_text = _format_character_message(
                        character_message,
                        original_name,
                        member.mention,
                        duration_label,
                        actual_character_name,
                    )
                    # Append player number to formatted message
                    if player_number_text:
                        response_text = f"{response_text}{player_number_text}"
                else:
                    response_text = f"{member.mention} is now **{actual_character_name}**{player_number_text}!"
                
                if _format_special_reroll_hint:
                    special_hint = _format_special_reroll_hint(actual_character_name, character.folder if character else None)
                    if special_hint:
                        response_text = f"{response_text}\n{special_hint}"
                
                if _get_magic_emoji and ctx.guild:
                    emoji_prefix = _get_magic_emoji(ctx.guild)
                    response_text = f"{emoji_prefix} {response_text}"
                
                # Delete progress message before sending final response
                try:
                    await progress_msg.delete()
                except Exception:
                    pass
                await ctx.reply(response_text, mention_author=False)
            except Exception as msg_exc:
                logger.exception("Error sending assignment transformation message: %s", msg_exc)
                # Get player number for fallback message
                player_number = self._get_player_number(game_state, member.id)
                if player_number:
                    player_number_text = f" - Player {player_number}"
                else:
                    player_number_text = ""
                    logger.warning("No player number found for %s in fallback message", member.display_name)
                # Delete progress message before sending fallback
                try:
                    await progress_msg.delete()
                except Exception:
                    pass
                await ctx.reply(f"✅ {member.mention} is now **{actual_character_name}**{player_number_text}!", mention_author=False)
            
            await self._log_action(game_state, f"Player {member.display_name} added and assigned character: {actual_character_name}")
        else:
            # Delete progress message before sending final response
            try:
                await progress_msg.delete()
            except Exception:
                pass
            await ctx.reply(f"Added {member.mention} to the game.", mention_author=False)
            await self._log_action(game_state, f"Player {member.display_name} added")

    async def command_assign(self, ctx: commands.Context, member: Optional[discord.Member] = None, *, character_name: str = "") -> None:
        """Assign a character to a player (GM only). Supports: !assign @user character OR !assign character_name character OR !assign character_folder character"""
        async def _impl():
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
                
                # If member not provided, try to resolve from character_name (first token)
                resolved_member = member
                character_to_assign = character_name.strip()
                
                if not resolved_member and character_to_assign:
                    # Try to parse: !assign target character
                    tokens = character_to_assign.split(None, 1)  # Split into max 2 parts
                    if len(tokens) >= 1:
                        target_token = tokens[0]
                        if len(tokens) == 2:
                            # Format: !assign target character_to_assign
                            resolved_member = self._resolve_target_member(ctx, game_state, target_token)
                            character_to_assign = tokens[1]
                        else:
                            # Only one token - could be target or character
                            # Try to resolve as target first
                            resolved_member = self._resolve_target_member(ctx, game_state, target_token)
                            if resolved_member:
                                # Successfully resolved as target, but no character specified
                                await ctx.reply("Usage: `!assign @user <character>` or `!assign character_name <character>` or `!assign character_folder <character>`", mention_author=False)
                                return
                            # Not a target, so it's the character name (but we need a target)
                            await ctx.reply("Usage: `!assign @user <character>` or `!assign character_name <character>` or `!assign character_folder <character>`", mention_author=False)
                            return
                
                if not resolved_member:
                    await ctx.reply("Usage: `!assign @user <character>` or `!assign character_name <character>` or `!assign character_folder <character>`", mention_author=False)
                    return
                
                if not character_to_assign or not character_to_assign.strip():
                    await ctx.reply("Usage: `!assign @user <character>` or `!assign character_name <character>` or `!assign character_folder <character>`", mention_author=False)
                    return
                
                character_to_assign = character_to_assign.strip()
                logger.debug("Assigning character %s to member %s", character_to_assign, resolved_member.id)
                
                # Send immediate progress message
                progress_msg = await ctx.reply("⏳ Assigning character...", mention_author=False)
                
                if resolved_member.id not in game_state.players:
                    try:
                        await progress_msg.delete()
                    except Exception:
                        pass
                    await ctx.reply(f"{resolved_member.mention} is not in the game. Add them first with `!addplayer`.", mention_author=False)
                    return
                
                player = game_state.players[resolved_member.id]
                
                # Verify character exists before assigning (uses first name matching like !reroll)
                character = self._get_character_by_name(character_to_assign)
                if not character:
                    # Character not found - show error (no suggestions, just error)
                    try:
                        await progress_msg.delete()
                    except Exception:
                        pass
                    await ctx.reply(f"❌ Unable to locate '{character_to_assign}'.", mention_author=False)
                    logger.warning("Character assignment failed: '%s' not found", character_to_assign)
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
                
                # CRITICAL: Clear any existing game state for this player before creating new one
                # This ensures we don't have stale state interfering with assignment
                if resolved_member.id in game_state.player_states:
                    logger.info("Clearing existing game state for player %s before reassignment", resolved_member.id)
                    del game_state.player_states[resolved_member.id]
                
                state = await self._create_game_state_for_player(
                    player,
                    resolved_member.id,
                    ctx.guild.id,
                    actual_character_name,  # Use the character's actual name, not the lookup parameter
                )
                if not state:
                    # This should never happen if character lookup worked, but handle it anyway
                    try:
                        await progress_msg.delete()
                    except Exception:
                        pass
                    logger.error("CRITICAL: Character found but state creation failed for %s with character %s", resolved_member.id, character_to_assign)
                    await ctx.reply(f"❌ Error: Failed to create state for character '{character_to_assign}'. Assignment not completed.", mention_author=False)
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
                
                # CRITICAL: Always store the new state - this replaces any old state
                game_state.player_states[resolved_member.id] = state
                logger.info("Assigned character '%s' (lookup: '%s') to player %s (user_id=%s). State stored with character_name='%s'", 
                           actual_character_name, character_to_assign, resolved_member.display_name, resolved_member.id, state.character_name)
                
                # Final verification - state MUST match player
                if state.character_name != actual_character_name or player.character_name != actual_character_name:
                    logger.error("CRITICAL: Final verification failed! player.character_name='%s', state.character_name='%s', actual_character_name='%s'", 
                                player.character_name, state.character_name, actual_character_name)
                    # Force everything to match
                    player.character_name = actual_character_name
                    state.character_name = actual_character_name
                    game_state.player_states[resolved_member.id] = state
                    logger.info("Forced correction: player and state now both have character_name='%s'", actual_character_name)
                
                # If GM was assigned as a player, remove narrator role
                if resolved_member.id == game_state.gm_user_id and game_state.narrator_user_id == resolved_member.id:
                    game_state.narrator_user_id = None
                    logger.debug("GM assigned as player, removed narrator role")
                
                # Call pack's on_character_assigned if it exists
                pack = get_game_pack(game_state.game_type, self.packs_dir)
                if pack and pack.has_function("on_character_assigned"):
                    try:
                        pack.call("on_character_assigned", game_state, player, actual_character_name)
                    except KeyError as key_exc:
                        error_key = str(key_exc)
                        logger.error("Game pack error in on_character_assigned: KeyError - key '%s' not found in game data. This usually means the player wasn't properly initialized in the game pack.", error_key)
                        await ctx.reply(f"⚠️ Character assigned, but game pack error: Missing key '{error_key}' in game data. The assignment succeeded, but some game features may not work correctly.", mention_author=False)
                    except Exception as pack_exc:
                        logger.error("Game pack error in on_character_assigned: %s (%s)", type(pack_exc).__name__, pack_exc, exc_info=True)
                        await ctx.reply(f"⚠️ Character assigned, but game pack error: {type(pack_exc).__name__}: {str(pack_exc)}. The assignment succeeded, but some game features may not work correctly.", mention_author=False)
                
                # Update progress message
                try:
                    await progress_msg.edit(content="⏳ Updating board...")
                except Exception:
                    pass
                
                # Update board to show the new token (use !savegame to save manually)
                await self._update_board(game_state, error_channel=ctx.channel)
                
                # Delete progress message before sending final response
                try:
                    await progress_msg.delete()
                except Exception:
                    pass
                
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
                    original_name = member_profile_name(resolved_member)
                    character_message = character.message or ""
                    duration_label = "Game"
                    
                    # Get player number for display (always get it, regardless of formatting method)
                    player_number = self._get_player_number(game_state, resolved_member.id)
                    if player_number:
                        player_number_text = f" - Player {player_number}"
                        logger.info("Assignment message (assign): %s assigned as Player %d", resolved_member.display_name, player_number)
                    else:
                        player_number_text = ""
                        logger.warning("No player number found for %s (user_id=%s) in assign - player may not have been properly added to game", resolved_member.display_name, resolved_member.id)
                    
                    if _format_character_message:
                        response_text = _format_character_message(
                            character_message,
                            original_name,
                            resolved_member.mention,
                            duration_label,
                            actual_character_name,
                        )
                        # Append player number to formatted message
                        if player_number_text:
                            response_text = f"{response_text}{player_number_text}"
                    else:
                        # Fallback if function not available
                        response_text = f"{resolved_member.mention} is now **{actual_character_name}**{player_number_text}!"
                    
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
                    logger.info("Assignment transformation message sent as TEXT for %s as %s (Player %s)", resolved_member.display_name, actual_character_name, player_number or "?")
                    
                except Exception as msg_exc:
                    logger.exception("Error sending assignment transformation message: %s", msg_exc)
                    # Fallback to simple text message
                    # Get player number for fallback message
                    player_number = self._get_player_number(game_state, resolved_member.id)
                    if player_number:
                        player_number_text = f" - Player {player_number}"
                    else:
                        player_number_text = ""
                        logger.warning("No player number found for %s in fallback message", resolved_member.display_name)
                    await ctx.reply(f"✅ {resolved_member.mention} is now **{actual_character_name}**{player_number_text}!", mention_author=False)
                
                await self._log_action(game_state, f"{resolved_member.display_name} assigned character: {character_to_assign}")
                logger.info("Successfully assigned character %s to %s", character_to_assign, resolved_member.id)
            except Exception as exc:
                logger.exception("Error in command_assign: %s", exc)
                # Get a more useful error message
                error_type = type(exc).__name__
                error_msg = str(exc)
                
                # Provide context for common error types
                if error_type == "KeyError":
                    error_msg = f"KeyError: Key '{error_msg}' not found. This usually means the game pack data wasn't properly initialized for this player."
                elif error_type == "AttributeError":
                    error_msg = f"AttributeError: {error_msg}. This usually means a required attribute is missing."
                elif error_type == "TypeError":
                    error_msg = f"TypeError: {error_msg}. This usually means a wrong type was passed to a function."
                # If error message is just a number (like a user ID), provide more context
                elif error_msg.isdigit() or (error_msg and len(error_msg) > 10 and error_msg.replace('-', '').isdigit()):
                    error_msg = f"{error_type}: {error_msg} (this looks like an ID - check if the player was properly initialized)"
                
                if not error_msg or error_msg == "None":
                    error_msg = f"{error_type} occurred"
                
                await ctx.reply(f"❌ Error assigning character: {error_msg}", mention_author=False)
        
        await self._execute_gameboard_command(ctx, _impl)

    async def command_reroll(self, ctx: commands.Context, member: Optional[discord.Member] = None, token: Optional[str] = None) -> None:
        """Randomly reroll a player's character (GM only). Supports: !reroll @user OR !reroll character_name OR !reroll character_folder"""
        async def _impl():
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
            
            # Resolve member from token if provided, otherwise use provided member
            resolved_member = member
            if token and not resolved_member:
                resolved_member = self._resolve_target_member(ctx, game_state, token)
            
            if not resolved_member:
                await ctx.reply("Usage: `!reroll @user` or `!reroll character_name` or `!reroll character_folder`", mention_author=False)
                return
            
            if resolved_member.id not in game_state.players:
                await ctx.reply(f"{resolved_member.mention} is not in the game.", mention_author=False)
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
            old_character = game_state.players[resolved_member.id].character_name
            new_character = random.choice(available_characters)
            
            # CRITICAL: Only modify game state, never global state
            game_state.players[resolved_member.id].character_name = new_character
            
            # Update player state if it exists
            if resolved_member.id in game_state.player_states:
                state = game_state.player_states[resolved_member.id]
                character = self._get_character_by_name(new_character)
                if character:
                    state.character_name = new_character
                    state.character_folder = character.folder
                    state.character_avatar_path = character.avatar_path or ""
                    state.character_message = character.message or ""
                    state.identity_display_name = new_character
            
            # Update board to show new character image
            await self._update_board(game_state, error_channel=ctx.channel)
            
            # Note: Auto-save removed - use !savegame to save manually
            await ctx.reply(
                f"Rerolled {resolved_member.mention}'s character from {old_character or 'none'} to {new_character}.",
                mention_author=False
            )
            await self._log_action(game_state, f"{resolved_member.display_name} rerolled from {old_character} to {new_character}")
        
        await self._execute_gameboard_command(ctx, _impl)

    async def command_swap(self, ctx: commands.Context, member1: Optional[discord.Member] = None, member2: Optional[discord.Member] = None, token1: Optional[str] = None, token2: Optional[str] = None) -> None:
        """Swap characters between two players (GM only). Supports: !swap @user1 @user2 OR !swap character1 character2"""
        async def _impl():
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
            
            # Resolve members from tokens if provided, otherwise use provided members
            resolved_member1 = member1
            resolved_member2 = member2
            
            if token1 and not resolved_member1:
                resolved_member1 = self._resolve_target_member(ctx, game_state, token1)
            if token2 and not resolved_member2:
                resolved_member2 = self._resolve_target_member(ctx, game_state, token2)
            
            if not resolved_member1 or not resolved_member2:
                await ctx.reply("Usage: `!swap @user1 @user2` or `!swap character1 character2`", mention_author=False)
                return
            
            if resolved_member1.id not in game_state.players or resolved_member2.id not in game_state.players:
                await ctx.reply("Both players must be in the game.", mention_author=False)
                return
            
            # Swap characters
            char1 = game_state.players[resolved_member1.id].character_name
            char2 = game_state.players[resolved_member2.id].character_name
            game_state.players[resolved_member1.id].character_name = char2
            game_state.players[resolved_member2.id].character_name = char1
            
            # Swap grid positions (board locations) - true body swap
            pos1 = game_state.players[resolved_member1.id].grid_position
            pos2 = game_state.players[resolved_member2.id].grid_position
            game_state.players[resolved_member1.id].grid_position = pos2
            game_state.players[resolved_member2.id].grid_position = pos1
            
            # Swap TransformationState objects in game_state.player_states (preserve user identity)
            state1 = game_state.player_states.get(resolved_member1.id)
            state2 = game_state.player_states.get(resolved_member2.id)
            
            if state1 and state2:
                # Extract character data from both states
                char1_name = state1.character_name
                char1_folder = state1.character_folder
                char1_avatar = state1.character_avatar_path
                char1_message = state1.character_message
                char1_inanimate = state1.is_inanimate
                char1_responses = state1.inanimate_responses
                
                char2_name = state2.character_name
                char2_folder = state2.character_folder
                char2_avatar = state2.character_avatar_path
                char2_message = state2.character_message
                char2_inanimate = state2.is_inanimate
                char2_responses = state2.inanimate_responses
                
                # Preserve user identity (original_display_name, identity_display_name, etc.)
                identity1 = state1.identity_display_name or state1.character_name
                identity2 = state2.identity_display_name or state2.character_name
                
                # Create new states with swapped character data but preserved user identity
                from tfbot.models import TransformationState
                from datetime import datetime, timezone
                
                new_state1 = TransformationState(
                    user_id=resolved_member1.id,
                    guild_id=state1.guild_id,
                    character_name=char2_name,
                    character_folder=char2_folder,
                    character_avatar_path=char2_avatar,
                    character_message=char2_message,
                    original_nick=state1.original_nick,
                    started_at=state1.started_at,
                    expires_at=state1.expires_at,
                    duration_label=state1.duration_label,
                    avatar_applied=state1.avatar_applied,
                    original_display_name=state1.original_display_name,
                    is_inanimate=char2_inanimate,
                    inanimate_responses=char2_responses,
                    form_owner_user_id=state2.form_owner_user_id or resolved_member2.id,
                    identity_display_name=identity1,  # Keep player's original identity
                    is_pillow=state1.is_pillow,
                )
                
                new_state2 = TransformationState(
                    user_id=resolved_member2.id,
                    guild_id=state2.guild_id,
                    character_name=char1_name,
                    character_folder=char1_folder,
                    character_avatar_path=char1_avatar,
                    character_message=char1_message,
                    original_nick=state2.original_nick,
                    started_at=state2.started_at,
                    expires_at=state2.expires_at,
                    duration_label=state2.duration_label,
                    avatar_applied=state2.avatar_applied,
                    original_display_name=state2.original_display_name,
                    is_inanimate=char1_inanimate,
                    inanimate_responses=char1_responses,
                    form_owner_user_id=state1.form_owner_user_id or resolved_member1.id,
                    identity_display_name=identity2,  # Keep player's original identity
                    is_pillow=state2.is_pillow,
                )
                
                # Update player_states
                game_state.player_states[resolved_member1.id] = new_state1
                game_state.player_states[resolved_member2.id] = new_state2
            
            # Update pack-specific metadata (tile numbers, turn order, etc.)
            self._swap_pack_player_metadata(game_state, resolved_member1.id, resolved_member2.id)
            
            # Update board to show swapped positions and character images
            await self._update_board(game_state, error_channel=ctx.channel)
            
            # Note: Auto-save removed - use !savegame to save manually
            await ctx.reply(f"Swapped characters and positions: {resolved_member1.mention} ↔ {resolved_member2.mention}", mention_author=False)
            await self._log_action(game_state, f"{resolved_member1.display_name} and {resolved_member2.display_name} swapped characters and positions")
        
        await self._execute_gameboard_command(ctx, _impl)

    async def command_movetoken(self, ctx: commands.Context, member: Optional[discord.Member] = None, *, position: str = "") -> None:
        """Move a player's token (GM only). Supports: !movetoken @user <coord> OR !movetoken character_name <coord> OR !movetoken character_folder <coord>"""
        async def _impl():
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
            
            # If member not provided, try to resolve from position (first token)
            resolved_member = member
            position_value = position.strip()
            
            if not resolved_member and position_value:
                # Try to parse: !movetoken target position
                tokens = position_value.split(None, 1)  # Split into max 2 parts
                if len(tokens) == 2:
                    # Format: !movetoken target position
                    target_token = tokens[0]
                    position_value = tokens[1]
                    resolved_member = self._resolve_target_member(ctx, game_state, target_token)
                    if not resolved_member:
                        await ctx.reply(f"Could not find player '{target_token}'. Use `@user`, character name, or character folder.", mention_author=False)
                        return
                else:
                    # Only one token - could be target or position
                    # Try to resolve as target first
                    resolved_member = self._resolve_target_member(ctx, game_state, tokens[0])
                    if resolved_member:
                        # Successfully resolved as target, but no position specified
                        await ctx.reply("Usage: `!movetoken @user <coord>` or `!movetoken character_name <coord>` or `!movetoken character_folder <coord>` (e.g., `!movetoken @user A1`)", mention_author=False)
                        return
                    # Not a target, so it's the position (but we need a target)
                    await ctx.reply("Usage: `!movetoken @user <coord>` or `!movetoken character_name <coord>` or `!movetoken character_folder <coord>` (e.g., `!movetoken @user A1`)", mention_author=False)
                    return
            
            if not resolved_member or not position_value:
                await ctx.reply("Usage: `!movetoken @user <coord>` or `!movetoken character_name <coord>` or `!movetoken character_folder <coord>` (e.g., `!movetoken @user A1`)", mention_author=False)
                return
            
            if resolved_member.id not in game_state.players:
                await ctx.reply(f"{resolved_member.mention} is not in the game.", mention_author=False)
                return
            
            position_value = position_value.strip().upper()
            game_config = self.get_game_config(game_state.game_type)
            if not game_config:
                await ctx.reply("Game configuration not found.", mention_author=False)
                return
            
            # Validate coordinate using core validation
            if not validate_coordinate(position_value, game_config):
                await ctx.reply(f"Invalid coordinate: `{position_value}`. Check bounds and blocked cells.", mention_author=False)
                return
            
            # Validate move using pack-specific validation (if available)
            pack = get_game_pack(game_state.game_type, self.packs_dir)
            if pack and pack.has_function("validate_move"):
                is_valid, error_msg = pack.call("validate_move", game_state, game_state.players[resolved_member.id], position_value, game_config)
                if not is_valid:
                    await ctx.reply(error_msg or f"Invalid move: `{position_value}`", mention_author=False)
                    return
            
            old_pos = game_state.players[resolved_member.id].grid_position
            game_state.players[resolved_member.id].grid_position = position_value
            
            # CRITICAL: Update tile_numbers in game data to match GM movement
            # This ensures GM movement persists through dice rolls
            pack = get_game_pack(game_state.game_type, self.packs_dir)
            if pack and hasattr(pack.module, 'alphanumeric_to_tile_number'):
                # Get the function from the pack module
                alphanumeric_to_tile_number = getattr(pack.module, 'alphanumeric_to_tile_number')
                tile_number = alphanumeric_to_tile_number(position_value, game_config)
                if tile_number is not None:
                    # Update game data tile_numbers - use pack's get_game_data function
                    if hasattr(pack.module, 'get_game_data'):
                        get_game_data = getattr(pack.module, 'get_game_data')
                        data = get_game_data(game_state)
                        data['tile_numbers'][resolved_member.id] = tile_number
                        logger.info("Updated tile_number for player %s to %d (GM movement to %s)", resolved_member.id, tile_number, position_value)
                    else:
                        logger.warning("Pack %s does not have get_game_data function, cannot update tile_numbers", game_state.game_type)
            
            # Check if pack wants board update on move
            should_update = True
            if pack and pack.has_function("should_update_board"):
                should_update = pack.call("should_update_board", game_state, "move")
            
            if should_update:
                await self._update_board(game_state, error_channel=ctx.channel)
            await ctx.reply(f"Moved {resolved_member.mention}'s token from {old_pos} to {position_value}.", mention_author=False)
            await self._log_action(game_state, f"{resolved_member.display_name} token moved to {position_value}")
        
        await self._execute_gameboard_command(ctx, _impl)

    async def command_dice(self, ctx: commands.Context, target_player: Optional[discord.Member] = None) -> None:
        """
        Roll dice (player command).
        
        GM can use: !dice @playername to force a roll for that player (skips turn order).
        """
        async def _impl():
            # CRITICAL: Wrap entire function in try-except to prevent crashes
            try:
                game_state = await self._get_game_state_for_context(ctx)
                if not game_state:
                    await ctx.reply("No active game in this thread.", mention_author=False)
                    return
                
                if game_state.is_locked:
                    await ctx.reply("Game is locked.", mention_author=False)
                    return
                
                # Check if game is paused - block dice commands for everyone (including GM/admin)
                if game_state.is_paused:
                    # Game is paused - delete message silently
                    try:
                        await ctx.message.delete()
                    except discord.HTTPException:
                        pass
                    return
                
                is_gm = self._is_gm(ctx.author, game_state)
                is_gm_override = False
                
                # Check if GM is forcing a roll for another player
                if target_player and is_gm:
                    # GM override - roll for the target player (bypasses game_started check)
                    if target_player.id not in game_state.players:
                        await ctx.reply(f"{target_player.display_name} is not in this game.", mention_author=False)
                        return
                    player = game_state.players[target_player.id]
                    is_gm_override = True
                elif is_gm and not target_player:
                    # GM rolling for themselves - check if they're a player
                    if ctx.author.id in game_state.players:
                        player = game_state.players[ctx.author.id]
                        # GM can roll even if game hasn't started
                        is_gm_override = True
                    else:
                        # GM is not a player - they can't roll
                        await ctx.reply("You're not a player in this game. Use `!dice @player` or `!dice character_name` to roll for a player.", mention_author=False)
                        return
                else:
                    # Normal player roll - check if game has started
                    if not game_state.game_started:
                        await ctx.reply("⏸️ Game hasn't started yet! The GM needs to use `!start` to begin the game.", mention_author=False)
                        return
                    
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
                
                auto_move_requested = False
                turn_complete_requested = False
                summary_msg = None

                # Call pack's on_dice_rolled if it exists
                pack = get_game_pack(game_state.game_type, self.packs_dir)
                if pack and pack.has_function("on_dice_rolled"):
                    try:
                        pack_result = pack.call("on_dice_rolled", game_state, player, total, game_config)
                    except Exception as exc:
                        logger.exception("CRITICAL: Error in pack.on_dice_rolled: %s", exc)
                        try:
                            await ctx.reply(f"❌ Error processing dice roll: {exc}", mention_author=False)
                        except Exception:
                            pass
                        return
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
                            
                            special_notice = pack_msg.strip()
                            if (
                                not auto_move_requested
                                and not turn_complete_requested
                                and not transformation_char
                                and (
                                    special_notice.startswith("You've already rolled this turn! Wait for the turn summary.")
                                    or special_notice.startswith("It's not your turn yet!")
                                    or special_notice.startswith("All players have rolled! Turn summary should be shown.")
                                )
                            ):
                                await ctx.reply(pack_msg, mention_author=False)
                                return
                        
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
                            auto_move_requested = True
                        
                        # Handle turn completion
                        if is_turn_complete:
                            turn_complete_requested = True
                            # Get turn summary from pack
                            if pack and pack.has_function("get_turn_summary"):
                                summary_msg = pack.call("get_turn_summary", game_state, game_config)

                # Get player number from player_numbers dict (based on add order)
                player_number = self._get_player_number(game_state, player.user_id)
                player_number_text = f" (Player {player_number})" if player_number else ""
                
                # Build embed with roll result and player's current board position
                embed_color = discord.Color.random()
                player_position = player.grid_position or "Unknown"
                embed_description = f"{result}\n\n**New Position:** `{player_position}`"
                roll_embed = discord.Embed(
                    title=f"Dice Roll{player_number_text}",
                    description=embed_description,
                    color=embed_color,
                )
                face_file = None
                face_attachment_url = None
                if player.character_name:
                    face_path = _resolve_face_cache_path(player.character_name)
                    if face_path and face_path.exists():
                        face_filename = f"dice_face_{player.user_id}.png"
                        face_file = discord.File(face_path, filename=face_filename)
                        face_attachment_url = f"attachment://{face_filename}"
                        roll_embed.set_thumbnail(url=face_attachment_url)
                        roll_embed.set_author(name=player.character_name, icon_url=face_attachment_url)
                if not face_file:
                    player_member = None
                    if ctx.guild:
                        if isinstance(ctx.channel, discord.Thread):
                            player_member = ctx.channel.guild.get_member(player.user_id)
                        else:
                            player_member = ctx.guild.get_member(player.user_id)
                    if player_member:
                        avatar_url = player_member.display_avatar.url
                        roll_embed.set_author(name=player_member.display_name, icon_url=avatar_url)
                        roll_embed.set_thumbnail(url=avatar_url)
                    else:
                        roll_embed.set_author(name=getattr(ctx.author, "display_name", "Dice Roller"))
                reply_kwargs = {
                    "embed": roll_embed,
                    "mention_author": False,
                }
                if face_file:
                    reply_kwargs["file"] = face_file
                try:
                    await ctx.reply(**reply_kwargs)
                except Exception as exc:
                    logger.exception("CRITICAL: Error sending dice roll reply: %s", exc)
                    return

                # Update board(s) and handle summaries after showing the roll result
                # CRITICAL: Only update board ONCE per dice roll
                # - If turn completes: Update ONCE at turn end (includes movement)
                # - If turn doesn't complete but player moved: Update ONCE for movement
                has_separate_map_thread = bool(
                    game_state.map_thread_id
                    and game_state.game_thread_id
                    and game_state.map_thread_id != game_state.game_thread_id
                )
                
                # Check for win condition and game end (stored in pack data by on_dice_rolled)
                game_ended = False
                has_winners = False
                if pack and hasattr(game_state, '_pack_data'):
                    pack_data = game_state._pack_data
                    game_ended = pack_data.get('game_ended', False)
                    winners = pack_data.get('winners', [])
                    has_winners = len(winners) > 0
                
                # CRITICAL: Only update board ONCE per dice roll
                if turn_complete_requested:
                    # Turn is completing - update board ONCE at end of turn (includes all movement)
                    if summary_msg:
                        await ctx.channel.send(summary_msg, allowed_mentions=discord.AllowedMentions.none())
                    if pack and pack.has_function("advance_turn"):
                        pack.call("advance_turn", game_state)
                    
                    # Increment turn count and auto-save at end of turn
                    game_state.turn_count += 1
                    await self._save_auto_save(game_state, ctx)
                    logger.info("Turn %d completed, auto-save created", game_state.turn_count)
                    
                    # Update board ONCE at end of turn (this includes all movement from the turn)
                    # CRITICAL: Always post to game thread at turn end for visibility
                    also_post_to_game = True  # Always show board in game thread at turn end
                    try:
                        await self._update_board(game_state, error_channel=ctx.channel, target_thread="map", also_post_to_game=also_post_to_game)
                        if game_ended:
                            logger.info("Board updated at end of turn %d - game ended", game_state.turn_count)
                        elif has_winners:
                            logger.info("Board updated at end of turn %d - winner(s) detected", game_state.turn_count)
                        else:
                            logger.info("Board updated at end of turn %d", game_state.turn_count)
                    except Exception as exc:
                        logger.exception("CRITICAL: Error updating board at turn end: %s", exc)
                        try:
                            await ctx.reply("❌ Error updating board. The turn was processed, but the board may not have updated.", mention_author=False)
                        except Exception:
                            pass
                elif auto_move_requested:
                    # Turn not completing but player moved - update board ONCE for movement
                    # CRITICAL: Always post to game thread when player moves (for visibility)
                    target_thread = "map" if has_separate_map_thread else "game"
                    also_post_to_game = True  # Always show board in game thread when player moves
                    try:
                        await self._update_board(game_state, error_channel=ctx.channel, target_thread=target_thread, also_post_to_game=also_post_to_game)
                        logger.info("Board updated after movement (turn not complete)")
                    except Exception as exc:
                        logger.exception("CRITICAL: Error updating board after movement: %s", exc)
                        try:
                            await ctx.reply("❌ Error updating board. The roll was processed, but the board may not have updated.", mention_author=False)
                        except Exception:
                            pass
                
                # After board/turn handling, show next turn info or game over standings
                try:
                    if pack and hasattr(pack, "get_game_data"):
                        data = pack.get_game_data(game_state)
                        game_ended = data.get("game_ended", False)
                        goal_turns = data.get("goal_reached_turn", {}) or {}
                        turn_order = data.get("turn_order", [])
                        player_numbers = data.get("player_numbers", {})
                        turn_order_index = {uid: idx for idx, uid in enumerate(turn_order)}
                        
                        if game_ended:
                            # Build finish order sorted by turn reached, then turn order
                            ordered_finishers = sorted(
                                goal_turns.items(),
                                key=lambda item: (
                                    item[1],
                                    turn_order_index.get(item[0], 10_000_000),
                                    item[0],
                                ),
                            )
                            lines = ["🏁 **Game Over! All players reached the goal.**", "", "**Finish order:**"]
                            for rank, (user_id, turn_num) in enumerate(ordered_finishers, start=1):
                                player_obj = game_state.players.get(user_id)
                                pnum = player_numbers.get(user_id, "?")
                                name = player_obj.character_name if player_obj and player_obj.character_name else f"Player {pnum}"
                                mention = f"<@{user_id}>"
                                lines.append(f"{rank}) {name} (Player {pnum}) {mention} — Turn {turn_num}")
                            await ctx.channel.send("\n".join(lines), allowed_mentions=discord.AllowedMentions.none())
                        else:
                            players_rolled = set(data.get("players_rolled_this_turn", []))
                            # Skip players who already reached the goal (win_tile from rules)
                            rules = game_config.rules or {}
                            win_tile = int(rules.get("win_tile", 100))
                            pending = []
                            for user_id in turn_order:
                                if user_id in players_rolled:
                                    continue
                                player_obj = game_state.players.get(user_id)
                                if not player_obj:
                                    continue
                                tile_num = data.get("tile_numbers", {}).get(user_id, 1)
                                if tile_num >= win_tile:
                                    continue
                                pnum = player_numbers.get(user_id, "?")
                                name = player_obj.character_name or f"Player {pnum}"
                                mention = f"<@{user_id}>"
                                pending.append(f"Player {pnum} - {name} ({mention})")
                            
                            if pending:
                                lines = ["➡️ **Next to roll:**", *pending]
                                await ctx.channel.send("\n".join(lines), allowed_mentions=discord.AllowedMentions.none())
                except Exception as exc:
                    logger.debug("Failed to post next-turn info: %s", exc)
                
                try:
                    await self._log_action(game_state, f"{ctx.author.display_name} rolled {result}")
                except Exception as exc:
                    logger.exception("Error logging dice roll action: %s", exc)
            except Exception as exc:
                # CRITICAL: Catch any unhandled exceptions in the entire function to prevent crashes
                logger.exception("CRITICAL: Unhandled error in command_dice: %s", exc)
                try:
                    await ctx.reply("❌ An unexpected error occurred during the dice roll. The roll may have been processed, but some features may not have worked correctly.", mention_author=False)
                except Exception:
                    pass
        
        # CRITICAL: Use command lock to prevent concurrent execution and ensure proper ordering
        # This ensures player commands don't interrupt GM commands and board updates happen in order
        processed_in_thread = isinstance(ctx.channel, discord.Thread)
        if processed_in_thread:
            thread_id = ctx.channel.id
            command_lock = self._get_command_lock(thread_id)
            
            # Check if lock is already held (GM command is processing)
            if command_lock.locked():
                logger.debug("Blocking player dice command from %s - GM command is processing", ctx.author.id)
                await ctx.reply("⏸️ A command is currently processing. Please wait until it completes and the board is shown.", mention_author=False)
                return
            
            # Acquire lock and execute - this ensures all messages from this command appear in order
            async with command_lock:
                await _impl()
        else:
            # Not in a thread, execute without lock
            await _impl()

        # Process queued messages once the command has fully completed (lock released if used)
        if processed_in_thread:
            game_state = await self._get_game_state_for_context(ctx)
            if game_state:
                await self._process_queued_messages(game_state)

    async def command_start(self, ctx: commands.Context) -> None:
        """Start the game - render board and allow dice rolls (GM only)."""
        async def _impl():
            if not isinstance(ctx.author, discord.Member):
                await ctx.reply("This command can only be used inside a server.", mention_author=False)
                return
            
            game_state = await self._get_game_state_for_context(ctx)
            if not game_state:
                await ctx.reply("No active game in this thread.", mention_author=False)
                return
            
            if not self._is_gm(ctx.author, game_state):
                await ctx.reply("Only the GM can start the game.", mention_author=False)
                return
        
            if game_state.is_locked:
                await ctx.reply("Game is locked and cannot be started.", mention_author=False)
                return
            
            # If game is paused, resume it
            if game_state.is_paused:
                game_state.is_paused = False
                await ctx.reply("✅ Game resumed! Players can now roll dice.", mention_author=False)
                await self._log_action(game_state, f"Game resumed by {ctx.author.display_name}")
                return
            
            # If game already started and not paused, show error
            if game_state.game_started:
                await ctx.reply("Game has already started! Use `!endgame` to end it first.", mention_author=False)
                return
            
            # Check if there are any players with assigned characters
            players_with_characters = [p for p in game_state.players.values() if p.character_name]
            if not players_with_characters:
                await ctx.reply("⚠️ No players have been assigned characters yet. Assign characters before starting the game.", mention_author=False)
                return
        
            # Mark game as started and unpause
            game_state.game_started = True
            game_state.is_paused = False
            
            # Send "Starting game..." message first
            await ctx.reply("Starting game...", mention_author=False)
            
            # Send start message
            player_count = len(players_with_characters)
            player_list = ", ".join([f"**{p.character_name}**" for p in players_with_characters])
            await ctx.reply(
                f"🎮 **Game Started!**\n\n"
                f"Players ready ({player_count}): {player_list}\n\n"
                f"Turn 1 can now begin! Players can use `!dice` to roll.",
                mention_author=False
            )
            
            # Determine if we have separate map thread
            has_separate_map_thread = bool(
                game_state.map_thread_id
                and game_state.game_thread_id
                and game_state.map_thread_id != game_state.game_thread_id
            )
            
            # Always post to map forum, and also post to game thread for visibility at game start
            # Map follows text messages
            await self._update_board(game_state, error_channel=ctx.channel, target_thread="map", also_post_to_game=has_separate_map_thread)
            
            await self._log_action(game_state, f"Game started by {ctx.author.display_name} with {player_count} players")
            logger.info("Game started: %d players ready, turn 1 can begin", player_count)
        
        await self._execute_gameboard_command(ctx, _impl)

    async def command_endgame(self, ctx: commands.Context) -> None:
        """End the current game and lock the thread (GM only)."""
        async def _impl():
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
            # Note: Auto-save removed - use !savegame to save manually
            
            # Lock the thread
            if isinstance(ctx.channel, discord.Thread):
                try:
                    await ctx.channel.edit(locked=True)
                except discord.HTTPException as exc:
                    logger.warning("Failed to lock thread: %s", exc)
            
            await ctx.reply("Game ended. Thread locked.", mention_author=False)
            await self._log_action(game_state, f"Game ended by {ctx.author.display_name}")
        
        await self._execute_gameboard_command(ctx, _impl)

    async def command_removeplayer(self, ctx: commands.Context, member: Optional[discord.Member] = None, token: Optional[str] = None) -> None:
        """Remove a player from the game (GM only). Supports: !removeplayer @user OR !removeplayer character_name OR !removeplayer character_folder"""
        async def _impl():
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
            
            # Resolve member from token if provided, otherwise use provided member
            resolved_member = member
            if token and not resolved_member:
                resolved_member = self._resolve_target_member(ctx, game_state, token)
            
            if not resolved_member:
                await ctx.reply("Usage: `!removeplayer @user` or `!removeplayer character_name` or `!removeplayer character_folder`", mention_author=False)
                return
            
            if resolved_member.id not in game_state.players:
                await ctx.reply(f"{resolved_member.mention} is not in the game.", mention_author=False)
                return
            
            del game_state.players[resolved_member.id]
            # Note: Auto-save removed - use !savegame to save manually
            await ctx.reply(f"Removed {resolved_member.mention} from the game.", mention_author=False)
            await self._log_action(game_state, f"Player {resolved_member.display_name} removed")
        
        await self._execute_gameboard_command(ctx, _impl)

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
        """Set background for a player or all players (GM only). Supports: !bg @user <number> OR !bg character_name <number> OR !bg character_folder <number> OR !bg all <number>"""
        async def _impl():
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
            
            bg_id_str = bg_id.strip()
            if not bg_id_str:
                await self.command_bg_list(ctx)
                return
            
            # Try to parse: !bg target bg_id OR !bg all bg_id
            tokens = bg_id_str.split(None, 1)  # Split into max 2 parts
            resolved_target = target
            bg_id_value = bg_id_str
            
            if not resolved_target and len(tokens) >= 1:
                first_token = tokens[0].lower()
                if first_token == "all":
                    # Format: !bg all <number>
                    resolved_target = None  # None means "all"
                    if len(tokens) == 2:
                        bg_id_value = tokens[1]
                    else:
                        await ctx.reply("Usage: `!bg all <number>`", mention_author=False)
                        return
                elif len(tokens) == 2:
                    # Format: !bg target bg_id
                    target_token = tokens[0]
                    bg_id_value = tokens[1]
                    resolved_target = self._resolve_target_member(ctx, game_state, target_token)
                    if not resolved_target:
                        await ctx.reply(f"Could not find player '{target_token}'. Use `@user`, character name, character folder, or `all`.", mention_author=False)
                        return
            
            try:
                bg_id_int = int(bg_id_value)
            except ValueError:
                await ctx.reply("Background ID must be a number.", mention_author=False)
                return
            
            # Validate background ID exists
            from tfbot.panels import list_background_choices
            choices = list_background_choices()
            if bg_id_int < 1 or bg_id_int > len(choices):
                await ctx.reply(f"Background ID must be between 1 and {len(choices)}.", mention_author=False)
                return
            
            # Check if target is None (meaning "all" was passed)
            if resolved_target is None:
                # Set for all players (game-specific only, doesn't touch global state)
                for player in game_state.players.values():
                    player.background_id = bg_id_int
                await ctx.reply(f"Set background {bg_id_int} for all players (game VN only).", mention_author=False)
                await self._log_action(game_state, f"All players background set to {bg_id_int}")
            elif resolved_target.id in game_state.players:
                game_state.players[resolved_target.id].background_id = bg_id_int
                await ctx.reply(f"Set background {bg_id_int} for {resolved_target.mention} (game VN only).", mention_author=False)
                await self._log_action(game_state, f"{resolved_target.display_name} background set to {bg_id_int}")
            else:
                await ctx.reply(f"{resolved_target.mention} is not in the game.", mention_author=False)
            
            # Note: Auto-save removed - use !savegame to save manually
        
        await self._execute_gameboard_command(ctx, _impl)

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
        """Set outfit for a player (GM only). Supports: !outfit @user <outfit> OR !outfit character_name <outfit> OR !outfit character_folder <outfit>"""
        async def _impl():
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
            
            # If member not provided, try to resolve from outfit_name (first token)
            resolved_member = member
            outfit_to_set = outfit_name.strip()
            
            if not resolved_member and outfit_to_set:
                # Try to parse: !outfit target outfit
                tokens = outfit_to_set.split(None, 1)  # Split into max 2 parts
                if len(tokens) >= 1:
                    target_token = tokens[0]
                    if len(tokens) == 2:
                        # Format: !outfit target outfit_to_set
                        resolved_member = self._resolve_target_member(ctx, game_state, target_token)
                        outfit_to_set = tokens[1]
                    else:
                        # Only one token - could be target or outfit
                        # Try to resolve as target first
                        resolved_member = self._resolve_target_member(ctx, game_state, target_token)
                        if resolved_member:
                            # Successfully resolved as target, but no outfit specified - show list
                            if resolved_member.id not in game_state.players:
                                await ctx.reply(f"{resolved_member.mention} is not in the game.", mention_author=False)
                                return
                            player = game_state.players[resolved_member.id]
                            if not player.character_name:
                                await ctx.reply(f"{resolved_member.mention} doesn't have a character assigned yet.", mention_author=False)
                                return
                            await self.command_outfit_list(ctx, character_name=player.character_name)
                            return
                        # Not a target, so it's the outfit name (but we need a target)
                        await ctx.reply("Usage: `!outfit @user <outfit>` or `!outfit character_name <outfit>` or `!outfit character_folder <outfit>`", mention_author=False)
                        return
            
            if not resolved_member:
                # Show list of current players
                await self.command_outfit_list(ctx)
                return
            
            if not outfit_to_set:
                # Show outfits for this player's character
                if resolved_member.id not in game_state.players:
                    await ctx.reply(f"{resolved_member.mention} is not in the game.", mention_author=False)
                    return
                player = game_state.players[resolved_member.id]
                if not player.character_name:
                    await ctx.reply(f"{resolved_member.mention} doesn't have a character assigned yet.", mention_author=False)
                    return
                await self.command_outfit_list(ctx, character_name=player.character_name)
                return
            
            if resolved_member.id not in game_state.players:
                await ctx.reply(f"{resolved_member.mention} is not in the game.", mention_author=False)
                return
            
            player = game_state.players[resolved_member.id]
            if not player.character_name:
                await ctx.reply(f"{resolved_member.mention} doesn't have a character assigned yet. Use `!assign` first.", mention_author=False)
                return
            
            # Set outfit (game-specific only, doesn't touch global vn_outfit_selection)
            game_state.players[resolved_member.id].outfit_name = outfit_to_set.strip()
            # Note: Auto-save removed - use !savegame to save manually
            await ctx.reply(f"Set outfit '{outfit_to_set}' for {resolved_member.mention} (game VN only).", mention_author=False)
            await self._log_action(game_state, f"{resolved_member.display_name} outfit set to {outfit_to_set}")
        
        await self._execute_gameboard_command(ctx, _impl)

    async def command_savegame(self, ctx: commands.Context) -> None:
        """Save the current game state (GM only)."""
        async def _impl():
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
        
        await self._execute_gameboard_command(ctx, _impl)

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
                debug_mode=bool(data.get("debug_mode", False)),
                game_started=bool(data.get("game_started", False)),  # Default to False for old saves
                is_paused=bool(data.get("is_paused", False)),  # Default to False for old saves
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
        rules_text += "• Land on snakes to slide down, ladders to climb up\n\n"
        
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
        
        # Starting position
        starting_tile = rules.get("starting_tile", 1)
        starting_pos = rules.get("starting_position", "A1")
        rules_text += f"**Starting Position:** Tile {starting_tile} ({starting_pos})\n"
        
        # Ensure under 2000 characters
        if len(rules_text) > 1950:
            rules_text = rules_text[:1950] + "\n\n*(Rules truncated - ask GM for full details)*"
        
        await ctx.reply(rules_text, mention_author=False)

    async def command_pause(self, ctx: commands.Context) -> None:
        """Pause the game - blocks dice rolls (GM only)."""
        async def _impl():
            if not isinstance(ctx.author, discord.Member):
                await ctx.reply("This command can only be used inside a server.", mention_author=False)
                return
            
            game_state = await self._get_game_state_for_context(ctx)
            if not game_state:
                await ctx.reply("No active game in this thread.", mention_author=False)
                return
            
            if not self._is_gm(ctx.author, game_state):
                await ctx.reply("Only the GM can pause the game.", mention_author=False)
                return
            
            if game_state.is_locked:
                await ctx.reply("Game is locked and cannot be paused.", mention_author=False)
                return
            
            if not game_state.game_started:
                await ctx.reply("Game hasn't started yet. Use `!start` to begin the game.", mention_author=False)
                return
            
            if game_state.is_paused:
                await ctx.reply("Game is already paused.", mention_author=False)
                return
            
            game_state.is_paused = True
            await ctx.reply("⏸️ Game paused. Dice rolls are blocked until resumed.", mention_author=False)
            await self._log_action(game_state, "Game paused")
        
        await self._execute_gameboard_command(ctx, _impl)
    
    async def command_resume(self, ctx: commands.Context) -> None:
        """Resume the game - allows dice rolls again (GM only)."""
        async def _impl():
            if not isinstance(ctx.author, discord.Member):
                await ctx.reply("This command can only be used inside a server.", mention_author=False)
                return
            
            game_state = await self._get_game_state_for_context(ctx)
            if not game_state:
                await ctx.reply("No active game in this thread.", mention_author=False)
                return
            
            if not self._is_gm(ctx.author, game_state):
                await ctx.reply("Only the GM can resume the game.", mention_author=False)
                return
            
            if game_state.is_locked:
                await ctx.reply("Game is locked and cannot be resumed.", mention_author=False)
                return
            
            if not game_state.is_paused:
                await ctx.reply("Game is not paused.", mention_author=False)
                return
            
            game_state.is_paused = False
            await ctx.reply("▶️ Game resumed. Dice rolls are now allowed.", mention_author=False)
            await self._log_action(game_state, "Game resumed")
        
        await self._execute_gameboard_command(ctx, _impl)

    async def command_debug(self, ctx: commands.Context) -> None:
        """Toggle debug mode on/off (Admin & GM only). Shows coordinate labels on board."""
        async def _impl():
            if not isinstance(ctx.author, discord.Member):
                await ctx.reply("This command can only be used inside a server.", mention_author=False)
                return
            
            game_state = await self._get_game_state_for_context(ctx)
            if not game_state:
                await ctx.reply("No active game in this thread.", mention_author=False)
                return
            
            # Check if user is admin or GM
            is_admin_user = is_admin(ctx.author)
            is_gm = self._is_gm(ctx.author, game_state)
            
            if not (is_admin_user or is_gm):
                await ctx.reply("Only admins and GMs can toggle debug mode.", mention_author=False)
                return
            
            # Toggle debug mode
            game_state.debug_mode = not game_state.debug_mode
            # Note: Auto-save removed - use !savegame to save manually
            
            status = "ON" if game_state.debug_mode else "OFF"
            await ctx.reply(f"Debug mode is now **{status}**. Board will show coordinate labels when debug is enabled.", mention_author=False)
            
            # Update board to show/hide debug layer
            await self._update_board(game_state, error_channel=ctx.channel)
            await self._log_action(game_state, f"Debug mode toggled {status} by {ctx.author.display_name}")
        
        await self._execute_gameboard_command(ctx, _impl)

    async def command_transfergm(self, ctx: commands.Context, member: Optional[discord.Member] = None) -> None:
        """Transfer GM role to another user (current GM only)."""
        async def _impl():
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
            
            # Note: Auto-save removed - use !savegame to save manually
            await ctx.reply(f"GM role transferred to {member.mention}.", mention_author=False)
            await self._log_action(game_state, f"GM role transferred from {ctx.author.display_name} to {member.display_name}")
        
        await self._execute_gameboard_command(ctx, _impl)

    async def command_help(self, ctx: commands.Context) -> None:
        """Show available player commands."""
        game_state = await self._get_game_state_for_context(ctx)
        if not game_state:
            await ctx.reply("No active game in this thread.", mention_author=False)
            return
        
        help_text = """**Player Commands:**
`!dice` - Roll dice (follows turn order)
`!dice @player` or `!dice character_name` - GM can force a roll
`!rules` - Show game rules
`!help` - Show this help

**Game Management (GM Only):**
`!startgame <game_type>` - Start new game (Admin only)
`!start` - Begin game (also resumes if paused)
`!pause` - Pause game (blocks dice rolls)
`!resume` - Resume game (allows dice rolls)
`!endgame` - End game and lock thread
`!transfergm @user` - Transfer GM role
`!listgames` - List available games

**Player Management (GM Only):**
`!addplayer @user [character]` - Add player
`!removeplayer @user` or `character_name` - Remove player
`!assign @user <character>` or `character_name <character>` - Assign character
`!reroll @user` or `character_name` - Reroll character
`!swap @user1 @user2` or `character1 character2` - Swap characters & positions

**Token Movement (GM Only):**
`!movetoken @user <coord>` or `character_name <coord>` - Move token (e.g., A1)

**Visual Customization (GM Only):**
`!bg @user <bg_id>` or `all <bg_id>` - Set background
`!bg_list` - List backgrounds
`!outfit @user <outfit>` - Set outfit
`!outfit_list [character]` - List outfits

**Save/Load (GM Only):**
`!savegame` - Save game state
`!loadgame <file>` - Load game state (Admin only)

**Debug (GM/Admin):**
`!debug` - Toggle debug mode

**Notes:**
- Commands are case-insensitive
- GM commands support @user, character names, or folders
- `!swap` swaps characters AND board positions
- When paused, `!dice` deleted silently (GM can force rolls)"""
        
        await ctx.reply(help_text, mention_author=False)
