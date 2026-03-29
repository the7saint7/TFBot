"""Game board framework for managing board games with visual boards and VN chat."""

from __future__ import annotations

import asyncio
import aiohttp
import io
import json
import logging
import os
import time
import random
import secrets
from collections import deque
from datetime import datetime, timedelta

# SystemRandom instance for statistically accurate dice rolls
_dice_rng = random.SystemRandom()
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import discord
from discord.ext import commands

from .game_models import GameConfig, GamePlayer, GameState
from .game_board import render_game_board, validate_coordinate, _resolve_face_cache_path
from .panel_executor import run_panel_render_gif, run_panel_render_vn
from .game_pack_loader import get_game_pack
from .utils import get_channel_id, is_admin, is_bot_mod, int_from_env, path_from_env
from .models import TransformationState, TFCharacter
from .swaps import ensure_form_owner
from .state import serialize_state, deserialize_state
from .animation_perf_log import log_event as log_animation_perf_event
from tfbot.transition_constants import (
    GIF_COLORS,
    GIF_DITHER_MODE,
    GIF_SHARED_PALETTE,
    GIF_TARGET_BYTES,
    TRANSITION_FALLBACK_FORMAT,
    TRANSITION_PRIMARY_FORMAT,
    WEBP_METHOD,
    WEBP_QUALITY,
    WEBP_TARGET_BYTES,
)

logger = logging.getLogger("tfbot.games")
_SEND_TIMINGS: deque[float] = deque(maxlen=200)
_SEND_COUNT = 0


def _send_pctl(values: deque[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[idx]


async def _timed_send(label: str, awaitable):
    global _SEND_COUNT
    started = time.perf_counter()
    result = await awaitable
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    _SEND_TIMINGS.append(elapsed_ms)
    _SEND_COUNT += 1
    if _SEND_COUNT % 25 == 0:
        logger.info(
            "games-send n=%d upload p50=%.0fms p95=%.0fms",
            _SEND_COUNT,
            _send_pctl(_SEND_TIMINGS, 0.5),
            _send_pctl(_SEND_TIMINGS, 0.95),
        )
    if elapsed_ms >= 3000:
        logger.warning("games-send slow label=%s upload=%.0fms", label, elapsed_ms)
    return result


async def _timed_send_ms(label: str, awaitable):
    started = time.perf_counter()
    result = await _timed_send(label, awaitable)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return result, elapsed_ms


def _discord_file_size_bytes(file_obj) -> Optional[int]:
    if file_obj is None:
        return None
    fp = getattr(file_obj, "fp", None)
    if fp is None:
        return None
    try:
        if hasattr(fp, "getvalue"):
            data = fp.getvalue()
            if isinstance(data, (bytes, bytearray)):
                return len(data)
        pos = fp.tell() if hasattr(fp, "tell") else None
        if hasattr(fp, "seek"):
            fp.seek(0)
        data = fp.read()
        if pos is not None and hasattr(fp, "seek"):
            fp.seek(pos)
        if isinstance(data, (bytes, bytearray)):
            return len(data)
    except Exception:
        return None
    return None


def _log_transition_send_metrics(
    *,
    label: str,
    render_ms: float,
    upload_ms: float,
    total_ms: float,
    payload_bytes: Optional[int],
) -> None:
    logger.info(
        "game-transition-send label=%s bytes=%s render=%.0fms upload=%.0fms total=%.0fms",
        label,
        payload_bytes if payload_bytes is not None else "unknown",
        render_ms,
        upload_ms,
        total_ms,
    )
    log_animation_perf_event(
        "game_transition_send",
        label=label,
        bytes=payload_bytes if payload_bytes is not None else "unknown",
        render_ms=f"{render_ms:.0f}",
        upload_ms=f"{upload_ms:.0f}",
        total_ms=f"{total_ms:.0f}",
        format=TRANSITION_PRIMARY_FORMAT,
        fallback_format=TRANSITION_FALLBACK_FORMAT,
        webp_quality=str(WEBP_QUALITY),
        webp_method=str(WEBP_METHOD),
        webp_target_bytes=str(WEBP_TARGET_BYTES),
        gif_colors=str(GIF_COLORS),
        gif_dither=GIF_DITHER_MODE,
        gif_shared_palette="1" if GIF_SHARED_PALETTE else "0",
        gif_target_bytes=str(GIF_TARGET_BYTES),
    )


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
        
        # Read TFBOT_TEST flag (same logic as bot.py: not defined or invalid → LIVE)
        TFBOT_TEST_RAW = os.getenv("TFBOT_TEST", "").strip().upper()
        if not TFBOT_TEST_RAW:
            test_mode: Optional[bool] = False  # Not defined → LIVE
        elif TFBOT_TEST_RAW in ("YES", "TRUE", "1", "ON"):
            test_mode = True  # TEST mode
        elif TFBOT_TEST_RAW in ("NO", "FALSE", "0", "OFF"):
            test_mode = False  # LIVE mode
        else:
            test_mode = False  # Invalid value → LIVE
        
        # Use environment variable if available, otherwise use config file
        env_forum_id = get_channel_id("TFBOT_GAME_FORUM_CHANNEL_ID", 0, test_mode)
        self.forum_channel_id = env_forum_id if env_forum_id > 0 else int(self._config.get("forum_channel_id", 0))
        env_map_forum_id = get_channel_id("TFBOT_GAME_MAP_FORUM_CHANNEL_ID", 0, test_mode)
        self.map_forum_channel_id = env_map_forum_id if env_map_forum_id > 0 else int(self._config.get("map_forum_channel_id", 0))
        # If map forum not configured, fall back to game forum (backwards compatibility)
        if self.map_forum_channel_id == 0:
            self.map_forum_channel_id = self.forum_channel_id
        env_dm_id = get_channel_id("TFBOT_GAME_DM_CHANNEL_ID", 0, test_mode)
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
        self._players_command_cooldowns: Dict[Tuple[int, int], float] = {}  # (thread_id, user_id) -> last_used
        
        # States directory - save in bot folder/vn_states/games
        # Path from tfbot/games.py -> tfbot/ -> TFBot/ -> vn_states/games
        bot_root = Path(__file__).parent.parent  # Go up from tfbot/games.py to TFBot/
        self.states_dir = bot_root / "vn_states" / "games"
        self.states_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Game states directory: %s", self.states_dir.absolute())
        
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
        
        # Load only the newest autosave per game (mtime-based, no filename changes)
        autosave_files = []
        for candidate in self.states_dir.glob("save_*_autosave*.json"):
            if candidate.is_file():
                autosave_files.append(candidate)

        if not autosave_files:
            return

        def _parse_save_game_number(path: Path) -> Optional[int]:
            name = path.name
            if not name.startswith("save_"):
                return None
            try:
                num_part = name.split("_", 2)[1]
                return int(num_part)
            except (ValueError, IndexError):
                return None

        newest_by_game: Dict[int, Path] = {}
        for path in autosave_files:
            game_num = _parse_save_game_number(path)
            if game_num is None:
                continue
            existing = newest_by_game.get(game_num)
            if not existing:
                newest_by_game[game_num] = path
                continue
            if path.stat().st_mtime > existing.stat().st_mtime:
                newest_by_game[game_num] = path

        for state_file in newest_by_game.values():
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
                
                # Always re-read enabled_packs from current config (ignore saved value)
                # This ensures games use current pack configuration, not outdated saved values
                game_type = str(data.get("game_type", ""))
                enabled_packs = None
                if game_type:
                    from tf_characters import get_enabled_packs_for_game, BOT_NAME
                    enabled_packs = get_enabled_packs_for_game(game_type, BOT_NAME)
                    saved_enabled_packs = data.get("enabled_packs")
                    if saved_enabled_packs and set(saved_enabled_packs) != enabled_packs:
                        logger.info("Game %s: enabled_packs updated from saved %s to current config %s", 
                                   thread_id, saved_enabled_packs, sorted(enabled_packs))
                
                game_state = GameState(
                    game_thread_id=thread_id,
                    forum_channel_id=int(data.get("forum_channel_id", 0)),
                    dm_channel_id=int(data.get("dm_channel_id", 0)),
                    gm_user_id=int(data.get("gm_user_id", 0)),
                    game_type=game_type,
                    players=players,
                    current_turn=data.get("current_turn"),
                    board_message_id=data.get("board_message_id"),
                    is_locked=bool(data.get("is_locked", False)),
                    narrator_user_id=data.get("narrator_user_id"),
                    debug_mode=bool(data.get("debug_mode", False)),
                    turn_count=int(data.get("turn_count", 0)),
                    game_started=bool(data.get("game_started", False)),  # Default to False for old saves
                    is_paused=bool(data.get("is_paused", False)),  # Default to False for old saves
                    bot_user_id=data.get("bot_user_id"),  # Load bot ownership (None for old saves)
                    enabled_packs=enabled_packs,  # Restore saved enabled packs (or from config for old saves)
                    map_thread_id=data.get("map_thread_id"),  # Restore so map thread receives board updates after reload
                )
                
                # ADD pack_data restoration if it exists
                if "pack_data" in data and data["pack_data"]:
                    game_state._pack_data = data["pack_data"]
                    
                    # CRITICAL: Sync grid_position from tile_numbers after loading pack_data
                    # This ensures board positions are restored correctly
                    if game_state._pack_data and "tile_numbers" in game_state._pack_data:
                        pack = get_game_pack(game_state.game_type, self.packs_dir)
                        if pack and hasattr(pack.module, 'tile_number_to_alphanumeric'):
                            game_config = self._game_configs.get(game_state.game_type)
                            if game_config:
                                tile_number_to_alphanumeric = getattr(pack.module, 'tile_number_to_alphanumeric')
                                for user_id_str, tile_num in game_state._pack_data["tile_numbers"].items():
                                    try:
                                        user_id = int(user_id_str)
                                        if user_id in game_state.players:
                                            new_pos = tile_number_to_alphanumeric(tile_num, game_config)
                                            if new_pos:
                                                game_state.players[user_id].grid_position = new_pos
                                                logger.debug("Synced grid_position for player %s: tile %s -> %s", user_id, tile_num, new_pos)
                                    except (ValueError, TypeError):
                                        continue
                
                # CRITICAL: Sanitize pack_data after loading (fixes #10, #14, #15)
                # Add None/empty checks first (handles old save files without pack_data)
                if not hasattr(game_state, '_pack_data') or not game_state._pack_data:
                    # Initialize empty pack_data if missing (get_game_data will handle defaults)
                    pack = get_game_pack(game_state.game_type, self.packs_dir)
                    if pack and pack.has_function("get_game_data"):
                        try:
                            game_state._pack_data = pack.call("get_game_data", game_state)
                        except Exception as exc:
                            logger.warning("Failed to call pack.get_game_data during load: %s", exc)
                            # Fallback to minimal structure
                            game_state._pack_data = {
                                'tile_numbers': {},
                                'turn_order': [],
                                'player_numbers': {},
                                'players_rolled_this_turn': [],
                                'winners': [],
                                'forfeited_players': [],
                            }
                    else:
                        # Fallback: create minimal structure
                        game_state._pack_data = {
                            'tile_numbers': {},
                            'turn_order': [],
                            'player_numbers': {},
                            'players_rolled_this_turn': [],
                            'winners': [],
                            'forfeited_players': [],
                        }
                
                # Verify pack_data is a dict (defensive programming)
                if not isinstance(game_state._pack_data, dict):
                    logger.warning("Invalid pack_data type in save file, initializing new dict")
                    game_state._pack_data = {}
                
                # Now sanitize all fields:
                if game_state._pack_data:
                    # Sanitize turn_order: convert to int, deduplicate, filter existing players
                    if "turn_order" in game_state._pack_data:
                        # Convert all to int, deduplicate, filter
                        seen = set()
                        turn_order_clean = []
                        for uid in game_state._pack_data['turn_order']:
                            try:
                                uid_int = int(uid) if isinstance(uid, str) else uid
                                if uid_int in game_state.players and uid_int not in seen:
                                    turn_order_clean.append(uid_int)
                                    seen.add(uid_int)
                            except (ValueError, TypeError) as exc:
                                logger.debug("Failed to sanitize turn_order entry %s: %s", uid, exc)
                                continue
                        game_state._pack_data['turn_order'] = turn_order_clean
                    
                    # Sanitize player_numbers: convert keys to int, filter existing players
                    if "player_numbers" in game_state._pack_data:
                        player_numbers_clean = {}
                        for uid_str, num in game_state._pack_data['player_numbers'].items():
                            try:
                                uid_int = int(uid_str) if isinstance(uid_str, str) else uid_str
                                if uid_int in game_state.players:
                                    player_numbers_clean[uid_int] = num
                            except (ValueError, TypeError) as exc:
                                logger.debug("Failed to sanitize player_numbers entry %s: %s", uid_str, exc)
                                continue
                        game_state._pack_data['player_numbers'] = player_numbers_clean
                    
                    # Clean up tile_numbers: remove entries for non-existent players (for consistency)
                    if "tile_numbers" in game_state._pack_data:
                        tile_numbers_clean = {}
                        for uid_str, tile_num in game_state._pack_data['tile_numbers'].items():
                            try:
                                uid_int = int(uid_str) if isinstance(uid_str, str) else uid_str
                                if uid_int in game_state.players:
                                    tile_numbers_clean[uid_int] = tile_num
                            except (ValueError, TypeError) as exc:
                                logger.debug("Failed to sanitize tile_numbers entry %s: %s", uid_str, exc)
                                continue
                        game_state._pack_data['tile_numbers'] = tile_numbers_clean
                    
                    # Deduplicate other lists (defensive programming)
                    for list_key in ['winners', 'forfeited_players', 'players_rolled_this_turn', 'players_reached_end_this_turn']:
                        if list_key in game_state._pack_data and isinstance(game_state._pack_data[list_key], list):
                            try:
                                game_state._pack_data[list_key] = list(dict.fromkeys(
                                    [int(uid) if isinstance(uid, str) else uid for uid in game_state._pack_data[list_key]]
                                ))
                            except (ValueError, TypeError) as exc:
                                logger.debug("Failed to sanitize %s list: %s", list_key, exc)
                                # Keep original list if sanitization fails (better than losing data)
                                continue
                    
                    # Log warning if turn_order becomes empty after sanitization (game might be invalid)
                    if game_state._pack_data.get('turn_order') == [] and game_state.players:
                        logger.warning("turn_order is empty after sanitization but players exist - game state may be corrupted")

                # Apply default background for active players missing one (skip forfeited/removed)
                forfeited_players = set()
                if isinstance(game_state._pack_data, dict):
                    raw_forfeited = game_state._pack_data.get("forfeited_players", [])
                    if isinstance(raw_forfeited, list):
                        for uid in raw_forfeited:
                            try:
                                forfeited_players.add(int(uid) if isinstance(uid, str) else uid)
                            except (ValueError, TypeError):
                                continue
                for user_id, player in game_state.players.items():
                    if player.background_id is None and user_id not in forfeited_players:
                        player.background_id = 415
                
                # RESTORE player_states from saved data if they exist
                if "player_states" in data and data["player_states"]:
                    try:
                        for user_id_str, state_dict in data["player_states"].items():
                            try:
                                user_id = int(user_id_str)
                                # Use deserialize_state to recreate TransformationState
                                restored_state = deserialize_state(state_dict)
                                game_state.player_states[user_id] = restored_state
                                logger.debug("Restored player state for user %s from saved data", user_id)
                            except (ValueError, KeyError, TypeError) as exc:
                                logger.warning("Failed to restore player state for user %s: %s", user_id_str, exc)
                    except Exception as exc:
                        logger.warning("Error restoring player_states: %s", exc)
                
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

    def _extract_game_number(self, game_state: GameState) -> Optional[int]:
        """Extract game number from thread name pattern: 'Game Name #123 - GM Name'."""
        try:
            thread = self.bot.get_channel(game_state.game_thread_id)
            if not isinstance(thread, discord.Thread) or not thread.name:
                return None
            
            # Parse: split by "#", take part after "#", split by " - ", convert to int
            if "#" in thread.name:
                parts = thread.name.split("#", 1)
                if len(parts) > 1:
                    num_part = parts[1].split(" - ", 1)[0].strip()
                    return int(num_part)
        except (ValueError, IndexError, AttributeError) as exc:
            logger.debug("Failed to extract game number from thread name: %s", exc)
        
        return None

    def _get_next_autosave_number(self, game_state: GameState, date_str: str) -> int:
        """Get next auto-save number (1-3) cycling through existing saves."""
        game_number = self._extract_game_number(game_state)
        if game_number is None:
            # Fallback: use thread_id if game number extraction fails
            game_number = game_state.game_thread_id
        
        # Find existing auto-saves for this game and date
        pattern = f"save_{game_number}_{date_str}_autosave*.json"
        existing_numbers = set()
        
        for save_file in self.states_dir.glob(pattern):
            try:
                # Extract number from filename: save_{gamenumber}_{date}_autosave{number}.json
                name = save_file.stem  # filename without .json
                if f"_autosave" in name:
                    num_part = name.split("_autosave", 1)[1]
                    if num_part.isdigit():
                        existing_numbers.add(int(num_part))
            except (ValueError, IndexError):
                continue
        
        # Return next in cycle: if all exist (1,2,3), return 1 (overwrite oldest)
        # Otherwise return lowest missing number, or 1 if none exist
        if len(existing_numbers) == 3:
            return 1  # Cycle back to 1
        elif not existing_numbers:
            return 1
        else:
            # Find lowest missing number from 1-3
            for num in [1, 2, 3]:
                if num not in existing_numbers:
                    return num
            return 1  # Fallback

    def _get_next_manualsave_number(self, game_state: GameState, date_str: str) -> int:
        """Get next manual save number (increments indefinitely)."""
        game_number = self._extract_game_number(game_state)
        if game_number is None:
            # Fallback: use thread_id if game number extraction fails
            game_number = game_state.game_thread_id
        
        # Find existing manual saves for this game and date
        pattern = f"save_{game_number}_{date_str}_manualsave*.json"
        existing_numbers = []
        
        for save_file in self.states_dir.glob(pattern):
            try:
                # Extract number from filename: save_{gamenumber}_{date}_manualsave{number}_turn*.json
                name = save_file.stem  # filename without .json
                if "_manualsave" in name:
                    num_part = name.split("_manualsave", 1)[1].split("_turn", 1)[0]
                    if num_part.isdigit():
                        existing_numbers.append(int(num_part))
            except (ValueError, IndexError):
                continue
        
        # Return highest number + 1, or 1 if none exist
        if not existing_numbers:
            return 1
        return max(existing_numbers) + 1

    async def _delete_game_saves(self, game_state: GameState) -> None:
        """Delete all save files for a specific game."""
        game_number = self._extract_game_number(game_state)
        if game_number is None:
            # Fallback: use thread_id if game number extraction fails
            game_number = game_state.game_thread_id
        
        # Delete all files matching: save_{gamenumber}_*.json
        pattern = f"save_{game_number}_*.json"
        deleted_count = 0
        
        for save_file in self.states_dir.glob(pattern):
            try:
                save_file.unlink()
                deleted_count += 1
                logger.info("Deleted game save: %s", save_file.name)
            except Exception as exc:
                logger.warning("Failed to delete save file %s: %s", save_file.name, exc)
        
        if deleted_count > 0:
            logger.info("Deleted %d save file(s) for game %d", deleted_count, game_number)

    def is_game_thread(self, channel: Optional[discord.abc.GuildChannel]) -> bool:
        """Check if a channel is a game thread (active or by name pattern)."""
        # If no packs available, gameboard is disabled - no threads are game threads
        if not self._has_packs:
            return False
        
        if not isinstance(channel, discord.Thread):
            return False
        
        # CRITICAL: Check if thread's parent forum channel matches this bot's configured forum
        # This prevents multiple bot instances from processing each other's threads
        if self.forum_channel_id > 0:
            parent_forum = getattr(channel, 'parent', None)
            if parent_forum and hasattr(parent_forum, 'id'):
                if parent_forum.id != self.forum_channel_id:
                    # Thread belongs to a different forum - not this bot's thread
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
                    bot_user_id=data.get("bot_user_id"),  # Load bot ownership (None for old saves)
                    )

                    # Apply default background for active players missing one (skip forfeited/removed)
                    forfeited_players = set()
                    pack_data = data.get("pack_data")
                    if isinstance(pack_data, dict):
                        raw_forfeited = pack_data.get("forfeited_players", [])
                        if isinstance(raw_forfeited, list):
                            for uid in raw_forfeited:
                                try:
                                    forfeited_players.add(int(uid) if isinstance(uid, str) else uid)
                                except (ValueError, TypeError):
                                    continue
                    for user_id, player in players.items():
                        if player.background_id is None and user_id not in forfeited_players:
                            player.background_id = 415
                    
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
                                    game_state=game_state,
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
        
        # Capture enabled packs from tf_characters.json (single source of truth)
        from tf_characters import get_enabled_packs_for_game, BOT_NAME
        enabled_packs = get_enabled_packs_for_game(matched_game_type, BOT_NAME)
        
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
            enabled_packs=enabled_packs,  # Capture enabled packs at game creation (frozen for this game)
        )
        
        self._active_games[thread.id] = game_state
        # Note: Auto-save removed - use !savegame to save manually
        logger.info("Created new game state for existing thread: thread_id=%s, game_type=%s, gm=%s", thread.id, matched_game_type, gm_user_id)
        return game_state

    def _get_filtered_character_pool(self, game_type: str, enabled_packs: Optional[Set[str]] = None) -> List[TFCharacter]:
        """
        Get filtered character pool for a specific game type.
        Only used in gameboard mode - filters characters based on pack enable flags.
        
        Args:
            game_type: Game type identifier (e.g., "snakes_ladders")
            enabled_packs: Optional set of pack names (from game_state.enabled_packs)
        
        Returns:
            List of TFCharacter objects from enabled packs
        """
        try:
            # Import filtering functions
            from tf_characters import get_filtered_characters_for_game, BOT_NAME
            
            # Get filtered character dicts (pass enabled_packs if provided from saved game state)
            filtered_chars = get_filtered_characters_for_game(game_type, BOT_NAME, enabled_packs=enabled_packs)
            
            if not filtered_chars:
                logger.debug("No characters enabled for game %s", game_type)
                return []
            
            # Get avatar root from bot module
            import sys
            bot_module = sys.modules.get('bot') or sys.modules.get('__main__')
            if not bot_module:
                try:
                    import bot as bot_module
                except ImportError:
                    pass
            
            avatar_root = None
            if bot_module:
                _resolve_avatar_root = getattr(bot_module, '_resolve_avatar_root', None)
                if _resolve_avatar_root:
                    avatar_root = _resolve_avatar_root()
            
            # Build TFCharacter objects from filtered dicts (matching _build_character_pool logic)
            pool: List[TFCharacter] = []
            for entry in filtered_chars:
                try:
                    avatar_path = str(entry.get("avatar_path", "")).strip()
                    if (
                        avatar_root
                        and avatar_path
                        and not avatar_path.startswith(("http://", "https://"))
                    ):
                        candidate = Path(avatar_path)
                        if not candidate.is_absolute():
                            avatar_path = str((avatar_root / candidate).resolve())
                        else:
                            avatar_path = str(candidate)
                    folder_name = str(entry.get("folder", "")).strip()
                    if not folder_name or folder_name.upper() == "TODO":
                        logger.debug("Skipping character %s: missing folder assignment.", entry.get("name", "Unnamed"))
                        continue
                    
                    # Replace {BOT_NAME} placeholder in messages
                    message_text = str(entry.get("message", ""))
                    if "{BOT_NAME}" in message_text:
                        message_text = message_text.replace("{BOT_NAME}", BOT_NAME)
                    
                    # Fix common encoding issues
                    encoding_fixes = {
                        'â€™': "'",
                        'â€œ': '"',
                        'â€': '"',
                        'â€"': '—',
                        'â€"': '–',
                        'â€¦': '…',
                    }
                    for wrong, correct in encoding_fixes.items():
                        message_text = message_text.replace(wrong, correct)
                    
                    pool.append(
                        TFCharacter(
                            name=entry["name"],
                            avatar_path=avatar_path,
                            message=message_text,
                            folder=folder_name,
                        )
                    )
                except KeyError as exc:
                    logger.warning("Skipping character entry missing %s", exc)
            
            logger.debug("Built filtered character pool: %d characters for game %s", len(pool), game_type)
            return pool
            
        except Exception as exc:
            logger.warning("Failed to build filtered character pool for game %s: %s", game_type, exc, exc_info=True)
            return []

    def _get_character_by_name(self, character_name: str, game_state: Optional[GameState] = None):
        """
        Get character by name using the SAME function that !reroll uses.
        Uses _find_character_by_token from bot.py - no reimplementation!
        
        If game_state is provided (gameboard mode), filters characters based on pack enable flags.
        If game_state is None (VN mode or non-game context), uses global CHARACTER_POOL.
        
        Args:
            character_name: Character name or token to search for
            game_state: Optional GameState for gameboard mode filtering
        """
        try:
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
            
            # Get the character search function
            _find_character_by_token = getattr(bot_module, '_find_character_by_token', None)
            if not _find_character_by_token:
                logger.warning("_find_character_by_token not found in bot module")
                return None
            
            # If game_state is provided, use filtered character pool (gameboard mode)
            if game_state and game_state.game_type:
                # Use saved enabled_packs if available, otherwise None (will read from config)
                enabled_packs = game_state.enabled_packs if game_state.enabled_packs else None
                filtered_pool = self._get_filtered_character_pool(game_state.game_type, enabled_packs=enabled_packs)
                if filtered_pool:
                    # Search in filtered pool using same logic as _find_character_by_token
                    normalized = (character_name or "").strip()
                    if not normalized:
                        return None
                    
                    # Get folder lookup tokens helper
                    _folder_lookup_tokens = getattr(bot_module, '_folder_lookup_tokens', None)
                    _normalize_folder_token = getattr(bot_module, '_normalize_folder_token', None)
                    _character_matches_token = getattr(bot_module, '_character_matches_token', None)
                    
                    if _folder_lookup_tokens and _normalize_folder_token:
                        # Build CHARACTER_BY_FOLDER for filtered pool
                        filtered_by_folder = {}
                        for char in filtered_pool:
                            folder_token = _normalize_folder_token(char.folder or char.name)
                            if folder_token and folder_token not in filtered_by_folder:
                                filtered_by_folder[folder_token] = char
                        
                        # Try folder lookup first
                        for folder_token in _folder_lookup_tokens(normalized):
                            match = filtered_by_folder.get(folder_token)
                            if match:
                                logger.info("Found character for '%s' in filtered pool: %s", character_name, match.name)
                                return match
                    
                    # Fall back to name matching
                    if _character_matches_token:
                        for char in filtered_pool:
                            if _character_matches_token(char, normalized):
                                logger.info("Found character for '%s' in filtered pool: %s", character_name, char.name)
                                return char
                    
                    logger.debug("No character found for '%s' in filtered pool for game %s", character_name, game_state.game_type)
                    return None
                else:
                    # No characters enabled for this game
                    logger.debug("No characters enabled for game %s", game_state.game_type)
                    return None
            
            # No game_state provided - use global CHARACTER_POOL (VN mode behavior)
            result = _find_character_by_token(character_name)
            if result:
                logger.info("Found character for '%s': %s", character_name, result.name)
            else:
                logger.debug("No character found for '%s'", character_name)
            return result
            
        except Exception as exc:
            logger.warning("Failed to lookup character %s: %s", character_name, exc, exc_info=True)
        return None

    async def _trigger_face_grab_on_assignment(
        self,
        character_name: str,
        force: bool = False,
    ) -> None:
        """
        Trigger face grab when a character is assigned in gameboard mode.
        This runs in a background task to avoid blocking the assignment command.
        
        Args:
            character_name: Name of the character to grab face for
            force: If True, always regenerate face even if it exists (default: False)
        """
        try:
            from tfbot.panels import (
                compose_game_avatar,
                resolve_character_directory,
                _cache_character_face,
                _load_character_config,
                _ordered_variant_dirs,
            )
            import tf_characters
            
            # Check config option from tf_characters.json
            # The config file is now automatically updated to include this option
            config_enabled = True  # Default to True
            if tf_characters._config_path.exists():
                try:
                    import json
                    with open(tf_characters._config_path, 'r', encoding='utf-8') as f:
                        config_data = json.load(f)
                    # Handle both list format (legacy) and object format (new)
                    if isinstance(config_data, dict):
                        config_enabled = config_data.get("always_grab_faces_in_gameboard", True)
                    # If it's a list (legacy format), default to True
                except Exception as exc:
                    logger.warning("Failed to read always_grab_faces_in_gameboard config: %s", exc)
                    config_enabled = True  # Default to True on error
            
            # If config is OFF and not forcing, use normal behavior (only grab if missing)
            # If config is ON or force=True, always regenerate
            should_force = force or config_enabled
            
            # Compose avatar to get the image
            avatar_image = compose_game_avatar(character_name)
            if not avatar_image:
                logger.warning("Failed to compose avatar for face grab: %s", character_name)
                return
            
            # Get character directory
            character_dir, attempted = resolve_character_directory(character_name)
            if not character_dir:
                logger.warning("Failed to resolve character directory for face grab: %s (tried: %s)", character_name, attempted)
                return
            
            # Get character config to find variant directories
            config = _load_character_config(character_dir)
            if not config:
                logger.warning("Failed to load character config for face grab: %s", character_name)
                return
            
            # Get first/default variant directory
            variant_dirs = _ordered_variant_dirs(character_dir, config)
            if not variant_dirs:
                logger.warning("No variant directories found for face grab: %s", character_name)
                return
            
            variant_dir = variant_dirs[0]  # Use first/default variant
            
            # Trigger face grab with appropriate force setting
            _cache_character_face(character_dir, variant_dir, avatar_image, force=should_force)
            logger.info("Triggered face grab for %s (force=%s, config_enabled=%s)", character_name, should_force, config_enabled)
        except Exception as exc:
            logger.warning("Error triggering face grab for %s: %s", character_name, exc, exc_info=True)
    
    async def _create_game_state_for_player(
        self,
        player: GamePlayer,
        user_id: int,
        guild_id: int,
        character_name: str,
        member: Optional[discord.Member] = None,
        game_state: Optional[GameState] = None,
    ) -> Optional[TransformationState]:
        """
        Create a TransformationState using the EXACT same function VN roll uses.
        
        CRITICAL: This state is ONLY stored in game_state.player_states.
        It is NEVER added to global active_transformations.
        This allows player to be one character in game, another in VN simultaneously!
        
        Args:
            player: GamePlayer object
            user_id: Discord user ID
            guild_id: Discord guild ID
            character_name: Character name to assign
            member: Optional discord.Member object. If provided, uses this directly instead of fetching.
            game_state: Optional GameState for gameboard mode character filtering
        """
        character = self._get_character_by_name(character_name, game_state=game_state)
        if not character:
            logger.warning("Character not found for game player: %s", character_name)
            return None
        
        # Import bot_module FIRST (always needed for _build_roleplay_state)
        import sys
        bot_module = sys.modules.get('bot') or sys.modules.get('__main__')
        if not bot_module:
            try:
                import bot as bot_module
            except ImportError:
                logger.warning("Cannot import bot module")
                return None
        
        if not hasattr(bot_module, 'bot'):
            logger.warning("Bot instance not found in module")
            return None
        
        bot_instance = getattr(bot_module, 'bot')
        
        # Use provided member if available, otherwise fetch it
        guild = None
        if member:
            # Use provided member directly (avoids lookup failures)
            guild = member.guild
            logger.debug("Using provided member object for user %s (display: %s)", user_id, member.display_name)
        else:
            # Fetch member from guild
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
        # Import here to avoid circular dependency
        from tfbot.panels import get_background_root, list_background_choices, VN_BACKGROUND_DEFAULT_RELATIVE
        
        if background_id is None:
            # Use default background when background_id is None
            bg_root = get_background_root()
            if bg_root and VN_BACKGROUND_DEFAULT_RELATIVE:
                default_path = bg_root / VN_BACKGROUND_DEFAULT_RELATIVE
                if default_path.exists():
                    return default_path
            return None
        
        backgrounds = list_background_choices()
        if 1 <= background_id <= len(backgrounds):
            return backgrounds[background_id - 1]  # 1-indexed
        
        # Fall back to default (not first VN background)
        bg_root = get_background_root()
        if bg_root and VN_BACKGROUND_DEFAULT_RELATIVE:
            default_path = bg_root / VN_BACKGROUND_DEFAULT_RELATIVE
            if default_path.exists():
                return default_path
        
        return None  # Don't fall back to first VN background

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
        
        # CRITICAL: Verify thread belongs to this bot's configured forum channel
        # This prevents multiple bot instances from processing each other's messages
        if self.forum_channel_id > 0:
            parent_forum = getattr(message.channel, 'parent', None)
            if parent_forum and hasattr(parent_forum, 'id'):
                if parent_forum.id != self.forum_channel_id:
                    # Thread belongs to different forum - ignore message
                    return False
        
        if not message.guild:
            return False

        # Capture URLs before strip_urls removes them (for link-based GIF handling)
        try:
            from tfbot.panels import URL_RE
            link_urls = URL_RE.findall(message.content) if message.content else []
        except Exception:
            link_urls = []
        
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
        
        # CRITICAL: Only process messages if this bot owns the game
        # This prevents multiple bots from processing the same game
        bot_user_id = self.bot.user.id if self.bot.user else None
        if game_state.bot_user_id is not None and game_state.bot_user_id != bot_user_id:
            logger.debug("Skipping gameboard message - owned by bot %s, this bot is %s", game_state.bot_user_id, bot_user_id)
            return False
        
        # CRITICAL: Block messages from removed/forfeited players
        # Check if player is in game_state.players
        if message.author.id not in game_state.players:
            # Allow GM/narrator to speak even if not a player
            if message.author.id == game_state.gm_user_id or message.author.id == game_state.narrator_user_id:
                logger.debug("Allowing GM/narrator message even though not in players (author_id=%s)", message.author.id)
            else:
                # Player is not in game - delete immediately (no caching)
                try:
                    await message.delete()
                except discord.HTTPException:
                    pass
                return True
        
        # CRITICAL: Also block messages from forfeited players (they stay in game_state.players but cannot speak)
        # Check if player is forfeited
        pack = get_game_pack(game_state.game_type, self.packs_dir)
        if pack and pack.has_function("get_game_data"):
            pack_module = pack.module
            get_game_data = getattr(pack_module, "get_game_data", None)
            if callable(get_game_data):
                try:
                    data = get_game_data(game_state)
                    forfeited_players = data.get('forfeited_players', [])
                    if message.author.id in forfeited_players:
                        # Player is forfeited - cache and delete message
                        await self._cache_and_delete_message(message, thread_id, "forfeited player")
                        return True
                except Exception as exc:
                    logger.debug("Failed to check forfeited_players during message blocking: %s", exc)
                    # Continue - if check fails, allow message through (fail open)
        
        # Handle map thread: auto-delete all messages except admin commands
        if game_state.map_thread_id and thread_id == game_state.map_thread_id:
            is_admin_user = is_admin(message.author) or is_bot_mod(message.author)
            # Allow admin commands, cache and delete everything else
            if not is_admin_user or not (message.content and message.content.strip().startswith('!')):
                await self._cache_and_delete_message(
                    message,
                    thread_id,
                    "map thread message",
                    queue_thread_id=game_state.game_thread_id,
                )
                return True
            # Admin command in map thread - let it through
            return False
        
        if game_state.is_locked:
            return False
        
        # CRITICAL: Check if a command/operation is currently processing (lock is held)
        # This includes: GM commands, board rendering, movement calculations, etc.
        # If so, delete message immediately and queue it for processing after operation completes
        # This applies to ALL messages (including GM/admin) to ensure proper ordering
        # EXCEPTION: Skip this check if this is a queued message being reprocessed
        # EXCEPTION: Skip this check if message is from bot (bot messages should never be queued)
        if not is_queued:
            # CRITICAL: Never queue bot messages - they are command replies that must appear immediately
            # Bot messages are already excluded at line 1098, but this is a safety check to ensure
            # they never get queued even if they somehow reach this point
            if message.author.bot:
                logger.debug("Skipping queuing for bot message (message_id=%s, author_id=%s) - bot messages should never be queued", 
                           message.id, message.author.id)
                # Let bot messages through normally - don't queue them
                # Fall through to normal message processing below (though they should have been excluded at line 1098)
            else:
                command_lock = self._get_command_lock(thread_id)
                if command_lock.locked():
                    # Operation is processing (command, board rendering, movement calculation, etc.)
                    # Extract message data BEFORE deleting
                    # DIAGNOSTIC: Track admin/player status for queuing
                    is_gm_queuing = self._is_actual_gm(message.author, game_state) if game_state else False
                    is_admin_queuing = is_admin(message.author) or is_bot_mod(message.author)
                    player_queuing = game_state.players.get(message.author.id) if game_state else None
                    has_character_queuing = player_queuing and player_queuing.character_name
                    logger.info("[QUEUE-STEP-1] Command lock detected - queuing message (author_id=%s, message_id=%s, thread_id=%s, is_gm=%s, is_admin=%s, has_character=%s, character_name=%s)", 
                               message.author.id, message.id, thread_id, is_gm_queuing, is_admin_queuing, 
                               has_character_queuing, player_queuing.character_name if player_queuing else None)
                    
                    # Download attachments before deleting message (attachments become inaccessible after deletion)
                    attachment_data = []
                    if message.attachments:
                        logger.info("[QUEUE-STEP-2] Attachment detection: Found %d attachment(s) (message_id=%s)", 
                                   len(message.attachments), message.id)
                        for idx, attachment in enumerate(message.attachments, 1):
                            logger.info("[QUEUE-STEP-3] Downloading attachment %d/%d: %s (message_id=%s, content_type=%s)", 
                                       idx, len(message.attachments), attachment.filename, message.id, attachment.content_type)
                            try:
                                attachment_bytes = await attachment.read()
                                byte_count = len(attachment_bytes) if attachment_bytes else 0
                                logger.info("[QUEUE-STEP-3] Read attachment bytes: %s (byte_count=%d)", attachment.filename, byte_count)
                                
                                # Validate bytes are not empty before storing
                                if not attachment_bytes or len(attachment_bytes) == 0:
                                    logger.error("[QUEUE-STEP-3] ERROR: Attachment %s has empty bytes (0 bytes) - cannot queue! (message_id=%s)", 
                                               attachment.filename, message.id, exc_info=True)
                                    continue  # Skip storing empty attachment
                                
                                attachment_data.append({
                                    'filename': attachment.filename,
                                    'bytes': attachment_bytes,
                                    'content_type': attachment.content_type,
                                })
                                logger.info("[QUEUE-STEP-3] SUCCESS: Stored attachment %s (byte_count=%d, content_type=%s)", 
                                           attachment.filename, byte_count, attachment.content_type)
                            except Exception as exc:
                                logger.error("[QUEUE-STEP-3] ERROR: Failed to download attachment %s for queuing (message_id=%s): %s", 
                                           attachment.filename, message.id, exc, exc_info=True)
                    else:
                        logger.info("[QUEUE-STEP-2] Attachment detection: No attachments found (message_id=%s)", message.id)
                    
                    # Download stickers before deleting message (stickers become inaccessible after deletion)
                    sticker_data = []
                    has_stickers = hasattr(message, 'stickers') and message.stickers
                    if has_stickers:
                        sticker_count = len(message.stickers)
                        logger.info("[QUEUE-STEP-4] Sticker detection: Found %d sticker(s) (message_id=%s)", sticker_count, message.id)
                        for idx, sticker in enumerate(message.stickers, 1):
                            sticker_id = sticker.id if hasattr(sticker, 'id') else 'unknown'
                            sticker_name = sticker.name if hasattr(sticker, 'name') else 'sticker'
                            logger.info("[QUEUE-STEP-4] Downloading sticker %d/%d: %s (id=%s, message_id=%s)", 
                                       idx, sticker_count, sticker_name, sticker_id, message.id)
                            try:
                                # Download sticker image
                                if hasattr(sticker, 'url') and sticker.url:
                                    logger.info("[QUEUE-STEP-4] Fetching sticker from URL: %s (sticker_id=%s)", sticker.url, sticker_id)
                                    async with aiohttp.ClientSession() as session:
                                        async with session.get(sticker.url) as resp:
                                            if resp.status == 200:
                                                sticker_bytes = await resp.read()
                                                byte_count = len(sticker_bytes) if sticker_bytes else 0
                                                logger.info("[QUEUE-STEP-4] Read sticker bytes: %s (byte_count=%d, sticker_id=%s)", 
                                                           sticker_name, byte_count, sticker_id)
                                                
                                                if not sticker_bytes or len(sticker_bytes) == 0:
                                                    logger.error("[QUEUE-STEP-4] ERROR: Sticker %s has empty bytes (0 bytes) - cannot queue! (sticker_id=%s, message_id=%s)", 
                                                               sticker_name, sticker_id, message.id, exc_info=True)
                                                    continue
                                                
                                                sticker_data.append({
                                                    'name': sticker_name,
                                                    'bytes': sticker_bytes,
                                                    'filename': f"sticker_{sticker_id}.png",  # Stickers are typically PNG
                                                })
                                                logger.info("[QUEUE-STEP-4] SUCCESS: Stored sticker %s (byte_count=%d, sticker_id=%s)", 
                                                           sticker_name, byte_count, sticker_id)
                                            else:
                                                logger.error("[QUEUE-STEP-4] ERROR: Failed to fetch sticker - HTTP status %d (sticker_id=%s, url=%s, message_id=%s)", 
                                                           resp.status, sticker_id, sticker.url, message.id, exc_info=True)
                                else:
                                    logger.error("[QUEUE-STEP-4] ERROR: Sticker has no URL (sticker_id=%s, message_id=%s)", 
                                               sticker_id, message.id, exc_info=True)
                            except Exception as exc:
                                logger.error("[QUEUE-STEP-4] ERROR: Failed to download sticker %s for queuing (sticker_id=%s, message_id=%s): %s", 
                                           sticker_name, sticker_id, message.id, exc, exc_info=True)
                    else:
                        logger.info("[QUEUE-STEP-4] Sticker detection: No stickers found (message_id=%s)", message.id)
                    
                    # Download embed images before deleting message (embeds become inaccessible after deletion)
                    embed_data = []
                    if message.embeds:
                        embed_count = len(message.embeds)
                        logger.info("[QUEUE-STEP-4.5] Embed detection: Found %d embed(s) (message_id=%s)", embed_count, message.id)
                        for idx, embed in enumerate(message.embeds, 1):
                            logger.info("[QUEUE-STEP-4.5] Processing embed %d/%d (message_id=%s)", idx, embed_count, message.id)
                            
                            # Extract image URL from embed (check image, video, thumbnail)
                            image_url = None
                            if embed.image and embed.image.url:
                                image_url = embed.image.url
                                logger.info("[QUEUE-STEP-4.5] Found embed.image.url: %s", image_url)
                            elif embed.video and embed.video.url:
                                image_url = embed.video.url
                                logger.info("[QUEUE-STEP-4.5] Found embed.video.url: %s", image_url)
                            elif embed.thumbnail and embed.thumbnail.url:
                                image_url = embed.thumbnail.url
                                logger.info("[QUEUE-STEP-4.5] Found embed.thumbnail.url: %s", image_url)
                            
                            if image_url:
                                try:
                                    logger.info("[QUEUE-STEP-4.5] Downloading embed image from URL: %s (message_id=%s)", image_url, message.id)
                                    async with aiohttp.ClientSession() as session:
                                        async with session.get(image_url) as resp:
                                            if resp.status == 200:
                                                embed_bytes = await resp.read()
                                                byte_count = len(embed_bytes) if embed_bytes else 0
                                                logger.info("[QUEUE-STEP-4.5] Read embed bytes: %s (byte_count=%d, message_id=%s)", 
                                                           image_url, byte_count, message.id)
                                                
                                                if not embed_bytes or len(embed_bytes) == 0:
                                                    logger.error("[QUEUE-STEP-4.5] ERROR: Embed image has empty bytes (0 bytes) - cannot queue! (url=%s, message_id=%s)", 
                                                               image_url, message.id, exc_info=True)
                                                    continue
                                                
                                                # Determine filename from URL or content-type
                                                filename = image_url.split('/')[-1].split('?')[0] or 'embed_image.gif'
                                                content_type = resp.headers.get('Content-Type', 'image/gif')
                                                
                                                embed_data.append({
                                                    'filename': filename,
                                                    'bytes': embed_bytes,
                                                    'content_type': content_type,
                                                })
                                                logger.info("[QUEUE-STEP-4.5] SUCCESS: Stored embed image %s (byte_count=%d, content_type=%s, message_id=%s)", 
                                                           filename, byte_count, content_type, message.id)
                                            else:
                                                logger.error("[QUEUE-STEP-4.5] ERROR: Failed to fetch embed image - HTTP status %d (url=%s, message_id=%s)", 
                                                           resp.status, image_url, message.id, exc_info=True)
                                except Exception as exc:
                                    logger.error("[QUEUE-STEP-4.5] ERROR: Failed to download embed image for queuing (url=%s, message_id=%s): %s", 
                                               image_url, message.id, exc, exc_info=True)
                            else:
                                logger.debug("[QUEUE-STEP-4.5] Embed %d/%d has no image/video/thumbnail URL (message_id=%s)", 
                                           idx, embed_count, message.id)
                    else:
                        logger.info("[QUEUE-STEP-4.5] Embed detection: No embeds found (message_id=%s)", message.id)
                    
                    # If no embeds were created, try link URLs (GIF links without embeds)
                    if not message.embeds and link_urls:
                        for url_idx, link_url in enumerate(link_urls, 1):
                            try:
                                async with aiohttp.ClientSession() as session:
                                    async with session.get(link_url) as resp:
                                        if resp.status != 200:
                                            continue
                                        content_type = resp.headers.get("Content-Type", "")
                                        if content_type and not (content_type.startswith("image/") or content_type.startswith("video/")):
                                            continue
                                        link_bytes = await resp.read()
                                        if not link_bytes:
                                            continue
                                        filename = link_url.split("/")[-1].split("?")[0] or "embed_link.gif"
                                        embed_data.append({
                                            "filename": filename,
                                            "bytes": link_bytes,
                                            "content_type": content_type or "image/gif",
                                        })
                            except Exception as exc:
                                logger.debug("[QUEUE-STEP-4.5] Failed to download link URL %d/%d for queuing: %s", url_idx, len(link_urls), exc)
                    
                    # Extract all necessary message data before deletion
                    content_length = len(message.content) if message.content else 0
                    logger.info("[QUEUE-STEP-5] Creating message data dict (message_id=%s, author_id=%s, attachments=%d, stickers=%d, embeds=%d, content_length=%d)", 
                               message.id, message.author.id, len(attachment_data), len(sticker_data), len(embed_data), content_length)
                    message_data = {
                        'content': message.content,
                        'author': message.author,
                        'channel': message.channel,
                        'guild': message.guild,
                        'reference': message.reference,
                        'attachments': attachment_data,  # Store downloaded attachment data
                        'stickers': sticker_data,  # Store downloaded sticker data, not just sticker objects
                        'embeds': embed_data,  # Store downloaded embed image data
                        'id': message.id,
                    }
                    logger.info("[QUEUE-STEP-5] SUCCESS: Message data dict created (message_id=%s, attachment_data_items=%d, sticker_data_items=%d)", 
                               message.id, len(message_data.get('attachments', [])), len(message_data.get('stickers', [])))
                    
                    # Delete message immediately
                    logger.info("[QUEUE-STEP-6] Attempting to delete message (message_id=%s)", message.id)
                    try:
                        await message.delete()
                        logger.info("[QUEUE-STEP-6] SUCCESS: Message deleted (message_id=%s)", message.id)
                    except discord.HTTPException as exc:
                        logger.warning("[QUEUE-STEP-6] WARNING: Failed to delete message (message_id=%s) - may already be deleted: %s", 
                                     message.id, exc)
                    except Exception as exc:
                        logger.error("[QUEUE-STEP-6] ERROR: Unexpected error deleting message (message_id=%s): %s", 
                                   message.id, exc, exc_info=True)
                    
                    # Queue message data for processing after operation completes
                    # CRITICAL: Queue ALL messages (including GM/admin) to ensure proper ordering
                    if thread_id not in self._message_queues:
                        self._message_queues[thread_id] = []
                        logger.info("[QUEUE-STEP-7] Created new queue for thread_id=%s", thread_id)
                    
                    self._message_queues[thread_id].append(message_data)
                    queue_size = len(self._message_queues[thread_id])
                    logger.info("[QUEUE-STEP-7] SUCCESS: Message queued (message_id=%s, author_id=%s, thread_id=%s, queue_size=%d, attachments=%d, stickers=%d, embeds=%d, content_length=%d)", 
                               message.id, message.author.id, thread_id, queue_size, len(attachment_data), 
                               len(sticker_data), len(embed_data), content_length)
                    return True
        
        # Check if author is GM or admin
        is_gm = self._is_actual_gm(message.author, game_state)
        is_admin_user = is_admin(message.author) or is_bot_mod(message.author)
        is_narrator = game_state.narrator_user_id == message.author.id
        
        # Get player info first
        player = game_state.players.get(message.author.id)
        has_character = player and player.character_name
        
        # Handle actual GM: If GM doesn't have character, always use Narrator
        if is_gm:
            if has_character:
                # GM has character - fall through to VN panel rendering below
                pass
            else:
                # GM without character - always use Narrator
                # Check if message looks like a command - if so, let it through
                if message.content and message.content.strip().startswith('!'):
                    # Command attempt - let command handler process it
                    return False  # Let command handler process it (will show error if command doesn't exist)
                # Otherwise handle as narrator message
                await self._handle_narrator_message(message, game_state)
                return True
        elif is_admin_user:
            # Non-GM admin: Must have character assigned to speak
            if has_character:
                # Admin has character - fall through to VN panel rendering below
                pass
            else:
                # Admin without character - cache and delete message
                logger.debug("Deleting message from admin %s without assigned character", message.author.id)
                await self._cache_and_delete_message(message, thread_id, "admin without character")
                return True
        else:
            # Not GM/admin - check if player is in the game and has a character assigned
            # (player and has_character already set above)
            
            # Block messages from non-GM, non-admin players without assigned character
            # This includes players not in the game (player is None) or players without character assigned
            if not has_character:
                logger.debug("Deleting message from unassigned player %s (player=%s, has_character=%s)", 
                            message.author.id, player is not None, has_character)
                await self._cache_and_delete_message(message, thread_id, "unassigned player")
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
                game_state=game_state,
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
            # CRITICAL: Use message.author directly to avoid member lookup failures
            state = await self._create_game_state_for_player(
                player,
                message.author.id,
                message.guild.id,
                player.character_name,
                game_state=game_state,
                member=message.author,  # Pass member directly to avoid lookup failures
            )
            if state:
                game_state.player_states[message.author.id] = state
                logger.info("Recreated game state for player %s with correct character '%s' (was '%s')", 
                           message.author.id, state.character_name, player.character_name)
            else:
                logger.error("Failed to recreate state for player %s with character '%s'", message.author.id, player.character_name)
                # Don't block message - allow it through with warning
                logger.warning("Allowing message through despite state recreation failure - player may need to be re-assigned")
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
            
            # Check if message has attachments (images) or stickers - these should be allowed even without text
            # CRITICAL: For queued messages, check _attachment_data FIRST since message.attachments might be empty AttachmentProxy objects
            original_attachment_count = len(message.attachments) if message.attachments else 0
            has_attachment_data = hasattr(message, '_attachment_data') and bool(message._attachment_data)
            attachment_data_count = len(message._attachment_data) if has_attachment_data else 0
            
            logger.info("[ATTACHMENT-DETECT] Early detection: message_id=%s, is_queued=%s, original_attachment_count=%d, has_attachment_data=%s, attachment_data_count=%d",
                       message.id if hasattr(message, 'id') else 'unknown', is_queued, original_attachment_count, has_attachment_data, attachment_data_count)
            
            # For queued messages, prioritize _attachment_data since AttachmentProxy might not work
            if is_queued and has_attachment_data:
                has_attachments = True  # Queued message has attachments in _attachment_data
                logger.info("[ATTACHMENT-DETECT] Queued message detected with _attachment_data: %d attachment(s) (message_id=%s)", 
                           attachment_data_count, message.id if hasattr(message, 'id') else 'unknown')
            else:
                # CRITICAL FIX: Use len() check instead of bool() to properly detect attachments
                # bool(message.attachments) can be False even when attachments exist (empty list or invalid AttachmentProxy)
                # Check both attachment count and _attachment_data for non-queued messages
                has_attachments_from_list = original_attachment_count > 0
                has_attachments = has_attachments_from_list or has_attachment_data
                
                # Additional validation: if message.attachments exists but count is 0, log warning
                if message.attachments is not None and original_attachment_count == 0:
                    logger.warning("[ATTACHMENT-DETECT] message.attachments exists but count is 0 (message_id=%s)", 
                                  message.id if hasattr(message, 'id') else 'unknown')
                
                if has_attachments:
                    logger.info("[ATTACHMENT-DETECT] Non-queued message: has_attachments=%s (from_list=%s, original_count=%d, attachment_data=%s, message_id=%s)", 
                               has_attachments, has_attachments_from_list, original_attachment_count, has_attachment_data,
                               message.id if hasattr(message, 'id') else 'unknown')
            
            has_stickers = bool(message.stickers) or (hasattr(message, 'sticker_files') and bool(message.sticker_files))
            has_real_embeds = bool(message.embeds)
            has_link_embeds = bool(link_urls) and not has_real_embeds
            has_embeds = has_real_embeds or has_link_embeds or (hasattr(message, 'embed_files') and bool(message.embed_files))
            logger.info("[ATTACHMENT-DETECT] Final detection: has_attachments=%s (original_count=%d, attachment_data_count=%d), has_stickers=%s, has_embeds=%s (message_id=%s)", 
                       has_attachments, original_attachment_count, attachment_data_count, has_stickers, has_embeds,
                       message.id if hasattr(message, 'id') else 'unknown')
            
            if not cleaned_content and not has_attachments and not has_stickers and not has_embeds:
                # No content, no attachments, and no stickers - ignore
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
            # CRITICAL: Uses player.background_id, NOT form_owner_user_id, ensuring !swap and !pswap behave identically
            background_path = self._get_game_background_path(player.background_id)
            
            # Defensive check: if background_path is None but player.background_id is set, recalculate it
            if background_path is None and player.background_id is not None:
                logger.warning("Background path was None for player %s (user_id=%s) with background_id=%s, recalculating...", 
                             player.character_name, message.author.id, player.background_id)
                background_path = self._get_game_background_path(player.background_id)
                if background_path is None:
                    logger.error("Failed to get background path for player %s (user_id=%s) with background_id=%s after recalculation", 
                               player.character_name, message.author.id, player.background_id)
            
            logger.info("Background selection for player %s (user_id=%s): background_id=%s, path=%s, state.user_id=%s", 
                       player.character_name, message.author.id, player.background_id, background_path, 
                       state.user_id if state else "None")
            # Verify state.user_id matches message.author.id (required for monkey-patch to work)
            if state and state.user_id != message.author.id:
                logger.warning("MISMATCH: state.user_id=%s != message.author.id=%s - monkey-patch may not work correctly!", 
                             state.user_id, message.author.id)
            
            # Get character display name (used for logging and display)
            character_display_name = state.identity_display_name or player.character_name
            
            # Render VN panel (only if VN style is enabled)
            # Skip VN panel rendering if only stickers/GIFs are present (no text) - send them directly instead
            files = []
            if MESSAGE_STYLE == "vn" and cleaned_content:
                # Gameboard background is passed into render_vn_panel (no global monkey-patch—thread-safe with run_panel_render).
                # CRITICAL: Always use player.character_name (the source of truth) not state.character_name
                if state.character_name != player.character_name:
                    logger.error("CRITICAL MISMATCH before render: state.character_name='%s' != player.character_name='%s'. Fixing state...", 
                                state.character_name, player.character_name)
                    character = self._get_character_by_name(player.character_name, game_state=game_state)
                    if character:
                        state.character_name = player.character_name
                        state.character_folder = character.folder
                        state.character_avatar_path = character.avatar_path or ""
                        state.character_message = character.message or ""
                        logger.info("Updated state with character '%s' (folder='%s', avatar='%s')", 
                                   character.name, character.folder, character.avatar_path)
                    else:
                        logger.warning("Character lookup failed for '%s', only updating name", player.character_name)
                        state.character_name = player.character_name
                    game_state.player_states[message.author.id] = state
                    logger.info("Fixed state: now state.character_name='%s', folder='%s', avatar='%s'", 
                               state.character_name, state.character_folder, state.character_avatar_path)

                logger.info("Rendering VN panel for game player: user_id=%s, player.character_name='%s', state.character_name='%s', character_display_name='%s'", 
                           message.author.id, player.character_name, state.character_name, character_display_name)

                if not state.character_name:
                    logger.error("State missing character_name! Cannot render VN panel.")
                elif not state.character_avatar_path:
                    logger.warning("State missing character_avatar_path for %s", state.character_name)

                vn_file = await run_panel_render_vn(
                    render_vn_panel,
                    state=state,
                    message_content=cleaned_content,
                    character_display_name=character_display_name,
                    original_name=message.author.display_name,
                    attachment_id=str(message.id),
                    formatted_segments=formatted_segments,
                    custom_emoji_images=custom_emoji_images,
                    reply_context=reply_context,
                    gacha_outfit_override=player.outfit_name if player.outfit_name else None,
                    panel_background_path=background_path,
                )
                if vn_file:
                    logger.info("VN panel rendered successfully for %s", character_display_name)
                    files.append(vn_file)
                else:
                    logger.warning("render_vn_panel returned None for %s (character_name='%s', state.character_name='%s')", 
                                 character_display_name, player.character_name, state.character_name)

            # If only stickers/GIFs are present (no text), send them directly without VN panel
            if not cleaned_content and (has_attachments or has_stickers or has_embeds):
                if not is_queued:
                    # Preserve original message for non-queued media-only posts (VN behavior)
                    if has_links or has_attachments or has_stickers or has_real_embeds:
                        return True
                author_id = message.author.id if message.author else 'unknown'
                message_id = message.id if hasattr(message, 'id') else 'unknown'
                # DIAGNOSTIC: Track admin/player status during processing
                is_gm_process = self._is_actual_gm(message.author, game_state) if message.author and game_state else False
                is_admin_process = (is_admin(message.author) or is_bot_mod(message.author)) if message.author else False
                player_process = game_state.players.get(author_id) if game_state else None
                has_character_process = player_process and player_process.character_name
                logger.info("[PROCESS-STEP-1] Processing start: Direct send (no text) (author_id=%s, message_id=%s, has_attachments=%s, has_stickers=%s, is_gm=%s, is_admin=%s, has_character=%s, character_name=%s)", 
                           author_id, message_id, has_attachments, has_stickers, is_gm_process, is_admin_process, 
                           has_character_process, player_process.character_name if player_process else None)
                attachment_files = []
                
                # CRITICAL: For queued messages, use _attachment_data directly instead of trying AttachmentProxy
                # AttachmentProxy.read() often fails or returns empty bytes for queued messages
                if is_queued and hasattr(message, '_attachment_data') and message._attachment_data:
                    logger.info("[PROCESS-STEP-2] Queued message: Using _attachment_data directly (%d items, message_id=%s)", 
                               len(message._attachment_data), message_id)
                    for att_idx, att_data in enumerate(message._attachment_data, 1):
                        filename = att_data.get('filename', 'unknown')
                        logger.info("[PROCESS-STEP-2] Processing queued attachment %d/%d: %s (message_id=%s)", 
                                   att_idx, len(message._attachment_data), filename, message_id)
                        try:
                            att_bytes = att_data.get('bytes', b'')
                            byte_count = len(att_bytes) if att_bytes else 0
                            logger.info("[PROCESS-STEP-2] Extracted bytes from _attachment_data: %s (byte_count=%d)", 
                                       filename, byte_count)
                            
                            if att_bytes and len(att_bytes) > 0:
                                logger.info("[PROCESS-STEP-4] Creating discord.File from _attachment_data: %s (byte_count=%d)", 
                                           filename, byte_count)
                                attachment_file = discord.File(
                                    io.BytesIO(att_bytes),
                                    filename=filename
                                )
                                attachment_files.append(attachment_file)
                                logger.info("[PROCESS-STEP-4] SUCCESS: Created attachment file from _attachment_data (queued message): %s (byte_count=%d)", 
                                           filename, byte_count)
                            else:
                                logger.error("[PROCESS-STEP-2] ERROR: Bytes are empty for queued attachment %s (message_id=%s)", 
                                           filename, message_id, exc_info=True)
                        except Exception as exc:
                            logger.error("[PROCESS-STEP-2] ERROR: Failed to create file from _attachment_data for queued message %s: %s", 
                                       filename, exc, exc_info=True)
                elif message.attachments and len(message.attachments) > 0:
                    # CRITICAL FIX: Validate attachment count > 0, not just truthy check
                    # This ensures we only process when attachments actually exist
                    attachment_count = len(message.attachments)
                    logger.info("[PROCESS-STEP-2] Attachment read loop: Processing %d attachment(s) (message_id=%s)", 
                               attachment_count, message_id)
                    for att_idx, attachment in enumerate(message.attachments, 1):
                        # Validate attachment object before processing
                        if not attachment or not hasattr(attachment, 'filename'):
                            logger.warning("[PROCESS-STEP-2] WARNING: Invalid attachment object at index %d (message_id=%s), skipping", 
                                          att_idx, message_id)
                            continue
                        
                        logger.info("[PROCESS-STEP-2] Reading attachment %d/%d: %s (message_id=%s)", 
                                   att_idx, attachment_count, attachment.filename, message_id)
                        attachment_bytes = None
                        try:
                            logger.info("[PROCESS-STEP-2] Attempting to read via AttachmentProxy.read(): %s", attachment.filename)
                            attachment_bytes = await attachment.read()
                            byte_count = len(attachment_bytes) if attachment_bytes else 0
                            logger.info("[PROCESS-STEP-2] AttachmentProxy.read() completed: %s (byte_count=%d)", 
                                       attachment.filename, byte_count)
                            
                            # Check if bytes are empty (not just if exception occurs)
                            if not attachment_bytes or len(attachment_bytes) == 0:
                                logger.error("[PROCESS-STEP-2] ERROR: AttachmentProxy.read() returned empty bytes for %s (message_id=%s)", 
                                           attachment.filename, message_id, exc_info=True)
                                # Fall through to fallback logic below
                            else:
                                # Bytes are valid - create file
                                logger.info("[PROCESS-STEP-4] Creating discord.File from AttachmentProxy bytes: %s (byte_count=%d)", 
                                           attachment.filename, byte_count)
                                try:
                                    attachment_file = discord.File(
                                        io.BytesIO(attachment_bytes),
                                        filename=attachment.filename
                                    )
                                    attachment_files.append(attachment_file)
                                    logger.info("[PROCESS-STEP-4] SUCCESS: Added attachment %s to files list (byte_count=%d)", 
                                               attachment.filename, byte_count)
                                    continue  # Successfully created file, move to next attachment
                                except Exception as file_exc:
                                    logger.error("[PROCESS-STEP-4] ERROR: Failed to create discord.File for %s: %s", 
                                               attachment.filename, file_exc, exc_info=True)
                        except Exception as exc:
                            logger.error("[PROCESS-STEP-2] ERROR: AttachmentProxy.read() failed for %s: %s", 
                                       attachment.filename, exc, exc_info=True)
                            attachment_bytes = None  # Mark as failed
                elif message.attachments is not None and len(message.attachments) == 0:
                    # CRITICAL FIX: If message.attachments is empty but _attachment_data exists, process all items directly
                    # This ensures non-admin players can use GIFs just like admins
                    if hasattr(message, '_attachment_data') and message._attachment_data:
                        logger.info("[PROCESS-STEP-2] message.attachments is empty, processing _attachment_data directly (%d items, message_id=%s)", 
                                   len(message._attachment_data), message_id)
                        for att_idx, att_data in enumerate(message._attachment_data, 1):
                            filename = att_data.get('filename', 'unknown')
                            logger.info("[PROCESS-STEP-2] Processing attachment %d/%d from _attachment_data: %s (message_id=%s)", 
                                       att_idx, len(message._attachment_data), filename, message_id)
                            try:
                                att_bytes = att_data.get('bytes', b'')
                                byte_count = len(att_bytes) if att_bytes else 0
                                logger.info("[PROCESS-STEP-2] Extracted bytes from _attachment_data: %s (byte_count=%d)", 
                                           filename, byte_count)
                                
                                if att_bytes and len(att_bytes) > 0:
                                    logger.info("[PROCESS-STEP-4] Creating discord.File from _attachment_data: %s (byte_count=%d)", 
                                               filename, byte_count)
                                    attachment_file = discord.File(
                                        io.BytesIO(att_bytes),
                                        filename=filename
                                    )
                                    attachment_files.append(attachment_file)
                                    logger.info("[PROCESS-STEP-4] SUCCESS: Created attachment file from _attachment_data: %s (byte_count=%d)", 
                                               filename, byte_count)
                                else:
                                    logger.error("[PROCESS-STEP-2] ERROR: Bytes are empty for attachment %s (message_id=%s)", 
                                               filename, message_id, exc_info=True)
                            except Exception as exc:
                                logger.error("[PROCESS-STEP-2] ERROR: Failed to create file from _attachment_data for %s: %s", 
                                           filename, exc, exc_info=True)
                
                # Fallback: if no attachments were created but _attachment_data exists, create files directly
                if not attachment_files and hasattr(message, '_attachment_data') and message._attachment_data:
                    logger.info("[PROCESS-STEP-3] No attachments created from message.attachments, trying _attachment_data directly (%d items, message_id=%s)", 
                              len(message._attachment_data), message_id)
                    for fallback_idx, att_data in enumerate(message._attachment_data, 1):
                        logger.info("[PROCESS-STEP-3] Processing fallback attachment %d/%d: %s", 
                                   fallback_idx, len(message._attachment_data), att_data.get('filename', 'unknown'))
                        try:
                            att_bytes = att_data.get('bytes', b'')
                            byte_count = len(att_bytes) if att_bytes else 0
                            logger.info("[PROCESS-STEP-3] Extracted bytes from _attachment_data: %s (byte_count=%d)", 
                                       att_data.get('filename', 'unknown'), byte_count)
                            if att_bytes and len(att_bytes) > 0:
                                logger.info("[PROCESS-STEP-4] Creating discord.File from fallback _attachment_data: %s (byte_count=%d)", 
                                           att_data.get('filename', 'unknown'), byte_count)
                                fallback_file = discord.File(
                                    io.BytesIO(att_bytes),
                                    filename=att_data.get('filename', 'attachment')
                                )
                                attachment_files.append(fallback_file)
                                logger.info("[PROCESS-STEP-4] SUCCESS: Created attachment file directly from _attachment_data: %s (byte_count=%d)", 
                                           att_data.get('filename', 'attachment'), byte_count)
                            else:
                                logger.error("[PROCESS-STEP-3] ERROR: Fallback bytes are empty for %s (message_id=%s)", 
                                           att_data.get('filename', 'unknown'), message_id, exc_info=True)
                        except Exception as exc:
                            logger.error("[PROCESS-STEP-3] ERROR: Failed to create file from _attachment_data item %s: %s", 
                                       att_data.get('filename', 'unknown'), exc, exc_info=True)
                
                # Include sticker files if available (from queued messages)
                sticker_files = []
                if hasattr(message, 'sticker_files') and message.sticker_files:
                    sticker_files = message.sticker_files
                    logger.info("[PROCESS-STEP-1] Including %d sticker file(s) from queued message (message_id=%s)", 
                               len(sticker_files), message_id)
                else:
                    logger.info("[PROCESS-STEP-1] No sticker files available (message_id=%s)", message_id)
                
                # Include embed files if available (from queued messages)
                embed_files = []
                if hasattr(message, 'embed_files') and message.embed_files:
                    embed_files = message.embed_files
                    logger.info("[PROCESS-STEP-1] Including %d embed file(s) from queued message (message_id=%s)", 
                               len(embed_files), message_id)
                
                # Process embeds from message.embeds if not already processed (for non-queued messages)
                if not embed_files and message.embeds:
                    for embed in message.embeds:
                        image_url = None
                        if embed.image and embed.image.url:
                            image_url = embed.image.url
                        elif embed.video and embed.video.url:
                            image_url = embed.video.url
                        elif embed.thumbnail and embed.thumbnail.url:
                            image_url = embed.thumbnail.url
                        
                        if image_url:
                            try:
                                async with aiohttp.ClientSession() as session:
                                    async with session.get(image_url) as resp:
                                        if resp.status == 200:
                                            embed_bytes = await resp.read()
                                            if embed_bytes and len(embed_bytes) > 0:
                                                filename = image_url.split('/')[-1].split('?')[0] or 'embed_image.gif'
                                                embed_files.append(discord.File(
                                                    io.BytesIO(embed_bytes),
                                                    filename=filename
                                                ))
                                                logger.info("[PROCESS-STEP-4] Created embed file: %s (byte_count=%d)", filename, len(embed_bytes))
                            except Exception as exc:
                                logger.error("[PROCESS-STEP-4] Failed to process embed in direct send: %s", exc)
                
                # Process link URLs when embeds are missing (link-based GIFs)
                if not embed_files and has_link_embeds and link_urls:
                    for url_idx, link_url in enumerate(link_urls, 1):
                        try:
                            async with aiohttp.ClientSession() as session:
                                async with session.get(link_url) as resp:
                                    if resp.status != 200:
                                        continue
                                    content_type = resp.headers.get("Content-Type", "")
                                    if content_type and not (content_type.startswith("image/") or content_type.startswith("video/")):
                                        continue
                                    link_bytes = await resp.read()
                                    if not link_bytes:
                                        continue
                                    filename = link_url.split("/")[-1].split("?")[0] or "embed_link.gif"
                                    embed_files.append(discord.File(io.BytesIO(link_bytes), filename=filename))
                        except Exception as exc:
                            logger.debug("[PROCESS-STEP-4] Failed to download link URL %d/%d for direct send: %s", url_idx, len(link_urls), exc)
                
                # Combine attachments, stickers, and embeds
                all_attachment_files = attachment_files + sticker_files + embed_files
                logger.info("[PROCESS-STEP-4] File preparation summary: attachment_files=%d, sticker_files=%d, embed_files=%d, total=%d (message_id=%s)", 
                           len(attachment_files), len(sticker_files), len(embed_files), len(all_attachment_files), message_id)
                
                if all_attachment_files:
                    logger.info("[PROCESS-STEP-6] Preparing to send files (attachment_files=%d, sticker_files=%d, embed_files=%d, total=%d, message_id=%s)", 
                               len(attachment_files), len(sticker_files), len(all_attachment_files), message_id)
                    send_kwargs: Dict[str, object] = {
                        "files": all_attachment_files,
                        "allowed_mentions": discord.AllowedMentions.none(),
                    }
                    
                    if message.reference:
                        send_kwargs["reference"] = message.reference
                        logger.info("[PROCESS-STEP-6] Added message reference to send_kwargs (reference_id=%s)", 
                                   message.reference.message_id if message.reference else 'None')
                    
                    try:
                        logger.info("[PROCESS-STEP-6] Sending files to channel (channel_id=%s, file_count=%d)", 
                                   message.channel.id if message.channel else 'unknown', len(all_attachment_files))
                        await message.channel.send(**send_kwargs)
                        logger.info("[PROCESS-STEP-6] SUCCESS: Sent %d file(s) (attachments: %d, stickers: %d) for game player %s (message_id=%s)", 
                                   len(all_attachment_files), len(attachment_files), len(sticker_files), author_id, message_id)
                        
                        # Handle original message - match VN mode behavior exactly
                        # For queued messages: Skip deletion (already deleted when queued)
                        # For non-queued messages: Edit to placeholder if attachments exist, delete if no attachments
                        if not is_queued:
                            # Match VN bot: preserve_original = has_attachments or has_stickers or has_links
                            preserve_original = has_attachments or has_stickers or has_links
                            
                            if not preserve_original:
                                # No attachments, stickers, or links - delete original message
                                try:
                                    await message.delete()
                                    logger.info("[PROCESS-STEP-6] Deleted original message (no attachments/stickers/links, message_id=%s)", message_id)
                                except discord.Forbidden:
                                    logger.debug("Missing permission to delete message %s for game relay in channel %s",
                                               message.id, message.channel.id if message.channel else 'unknown')
                                except discord.HTTPException as del_exc:
                                    logger.warning("[PROCESS-STEP-6] WARNING: Failed to delete original message (message_id=%s): %s", 
                                                 message_id, del_exc)
                            else:
                                # Has attachments/stickers/links - edit to placeholder (match VN bot)
                                # Note: Stickers cannot be edited out, so we only edit if there are attachments
                                if has_attachments and not has_links:
                                    placeholder = "\u200b"
                                    if message.content != placeholder:
                                        try:
                                            await message.edit(content=placeholder, attachments=message.attachments, suppress=True)
                                            logger.info("[PROCESS-STEP-6] Edited original message to placeholder (has attachments, message_id=%s)", message_id)
                                        except discord.HTTPException as edit_exc:
                                            logger.debug("Unable to clear attachment message %s: %s", message.id, edit_exc)
                        else:
                            # Queued message - already deleted when queued, skip deletion
                            logger.info("[PROCESS-STEP-6] Skipping deletion for queued message (already deleted, message_id=%s)", message_id)
                    except Exception as exc:
                        logger.error("[PROCESS-STEP-6] ERROR: Failed to send attachments/stickers (message_id=%s, files: %d attachment, %d sticker): %s", 
                                   message_id, len(attachment_files), len(sticker_files), exc, exc_info=True)
                else:
                    # Final recovery attempt: if attachments were detected but no files created, try one more time
                    logger.info("[PROCESS-STEP-5] No files to send - checking if final recovery needed (has_attachments=%s, attachment_files=%d, message_id=%s)", 
                               has_attachments, len(attachment_files), message_id)
                    if has_attachments and hasattr(message, '_attachment_data') and message._attachment_data:
                        logger.warning("[PROCESS-STEP-5] Final recovery triggered: No files created despite attachments detected (message_id=%s, _attachment_data_items=%d)", 
                                     message_id, len(message._attachment_data))
                        recovered_files = []
                        for recovery_idx, att_data in enumerate(message._attachment_data, 1):
                            logger.info("[PROCESS-STEP-5] Final recovery attempt %d/%d: %s", 
                                       recovery_idx, len(message._attachment_data), att_data.get('filename', 'unknown'))
                            try:
                                att_bytes = att_data.get('bytes', b'')
                                byte_count = len(att_bytes) if att_bytes else 0
                                logger.info("[PROCESS-STEP-5] Extracted bytes for final recovery: %s (byte_count=%d)", 
                                           att_data.get('filename', 'unknown'), byte_count)
                                if att_bytes and len(att_bytes) > 0:
                                    logger.info("[PROCESS-STEP-4] Creating discord.File from final recovery: %s (byte_count=%d)", 
                                               att_data.get('filename', 'unknown'), byte_count)
                                    final_fallback_file = discord.File(
                                        io.BytesIO(att_bytes),
                                        filename=att_data.get('filename', 'attachment')
                                    )
                                    recovered_files.append(final_fallback_file)
                                    logger.info("[PROCESS-STEP-5] SUCCESS: Final recovery created attachment file: %s (byte_count=%d)", 
                                               att_data.get('filename', 'attachment'), byte_count)
                                else:
                                    logger.error("[PROCESS-STEP-5] ERROR: Final recovery bytes are empty for %s (message_id=%s)", 
                                               att_data.get('filename', 'unknown'), message_id, exc_info=True)
                            except Exception as exc:
                                logger.error("[PROCESS-STEP-5] ERROR: Final recovery failed to create file from _attachment_data for %s: %s", 
                                           att_data.get('filename', 'unknown'), exc, exc_info=True)
                        
                        # Try sending again if we recovered any files
                        if recovered_files:
                            logger.info("[PROCESS-STEP-6] Final recovery successful: Attempting to send %d recovered file(s) (message_id=%s)", 
                                       len(recovered_files), message_id)
                            all_recovered_files = recovered_files + sticker_files
                            send_kwargs: Dict[str, object] = {
                                "files": all_recovered_files,
                                "allowed_mentions": discord.AllowedMentions.none(),
                            }
                            if message.reference:
                                send_kwargs["reference"] = message.reference
                                logger.info("[PROCESS-STEP-6] Added message reference to final recovery send_kwargs")
                            try:
                                logger.info("[PROCESS-STEP-6] Sending final recovery files (file_count=%d, message_id=%s)", 
                                           len(all_recovered_files), message_id)
                                await message.channel.send(**send_kwargs)
                                logger.info("[PROCESS-STEP-6] SUCCESS: Sent %d file(s) after final recovery (attachments: %d, stickers: %d) for game player %s (message_id=%s)", 
                                           len(all_recovered_files), len(recovered_files), len(sticker_files), author_id, message_id)
                                return True
                            except Exception as exc:
                                logger.error("[PROCESS-STEP-6] ERROR: Failed to send files after final recovery (message_id=%s): %s", 
                                           message_id, exc, exc_info=True)
                        else:
                            logger.error("[PROCESS-STEP-5] ERROR: Final recovery created no files (message_id=%s)", message_id, exc_info=True)
                    
                    # Only log warning if all recovery attempts failed
                    logger.error("[PROCESS-STEP-7] FINAL STATE: No files to send for queued message (author_id=%s, message_id=%s, has_attachments=%s, has_stickers=%s, attachment_files=%d, sticker_files=%d, _attachment_data=%d items)", 
                               author_id, message_id, has_attachments, has_stickers, len(attachment_files), len(sticker_files),
                               len(message._attachment_data) if hasattr(message, '_attachment_data') and message._attachment_data else 0, exc_info=True)
                
                return True
            
            # If we have files to send, send them and handle original message like VN bot
            if files:
                author_id = message.author.id if message.author else 'unknown'
                message_id = message.id if hasattr(message, 'id') else 'unknown'
                # DIAGNOSTIC: Track admin/player status during VN panel processing
                is_gm_vn = self._is_actual_gm(message.author, game_state) if message.author and game_state else False
                is_admin_vn = (is_admin(message.author) or is_bot_mod(message.author)) if message.author else False
                player_vn = game_state.players.get(author_id) if game_state else None
                has_character_vn = player_vn and player_vn.character_name
                logger.info("[VN-PANEL-STEP-1] Processing start: VN panel send (author_id=%s, message_id=%s, character=%s, has_attachments=%s, has_stickers=%s, is_gm=%s, is_admin=%s, has_character=%s)", 
                           author_id, message_id, character_display_name, has_attachments, has_stickers, 
                           is_gm_vn, is_admin_vn, has_character_vn)
                
                # Include original message attachments (GIFs, images, etc.) so they show as previews
                # Download and convert attachments to discord.File objects
                attachment_files = []
                
                # CRITICAL: For queued messages, use _attachment_data directly instead of trying AttachmentProxy
                # AttachmentProxy.read() often fails or returns empty bytes for queued messages
                if is_queued and hasattr(message, '_attachment_data') and message._attachment_data:
                    logger.info("[VN-PANEL-STEP-2] Queued message: Using _attachment_data directly (%d items, message_id=%s)", 
                               len(message._attachment_data), message_id)
                    for att_idx, att_data in enumerate(message._attachment_data, 1):
                        filename = att_data.get('filename', 'unknown')
                        logger.info("[VN-PANEL-STEP-2] Processing queued attachment %d/%d: %s (message_id=%s)", 
                                   att_idx, len(message._attachment_data), filename, message_id)
                        try:
                            att_bytes = att_data.get('bytes', b'')
                            byte_count = len(att_bytes) if att_bytes else 0
                            logger.info("[VN-PANEL-STEP-2] Extracted bytes from _attachment_data: %s (byte_count=%d)", 
                                       filename, byte_count)
                            
                            if att_bytes and len(att_bytes) > 0:
                                logger.info("[VN-PANEL-STEP-4] Creating discord.File from _attachment_data: %s (byte_count=%d)", 
                                           filename, byte_count)
                                attachment_file = discord.File(
                                    io.BytesIO(att_bytes),
                                    filename=filename
                                )
                                attachment_files.append(attachment_file)
                                logger.info("[VN-PANEL-STEP-4] SUCCESS: Created attachment file from _attachment_data (queued message): %s (byte_count=%d)", 
                                           filename, byte_count)
                            else:
                                logger.error("[VN-PANEL-STEP-2] ERROR: Bytes are empty for queued attachment %s (message_id=%s)", 
                                           filename, message_id, exc_info=True)
                        except Exception as exc:
                            logger.error("[VN-PANEL-STEP-2] ERROR: Failed to create file from _attachment_data for queued message %s: %s", 
                                       filename, exc, exc_info=True)
                elif message.attachments and len(message.attachments) > 0:
                    # CRITICAL FIX: Validate attachment count > 0, not just truthy check
                    # This ensures we only process when attachments actually exist (same logic as direct send path)
                    attachment_count = len(message.attachments)
                    logger.info("[VN-PANEL-STEP-2] Attachment read loop: Processing %d attachment(s) (message_id=%s)", 
                               attachment_count, message_id)
                    for att_idx, attachment in enumerate(message.attachments, 1):
                        # Validate attachment object before processing
                        if not attachment or not hasattr(attachment, 'filename'):
                            logger.warning("[VN-PANEL-STEP-2] WARNING: Invalid attachment object at index %d (VN panel, message_id=%s), skipping", 
                                          att_idx, message_id)
                            continue
                        
                        logger.info("[VN-PANEL-STEP-2] Reading attachment %d/%d: %s (message_id=%s)", 
                                   att_idx, attachment_count, attachment.filename, message_id)
                        attachment_bytes = None
                        try:
                            logger.info("[VN-PANEL-STEP-2] Attempting to read via AttachmentProxy.read(): %s", attachment.filename)
                            attachment_bytes = await attachment.read()
                            byte_count = len(attachment_bytes) if attachment_bytes else 0
                            logger.info("[VN-PANEL-STEP-2] AttachmentProxy.read() completed: %s (byte_count=%d)", 
                                       attachment.filename, byte_count)
                            
                            # Check if bytes are empty (not just if exception occurs)
                            if not attachment_bytes or len(attachment_bytes) == 0:
                                logger.error("[VN-PANEL-STEP-2] ERROR: AttachmentProxy.read() returned empty bytes for %s (VN panel, message_id=%s)", 
                                           attachment.filename, message_id, exc_info=True)
                                # Fall through to fallback logic below
                            else:
                                # Bytes are valid - create file
                                logger.info("[VN-PANEL-STEP-4] Creating discord.File from AttachmentProxy bytes: %s (byte_count=%d)", 
                                           attachment.filename, byte_count)
                                try:
                                    attachment_file = discord.File(
                                        io.BytesIO(attachment_bytes),
                                        filename=attachment.filename
                                    )
                                    attachment_files.append(attachment_file)
                                    logger.info("[VN-PANEL-STEP-4] SUCCESS: Added original attachment %s to files list (byte_count=%d)", 
                                               attachment.filename, byte_count)
                                    continue  # Successfully created file, move to next attachment
                                except Exception as file_exc:
                                    logger.error("[VN-PANEL-STEP-4] ERROR: Failed to create discord.File for %s: %s", 
                                               attachment.filename, file_exc, exc_info=True)
                        except Exception as exc:
                            logger.error("[VN-PANEL-STEP-2] ERROR: AttachmentProxy.read() failed for %s (VN panel): %s", 
                                       attachment.filename, exc, exc_info=True)
                            attachment_bytes = None  # Mark as failed
                elif message.attachments is not None and len(message.attachments) == 0:
                    # CRITICAL FIX: If message.attachments is empty but _attachment_data exists, process all items directly
                    # This ensures non-admin players can use GIFs just like admins
                    if hasattr(message, '_attachment_data') and message._attachment_data:
                        logger.info("[VN-PANEL-STEP-2] message.attachments is empty, processing _attachment_data directly (%d items, message_id=%s)", 
                                   len(message._attachment_data), message_id)
                        for att_idx, att_data in enumerate(message._attachment_data, 1):
                            filename = att_data.get('filename', 'unknown')
                            logger.info("[VN-PANEL-STEP-2] Processing attachment %d/%d from _attachment_data: %s (message_id=%s)", 
                                       att_idx, len(message._attachment_data), filename, message_id)
                            try:
                                att_bytes = att_data.get('bytes', b'')
                                byte_count = len(att_bytes) if att_bytes else 0
                                logger.info("[VN-PANEL-STEP-2] Extracted bytes from _attachment_data: %s (byte_count=%d)", 
                                           filename, byte_count)
                                
                                if att_bytes and len(att_bytes) > 0:
                                    logger.info("[VN-PANEL-STEP-4] Creating discord.File from _attachment_data: %s (byte_count=%d)", 
                                               filename, byte_count)
                                    attachment_file = discord.File(
                                        io.BytesIO(att_bytes),
                                        filename=filename
                                    )
                                    attachment_files.append(attachment_file)
                                    logger.info("[VN-PANEL-STEP-4] SUCCESS: Created attachment file from _attachment_data: %s (byte_count=%d)", 
                                               filename, byte_count)
                                else:
                                    logger.error("[VN-PANEL-STEP-2] ERROR: Bytes are empty for attachment %s (message_id=%s)", 
                                               filename, message_id, exc_info=True)
                            except Exception as exc:
                                logger.error("[VN-PANEL-STEP-2] ERROR: Failed to create file from _attachment_data for %s: %s", 
                                           filename, exc, exc_info=True)
                
                # Fallback: if no attachments were created but _attachment_data exists, create files directly
                if not attachment_files and hasattr(message, '_attachment_data') and message._attachment_data:
                    logger.info("[VN-PANEL-STEP-3] No attachments created from message.attachments, trying _attachment_data directly (%d items, message_id=%s)", 
                              len(message._attachment_data), message_id)
                    for fallback_idx, att_data in enumerate(message._attachment_data, 1):
                        logger.info("[VN-PANEL-STEP-3] Processing fallback attachment %d/%d: %s", 
                                   fallback_idx, len(message._attachment_data), att_data.get('filename', 'unknown'))
                        try:
                            att_bytes = att_data.get('bytes', b'')
                            byte_count = len(att_bytes) if att_bytes else 0
                            logger.info("[VN-PANEL-STEP-3] Extracted bytes from _attachment_data: %s (byte_count=%d)", 
                                       att_data.get('filename', 'unknown'), byte_count)
                            if att_bytes and len(att_bytes) > 0:
                                logger.info("[VN-PANEL-STEP-4] Creating discord.File from fallback _attachment_data: %s (byte_count=%d)", 
                                           att_data.get('filename', 'unknown'), byte_count)
                                fallback_file = discord.File(
                                    io.BytesIO(att_bytes),
                                    filename=att_data.get('filename', 'attachment')
                                )
                                attachment_files.append(fallback_file)
                                logger.info("[VN-PANEL-STEP-4] SUCCESS: Created attachment file directly from _attachment_data (VN panel): %s (byte_count=%d)", 
                                           att_data.get('filename', 'attachment'), byte_count)
                            else:
                                logger.error("[VN-PANEL-STEP-3] ERROR: Fallback bytes are empty for %s (VN panel, message_id=%s)", 
                                           att_data.get('filename', 'unknown'), message_id, exc_info=True)
                        except Exception as exc:
                            logger.error("[VN-PANEL-STEP-3] ERROR: Failed to create file from _attachment_data item %s (VN panel): %s", 
                                       att_data.get('filename', 'unknown'), exc, exc_info=True)
                
                # Include sticker files if available (from queued messages)
                sticker_files = []
                if hasattr(message, 'sticker_files') and message.sticker_files:
                    sticker_files = message.sticker_files
                    logger.info("[VN-PANEL-STEP-1] Including %d sticker file(s) from queued message in VN panel (message_id=%s)", 
                               len(sticker_files), message_id)
                else:
                    logger.info("[VN-PANEL-STEP-1] No sticker files available for VN panel (message_id=%s)", message_id)
                
                # Include embed files if available (from queued messages)
                embed_files = []
                if hasattr(message, 'embed_files') and message.embed_files:
                    embed_files = message.embed_files
                    logger.info("[VN-PANEL-STEP-1] Including %d embed file(s) from queued message in VN panel (message_id=%s)", 
                               len(embed_files), message_id)
                
                # Process embeds from message.embeds if not already processed (for non-queued messages)
                if not embed_files and message.embeds:
                    for embed in message.embeds:
                        image_url = None
                        if embed.image and embed.image.url:
                            image_url = embed.image.url
                        elif embed.video and embed.video.url:
                            image_url = embed.video.url
                        elif embed.thumbnail and embed.thumbnail.url:
                            image_url = embed.thumbnail.url
                        
                        if image_url:
                            try:
                                async with aiohttp.ClientSession() as session:
                                    async with session.get(image_url) as resp:
                                        if resp.status == 200:
                                            embed_bytes = await resp.read()
                                            if embed_bytes and len(embed_bytes) > 0:
                                                filename = image_url.split('/')[-1].split('?')[0] or 'embed_image.gif'
                                                embed_files.append(discord.File(
                                                    io.BytesIO(embed_bytes),
                                                    filename=filename
                                                ))
                                                logger.info("[VN-PANEL-STEP-4] Created embed file: %s (byte_count=%d)", filename, len(embed_bytes))
                            except Exception as exc:
                                logger.error("[VN-PANEL-STEP-4] Failed to process embed: %s", exc)
                
                # Process link URLs when embeds are missing (link-based GIFs)
                if not embed_files and has_link_embeds and link_urls:
                    for url_idx, link_url in enumerate(link_urls, 1):
                        try:
                            async with aiohttp.ClientSession() as session:
                                async with session.get(link_url) as resp:
                                    if resp.status != 200:
                                        continue
                                    content_type = resp.headers.get("Content-Type", "")
                                    if content_type and not (content_type.startswith("image/") or content_type.startswith("video/")):
                                        continue
                                    link_bytes = await resp.read()
                                    if not link_bytes:
                                        continue
                                    filename = link_url.split("/")[-1].split("?")[0] or "embed_link.gif"
                                    embed_files.append(discord.File(io.BytesIO(link_bytes), filename=filename))
                        except Exception as exc:
                            logger.debug("[VN-PANEL-STEP-4] Failed to download link URL %d/%d for VN panel: %s", url_idx, len(link_urls), exc)
                
                # Combine VN panel file with original attachments, stickers, and embeds
                # For non-queued messages with text, keep original media and only send VN panel text
                if cleaned_content and not is_queued:
                    all_files = files
                else:
                    all_files = files + attachment_files + sticker_files + embed_files
                logger.info("[VN-PANEL-STEP-4] File preparation summary: VN panel files=%d, attachment_files=%d, sticker_files=%d, embed_files=%d, total=%d (message_id=%s)", 
                           len(files), len(attachment_files), len(sticker_files), len(embed_files), len(all_files), message_id)
                
                # Ensure we have files to send - if attachments were detected but not created, try one more time
                if not attachment_files and hasattr(message, '_attachment_data') and message._attachment_data:
                    logger.warning("[VN-PANEL-STEP-5] No attachment files created for VN panel despite _attachment_data existing, attempting final fallback (message_id=%s, _attachment_data_items=%d)", 
                                 message_id, len(message._attachment_data))
                    for final_idx, att_data in enumerate(message._attachment_data, 1):
                        logger.info("[VN-PANEL-STEP-5] Final fallback attempt %d/%d: %s", 
                                   final_idx, len(message._attachment_data), att_data.get('filename', 'unknown'))
                        try:
                            att_bytes = att_data.get('bytes', b'')
                            byte_count = len(att_bytes) if att_bytes else 0
                            logger.info("[VN-PANEL-STEP-5] Extracted bytes for final fallback: %s (byte_count=%d)", 
                                       att_data.get('filename', 'unknown'), byte_count)
                            if att_bytes and len(att_bytes) > 0:
                                logger.info("[VN-PANEL-STEP-4] Creating discord.File from final fallback: %s (byte_count=%d)", 
                                           att_data.get('filename', 'unknown'), byte_count)
                                final_fallback_file = discord.File(
                                    io.BytesIO(att_bytes),
                                    filename=att_data.get('filename', 'attachment')
                                )
                                attachment_files.append(final_fallback_file)
                                all_files.append(final_fallback_file)
                                logger.info("[VN-PANEL-STEP-5] SUCCESS: Final fallback created attachment file: %s (byte_count=%d)", 
                                           att_data.get('filename', 'attachment'), byte_count)
                            else:
                                logger.error("[VN-PANEL-STEP-5] ERROR: Final fallback bytes are empty for %s (VN panel, message_id=%s)", 
                                           att_data.get('filename', 'unknown'), message_id, exc_info=True)
                        except Exception as exc:
                            logger.error("[VN-PANEL-STEP-5] ERROR: Final fallback failed to create file from _attachment_data for %s: %s", 
                                       att_data.get('filename', 'unknown'), exc, exc_info=True)
                
                logger.info("[VN-PANEL-STEP-6] Preparing to send VN panel with files (VN panel files=%d, attachment_files=%d, sticker_files=%d, total=%d, message_id=%s)", 
                           len(files), len(attachment_files), len(sticker_files), len(all_files), message_id)
                send_kwargs: Dict[str, object] = {
                    "files": all_files,
                    "allowed_mentions": discord.AllowedMentions.none(),
                }
                
                # Preserve reply reference if present
                if message.reference:
                    send_kwargs["reference"] = message.reference
                    logger.info("[VN-PANEL-STEP-6] Added message reference to send_kwargs (reference_id=%s)", 
                               message.reference.message_id if message.reference else 'None')
                
                # Match VN bot: preserve_original = has_attachments or has_stickers or has_links
                # Note: VN bot should also check stickers, but for now we'll add it here to match expected behavior
                preserve_original = has_attachments or has_stickers or has_links
                deleted = False
                
                try:
                    logger.info("[VN-PANEL-STEP-6] Sending VN panel to channel (channel_id=%s, file_count=%d, message_id=%s)", 
                               message.channel.id if message.channel else 'unknown', len(all_files), message_id)
                    await message.channel.send(**send_kwargs)
                    logger.info("[VN-PANEL-STEP-6] SUCCESS: Sent VN panel with %d file(s) (VN panel: %d, attachments: %d, stickers: %d) for game player %s (message_id=%s)", 
                               len(all_files), len(files), len(attachment_files), len(sticker_files), author_id, message_id)
                    
                    # Handle original message - match VN bot behavior exactly
                    # For queued messages: Skip deletion/editing (already deleted when queued)
                    # For non-queued messages: Delete or edit based on preserve_original
                    if not is_queued:
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
                        # Note: Stickers cannot be edited out, so we only edit if there are attachments
                        if has_attachments and not has_links:
                            placeholder = "\u200b"
                            if message.content != placeholder:
                                try:
                                    await message.edit(content=placeholder, attachments=message.attachments, suppress=True)
                                except discord.HTTPException as exc:
                                    logger.debug("Unable to clear attachment message %s: %s", message.id, exc)
                    else:
                        # Queued message - already deleted when queued, skip deletion/editing
                        logger.info("[VN-PANEL-STEP-6] Skipping deletion/editing for queued message (already deleted, message_id=%s)", message_id)
                except discord.HTTPException as exc:
                    logger.error("[VN-PANEL-STEP-6] ERROR: Failed to send game VN panel (message_id=%s, files: VN panel=%d, attachments=%d, stickers=%d, total=%d): %s", 
                               message_id, len(files), len(attachment_files), len(sticker_files), len(all_files), exc, exc_info=True)
                    # Log final state if send failed
                    logger.error("[VN-PANEL-STEP-7] FINAL STATE: VN panel send failed (author_id=%s, message_id=%s, has_attachments=%s, has_stickers=%s, VN_panel_files=%d, attachment_files=%d, sticker_files=%d, total_files=%d, _attachment_data=%d items)", 
                               author_id, message_id, has_attachments, has_stickers, len(files), len(attachment_files), len(sticker_files), len(all_files),
                               len(message._attachment_data) if hasattr(message, '_attachment_data') and message._attachment_data else 0, exc_info=True)
            elif has_attachments or has_stickers:
                # No VN panel but message has attachments or stickers - preserve original message (match VN bot)
                # VN bot doesn't delete messages with attachments/stickers unless they interrupt calculations
                logger.info("[VN-PANEL-STEP-7] Preserving message with attachments/stickers (no VN panel) for game player %s (message_id=%s, has_attachments=%s, has_stickers=%s)", 
                          message.author.id, message.id if hasattr(message, 'id') else 'unknown', has_attachments, has_stickers)
                # Don't delete - attachments/stickers should remain in original message
            else:
                logger.warning("[VN-PANEL-STEP-7] FINAL STATE: No VN panel file created for game player %s as %s (message_id=%s, MESSAGE_STYLE=%s, files=%s, has_attachments=%s, has_stickers=%s)", 
                             message.author.id, character_display_name, message.id if hasattr(message, 'id') else 'unknown', MESSAGE_STYLE, len(files) if 'files' in locals() else 0, has_attachments, has_stickers)
            
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
            
            # Process attachments, stickers, and embeds (same as VN panel path)
            attachment_files = []
            sticker_files = []
            embed_files = []
            
            # Process attachments (check _attachment_data first, fallback to message.attachments)
            if hasattr(message, '_attachment_data') and message._attachment_data:
                for att_data in message._attachment_data:
                    try:
                        att_bytes = att_data.get('bytes', b'')
                        if att_bytes and len(att_bytes) > 0:
                            attachment_files.append(discord.File(
                                io.BytesIO(att_bytes),
                                filename=att_data.get('filename', 'attachment')
                            ))
                    except Exception as exc:
                        logger.error("Failed to create attachment file for narrator: %s", exc)
            elif message.attachments:
                for attachment in message.attachments:
                    try:
                        attachment_bytes = await attachment.read()
                        if attachment_bytes and len(attachment_bytes) > 0:
                            attachment_files.append(discord.File(
                                io.BytesIO(attachment_bytes),
                                filename=attachment.filename
                            ))
                    except Exception as exc:
                        logger.error("Failed to read attachment for narrator: %s", exc)
            
            # Process stickers (check sticker_files first, fallback to message.stickers)
            if hasattr(message, 'sticker_files') and message.sticker_files:
                sticker_files = message.sticker_files
            elif message.stickers:
                for sticker in message.stickers:
                    try:
                        if hasattr(sticker, 'url') and sticker.url:
                            async with aiohttp.ClientSession() as session:
                                async with session.get(sticker.url) as resp:
                                    if resp.status == 200:
                                        sticker_bytes = await resp.read()
                                        if sticker_bytes and len(sticker_bytes) > 0:
                                            sticker_files.append(discord.File(
                                                io.BytesIO(sticker_bytes),
                                                filename=f"sticker_{sticker.id}.png"
                                            ))
                    except Exception as exc:
                        logger.error("Failed to process sticker for narrator: %s", exc)
            
            # Process embeds (extract images from embeds)
            if message.embeds:
                for embed in message.embeds:
                    image_url = None
                    if embed.image and embed.image.url:
                        image_url = embed.image.url
                    elif embed.video and embed.video.url:
                        image_url = embed.video.url
                    elif embed.thumbnail and embed.thumbnail.url:
                        image_url = embed.thumbnail.url
                    
                    if image_url:
                        try:
                            async with aiohttp.ClientSession() as session:
                                async with session.get(image_url) as resp:
                                    if resp.status == 200:
                                        embed_bytes = await resp.read()
                                        if embed_bytes and len(embed_bytes) > 0:
                                            filename = image_url.split('/')[-1].split('?')[0] or 'embed_image.gif'
                                            embed_files.append(discord.File(
                                                io.BytesIO(embed_bytes),
                                                filename=filename
                                            ))
                        except Exception as exc:
                            logger.error("Failed to process embed for narrator: %s", exc)
            
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
            vn_file = await run_panel_render_vn(
                render_vn_panel,
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
                    "files": [vn_file] + attachment_files + sticker_files + embed_files,
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
            try:
                result = pack.call("get_player_number", game_state, user_id)
                if result is not None:
                    return result
            except Exception as exc:
                logger.warning("Failed to call pack.get_player_number for user %s: %s", user_id, exc)
                # Fall through to fallback logic
        
        # Fallback: read directly from pack_data if pack function not available or returns None
        if hasattr(game_state, '_pack_data') and game_state._pack_data:
            player_numbers = game_state._pack_data.get('player_numbers', {})
            if isinstance(player_numbers, dict):
                # Try direct lookup first (user_id as int)
                if user_id in player_numbers:
                    return player_numbers[user_id]
                # Fallback: try string key (JSON deserialization might store keys as strings)
                if str(user_id) in player_numbers:
                    return player_numbers[str(user_id)]
        
        return None
    
    def _get_next_player_info(self, game_state: GameState, pack: Optional[Any], guild: Optional[discord.Guild]) -> Optional[Tuple[int, str, int, str]]:
        """Get next player info for turn indicator. Returns (player_number, character_name, user_id, username) or None."""
        if not pack:
            return None
        
        try:
            if not pack.has_function("get_game_data"):
                return None
            
            try:
                data = pack.call("get_game_data", game_state)
            except Exception as exc:
                logger.warning("Failed to call pack.get_game_data in _get_next_player_info: %s", exc)
                return None
            turn_order = data.get('turn_order', [])
            player_numbers = data.get('player_numbers', {})
            
            # Get win tile for filtering
            game_config = self.get_game_config(game_state.game_type)
            if game_config:
                rules = game_config.rules or {}
                win_tile = int(rules.get("win_tile", 100))
            else:
                win_tile = 100
            
            # Find first player in turn_order who hasn't rolled AND isn't at goal AND hasn't forfeited
            forfeited_players = set(data.get('forfeited_players', []))
            for user_id in turn_order:
                # Skip forfeited players
                if user_id in forfeited_players:
                    continue
                
                # Skip players at goal tile
                tile_num = data.get('tile_numbers', {}).get(user_id, 1)
                if tile_num >= win_tile:
                    continue
                
                # Check if player has rolled (for end-of-turn indicator)
                # For game start, all players are eligible
                if data.get('players_rolled_this_turn'):
                    if user_id in data['players_rolled_this_turn']:
                        continue
                
                # Found next player
                player = game_state.players.get(user_id)
                if not player:
                    continue
                
                # Get player number (int, or None if not found)
                player_num = player_numbers.get(user_id)
                if player_num is None:
                    # Player number not found - skip this player (shouldn't happen, but handle gracefully)
                    logger.warning("Player %s found in turn_order but no player_number assigned", user_id)
                    continue
                
                character_name = player.character_name or f"Player {player_num}"
                
                # Get username
                username = f"User {user_id}"
                if guild:
                    member = guild.get_member(user_id)
                    if member:
                        username = member.display_name
                
                return (player_num, character_name, user_id, username)
            
            return None
        except Exception as exc:
            logger.warning("Error getting next player info: %s", exc)
            return None
    
    def _swap_pack_player_metadata(self, game_state: GameState, user_id1: int, user_id2: int) -> None:
        """Swap character-specific metadata only. Does NOT swap player_numbers or turn_order."""
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
        
        # Swap tile_numbers (positions swap with characters - tokens are tied to characters)
        # This is handled in command_swap before calling this function
        # tile_numbers swapping is done there to ensure proper order
        
        # CRITICAL: Do NOT swap player_numbers - player numbers never change
        # player_numbers = data.get("player_numbers")
        # if isinstance(player_numbers, dict):
        #     _swap_dict_entry(player_numbers)
        
        # CRITICAL: Do NOT swap turn_order - turn order never changes
        # turn_order = data.get("turn_order")
        # if isinstance(turn_order, list):
        #     try:
        #         idx1 = turn_order.index(user_id1)
        #         idx2 = turn_order.index(user_id2)
        #     except ValueError:
        #         idx1 = idx2 = None
        #     if idx1 is not None and idx2 is not None:
        #         turn_order[idx1], turn_order[idx2] = turn_order[idx2], turn_order[idx1]
        
        # CRITICAL: Do NOT swap player-specific lists - these are tied to player numbers, not characters
        # for list_key in ("players_rolled_this_turn", "winners", "players_reached_end_this_turn"):
        #     seq = data.get(list_key)
        #     if isinstance(seq, list):
        #         _swap_list_entries(seq)
        
        # CRITICAL: Do NOT swap goal_reached_turn - this is tied to player numbers, not characters
        # goal_reached_turn = data.get("goal_reached_turn")
        # if isinstance(goal_reached_turn, dict):
        #     _swap_dict_entry(goal_reached_turn)
        
        # Only swap character-specific metadata (tied to character, not player)
        def _swap_dict_entry(mapping: Dict[int, object]) -> None:
            if not isinstance(mapping, dict):
                return
            val1 = mapping.get(user_id1)
            val2 = mapping.get(user_id2)
            if val1 is None and val2 is None:
                return
            mapping[user_id1], mapping[user_id2] = val2, val1
        
        for char_key in ("original_characters", "real_body_characters", "transformation_counts", "mind_changed"):
            char_dict = data.get(char_key)
            if isinstance(char_dict, dict):
                _swap_dict_entry(char_dict)

    def list_available_games(self) -> List[str]:
        """Get list of available game types."""
        if not self._has_packs:
            return []  # No games available if no packs
        return list(self._game_configs.keys())

    def _is_gm(self, member: Optional[discord.Member], game_state: Optional[GameState] = None) -> bool:
        """Check if member is GM for a game or is admin."""
        if not isinstance(member, discord.Member):
            return False
        if is_admin(member) or is_bot_mod(member):
            return True
        if game_state and game_state.gm_user_id == member.id:
            return True
        return False
    
    def _is_actual_gm(self, member: Optional[discord.Member], game_state: Optional[GameState] = None) -> bool:
        """Check if member is the actual GM for a game (not just admin)."""
        if not isinstance(member, discord.Member):
            return False
        if game_state and game_state.gm_user_id == member.id:
            return True
        return False
    
    def _build_swap_chain(self, game_state: GameState, start_user_id: int) -> List[Tuple[int, int]]:
        """
        Build swap chain starting from a user_id.
        Returns list of (user_id1, user_id2) pairs representing swaps in order.
        Example: [(1, 2), (2, 3)] means Player 1 swapped with 2, then 2 swapped with 3.
        
        For a simple two-player swap (A <-> B), this will return [(A, B)].
        For chain swaps (A <-> B <-> C), this will return [(A, B), (B, C)].
        """
        chain = []
        visited = set()
        current_id = start_user_id
        
        # Follow form_owner_user_id links to build chain
        while current_id and current_id not in visited:
            visited.add(current_id)
            state = game_state.player_states.get(current_id)
            if not state or not state.form_owner_user_id:
                break
            if state.form_owner_user_id == current_id:
                # Not swapped
                break
            # Found a swap: current_id is swapped with form_owner_user_id
            other_id = state.form_owner_user_id
            
            # Check if this swap pair is already in the chain (avoid duplicates)
            swap_pair = (current_id, other_id)
            reverse_pair = (other_id, current_id)
            if swap_pair not in chain and reverse_pair not in chain:
                chain.append(swap_pair)
            
            # Move to the other player to continue building the chain
            current_id = other_id
        
        return chain
    
    async def _revert_swap_chain(self, ctx: commands.Context, game_state: GameState, swap_chain: List[Tuple[int, int]]) -> None:
        """
        Revert all swaps in a chain in reverse order.
        Example: Chain [(1, 2), (2, 3)] reverts by swapping (2, 3) back, then (1, 2) back.
        
        For a simple two-player swap (A <-> B), reverting [(A, B)] will swap them back.
        """
        if not swap_chain:
            return
        
        # Revert in reverse order
        for user_id1, user_id2 in reversed(swap_chain):
            await self._revert_swap_direct(game_state, user_id1, user_id2, ctx)
        
        # Update board and save after all swaps are reverted
        await self._update_board(game_state, error_channel=ctx.channel, description_text="Swap reverted")
        await self._save_auto_save(game_state, ctx)
    
    async def _revert_swap_direct(self, game_state: GameState, user_id1: int, user_id2: int, ctx: commands.Context) -> None:
        """
        Directly revert a swap between two players without going through command flow.
        This is used internally by _revert_swap_chain to ensure proper state management.
        """
        # Get players and states
        player1 = game_state.players.get(user_id1)
        player2 = game_state.players.get(user_id2)
        state1 = game_state.player_states.get(user_id1)
        state2 = game_state.player_states.get(user_id2)
        
        if not player1 or not player2 or not state1 or not state2:
            logger.warning("Cannot revert swap: missing player or state for %s or %s", user_id1, user_id2)
            return
        
        # Swap characters
        char1 = player1.character_name
        char2 = player2.character_name
        player1.character_name = char2
        player2.character_name = char1
        
        # Swap grid positions
        pos1 = player1.grid_position
        pos2 = player2.grid_position
        player1.grid_position = pos2
        player2.grid_position = pos1
        
        # Swap backgrounds
        bg1 = player1.background_id
        bg2 = player2.background_id
        player1.background_id = bg2
        player2.background_id = bg1
        
        # Swap character data from states
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
        
        # Preserve user identity
        identity1 = state1.identity_display_name or state1.character_name
        identity2 = state2.identity_display_name or state2.character_name
        
        # Create new states with swapped character data
        # CRITICAL: Set form_owner_user_id to player's own ID (not swapped)
        from tfbot.models import TransformationState
        from datetime import datetime, timezone
        
        new_state1 = TransformationState(
            user_id=user_id1,
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
            form_owner_user_id=user_id1,  # Reset to own ID (not swapped)
            identity_display_name=identity1,
            is_pillow=state1.is_pillow,
        )
        
        new_state2 = TransformationState(
            user_id=user_id2,
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
            form_owner_user_id=user_id2,  # Reset to own ID (not swapped)
            identity_display_name=identity2,
            is_pillow=state2.is_pillow,
        )
        
        # Update player_states
        game_state.player_states[user_id1] = new_state1
        game_state.player_states[user_id2] = new_state2
        
        # Update pack-specific metadata (tile_numbers)
        pack = get_game_pack(game_state.game_type, self.packs_dir)
        if pack and pack.has_function("get_game_data"):
            pack_module = pack.module
            get_game_data = getattr(pack_module, "get_game_data", None)
            if callable(get_game_data):
                try:
                    data = get_game_data(game_state)
                    tile_numbers = data.get('tile_numbers', {})
                    if user_id1 in tile_numbers and user_id2 in tile_numbers:
                        tile1 = tile_numbers[user_id1]
                        tile2 = tile_numbers[user_id2]
                        tile_numbers[user_id1] = tile2
                        tile_numbers[user_id2] = tile1
                        data['tile_numbers'] = tile_numbers
                except Exception as exc:
                    logger.warning("Failed to update tile_numbers during swap reversion: %s", exc)
        
        # Swap character-related metadata
        self._swap_pack_player_metadata(game_state, user_id1, user_id2)
        
        # Notify pack about character swaps
        if pack and pack.has_function("on_character_assigned"):
            try:
                pack.call("on_character_assigned", game_state, player1, char2)
            except Exception as exc:
                logger.exception("Error in pack.on_character_assigned for player1 during swap reversion: %s", exc)
            
            try:
                pack.call("on_character_assigned", game_state, player2, char1)
            except Exception as exc:
                logger.exception("Error in pack.on_character_assigned for player2 during swap reversion: %s", exc)
        
        logger.info("Reverted swap between %s and %s", user_id1, user_id2)
    
    def _is_invalid_command(self, message: discord.Message) -> bool:
        """Check if message starts with ! but is not a valid command for a player."""
        if not message.content or not message.content.strip().startswith('!'):
            return False
        # Check if it's a valid player command
        # Valid player commands in gameboard: dice, gamequit, players, help, rules
        # All other ! commands are invalid for players
        content = message.content.strip()
        valid_player_commands = {'!dice', '!gamequit', '!players', '!help', '!rules'}
        # Check if it matches any valid command (case-insensitive)
        cmd_lower = content.lower().split()[0] if content.split() else ''
        return cmd_lower not in valid_player_commands
    
    async def _cache_and_delete_message(
        self,
        message: discord.Message,
        thread_id: int,
        reason: str = "",
        queue_thread_id: Optional[int] = None,
    ) -> None:
        """Cache message data and delete it, to be reprinted after command completes.
        
        Only caches if a command is currently processing (lock is held).
        Invalid ! commands are deleted immediately without caching.
        """
        # Check if this is an invalid command - delete immediately without caching
        if self._is_invalid_command(message):
            try:
                await message.delete()
                logger.debug("Deleted invalid command message from %s: %s", message.author.id, reason)
            except discord.HTTPException:
                pass
            return
        
        # Check if command is processing (lock is held)
        queue_id = queue_thread_id or thread_id
        command_lock = self._get_command_lock(queue_id)
        if command_lock.locked():
            try:
                # Command is processing - cache message for later reprinting
                logger.debug("Command processing - caching message from %s: %s", message.author.id, reason)
                
                # Download attachments before deleting message
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
                        except Exception as exc:
                            logger.warning("Failed to download attachment %s for queuing: %s", attachment.filename, exc)
                
                # Download stickers before deleting message (stickers become inaccessible after deletion)
                sticker_data = []
                if hasattr(message, 'stickers') and message.stickers:
                    for sticker in message.stickers:
                        try:
                            # Download sticker image
                            if hasattr(sticker, 'url') and sticker.url:
                                async with aiohttp.ClientSession() as session:
                                    async with session.get(sticker.url) as resp:
                                        if resp.status == 200:
                                            sticker_bytes = await resp.read()
                                            sticker_id = sticker.id if hasattr(sticker, 'id') else 'unknown'
                                            sticker_data.append({
                                                'name': sticker.name if hasattr(sticker, 'name') else 'sticker',
                                                'bytes': sticker_bytes,
                                                'filename': f"sticker_{sticker_id}.png",  # Stickers are typically PNG
                                            })
                                            logger.debug("Downloaded sticker %s (%d bytes) for queuing", 
                                                       sticker.id if hasattr(sticker, 'id') else 'unknown', len(sticker_bytes))
                        except Exception as exc:
                            logger.warning("Failed to download sticker %s for queuing: %s", 
                                         sticker.id if hasattr(sticker, 'id') else 'unknown', exc)
                
                # Extract all necessary message data before deletion
                message_data = {
                    'content': message.content,
                    'author': message.author,
                    'channel': message.channel,
                    'guild': message.guild,
                    'reference': message.reference,
                    'attachments': attachment_data,
                    'stickers': sticker_data,  # Store downloaded sticker data, not just sticker objects
                    'id': message.id,
                }
                
                # Delete message immediately
                try:
                    await message.delete()
                except discord.HTTPException:
                    pass
                
                # Queue message data for processing after operation completes
                if queue_id not in self._message_queues:
                    self._message_queues[queue_id] = []
                self._message_queues[queue_id].append(message_data)
                logger.debug("Queued message from %s (queue size: %d): %s", 
                           message.author.id, len(self._message_queues[queue_id]), reason)
            except Exception as exc:
                logger.warning("Failed to cache message %s from %s: %s", message.id, message.author.id, exc, exc_info=True)
                await self._warn_queue_drop(
                    message.channel,
                    f"failed to cache message: {exc}",
                    message_id=getattr(message, "id", None),
                    author_id=getattr(message.author, "id", None) if message.author else None,
                )
                try:
                    await message.delete()
                except discord.HTTPException:
                    pass
        else:
            # No command processing - delete immediately without caching
            try:
                await message.delete()
                logger.debug("Deleted message from %s (no command processing): %s", message.author.id, reason)
            except discord.HTTPException:
                pass

    async def _warn_queue_drop(
        self,
        channel: Optional[discord.abc.Messageable],
        reason: str,
        *,
        message_id: Optional[int] = None,
        author_id: Optional[int] = None,
    ) -> None:
        """Log and optionally post a warning when a queued message is dropped."""
        channel_id = getattr(channel, "id", None)
        logger.warning(
            "Queued message dropped: %s (author_id=%s, message_id=%s, channel_id=%s)",
            reason,
            author_id,
            message_id,
            channel_id,
        )
        if channel and isinstance(channel, (discord.Thread, discord.TextChannel)):
            try:
                details = []
                if author_id:
                    details.append(f"author_id={author_id}")
                if message_id:
                    details.append(f"message_id={message_id}")
                detail_text = f" ({', '.join(details)})" if details else ""
                await channel.send(
                    f"WARNING: A message was dropped while processing the game queue: {reason}.{detail_text}",
                    allowed_mentions=discord.AllowedMentions.none(),
                )
            except Exception:
                pass
    
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
        
        # CRITICAL: Verify thread belongs to this bot's configured forum channel
        # This prevents multiple bot instances from processing each other's commands
        if self.forum_channel_id > 0:
            parent_forum = getattr(ctx.channel, 'parent', None)
            if parent_forum and hasattr(parent_forum, 'id'):
                if parent_forum.id != self.forum_channel_id:
                    # Thread belongs to different forum - ignore command silently
                    logger.debug("Ignoring gameboard command in thread %s (parent forum %s != configured %s)", 
                               ctx.channel.id, parent_forum.id, self.forum_channel_id)
                    return
        
        thread_id = ctx.channel.id
        command_lock = self._get_command_lock(thread_id)
        
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
        character = self._get_character_by_name(token, game_state=game_state)
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
                game_state = self._active_games[thread_id]
                # CRITICAL: Only return game state if this bot owns it
                bot_user_id = self.bot.user.id if self.bot.user else None
                if game_state.bot_user_id is not None and game_state.bot_user_id != bot_user_id:
                    logger.debug("Skipping gameboard command - owned by bot %s, this bot is %s", game_state.bot_user_id, bot_user_id)
                    return None
                return game_state
            
            # Try to detect and load existing game thread
            game_state = await self._detect_and_load_game_thread(ctx.channel)
            if game_state:
                # CRITICAL: Only return game state if this bot owns it
                bot_user_id = self.bot.user.id if self.bot.user else None
                if game_state.bot_user_id is not None and game_state.bot_user_id != bot_user_id:
                    logger.debug("Skipping gameboard command - owned by bot %s, this bot is %s", game_state.bot_user_id, bot_user_id)
                    return None
            return game_state
        
        # Could also check DM channel
        return None

    async def _save_game_state(self, game_state: GameState) -> None:
        """Save game state to disk."""
        async with self._lock:
            # Generate filename with game number, date, manual save number, and turn number
            from datetime import datetime
            now = datetime.now()
            date_str = now.strftime("%d-%m-%Y")
            
            # Extract game number and get next manual save number
            game_number = self._extract_game_number(game_state)
            if game_number is None:
                # Fallback: use thread_id if game number extraction fails
                game_number = game_state.game_thread_id
            
            manual_save_num = self._get_next_manualsave_number(game_state, date_str)
            turn_num = game_state.turn_count
            
            # Filename: save_{gamenumber}_{date}_manualsave{number}_turn{turn_num}.json
            filename = f"save_{game_number}_{date_str}_manualsave{manual_save_num}_turn{turn_num}.json"
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
                "enabled_packs": list(game_state.enabled_packs) if game_state.enabled_packs else None,
            }
            # ADD pack_data if it exists
            if hasattr(game_state, '_pack_data') and game_state._pack_data:
                data["pack_data"] = game_state._pack_data
            # ADD player_states serialization if they exist
            if game_state.player_states:
                player_states_data = {}
                for user_id, state in game_state.player_states.items():
                    # Convert TransformationState to dict using serialize_state
                    player_states_data[str(user_id)] = serialize_state(state)
                data["player_states"] = player_states_data
            # ADD bot_user_id if it exists
            if game_state.bot_user_id:
                data["bot_user_id"] = game_state.bot_user_id
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
                # Get date
                from datetime import datetime
                now = datetime.now()
                date_str = now.strftime("%d-%m-%Y")
                
                # Extract game number and get next auto-save number (1-3)
                game_number = self._extract_game_number(game_state)
                if game_number is None:
                    # Fallback: use thread_id if game number extraction fails
                    game_number = game_state.game_thread_id
                
                autosave_num = self._get_next_autosave_number(game_state, date_str)
                
                # Filename: save_{gamenumber}_{date}_autosave{number}.json
                filename = f"save_{game_number}_{date_str}_autosave{autosave_num}.json"
                state_file = self.states_dir / filename
                
                # Delete old auto-save with same number before saving (cycle overwrites)
                if state_file.exists():
                    try:
                        state_file.unlink()
                        logger.info("Deleted old auto-save for cycling: %s", state_file.name)
                    except Exception as exc:
                        logger.warning("Failed to delete old auto-save %s: %s", state_file.name, exc)
                
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
                    "enabled_packs": list(game_state.enabled_packs) if game_state.enabled_packs else None,
                }
                # ADD pack_data if it exists
                if hasattr(game_state, '_pack_data') and game_state._pack_data:
                    data["pack_data"] = game_state._pack_data
                # ADD player_states serialization if they exist
                if game_state.player_states:
                    player_states_data = {}
                    for user_id, state in game_state.player_states.items():
                        # Convert TransformationState to dict using serialize_state
                        player_states_data[str(user_id)] = serialize_state(state)
                    data["player_states"] = player_states_data
                # ADD bot_user_id if it exists
                if game_state.bot_user_id:
                    data["bot_user_id"] = game_state.bot_user_id
                
                # Ensure directory exists
                state_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Log where we're saving
                logger.debug("Saving auto-save to: %s", state_file.absolute())
                
                # Write file and verify
                json_content = json.dumps(data, indent=2)
                state_file.write_text(json_content, encoding="utf-8")
                
                # Verify file was written
                if not state_file.exists():
                    logger.error("CRITICAL: Auto-save file was not created: %s (absolute: %s)", state_file, state_file.absolute())
                    return
                
                file_size = state_file.stat().st_size
                if file_size == 0:
                    logger.error("CRITICAL: Auto-save file is 0 bytes: %s (absolute: %s)", state_file, state_file.absolute())
                    return
                
                logger.info("Auto-save created successfully: %s (%d bytes) at %s", filename, file_size, state_file.absolute())

                # Prune autosaves to newest 3 per game (mtime-based, keep filename format)
                autosave_pattern = f"save_{game_number}_*_autosave*.json"
                autosave_files = list(self.states_dir.glob(autosave_pattern))
                if len(autosave_files) > 3:
                    autosave_files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
                    for old_file in autosave_files[3:]:
                        try:
                            old_file.unlink()
                            logger.info("Deleted old auto-save: %s", old_file.name)
                        except Exception as exc:
                            logger.warning("Failed to delete auto-save %s: %s", old_file.name, exc)
            except Exception as exc:
                logger.error("CRITICAL: Failed to create auto-save: %s", exc, exc_info=True)

    async def _update_board(
        self,
        game_state: GameState,
        error_channel: Optional[discord.abc.Messageable] = None,
        target_thread: str = "map",
        also_post_to_game: bool = False,
        description_text: Optional[str] = None
    ) -> None:
        """
        Update board image.
        
        CRITICAL: This function holds the command lock during board rendering to ensure
        messages sent during rendering are cached, deleted, and reprinted in order.
        If the lock is already held (e.g., called from within a command), it proceeds without
        acquiring it again to avoid deadlock.
        
        Args:
            game_state: The game state
            error_channel: Channel to send error messages to
            target_thread: "game" to send to game thread, "map" to send to map thread (default)
            also_post_to_game: If True, also post to game thread in addition to map thread (for game start and turn start)
        """
        thread_id = game_state.game_thread_id
        command_lock = self._get_command_lock(thread_id)
        
        # CRITICAL: Acquire lock during board rendering to cache messages
        # This ensures messages sent during rendering are cached, deleted, and reprinted in order
        # If lock is already held (e.g., from parent command), proceed without acquiring (avoid deadlock)
        lock_already_held = command_lock.locked()
        
        if lock_already_held:
            # Lock already held by parent command - proceed without acquiring
            logger.debug("Board update called from within locked command - proceeding without acquiring lock")
            await self._update_board_impl(game_state, error_channel, target_thread, also_post_to_game, description_text)
        else:
            # Acquire lock and update board
            async with command_lock:
                await self._update_board_impl(game_state, error_channel, target_thread, also_post_to_game, description_text)
            
            # CRITICAL: Process queued messages after board update completes
            # This ensures messages sent during board rendering are reprinted in order
            await self._process_queued_messages(game_state)
    
    async def _update_board_impl(
        self,
        game_state: GameState,
        error_channel: Optional[discord.abc.Messageable] = None,
        target_thread: str = "map",
        also_post_to_game: bool = False,
        description_text: Optional[str] = None
    ) -> None:
        """Internal implementation of board update (called with or without lock)."""
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
            # Fallback: if map thread missing, try game thread
            if target_thread_id == game_state.map_thread_id and game_state.game_thread_id:
                target_thread_id = game_state.game_thread_id
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
        
        # OPTIMIZATION: Extract board bytes before sending (for reuse if posting to both threads)
        # discord.File objects can only be sent once, so we need the bytes to create a second File
        board_bytes = None
        board_filename = None
        if also_post_to_game and game_state.game_thread_id and game_state.game_thread_id != target_thread_id:
            # Extract bytes from the file object before it's consumed by send()
            if hasattr(board_file, 'fp') and hasattr(board_file.fp, 'read'):
                try:
                    board_file.fp.seek(0)
                    board_bytes = board_file.fp.read()
                    board_file.fp.seek(0)  # Reset for first send
                    # Extract filename to maintain format extension
                    if hasattr(board_file, 'filename'):
                        board_filename = board_file.filename
                except Exception as exc:
                    logger.debug("Could not extract board bytes for reuse: %s", exc)
        
        # Post to primary target thread (map forum by default)
        logger.info("Board image regenerated, posting to %s thread", target_thread)
        try:
            # Send description text first if provided and posting to map thread
            if description_text and target_thread == "map":
                try:
                    await thread.send(description_text, allowed_mentions=discord.AllowedMentions.none())
                    logger.debug("Sent description text to map thread: %s", description_text)
                except Exception as exc:
                    logger.warning("Failed to send description text to map thread: %s", exc)
            
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
                    # Reuse board bytes if available (avoid re-rendering)
                    if board_bytes:
                        import io
                        # Extract filename from original board_file to match format (WEBP or PNG)
                        original_filename = board_filename if board_filename else (board_file.filename if hasattr(board_file, 'filename') else "game_board.webp")
                        game_board_file = discord.File(io.BytesIO(board_bytes), filename=original_filename)
                        await game_thread.send(
                            file=game_board_file,
                            allowed_mentions=discord.AllowedMentions.none()
                        )
                        logger.info("Board also posted to game thread for visibility (using same render)")
                    else:
                        # Fallback: re-render if bytes extraction failed
                        logger.warning("Could not reuse board bytes, re-rendering for game thread")
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
            logger.info("[RECONSTRUCT-STEP-1] No queue found for thread_id=%s", thread_id)
            return
        
        queue = self._message_queues[thread_id]
        if not queue:
            logger.info("[RECONSTRUCT-STEP-1] Queue is empty for thread_id=%s", thread_id)
            return
        
        queue_size = len(queue)
        logger.info("[RECONSTRUCT-STEP-1] Queue processing start: %d queued message(s) for thread_id=%s", queue_size, thread_id)
        
        # Process all queued messages
        messages_to_process = queue.copy()
        queue.clear()  # Clear queue immediately to prevent duplicates
        logger.info("[RECONSTRUCT-STEP-1] Copied %d message(s) to process, cleared original queue", len(messages_to_process))
        
        for msg_idx, message_data in enumerate(messages_to_process, 1):
            message_id = message_data.get('id', 'unknown')
            author_obj = message_data.get('author')
            author_id = author_obj.id if author_obj else 'unknown'
            # DIAGNOSTIC: Track admin/player status during reconstruction
            is_gm_reconstruct = self._is_actual_gm(author_obj, game_state) if author_obj and game_state else False
            is_admin_reconstruct = (is_admin(author_obj) or is_bot_mod(author_obj)) if author_obj else False
            player_reconstruct = game_state.players.get(author_id) if game_state else None
            has_character_reconstruct = player_reconstruct and player_reconstruct.character_name
            logger.info("[RECONSTRUCT-STEP-2] Processing queued message %d/%d (message_id=%s, author_id=%s, is_gm=%s, is_admin=%s, has_character=%s, character_name=%s)", 
                       msg_idx, len(messages_to_process), message_id, author_id, is_gm_reconstruct, 
                       is_admin_reconstruct, has_character_reconstruct, player_reconstruct.character_name if player_reconstruct else None)
            try:
                # Recreate a message-like object from stored data
                class QueuedMessage:
                    def __init__(self, data):
                        logger.info("[RECONSTRUCT-STEP-2] Creating QueuedMessage object (message_id=%s)", data.get('id', 'unknown'))
                        self.content = data.get('content', '')
                        content_len = len(self.content) if self.content else 0
                        logger.info("[RECONSTRUCT-STEP-2] Set content (length=%d)", content_len)
                        self.author = data.get('author')
                        logger.info("[RECONSTRUCT-STEP-2] Set author (author_id=%s)", self.author.id if self.author else 'None')
                        self.channel = data.get('channel')
                        logger.info("[RECONSTRUCT-STEP-2] Set channel (channel_id=%s)", self.channel.id if self.channel else 'None')
                        self.guild = data.get('guild')
                        logger.info("[RECONSTRUCT-STEP-2] Set guild (guild_id=%s)", self.guild.id if self.guild else 'None')
                        self.reference = data.get('reference')
                        logger.info("[RECONSTRUCT-STEP-2] Set reference (reference_id=%s)", self.reference.message_id if self.reference else 'None')
                        
                        # Reconstruct attachment objects from stored data
                        self._attachment_data = data.get('attachments', [])
                        attachment_count = len(self._attachment_data)
                        logger.info("[RECONSTRUCT-STEP-3] Attachment data extraction: Found %d attachment(s) in stored data (message_id=%s)", 
                                   attachment_count, data.get('id', 'unknown'))
                        
                        # Create attachment-like objects for compatibility
                        self.attachments = []
                        skipped_count = 0
                        for att_idx, att_data in enumerate(self._attachment_data, 1):
                            filename = att_data.get('filename', 'unknown')
                            logger.info("[RECONSTRUCT-STEP-4] Processing attachment %d/%d: %s (message_id=%s)", 
                                       att_idx, attachment_count, filename, data.get('id', 'unknown'))
                            
                            # Verify bytes are present and not empty
                            att_bytes = att_data.get('bytes', b'')
                            byte_count = len(att_bytes) if att_bytes else 0
                            logger.info("[RECONSTRUCT-STEP-4] Byte validation: %s (byte_count=%d)", filename, byte_count)
                            
                            if not att_bytes or len(att_bytes) == 0:
                                logger.error("[RECONSTRUCT-STEP-4] ERROR: AttachmentProxy creation failed - bytes are empty for %s (message_id=%s)", 
                                           filename, data.get('id', 'unknown'), exc_info=True)
                                skipped_count += 1
                                continue  # Skip creating AttachmentProxy with empty bytes - fallback will handle it
                            
                            # Create a simple object that mimics discord.Attachment
                            class AttachmentProxy:
                                def __init__(self, att_data):
                                    self.filename = att_data.get('filename', 'unknown')
                                    self._bytes = att_data.get('bytes', b'')
                                    self.content_type = att_data.get('content_type')
                                    byte_count = len(self._bytes) if self._bytes else 0
                                    # Verify bytes are not empty
                                    if not self._bytes or len(self._bytes) == 0:
                                        logger.error("[RECONSTRUCT-STEP-4] ERROR: AttachmentProxy created with empty bytes for %s", 
                                                   self.filename, exc_info=True)
                                    else:
                                        logger.info("[RECONSTRUCT-STEP-4] AttachmentProxy initialized: %s (byte_count=%d, content_type=%s)", 
                                                   self.filename, byte_count, self.content_type)
                                
                                async def read(self):
                                    byte_count = len(self._bytes) if self._bytes else 0
                                    if not self._bytes or len(self._bytes) == 0:
                                        logger.error("[RECONSTRUCT-STEP-4] ERROR: AttachmentProxy.read() called but _bytes is empty for %s (message_id=%s)", 
                                                   self.filename, data.get('id', 'unknown'), exc_info=True)
                                        return b''  # Return empty bytes, fallback will handle it
                                    logger.info("[RECONSTRUCT-STEP-4] AttachmentProxy.read() returning bytes: %s (byte_count=%d, message_id=%s)", 
                                               self.filename, byte_count, data.get('id', 'unknown'))
                                    return self._bytes
                            
                            try:
                                proxy = AttachmentProxy(att_data)
                                self.attachments.append(proxy)
                                logger.info("[RECONSTRUCT-STEP-4] SUCCESS: Created AttachmentProxy for %s (byte_count=%d)", 
                                           proxy.filename, len(proxy._bytes))
                            except Exception as exc:
                                logger.error("[RECONSTRUCT-STEP-4] ERROR: Failed to create AttachmentProxy for %s: %s", 
                                           filename, exc, exc_info=True)
                                skipped_count += 1
                        
                        logger.info("[RECONSTRUCT-STEP-4] AttachmentProxy creation summary: Created %d, skipped %d (message_id=%s)", 
                                   len(self.attachments), skipped_count, data.get('id', 'unknown'))
                        
                        # Handle sticker data - convert stored sticker data to files
                        self._sticker_data = data.get('stickers', [])
                        sticker_data_count = len(self._sticker_data)
                        logger.info("[RECONSTRUCT-STEP-5] Sticker file creation: Found %d sticker(s) in stored data (message_id=%s)", 
                                   sticker_data_count, data.get('id', 'unknown'))
                        self.stickers = []  # Keep for compatibility
                        # Create sticker files from downloaded data
                        self.sticker_files = []
                        created_sticker_count = 0
                        for sticker_idx, sticker_data_item in enumerate(self._sticker_data, 1):
                            logger.info("[RECONSTRUCT-STEP-5] Processing sticker %d/%d (message_id=%s)", 
                                       sticker_idx, sticker_data_count, data.get('id', 'unknown'))
                            try:
                                if isinstance(sticker_data_item, dict) and 'bytes' in sticker_data_item:
                                    sticker_bytes = sticker_data_item.get('bytes', b'')
                                    byte_count = len(sticker_bytes) if sticker_bytes else 0
                                    filename = sticker_data_item.get('filename', 'sticker.png')
                                    logger.info("[RECONSTRUCT-STEP-5] Creating sticker file: %s (byte_count=%d)", filename, byte_count)
                                    
                                    if not sticker_bytes or len(sticker_bytes) == 0:
                                        logger.error("[RECONSTRUCT-STEP-5] ERROR: Sticker bytes are empty for %s (message_id=%s)", 
                                                   filename, data.get('id', 'unknown'), exc_info=True)
                                        continue
                                    
                                    sticker_file = discord.File(
                                        io.BytesIO(sticker_bytes),
                                        filename=filename
                                    )
                                    self.sticker_files.append(sticker_file)
                                    created_sticker_count += 1
                                    logger.info("[RECONSTRUCT-STEP-5] SUCCESS: Created sticker file: %s (byte_count=%d)", 
                                               filename, byte_count)
                                else:
                                    logger.error("[RECONSTRUCT-STEP-5] ERROR: Invalid sticker data format (not dict or missing 'bytes' key) (message_id=%s)", 
                                               data.get('id', 'unknown'), exc_info=True)
                            except Exception as exc:
                                logger.error("[RECONSTRUCT-STEP-5] ERROR: Failed to create sticker file: %s", exc, exc_info=True)
                        
                        logger.info("[RECONSTRUCT-STEP-5] Sticker file creation summary: Created %d/%d (message_id=%s)", 
                                   created_sticker_count, sticker_data_count, data.get('id', 'unknown'))
                        
                        # Handle embed data - convert stored embed data to files
                        self._embed_data = data.get('embeds', [])
                        embed_data_count = len(self._embed_data)
                        logger.info("[RECONSTRUCT-STEP-5.5] Embed file creation: Found %d embed(s) in stored data (message_id=%s)", 
                                   embed_data_count, data.get('id', 'unknown'))
                        # Create embed files from downloaded data
                        self.embed_files = []
                        # Ensure embeds attribute exists for handle_message compatibility
                        self.embeds = []
                        created_embed_count = 0
                        for embed_idx, embed_data_item in enumerate(self._embed_data, 1):
                            logger.info("[RECONSTRUCT-STEP-5.5] Processing embed %d/%d (message_id=%s)", 
                                       embed_idx, embed_data_count, data.get('id', 'unknown'))
                            try:
                                if isinstance(embed_data_item, dict) and 'bytes' in embed_data_item:
                                    embed_bytes = embed_data_item.get('bytes', b'')
                                    byte_count = len(embed_bytes) if embed_bytes else 0
                                    filename = embed_data_item.get('filename', 'embed_image.gif')
                                    logger.info("[RECONSTRUCT-STEP-5.5] Creating embed file: %s (byte_count=%d)", filename, byte_count)
                                    
                                    if not embed_bytes or len(embed_bytes) == 0:
                                        logger.error("[RECONSTRUCT-STEP-5.5] ERROR: Embed bytes are empty for %s (message_id=%s)", 
                                                   filename, data.get('id', 'unknown'), exc_info=True)
                                        continue
                                    
                                    embed_file = discord.File(
                                        io.BytesIO(embed_bytes),
                                        filename=filename
                                    )
                                    self.embed_files.append(embed_file)
                                    created_embed_count += 1
                                    logger.info("[RECONSTRUCT-STEP-5.5] SUCCESS: Created embed file: %s (byte_count=%d)", 
                                               filename, byte_count)
                                else:
                                    logger.error("[RECONSTRUCT-STEP-5.5] ERROR: Invalid embed data format (not dict or missing 'bytes' key) (message_id=%s)", 
                                               data.get('id', 'unknown'), exc_info=True)
                            except Exception as exc:
                                logger.error("[RECONSTRUCT-STEP-5.5] ERROR: Failed to create embed file: %s", exc, exc_info=True)
                        
                        logger.info("[RECONSTRUCT-STEP-5.5] Embed file creation summary: Created %d/%d (message_id=%s)", 
                                   created_embed_count, embed_data_count, data.get('id', 'unknown'))
                        self.id = data.get('id', 0)
                        # Note: Admin/player status logged outside this class after object creation
                        logger.info("[RECONSTRUCT-STEP-6] Final reconstruction summary: QueuedMessage created (message_id=%s, attachments=%d, sticker_files=%d, embed_files=%d, content_length=%d)", 
                                   self.id, len(self.attachments), len(self.sticker_files), len(self.embed_files), len(self.content) if self.content else 0)
                    
                    async def delete(self):
                        # Message was already deleted, so this is a no-op
                        pass
                
                queued_message = QueuedMessage(message_data)
                logger.info("[RECONSTRUCT-STEP-6] QueuedMessage object created successfully (message_id=%s)", queued_message.id)
                has_attachments = len(queued_message.attachments) > 0
                has_stickers = len(queued_message.sticker_files) > 0 if hasattr(queued_message, 'sticker_files') else False
                has_embeds = len(queued_message.embed_files) > 0 if hasattr(queued_message, 'embed_files') else False
                # DIAGNOSTIC: Log admin/player status with attachment state
                logger.info("[RECONSTRUCT-STEP-6] QueuedMessage state: has_attachments=%s, has_stickers=%s, has_embeds=%s, attachment_count=%d, sticker_count=%d, embed_count=%d, is_gm=%s, is_admin=%s, has_character=%s", 
                           has_attachments, has_stickers, has_embeds, len(queued_message.attachments), 
                           len(queued_message.sticker_files) if hasattr(queued_message, 'sticker_files') else 0,
                           len(queued_message.embed_files) if hasattr(queued_message, 'embed_files') else 0,
                           is_gm_reconstruct, is_admin_reconstruct, has_character_reconstruct)
                has_attachment_data = hasattr(queued_message, '_attachment_data') and bool(queued_message._attachment_data)
                logger.info("Processing queued message from %s: content_length=%d, attachments=%d, _attachment_data=%s, stickers=%d, embeds=%d", 
                           queued_message.author.id if queued_message.author else "?", 
                           len(queued_message.content) if queued_message.content else 0,
                           len(queued_message.attachments) if queued_message.attachments else 0,
                           len(queued_message._attachment_data) if hasattr(queued_message, '_attachment_data') and queued_message._attachment_data else 0,
                           len(queued_message.sticker_files) if hasattr(queued_message, 'sticker_files') and queued_message.sticker_files else 0,
                           len(queued_message.embed_files) if hasattr(queued_message, 'embed_files') and queued_message.embed_files else 0)
                
                # CRITICAL: Check if lock is still held - if so, skip processing (shouldn't happen, but safety check)
                command_lock = self._get_command_lock(thread_id)
                if command_lock.locked():
                    logger.warning("Command lock still held when processing queued messages - skipping to prevent recursion")
                    continue
                
                # Re-call handle_message - it will process the message normally
                # The message.delete() call in handle_message will fail silently since message is already deleted
                # CRITICAL: This will process ALL queued messages (including GM/admin) in order
                # Pass is_queued=True to skip the lock check and prevent re-queuing
                handled = await self.handle_message(queued_message, command_invoked=False, is_queued=True)
                if not handled:
                    await self._warn_queue_drop(
                        queued_message.channel,
                        "queued message was not handled",
                        message_id=message_id if isinstance(message_id, int) else None,
                        author_id=author_id if isinstance(author_id, int) else None,
                    )
            except Exception as exc:
                logger.exception("Error processing queued message from %s: %s", message_data.get('author').id if message_data.get('author') else "unknown", exc)
                await self._warn_queue_drop(
                    message_data.get('channel'),
                    f"exception during queued message replay: {exc}",
                    message_id=message_id if isinstance(message_id, int) else None,
                    author_id=author_id if isinstance(author_id, int) else None,
                )

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
            await ctx.reply("You are not an admin. Only admins can start new games. The admin who starts the game becomes the GM.", mention_author=False)
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
                    await self._update_board(existing_state, error_channel=ctx.channel, description_text="Game resumed")
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
        bot_user_id = self.bot.user.id if self.bot.user else None
        logger.info("command_startgame: Setting bot_user_id=%s for game ownership", bot_user_id)
        
        # Capture enabled packs from tf_characters.json (single source of truth)
        from tf_characters import get_enabled_packs_for_game, BOT_NAME
        enabled_packs = get_enabled_packs_for_game(game_type, BOT_NAME)
        logger.info("command_startgame: Captured enabled packs for game %s: %s", game_type, enabled_packs)
        
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
            bot_user_id=bot_user_id,  # Track which bot owns this game (prevents multiple bots from processing same game)
            enabled_packs=enabled_packs,  # Capture enabled packs at game creation (frozen for this game)
        )
        
        logger.info("command_startgame: Storing game state in active games")
        self._active_games[thread.id] = game_state
        
        # Send progress message
        progress_msg = await ctx.reply("⏳ Creating initial board...", mention_author=False)
        
        # Post initial blank board (no players yet - board will update when characters are assigned)
        logger.info("command_startgame: Creating initial blank board")
        await self._update_board(game_state, error_channel=ctx.channel, description_text="Game created")
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
        
        if not self._is_actual_gm(ctx.author, game_state):
            await ctx.reply("Only the GM can add players.", mention_author=False)
            return
        
        if not member:
            await ctx.reply("Usage: `!addplayer @user [character_name]`\nExample: `!addplayer @user kiyoshi`", mention_author=False)
            return
        
        # Check if user is already a player
        is_re_adding = False
        if member.id in game_state.players:
            # Check if player is forfeited - if so, allow re-adding
            pack = get_game_pack(game_state.game_type, self.packs_dir)
            is_forfeited = False
            if pack and pack.has_function("get_game_data"):
                pack_module = pack.module
                get_game_data = getattr(pack_module, "get_game_data", None)
                if callable(get_game_data):
                    try:
                        data = get_game_data(game_state)
                        forfeited_players = data.get('forfeited_players', [])
                        if member.id in forfeited_players:
                            is_forfeited = True
                            is_re_adding = True
                            # Remove from forfeited_players list
                            if member.id in forfeited_players:
                                forfeited_players.remove(member.id)
                                data['forfeited_players'] = forfeited_players
                            logger.info("Re-adding forfeited player %s to game", member.id)
                            # Player will be reset below (position, etc.)
                    except Exception as exc:
                        logger.warning("Failed to call get_game_data during addplayer forfeit check: %s", exc)
                        # Continue - assume not forfeited if check fails
            
            if not is_forfeited:
                # Player is already in game and not forfeited - show error
                player_number = self._get_player_number(game_state, member.id)
                player = game_state.players[member.id]
                if player_number:
                    if player.character_name:
                        await ctx.reply(f"{member.display_name} ({player.character_name}) is already Player {player_number} in this game.", mention_author=False)
                    else:
                        await ctx.reply(f"{member.display_name} is already Player {player_number} in this game.", mention_author=False)
                else:
                    if player.character_name:
                        await ctx.reply(f"{member.display_name} ({player.character_name}) is already in this game.", mention_author=False)
                    else:
                        await ctx.reply(f"{member.display_name} is already in this game.", mention_author=False)
                return
        
        # Check if character name is already assigned to another player (if character_name provided)
        if character_name and character_name.strip():
            character_name = character_name.strip()
            # Check if any player already has this character
            for existing_user_id, existing_player in game_state.players.items():
                if existing_player.character_name and existing_player.character_name.lower() == character_name.lower():
                    existing_member = ctx.guild.get_member(existing_user_id) if ctx.guild else None
                    existing_player_number = self._get_player_number(game_state, existing_user_id)
                    if existing_player_number:
                        if existing_member:
                            await ctx.reply(f"Character '{character_name}' is already assigned to {existing_member.display_name} (Player {existing_player_number}).", mention_author=False)
                        else:
                            await ctx.reply(f"Character '{character_name}' is already assigned to Player {existing_player_number}.", mention_author=False)
                    else:
                        if existing_member:
                            await ctx.reply(f"Character '{character_name}' is already assigned to {existing_member.display_name}.", mention_author=False)
                        else:
                            await ctx.reply(f"Character '{character_name}' is already assigned to another player.", mention_author=False)
                    return
        
        # Send immediate progress message
        progress_msg = await ctx.reply("⏳ Adding player...", mention_author=False)
        
        # Check if player already exists (forfeited player being re-added)
        # is_re_adding is already set above if they were forfeited
        is_re_adding_player = is_re_adding
        
        # Check if player has a previous player_number (from before they quit)
        pack = get_game_pack(game_state.game_type, self.packs_dir)
        previous_player_number = None
        if pack and pack.has_function("get_game_data"):
            pack_module = pack.module
            get_game_data = getattr(pack_module, "get_game_data", None)
            if callable(get_game_data):
                try:
                    data = get_game_data(game_state)
                    player_numbers = data.get('player_numbers', {})
                    previous_player_number = player_numbers.get(member.id)
                except Exception as exc:
                    logger.warning("Failed to call get_game_data during addplayer player_number check: %s", exc)
                    previous_player_number = None
        
        # Check if player has preserved position data (from before they were removed)
        preserved_tile = None
        preserved_grid_position = None
        if pack and pack.has_function("get_game_data"):
            pack_module = pack.module
            get_game_data = getattr(pack_module, "get_game_data", None)
            if callable(get_game_data):
                try:
                    data = get_game_data(game_state)
                    # Check for preserved tile_number
                    tile_numbers = data.get('tile_numbers', {})
                    preserved_tile = tile_numbers.get(member.id)
                    # Check for preserved grid_position
                    removed_positions = data.get('removed_player_positions', {})
                except Exception as exc:
                    logger.warning("Failed to call get_game_data during addplayer position check: %s", exc)
                    preserved_tile = None
                    removed_positions = {}
                preserved_grid_position = removed_positions.get(member.id)
        
        if is_re_adding_player:
            # Restore existing player for re-adding
            player = game_state.players[member.id]
            # Restore grid_position if preserved, otherwise reset to A1
            if preserved_grid_position:
                player.grid_position = preserved_grid_position
                logger.info("Restored grid_position %s for re-added player %s", preserved_grid_position, member.id)
            else:
                player.grid_position = "A1"  # Reset position if no preserved data
            # Character will be preserved if not reassigning, or reset if character_name provided
            logger.info("Re-adding forfeited player %s (was Player %s)", member.id, self._get_player_number(game_state, member.id))
        else:
            # Create new player
            # Use preserved grid_position if available, otherwise A1
            initial_position = preserved_grid_position if preserved_grid_position else "A1"
            player = GamePlayer(user_id=member.id, grid_position=initial_position, background_id=415)
            game_state.players[member.id] = player
        
        # Call pack's on_player_added if it exists (resets game data for re-added forfeited players)
        game_config = self.get_game_config(game_state.game_type)
        if pack and pack.has_function("on_player_added") and game_config:
            # For re-added forfeited players, this will reset their tile position and game data
            # The pack's on_player_added will preserve existing player_number if present
            try:
                pack.call("on_player_added", game_state, player, game_config)
            except Exception as exc:
                logger.exception("Error in pack.on_player_added: %s", exc)
                # Continue - player is still added, just pack callback failed
            
            # CRITICAL: Restore preserved tile_number and grid_position after on_player_added
            # (on_player_added will reset them, so we restore after)
            if preserved_tile is not None:
                try:
                    data = get_game_data(game_state)
                    tile_numbers = data.get('tile_numbers', {})
                    tile_numbers[member.id] = preserved_tile
                    data['tile_numbers'] = tile_numbers
                    logger.info("Restored tile_number %s for re-added player %s", preserved_tile, member.id)
                except Exception as exc:
                    logger.warning("Failed to restore tile_number for re-added player: %s", exc)
            
            if preserved_grid_position:
                player.grid_position = preserved_grid_position
                logger.info("Restored grid_position %s for re-added player %s", preserved_grid_position, member.id)
            
            # CRITICAL: Restore previous player_number if they had one and it wasn't preserved
            if previous_player_number is not None:
                if pack.has_function("get_game_data"):
                    pack_module = pack.module
                    get_game_data = getattr(pack_module, "get_game_data", None)
                    if callable(get_game_data):
                        try:
                            data = get_game_data(game_state)
                            player_numbers = data.get('player_numbers', {})
                            current_number = player_numbers.get(member.id)
                            # Only restore if it was overwritten (shouldn't happen now, but safety check)
                        except Exception as exc:
                            logger.warning("Failed to restore player_number for re-added player: %s", exc)
                            current_number = None
                        if current_number is None or current_number != previous_player_number:
                            try:
                                data = get_game_data(game_state)
                                player_numbers = data.get('player_numbers', {})
                                player_numbers[member.id] = previous_player_number
                                data['player_numbers'] = player_numbers
                                logger.info("Restored player number %s for %s (was %s)", previous_player_number, member.id, current_number)
                            except Exception as exc:
                                logger.warning("Failed to update player_number for re-added player: %s", exc)
        
        # Check if pack wants board update on player added
        should_update = False
        if pack and pack.has_function("should_update_board"):
            try:
                should_update = pack.call("should_update_board", game_state, "player_added")
            except Exception as exc:
                logger.warning("Error in pack.should_update_board: %s", exc)
                should_update = False  # Default to False on error
        
        # Board will be updated when character is assigned (not when player is added)
        # Unless pack specifically requests it
        if should_update:
            player_number = self._get_player_number(game_state, member.id)
            description_text = f"Player {player_number} added" if player_number else "Player added"
            await self._update_board(game_state, error_channel=ctx.channel, description_text=description_text)
        
        # If character_name provided, assign character
        if character_name and character_name.strip():
            character_name = character_name.strip()
            logger.info("command_addplayer: Auto-assigning character '%s' to %s", character_name, member.id)
            
            # Verify character exists
            character = self._get_character_by_name(character_name, game_state=game_state)
            if not character:
                await ctx.reply(f"Added {member.display_name} to the game.\n❌ Unable to locate character '{character_name}'. Use `!assign @{member.display_name} <character>` to assign later.", mention_author=False)
                await self._log_action(game_state, f"Player {member.display_name} added (character '{character_name}' not found)")
                return
            
            # Assign character using the same logic as command_assign
            actual_character_name = character.name
            player.character_name = actual_character_name
            
            # Create TransformationState for the player
            if not ctx.guild:
                await ctx.reply(f"Added {member.display_name} to the game.\n❌ Error: Cannot assign character outside of a server.", mention_author=False)
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
                member=member,  # Pass member directly to avoid lookup failures
            )
            if not state:
                logger.error("CRITICAL: Failed to create state for player %s with character %s", member.id, actual_character_name)
                await ctx.reply(f"Added {member.display_name} to the game.\n❌ Error: Failed to create state for character '{character_name}'. Use `!assign @{member.display_name} <character>` to assign later.", mention_author=False)
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
            
            # Trigger face grab on assignment (runs in background, doesn't block)
            # This ensures faces are grabbed/updated when characters are assigned
            asyncio.create_task(self._trigger_face_grab_on_assignment(actual_character_name, force=False))
            
            # If GM was assigned as a player, remove narrator role
            if member.id == game_state.gm_user_id and game_state.narrator_user_id == member.id:
                game_state.narrator_user_id = None
            
            # Call pack's on_character_assigned if it exists
            if pack and pack.has_function("on_character_assigned"):
                try:
                    pack.call("on_character_assigned", game_state, player, actual_character_name)
                except Exception as exc:
                    logger.exception("Error in pack.on_character_assigned: %s", exc)
                    # Continue - character is still assigned, just pack callback failed
            
            # Check if pack wants board update on character assignment
            should_update = True
            if pack and pack.has_function("should_update_board"):
                try:
                    should_update = pack.call("should_update_board", game_state, "character_assigned")
                except Exception as exc:
                    logger.warning("Error in pack.should_update_board: %s", exc)
                    should_update = True  # Default to True on error (safe default)
            
            if should_update:
                player_number = self._get_player_number(game_state, member.id)
                description_text = f"Player {player_number} added as {actual_character_name}" if player_number else f"Player added as {actual_character_name}"
                await self._update_board(game_state, error_channel=ctx.channel, description_text=description_text)
            
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
                        member.display_name,
                        duration_label,
                        actual_character_name,
                    )
                    # Append player number to formatted message
                    if player_number_text:
                        response_text = f"{response_text}{player_number_text}"
                else:
                    response_text = f"{member.display_name} is now **{actual_character_name}**{player_number_text}!"
                
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
                await ctx.reply(f"✅ {member.display_name} is now **{actual_character_name}**{player_number_text}!", mention_author=False)
            
            await self._log_action(game_state, f"Player {member.display_name} added and assigned character: {actual_character_name}")
            
            # Auto-save after player is added and character is assigned
            await self._save_auto_save(game_state, ctx)
        else:
            # Delete progress message before sending final response
            try:
                await progress_msg.delete()
            except Exception:
                pass
            # Check if this was a re-add of a forfeited player
            if is_re_adding:
                player_number = self._get_player_number(game_state, member.id)
                if player_number:
                    await ctx.reply(f"✅ Re-added {member.display_name} to the game (Player {player_number}). Position reset to starting tile.", mention_author=False)
                else:
                    await ctx.reply(f"✅ Re-added {member.display_name} to the game. Position reset to starting tile.", mention_author=False)
                await self._log_action(game_state, f"Forfeited player {member.display_name} re-added to game")
            else:
                player_number = self._get_player_number(game_state, member.id)
                if player_number:
                    await ctx.reply(f"Added {member.display_name} to the game assigned Player {player_number}.", mention_author=False)
                else:
                    await ctx.reply(f"Added {member.display_name} to the game.", mention_author=False)
                await self._log_action(game_state, f"Player {member.display_name} added")
            
            # Auto-save after player is added
            await self._save_auto_save(game_state, ctx)

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
                
                if not self._is_actual_gm(ctx.author, game_state):
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
                    await ctx.reply(f"{resolved_member.display_name} is not in the game. Add them first with `!addplayer`.", mention_author=False)
                    return
                
                player = game_state.players[resolved_member.id]
                
                # Check if player already has character (Issue #7 - prevent reassignment)
                if player.character_name and player.character_name.strip():
                    # Player already has character - check if they're forfeited/re-added
                    # If in game_state.players, they're active (reject reassignment)
                    player_number = self._get_player_number(game_state, resolved_member.id)
                    if player_number:
                        try:
                            await progress_msg.delete()
                        except Exception:
                            pass
                        await ctx.reply(f"{resolved_member.display_name} ({player.character_name}) is already Player {player_number} in this game.", mention_author=False)
                        return
                    else:
                        try:
                            await progress_msg.delete()
                        except Exception:
                            pass
                        await ctx.reply(f"{resolved_member.display_name} ({player.character_name}) is already assigned a character in this game.", mention_author=False)
                        return
                
                # Verify character exists before assigning (uses first name matching like !reroll)
                character = self._get_character_by_name(character_to_assign, game_state=game_state)
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
                    member=resolved_member,  # Pass member directly to avoid lookup failures
                    game_state=game_state,
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
                
                # Trigger face grab on assignment (runs in background, doesn't block)
                # This ensures faces are grabbed/updated when characters are assigned
                asyncio.create_task(self._trigger_face_grab_on_assignment(actual_character_name, force=False))
                
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
                player_number = self._get_player_number(game_state, resolved_member.id)
                description_text = f"Player {player_number} assigned as {actual_character_name}" if player_number else f"Player assigned as {actual_character_name}"
                await self._update_board(game_state, error_channel=ctx.channel, description_text=description_text)
                
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
                            resolved_member.display_name,
                            duration_label,
                            actual_character_name,
                        )
                        # Append player number to formatted message
                        if player_number_text:
                            response_text = f"{response_text}{player_number_text}"
                    else:
                        # Fallback if function not available
                        response_text = f"{resolved_member.display_name} is now **{actual_character_name}**{player_number_text}!"
                    
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
                    await ctx.reply(f"✅ {resolved_member.display_name} is now **{actual_character_name}**{player_number_text}!", mention_author=False)
                
                await self._log_action(game_state, f"{resolved_member.display_name} assigned character: {character_to_assign}")
                logger.info("Successfully assigned character %s to %s", character_to_assign, resolved_member.id)
                
                # Auto-save after character is assigned
                await self._save_auto_save(game_state, ctx)
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

    async def command_reroll(self, ctx: commands.Context, member: Optional[discord.Member] = None, token: Optional[str] = None, forced_character: Optional[str] = None) -> None:
        """Reroll a player's character in gameboard mode ONLY (GM only). Completely separate from VN reroll. Supports: !reroll @user OR !reroll character_name OR !reroll character_name target_character"""
        async def _impl():
            if not isinstance(ctx.author, discord.Member):
                await ctx.reply("This command can only be used inside a server.", mention_author=False)
                return
            
            game_state = await self._get_game_state_for_context(ctx)
            if not game_state:
                await ctx.reply("No active game in this thread.", mention_author=False)
                return
            
            if not self._is_actual_gm(ctx.author, game_state):
                await ctx.reply("Only the GM can reroll characters.", mention_author=False)
                return
            
            # Enhanced parsing: support multiple formats
            # !reroll tori = random reroll for tori
            # !reroll tori Kiyoshi = roll tori into Kiyoshi
            # !reroll @user character = reroll character
            # !reroll @user = random reroll
            resolved_member = member
            resolved_forced_character = forced_character
            
            if token:
                if not resolved_member:
                    # Try to resolve token as member/character
                    resolved_member = self._resolve_target_member(ctx, game_state, token)
                    
                    # If token wasn't used to resolve member, check if it's a character name for forced_character
                    if not resolved_member:
                        # Check if token could be a character name
                        potential_character = self._get_character_by_name(token, game_state=game_state)
                        if potential_character:
                            # Token is a character name, but we need a member first
                            # This is an error case - need member before character
                            character_names_in_game = [
                                player.character_name 
                                for player in game_state.players.values() 
                                if player.character_name
                            ]
                            if character_names_in_game:
                                await ctx.reply(
                                    f"Could not find player for '{token}'. Usage: `!reroll @user` or `!reroll character_name` or `!reroll character_name target_character`\n"
                                    f"Available characters: {', '.join(character_names_in_game)}",
                                    mention_author=False
                                )
                            else:
                                await ctx.reply("Usage: `!reroll @user` or `!reroll character_name` or `!reroll character_name target_character`", mention_author=False)
                            return
                else:
                    # Member provided, token might be forced_character
                    if not resolved_forced_character:
                        resolved_forced_character = token
            
            # If we have a member but no forced_character, check if there's a second token
            if resolved_member and not resolved_forced_character and token:
                # Token was already used, check if there's more in the message
                # This is handled by the bot.py routing which parses the full command
                pass
            
            if not resolved_member:
                # Provide more helpful error message
                character_names_in_game = [
                    player.character_name 
                    for player in game_state.players.values() 
                    if player.character_name
                ]
                token_display = token if token else "provided token"
                if character_names_in_game:
                    await ctx.reply(
                        f"Could not find player for '{token_display}'. Usage: `!reroll @user` or `!reroll character_name` or `!reroll character_name target_character`\n"
                        f"Available characters: {', '.join(character_names_in_game)}",
                        mention_author=False
                    )
                else:
                    await ctx.reply("Usage: `!reroll @user` or `!reroll character_name` or `!reroll character_name target_character`", mention_author=False)
                return
            
            if resolved_member.id not in game_state.players:
                await ctx.reply(f"{resolved_member.display_name} is not in the game.", mention_author=False)
                return
            
            # Get player state first
            player_state = game_state.player_states.get(resolved_member.id)
            
            # Check if player is swapped - if so, revert all swaps in chain first
            if player_state and player_state.form_owner_user_id and player_state.form_owner_user_id != resolved_member.id:
                # Player is swapped - build swap chain and revert it
                swap_chain = self._build_swap_chain(game_state, resolved_member.id)
                if swap_chain:
                    logger.info("Reverting swap chain for %s before reroll: %s", resolved_member.id, swap_chain)
                    await self._revert_swap_chain(ctx, game_state, swap_chain)
                    # Refresh game_state and player_state after reverting swaps
                    game_state = await self._get_game_state_for_context(ctx)
                    if game_state:
                        player_state = game_state.player_states.get(resolved_member.id)
                    else:
                        player_state = None
                    
                    # Verify swap was reverted - if still swapped, log warning
                    if player_state and player_state.form_owner_user_id and player_state.form_owner_user_id != resolved_member.id:
                        logger.warning("Swap reversion may have failed for %s - still swapped with %s", resolved_member.id, player_state.form_owner_user_id)
            
            # Get player state (gameboard-only, never touches VN active_transformations)
            # Use original character name from identity_display_name if available
            original_character_name = None
            if player_state:
                original_character_name = player_state.identity_display_name or player_state.character_name
            
            if not player_state:
                # Create state if it doesn't exist
                player = game_state.players.get(resolved_member.id)
                if not player:
                    logger.error("Player %s not found in game_state.players during reroll state creation", resolved_member.id)
                    await ctx.reply("Error: Player not found in game state.", mention_author=False)
                    return
                player_state = await self._create_game_state_for_player(
                    player,
                    resolved_member.id,
                    ctx.guild.id,
                    player.character_name or "Unknown",
                    member=resolved_member,  # Pass member directly to avoid lookup failures
                    game_state=game_state,
                )
                if player_state:
                    game_state.player_states[resolved_member.id] = player_state
            
            if not player_state:
                await ctx.reply(f"Failed to create transformation state for {resolved_member.display_name}.", mention_author=False)
                return
            
            # Get character pool and helper functions from bot module (for character selection logic)
            import sys
            bot_module = sys.modules.get('bot')
            # Try fallback methods if bot module not found
            if not bot_module:
                try:
                    import bot as bot_module
                except ImportError:
                    bot_module = sys.modules.get('__main__')
            if not bot_module:
                logger.error("Failed to access bot module for character system")
                await ctx.reply("Failed to access character system. Please try again.", mention_author=False)
                return
            
            CHARACTER_POOL = getattr(bot_module, 'CHARACTER_POOL', None)
            _format_character_message = getattr(bot_module, '_format_character_message', None)
            _get_magic_emoji = getattr(bot_module, '_get_magic_emoji', None)
            member_profile_name = getattr(bot_module, 'member_profile_name', None)
            TFBOT_NAME = getattr(bot_module, 'TFBOT_NAME', 'TFBot')
            
            if not CHARACTER_POOL:
                await ctx.reply("Character pool not available.", mention_author=False)
                return
            
            # Check if forcing a specific character
            forced_character_obj = None
            if resolved_forced_character:
                forced_character_obj = self._get_character_by_name(resolved_forced_character, game_state=game_state)
                if not forced_character_obj:
                    await ctx.reply(
                        f"Unknown target `{resolved_forced_character}`. Provide a valid first name.",
                        mention_author=False,
                    )
                    return
            
            # Get used character names (from gameboard states only)
            used_names = {
                state.character_name
                for state in game_state.player_states.values()
                if state.character_name and state.user_id != resolved_member.id
            }
            
            # Select new character
            if forced_character_obj:
                new_character = forced_character_obj
                new_name = forced_character_obj.name
                new_folder = forced_character_obj.folder
                new_avatar_path = forced_character_obj.avatar_path or ""
                new_message = forced_character_obj.message or ""
            else:
                # Random selection - use filtered gameboard pool, exclude used names and current character
                # Get filtered character pool for this game
                enabled_packs = game_state.enabled_packs if game_state.enabled_packs else None
                filtered_pool = self._get_filtered_character_pool(game_state.game_type, enabled_packs=enabled_packs)
                
                if not filtered_pool:
                    await ctx.reply("No characters available for this game. Check pack enable settings.", mention_author=False)
                    return
                
                # Use original character name for exclusion (not swapped name)
                current_char_name = original_character_name or (player_state.character_name if player_state else None)
                available_characters = [
                    char for char in filtered_pool
                    if char.name not in used_names and char.name != current_char_name
                ]
                
                if not available_characters:
                    await ctx.reply("No alternative characters available for reroll.", mention_author=False)
                    return
                
                chosen = random.choice(available_characters)
                new_character = chosen
                new_name = chosen.name
                new_folder = chosen.folder
                new_avatar_path = chosen.avatar_path or ""
                new_message = chosen.message or ""
            
            if new_name == player_state.character_name:
                await ctx.reply(f"{resolved_member.display_name} is already {new_name}.", mention_author=False)
                return
            
            if new_name in used_names:
                await ctx.reply(f"{new_name} is already in use by another player.", mention_author=False)
                return
            
            # Update player state (gameboard-only, never touches VN state)
            previous_character = player_state.character_name
            player_state.character_name = new_name
            player_state.character_folder = new_folder
            player_state.character_avatar_path = new_avatar_path
            player_state.character_message = new_message
            player_state.identity_display_name = new_name
            player_state.avatar_applied = False
            
            # Update game player
            player = game_state.players.get(resolved_member.id)
            if not player:
                logger.error("Player %s not found in game_state.players during reroll character update", resolved_member.id)
                await ctx.reply("Error: Player not found in game state.", mention_author=False)
                return
            player.character_name = new_name
            
            # Call pack's on_character_assigned if available
            pack = get_game_pack(game_state.game_type, self.packs_dir)
            if pack and pack.has_function("on_character_assigned"):
                player = game_state.players.get(resolved_member.id)
                if not player:
                    logger.error("Player %s not found in game_state.players during reroll", resolved_member.id)
                    await ctx.reply("Error: Player not found in game state.", mention_author=False)
                    return
                
                try:
                    pack.call("on_character_assigned", game_state, player, new_name)
                except Exception as exc:
                    logger.exception("Error in pack.on_character_assigned during reroll: %s", exc)
                    # Continue - character is still assigned, just pack callback failed
            
            # Update board to show new character image/token
            player_number = self._get_player_number(game_state, resolved_member.id)
            description_text = f"Player {player_number} rerolled to {new_name}" if player_number else f"Player rerolled to {new_name}"
            await self._update_board(game_state, error_channel=ctx.channel, description_text=description_text)
            
            # Send VN-style message (same format as VN reroll but gameboard-only)
            if _format_character_message and member_profile_name:
                original_name = member_profile_name(resolved_member)
                base_message = _format_character_message(
                    new_message,
                    original_name,
                    resolved_member.display_name,
                    "good",  # Gameboard characters are permanent (reads as "for good")
                    new_name,
                )
                response_text = base_message
                
                if _get_magic_emoji:
                    emoji_prefix = _get_magic_emoji(ctx.guild)
                    try:
                        await ctx.channel.send(
                            f"{emoji_prefix} {response_text}",
                            allowed_mentions=discord.AllowedMentions.none(),
                        )
                    except discord.HTTPException as exc:
                        logger.warning("Failed to send reroll message: %s", exc)
                
                try:
                    await ctx.message.delete()
                except discord.HTTPException:
                    pass
                
                summary_message = f"{resolved_member.display_name} has been rerolled into **{new_name}**."
                await ctx.send(summary_message, delete_after=10)
            else:
                # Fallback if helper functions not available
                await ctx.reply(
                    f"Rerolled {resolved_member.display_name}'s character from {previous_character or 'none'} to {new_name}.",
                    mention_author=False
                )
            
            # Update board to show new character image (second update after message sent)
            player_number = self._get_player_number(game_state, resolved_member.id)
            description_text = f"Player {player_number} rerolled to {new_name}" if player_number else f"Player rerolled to {new_name}"
            await self._update_board(game_state, error_channel=ctx.channel, description_text=description_text)
            
            await self._log_action(game_state, f"{resolved_member.display_name} rerolled from {previous_character} to {new_name}")
            
            # Auto-save after character is rerolled
            await self._save_auto_save(game_state, ctx)
        
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
            
            if not self._is_actual_gm(ctx.author, game_state):
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
            
            # Swap characters and positions (tokens are tied to characters, NOT players)
            # CRITICAL: Player numbers NEVER swap - they stay with the player (user_id)
            # Example: Player 1 (Character A on tile 10) swaps with Player 2 (Character B on tile 8)
            # Result: 
            #   - Player 1 (now Character B) moves to tile 8 (token moves with character)
            #   - Player 2 (now Character A) moves to tile 10 (token moves with character)
            #   - Player numbers NEVER change: Player 1 stays Player 1, Player 2 stays Player 2
            #   - Turn order NEVER changes: Turn order stays the same
            player1 = game_state.players.get(resolved_member1.id)
            player2 = game_state.players.get(resolved_member2.id)
            if not player1 or not player2:
                logger.error("One or both players not found during swap: player1=%s, player2=%s", player1 is not None, player2 is not None)
                await ctx.reply("Error: One or both players not found in game state.", mention_author=False)
                return
            
            char1 = player1.character_name
            char2 = player2.character_name
            player1.character_name = char2
            player2.character_name = char1
            
            # Swap grid positions (tokens are tied to characters, so positions swap with characters)
            pos1 = player1.grid_position
            pos2 = player2.grid_position
            player1.grid_position = pos2
            player2.grid_position = pos1
            
            # Swap backgrounds (to give illusion they changed places if in different locations)
            bg1 = player1.background_id
            bg2 = player2.background_id
            player1.background_id = bg2
            player2.background_id = bg1
            logger.info("Swapped backgrounds (!swap): player1 (user_id=%s, character=%s) bg_id %s -> %s, player2 (user_id=%s, character=%s) bg_id %s -> %s", 
                       resolved_member1.id, player1.character_name, bg1, bg2, 
                       resolved_member2.id, player2.character_name, bg2, bg1)
            # Verify swap persisted correctly
            if player1.background_id != bg2 or player2.background_id != bg1:
                logger.error("CRITICAL: Background swap verification failed! player1.bg_id=%s (expected %s), player2.bg_id=%s (expected %s)", 
                           player1.background_id, bg2, player2.background_id, bg1)
            else:
                logger.debug("Background swap verified: player1.bg_id=%s, player2.bg_id=%s", 
                           player1.background_id, player2.background_id)
            
            # Swap outfits (outfits stay with characters)
            outfit1 = player1.outfit_name
            outfit2 = player2.outfit_name
            player1.outfit_name = outfit2
            player2.outfit_name = outfit1
            
            # Swap TransformationState objects in game_state.player_states (preserve user identity)
            state1 = game_state.player_states.get(resolved_member1.id)
            state2 = game_state.player_states.get(resolved_member2.id)
            new_state1 = state1
            new_state2 = state2
            new_state1 = state1
            new_state2 = state2
            
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
            
            # Update pack-specific metadata
            # CRITICAL: Swap tile_numbers (positions swap with characters)
            # Do NOT swap player_numbers or turn_order (these stay with player numbers)
            pack = get_game_pack(game_state.game_type, self.packs_dir)
            if pack and pack.has_function("get_game_data"):
                pack_module = pack.module
                get_game_data = getattr(pack_module, "get_game_data", None)
                if callable(get_game_data):
                    try:
                        data = get_game_data(game_state)
                        
                        # Swap tile_numbers (positions swap with characters)
                        tile_numbers = data.get('tile_numbers', {})
                    except Exception as exc:
                        logger.warning("Failed to call get_game_data during swap: %s", exc)
                        data = None
                    
                    if data:
                        if isinstance(tile_numbers, dict):
                            tile1 = tile_numbers.get(resolved_member1.id)
                            tile2 = tile_numbers.get(resolved_member2.id)
                            if tile1 is not None and tile2 is not None:
                                tile_numbers[resolved_member1.id] = tile2
                                tile_numbers[resolved_member2.id] = tile1
                                data['tile_numbers'] = tile_numbers
                                logger.info("Swapped tile_numbers: player1=%s (tile %s -> %s), player2=%s (tile %s -> %s)", 
                                          resolved_member1.id, tile1, tile2, resolved_member2.id, tile2, tile1)
                    
                    # Swap character-related metadata
                    self._swap_pack_player_metadata(game_state, resolved_member1.id, resolved_member2.id)
            
            # Notify pack about character swaps
            if pack and pack.has_function("on_character_assigned"):
                player1 = game_state.players.get(resolved_member1.id)
                player2 = game_state.players.get(resolved_member2.id)
                if not player1 or not player2:
                    logger.error("One or both players not found during swap: player1=%s, player2=%s", player1 is not None, player2 is not None)
                    await ctx.reply("Error: One or both players not found in game state.", mention_author=False)
                    return
                
                try:
                    pack.call("on_character_assigned", game_state, player1, char2)
                except Exception as exc:
                    logger.exception("Error in pack.on_character_assigned for player1 during swap: %s", exc)
                
                try:
                    pack.call("on_character_assigned", game_state, player2, char1)
                except Exception as exc:
                    logger.exception("Error in pack.on_character_assigned for player2 during swap: %s", exc)
            
            # Update board to show swapped positions and character images
            player1_number = self._get_player_number(game_state, resolved_member1.id)
            player2_number = self._get_player_number(game_state, resolved_member2.id)
            description_text = f"Player {player1_number} and Player {player2_number} swapped" if (player1_number and player2_number) else "Players swapped"
            await self._update_board(game_state, error_channel=ctx.channel, description_text=description_text)
            
            # Auto-save after characters are swapped
            await self._save_auto_save(game_state, ctx)
            from tfbot.panels import render_swap_transition_panel, render_swap_transition_panel_gif

            left_label = f"{(new_state1.character_name if new_state1 and new_state1.character_name else resolved_member1.display_name)}({resolved_member1.display_name})"
            right_label = f"{(new_state2.character_name if new_state2 and new_state2.character_name else resolved_member2.display_name)}({resolved_member2.display_name})"
            message_text = f"<@{ctx.author.id}> swapped {left_label} with {right_label}"
            transition_started = time.perf_counter()
            transition_file = await run_panel_render_gif(
                render_swap_transition_panel_gif,
                before_left_state=state1,
                before_right_state=state2,
                after_left_state=new_state1,
                after_right_state=new_state2,
                left_background_user_id=resolved_member1.id,
                right_background_user_id=resolved_member2.id,
                filename=f"swap_{resolved_member1.id}_{resolved_member2.id}.webp",
            )
            if transition_file is None:
                transition_file = await run_panel_render_gif(
                    render_swap_transition_panel,
                    left_state=new_state1,
                    right_state=new_state2,
                    left_background_user_id=resolved_member1.id,
                    right_background_user_id=resolved_member2.id,
                    filename=f"swap_{resolved_member1.id}_{resolved_member2.id}.png",
                )
            if transition_file is not None:
                render_ms = (time.perf_counter() - transition_started) * 1000.0
                _, upload_ms = await _timed_send_ms(
                    "game_swap_transition_send",
                    ctx.reply(
                        content=message_text,
                        file=transition_file,
                        mention_author=False,
                        allowed_mentions=discord.AllowedMentions.none(),
                    ),
                )
                _log_transition_send_metrics(
                    label="game_swap_transition_send",
                    render_ms=render_ms,
                    upload_ms=upload_ms,
                    total_ms=(time.perf_counter() - transition_started) * 1000.0,
                    payload_bytes=_discord_file_size_bytes(transition_file),
                )
            else:
                await _timed_send(
                    "game_swap_text_send",
                    ctx.reply(
                        message_text,
                        mention_author=False,
                        allowed_mentions=discord.AllowedMentions.none(),
                    ),
                )
            await self._log_action(game_state, f"{resolved_member1.display_name} and {resolved_member2.display_name} swapped characters and positions")
        
        await self._execute_gameboard_command(ctx, _impl)

    async def command_pswap(self, ctx: commands.Context, member1: Optional[discord.Member] = None, member2: Optional[discord.Member] = None, token1: Optional[str] = None, token2: Optional[str] = None) -> None:
        """Permanent swap characters between two players (GM only, gameboard only). Same as !swap but without reroll block or swap reversal. Supports: !pswap @user1 @user2 OR !pswap character1 character2"""
        async def _impl():
            if not isinstance(ctx.author, discord.Member):
                await ctx.reply("This command can only be used inside a server.", mention_author=False)
                return
            
            game_state = await self._get_game_state_for_context(ctx)
            if not game_state:
                await ctx.reply("No active game in this thread.", mention_author=False)
                return
            
            if not self._is_actual_gm(ctx.author, game_state):
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
                await ctx.reply("Usage: `!pswap @user1 @user2` or `!pswap character1 character2`", mention_author=False)
                return
            
            if resolved_member1.id not in game_state.players or resolved_member2.id not in game_state.players:
                await ctx.reply("Both players must be in the game.", mention_author=False)
                return
            
            # Swap characters and positions (tokens are tied to characters, NOT players)
            # CRITICAL: Player numbers NEVER swap - they stay with the player (user_id)
            player1 = game_state.players.get(resolved_member1.id)
            player2 = game_state.players.get(resolved_member2.id)
            if not player1 or not player2:
                logger.error("One or both players not found during pswap: player1=%s, player2=%s", player1 is not None, player2 is not None)
                await ctx.reply("Error: One or both players not found in game state.", mention_author=False)
                return
            
            char1 = player1.character_name
            char2 = player2.character_name
            player1.character_name = char2
            player2.character_name = char1
            
            # Swap grid positions (tokens are tied to characters, so positions swap with characters)
            pos1 = player1.grid_position
            pos2 = player2.grid_position
            player1.grid_position = pos2
            player2.grid_position = pos1
            
            # Swap backgrounds (to give illusion they changed places if in different locations)
            bg1 = player1.background_id
            bg2 = player2.background_id
            player1.background_id = bg2
            player2.background_id = bg1
            logger.info("Swapped backgrounds (!pswap): player1 (user_id=%s, character=%s) bg_id %s -> %s, player2 (user_id=%s, character=%s) bg_id %s -> %s", 
                       resolved_member1.id, player1.character_name, bg1, bg2, 
                       resolved_member2.id, player2.character_name, bg2, bg1)
            # Verify swap persisted correctly
            if player1.background_id != bg2 or player2.background_id != bg1:
                logger.error("CRITICAL: Background swap verification failed! player1.bg_id=%s (expected %s), player2.bg_id=%s (expected %s)", 
                           player1.background_id, bg2, player2.background_id, bg1)
            else:
                logger.debug("Background swap verified: player1.bg_id=%s, player2.bg_id=%s", 
                           player1.background_id, player2.background_id)
            
            # Swap outfits (outfits stay with characters)
            outfit1 = player1.outfit_name
            outfit2 = player2.outfit_name
            player1.outfit_name = outfit2
            player2.outfit_name = outfit1
            
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
                # CRITICAL: Set form_owner_user_id to player's own ID (not swapped) - this prevents reroll from reverting
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
                    form_owner_user_id=resolved_member1.id,  # Set to own ID (permanent swap - no reversal)
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
                    form_owner_user_id=resolved_member2.id,  # Set to own ID (permanent swap - no reversal)
                    identity_display_name=identity2,  # Keep player's original identity
                    is_pillow=state2.is_pillow,
                )
                
                # Update player_states
                game_state.player_states[resolved_member1.id] = new_state1
                game_state.player_states[resolved_member2.id] = new_state2
            
            # Update pack-specific metadata
            # CRITICAL: Swap tile_numbers (positions swap with characters)
            # Do NOT swap player_numbers or turn_order (these stay with player numbers)
            pack = get_game_pack(game_state.game_type, self.packs_dir)
            if pack and pack.has_function("get_game_data"):
                pack_module = pack.module
                get_game_data = getattr(pack_module, "get_game_data", None)
                if callable(get_game_data):
                    try:
                        data = get_game_data(game_state)
                        
                        # Swap tile_numbers (positions swap with characters)
                        tile_numbers = data.get('tile_numbers', {})
                    except Exception as exc:
                        logger.warning("Failed to call get_game_data during pswap: %s", exc)
                        data = None
                    
                    if data:
                        if isinstance(tile_numbers, dict):
                            tile1 = tile_numbers.get(resolved_member1.id)
                            tile2 = tile_numbers.get(resolved_member2.id)
                            if tile1 is not None and tile2 is not None:
                                tile_numbers[resolved_member1.id] = tile2
                                tile_numbers[resolved_member2.id] = tile1
                                data['tile_numbers'] = tile_numbers
                                logger.info("Swapped tile_numbers: player1=%s (tile %s -> %s), player2=%s (tile %s -> %s)", 
                                          resolved_member1.id, tile1, tile2, resolved_member2.id, tile2, tile1)
                    
                    # Swap character-related metadata
                    self._swap_pack_player_metadata(game_state, resolved_member1.id, resolved_member2.id)
            
            # Notify pack about character swaps
            if pack and pack.has_function("on_character_assigned"):
                player1 = game_state.players.get(resolved_member1.id)
                player2 = game_state.players.get(resolved_member2.id)
                if not player1 or not player2:
                    logger.error("One or both players not found during pswap: player1=%s, player2=%s", player1 is not None, player2 is not None)
                    await ctx.reply("Error: One or both players not found in game state.", mention_author=False)
                    return
                
                try:
                    pack.call("on_character_assigned", game_state, player1, char2)
                except Exception as exc:
                    logger.exception("Error in pack.on_character_assigned for player1 during pswap: %s", exc)
                
                try:
                    pack.call("on_character_assigned", game_state, player2, char1)
                except Exception as exc:
                    logger.exception("Error in pack.on_character_assigned for player2 during pswap: %s", exc)
            
            # Update board to show swapped positions and character images
            player1_number = self._get_player_number(game_state, resolved_member1.id)
            player2_number = self._get_player_number(game_state, resolved_member2.id)
            description_text = f"Player {player1_number} and Player {player2_number} permanently swapped" if (player1_number and player2_number) else "Players permanently swapped"
            await self._update_board(game_state, error_channel=ctx.channel, description_text=description_text)
            
            # Auto-save after characters are swapped
            await self._save_auto_save(game_state, ctx)
            from tfbot.panels import render_swap_transition_panel, render_swap_transition_panel_gif

            left_label = f"{(new_state1.character_name if new_state1 and new_state1.character_name else resolved_member1.display_name)}({resolved_member1.display_name})"
            right_label = f"{(new_state2.character_name if new_state2 and new_state2.character_name else resolved_member2.display_name)}({resolved_member2.display_name})"
            message_text = f"<@{ctx.author.id}> permanently swapped {left_label} with {right_label}"
            transition_started = time.perf_counter()
            transition_file = await run_panel_render_gif(
                render_swap_transition_panel_gif,
                before_left_state=state1,
                before_right_state=state2,
                after_left_state=new_state1,
                after_right_state=new_state2,
                left_background_user_id=resolved_member1.id,
                right_background_user_id=resolved_member2.id,
                filename=f"pswap_{resolved_member1.id}_{resolved_member2.id}.webp",
            )
            if transition_file is None:
                transition_file = await run_panel_render_gif(
                    render_swap_transition_panel,
                    left_state=new_state1,
                    right_state=new_state2,
                    left_background_user_id=resolved_member1.id,
                    right_background_user_id=resolved_member2.id,
                    filename=f"pswap_{resolved_member1.id}_{resolved_member2.id}.png",
                )
            if transition_file is not None:
                render_ms = (time.perf_counter() - transition_started) * 1000.0
                _, upload_ms = await _timed_send_ms(
                    "game_pswap_transition_send",
                    ctx.reply(
                        content=message_text,
                        file=transition_file,
                        mention_author=False,
                        allowed_mentions=discord.AllowedMentions.none(),
                    ),
                )
                _log_transition_send_metrics(
                    label="game_pswap_transition_send",
                    render_ms=render_ms,
                    upload_ms=upload_ms,
                    total_ms=(time.perf_counter() - transition_started) * 1000.0,
                    payload_bytes=_discord_file_size_bytes(transition_file),
                )
            else:
                await _timed_send(
                    "game_pswap_text_send",
                    ctx.reply(
                        message_text,
                        mention_author=False,
                        allowed_mentions=discord.AllowedMentions.none(),
                    ),
                )
            await self._log_action(game_state, f"{resolved_member1.display_name} and {resolved_member2.display_name} permanently swapped characters and positions")
        
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
            
            if not self._is_actual_gm(ctx.author, game_state):
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
                await ctx.reply(f"{resolved_member.display_name} is not in the game.", mention_author=False)
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
            player = game_state.players.get(resolved_member.id)
            if not player:
                await ctx.reply("Error: Player not found in game state.", mention_author=False)
                return
            
            if pack and pack.has_function("validate_move"):
                try:
                    is_valid, error_msg = pack.call("validate_move", game_state, player, position_value, game_config)
                    if not is_valid:
                        await ctx.reply(error_msg or f"Invalid move: `{position_value}`", mention_author=False)
                        return
                except Exception as exc:
                    logger.exception("Error in pack.validate_move: %s", exc)
                    await ctx.reply(f"Error validating move: {exc}", mention_author=False)
                    return
            
            # Use player variable (already checked above)
            old_pos = player.grid_position
            player.grid_position = position_value
            
            # CRITICAL: Update tile_numbers in game data to match GM movement
            # This ensures GM movement persists through dice rolls
            pack = get_game_pack(game_state.game_type, self.packs_dir)
            move_messages = []  # Initialize messages list for snake/ladder/win messages
            if pack and hasattr(pack.module, 'alphanumeric_to_tile_number'):
                # Get the function from the pack module
                alphanumeric_to_tile_number = getattr(pack.module, 'alphanumeric_to_tile_number')
                tile_number = alphanumeric_to_tile_number(position_value, game_config)
                if tile_number is not None:
                    # Update game data tile_numbers - use pack's get_game_data function
                    if hasattr(pack.module, 'get_game_data'):
                        get_game_data = getattr(pack.module, 'get_game_data')
                        try:
                            data = get_game_data(game_state)
                            data['tile_numbers'][resolved_member.id] = tile_number
                        except Exception as exc:
                            logger.warning("Failed to call get_game_data during movetoken: %s", exc)
                            data = None
                        
                        if data:
                            # ADD snakes/ladders check (same logic as on_dice_rolled)
                            rules = game_config.rules or {}
                            snakes_raw = rules.get("snakes", {}) if rules else {}
                            ladders_raw = rules.get("ladders", {}) if rules else {}
                            snakes = {int(k): int(v) for k, v in snakes_raw.items()} if snakes_raw else {}
                            ladders = {int(k): int(v) for k, v in ladders_raw.items()} if ladders_raw else {}
                            
                            final_tile = tile_number
                            
                            # Check snake
                            if tile_number in snakes:
                                tail_tile = snakes[tile_number]
                                data['tile_numbers'][resolved_member.id] = tail_tile
                                final_tile = tail_tile
                                if hasattr(pack.module, 'tile_number_to_alphanumeric'):
                                    tile_number_to_alphanumeric = getattr(pack.module, 'tile_number_to_alphanumeric')
                                    new_pos = tile_number_to_alphanumeric(tail_tile, game_config)
                                    if new_pos:
                                        player.grid_position = new_pos
                                move_messages.append(f"🐍 Snake! Slid down to tile {tail_tile}")
                            
                            # Check ladder
                            elif tile_number in ladders:
                                top_tile = ladders[tile_number]
                                data['tile_numbers'][resolved_member.id] = top_tile
                                final_tile = top_tile
                                if hasattr(pack.module, 'tile_number_to_alphanumeric'):
                                    tile_number_to_alphanumeric = getattr(pack.module, 'tile_number_to_alphanumeric')
                                    new_pos = tile_number_to_alphanumeric(top_tile, game_config)
                                    if new_pos:
                                        player.grid_position = new_pos
                                move_messages.append(f"🪜 Ladder! Climbed up to tile {top_tile}")
                        
                        # Check win condition
                        if pack and pack.has_function("check_win_condition"):
                            try:
                                win_msg, game_ended = pack.call("check_win_condition", game_state, game_config)
                                if win_msg:
                                    move_messages.append(win_msg)
                            except Exception as exc:
                                logger.exception("Error in pack.check_win_condition: %s", exc)
                                # Continue - move still succeeds, just win check failed
                        
                        logger.info("Updated tile_number for player %s to %d (GM movement to %s, final tile %d)", resolved_member.id, tile_number, position_value, final_tile)
                    else:
                        logger.warning("Pack %s does not have get_game_data function, cannot update tile_numbers", game_state.game_type)
            
            # Check if pack wants board update on move
            should_update = True
            if pack and pack.has_function("should_update_board"):
                try:
                    should_update = pack.call("should_update_board", game_state, "move")
                except Exception as exc:
                    logger.warning("Error in pack.should_update_board: %s", exc)
                    should_update = True  # Default to True on error
            
            if should_update:
                player_number = self._get_player_number(game_state, resolved_member.id)
                description_text = f"Player {player_number} moved to {position_value}" if player_number else f"Player moved to {position_value}"
                await self._update_board(game_state, error_channel=ctx.channel, description_text=description_text)
            
            # Build reply message with snake/ladder messages if any
            move_msg = f"Moved {resolved_member.display_name}'s token from {old_pos} to {position_value}."
            if move_messages:
                move_msg += "\n" + "\n".join(move_messages)
            await ctx.reply(move_msg, mention_author=False)
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
                
                is_gm = self._is_actual_gm(ctx.author, game_state)
                is_admin_user = is_admin(ctx.author) or is_bot_mod(ctx.author)
                is_gm_override = False
                
                # Check if GM is forcing a roll for another player (admins cannot force rolls)
                if target_player and is_gm:
                    # GM/Admin override - roll for the target player (but still requires game to start)
                    if not game_state.game_started:
                        await ctx.reply("⏸️ Game hasn't started yet! The GM needs to use `!start` to begin the game.", mention_author=False)
                        return
                    if target_player.id not in game_state.players:
                        await ctx.reply(f"{target_player.display_name} is not in this game.", mention_author=False)
                        return
                    player = game_state.players[target_player.id]
                    is_gm_override = True
                elif is_gm and not target_player:
                    # GM rolling for themselves - check if they're a player
                    if ctx.author.id in game_state.players:
                        player = game_state.players[ctx.author.id]
                        # GM must wait for game to start (same as normal players)
                        if not game_state.game_started:
                            await ctx.reply("⏸️ Game hasn't started yet! The GM needs to use `!start` to begin the game.", mention_author=False)
                            return
                        # Game has started, GM can roll
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
                
                # Use SystemRandom for statistically accurate randomness (OS-level entropy, optimized for rapid calls)
                rolls = [_dice_rng.randint(1, dice_faces) for _ in range(dice_count)]
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
                                transformation_char,
                                game_state=game_state,
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
                                
                                vn_file = await run_panel_render_vn(
                                    render_vn_panel,
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
                                try:
                                    summary_msg = pack.call("get_turn_summary", game_state, game_config, ctx.guild)
                                except Exception as exc:
                                    logger.warning("Error in pack.get_turn_summary: %s", exc)
                                    summary_msg = None

                # Get player number from player_numbers dict (based on add order)
                player_number = self._get_player_number(game_state, player.user_id)
                player_number_text = f" (Player {player_number})" if player_number else ""
                
                # Get player member for username display
                player_member = None
                if ctx.guild:
                    if isinstance(ctx.channel, discord.Thread):
                        player_member = ctx.channel.guild.get_member(player.user_id)
                    else:
                        player_member = ctx.guild.get_member(player.user_id)
                
                # Build username text for title
                username_text = ""
                if player_member:
                    username_text = f" - {player_member.display_name}"
                elif player.user_id:
                    username_text = f" - <@{player.user_id}>"
                
                # Build embed with roll result and player's current board position
                embed_color = discord.Color.random()
                player_position = player.grid_position or "Unknown"
                turn_number = game_state.turn_count + 1
                embed_description = f"{result}\n\n**New Position:** `{player_position}`\n**Turn:** {turn_number}"
                roll_embed = discord.Embed(
                    title=f"Dice Roll{player_number_text}{username_text} - Turn {turn_number}",
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
                        # Author shows character name, but title already has username
                        roll_embed.set_author(name=player.character_name, icon_url=face_attachment_url)
                if not face_file:
                    if player_member:
                        avatar_url = player_member.display_avatar.url
                        # Author shows character name if available, otherwise display name
                        author_name = player.character_name if player.character_name else player_member.display_name
                        roll_embed.set_author(name=author_name, icon_url=avatar_url)
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
                autosave_done = False
                if turn_complete_requested:
                    # Turn is completing - update board ONCE at end of turn (includes all movement)
                    if summary_msg:
                        await ctx.channel.send(summary_msg, allowed_mentions=discord.AllowedMentions.none())
                    
                    # ADD turn completion message - show turn number
                    next_player_info = self._get_next_player_info(game_state, pack, ctx.guild)
                    if next_player_info:
                        player_num, character_name, user_id, username = next_player_info
                        member = ctx.guild.get_member(user_id) if ctx.guild else None
                        player_name = member.display_name if member else f"User {user_id}"
                        await ctx.channel.send(f"**Turn {game_state.turn_count} ended. Turn {game_state.turn_count + 1} start. (Player {player_num} - {character_name} - {player_name})**", allowed_mentions=discord.AllowedMentions.none())
                    
                    if pack and pack.has_function("advance_turn"):
                        try:
                            pack.call("advance_turn", game_state)
                        except Exception as exc:
                            logger.exception("Error in pack.advance_turn: %s", exc)
                            # Continue - turn count still increments, just pack callback failed
                    
                    # Increment turn count and auto-save at end of turn
                    game_state.turn_count += 1
                    await self._save_auto_save(game_state, ctx)
                    autosave_done = True
                    logger.info("Turn %d completed, auto-save created", game_state.turn_count)
                    
                    # Update board ONCE at end of turn (this includes all movement from the turn)
                    # CRITICAL: Always post to game thread at turn end for visibility
                    also_post_to_game = True  # Always show board in game thread at turn end
                    description_text = f"Turn {game_state.turn_count + 1} start"
                    try:
                        await self._update_board(game_state, error_channel=ctx.channel, target_thread="map", also_post_to_game=also_post_to_game, description_text=description_text)
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
                    player_number = self._get_player_number(game_state, player.user_id)
                    turn_number = game_state.turn_count + 1
                    description_text = f"Turn {turn_number} Player {player_number}" if player_number and target_thread == "map" else None
                    try:
                        await self._update_board(game_state, error_channel=ctx.channel, target_thread=target_thread, also_post_to_game=also_post_to_game, description_text=description_text)
                        logger.info("Board updated after movement (turn not complete)")
                    except Exception as exc:
                        logger.exception("CRITICAL: Error updating board after movement: %s", exc)
                        try:
                            await ctx.reply("❌ Error updating board. The roll was processed, but the board may not have updated.", mention_author=False)
                        except Exception:
                            pass
                    # Auto-save after movement (turn not complete)
                    await self._save_auto_save(game_state, ctx)
                    autosave_done = True

                # Auto-save after roll even if no movement/turn complete
                if not autosave_done:
                    await self._save_auto_save(game_state, ctx)
                
                # After board/turn handling, show next turn info or game over standings
                try:
                    if pack and hasattr(pack, "get_game_data"):
                        try:
                            data = pack.get_game_data(game_state)
                            game_ended = data.get("game_ended", False)
                        except Exception as exc:
                            logger.warning("Failed to call pack.get_game_data during dice game_ended check: %s", exc)
                            game_ended = False
                            data = None
                        
                        if data:
                            goal_turns = data.get("goal_reached_turn", {}) or {}
                            turn_order = data.get("turn_order", [])
                            player_numbers = data.get("player_numbers", {})
                            turn_order_index = {uid: idx for idx, uid in enumerate(turn_order)}
                            
                            if game_ended:
                                # Get winners and forfeited players
                                winners = data.get("winners", [])
                                forfeited_players = set(data.get("forfeited_players", []))
                                
                                # Build finish order sorted by turn reached, then turn order (for finished players only)
                                ordered_finishers = sorted(
                                    goal_turns.items(),
                                    key=lambda item: (
                                        item[1],
                                        turn_order_index.get(item[0], 10_000_000),
                                        item[0],
                                    ),
                                )
                                
                                # Get forfeited players who didn't finish (not in goal_turns)
                                forfeited_not_finished = [
                                    user_id for user_id in forfeited_players
                                    if user_id not in goal_turns
                                ]
                                
                                # Build leaderboard message
                                lines = ["🏁 **Game Over!**", ""]
                                
                                # Show WINNER(s) first at top
                                if winners:
                                    winner_mentions = []
                                    for user_id in winners:
                                        player_obj = game_state.players.get(user_id)
                                        pnum = player_numbers.get(user_id, "?")
                                        name = player_obj.character_name if player_obj and player_obj.character_name else f"Player {pnum}"
                                        mention = f"<@{user_id}>"
                                        winner_mentions.append(f"{name} ({mention})")
                                    
                                    if len(winners) == 1:
                                        lines.append(f"🏆 **WINNER:** {winner_mentions[0]}")
                                    else:
                                        lines.append(f"🏆 **WINNERS:** {', '.join(winner_mentions)}")
                                    lines.append("")
                                
                                # Show finish order: all finished players (1st, 2nd, 3rd, etc.)
                                if ordered_finishers:
                                    lines.append("**Finish Order:**")
                                    for rank, (user_id, turn_num) in enumerate(ordered_finishers, start=1):
                                        player_obj = game_state.players.get(user_id)
                                        pnum = player_numbers.get(user_id, "?")
                                        name = player_obj.character_name if player_obj and player_obj.character_name else f"Player {pnum}"
                                        mention = f"<@{user_id}>"
                                        lines.append(f"{rank}) {name} (Player {pnum}) {mention} — Turn {turn_num}")
                                    
                                    # Add forfeited players at end if any
                                    if forfeited_not_finished:
                                        if ordered_finishers:
                                            lines.append("")  # Empty line separator
                                        for user_id in forfeited_not_finished:
                                            player_obj = game_state.players.get(user_id)
                                            pnum = player_numbers.get(user_id, "?")
                                            name = player_obj.character_name if player_obj and player_obj.character_name else f"Player {pnum}"
                                            mention = f"<@{user_id}>"
                                            lines.append(f"❌ {name} (Player {pnum}) {mention} — **FORFEIT/QUIT**")
                                
                                await ctx.channel.send("\n".join(lines), allowed_mentions=discord.AllowedMentions.none())
                            else:
                                players_rolled = set(data.get("players_rolled_this_turn", []))
                                forfeited_players = set(data.get("forfeited_players", []))
                                # Skip players who already reached the goal (win_tile from rules) and forfeited players
                                rules = game_config.rules or {}
                                win_tile = int(rules.get("win_tile", 100))
                                pending = []
                                for user_id in turn_order:
                                    if user_id in players_rolled:
                                        continue
                                    # Skip forfeited players
                                    if user_id in forfeited_players:
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
                # Queue the message instead of just replying
                await self._cache_and_delete_message(ctx.message, thread_id, "dice command - another command processing")
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
            
            if not self._is_actual_gm(ctx.author, game_state):
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
            await self._update_board(game_state, error_channel=ctx.channel, target_thread="map", also_post_to_game=has_separate_map_thread, description_text="Game started")
            
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
            
            # Allow both actual GM and admins to end games
            is_actual_gm = self._is_actual_gm(ctx.author, game_state)
            is_admin_user = is_admin(ctx.author) or is_bot_mod(ctx.author)
            if not (is_actual_gm or is_admin_user):
                await ctx.reply("Only the GM or an admin can end games.", mention_author=False)
                return
            
            game_state.is_locked = True
            
            # Delete all saves for this game
            await self._delete_game_saves(game_state)
            
            # Lock the thread
            if isinstance(ctx.channel, discord.Thread):
                try:
                    await ctx.channel.edit(locked=True)
                except discord.HTTPException as exc:
                    logger.warning("Failed to lock thread: %s", exc)
            
            await ctx.reply("Game ended. Thread locked.", mention_author=False)
            await self._log_action(game_state, f"Game ended by {ctx.author.display_name}")
        
        await self._execute_gameboard_command(ctx, _impl)

    async def command_gamequit(self, ctx: commands.Context) -> None:
        """Forfeit the game (player command). Removes player from game entirely (opposite of addplayer)."""
        async def _impl():
            game_state = await self._get_game_state_for_context(ctx)
            if not game_state:
                await ctx.reply("No active game in this thread.", mention_author=False)
                return
            
            if game_state.is_locked:
                await ctx.reply("Game is locked.", mention_author=False)
                return
            
            # Check if player is in game
            if ctx.author.id not in game_state.players:
                await ctx.reply("You're not in this game.", mention_author=False)
                return
            
            # Get pack_data
            pack = get_game_pack(game_state.game_type, self.packs_dir)
            if not pack or not hasattr(pack, 'module'):
                await ctx.reply("Game pack not found.", mention_author=False)
                return
            
            pack_module = pack.module
            get_game_data = getattr(pack_module, "get_game_data", None)
            if not callable(get_game_data):
                await ctx.reply("Game data function not found.", mention_author=False)
                return
            
            try:
                data = get_game_data(game_state)
            except Exception as exc:
                logger.warning("Failed to call get_game_data during gamequit: %s", exc)
                await ctx.reply("Error accessing game data.", mention_author=False)
                return
            
            # Get player info before removal
            player = game_state.players.get(ctx.author.id)
            player_name = player.character_name if player and player.character_name else ctx.author.display_name
            player_number = self._get_player_number(game_state, ctx.author.id)
            
            # CRITICAL: Preserve grid_position before removing player
            # Store it in pack_data so it can be restored when re-added
            if player:
                # Store grid_position in a preserved positions dict
                if 'removed_player_positions' not in data:
                    data['removed_player_positions'] = {}
                data['removed_player_positions'][ctx.author.id] = player.grid_position
                logger.debug("Preserved grid_position %s for removed player %s", player.grid_position, ctx.author.id)
            
            # CRITICAL: Do NOT remove player from game_state.players - they should stay on the board
            # Player stays in turn_order, stays on gameboard, but cannot roll dice
            # Their turns will be skipped via forfeited_players check during turn processing
            # DO NOT delete: del game_state.players[ctx.author.id]  # Keep player on board
            
            # CRITICAL: Do NOT remove from turn_order - players should stay in turn_order when forfeited
            # Their turns will be skipped via forfeited_players check during turn processing
            
            # CRITICAL: Keep player_numbers - player number is preserved for re-adding
            # Example: Players 1, 2, 3, 4 -> Player 3 quits -> Still 1, 2, 4 (not renumbered)
            # If Player 3 is re-added, they remain Player 3
            # player_numbers are NOT removed - they stay assigned
            
            # CRITICAL: Keep tile_numbers - position is preserved for re-adding
            # Example: Player 3 on tile 25 quits -> tile_numbers[user_id] = 25 stays
            # If Player 3 is re-added, they return to tile 25
            # tile_numbers are NOT removed - they stay assigned
            
            # CRITICAL: Add to forfeited_players so their turns are skipped
            # They stay in turn_order but cannot roll dice
            forfeited_players = data.get('forfeited_players', [])
            if ctx.author.id not in forfeited_players:
                forfeited_players.append(ctx.author.id)
                data['forfeited_players'] = forfeited_players
            
            # Remove from winners (if present)
            winners = data.get('winners', [])
            if ctx.author.id in winners:
                winners.remove(ctx.author.id)
                data['winners'] = winners
            
            # Remove from players_rolled_this_turn (if present)
            players_rolled = data.get('players_rolled_this_turn', [])
            if ctx.author.id in players_rolled:
                players_rolled.remove(ctx.author.id)
                data['players_rolled_this_turn'] = players_rolled
            
            # Remove from other character-specific metadata
            for key in ('original_characters', 'real_body_characters', 'transformation_counts', 'mind_changed', 'goal_reached_turn'):
                metadata_dict = data.get(key, {})
                if isinstance(metadata_dict, dict) and ctx.author.id in metadata_dict:
                    del metadata_dict[ctx.author.id]
                    data[key] = metadata_dict
            
            # Remove from players_reached_end_this_turn
            players_reached_end = data.get('players_reached_end_this_turn', [])
            if ctx.author.id in players_reached_end:
                players_reached_end.remove(ctx.author.id)
                data['players_reached_end_this_turn'] = players_reached_end
            
            # Update board (player stays on board but is marked as forfeited)
            description_text = f"Player {player_number} quit" if player_number else "Player quit"
            await self._update_board(game_state, error_channel=ctx.channel, description_text=description_text)
            
            # Reply to player
            if player_number:
                await ctx.reply(f"😔 **{player_name}** (Player {player_number}) has forfeited. Your token stays on the board, but you cannot roll dice. You can be re-activated with `!addplayer` and `!assign` if needed.", mention_author=False)
            else:
                await ctx.reply(f"😔 **{player_name}** has forfeited. Your token stays on the board, but you cannot roll dice. You can be re-activated with `!addplayer` and `!assign` if needed.", mention_author=False)
            
            await self._log_action(game_state, f"{ctx.author.display_name} forfeited (stays on board, cannot roll)")
            
            # Auto-save after player quits
            await self._save_auto_save(game_state, ctx)
        
        await self._execute_gameboard_command(ctx, _impl)

    async def command_players(self, ctx: commands.Context) -> None:
        """List players in player-number order with status (gameboard)."""
        if not isinstance(ctx.author, discord.Member):
            return

        game_state = await self._get_game_state_for_context(ctx)
        if not game_state:
            return

        is_gm = self._is_actual_gm(ctx.author, game_state)
        if not is_gm:
            key = (game_state.game_thread_id, ctx.author.id)
            now = time.monotonic()
            last_used = self._players_command_cooldowns.get(key, 0.0)
            if now - last_used < 30:
                return
            self._players_command_cooldowns[key] = now

        pack = get_game_pack(game_state.game_type, self.packs_dir)
        data = {}
        if pack and pack.has_function("get_game_data"):
            try:
                data = pack.call("get_game_data", game_state)
            except Exception as exc:
                logger.warning("Failed to call get_game_data for players list: %s", exc)
                data = {}

        forfeited_players = set(data.get("forfeited_players", [])) if isinstance(data, dict) else set()
        turn_order = data.get("turn_order", []) if isinstance(data, dict) else []
        forfeited_players = set(data.get("forfeited_players", [])) if isinstance(data, dict) else set()
        current_turn_user_id = None
        if turn_order:
            # Mirror pack turn logic: first eligible player who hasn't rolled, isn't forfeited, and isn't at goal
            rules = self.get_game_config(game_state.game_type).rules if self.get_game_config(game_state.game_type) else {}
            win_tile = int((rules or {}).get("win_tile", 100))
            players_rolled = set(data.get("players_rolled_this_turn", [])) if isinstance(data, dict) else set()
            for user_id in turn_order:
                if user_id not in game_state.players:
                    continue
                if user_id in forfeited_players:
                    continue
                if user_id in players_rolled:
                    continue
                tile_num = data.get("tile_numbers", {}).get(user_id, 1) if isinstance(data, dict) else 1
                if tile_num >= win_tile:
                    continue
                current_turn_user_id = user_id
                break

        entries: List[Tuple[int, int, GamePlayer, Optional[int]]] = []
        for user_id, player in game_state.players.items():
            player_number = self._get_player_number(game_state, user_id)
            sort_key = player_number if player_number is not None else 9_999
            entries.append((sort_key, user_id, player, player_number))

        entries.sort(key=lambda item: item[0])

        lines: List[str] = ["🏅 **Players**", ""]
        for _, user_id, player, player_number in entries:
            member = ctx.guild.get_member(user_id) if ctx.guild else None
            username = member.display_name if isinstance(member, discord.Member) else f"User {user_id}"
            character = player.character_name or "Unassigned"
            mention = f"<@{user_id}>"

            status_parts: List[str] = []
            if user_id in forfeited_players:
                status_parts.append("removed/quit")

            state = game_state.player_states.get(user_id)
            if state and state.form_owner_user_id and state.form_owner_user_id != user_id:
                swap_partner_id = None
                swap_chain = self._build_swap_chain(game_state, user_id)
                for left_id, right_id in swap_chain:
                    if left_id == user_id:
                        swap_partner_id = right_id
                        break
                    if right_id == user_id:
                        swap_partner_id = left_id
                        break
                if swap_partner_id is None:
                    swap_partner_id = state.form_owner_user_id

                if swap_partner_id:
                    partner_member = ctx.guild.get_member(swap_partner_id) if ctx.guild else None
                    partner_name = (
                        partner_member.display_name
                        if isinstance(partner_member, discord.Member)
                        else f"User {swap_partner_id}"
                    )
                    partner_num = self._get_player_number(game_state, swap_partner_id)
                    partner_num_display = partner_num if partner_num is not None else "?"
                    status_parts.append(f"swapped with {partner_name} (Player {partner_num_display})")
                else:
                    status_parts.append("swapped")

            status = ", ".join(status_parts) if status_parts else "active"
            player_num_display = player_number if player_number is not None else "?"
            lines.append(f"{player_num_display}) {character} — {username} ({mention}) — {status}")

        if current_turn_user_id is not None:
            current_player = game_state.players.get(current_turn_user_id)
            current_member = ctx.guild.get_member(current_turn_user_id) if ctx.guild else None
            current_name = (
                current_member.display_name if isinstance(current_member, discord.Member) else f"User {current_turn_user_id}"
            )
            current_character = current_player.character_name if current_player and current_player.character_name else "Unassigned"
            current_num = self._get_player_number(game_state, current_turn_user_id)
            current_num_display = current_num if current_num is not None else "?"
            current_mention = f"<@{current_turn_user_id}>"
            lines.extend([
                "",
                "➡️ **Current turn:**",
                f"Player {current_num_display} — {current_name} ({current_mention}) — {current_character}",
            ])
        else:
            lines.extend(["", "➡️ **Current turn:**", "Unknown"])

        if not lines:
            await ctx.send("No players in this game.", allowed_mentions=discord.AllowedMentions.none())
            return

        await ctx.send("\n".join(lines), allowed_mentions=discord.AllowedMentions.none())

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
            
            if not self._is_actual_gm(ctx.author, game_state):
                await ctx.reply("Only the GM can remove players.", mention_author=False)
                return
            
            # Resolve member from token if provided, otherwise use provided member
            # Use _resolve_target_member for consistent parsing (supports character names/folders)
            resolved_member = member
            if token and not resolved_member:
                resolved_member = self._resolve_target_member(ctx, game_state, token)
            
            if not resolved_member:
                await ctx.reply("Usage: `!removeplayer @user` or `!removeplayer character_name` or `!removeplayer character_folder`", mention_author=False)
                return
            
            if resolved_member.id not in game_state.players:
                await ctx.reply(f"{resolved_member.display_name} is not in the game.", mention_author=False)
                return
            
            # CRITICAL: Preserve position data before removing
            player = game_state.players.get(resolved_member.id)
            if player:
                # Get pack_data to store preserved positions
                pack = get_game_pack(game_state.game_type, self.packs_dir)
                if pack and pack.has_function("get_game_data"):
                    pack_module = pack.module
                    get_game_data = getattr(pack_module, "get_game_data", None)
                    if callable(get_game_data):
                        try:
                            data = get_game_data(game_state)
                            
                            # CRITICAL: Keep tile_numbers - position is preserved for re-adding
                            # tile_numbers are NOT removed - they stay assigned
                            
                            # Store grid_position in preserved positions dict
                            if 'removed_player_positions' not in data:
                                data['removed_player_positions'] = {}
                            data['removed_player_positions'][resolved_member.id] = player.grid_position
                            logger.debug("Preserved grid_position %s for removed player %s", player.grid_position, resolved_member.id)
                        except Exception as exc:
                            logger.warning("Failed to call get_game_data during removeplayer position preservation: %s", exc)
                
                # Also remove from pack data structures (turn_order, etc.)
                if pack and pack.has_function("get_game_data"):
                    pack_module = pack.module
                    get_game_data = getattr(pack_module, "get_game_data", None)
                    if callable(get_game_data):
                        try:
                            data = get_game_data(game_state)
                            # CRITICAL: Do NOT remove from turn_order - players should stay in turn_order when removed
                            # Their turns will be skipped via forfeited_players check during turn processing
                        except Exception as exc:
                            logger.warning("Failed to call get_game_data during removeplayer cleanup: %s", exc)
                            data = None
                        
                        if data:
                            # Remove from winners (if present)
                            winners = data.get('winners', [])
                            if resolved_member.id in winners:
                                winners.remove(resolved_member.id)
                                data['winners'] = winners
                            # Remove from players_rolled_this_turn (if present)
                            players_rolled = data.get('players_rolled_this_turn', [])
                            if resolved_member.id in players_rolled:
                                players_rolled.remove(resolved_member.id)
                                data['players_rolled_this_turn'] = players_rolled
                            # CRITICAL: Add to forfeited_players so their turns are skipped
                            # They stay in turn_order and on gameboard, but cannot roll dice
                            forfeited_players = data.get('forfeited_players', [])
                            if resolved_member.id not in forfeited_players:
                                forfeited_players.append(resolved_member.id)
                                data['forfeited_players'] = forfeited_players
                            
                            # Remove from other character-specific metadata
                            for key in ('original_characters', 'real_body_characters', 'transformation_counts', 'mind_changed', 'goal_reached_turn'):
                                metadata_dict = data.get(key, {})
                                if isinstance(metadata_dict, dict) and resolved_member.id in metadata_dict:
                                    del metadata_dict[resolved_member.id]
                                    data[key] = metadata_dict
                            
                            # Remove from players_reached_end_this_turn
                            players_reached_end = data.get('players_reached_end_this_turn', [])
                            if resolved_member.id in players_reached_end:
                                players_reached_end.remove(resolved_member.id)
                                data['players_reached_end_this_turn'] = players_reached_end
            
            # CRITICAL: Do NOT remove from game_state.players - they should stay on the board
            # Player stays in turn_order, stays on gameboard, but cannot roll dice
            # Their turns will be skipped via forfeited_players check during turn processing
            # DO NOT delete: del game_state.players[resolved_member.id]  # Keep player on board
            
            # Update board (player stays on board but is marked as forfeited)
            player_number = self._get_player_number(game_state, resolved_member.id)
            description_text = f"Player {player_number} removed" if player_number else "Player removed"
            await self._update_board(game_state, error_channel=ctx.channel, description_text=description_text)
            
            # Auto-save after player is removed
            await self._save_auto_save(game_state, ctx)
            await ctx.reply(f"Removed {resolved_member.display_name} from active play. Token stays on board, but they cannot roll dice.", mention_author=False)
            await self._log_action(game_state, f"Player {resolved_member.display_name} removed (stays on board, cannot roll)")
        
        await self._execute_gameboard_command(ctx, _impl)

    async def command_bg_list(self, ctx: commands.Context) -> None:
        """List available backgrounds (game-specific, isolated from global VN). DMs the GM like VN mode."""
        from tfbot.panels import get_background_root, list_background_choices, VN_BACKGROUND_DEFAULT_RELATIVE
        
        game_state = await self._get_game_state_for_context(ctx)
        if not game_state:
            await ctx.reply("No active game in this thread.", mention_author=False)
            return
        
        # Get GM user (author if they're actual GM, otherwise the game's GM)
        gm_user = ctx.author
        if not self._is_actual_gm(ctx.author, game_state):
            # Not actual GM - get the actual GM
            if game_state.gm_user_id and ctx.guild:
                gm_user = ctx.guild.get_member(game_state.gm_user_id)
                if not gm_user:
                    await ctx.reply("GM not found. Only the GM can view the background list.", mention_author=False)
                    return
            else:
                await ctx.reply("Only the GM can view the background list.", mention_author=False)
                return
        
        bg_root = get_background_root()
        if bg_root is None:
            try:
                await gm_user.send("Backgrounds are not configured on this bot.")
            except discord.Forbidden:
                if ctx.guild:
                    await ctx.send("I couldn't DM you. Please enable direct messages.", delete_after=10)
            return
        
        choices = list_background_choices()
        if not choices:
            try:
                await gm_user.send("No background images were found in the configured directory.")
            except discord.Forbidden:
                if ctx.guild:
                    await ctx.send("I couldn't DM you. Please enable direct messages.", delete_after=10)
            return
        
        lines: list[str] = []
        for idx, path in enumerate(choices, start=1):
            try:
                if bg_root:
                    relative = path.resolve().relative_to(bg_root.resolve())
                    display = relative.as_posix()
                else:
                    display = path.name
            except ValueError:
                display = str(path)
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
        
        default_display = (
            VN_BACKGROUND_DEFAULT_RELATIVE.as_posix()
            if VN_BACKGROUND_DEFAULT_RELATIVE
            else "system default"
        )
        instructions = (
            "Use `!bg @user <number>` or `!bg all <number>` to apply that background in gameboard mode.\n"
            f"Example: `!bg @user 45` selects option 45 from the list.\n"
            f"The default background is `{default_display}`."
        )
        
        try:
            for chunk in chunks:
                await gm_user.send(f"```\n{chunk}\n```")
            await gm_user.send(instructions)
            if ctx.author != gm_user:
                await ctx.reply(f"Background list sent to {gm_user.mention} via DM.", mention_author=False)
            else:
                await ctx.reply("Background list sent to you via DM.", mention_author=False)
        except discord.Forbidden:
            if ctx.guild:
                await ctx.send("I couldn't DM you. Please enable direct messages, then rerun `!bg_list`.", delete_after=10)

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
            
            if not self._is_actual_gm(ctx.author, game_state):
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
            is_all_explicit = False  # Track if "all" was explicitly used
            
            # CRITICAL: If target is None and bg_id is just a number, it means "all" was used in bot.py
            # Check if bg_id_str is just a number (no spaces, no "all" token)
            if not resolved_target:
                # Check if bg_id_str is just a number (called from bot.py with !bg all <number>)
                try:
                    int(bg_id_str)  # Try to parse as number
                    # If it's just a number and target is None, it means "all" was used
                    is_all_explicit = True
                    resolved_target = None  # None means "all"
                    bg_id_value = bg_id_str
                except ValueError:
                    # Not just a number - parse tokens
                    if len(tokens) >= 1:
                        first_token = tokens[0].lower()
                        if first_token == "all":
                            # Format: !bg all <number>
                            resolved_target = None  # None means "all"
                            is_all_explicit = True  # Mark that "all" was explicitly used
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
                        else:
                            # Only one token - treat it as bg_id (no target specified)
                            # This means user provided just a number, so we need a target
                            # If no target was provided, show error
                            await ctx.reply("Usage: `!bg @user <number>` or `!bg character_name <number>` or `!bg all <number>`", mention_author=False)
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
            # CRITICAL: Only apply to all if explicitly "all" was used, not if target resolution failed
            if resolved_target is None:
                if not is_all_explicit:
                    # Target resolution failed - don't apply to all
                    await ctx.reply("Could not find target. Use `!bg @user <number>`, `!bg character_name <number>`, or `!bg all <number>`.", mention_author=False)
                    return
                # Set for all players (game-specific only, doesn't touch global state)
                for player in game_state.players.values():
                    player.background_id = bg_id_int
                await ctx.reply(f"Set background {bg_id_int} for all players (game VN only).", mention_author=False)
                await self._log_action(game_state, f"All players background set to {bg_id_int}")
            elif resolved_target.id in game_state.players:
                game_state.players[resolved_target.id].background_id = bg_id_int
                await ctx.reply(f"Set background {bg_id_int} for {resolved_target.display_name} (game VN only).", mention_author=False)
                await self._log_action(game_state, f"{resolved_target.display_name} background set to {bg_id_int}")
            else:
                await ctx.reply(f"{resolved_target.display_name} is not in the game.", mention_author=False)
            
            # Note: Auto-save removed - use !savegame to save manually
        
        await self._execute_gameboard_command(ctx, _impl)

    async def command_outfit_list(self, ctx: commands.Context, target: Optional[str] = None) -> None:
        """List available outfits for a character or user's character. Supports: !outfit_list character_name OR !outfit_list @user OR !outfit_list user_name"""
        game_state = await self._get_game_state_for_context(ctx)
        if not game_state:
            await ctx.reply("No active game in this thread.", mention_author=False)
            return
        
        resolved_character = None
        
        if target:
            # Try to resolve as member first (user mention, username, or display name)
            resolved_member = None
            if ctx.guild:
                # Use the same resolution logic as other commands (supports @user, character_name, username)
                resolved_member = self._resolve_target_member(ctx, game_state, target)
            
            if resolved_member:
                # User was specified - get their character
                if resolved_member.id in game_state.players:
                    player = game_state.players[resolved_member.id]
                    resolved_character = player.character_name
                    if not resolved_character:
                        await ctx.reply(f"{resolved_member.display_name} doesn't have a character assigned yet.", mention_author=False)
                        return
                else:
                    await ctx.reply(f"{resolved_member.display_name} is not in the game.", mention_author=False)
                    return
            else:
                # Treat as character name (not found as member)
                resolved_character = target.strip()
        else:
            # No target - show current game players and their characters
            lines = ["**Current Game Players:**"]
            for player in game_state.players.values():
                if player.character_name:
                    char_info = f"- {player.character_name}"
                    if player.outfit_name:
                        char_info += f" (current outfit: {player.outfit_name})"
                    lines.append(char_info)
            await ctx.reply("\n".join(lines) + "\n\nUse `!outfit_list character_name` or `!outfit_list @user` to see outfits.", mention_author=False)
            return
        
        # List outfits for specific character (use categorized format like VN mode)
        from tfbot.panels import list_pose_outfits
        
        pose_outfits = list_pose_outfits(resolved_character)
        if not pose_outfits:
            await ctx.reply(f"No outfits found for character: {resolved_character}", mention_author=False)
            return
        
        # Format output to match VN mode exactly: pose: outfit1, outfit2, outfit3, ...
        pose_lines = []
        for pose, options in sorted(pose_outfits.items()):
            pose_lines.append(f"{pose}: {', '.join(sorted(options))}")
        lines = [f"**Available Outfits for {resolved_character} (Game VN):**"]
        lines.extend(f"- {line}" for line in pose_lines)
        
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
            
            if not self._is_actual_gm(ctx.author, game_state):
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
                                await ctx.reply(f"{resolved_member.display_name} is not in the game.", mention_author=False)
                                return
                            player = game_state.players[resolved_member.id]
                            if not player.character_name:
                                await ctx.reply(f"{resolved_member.display_name} doesn't have a character assigned yet.", mention_author=False)
                                return
                            await self.command_outfit_list(ctx, target=player.character_name)
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
                    await ctx.reply(f"{resolved_member.display_name} is not in the game.", mention_author=False)
                    return
                player = game_state.players[resolved_member.id]
                if not player.character_name:
                    await ctx.reply(f"{resolved_member.display_name} doesn't have a character assigned yet.", mention_author=False)
                    return
                await self.command_outfit_list(ctx, target=player.character_name)
                return
            
            if resolved_member.id not in game_state.players:
                await ctx.reply(f"{resolved_member.display_name} is not in the game.", mention_author=False)
                return
            
            player = game_state.players[resolved_member.id]
            if not player.character_name:
                await ctx.reply(f"{resolved_member.display_name} doesn't have a character assigned yet. Use `!assign` first.", mention_author=False)
                return
            
            # Set outfit (game-specific only, doesn't touch global vn_outfit_selection)
            player.outfit_name = outfit_to_set.strip()
            # Note: Auto-save removed - use !savegame to save manually
            await ctx.reply(f"Set outfit '{outfit_to_set}' for {resolved_member.display_name} (game VN only).", mention_author=False)
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
            
            if not self._is_actual_gm(ctx.author, game_state):
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
        
        game_state = await self._get_game_state_for_context(ctx)
        if not game_state:
            await ctx.reply("No active game in this thread.", mention_author=False)
            return

        if not self._is_actual_gm(ctx.author, game_state):
            await ctx.reply("Only the GM can load games.", mention_author=False)
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
            
            # Always re-read enabled_packs from current config (ignore saved value)
            # This ensures games use current pack configuration, not outdated saved values
            game_type = str(data.get("game_type", ""))
            enabled_packs = None
            if game_type:
                from tf_characters import get_enabled_packs_for_game, BOT_NAME
                enabled_packs = get_enabled_packs_for_game(game_type, BOT_NAME)
                saved_enabled_packs = data.get("enabled_packs")
                if saved_enabled_packs and set(saved_enabled_packs) != enabled_packs:
                    logger.info("Game %s: enabled_packs updated from saved %s to current config %s", 
                               thread_id, saved_enabled_packs, sorted(enabled_packs))
            
            game_state = GameState(
                game_thread_id=thread_id,
                forum_channel_id=int(data.get("forum_channel_id", 0)),
                dm_channel_id=int(data.get("dm_channel_id", 0)),
                gm_user_id=int(data.get("gm_user_id", 0)),
                game_type=game_type,
                map_thread_id=data.get("map_thread_id"),  # CRITICAL: Restore map_thread_id so map thread continues to receive updates
                players=players,
                current_turn=data.get("current_turn"),
                board_message_id=data.get("board_message_id"),
                is_locked=bool(data.get("is_locked", False)),
                narrator_user_id=data.get("narrator_user_id"),  # Restore narrator_user_id
                debug_mode=bool(data.get("debug_mode", False)),
                turn_count=int(data.get("turn_count", 0)),
                game_started=bool(data.get("game_started", False)),  # Default to False for old saves
                is_paused=bool(data.get("is_paused", False)),  # Default to False for old saves
                bot_user_id=data.get("bot_user_id"),  # Load bot ownership (None for old saves)
                enabled_packs=enabled_packs,  # Restore saved enabled packs (or from config for old saves)
            )
            
            # ADD pack_data restoration if it exists
            if "pack_data" in data and data["pack_data"]:
                game_state._pack_data = data["pack_data"]
            
            # CRITICAL: Sanitize pack_data after loading (handles old save files without pack_data)
            # Add None/empty checks first (handles old save files without pack_data)
            if not hasattr(game_state, '_pack_data') or not game_state._pack_data:
                # Initialize empty pack_data if missing (get_game_data will handle defaults)
                pack = get_game_pack(game_state.game_type, self.packs_dir)
                if pack and pack.has_function("get_game_data"):
                    try:
                        game_state._pack_data = pack.call("get_game_data", game_state)
                    except Exception as exc:
                        logger.warning("Failed to call pack.get_game_data during loadgame: %s", exc)
                        # Fallback to minimal structure
                        game_state._pack_data = {
                            'tile_numbers': {},
                            'turn_order': [],
                            'player_numbers': {},
                            'players_rolled_this_turn': [],
                            'winners': [],
                            'forfeited_players': [],
                        }
                else:
                    # Fallback: create minimal structure
                    game_state._pack_data = {
                        'tile_numbers': {},
                        'turn_order': [],
                        'player_numbers': {},
                        'players_rolled_this_turn': [],
                        'winners': [],
                        'forfeited_players': [],
                    }
            
            # Verify pack_data is a dict (defensive programming)
            if not isinstance(game_state._pack_data, dict):
                logger.warning("Invalid pack_data type in save file, initializing new dict")
                game_state._pack_data = {}
            
            # Now sanitize all fields:
            if game_state._pack_data:
                # Sanitize turn_order: convert to int, deduplicate, filter existing players
                if "turn_order" in game_state._pack_data:
                    # Convert all to int, deduplicate, filter
                    seen = set()
                    turn_order_clean = []
                    for uid in game_state._pack_data['turn_order']:
                        try:
                            uid_int = int(uid) if isinstance(uid, str) else uid
                            if uid_int in game_state.players and uid_int not in seen:
                                turn_order_clean.append(uid_int)
                                seen.add(uid_int)
                        except (ValueError, TypeError) as exc:
                            logger.debug("Failed to sanitize turn_order entry %s: %s", uid, exc)
                            continue
                    game_state._pack_data['turn_order'] = turn_order_clean
                
                # Sanitize player_numbers: convert keys to int, filter existing players
                if "player_numbers" in game_state._pack_data:
                    player_numbers_clean = {}
                    for uid_str, num in game_state._pack_data['player_numbers'].items():
                        try:
                            uid_int = int(uid_str) if isinstance(uid_str, str) else uid_str
                            if uid_int in game_state.players:
                                player_numbers_clean[uid_int] = num
                        except (ValueError, TypeError) as exc:
                            logger.debug("Failed to sanitize player_numbers entry %s: %s", uid_str, exc)
                            continue
                    game_state._pack_data['player_numbers'] = player_numbers_clean
                
                # Clean up tile_numbers: remove entries for non-existent players (for consistency)
                if "tile_numbers" in game_state._pack_data:
                    tile_numbers_clean = {}
                    for uid_str, tile_num in game_state._pack_data['tile_numbers'].items():
                        try:
                            uid_int = int(uid_str) if isinstance(uid_str, str) else uid_str
                            if uid_int in game_state.players:
                                tile_numbers_clean[uid_int] = tile_num
                        except (ValueError, TypeError) as exc:
                            logger.debug("Failed to sanitize tile_numbers entry %s: %s", uid_str, exc)
                            continue
                    game_state._pack_data['tile_numbers'] = tile_numbers_clean
                
                # Deduplicate other lists (defensive programming)
                for list_key in ['winners', 'forfeited_players', 'players_rolled_this_turn', 'players_reached_end_this_turn']:
                    if list_key in game_state._pack_data and isinstance(game_state._pack_data[list_key], list):
                        try:
                            game_state._pack_data[list_key] = list(dict.fromkeys(
                                [int(uid) if isinstance(uid, str) else uid for uid in game_state._pack_data[list_key]]
                            ))
                        except (ValueError, TypeError) as exc:
                            logger.debug("Failed to sanitize %s list: %s", list_key, exc)
                            # Keep original list if sanitization fails (better than losing data)
                            continue
                
                # Log warning if turn_order becomes empty after sanitization (game might be invalid)
                if game_state._pack_data.get('turn_order') == [] and game_state.players:
                    logger.warning("turn_order is empty after sanitization but players exist - game state may be corrupted")

            # Apply default background for active players missing one (skip forfeited/removed)
            forfeited_players = set()
            if isinstance(game_state._pack_data, dict):
                raw_forfeited = game_state._pack_data.get("forfeited_players", [])
                if isinstance(raw_forfeited, list):
                    for uid in raw_forfeited:
                        try:
                            forfeited_players.add(int(uid) if isinstance(uid, str) else uid)
                        except (ValueError, TypeError):
                            continue
            for user_id, player in game_state.players.items():
                if player.background_id is None and user_id not in forfeited_players:
                    player.background_id = 415
            
            # RESTORE player_states from saved data if they exist
            if "player_states" in data and data["player_states"]:
                try:
                    for user_id_str, state_dict in data["player_states"].items():
                        try:
                            user_id = int(user_id_str)
                            # Use deserialize_state to recreate TransformationState
                            restored_state = deserialize_state(state_dict)
                            game_state.player_states[user_id] = restored_state
                            logger.debug("Restored player state for user %s from saved data", user_id)
                        except (ValueError, KeyError, TypeError) as exc:
                            logger.warning("Failed to restore player state for user %s: %s", user_id_str, exc)
                except Exception as exc:
                    logger.warning("Error restoring player_states: %s", exc)
            
            # Replace active game state in memory
            self._active_games[thread_id] = game_state
            
            # Update board display
            await self._update_board(game_state, error_channel=ctx.channel, description_text=f"Game loaded from {state_file_path.name}")
            
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
        rules_text += f"**Starting Position:** Tile {starting_tile} ({starting_pos})\n\n"
        
        # Tile Colors section
        rules_text += "**Tile Colors (GM Controlled):**\n"
        rules_text += "Tiles show colors when landed on. GM must manually trigger effects:\n"
        rules_text += "• Orange - Gender Swap\n"
        rules_text += "• Dark Blue - Age Regression or Progression depending on character\n"
        rules_text += "• Light Blue - Restore player's original body\n"
        rules_text += "• Purple - Load Saved Body (or original if none saved)\n"
        rules_text += "• Yellow - Random Transformation\n"
        rules_text += "• Red - Random Transformation for Someone Else\n"
        rules_text += "• Green - Body Swap\n"
        rules_text += "• Pink - Mind Change/Command (GM sets an RP condition that the player must follow)\n\n"
        rules_text += "**Note:** Tile colors are informational only. GM uses commands like `!reroll`, `!swap`, etc. to apply effects.\n"
        
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
            
            if not self._is_actual_gm(ctx.author, game_state):
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
            
            if not self._is_actual_gm(ctx.author, game_state):
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
        """Toggle debug mode on/off (GM only). Shows coordinate labels on board."""
        async def _impl():
            if not isinstance(ctx.author, discord.Member):
                await ctx.reply("This command can only be used inside a server.", mention_author=False)
                return
            
            game_state = await self._get_game_state_for_context(ctx)
            if not game_state:
                await ctx.reply("No active game in this thread.", mention_author=False)
                return
            
            # Check if user is GM
            if not self._is_actual_gm(ctx.author, game_state):
                await ctx.reply("Only the GM can toggle debug mode.", mention_author=False)
                return
            
            # Toggle debug mode
            game_state.debug_mode = not game_state.debug_mode
            # Note: Auto-save removed - use !savegame to save manually
            
            status = "ON" if game_state.debug_mode else "OFF"
            await ctx.reply(f"Debug mode is now **{status}**. Board will show coordinate labels when debug is enabled.", mention_author=False)
            
            # Update board to show/hide debug layer
            await self._update_board(game_state, error_channel=ctx.channel, description_text=f"Debug mode {status.lower()}")
            await self._log_action(game_state, f"Debug mode toggled {status} by {ctx.author.display_name}")
        
        await self._execute_gameboard_command(ctx, _impl)

    async def command_transfergm(self, ctx: commands.Context, member: Optional[discord.Member] = None) -> None:
        """Transfer GM role to another user (current GM or admin only)."""
        async def _impl():
            if not isinstance(ctx.author, discord.Member):
                await ctx.reply("This command can only be used inside a server.", mention_author=False)
                return
            
            game_state = await self._get_game_state_for_context(ctx)
            if not game_state:
                await ctx.reply("No active game in this thread.", mention_author=False)
                return
            
            # Allow both actual GM and admins to transfer GM (admins can override in case of disconnect/abuse)
            is_actual_gm = self._is_actual_gm(ctx.author, game_state)
            is_admin_user = is_admin(ctx.author) or is_bot_mod(ctx.author)
            if not (is_actual_gm or is_admin_user):
                await ctx.reply("Only the current GM or an admin can transfer the GM role.", mention_author=False)
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
            
            # Always remove narrator status from old GM (they become a regular player)
            if game_state.narrator_user_id == old_gm_id:
                # Old GM was narrator - remove narrator role (they're now a regular player)
                game_state.narrator_user_id = None
                logger.info("Removed narrator status from old GM (user_id: %s)", old_gm_id)
            
            # Set narrator for new GM if they're not a player
            new_gm_player = game_state.players.get(member.id)
            if not (new_gm_player and new_gm_player.character_name):
                # New GM is not a player - make them narrator
                game_state.narrator_user_id = member.id
                logger.info("Set new GM as narrator (user_id: %s)", member.id)
            else:
                # New GM is a player - no narrator role needed
                if game_state.narrator_user_id == member.id:
                    game_state.narrator_user_id = None
                    logger.info("New GM is a player, removed narrator role")
            
            # Note: Auto-save removed - use !savegame to save manually
            await ctx.reply(f"GM role transferred to {member.display_name}.", mention_author=False)
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
`!rules` - Show game rules
`!gamequit` - Forfeit (token stays on board)
`!help` - Show this help

**Game Control (GM Only):**
`!startgame <type>` - Start new game (Admin only)
`!start` - Begin or resume game
`!pause` / `!resume` - Pause or resume dice rolls
`!endgame` - End game and lock thread
`!transfergm @user` - Transfer GM role
`!listgames` - List available games

**Player Management (GM Only):**
`!addplayer @user [char]` - Add player
`!removeplayer @user/char` - Remove player
`!assign @user/char <char>` - Assign character
`!reroll @user/char` - Reroll character
`!swap @user1 @user2` - Swap characters, positions, and backgrounds
`!pswap @user1 @user2` - Permanent swap (no reroll block/reversal)

**Token Movement (GM Only):**
`!movetoken @user/char <coord>` - Move token (e.g., A1)

**Visual Customization (GM Only):**
`!bg @user/all <id>` - Set background
`!bg` (no args) - List backgrounds (DMs GM)
`!outfit @user/char <outfit>` - Set outfit
`!outfit` (no args) - List outfits for your character; add target for others

**Save/Load (GM Only):**
`!savegame` - Save game state
`!loadgame <file>` - Load game state

**Debug (GM/Admin Only):**
`!debug` - Toggle coordinate labels

**Tips:**
- Commands support @user, character names, or folders
- `!gamequit` preserves position for re-adding
- Paused games block dice (GM can force rolls)"""
        
        await ctx.reply(help_text, mention_author=False)
