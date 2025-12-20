"""Data models for game board framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

# Import here to avoid circular dependency - only used in type annotation
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import TransformationState


@dataclass
class GamePlayer:
    """Represents a player in a game."""
    user_id: int
    character_name: Optional[str] = None
    grid_position: str = "A1"  # Alphanumeric grid position (e.g., "A1", "B5")
                                # Format: Letter (column) + Number (row)
                                # A1 = bottom left, increasing right and up (1-indexed)
    background_id: Optional[int] = None
    outfit_name: Optional[str] = None
    token_image: str = "default.png"  # Planned: path to token image (deferred - use colored markers initially)


@dataclass
class GameState:
    """Represents the state of an active game."""
    game_thread_id: int         # Forum thread ID where game runs (chat/commands)
    forum_channel_id: int       # Parent forum channel ID
    dm_channel_id: int          # GM command channel ID
    gm_user_id: int
    game_type: str              # References game config file name (e.g., "snakes_ladders")
    map_thread_id: Optional[int] = None  # Forum thread ID for board images (separate post in same channel)
    players: Dict[int, GamePlayer] = field(default_factory=dict)
    current_turn: Optional[int] = None
    board_message_id: Optional[int] = None  # Latest board message ID in map thread (for reference, not deletion)
    is_locked: bool = False     # Game ended - thread locked but viewable
    narrator_user_id: Optional[int] = None  # User ID of narrator (GM unless GM is a player)
    player_states: Dict[int, "TransformationState"] = field(default_factory=dict)  # Player user_id -> TransformationState
    debug_mode: bool = False    # Debug mode: shows coordinate labels on board
    turn_count: int = 0  # Current turn number (increments each turn)
    game_started: bool = False  # Game ready to start - blocks dice rolls until GM issues !start command


@dataclass
class GameConfig:
    """Configuration for a game type loaded from JSON."""
    name: str
    board_image: str
    grid: Dict[str, object]     # Grid configuration (rows, cols, tile_width, etc.)
    dice: Dict[str, object]     # Dice configuration (count, faces)
    rules: Optional[Dict[str, object]] = None
    tokens: Optional[Dict[str, object]] = None


__all__ = [
    "GamePlayer",
    "GameState",
    "GameConfig",
]

