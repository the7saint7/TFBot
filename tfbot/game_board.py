"""Board rendering for game board framework."""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import discord

from .game_models import GameConfig, GameState

logger = logging.getLogger("tfbot.game_board")


def parse_alphanumeric_coordinate(coord: str) -> Optional[Tuple[int, int]]:
    """
    Parse alphanumeric coordinate (e.g., "A1", "B5") to (column_index, row_index).
    
    Returns (column_index, row_index) where both are 1-indexed.
    A1 = (1, 1) = bottom left
    Returns None if invalid format.
    """
    coord = coord.strip().upper()
    if not coord:
        return None
    
    # Extract letter part (columns) and number part (rows)
    match = re.match(r"^([A-Z]+)(\d+)$", coord)
    if not match:
        return None
    
    letters = match.group(1)
    numbers = match.group(2)
    
    # Convert letters to column number (A=1, B=2, ..., Z=26, AA=27, etc.)
    column = 0
    for char in letters:
        column = column * 26 + (ord(char) - ord('A') + 1)
    
    # Convert number string to row number (1-indexed)
    row = int(numbers)
    
    return (column, row)


def alphanumeric_to_pixel(
    coord: str,
    game_config: GameConfig,
    board_width: int,
    board_height: int,
) -> Optional[Tuple[int, int]]:
    """
    Convert alphanumeric coordinate to pixel position.
    
    Args:
        coord: Alphanumeric coordinate (e.g., "A1")
        game_config: Game configuration with grid settings
        board_width: Width of board image in pixels
        board_height: Height of board image in pixels
    
    Returns:
        (pixel_x, pixel_y) tuple or None if invalid
    """
    parsed = parse_alphanumeric_coordinate(coord)
    if not parsed:
        return None
    
    column_index_1, row_index_1 = parsed
    
    grid_config = game_config.grid
    rows = int(grid_config.get("rows", 10))
    cols = int(grid_config.get("cols", 10))
    tile_width = int(grid_config.get("tile_width", 60))
    tile_height = int(grid_config.get("tile_height", 60))
    start_x = int(grid_config.get("start_x", 50))
    start_y = int(grid_config.get("start_y", 50))
    
    # Validate bounds
    if column_index_1 < 1 or column_index_1 > cols:
        return None
    if row_index_1 < 1 or row_index_1 > rows:
        return None
    
    # Convert to 0-indexed for pixel calculation
    column_index_0 = column_index_1 - 1
    row_index_0 = row_index_1 - 1
    
    # Calculate pixel position
    # Column: left to right
    pixel_x = start_x + (column_index_0 * tile_width) + (tile_width // 2)
    
    # Row: bottom to top (invert because row 1 is bottom)
    pixel_y = start_y + ((rows - row_index_0 - 1) * tile_height) + (tile_height // 2)
    
    return (pixel_x, pixel_y)


def render_game_board(
    game_state: GameState,
    game_config: GameConfig,
    assets_dir: Path,
) -> Optional[discord.File]:
    """
    Render game board image with token markers overlaid.
    
    Returns discord.File with board image, or None on error.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        logger.warning("PIL not available, cannot render board")
        return None
    
    # Load board image
    board_path = assets_dir / "boards" / game_config.board_image
    if not board_path.exists():
        logger.warning("Board image not found: %s", board_path)
        return None
    
    try:
        with Image.open(board_path) as board_img:
            board = board_img.convert("RGBA").copy()
    except Exception as exc:
        logger.warning("Failed to load board image %s: %s", board_path, exc)
        return None
    
    # Draw token markers
    draw = ImageDraw.Draw(board)
    board_width, board_height = board.size
    
    # Color palette for players
    colors = [
        (255, 0, 0),    # Red
        (0, 0, 255),    # Blue
        (0, 255,0),     # Green
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 165, 0),  # Orange
        (128, 0, 128),  # Purple
    ]
    
    player_list = list(game_state.players.values())
    for idx, player in enumerate(player_list):
        if not player.grid_position:
            continue
        
        # Convert alphanumeric to pixel coordinates
        pixel_pos = alphanumeric_to_pixel(
            player.grid_position,
            game_config,
            board_width,
            board_height,
        )
        
        if not pixel_pos:
            continue
        
        pixel_x, pixel_y = pixel_pos
        color = colors[idx % len(colors)]
        radius = 15  # Token marker radius
        
        # Draw filled circle for token marker
        bbox = (
            pixel_x - radius,
            pixel_y - radius,
            pixel_x + radius,
            pixel_y + radius,
        )
        draw.ellipse(bbox, fill=color, outline=(0, 0, 0), width=2)
    
    # Convert to bytes
    img_bytes = io.BytesIO()
    board.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    
    return discord.File(img_bytes, filename="game_board.png")


def validate_coordinate(coord: str, game_config: GameConfig) -> bool:
    """Validate if a coordinate is within bounds and not blocked."""
    parsed = parse_alphanumeric_coordinate(coord)
    if not parsed:
        return False
    
    column_index_1, row_index_1 = parsed
    
    grid_config = game_config.grid
    rows = int(grid_config.get("rows", 10))
    cols = int(grid_config.get("cols", 10))
    
    # Check bounds
    if column_index_1 < 1 or column_index_1 > cols:
        return False
    if row_index_1 < 1 or row_index_1 > rows:
        return False
    
    # Check blocked cells
    blocked = grid_config.get("blocked_cells", [])
    if isinstance(blocked, list):
        # Convert blocked cells to alphanumeric format if they're numbers
        # or check directly if they're already strings
        blocked_strs = {str(cell).strip().upper() for cell in blocked}
        if coord.strip().upper() in blocked_strs:
            return False
    
    return True


__all__ = [
    "parse_alphanumeric_coordinate",
    "alphanumeric_to_pixel",
    "render_game_board",
    "validate_coordinate",
]

