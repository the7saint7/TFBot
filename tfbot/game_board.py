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


def _resolve_face_cache_path(character_name: str) -> Optional[Path]:
    """
    Resolve face cache path for a character using the SAME method as VN bot.
    
    Uses _find_character_face_path from bot.py which:
    - Tries both normalized and lowercased folder names
    - Checks for face.png in variant directories
    - Falls back to any .png in variant directories
    - Falls back to .png files directly in folder
    """
    if not character_name:
        logger.debug("_resolve_face_cache_path: No character name provided")
        return None
    
    # Import here to avoid circular dependency
    import sys
    bot_module = sys.modules.get('bot') or sys.modules.get('__main__')
    if not bot_module:
        logger.debug("_resolve_face_cache_path: Bot module not found")
        return None
    
    # Try to use VN bot's face lookup function directly
    _find_character_face_path = getattr(bot_module, '_find_character_face_path', None)
    if _find_character_face_path:
        # Get character object to find folder name
        CHARACTER_BY_NAME = getattr(bot_module, 'CHARACTER_BY_NAME', None)
        if CHARACTER_BY_NAME:
            # Try exact match first (case-insensitive)
            character = CHARACTER_BY_NAME.get(character_name.strip().lower())
            # If not found, try case-insensitive search through all characters
            if not character:
                for name, char in CHARACTER_BY_NAME.items():
                    if name.lower() == character_name.strip().lower() or char.name.lower() == character_name.strip().lower():
                        character = char
                        logger.debug("_resolve_face_cache_path: Found character '%s' via case-insensitive search", character_name)
                        break
            
            if character and character.folder:
                # Use VN bot's function with folder name
                face_path = _find_character_face_path(character.folder)
                if face_path:
                    logger.info("_resolve_face_cache_path: Found face at %s (using VN bot method)", face_path)
                    return face_path
                else:
                    logger.debug("_resolve_face_cache_path: VN bot method returned None for folder '%s'", character.folder)
    
    # Fallback to manual lookup if VN bot function not available
    CHARACTER_FACES_ROOT = getattr(bot_module, 'CHARACTER_FACES_ROOT', None)
    if not CHARACTER_FACES_ROOT or not CHARACTER_FACES_ROOT.exists():
        logger.warning("_resolve_face_cache_path: CHARACTER_FACES_ROOT not available: %s", CHARACTER_FACES_ROOT)
        return None
    
    CHARACTER_BY_NAME = getattr(bot_module, 'CHARACTER_BY_NAME', None)
    if not CHARACTER_BY_NAME:
        logger.debug("_resolve_face_cache_path: CHARACTER_BY_NAME not available")
        return None
    
    # Try exact match first (case-insensitive)
    character = CHARACTER_BY_NAME.get(character_name.strip().lower())
    # If not found, try case-insensitive search through all characters
    if not character:
        for name, char in CHARACTER_BY_NAME.items():
            if name.lower() == character_name.strip().lower() or char.name.lower() == character_name.strip().lower():
                character = char
                logger.debug("_resolve_face_cache_path: Found character '%s' via case-insensitive search", character_name)
                break
    
    if not character:
        logger.warning("_resolve_face_cache_path: Character '%s' not found in CHARACTER_BY_NAME", character_name)
        return None
    
    # Use character's folder name (this matches the directory structure in characters_repo/faces/)
    folder_name = character.folder if character.folder else character_name.strip().lower()
    normalized = folder_name.strip()
    candidate_names = [normalized]
    lowered = normalized.lower()
    if lowered != normalized:
        candidate_names.append(lowered)
    
    # Replicate VN bot's lookup logic
    checked: set[Path] = set()
    for candidate in candidate_names:
        folder_dir = (CHARACTER_FACES_ROOT / candidate).resolve()
        if folder_dir in checked or not folder_dir.exists():
            continue
        checked.add(folder_dir)
        
        # Check variant directories first
        variant_dirs = sorted(
            (entry for entry in folder_dir.iterdir() if entry.is_dir()),
            key=lambda path: path.name.lower(),
        )
        for variant_dir in variant_dirs:
            face_candidate = variant_dir / "face.png"
            if face_candidate.exists():
                logger.info("_resolve_face_cache_path: Found face at %s", face_candidate)
                return face_candidate
            png_candidates = sorted(variant_dir.glob("*.png"))
            if png_candidates:
                logger.info("_resolve_face_cache_path: Found face at %s (fallback)", png_candidates[0])
                return png_candidates[0]
        
        # Fallback: check for .png files directly in folder
        direct_pngs = sorted(folder_dir.glob("*.png"))
        if direct_pngs:
            logger.info("_resolve_face_cache_path: Found face at %s (direct)", direct_pngs[0])
            return direct_pngs[0]
    
    logger.warning("_resolve_face_cache_path: No face found for character '%s' (folder: '%s')", character_name, folder_name)
    return None


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
    
    Grid system (per plan):
    - A1 = bottom left corner (1-indexed)
    - Letters (A, B, C...) = columns (left to right), A = column 1
    - Numbers (1, 2, 3...) = rows (bottom to top), 1 = bottom row
    - Both letters and numbers are 1-indexed
    
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
        logger.warning("Column index %d out of bounds (1-%d) for coord %s", column_index_1, cols, coord)
        return None
    if row_index_1 < 1 or row_index_1 > rows:
        logger.warning("Row index %d out of bounds (1-%d) for coord %s", row_index_1, rows, coord)
        return None
    
    # Calculate pixel positions with straight-through columns (A=leftmost) and rows counted
    # from bottom while the image rows count from the top.
    image_col = column_index_1 - 1
    image_row = rows - row_index_1
    
    # Calculate pixel position from image coordinates
    # Image coordinates: row 0 = top, row 9 = bottom; col 0 = left, col 9 = right
    pixel_x = start_x + (image_col * tile_width) + (tile_width // 2)
    pixel_y = start_y + (image_row * tile_height) + (tile_height // 2)
    
    # Debug: Verify A1 calculation
    if coord == "A1":
        expected_x = start_x + (tile_width // 2)  # First column, centered
        expected_y = start_y + ((rows - 1) * tile_height) + (tile_height // 2)  # Bottom row, centered
        logger.info("A1 calculation: pixel_x=%d (expected ~%d), pixel_y=%d (expected ~%d), start_x=%d, start_y=%d, rows=%d", 
                   pixel_x, expected_x, pixel_y, expected_y, start_x, start_y, rows)
    
    logger.debug("Grid coord %s (col=%d, row=%d) -> pixel (%d, %d)", coord, column_index_1, row_index_1, pixel_x, pixel_y)
    
    # Enhanced debug logging for problematic tiles
    if coord in ["A1", "A6", "J10"] or "60" in str(coord):
        logger.info("COORD DEBUG: %s -> col=%d, row=%d -> pixel (%d, %d) | start_x=%d, start_y=%d, tile_w=%d, tile_h=%d, rows=%d", 
                   coord, column_index_1, row_index_1, pixel_x, pixel_y, start_x, start_y, tile_width, tile_height, rows)
    
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
            actual_width, actual_height = board.size
            logger.info("Loaded board image: %s (%dx%d)", board_path, actual_width, actual_height)
    except Exception as exc:
        logger.warning("Failed to load board image %s: %s", board_path, exc)
        return None
    
    # Draw token markers
    draw = ImageDraw.Draw(board)
    board_width, board_height = board.size
    
    # Get tile dimensions from config (needed for debug layer)
    grid_config = game_config.grid
    tile_width = int(grid_config.get("tile_width", 60))
    tile_height = int(grid_config.get("tile_height", 60))
    rows = int(grid_config.get("rows", 10))
    cols = int(grid_config.get("cols", 10))
    
    # Draw debug layer if enabled (after board image, before tokens)
    if game_state.debug_mode:
        try:
            from PIL import ImageFont
        except ImportError:
            ImageFont = None
        
        # Try to load a font, fall back to default if not available
        try:
            # Try to use a default font
            font = ImageFont.truetype("arial.ttf", 12) if ImageFont else None
        except (OSError, IOError):
            try:
                # Try alternative font paths
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12) if ImageFont else None
            except (OSError, IOError):
                font = ImageFont.load_default() if ImageFont else None
        
        # Draw debug layer: red outlines and coordinate labels for all tiles
        for row in range(1, rows + 1):  # 1-indexed rows
            for col in range(1, cols + 1):  # 1-indexed columns
                # Convert to alphanumeric coordinate
                col_letter = chr(ord('A') + col - 1)
                coord = f"{col_letter}{row}"
                
                # Get pixel position for this coordinate
                pixel_pos = alphanumeric_to_pixel(
                    coord,
                    game_config,
                    board_width,
                    board_height,
                )
                
                if not pixel_pos:
                    continue
                
                pixel_x, pixel_y = pixel_pos
                
                # Draw red rectangle outline for the tile
                # Calculate tile bounds (centered on pixel position)
                half_width = tile_width // 2
                half_height = tile_height // 2
                rect_left = pixel_x - half_width
                rect_top = pixel_y - half_height
                rect_right = pixel_x + half_width
                rect_bottom = pixel_y + half_height
                
                # Draw white background with red outline (makes debug text legible)
                draw.rectangle(
                    [rect_left, rect_top, rect_right, rect_bottom],
                    fill=(255, 255, 255, 255),
                    outline=(255, 0, 0),  # Red
                    width=2
                )
                
                # Draw coordinate label text
                label_text = coord
                if font:
                    # Get text bounding box to center it
                    try:
                        bbox = draw.textbbox((0, 0), label_text, font=font)
                        text_width = bbox[2] - bbox[0]
                        text_height = bbox[3] - bbox[1]
                    except AttributeError:
                        # Fallback for older PIL versions
                        text_width, text_height = draw.textsize(label_text, font=font)
                else:
                    # Rough estimate if no font
                    text_width = len(label_text) * 6
                    text_height = 12
                
                # Center text in tile
                text_x = pixel_x - (text_width // 2)
                text_y = pixel_y - (text_height // 2)
                
                # Draw text with red color and white outline for visibility
                # Draw outline (white) by drawing text in multiple positions
                outline_color = (255, 255, 255)  # White
                text_color = (255, 0, 0)  # Red
                for adj_x in [-1, 0, 1]:
                    for adj_y in [-1, 0, 1]:
                        if adj_x != 0 or adj_y != 0:
                            draw.text(
                                (text_x + adj_x, text_y + adj_y),
                                label_text,
                                fill=outline_color,
                                font=font
                            )
                # Draw main text
                draw.text(
                    (text_x, text_y),
                    label_text,
                    fill=text_color,
                    font=font
                )
        
        logger.info("Debug layer drawn: showing all tile coordinates")
    
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
    
    # Group players by grid position to handle multiple tokens per tile
    players_by_position: Dict[str, list[GamePlayer]] = {}
    for player in game_state.players.values():
        if not player.grid_position:
            continue
        if player.grid_position not in players_by_position:
            players_by_position[player.grid_position] = []
        players_by_position[player.grid_position].append(player)
    
    # Tile dimensions already retrieved above for debug layer
    
    # Render tokens grouped by position
    for grid_pos, players_at_pos in players_by_position.items():
        # Convert alphanumeric to pixel coordinates (center of tile)
        logger.debug("Converting grid position %s to pixels", grid_pos)
        pixel_pos = alphanumeric_to_pixel(
            grid_pos,
            game_config,
            board_width,
            board_height,
        )
        
        if not pixel_pos:
            logger.warning("Failed to convert grid position %s to pixels", grid_pos)
            continue
        
        logger.debug("Grid position %s -> pixel (%d, %d)", grid_pos, pixel_pos[0], pixel_pos[1])
        
        pixel_x, pixel_y = pixel_pos
        num_players = len(players_at_pos)
        
        # Calculate token size based on number of players on this tile
        # Max 4 tokens per tile, scale down if more
        if num_players == 1:
            token_size = min(tile_width, tile_height) - 10  # Single token: almost full tile
        elif num_players == 2:
            token_size = min(tile_width, tile_height) // 2 - 5  # Two tokens: half tile each
        elif num_players == 3:
            token_size = min(tile_width, tile_height) // 2 - 5  # Three tokens: arrange in triangle
        else:
            token_size = min(tile_width, tile_height) // 2 - 5  # Four or more: 2x2 grid
        
        # Ensure minimum size
        token_size = max(20, token_size)
        
        # Calculate positions for multiple tokens within the tile
        token_positions = []
        if num_players == 1:
            token_positions = [(pixel_x, pixel_y)]  # Center
        elif num_players == 2:
            # Side by side
            offset = tile_width // 4
            token_positions = [
                (pixel_x - offset, pixel_y),
                (pixel_x + offset, pixel_y),
            ]
        elif num_players == 3:
            # Triangle formation
            offset_x = tile_width // 3
            offset_y = tile_height // 3
            token_positions = [
                (pixel_x, pixel_y - offset_y),  # Top
                (pixel_x - offset_x, pixel_y + offset_y),  # Bottom left
                (pixel_x + offset_x, pixel_y + offset_y),  # Bottom right
            ]
        else:  # 4 or more
            # 2x2 grid (or more, overlapping)
            offset_x = tile_width // 3
            offset_y = tile_height // 3
            positions_2x2 = [
                (pixel_x - offset_x, pixel_y - offset_y),  # Top left
                (pixel_x + offset_x, pixel_y - offset_y),  # Top right
                (pixel_x - offset_x, pixel_y + offset_y),  # Bottom left
                (pixel_x + offset_x, pixel_y + offset_y),  # Bottom right
            ]
            # Use first 4 positions, repeat if more than 4 players
            token_positions = [positions_2x2[i % 4] for i in range(num_players)]
        
        # Render each token at this position
        for idx, player in enumerate(players_at_pos):
            if idx >= len(token_positions):
                break
            
            token_x, token_y = token_positions[idx]
            color = colors[player.user_id % len(colors)]
            
            # Get player number from turn_order (if available)
            player_number = None
            if hasattr(game_state, '_pack_data') and game_state._pack_data:
                turn_order = game_state._pack_data.get('turn_order', [])
                if player.user_id in turn_order:
                    player_number = turn_order.index(player.user_id) + 1  # 1-indexed (P1, P2, etc.)
            
            # Try to load face image from face sync cache
            face_path = None
            if player.character_name:
                # Use the game-assigned character name directly (from face sync system)
                character_name_for_face = player.character_name.strip()
                logger.info("Looking up face for game character: '%s' (player: %s, position: %s)", character_name_for_face, player.user_id, grid_pos)
                face_path = _resolve_face_cache_path(character_name_for_face)
                if face_path and face_path.exists():
                    logger.info("Found face at: %s", face_path)
                else:
                    logger.warning("Face NOT found for game character: '%s' (player: %s). Checked path: %s", character_name_for_face, player.user_id, face_path)
            
            if face_path and face_path.exists():
                try:
                    # Load and resize face image from face sync cache
                    with Image.open(face_path) as face_img:
                        face = face_img.convert("RGBA")
                        # Resize to calculated token size
                        try:
                            # Try new Pillow 10+ API first
                            face = face.resize((token_size, token_size), Image.Resampling.LANCZOS)
                        except AttributeError:
                            # Fall back to old PIL API
                            face = face.resize((token_size, token_size), Image.LANCZOS)
                        
                        # Create circular mask for face
                        mask = Image.new("L", (token_size, token_size), 0)
                        mask_draw = ImageDraw.Draw(mask)
                        mask_draw.ellipse((0, 0, token_size, token_size), fill=255)
                        
                        # Apply circular mask
                        face.putalpha(mask)
                        
                        # Calculate position (center face on calculated position)
                        face_x = token_x - (token_size // 2)
                        face_y = token_y - (token_size // 2)
                        
                        # Ensure token stays within tile bounds
                        face_x = max(face_x, token_x - tile_width // 2 + 5)
                        face_y = max(face_y, token_y - tile_height // 2 + 5)
                        face_x = min(face_x, token_x + tile_width // 2 - token_size - 5)
                        face_y = min(face_y, token_y + tile_height // 2 - token_size - 5)
                        
                        # Paste face onto board
                        board.paste(face, (int(face_x), int(face_y)), face)
                        
                        # Draw colored border around face (2 pixels, thinner for smaller tokens)
                        border_width = max(2, token_size // 30)
                        border_bbox = (
                            int(face_x - border_width),
                            int(face_y - border_width),
                            int(face_x + token_size + border_width),
                            int(face_y + token_size + border_width),
                        )
                        draw.ellipse(border_bbox, outline=color, width=border_width)
                        
                        # Draw player number label (P1, P2, etc.) on token
                        if player_number is not None:
                            try:
                                from PIL import ImageFont
                                # Try to load a font for the label
                                try:
                                    label_font = ImageFont.truetype("arial.ttf", max(10, token_size // 6)) if ImageFont else None
                                except (OSError, IOError):
                                    try:
                                        label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", max(10, token_size // 6)) if ImageFont else None
                                    except (OSError, IOError):
                                        label_font = ImageFont.load_default() if ImageFont else None
                                
                                label_text = f"P{player_number}"
                                # Position label at bottom-right of token
                                label_x = int(face_x + token_size - token_size // 3)
                                label_y = int(face_y + token_size - token_size // 3)
                                
                                # Draw white background for label
                                if label_font:
                                    try:
                                        bbox = draw.textbbox((0, 0), label_text, font=label_font)
                                        text_width = bbox[2] - bbox[0]
                                        text_height = bbox[3] - bbox[1]
                                    except AttributeError:
                                        text_width, text_height = draw.textsize(label_text, font=label_font)
                                else:
                                    text_width = len(label_text) * 6
                                    text_height = 10
                                
                                # Draw background rectangle
                                bg_padding = 2
                                draw.rectangle(
                                    [label_x - bg_padding, label_y - bg_padding, 
                                     label_x + text_width + bg_padding, label_y + text_height + bg_padding],
                                    fill=(255, 255, 255, 200),  # Semi-transparent white
                                    outline=(0, 0, 0),  # Black outline
                                    width=1
                                )
                                
                                # Draw label text
                                draw.text(
                                    (label_x, label_y),
                                    label_text,
                                    fill=(0, 0, 0),  # Black text
                                    font=label_font
                                )
                            except Exception as exc:
                                logger.warning("Failed to draw player number label: %s", exc)
                except Exception as exc:
                    logger.warning("Failed to load face token for %s: %s", player.character_name, exc)
                    # Fall through to colored circle
                    face_path = None
            
            # Fall back to colored circle if no face found or face loading failed
            if not face_path:
                radius = max(8, token_size // 4)  # Scaled radius
                bbox = (
                    int(token_x - radius),
                    int(token_y - radius),
                    int(token_x + radius),
                    int(token_y + radius),
                )
                draw.ellipse(bbox, fill=color, outline=(0, 0, 0), width=2)
                
                # Draw player number label on colored circle token too
                if player_number is not None:
                    try:
                        from PIL import ImageFont
                        try:
                            label_font = ImageFont.truetype("arial.ttf", max(10, radius)) if ImageFont else None
                        except (OSError, IOError):
                            try:
                                label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", max(10, radius)) if ImageFont else None
                            except (OSError, IOError):
                                label_font = ImageFont.load_default() if ImageFont else None
                        
                        label_text = f"P{player_number}"
                        label_x = int(token_x + radius - radius // 2)
                        label_y = int(token_y + radius - radius // 2)
                        
                        if label_font:
                            try:
                                bbox = draw.textbbox((0, 0), label_text, font=label_font)
                                text_width = bbox[2] - bbox[0]
                                text_height = bbox[3] - bbox[1]
                            except AttributeError:
                                text_width, text_height = draw.textsize(label_text, font=label_font)
                        else:
                            text_width = len(label_text) * 6
                            text_height = 10
                        
                        bg_padding = 2
                        draw.rectangle(
                            [label_x - bg_padding, label_y - bg_padding,
                             label_x + text_width + bg_padding, label_y + text_height + bg_padding],
                            fill=(255, 255, 255, 200),
                            outline=(0, 0, 0),
                            width=1
                        )
                        
                        draw.text(
                            (label_x, label_y),
                            label_text,
                            fill=(0, 0, 0),
                            font=label_font
                        )
                    except Exception as exc:
                        logger.warning("Failed to draw player number label on circle: %s", exc)
    
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

