"""
Snakes and Ladders Game Pack

All game-specific rules and logic for Snakes & Ladders.
This pack is self-contained and doesn't modify core bot code.
"""

from __future__ import annotations

import random
from typing import Dict, Optional, Tuple, List

import sys
from pathlib import Path

# Add TFBot root to path for imports
# Pack file is at: games/packs/snakes_ladders.py
# Need to import from: tfbot/game_models.py
_pack_file = Path(__file__).resolve()
_tfbot_root = _pack_file.parent.parent.parent
if str(_tfbot_root) not in sys.path:
    sys.path.insert(0, str(_tfbot_root))

from tfbot.game_models import GameConfig, GamePlayer, GameState
from tfbot.game_board import parse_alphanumeric_coordinate


# Game-specific data stored in game state
def get_game_data(game_state: GameState) -> Dict:
    """Get or initialize game-specific data for this game."""
    if not hasattr(game_state, '_pack_data'):
        game_state._pack_data = {
            'tile_numbers': {},  # user_id -> tile_number (1-100)
            'transformation_counts': {},  # user_id -> count
            'original_characters': {},  # user_id -> original character
            'real_body_characters': {},  # user_id -> real body character
            'mind_changed': {},  # user_id -> bool
            'turn_order': [],  # List of user_ids
            'current_turn_index': 0,
            'players_rolled_this_turn': [],  # List of user_ids who have rolled this turn
        }
    return game_state._pack_data


def tile_number_to_alphanumeric(tile_num: int, game_config: GameConfig) -> Optional[str]:
    """
    Convert tile number (1-100) to alphanumeric coordinate (A1-J10).
    
    Snakes & Ladders numbering:
    - Row 1 (bottom): 1-10 (A1-J1, left to right)
    - Row 2: 11-20 (J2-A2, right to left)
    - Row 3: 21-30 (A3-J3, left to right)
    - Alternates direction each row
    """
    if tile_num < 1:
        return None
    
    grid_config = game_config.grid
    rows = int(grid_config.get("rows", 10))
    cols = int(grid_config.get("cols", 10))
    max_tile = rows * cols
    
    if tile_num > max_tile:
        return None
    
    # Calculate row (1-indexed, from bottom)
    row = ((tile_num - 1) // cols) + 1
    
    # Calculate column position within row
    position_in_row = ((tile_num - 1) % cols) + 1
    
    # Odd rows (1, 3, 5, ...) go left to right (A, B, C, ...)
    # Even rows (2, 4, 6, ...) go right to left (J, I, H, ...)
    if row % 2 == 1:
        # Left to right: A=1, B=2, ...
        col_letter = chr(ord('A') + position_in_row - 1)
    else:
        # Right to left: J=1, I=2, ...
        col_letter = chr(ord('A') + cols - position_in_row)
    
    result = f"{col_letter}{row}"
    
    # Debug logging for key tiles
    if tile_num in [1, 10, 11, 20, 51, 60, 100]:
        import logging
        logger = logging.getLogger("tfbot.games.snakes_ladders")
        logger.info("TILE_TO_COORD DEBUG: tile %d -> row=%d, pos_in_row=%d, col_letter=%s -> %s", 
                   tile_num, row, position_in_row, col_letter, result)
    
    return result


def alphanumeric_to_tile_number(coord: str, game_config: GameConfig) -> Optional[int]:
    """Convert alphanumeric coordinate (A1-J10) to tile number (1-100)."""
    parsed = parse_alphanumeric_coordinate(coord)
    if not parsed:
        return None
    
    column_index_1, row_index_1 = parsed
    
    grid_config = game_config.grid
    cols = int(grid_config.get("cols", 10))
    
    # Calculate position within row
    if row_index_1 % 2 == 1:
        # Odd rows: left to right
        position_in_row = column_index_1
    else:
        # Even rows: right to left
        position_in_row = cols - column_index_1 + 1
    
    # Calculate tile number
    tile_num = ((row_index_1 - 1) * cols) + position_in_row
    
    return tile_num


def on_player_added(game_state: GameState, player: GamePlayer, game_config: GameConfig) -> None:
    """Called when a player is added to the game."""
    data = get_game_data(game_state)
    
    # Initialize tile number and grid position
    rules = game_config.rules or {}
    starting_tile = int(rules.get("starting_tile", 1))
    starting_position = rules.get("starting_position", "A1")
    data['tile_numbers'][player.user_id] = starting_tile
    player.grid_position = str(starting_position)  # Set grid position from config
    
    # Initialize other fields
    data['transformation_counts'][player.user_id] = 0
    data['original_characters'][player.user_id] = None
    data['real_body_characters'][player.user_id] = None
    data['mind_changed'][player.user_id] = False
    
    # Add to turn order if not present
    if player.user_id not in data['turn_order']:
        data['turn_order'].append(player.user_id)
        if len(data['turn_order']) == 1:
            data['current_turn_index'] = 0


def on_character_assigned(game_state: GameState, player: GamePlayer, character_name: str) -> None:
    """Called when a character is assigned to a player."""
    data = get_game_data(game_state)
    
    # Store original character if not set
    if data['original_characters'][player.user_id] is None:
        data['original_characters'][player.user_id] = character_name


def on_dice_rolled(
    game_state: GameState,
    player: GamePlayer,
    dice_result: int,
    game_config: GameConfig,
) -> Tuple[Optional[str], bool, Optional[str], bool]:
    """
    Called when a player rolls dice.
    
    Returns: (message_to_send, should_auto_move, transformation_character_name, is_turn_complete)
    - message_to_send: Message about the roll and movement
    - should_auto_move: Whether to automatically update the board
    - transformation_character_name: Character name to transform to (None if no transformation)
    - is_turn_complete: Whether all players have rolled this turn
    """
    data = get_game_data(game_state)
    
    # Check if player already rolled this turn
    if player.user_id in data['players_rolled_this_turn']:
        return (f"You've already rolled this turn! Wait for the turn summary.", False, None, False)
    
    # Check if it's player's turn to roll (next player in turn_order who hasn't rolled)
    if data['turn_order']:
        # Find next player who hasn't rolled
        next_player_id = None
        for user_id in data['turn_order']:
            if user_id not in data['players_rolled_this_turn']:
                next_player_id = user_id
                break
        
        if next_player_id is None:
            # All players have rolled, turn should be complete
            return ("All players have rolled! Turn summary should be shown.", False, None, True)
        
        if player.user_id != next_player_id:
            # Not this player's turn yet
            return (f"It's not your turn yet! Waiting for others to roll.", False, None, False)
    
    # Get current tile
    current_tile = data['tile_numbers'].get(player.user_id, 1)
    new_tile = current_tile + dice_result
    
    # Check win condition
    rules = game_config.rules or {}
    win_tile = int(rules.get("win_tile", 100))
    if new_tile >= win_tile:
        new_tile = win_tile
    
    # Move player
    data['tile_numbers'][player.user_id] = new_tile
    
    # Update grid position
    new_pos = tile_number_to_alphanumeric(new_tile, game_config)
    if new_pos:
        player.grid_position = new_pos
    
    # Check for snakes and ladders
    # Convert string keys to ints for comparison
    snakes_raw = game_config.rules.get("snakes", {}) if game_config.rules else {}
    ladders_raw = game_config.rules.get("ladders", {}) if game_config.rules else {}
    snakes = {int(k): int(v) for k, v in snakes_raw.items()} if snakes_raw else {}
    ladders = {int(k): int(v) for k, v in ladders_raw.items()} if ladders_raw else {}
    
    message_parts = [f"Moved to tile {new_tile}"]
    final_tile = new_tile
    
    # Check snake
    if new_tile in snakes:
        tail_tile = snakes[new_tile]
        data['tile_numbers'][player.user_id] = tail_tile
        final_tile = tail_tile
        new_pos = tile_number_to_alphanumeric(tail_tile, game_config)
        if new_pos:
            player.grid_position = new_pos
        message_parts.append(f"🐍 Snake! Slid down to tile {tail_tile}")
    
    # Check ladder
    elif new_tile in ladders:
        top_tile = ladders[new_tile]
        data['tile_numbers'][player.user_id] = top_tile
        final_tile = top_tile
        new_pos = tile_number_to_alphanumeric(top_tile, game_config)
        if new_pos:
            player.grid_position = new_pos
        message_parts.append(f"🪜 Ladder! Climbed up to tile {top_tile}")
    
    # Apply tile transformation (tiles 2-99) - AFTER snake/ladder movement
    transformation_char = None
    if 2 <= final_tile <= 99:
        transform_msg, new_char = apply_tile_transformation(game_state, player, final_tile, game_config)
        if transform_msg:
            message_parts.append(transform_msg)
        if new_char:
            transformation_char = new_char
            # Update player's character name
            player.character_name = new_char
    
    # Mark player as having rolled this turn
    if player.user_id not in data['players_rolled_this_turn']:
        data['players_rolled_this_turn'].append(player.user_id)
    
    # Check if turn is complete (all players have rolled)
    is_turn_complete = False
    if data['turn_order']:
        is_turn_complete = len(data['players_rolled_this_turn']) >= len(data['turn_order'])
    
    # Check win condition
    win_msg = check_win_condition(game_state, game_config)
    if win_msg:
        message_parts.append(win_msg)
    
    return ("\n".join(message_parts), True, transformation_char, is_turn_complete)


def apply_tile_transformation(
    game_state: GameState,
    player: GamePlayer,
    tile_number: int,
    game_config: GameConfig,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Apply transformation based on tile color/effect.
    
    Returns: (message, new_character_name)
    - message: Message to display about the transformation
    - new_character_name: Character name to transform to (None if no transformation)
    """
    data = get_game_data(game_state)
    
    # Get tile color mapping from config
    if not game_config or not game_config.rules:
        return (None, None)
    
    tile_colors_map = game_config.rules.get("tile_colors_map", {})
    tile_color = tile_colors_map.get(str(tile_number))
    
    if not tile_color:
        return (None, None)
    
    # Map image colors to config colors
    # Image has: yellow, blue, purple, red, green (repeating pattern)
    # Config expects: yellow, light_blue/dark_blue, purple, red, green, orange, pink
    # Map "blue" from image to "dark_blue" (age_change)
    # Note: Some blue tiles might be light_blue (real_body) but we'll use dark_blue as default
    # GM can manually adjust if needed
    color_mapping = {
        "yellow": "yellow",      # random_transform
        "blue": "dark_blue",     # age_change (image blue → dark_blue)
        "purple": "purple",      # revert_body
        "red": "red",            # transform_other
        "green": "green"         # body_swap
    }
    
    # Convert image color to config color
    config_color = color_mapping.get(tile_color, tile_color)
    
    # Get effect mapping
    color_effects = game_config.rules.get("tile_colors", {})
    effect = color_effects.get(config_color)
    
    if not effect:
        return (None, None)
    
    # Import character data
    import sys
    bot_module = sys.modules.get('bot')
    if not bot_module:
        return (None, None)
    
    CHARACTER_BY_NAME = getattr(bot_module, 'CHARACTER_BY_NAME', None)
    if not CHARACTER_BY_NAME:
        return (None, None)
    
    available = list(CHARACTER_BY_NAME.keys())
    if not available:
        return (None, None)
    
    old_char = player.character_name
    
    # Check if mind change should be undone (any transformation undoes it)
    mind_was_changed = data['mind_changed'].get(player.user_id, False)
    mind_undo_msg = ""
    if mind_was_changed and effect in ("gender_swap", "random_transform", "age_change", "body_swap"):
        data['mind_changed'][player.user_id] = False
        mind_undo_msg = " (Mind change undone!)"
    
    # Apply effect
    if effect == "gender_swap":
        new_char = random.choice(available)
        message = f"🟠 Gender swap! {old_char} → {new_char}{mind_undo_msg}"
        data['transformation_counts'][player.user_id] += 1
        return (message, new_char)
    
    elif effect == "real_body":
        if player.character_name:
            data['real_body_characters'][player.user_id] = player.character_name
            message = f"🔵 Current body ({player.character_name}) is now your real body!"
            return (message, None)  # No transformation, just setting real body
    
    elif effect == "revert_body":
        real_body = data['real_body_characters'].get(player.user_id)
        if real_body:
            message = f"🟣 Reverted to real body: {real_body}"
            return (message, real_body)  # Transform to real body
        message = "🟣 No real body set - no change"
        return (message, None)
    
    elif effect == "body_swap":
        message = "🟢 Body swap! GM: use !swap command to choose target"
        return (message, None)  # No automatic transformation
    
    elif effect == "random_transform":
        new_char = random.choice(available)
        message = f"🟡 Random transformation! {old_char} → {new_char}{mind_undo_msg}"
        data['transformation_counts'][player.user_id] += 1
        return (message, new_char)
    
    elif effect == "mind_change":
        data['mind_changed'][player.user_id] = True
        message = "🩷 Mind changed! Can be undone if transformed before next turn"
        return (message, None)  # No transformation
    
    elif effect == "transform_other":
        message = "🔴 Transform other! GM: use !reroll @target to transform someone"
        return (message, None)  # No automatic transformation
    
    elif effect == "age_change":
        # Age change - transform to random character
        new_char = random.choice(available)
        message = f"🔵 Age change! {old_char} → {new_char}{mind_undo_msg}"
        data['transformation_counts'][player.user_id] += 1
        return (message, new_char)
    
    return (None, None)


def check_win_condition(game_state: GameState, game_config: GameConfig) -> Optional[str]:
    """Check if game win condition is met."""
    data = get_game_data(game_state)
    rules = game_config.rules or {}
    
    win_tile = int(rules.get("win_tile", 100))
    all_must_reach = bool(rules.get("game_ends_when_all_reach_end", False))
    
    if not all_must_reach:
        # First to reach wins
        for user_id, tile_num in data['tile_numbers'].items():
            if tile_num >= win_tile:
                return f"🎉 <@{user_id}> wins! First to reach tile {win_tile}!"
    else:
        # All must reach - check if all players are at end
        all_at_end = all(
            tile_num >= win_tile
            for tile_num in data['tile_numbers'].values()
        )
        
        if all_at_end:
            # Winner is player with lowest transformation count
            winner_id = min(
                data['transformation_counts'].items(),
                key=lambda x: x[1]
            )[0]
            
            count = data['transformation_counts'][winner_id]
            return (
                f"🎉 **GAME OVER!** 🎉\n"
                f"**Winner:** <@{winner_id}> (lowest transformations: {count})\n"
                f"**Reward:** Control all players' fates for 7 days in normal BunnyBot!"
            )
    
    return None


def get_player_tile_number(game_state: GameState, user_id: int) -> int:
    """Get player's current tile number."""
    data = get_game_data(game_state)
    return data['tile_numbers'].get(user_id, 1)


def get_transformation_count(game_state: GameState, user_id: int) -> int:
    """Get player's transformation count."""
    data = get_game_data(game_state)
    return data['transformation_counts'].get(user_id, 0)


def get_turn_summary(game_state: GameState, game_config: GameConfig) -> str:
    """
    Generate turn summary with leaderboard.
    
    Returns: Formatted message with leader and last place info.
    """
    data = get_game_data(game_state)
    
    # Get all players with their tile numbers
    player_positions = []
    for user_id, tile_num in data['tile_numbers'].items():
        player = game_state.players.get(user_id)
        if player:
            player_positions.append((user_id, tile_num, player))
    
    if not player_positions:
        return "No players in game."
    
    # Sort by tile number (highest = leader)
    player_positions.sort(key=lambda x: x[1], reverse=True)
    
    # Get leader and last place
    leader_id, leader_tile, leader_player = player_positions[0]
    last_id, last_tile, last_player = player_positions[-1]
    
    # Build summary
    summary_parts = ["**Turn Complete!**", "", "**Leaderboard:**"]
    
    # Leader
    leader_name = leader_player.character_name or f"Player {leader_id}"
    summary_parts.append(f"🥇 In Lead: {leader_name} (Tile {leader_tile})")
    
    # Last place (only if different from leader)
    if last_id != leader_id:
        last_name = last_player.character_name or f"Player {last_id}"
        summary_parts.append(f"🥉 In Last: {last_name} (Tile {last_tile})")
    
    summary_parts.append("")
    summary_parts.append("**Current Positions:**")
    for user_id, tile_num, player in player_positions:
        player_name = player.character_name or f"Player {user_id}"
        summary_parts.append(f"- {player_name}: Tile {tile_num}")
    
    return "\n".join(summary_parts)


def advance_turn(game_state: GameState) -> None:
    """Reset players_rolled_this_turn and advance to next turn."""
    data = get_game_data(game_state)
    
    # Reset players who have rolled
    data['players_rolled_this_turn'] = []
    
    # Advance turn index (this will be handled by the calling code if needed)
    # The turn index doesn't need to advance here since we're starting a new turn

