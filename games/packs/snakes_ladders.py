"""
Snakes and Ladders Game Pack

All game-specific rules and logic for Snakes & Ladders.
This pack is self-contained and doesn't modify core bot code.
"""

from __future__ import annotations

import random
from typing import Dict, Optional, Tuple, List
import logging

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

logger = logging.getLogger("tfbot.games.snakes_ladders")


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
            'turn_order': [],  # List of user_ids in order players were added
            'player_numbers': {},  # user_id -> player_number (1, 2, 3, etc.) based on add order
            'current_turn_index': 0,
            'players_rolled_this_turn': [],  # List of user_ids who have rolled this turn
            'winners': [],  # List of user_ids who have won (cannot roll dice anymore)
            'players_reached_end_this_turn': [],  # List of user_ids who reached the end tile this turn (for multi-winner detection)
            'goal_reached_turn': {},  # user_id -> turn_number when they reached the goal (for determining winners)
        }
    return game_state._pack_data


def tile_number_to_alphanumeric(tile_num: int, game_config: GameConfig) -> Optional[str]:
    """
    Convert tile number (1-100) to alphanumeric coordinate (A1-J10).
    
    Uses classic Snakes & Ladders zig-zag numbering:
    - Row 1 (bottom) runs left-to-right (A-J)
    - Row 2 runs right-to-left (J-A)
    - Alternates every row.
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
    
    # Zig-zag column calculation
    position_in_row = ((tile_num - 1) % cols) + 1
    if row % 2 == 1:
        column = position_in_row
    else:
        column = cols - position_in_row + 1
    col_letter = chr(ord('A') + column - 1)
    
    result = f"{col_letter}{row}"
    
    logger.debug(
        "tile_number_to_alphanumeric: tile=%s rows=%s cols=%s -> row=%s column=%s (%s)",
        tile_num,
        rows,
        cols,
        row,
        column,
        result,
    )
    
    return result


def alphanumeric_to_tile_number(coord: str, game_config: GameConfig) -> Optional[int]:
    """Convert alphanumeric coordinate (A1-J10) to tile number (1-100) using zig-zag layout."""
    parsed = parse_alphanumeric_coordinate(coord)
    if not parsed:
        return None
    
    column_index_1, row_index_1 = parsed
    
    grid_config = game_config.grid
    cols = int(grid_config.get("cols", 10))
    
    # Determine position within row based on zig-zag direction
    if row_index_1 % 2 == 1:
        position_in_row = column_index_1
    else:
        position_in_row = cols - column_index_1 + 1
    
    tile_num = ((row_index_1 - 1) * cols) + position_in_row
    
    logger.debug(
        "alphanumeric_to_tile_number: coord=%s -> row=%s column=%s (tile=%s)",
        coord,
        row_index_1,
        column_index_1,
        ((row_index_1 - 1) * cols) + position_in_row,
    )
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
    
    # Add to turn order if not present (maintain order players were added)
    if player.user_id not in data['turn_order']:
        data['turn_order'].append(player.user_id)
        # Assign player number based on order added (1-indexed)
        if 'player_numbers' not in data:
            data['player_numbers'] = {}
        data['player_numbers'][player.user_id] = len(data['turn_order'])
        if len(data['turn_order']) == 1:
            data['current_turn_index'] = 0


def on_character_assigned(game_state: GameState, player: GamePlayer, character_name: str) -> None:
    """Called when a character is assigned to a player."""
    data = get_game_data(game_state)
    
    # Ensure original_characters dict exists and has entry for this player
    if 'original_characters' not in data:
        data['original_characters'] = {}
    
    # Store original character if not set (use .get() to safely check)
    if data['original_characters'].get(player.user_id) is None:
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
    
    # Debug logging for turn order
    logger.debug("on_dice_rolled: player.user_id=%s, turn_order=%s, players_rolled_this_turn=%s", 
                player.user_id, data.get('turn_order', []), data.get('players_rolled_this_turn', []))
    
    # Get win tile for checks
    rules = game_config.rules or {}
    win_tile = int(rules.get("win_tile", 100))
    
    # CRITICAL: Check if player has already won - winners cannot roll dice
    if player.user_id in data.get('winners', []):
        return (f"🎉 You've already won! You cannot roll dice anymore. The game continues for other players.", False, None, False)
    
    # CRITICAL: Check if player is at the goal tile - they cannot roll (even if not in winners list yet)
    current_tile = data['tile_numbers'].get(player.user_id, 1)
    if current_tile >= win_tile:
        return (f"🎉 You've reached the goal (tile {win_tile})! You cannot roll dice anymore. The game continues for other players.", False, None, False)
    
    # Check if player already rolled this turn
    if player.user_id in data['players_rolled_this_turn']:
        return (f"You've already rolled this turn! Wait for the turn summary.", False, None, False)
    
    # Check if it's player's turn to roll (next player in turn_order who hasn't rolled AND isn't at goal)
    if data['turn_order']:
        # Find next player who hasn't rolled AND isn't at the goal tile
        next_player_id = None
        for user_id in data['turn_order']:
            # Skip players who have already rolled this turn
            if user_id in data['players_rolled_this_turn']:
                continue
            # Skip players who are at the goal tile (they cannot roll)
            player_tile = data['tile_numbers'].get(user_id, 1)
            if player_tile >= win_tile:
                logger.debug("Skipping player %s in turn order - at goal tile %d", user_id, player_tile)
                continue
            # Found next active player
            next_player_id = user_id
            break
        
        if next_player_id is None:
            # All active players have rolled, turn should be complete
            return ("All players have rolled! Turn summary should be shown.", False, None, True)
        
        if player.user_id != next_player_id:
            # Not this player's turn yet - provide helpful message
            next_player_num = data.get('player_numbers', {}).get(next_player_id, "?")
            current_player_num = data.get('player_numbers', {}).get(player.user_id, "?")
            logger.debug("Turn check: player %s (Player %s) tried to roll, but it's Player %s's turn (user_id=%s)", 
                        player.user_id, current_player_num, next_player_num, next_player_id)
            return (f"It's not your turn yet! Waiting for Player {next_player_num} to roll.", False, None, False)
    
    # Get current tile
    current_tile = data['tile_numbers'].get(player.user_id, 1)
    logger.debug(
        "on_dice_rolled: player=%s current_tile=%s dice_result=%s",
        player.user_id,
        current_tile,
        dice_result,
    )
    new_tile = current_tile + dice_result
    logger.debug("on_dice_rolled: tentative new_tile=%s", new_tile)
    
    # Check win condition
    rules = game_config.rules or {}
    win_tile = int(rules.get("win_tile", 100))
    if new_tile >= win_tile:
        new_tile = win_tile
        logger.debug("on_dice_rolled: capped to win_tile=%s", new_tile)
    
    # Move player
    data['tile_numbers'][player.user_id] = new_tile
    
    # Update grid position
    new_pos = tile_number_to_alphanumeric(new_tile, game_config)
    if new_pos:
        player.grid_position = new_pos
    logger.debug(
        "on_dice_rolled: updated grid position -> tile=%s coord=%s",
        new_tile,
        new_pos,
    )
    
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
        logger.debug(
            "on_dice_rolled: snake encountered head=%s tail=%s coord=%s",
            new_tile,
            tail_tile,
            new_pos,
        )
        message_parts.append(f"🐍 Snake! Slid down to tile {tail_tile}")
    
    # Check ladder
    elif new_tile in ladders:
        top_tile = ladders[new_tile]
        data['tile_numbers'][player.user_id] = top_tile
        final_tile = top_tile
        new_pos = tile_number_to_alphanumeric(top_tile, game_config)
        if new_pos:
            player.grid_position = new_pos
        logger.debug(
            "on_dice_rolled: ladder encountered base=%s top=%s coord=%s",
            new_tile,
            top_tile,
            new_pos,
        )
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
        logger.debug(
            "on_dice_rolled: transformation check tile=%s message=%s new_char=%s",
            final_tile,
            transform_msg,
            transformation_char,
        )
    
    # Mark player as having rolled this turn
    if player.user_id not in data['players_rolled_this_turn']:
        data['players_rolled_this_turn'].append(player.user_id)
    
    # Check if turn is complete (all ACTIVE players have rolled - skip players at goal)
    is_turn_complete = False
    if data['turn_order']:
        # Count only active players (those not at the goal tile)
        active_players = [
            uid for uid in data['turn_order']
            if data['tile_numbers'].get(uid, 1) < win_tile
        ]
        # Turn is complete when all active players have rolled
        active_players_rolled = [
            uid for uid in data['players_rolled_this_turn']
            if uid in active_players
        ]
        is_turn_complete = len(active_players_rolled) >= len(active_players) if active_players else True
    logger.debug(
        "on_dice_rolled: final_tile=%s coord=%s turn_complete=%s rolled=%s/%s",
        final_tile,
        player.grid_position,
        is_turn_complete,
        len(data['players_rolled_this_turn']),
        len(data['turn_order']) if data['turn_order'] else 0,
    )
    
    # Check win condition (returns message and game_ended flag)
    win_msg, game_ended = check_win_condition(game_state, game_config)
    if win_msg:
        message_parts.append(win_msg)
    
    # Store game_ended flag in pack data for later use
    data['game_ended'] = game_ended
    
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


def check_win_condition(game_state: GameState, game_config: GameConfig) -> Tuple[Optional[str], bool]:
    """
    Check if game win condition is met. Tracks when players reach the goal and determines winners.
    
    IMPORTANT: Turn order matters! Players roll in order (Player 1, then Player 2, etc.) within each turn.
    - If Player 1 and Player 2 both reach the goal during the same turn cycle (before turn_count increments),
      they BOTH win because they reached on the same turn number.
    - If Player 1 reaches on turn 10 and Player 2 reaches on turn 11, only Player 1 wins (reached first).
    
    Winners are determined by who reached the goal FIRST (lowest turn number).
    If multiple players reach on the same turn, they ALL win.
    Game ends when ALL players reach the goal.
    
    Returns: (win_message, game_ended)
    - win_message: Message about winners or None if no winners
    - game_ended: True if game has ended (all players reached end), False otherwise
    """
    data = get_game_data(game_state)
    rules = game_config.rules or {}
    
    win_tile = int(rules.get("win_tile", 100))
    # CRITICAL: Use current turn_count (before turn advances)
    # This ensures players who reach the goal in the same turn cycle get the same turn number
    current_turn = game_state.turn_count
    
    # Ensure data structures exist
    if 'winners' not in data:
        data['winners'] = []
    if 'players_reached_end_this_turn' not in data:
        data['players_reached_end_this_turn'] = []
    if 'goal_reached_turn' not in data:
        data['goal_reached_turn'] = {}
    
    # Track players who reached the goal this turn (for recording turn number)
    new_goal_reachers_this_turn = []
    for user_id, tile_num in data['tile_numbers'].items():
        if tile_num >= win_tile:
            # Player is at or past the goal
            if user_id not in data['goal_reached_turn']:
                # First time reaching the goal - record the turn number
                data['goal_reached_turn'][user_id] = current_turn
                new_goal_reachers_this_turn.append(user_id)
                logger.info("Player %s reached goal on turn %d", user_id, current_turn)
            # Add to winners list if not already there (prevents rolling)
            if user_id not in data['winners']:
                data['winners'].append(user_id)
                if user_id not in data['players_reached_end_this_turn']:
                    data['players_reached_end_this_turn'].append(user_id)
    
    # Check for game end condition (all players reached end)
    all_at_end = all(
        tile_num >= win_tile
        for tile_num in data['tile_numbers'].values()
    )
    
    if all_at_end:
        # Game ends automatically when all players reach the end
        # Determine winners based on who reached FIRST (lowest turn number)
        if not data.get('game_ended', False):
            data['game_ended'] = True
            
            # Find the earliest turn when someone reached the goal
            if data['goal_reached_turn']:
                earliest_turn = min(data['goal_reached_turn'].values())
                # CRITICAL: Clear winners list and rebuild with only actual winners
                # Only players who reached on the earliest turn are winners
                data['winners'] = [
                    user_id for user_id, turn_num in data['goal_reached_turn'].items()
                    if turn_num == earliest_turn
                ]
                winners = data['winners']
            else:
                # Fallback: all players are winners if no turn tracking
                data['winners'] = list(data['tile_numbers'].keys())
                winners = data['winners']
            
            # Build winner message using the corrected winners list
            winner_mentions = []
            for user_id in winners:
                player = game_state.players.get(user_id)
                if player:
                    player_name = player.character_name or f"Player {user_id}"
                    player_num = data.get('player_numbers', {}).get(user_id, "?")
                    turn_reached = data['goal_reached_turn'].get(user_id, "?")
                    winner_mentions.append(f"<@{user_id}> ({player_name} - Player {player_num}, Turn {turn_reached})")
                else:
                    winner_mentions.append(f"<@{user_id}>")
            
            # Build all players message
            all_player_names = []
            for user_id in data['tile_numbers'].keys():
                player = game_state.players.get(user_id)
                if player:
                    player_name = player.character_name or f"Player {user_id}"
                    player_num = data.get('player_numbers', {}).get(user_id, "?")
                    turn_reached = data['goal_reached_turn'].get(user_id, "?")
                    all_player_names.append(f"<@{user_id}> ({player_name} - Player {player_num}, Turn {turn_reached})")
                else:
                    all_player_names.append(f"<@{user_id}>")
            
            if len(winners) == 1:
                return (
                    f"🎉 **GAME OVER!** 🎉\n"
                    f"**🏆 WINNER:** {winner_mentions[0]}\n"
                    f"**All players have reached the end!**\n"
                    f"**Final Results:**\n" + "\n".join(all_player_names),
                    True
                )
            else:
                return (
                    f"🎉 **GAME OVER!** 🎉\n"
                    f"**🏆 WINNERS (Tied on Turn {earliest_turn}):** {', '.join(winner_mentions)}\n"
                    f"**All players have reached the end!**\n"
                    f"**Final Results:**\n" + "\n".join(all_player_names),
                    True
                )
        return (None, True)  # Game already ended, no new message
    
    # Check for new winners this turn (players who reached end within current turn)
    if new_goal_reachers_this_turn:
        # Multiple players can win if they reach the end on the same turn
        winner_mentions = []
        for user_id in new_goal_reachers_this_turn:
            player = game_state.players.get(user_id)
            if player:
                player_name = player.character_name or f"Player {user_id}"
                winner_mentions.append(f"<@{user_id}> ({player_name})")
            else:
                winner_mentions.append(f"<@{user_id}>")
        
        if len(new_goal_reachers_this_turn) == 1:
            return (f"🎉 {winner_mentions[0]} reached the goal (tile {win_tile}) on turn {current_turn}! They cannot roll dice anymore, but the game continues for others.", False)
        else:
            return (f"🎉 **WINNERS!** 🎉\n{', '.join(winner_mentions)} have all reached tile {win_tile} on turn {current_turn}! They cannot roll dice anymore, but the game continues for others.", False)
    
    return (None, False)


def get_player_tile_number(game_state: GameState, user_id: int) -> int:
    """Get player's current tile number."""
    data = get_game_data(game_state)
    return data['tile_numbers'].get(user_id, 1)


def get_transformation_count(game_state: GameState, user_id: int) -> int:
    """Get player's transformation count."""
    data = get_game_data(game_state)
    return data['transformation_counts'].get(user_id, 0)


def get_player_number(game_state: GameState, user_id: int) -> Optional[int]:
    """Get player number (1, 2, 3, etc.) based on order added to game."""
    data = get_game_data(game_state)
    return data.get('player_numbers', {}).get(user_id)


def should_update_board(game_state: GameState, event: str) -> bool:
    """
    Control when board should be updated.
    
    Events: "player_added", "character_assigned", "dice_rolled", "turn_complete", "move", "win", "game_end"
    
    For Snakes & Ladders: Update on assignment, movement, turn completion, wins, and game end.
    Do NOT update when player is just added (wait for assignment).
    """
    # Update board on these events
    update_events = ["character_assigned", "dice_rolled", "turn_complete", "move", "win", "game_end"]
    return event in update_events


def validate_move(game_state: GameState, player: GamePlayer, new_position: str, game_config: GameConfig) -> Tuple[bool, Optional[str]]:
    """
    Validate a GM move to a new position.
    
    Returns: (is_valid, error_message)
    """
    # Convert position to tile number to validate
    tile_num = alphanumeric_to_tile_number(new_position, game_config)
    if tile_num is None:
        return (False, f"Invalid coordinate: {new_position}")
    
    # Check bounds
    rules = game_config.rules or {}
    win_tile = int(rules.get("win_tile", 100))
    if tile_num < 1 or tile_num > win_tile:
        return (False, f"Position {new_position} (tile {tile_num}) is out of bounds (1-{win_tile})")
    
    return (True, None)


def get_turn_summary(game_state: GameState, game_config: GameConfig) -> str:
    """
    Generate turn summary with leaderboard showing ALL players.
    
    Returns: Formatted message with all players' positions.
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
    
    # Build summary
    summary_parts = ["**Turn Complete!**", "", "**Leaderboard (All Players):**"]
    
    # Show all players in order
    for idx, (user_id, tile_num, player) in enumerate(player_positions):
        player_name = player.character_name or f"Player {user_id}"
        player_num = data.get('player_numbers', {}).get(user_id, "?")
        
        # Add emoji for positions
        if idx == 0:
            emoji = "🥇"
        elif idx == len(player_positions) - 1:
            emoji = "🥉"
        else:
            emoji = "📍"
        
        # Check if player is a winner
        is_winner = user_id in data.get('winners', [])
        winner_text = " 🏆 WINNER" if is_winner else ""
        
        summary_parts.append(f"{emoji} **Player {player_num}** - {player_name}: Tile {tile_num}{winner_text}")
    
    return "\n".join(summary_parts)


def advance_turn(game_state: GameState) -> None:
    """Reset players_rolled_this_turn and advance to next turn."""
    data = get_game_data(game_state)
    
    # Reset players who have rolled
    data['players_rolled_this_turn'] = []
    
    # Reset players who reached end this turn (for next turn's win detection)
    data['players_reached_end_this_turn'] = []
    
    # Advance turn index (this will be handled by the calling code if needed)
    # The turn index doesn't need to advance here since we're starting a new turn

