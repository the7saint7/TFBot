# Game Quick Start Guide

## Starting a Game (Admin/GM Only)

### Step 1: List Available Games
```
!listgames
```
Shows all available game types (e.g., `snakes_ladders`)

### Step 2: Start a New Game
```
!startgame <game_type>
```
Example: `!startgame snakes_ladders`

**Note:** Only admins can start games. The admin who starts becomes the GM (Game Master).

This creates:
- A game thread (for chat/commands)
- A map thread (for board images)

---

## Setting Up Players

### Step 3: Add Players
```
!addplayer @user
```
Add each player you want in the game. Example:
```
!addplayer @Player1
!addplayer @Player2
```

### Step 4: Assign Characters
```
!assign @user <character_name>
```
Assign a character to each player. Example:
```
!assign @Player1 Alice
!assign @Player2 Bob
```

**Note:** Characters must exist in the bot's character database.

---

## Playing the Game

### Step 5: Roll Dice
```
!dice
```
or
```
!roll
```

Players roll dice on their turn. The game will:
- Move your token automatically
- Apply snake/ladder effects
- Apply tile transformations (if any)
- Update the board

### Step 6: View Game Rules
```
!rules
```
Shows the game rules, objectives, and tile effects.

### Step 7: View Help
```
!help
```
Shows all available commands.

---

## Example Gameplay Flow

1. **Admin starts game:**
   ```
   !startgame snakes_ladders
   ```

2. **GM adds players:**
   ```
   !addplayer @Player1
   !addplayer @Player2
   ```

3. **GM assigns characters:**
   ```
   !assign @Player1 Alice
   !assign @Player2 Bob
   ```

4. **Players take turns rolling:**
   ```
   Player1: !dice
   Player2: !dice
   Player1: !dice
   ...
   ```

5. **Game continues until someone wins!**

---

## GM Commands (Game Master Only)

### Player Management
- `!addplayer @user` - Add a player to the game
- `!removeplayer @user` - Remove a player
- `!assign @user <character>` - Assign character to player
- `!reroll @user` - Randomly reroll a player's character
- `!swap @user1 @user2` - Swap characters between two players

### Game Control
- `!startgame <game_type>` - Start a new game (admin only)
- `!endgame` - End the game and lock the thread
- `!savegame` - Save current game state
- `!loadgame <file>` - Load a saved game state
- `!transfergm @user` - Transfer GM role to another user

### Board Management
- `!movetoken @user <coord>` - Manually move a player's token (e.g., `!movetoken @user A5`)
- `!debug` - Toggle debug mode (shows coordinate labels on board)

### Visual Customization
- `!bg @user <id>` - Set background for a player
- `!bg all <id>` - Set background for all players
- `!bg_list` - List available backgrounds
- `!outfit @user <outfit>` - Set outfit for a player
- `!outfit_list` - List available outfits

---

## Player Commands

- `!dice` or `!roll` - Roll dice on your turn
- `!rules` - Show game rules
- `!help` - Show available commands

---

## Tips

1. **Check the map thread** - Board updates appear in the separate map thread
2. **Turn order** - The game tracks turn order automatically
3. **Transformations** - Landing on colored tiles may transform your character
4. **Snakes & Ladders** - Landing on snake/ladder tiles moves you automatically
5. **Debug mode** - Use `!debug` to see all tile coordinates (helpful for troubleshooting)

---

## Troubleshooting

- **"No active game in this thread"** - Make sure you're in the game thread created by `!startgame`
- **"Only the GM can..."** - Only the GM (or admins) can use GM commands
- **"You're not in this game"** - Ask the GM to add you with `!addplayer`
- **Character not found** - Make sure the character name exists in the bot's character database

---

## Available Games

Check available games with:
```
!listgames
```

Currently configured:
- `snakes_ladders` - Snakes and Ladders with transformation mechanics

