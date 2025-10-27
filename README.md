# TF Discord Bot

Basic Discord bot scaffold that listens for messages and has a configurable chance (default 10%) of triggering a TF response. Future iterations can expand the transformation narrative without reworking the scaffolding.

## Setup
1. Install Python 3.10+.
2. Create and activate a virtual environment (recommended).
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and fill in your values.

## Configuration
Environment variables (via `.env` or your shell) control runtime behavior:

| Variable | Description | Default |
| --- | --- | --- |
| `DISCORD_TOKEN` | Bot token from the Discord developer portal | _(required)_ |
| `TFBOT_CHANCE` | Decimal probability that a message triggers a TF | `0.10` |
| `TFBOT_IGNORE_CHANNELS` | Comma-separated channel IDs that should never trigger TFs | _(empty)_ |
| `TFBOT_PREFIX` | Command prefix for future bot commands | `!` |
| `TFBOT_LOG_LEVEL` | Python logging level | `INFO` |
| `TFBOT_HISTORY_CHANNEL_ID` | Channel that receives TF audit logs | `1432196317722972262` |
| `TFBOT_STATE_FILE` | Path for persisting active TF records | `tf_state.json` |
| `TFBOT_MAGIC_EMOJI_NAME` | Custom emoji name to prefix TF narration (falls back to `:name:` if missing) | `magic_emoji` |
| `TFBOT_STATS_FILE` | JSON file storing per-user TF counts | `tf_stats.json` |

## Running
```bash
python bot.py
```
The bot requires the `MESSAGE CONTENT INTENT` to be enabled in the Discord developer portal.

- Messages from members with the Discord Administrator permission (or a role literally named `Admin`) are ignored automatically.

### TF Flow
- When the 10% roll succeeds, the bot:
  - Picks a character from `tf_characters.py`.
  - Updates the member's nickname (and attempts to set their server avatar if the API/permissions allow it).
  - Stores the original nickname and TF metadata in `tf_state.json`.
  - Selects a random duration (10m, 1h, 10h, 24h) and schedules an automatic revert.
  - Sends an entry to the history channel defined by `TFBOT_HISTORY_CHANNEL_ID` without pinging the user or naming the guild.
- When the timer expires, the nickname and avatar revert and the history channel receives a 'TF Reverted' entry that references the member's username (not display name) and the character that expired.
- While a user is transformed, every message they send is mirrored by the bot as an embedded post that uses the character's avatar thumbnail (requires the bot to have **Manage Messages** to remove the original post cleanly).
- Members with the Discord Administrator permission (or a role literally named `Admin`) and ignored channels never trigger TFs.
- Server admins can run the hidden `!synreset` command to immediately revert all active TFs in the current server and log the action to the history channel.
- Any member can type `!tf` to see how many times they've transformed and how often each character has appeared.
- Set `TFBOT_MAGIC_EMOJI_NAME` to the custom emoji name you want prefixed on TF narration; the bot looks it up per guild and falls back to plain text if missing.
- Only one user can embody a given character at a time; if every character is occupied, additional messages simply won't trigger until someone reverts.

Update `tf_characters.py` with your own roster (names, local avatar paths under `avatars/`, narration). Each `message` should contain only the unique narration body; the bot automatically prefixes the user's name and appends “becomes **{character}** for {duration}!” around it. Drop your PNGs/JPGs into the `avatars` folder and point each `avatar_path` field at the correct file.

### Dev mode
Development servers can launch with:
```bash
python bot.py -dev
```
- TF chance is forced to 75%.
- Only channel `1432191400983662766` will receive responses; all others are ignored.
- The regular ignore-list is bypassed while dev mode is active.

## Next Steps
- Flesh out `tf_characters.py` with real art assets and narration.
- Add commands to view/force/end TFs or to inspect the backlog channel directly.
