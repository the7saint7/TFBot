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
| `TFBOT_CHANNEL_ID` | Channel ID where classic TF rolls are allowed | _(required in classic mode)_ |
| `TFBOT_PREFIX` | Command prefix for future bot commands | `!` |
| `TFBOT_LOG_LEVEL` | Python logging level | `INFO` |
| `TFBOT_HISTORY_CHANNEL_ID` | Channel that receives TF audit logs | `1432196317722972262` |
| `TFBOT_STATE_FILE` | Path for persisting active TF records | `tf_state.json` |
| `TFBOT_MAGIC_EMOJI_NAME` | Custom emoji name to prefix TF narration (falls back to `:name:` if missing) | `magic_emoji` |
| `TFBOT_MESSAGE_STYLE` | `classic` embed layout or `vn` for visual-novel image panels | `classic` |
| `TFBOT_VN_BASE` | Base PNG used for VN-style rendering (only when style is `vn`) | `vn_assets/vn_base.png` |
| `TFBOT_VN_FONT` | Optional TTF font path for VN text rendering | `fonts/Ubuntu-B.ttf` |
| `TFBOT_VN_EMOJI_FONT` | Optional emoji fallback font | `fonts/NotoEmoji-VariableFont_wght.ttf` |
| `TFBOT_VN_NAME_SIZE` | Font size (px) for VN character name | `34` |
| `TFBOT_VN_TEXT_SIZE` | Font size (px) for VN dialogue text | `26` |
| `TFBOT_VN_GAME_ROOT` | Path to Student Transfer game root for sprite composition | _(empty)_ |
| `TFBOT_VN_OUTFIT` | Default outfit filename when composing sprites | `casual.png` |
| `TFBOT_VN_FACE` | Default face sprite filename when composing sprites | `0.png` |
| `TFBOT_VN_AVATAR_MODE` | `game` to build sprites from Student Transfer assets or `user` to keep member avatars | `game` |
| `TFBOT_VN_AVATAR_SCALE` | Scaling factor applied to VN avatars before cropping into the panel | `1.0` |
| `TFBOT_VN_CACHE_DIR` | Directory that stores cached composite VN avatars (set empty to disable) | `vn_cache` |
| `TFBOT_VN_SELECTIONS` | JSON file storing per-character outfit overrides | `tf_outfits.json` |
| `TFBOT_STATS_FILE` | JSON file storing per-user TF counts | `tf_stats.json` |
| `TFBOT_AI_REWRITE` | Enable GPT paraphrasing for character voice (`true`/`false`) | `false` |
| `TFBOT_AI_MODEL` | OpenAI chat model used for rewrites | `gpt-3.5-turbo-1106` |
| `TFBOT_AI_MAX_TOKENS` | Maximum tokens to generate during rewrites | `80` |
| `TFBOT_AI_TEMPERATURE` | Sampling temperature for rewrite completions | `0.5` |
| `TFBOT_AI_TIMEOUT` | Seconds to wait before abandoning an AI rewrite | `2.0` |
| `TFBOT_AI_CONCURRENCY` | Maximum simultaneous rewrite requests | `1` |
| `TFBOT_AI_BACKOFF` | Seconds to pause after receiving a rate-limit response | `1.5` |
| `TFBOT_AI_MIN_INTERVAL` | Minimum seconds between rewrite requests | `0.75` |
| `TFBOT_AI_SYSTEM_PROMPT` | Override the system prompt used for rewrites | _(built-in)_ |
| `TFBOT_AI_API_KEY` / `OPENAI_API_KEY` | OpenAI API key (either variable works) | _(required when rewrites enabled)_ |

## Running
```bash
python bot.py
```
The bot requires the `MESSAGE CONTENT INTENT` to be enabled in the Discord developer portal.
Configure `TFBOT_CHANNEL_ID` with the numeric channel ID you want the bot to monitor; all other channels are ignored in classic mode.
Set `TFBOT_MODE=gacha` to run only the collection game. If you keep the default `classic` mode and also configure `TFBOT_GACHA_CHANNEL_ID`, the bot runs both: classic TF rolls in `TFBOT_CHANNEL_ID` and gacha relay panels in the gacha channel.

- Messages from members with the Discord Administrator permission (or a role literally named `Admin`) are ignored automatically.

### TF Flow
- When the 10% roll succeeds, the bot:
  - Picks a character from `tf_characters.py`.
  - Updates the member's nickname (and attempts to set their server avatar if the API/permissions allow it).
  - Stores the original nickname and TF metadata in `tf_state.json`.
  - Selects a random duration (10m, 1h, 10h, 24h) and schedules an automatic revert.
  - Sends an entry to the history channel defined by `TFBOT_HISTORY_CHANNEL_ID` without pinging the user or naming the guild.
- When the timer expires, the nickname and avatar revert and the history channel receives a 'TF Reverted' entry that references the member's username (not display name) and the character that expired.
- While a user is transformed, every message they send is mirrored by the bot; use `TFBOT_MESSAGE_STYLE=classic` for embeds with thumbnails or `vn` for generated visual-novel panels (requires **Manage Messages**). In `vn` mode the bot composites character sprites from the Student Transfer assets if `TFBOT_VN_GAME_ROOT` is provided. In VN mode, transformed users can switch sprites by running `!outfit <name>` after the bot DM shares available outfits.
- Members with the Discord Administrator permission (or a role literally named `Admin`) never trigger TFs. Messages outside the configured `TFBOT_CHANNEL_ID` channel are ignored.
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
- Transformations use a shortened two-minute duration.
- All other runtime settings remain the same, including the configured `TFBOT_CHANNEL_ID`.

## Next Steps
- Flesh out `tf_characters.py` with real art assets and narration.
- Add commands to view/force/end TFs or to inspect the backlog channel directly.
