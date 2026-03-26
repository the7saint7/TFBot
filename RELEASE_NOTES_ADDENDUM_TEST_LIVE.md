# Add to Release Notes (merge into BunniBot_4.8_Release_Notes.md or later version)

**Placement:** Add as a new section before "## Configuration Summary" (or at the end of the main notes).

---

### TEST and LIVE in the Same Server (Isolation)

When running **TEST** on your dev machine and **LIVE** on the server, both bots can be in the **same Discord server** using **different channels** (no-interaction setup by design). In that setup they should not interact, but limited cross-talk can occur (e.g. !outfits loading the wrong character) if state is keyed only by guild/user and not by bot.

**What we do about it:**

- **Gameboard** already isolates by bot: each game stores `bot_user_id`; a bot ignores games owned by the other bot.
- **Gacha** (and other shared state where applicable) is being updated so state is keyed by **bot identity** (e.g. `bot_user_id`) where needed, so TEST and LIVE never use each other's data.
- **Folder/path separation** is not relevant when the two instances run on different machines; Discord is the only connection.

**Config / .env:**

- When you add or change variables or comments in **`.env.example`**, mirror those changes into your active **`.env`** (e.g. add new keys with empty or placeholder values only when the key is missing). **Do not** change or overwrite existing values in `.env`.

---

## Configuration Summary (additional row if needed)

| Variable | Description |
|----------|-------------|
| *(No new variables for isolation; optional _LIVE/_TEST path vars only if running both instances on the same machine.)* |

---
