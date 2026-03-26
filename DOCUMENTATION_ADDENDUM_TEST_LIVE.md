# Add to Full Documentation (merge into BunniBot_4.8_Full_Documentation.md)

**Purpose:** Document TEST/LIVE same-server setup, isolation by bot identity, and the .env.example ↔ .env mirror rule.

---

## Placement 1: After §1.4 TEST vs LIVE (new subsection 1.5)

Insert the following after the paragraph that ends with "...the default (0) is used." and before the "---" that precedes "## 2. Core Configuration".

### 1.5 TEST and LIVE in the Same Server (Isolation)

- **Typical setup:** The two instances are **never the same build** and usually run on **different machines** (e.g. you run TEST on your dev machine, LIVE on the server). **Discord is the only connection**; there is no shared filesystem.
- **Same server, different channels:** TEST and LIVE can both be in the same Discord server, using different channels with a no-interaction setup by design. They **should not** interact.
- **Limited cross-talk:** If state is keyed only by guild_id/user_id (or channel) without **bot identity**, limited interaction can still occur (e.g. !outfits loading the wrong character). The fix is to key Discord-scoped state by **bot instance** (e.g. `bot_user_id`) so each bot only uses its own data.
- **Gameboard:** Already isolates by bot: `GameState` stores `bot_user_id`; a bot skips processing games owned by another bot.
- **Gacha / other state:** Where applicable, state is keyed by bot identity (e.g. guild_id, user_id, **bot_user_id**) so TEST and LIVE do not share data.
- **Folder/path separation:** Not relevant when the two instances run on different machines. Mode-specific path env vars (_LIVE/_TEST) are only relevant when running both instances on the **same machine** (optional).

---

## Placement 2: In §2 Core Configuration (.env) – after the "Source of truth" paragraph

Insert the following immediately after the sentence "Never commit `.env`." and before "**Quick reference:**".

**Mirroring .env.example into .env:** When you add or change variables or comments in `.env.example`, **mirror** those changes into your active `.env`: add any **new** keys (with empty or placeholder value) only when the key is **not** already present. **Do not** change, overwrite, or delete existing values in the active `.env`.

---

## Placement 3 (optional): Table of Contents

If the doc has a Table of Contents, add an entry for the new subsection, for example:

- `1.5. [TEST and LIVE in the Same Server (Isolation)](#15-test-and-live-in-the-same-server-isolation)`

(Adjust anchor to match your markdown heading style, e.g. lowercase, hyphens.)
