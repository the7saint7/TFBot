"""Gacha mode subsystem for TFBot."""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import timedelta
from io import BytesIO
import json
import logging
import math
import random
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import discord
from discord.abc import Messageable
from discord.ext import commands

from tfbot.models import TFCharacter, TransformationState
from tfbot.panels import compose_game_avatar, list_pose_outfits, render_vn_panel
from tfbot.utils import float_from_env, int_from_env, is_admin, path_from_env, utc_now

logger = logging.getLogger("tfbot.gacha")

# 👇 add this helper
_roll_logger = logging.getLogger("tfbot.gacha.rolls")
if not _roll_logger.handlers:
    _roll_logger.setLevel(logging.DEBUG)          # lowest we care about
    _roll_logger.propagate = False                # don't let the app mute us

    _roll_handler = logging.StreamHandler()       # print to stdout
    _roll_handler.setLevel(logging.DEBUG)
    _roll_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))
    _roll_logger.addHandler(_roll_handler)

# Type aliases
RelayCallback = Callable[[discord.Message, TransformationState], asyncio.Future]

RARITY_STAR_FLOOR: Dict[str, int] = {
    "common": 1,
    "uncommon": 1,
    "rare": 2,
    "epic": 3,
    "ultra": 3,
    "legendary": 3,
}

RARITY_BORDER_MAP: Dict[str, Optional[str]] = {
    "common": "common",
    "uncommon": None,
    "rare": "gold",
    "epic": "epic",
    "ultra": "ultra",
    "legendary": "epic",
}

RARITY_EMBED_COLORS: Dict[str, int] = {
    "common": 0x95A5A6,
    "uncommon": 0x1ABC9C,
    "rare": 0xF1C40F,
    "epic": 0x9B59B6,
    "ultra": 0xA22424,
    "legendary": 0xF1C40F,
}
DEFAULT_EMBED_COLOR = 0x5865F2

EMBED_TOTAL_CHAR_LIMIT = 6000
EMBED_MAX_FIELDS = 25
EMBED_FIELD_VALUE_LIMIT = 1024


@dataclass(frozen=True)
class GachaOutfitDef:
    key: str
    pose: Optional[str]
    outfit: str
    rarity: str = "common"
    label: str = ""


@dataclass(frozen=True)
class GachaCharacterDef:
    slug: str
    display_name: str
    source_name: str
    rarity: str = "common"
    message: str = ""
    avatar_path: str = ""
    outfits: Mapping[str, GachaOutfitDef] = field(default_factory=dict)
    is_inanimate: bool = False
    inanimate_responses: Tuple[str, ...] = field(default_factory=tuple)


@dataclass
class GachaProfile:
    guild_id: int
    user_id: int
    rudy_coins: int
    frog_coins: int
    equipped_character: Optional[str]
    equipped_outfit: Optional[str]
    starter_granted: bool
    boost_rolls_remaining: int
    boost_bonus: float


def _normalize_key(value: str) -> str:
    if not value:
        return ""
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _load_gacha_config(path: Optional[Path]) -> dict:
    if not path:
        return {}
    try:
        if not path.exists():
            logger.warning("Gacha config %s not found; using defaults.", path)
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
        logger.warning("Gacha config %s must be a JSON object.", path)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse gacha config %s: %s", path, exc)
    return {}


class GachaManager:
    """Encapsulates gacha state, database, and command handlers."""

    def __init__(
        self,
        *,
        bot: commands.Bot,
        character_pool: Sequence[TFCharacter],
        relay_fn: Callable[..., asyncio.Future],
        config_path: Optional[Path],
    ) -> None:
        self.bot = bot
        self._relay = relay_fn
        self._config = _load_gacha_config(config_path)
        self._rarity_weights = self._load_rarity_weights()
        self._starter_rarities = self._load_starter_rarities()
        self._starter_characters = self._load_starter_character_set()
        self._starter_outfits = self._load_starter_outfit_map()
        self._catalog = self._build_catalog(character_pool)
        self._catalog_by_display = {char.display_name.lower(): char for char in self._catalog.values()}
        self._catalog_by_source = {char.source_name.lower(): char for char in self._catalog.values()}
        self._spam_tracker: Dict[Tuple[int, int], dict] = {}
        self._webhook_cache: Dict[int, discord.Webhook] = {}

        db_path = path_from_env("TFBOT_GACHA_DB_PATH") or Path("tfbot_gacha.sqlite3")
        if not db_path.is_absolute():
            db_path = Path.cwd() / db_path
        self._db_path = db_path
        self._conn = self._connect_db()
        self._lock = asyncio.Lock()
        self._create_tables()

        self._allowed_guild_id = int_from_env("TFBOT_GACHA_GUILD_ID", 0)

        channel_id = int_from_env("TFBOT_GACHA_CHANNEL_ID", 0)
        if not channel_id:
            raise RuntimeError("TFBOT_GACHA_CHANNEL_ID is required in gacha mode.")
        self.channel_id = channel_id

        self.starting_coins = int_from_env("TFBOT_GACHA_STARTING_COINS", 1200)
        self.coin_earn_per_message = int_from_env("TFBOT_GACHA_EARN_PER_MESSAGE", 3)
        self.min_words_for_reward = int_from_env("TFBOT_GACHA_MIN_WORDS", 3)
        self.character_roll_cost = int_from_env("TFBOT_GACHA_CHARACTER_ROLL_COST", 350)
        self.outfit_roll_cost = int_from_env("TFBOT_GACHA_OUTFIT_ROLL_COST", 150)
        self.duplicate_character_frog = int_from_env("TFBOT_GACHA_DUPLICATE_CHARACTER_FROG", 2)
        self.duplicate_outfit_frog = int_from_env("TFBOT_GACHA_DUPLICATE_OUTFIT_FROG", 1)
        self.frog_to_rudy_rate = int_from_env("TFBOT_GACHA_FROG_TO_RUDY_RATE", 50)
        self.frog_boost_cost = int_from_env("TFBOT_GACHA_FROG_BOOST_COST", 5)
        self.frog_boost_bonus = float_from_env(
            "TFBOT_GACHA_FROG_BOOST_BONUS",
            self._config.get("frog_boost_bonus", 0.15),
        )
        self.frog_boost_rolls = int_from_env(
            "TFBOT_GACHA_FROG_BOOST_ROLLS",
            int(self._config.get("frog_boost_rolls", 2)),
        )
        boost_targets = self._config.get("frog_boost_targets", ["rare", "epic", "ultra"])
        if isinstance(boost_targets, (list, tuple)):
            self._boost_target_rarities = tuple(str(item).lower() for item in boost_targets if str(item).strip())
        else:
            self._boost_target_rarities = ("rare", "epic", "ultra")
        self._gacha_channel: Optional[discord.TextChannel] = None
        self._rp_forum_post_id = int_from_env("TFBOT_RP_FORUM_POST_ID", 0)

    def _load_rarity_weights(self) -> Dict[str, float]:
        default_weights = {"common": 70.0, "rare": 25.0, "epic": 4.0, "ultra": 1.0}
        config_weights = self._config.get("rarities")
        if isinstance(config_weights, Mapping):
            weights: Dict[str, float] = {}
            for key, value in config_weights.items():
                try:
                    weights[str(key).lower()] = float(value)
                except (TypeError, ValueError):
                    logger.warning("Ignoring invalid weight for rarity %s", key)
            if weights:
                return weights
        return default_weights

    def _load_starter_rarities(self) -> Sequence[str]:
        entries = self._config.get("starter_rarities")
        if isinstance(entries, list):
            normalized = [str(item).lower() for item in entries if str(item).strip()]
            if normalized:
                return tuple(normalized)
        return ("common",)

    def _load_starter_character_set(self) -> Optional[Sequence[str]]:
        entries = self._config.get("starter_characters")
        if isinstance(entries, list):
            names = [str(item).strip() for item in entries if str(item).strip()]
            if names:
                return tuple(names)
        return None

    def _load_starter_outfit_map(self) -> Dict[str, Sequence[str]]:
        result: Dict[str, Sequence[str]] = {}
        entries = self._config.get("starter_outfits")
        if isinstance(entries, Mapping):
            for key, value in entries.items():
                if isinstance(value, list):
                    options = [str(item).strip() for item in value if str(item).strip()]
                    if options:
                        result[str(key).strip().lower()] = tuple(options)
        return result

    def _build_catalog(self, character_pool: Sequence[TFCharacter]) -> Dict[str, GachaCharacterDef]:
        catalog: Dict[str, GachaCharacterDef] = {}
        config_characters = self._normalize_character_config()

        for character in character_pool:
            first_token = (character.name.split()[0]) if character.name else ""
            hyphen_token = first_token.split("-")[0] if first_token else ""
            slug_candidates = []
            for token in (first_token, hyphen_token, character.name):
                normalized = _normalize_key(token)
                if normalized and normalized not in slug_candidates:
                    slug_candidates.append(normalized)
            slug = None
            entry = None
            for candidate in slug_candidates:
                candidate_entry = config_characters.get(candidate)
                if isinstance(candidate_entry, Mapping):
                    slug = candidate
                    entry = candidate_entry
                    break
            if slug is None:
                slug = slug_candidates[0] or _normalize_key(character.name)
                entry = config_characters.get(slug)

            if not isinstance(entry, Mapping):
                logger.debug(
                    "Skipping gacha character %s (%s); missing gacha_config entry.",
                    character.display_name or character.name,
                    slug or "unknown",
                )
                continue

            display_name = character.display_name or character.name
            if isinstance(entry, Mapping):
                configured_name = entry.get("display_name")
                if isinstance(configured_name, str) and configured_name.strip():
                    display_name = configured_name.strip()

            rarity = "common"
            if isinstance(entry, Mapping):
                rarity_value = entry.get("rarity")
                if isinstance(rarity_value, str) and rarity_value.strip():
                    rarity = rarity_value.strip().lower()

            outfits = self._build_outfits(character.name, entry)
            catalog[slug] = GachaCharacterDef(
                slug=slug,
                display_name=display_name,
                source_name=character.name,
                rarity=rarity,
                message=character.message,
                avatar_path=character.avatar_path,
                outfits=outfits,
                is_inanimate=False,
                inanimate_responses=(),
            )
        for slug, entry in config_characters.items():
            char_def = self._build_config_only_character(slug, entry)
            if char_def is None:
                continue
            if char_def.slug in catalog:
                continue
            catalog[char_def.slug] = char_def
        return catalog

    def _normalize_character_config(self) -> Dict[str, Mapping[str, object]]:
        raw_characters = self._config.get("characters")
        if not isinstance(raw_characters, Mapping):
            return {}
        normalized: Dict[str, Mapping[str, object]] = {}
        for raw_key, raw_entry in raw_characters.items():
            if not isinstance(raw_entry, Mapping):
                continue
            normalized_key = _normalize_key(str(raw_key))
            if not normalized_key:
                display_value = raw_entry.get("display_name") or raw_entry.get("name")
                if isinstance(display_value, str):
                    normalized_key = _normalize_key(display_value)
            if not normalized_key:
                logger.warning("Gacha config: skipping character entry %s (cannot normalize key)", raw_key)
                continue
            if normalized_key in normalized:
                logger.warning(
                    "Gacha config: duplicate character entry for %s; keeping first occurrence.",
                    normalized_key,
                )
                continue
            normalized[normalized_key] = raw_entry
        return normalized

    def _build_config_only_character(
        self,
        slug: str,
        entry: object,
    ) -> Optional[GachaCharacterDef]:
        if not isinstance(entry, Mapping):
            return None
        inanimate_flag = bool(entry.get("inanimate"))
        if not inanimate_flag:
            return None
        normalized_slug = _normalize_key(slug) or _normalize_key(str(entry.get("display_name") or ""))
        if not normalized_slug:
            return None
        display_name = str(entry.get("display_name") or entry.get("name") or slug).strip()
        if not display_name:
            display_name = normalized_slug
        message = str(entry.get("message") or "").strip()
        avatar_path = self._resolve_avatar_path(str(entry.get("avatar_path") or ""))
        responses = self._extract_inanimate_responses(entry, message)
        rarity = "epic"
        return GachaCharacterDef(
            slug=normalized_slug,
            display_name=display_name,
            source_name=display_name,
            rarity=rarity,
            message=message,
            avatar_path=avatar_path,
            outfits={},
            is_inanimate=True,
            inanimate_responses=responses,
        )

    @staticmethod
    def _resolve_avatar_path(raw_path: str) -> str:
        path = raw_path.strip()
        if not path:
            return ""
        if path.startswith("http://") or path.startswith("https://"):
            return path
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        return str(candidate)

    @staticmethod
    def _extract_inanimate_responses(entry: Mapping[str, object], fallback: str) -> Tuple[str, ...]:
        responses: List[str] = []
        raw_responses = entry.get("responses")
        if isinstance(raw_responses, Sequence) and not isinstance(raw_responses, (str, bytes)):
            for item in raw_responses:
                if isinstance(item, str):
                    cleaned = item.strip()
                    if cleaned:
                        responses.append(cleaned)
        if not responses and fallback:
            responses.append(fallback)
        return tuple(responses)

    @staticmethod
    def _load_static_avatar_image(avatar_path: str) -> Optional["Image.Image"]:
        if not avatar_path:
            return None
        try:
            from PIL import Image
        except ImportError:
            return None

        path = Path(avatar_path)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.exists():
            logger.warning("Gacha embed: avatar path %s missing", path)
            return None
        try:
            with Image.open(path) as img:
                return img.convert("RGBA")
        except OSError as exc:
            logger.warning("Gacha embed: failed to load avatar %s: %s", path, exc)
            return None

    def _build_outfits(
        self,
        character_name: str,
        config_entry: Optional[Mapping[str, object]],
    ) -> Dict[str, GachaOutfitDef]:
        outfits: Dict[str, GachaOutfitDef] = {}

        if isinstance(config_entry, Mapping):
            poses_entry = config_entry.get("poses")
            if isinstance(poses_entry, Mapping):
                for pose_name, pose_meta in poses_entry.items():
                    pose_default_rarity = "common"
                    pose_outfits: Optional[Mapping[str, object]] = None
                    if isinstance(pose_meta, Mapping):
                        raw_outfits = pose_meta.get("outfits")
                        if isinstance(raw_outfits, Mapping):
                            pose_outfits = raw_outfits
                        pose_rarity_raw = pose_meta.get("rarity")
                        if isinstance(pose_rarity_raw, str) and pose_rarity_raw.strip():
                            pose_default_rarity = pose_rarity_raw.strip().lower()
                    if not pose_outfits:
                        continue
                    for outfit_name, outfit_meta in pose_outfits.items():
                        rarity = pose_default_rarity
                        if isinstance(outfit_meta, Mapping):
                            rarity_value = outfit_meta.get("rarity")
                            if isinstance(rarity_value, str) and rarity_value.strip():
                                rarity = rarity_value.strip().lower()
                        elif isinstance(outfit_meta, str) and outfit_meta.strip():
                            rarity = outfit_meta.strip().lower()
                        key = self._make_outfit_key(pose_name, outfit_name)
                        outfits[key] = GachaOutfitDef(
                            key=key,
                            pose=pose_name,
                            outfit=outfit_name,
                            rarity=rarity,
                            label=self._format_combo_label(pose_name, outfit_name),
                        )
        if outfits:
            return outfits

        fallback = list_pose_outfits(character_name)
        for pose_name, pose_outfits in fallback.items():
            for outfit_name in pose_outfits:
                key = self._make_outfit_key(pose_name, outfit_name)
                outfits[key] = GachaOutfitDef(
                    key=key,
                    pose=pose_name,
                    outfit=outfit_name,
                    rarity="common",
                    label=self._format_combo_label(pose_name, outfit_name),
                )
        return outfits

    @staticmethod
    def _make_outfit_key(pose: Optional[str], outfit: str) -> str:
        pose_part = (pose or "").strip().lower()
        outfit_part = outfit.strip().lower()
        return f"{pose_part}::{outfit_part}"

    @staticmethod
    def _format_combo_label(pose: Optional[str], outfit: str) -> str:
        outfit_label = outfit.strip()
        pose_label = (pose or "").strip()
        return f"{pose_label} / {outfit_label}" if pose_label else outfit_label

    def _create_tables(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS gacha_profiles (
                    guild_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    rudy_coins INTEGER NOT NULL DEFAULT 0,
                    frog_coins INTEGER NOT NULL DEFAULT 0,
                    equipped_character TEXT,
                    equipped_outfit TEXT,
                    starter_granted INTEGER NOT NULL DEFAULT 0,
                    boost_rolls_remaining INTEGER NOT NULL DEFAULT 0,
                    boost_bonus REAL NOT NULL DEFAULT 0.0,
                    PRIMARY KEY (guild_id, user_id)
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS gacha_characters (
                    guild_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    character_name TEXT NOT NULL,
                    rarity TEXT,
                    obtained_at TEXT NOT NULL,
                    PRIMARY KEY (guild_id, user_id, character_name)
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS gacha_outfits (
                    guild_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    character_name TEXT NOT NULL,
                    outfit_name TEXT NOT NULL,
                    rarity TEXT,
                    obtained_at TEXT NOT NULL,
                    PRIMARY KEY (guild_id, user_id, character_name, outfit_name)
                )
                """
            )

    def _connect_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, isolation_level=None, check_same_thread=False)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _is_authorized_guild(self, guild: Optional[discord.Guild]) -> bool:
        if guild is None or self._allowed_guild_id == 0:
            return True
        return guild.id == self._allowed_guild_id

    async def _get_or_create_webhook(self, channel: discord.TextChannel) -> Optional[discord.Webhook]:
        webhook = self._webhook_cache.get(channel.id)
        if webhook is not None:
            return webhook
        try:
            existing = await channel.webhooks()
        except discord.HTTPException:
            return None

        target_name = "SynTF Gacha Relay"
        for hook in existing:
            if hook.name == target_name or (hook.user and hook.user.id == self.bot.user.id):
                self._webhook_cache[channel.id] = hook
                return hook

        try:
            webhook = await channel.create_webhook(name=target_name, reason="Gacha panel relay")
        except discord.HTTPException:
            return None

        self._webhook_cache[channel.id] = webhook
        return webhook

    # Database helpers -------------------------------------------------

    async def _fetch_profile(self, guild_id: int, user_id: int) -> GachaProfile:
        async with self._lock:
            cur = self._conn.execute(
                """
                SELECT guild_id, user_id, rudy_coins, frog_coins,
                       equipped_character, equipped_outfit,
                       starter_granted, boost_rolls_remaining, boost_bonus
                FROM gacha_profiles
                WHERE guild_id = ? AND user_id = ?
                """,
                (guild_id, user_id),
            )
            row = cur.fetchone()
            if row is None:
                self._conn.execute(
                    """
                    INSERT INTO gacha_profiles (
                        guild_id, user_id, rudy_coins, frog_coins,
                        equipped_character, equipped_outfit,
                        starter_granted, boost_rolls_remaining, boost_bonus
                    ) VALUES (?, ?, 0, 0, NULL, NULL, 0, 0, 0)
                    """,
                    (guild_id, user_id),
                )
                return GachaProfile(
                    guild_id=guild_id,
                    user_id=user_id,
                    rudy_coins=0,
                    frog_coins=0,
                    equipped_character=None,
                    equipped_outfit=None,
                    starter_granted=False,
                    boost_rolls_remaining=0,
                    boost_bonus=0.0,
                )
            return GachaProfile(
                guild_id=row[0],
                user_id=row[1],
                rudy_coins=row[2],
                frog_coins=row[3],
                equipped_character=row[4],
                equipped_outfit=row[5],
                starter_granted=bool(row[6]),
                boost_rolls_remaining=row[7],
                boost_bonus=row[8],
            )

    async def _update_profile(self, profile: GachaProfile) -> None:
        async with self._lock:
            self._conn.execute(
                """
                UPDATE gacha_profiles
                SET rudy_coins = ?, frog_coins = ?,
                    equipped_character = ?, equipped_outfit = ?,
                    starter_granted = ?, boost_rolls_remaining = ?, boost_bonus = ?
                WHERE guild_id = ? AND user_id = ?
                """,
                (
                    profile.rudy_coins,
                    profile.frog_coins,
                    profile.equipped_character,
                    profile.equipped_outfit,
                    1 if profile.starter_granted else 0,
                    profile.boost_rolls_remaining,
                    profile.boost_bonus,
                    profile.guild_id,
                    profile.user_id,
                ),
            )

    async def _add_character(self, guild_id: int, user_id: int, character_name: str, rarity: str) -> bool:
        async with self._lock:
            try:
                self._conn.execute(
                    """
                    INSERT INTO gacha_characters (guild_id, user_id, character_name, rarity, obtained_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (guild_id, user_id, character_name, rarity, utc_now().isoformat()),
                )
                return True
            except sqlite3.IntegrityError:
                return False

    async def _add_outfit(
        self,
        guild_id: int,
        user_id: int,
        character_name: str,
        outfit_name: str,
        rarity: str,
    ) -> bool:
        async with self._lock:
            try:
                self._conn.execute(
                    """
                    INSERT INTO gacha_outfits (guild_id, user_id, character_name, outfit_name, rarity, obtained_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (guild_id, user_id, character_name, outfit_name, rarity, utc_now().isoformat()),
                )
                return True
            except sqlite3.IntegrityError:
                return False

    async def _list_owned_characters(self, guild_id: int, user_id: int) -> Dict[str, str]:
        async with self._lock:
            cur = self._conn.execute(
                """
                SELECT character_name, COALESCE(rarity, 'common')
                FROM gacha_characters
                WHERE guild_id = ? AND user_id = ?
                ORDER BY character_name COLLATE NOCASE
                """,
                (guild_id, user_id),
            )
            return {row[0]: row[1] for row in cur.fetchall()}

    async def _list_owned_outfits(
        self, guild_id: int, user_id: int, character_name: Optional[str] = None
    ) -> Dict[str, Dict[str, str]]:
        params: Tuple[object, ...]
        query = """
            SELECT character_name, outfit_name, COALESCE(rarity, 'common')
            FROM gacha_outfits
            WHERE guild_id = ? AND user_id = ?
        """
        if character_name:
            query += " AND character_name = ?"
            params = (guild_id, user_id, character_name)
        else:
            params = (guild_id, user_id)
        query += " ORDER BY character_name COLLATE NOCASE, outfit_name COLLATE NOCASE"
        async with self._lock:
            cur = self._conn.execute(query, params)
            owned: Dict[str, Dict[str, str]] = {}
            for row in cur.fetchall():
                owned.setdefault(row[0], {})[row[1]] = row[2]
            return owned

    # Catalog helpers --------------------------------------------------

    def _lookup_character(self, name: str) -> Optional[GachaCharacterDef]:
        if not name:
            return None
        first_token = name.split()[0]
        hyphen_token = first_token.split("-")[0] if first_token else ""
        for token in (first_token, hyphen_token, name):
            slug_candidate = _normalize_key(token)
            if slug_candidate in self._catalog:
                return self._catalog[slug_candidate]
        slug_full = _normalize_key(name)
        if slug_full in self._catalog:
            return self._catalog[slug_full]
        lower = name.lower()
        if lower in self._catalog_by_display:
            return self._catalog_by_display[lower]
        if lower in self._catalog_by_source:
            return self._catalog_by_source[lower]
        return None

    async def _reset_database(self) -> None:
        async with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass
            if self._db_path.exists():
                try:
                    self._db_path.unlink()
                except OSError:
                    pass
            self._conn = self._connect_db()
            self._create_tables()

    def _lookup_outfit(self, character: GachaCharacterDef, outfit_key: str) -> Optional[GachaOutfitDef]:
        if not outfit_key:
            return None
        if outfit_key in character.outfits:
            return character.outfits[outfit_key]
        normalized = outfit_key.lower()
        return next(
            (outfit for key, outfit in character.outfits.items() if key == normalized or outfit.label.lower() == normalized),
            None,
        )

    def _match_outfit_query(self, character: GachaCharacterDef, query: str) -> Optional[str]:
        if not query:
            return None
        lowered = query.strip().lower()
        if lowered in character.outfits:
            return lowered
        for key, combo in character.outfits.items():
            if combo.label.lower() == lowered:
                return key
        pose = None
        outfit = None
        if "/" in query:
            pose, outfit = [part.strip() for part in query.split("/", 1)]
        elif ":" in query:
            pose, outfit = [part.strip() for part in query.split(":", 1)]
        else:
            tokens = query.split()
            if len(tokens) >= 2:
                pose = tokens[0].strip()
                outfit = " ".join(tokens[1:]).strip()
            else:
                outfit = query.strip()
        matches: list[str] = []
        for key, combo in character.outfits.items():
            pose_match = pose is None or (combo.pose and combo.pose.lower() == pose.lower())
            outfit_match = outfit is None or combo.outfit.lower() == outfit.lower()
            if pose_match and outfit_match:
                matches.append(key)
        if len(matches) == 1:
            return matches[0]
        return None

    def _available_characters(self) -> Sequence[GachaCharacterDef]:
        return list(self._catalog.values())

    async def _ensure_gacha_channel(self) -> Optional[discord.TextChannel]:
        if self._gacha_channel and isinstance(self._gacha_channel, discord.TextChannel):
            return self._gacha_channel
        candidate = self.bot.get_channel(self.channel_id)
        if isinstance(candidate, discord.TextChannel):
            self._gacha_channel = candidate
            return candidate
        try:
            fetched = await self.bot.fetch_channel(self.channel_id)
        except discord.HTTPException:
            return None
        if isinstance(fetched, discord.TextChannel):
            self._gacha_channel = fetched
            return fetched
        return None

    def _resolve_display_combo(
        self,
        character: GachaCharacterDef,
        outfit_key: Optional[str],
    ) -> Optional[GachaOutfitDef]:
        if outfit_key and outfit_key in character.outfits:
            return character.outfits[outfit_key]
        if character.outfits:
            return min(character.outfits.values(), key=lambda combo: combo.label.lower())
        return None

    @staticmethod
    def _select_awarded_outfit(
        character: GachaCharacterDef,
        owned_for_character: Mapping[str, str],
    ) -> Optional[GachaOutfitDef]:
        if not character.outfits:
            return None
        commons = [
            combo
            for combo in character.outfits.values()
            if (combo.rarity or "common").lower() == "common"
        ]
        pool = commons or list(character.outfits.values())
        unowned = [combo for combo in pool if combo.key not in owned_for_character]
        if unowned:
            pool = unowned
        return random.choice(pool) if pool else None

    def _embed_color_for_rarity(self, rarity: Optional[str]) -> int:
        if not rarity:
            return DEFAULT_EMBED_COLOR
        return RARITY_EMBED_COLORS.get(rarity.lower(), DEFAULT_EMBED_COLOR)

    @staticmethod
    def _format_star_progress(star_count: int) -> str:
        capped = max(0, min(int(star_count), 3))
        stars = "★" * capped
        empties = "☆" * (3 - capped)
        return f"{stars}{empties} ({capped}/3)"

    def _build_avatar_embed(
        self,
        *,
        user_id: int,
        author_name: str,
        character: GachaCharacterDef,
        outfit_key: Optional[str],
        description: str,
        title: str,
        star_count: int,
        rudy_balance: Optional[int],
        frog_balance: Optional[int],
        outfit_label: Optional[str],
        pose_override: Optional[str] = None,
        outfit_override: Optional[str] = None,
    ) -> Tuple[discord.Embed, Optional[discord.File]]:
        now = utc_now()
        rarity_label = (character.rarity or "common").title()
        embed = discord.Embed(
            title=title,
            description=description,
            color=self._embed_color_for_rarity(character.rarity),
            timestamp=now,
        )
        embed.set_author(name=author_name)
        embed.add_field(
            name="Character",
            value=f"{character.display_name} ({rarity_label})",
            inline=False,
        )
        if outfit_label:
            embed.add_field(name="Outfit", value=outfit_label, inline=False)
        embed.add_field(
            name="Collection Progress",
            value=self._format_star_progress(star_count),
            inline=False,
        )
        balance_lines: List[str] = []
        if rudy_balance is not None:
            balance_lines.append(f"Rudy Coins: **{rudy_balance}**")
        if frog_balance is not None:
            balance_lines.append(f"Frog Coins: **{frog_balance}**")
        if balance_lines:
            embed.add_field(name="Wallet", value="\n".join(balance_lines), inline=False)

        combo = self._resolve_display_combo(character, outfit_key)
        pose = pose_override or (combo.pose if combo else None)
        outfit = outfit_override or (combo.outfit if combo else None)
        avatar_file: Optional[discord.File] = None
        avatar_image = None
        if not character.is_inanimate:
            avatar_image = compose_game_avatar(
                character.display_name,
                pose_override=pose,
                outfit_override=outfit,
            )
        if avatar_image is None:
            avatar_image = self._load_static_avatar_image(character.avatar_path)
        if avatar_image is not None:
            buffer = BytesIO()
            avatar_image.save(buffer, format="PNG")
            buffer.seek(0)
            filename = f"gacha-avatar-{user_id}-{int(now.timestamp() * 1000)}.png"
            avatar_file = discord.File(buffer, filename=filename)
            embed.set_image(url=f"attachment://{filename}")

        return embed, avatar_file

    async def _send_avatar_announcement(
        self,
        destination: Messageable,
        message_text: str,
        **embed_kwargs,
    ) -> None:
        embed, file = self._build_avatar_embed(**embed_kwargs)
        if file:
            try:
                await destination.send(message_text, embed=embed, file=file)
            finally:
                try:
                    file.close()
                except AttributeError:
                    fp = getattr(file, "fp", None)
                    if fp and hasattr(fp, "close"):
                        fp.close()
        else:
            await destination.send(message_text, embed=embed)

    def _build_panel_file(
        self,
        *,
        guild_id: int,
        user_id: int,
        author_name: str,
        character: GachaCharacterDef,
        outfit_key: Optional[str],
        message_text: str,
        star_count: int,
        rudy_balance: Optional[int],
        frog_balance: Optional[int],
        border_style: Optional[str],
        gacha_outfit_override: Optional[str] = None,
        gacha_pose_override: Optional[str] = None,
    ) -> Optional[discord.File]:
        auto_border = RARITY_BORDER_MAP.get((character.rarity or "common").lower())
        border_style = border_style or auto_border

        combo = self._resolve_display_combo(character, outfit_key)
        pose_override = gacha_pose_override or (combo.pose if combo else None)
        outfit_override = gacha_outfit_override or (combo.outfit if combo else None)
        now = utc_now()
        state = TransformationState(
            user_id=user_id,
            guild_id=guild_id,
            character_name=character.display_name,
            character_folder=character.slug,
            character_avatar_path=character.avatar_path,
            character_message=character.message,
            original_nick=None,
            started_at=now,
            expires_at=now,
            duration_label="gacha",
            avatar_applied=True,
            original_display_name=author_name,
            is_inanimate=character.is_inanimate,
            inanimate_responses=character.inanimate_responses,
        )
        return render_vn_panel(
            state=state,
            message_content=message_text,
            character_display_name=character.display_name,
            original_name=author_name,
            attachment_id=f"gacha-{user_id}-{int(now.timestamp() * 1000)}",
            formatted_segments=[],
            custom_emoji_images={},
            reply_context=None,
            gacha_star_count=star_count,
            gacha_rudy=rudy_balance,
            gacha_frog=frog_balance,
            gacha_outfit_override=outfit_override,
            gacha_pose_override=pose_override,
            gacha_border=border_style,
        )

    # Public API -------------------------------------------------------

    async def handle_message(self, message: discord.Message, *, command_invoked: bool) -> bool:
        if message.author.bot or command_invoked:
            return False
        if not isinstance(message.channel, discord.TextChannel):
            return False
        if message.guild is None:
            return False
        if not self._is_authorized_guild(message.guild):
            return False
        if message.channel.id != self.channel_id:
            return False

        profile = await self._fetch_profile(message.guild.id, message.author.id)
        if not profile.starter_granted:
            await self._grant_starter_pack(message, profile)
            # Profile is updated in-place by _grant_starter_pack, no need to refetch

        if not await self.enforce_spam_policy(message, profile=profile):
            return False

        resolved = await self._resolve_equipped_character(profile, message.guild.id, message.author.id)
        if resolved is None:
            logger.debug("User %s has no equipped character.", message.author.id)
            return False
        character_def, selected_combo, owned_for_character = resolved

        await self._maybe_award_message_reward(profile, message.content)

        if character_def is None:
            return False

        star_count = self._calculate_star_rating_from_owned(character_def, owned_for_character)
        border_style = self._rarity_border(character_def, allow_common_fallback=False)

        has_links_or_media = bool(message.attachments or message.embeds)
        if not has_links_or_media:
            content_lower = (message.content or "").lower()
            if "http://" in content_lower or "https://" in content_lower:
                has_links_or_media = True

        if has_links_or_media:
            return True

        # Compose transformation state for fallback relay.
        now = utc_now()
        state = TransformationState(
            user_id=message.author.id,
            guild_id=message.guild.id,
            character_name=character_def.display_name,
            character_folder=character_def.slug,
            character_avatar_path=character_def.avatar_path,
            character_message=character_def.message,
            original_nick=None,
            started_at=now,
            expires_at=now,
            duration_label="gacha",
            avatar_applied=True,
            original_display_name=message.author.display_name,
            is_inanimate=character_def.is_inanimate,
            inanimate_responses=character_def.inanimate_responses,
        )

        if character_def.is_inanimate:
            await self._relay(
                message,
                state,
                gacha_stars=star_count,
                gacha_outfit=None,
                gacha_pose=None,
                gacha_rudy=profile.rudy_coins,
                gacha_frog=profile.frog_coins,
                gacha_border=border_style,
            )
            return True

        panel_kwargs = dict(
            guild_id=message.guild.id,
            user_id=message.author.id,
            author_name=message.author.display_name,
            character=character_def,
            outfit_key=selected_combo.key if selected_combo else None,
            message_text=message.content or "",
            star_count=star_count,
            rudy_balance=profile.rudy_coins,
            frog_balance=profile.frog_coins,
            border_style=border_style,
            gacha_outfit_override=selected_combo.outfit if selected_combo else None,
            gacha_pose_override=selected_combo.pose if selected_combo else None,
        )

        panel_file = self._build_panel_file(**panel_kwargs)
        if panel_file is None:
            await self._relay(
                message,
                state,
                gacha_stars=star_count,
                gacha_outfit=selected_combo.outfit if selected_combo else None,
                gacha_pose=selected_combo.pose if selected_combo else None,
                gacha_rudy=profile.rudy_coins,
                gacha_frog=profile.frog_coins,
                gacha_border=border_style,
            )
            return True

        webhook = await self._get_or_create_webhook(message.channel)
        if webhook is None:
            panel_file.close()
            await self._relay(
                message,
                state,
                gacha_stars=star_count,
                gacha_outfit=selected_combo.outfit if selected_combo else None,
                gacha_pose=selected_combo.pose if selected_combo else None,
                gacha_rudy=profile.rudy_coins,
                gacha_frog=profile.frog_coins,
                gacha_border=border_style,
            )
            return True

        try:
            await message.delete()
        except discord.HTTPException:
            panel_file.close()
            await self._relay(
                message,
                state,
                gacha_stars=star_count,
                gacha_outfit=selected_combo.outfit if selected_combo else None,
                gacha_pose=selected_combo.pose if selected_combo else None,
                gacha_rudy=profile.rudy_coins,
                gacha_frog=profile.frog_coins,
                gacha_border=border_style,
            )
            return True

        try:
            await webhook.send(
                username=message.author.display_name,
                avatar_url=message.author.display_avatar.url,
                content=None,
                file=panel_file,
                allowed_mentions=discord.AllowedMentions.none(),
            )
        except discord.HTTPException:
            await self._relay(
                message,
                state,
                gacha_stars=star_count,
                gacha_outfit=selected_combo.outfit if selected_combo else None,
                gacha_pose=selected_combo.pose if selected_combo else None,
                gacha_rudy=profile.rudy_coins,
                gacha_frog=profile.frog_coins,
                gacha_border=border_style,
            )
        finally:
            panel_file.close()

        return True

    async def ensure_profile(self, guild_id: int, user_id: int) -> GachaProfile:
        """Ensure a gacha profile exists; create it if missing."""
        return await self._fetch_profile(guild_id, user_id)

    async def award_message_reward(
        self,
        message: discord.Message,
        *,
        profile: Optional[GachaProfile] = None,
    ) -> bool:
        """Grant Rudy coins for a qualifying message outside the gacha channel."""
        if message.guild is None or message.author.bot:
            return False
        if not self._is_authorized_guild(message.guild):
            return False
        if profile is None:
            profile = await self._fetch_profile(message.guild.id, message.author.id)
        rewarded = await self._maybe_award_message_reward(profile, message.content or "")
        if rewarded:
            logger.debug(
                "Awarded %s Rudy coins to %s in guild %s (channel %s)",
                self.coin_earn_per_message,
                message.author.id,
                message.guild.id,
                getattr(message.channel, "id", "dm"),
            )
        return rewarded

    async def relay_classic_message(
        self,
        message: discord.Message,
        *,
        profile: Optional[GachaProfile] = None,
    ) -> bool:
        """Relay a message in the classic channel using the equipped gacha form."""
        if message.guild is None or not isinstance(message.channel, discord.TextChannel):
            return False
        if message.author.bot or not self._is_authorized_guild(message.guild):
            return False
        if profile is None:
            profile = await self._fetch_profile(message.guild.id, message.author.id)

        resolved = await self._resolve_equipped_character(profile, message.guild.id, message.author.id)
        if resolved is None:
            return False
        character_def, selected_combo, _owned_for_character = resolved

        has_links_or_media = bool(message.attachments or message.embeds)
        if not has_links_or_media:
            content_lower = (message.content or "").lower()
            if "http://" in content_lower or "https://" in content_lower:
                has_links_or_media = True
        if has_links_or_media:
            return False

        pose_override = selected_combo.pose if selected_combo else None
        outfit_override = selected_combo.outfit if selected_combo else None
        border_style = self._rarity_border(character_def, allow_common_fallback=True)

        now = utc_now()
        state = TransformationState(
            user_id=message.author.id,
            guild_id=message.guild.id,
            character_name=character_def.display_name,
            character_folder=character_def.slug,
            character_avatar_path=character_def.avatar_path,
            character_message=character_def.message,
            original_nick=None,
            started_at=now,
            expires_at=now,
            duration_label="gacha",
            avatar_applied=True,
            original_display_name=message.author.display_name,
            is_inanimate=character_def.is_inanimate,
            inanimate_responses=character_def.inanimate_responses,
        )

        result = await self._relay(
            message,
            state,
            gacha_stars=None,
            gacha_outfit=outfit_override,
            gacha_pose=pose_override,
            gacha_rudy=None,
            gacha_frog=None,
            gacha_border=border_style,
        )
        return bool(result)

    async def enforce_spam_policy(
        self,
        message: discord.Message,
        *,
        profile: Optional[GachaProfile] = None,
    ) -> bool:
        """Apply spam cooldown and penalties. Returns True if the message is allowed."""
        if message.guild is None or message.author.bot:
            return True
        if not isinstance(message.channel, discord.TextChannel):
            return True
        if not self._is_authorized_guild(message.guild):
            return True

        now = utc_now()
        spam_key = (message.guild.id, message.author.id)
        record = self._spam_tracker.get(spam_key)
        if record is None:
            record = {
                "messages": deque(maxlen=3),
                "offenses": 0,
                "cooldown_until": now - timedelta(seconds=1),
            }
            self._spam_tracker[spam_key] = record

        if now < record["cooldown_until"]:
            remaining = max(0, int((record["cooldown_until"] - now).total_seconds()))
            try:
                await message.delete()
            except discord.HTTPException:
                pass
            await message.channel.send(
                f"{message.author.mention}, you're still on cooldown for {remaining}s. Slow down!"
            )
            return False

        normalized_content = re.sub(r"\s+", " ", message.content).strip().lower()
        if not normalized_content:
            return True

        record["messages"].append(normalized_content)
        if len(record["messages"]) < record["messages"].maxlen:
            return True

        if len(set(record["messages"])) != 1:
            return True

        record["messages"].clear()
        record["offenses"] += 1
        offense = record["offenses"]
        if offense == 1:
            cooldown = timedelta(seconds=10)
            penalty = 0
            penalty_text = "That's a warning."
        elif offense == 2:
            cooldown = timedelta(minutes=1)
            penalty = 10
            penalty_text = "Penalty: -10 Rudy coins."
        else:
            cooldown = timedelta(minutes=10)
            penalty = 100
            penalty_text = "Penalty: -100 Rudy coins."
        record["cooldown_until"] = now + cooldown

        if penalty > 0:
            if profile is None:
                profile = await self._fetch_profile(message.guild.id, message.author.id)
            profile.rudy_coins = max(0, profile.rudy_coins - penalty)
            await self._update_profile(profile)

        try:
            await message.delete()
        except discord.HTTPException:
            pass
        await message.channel.send(
            f"🚨 {message.author.mention}, you've been caught speeding! {penalty_text} "
            f"Please wait {int(cooldown.total_seconds())}s before chatting again."
        )
        return False

    async def user_has_equipped_character(self, guild_id: int, user_id: int) -> bool:
        if self._allowed_guild_id and guild_id != self._allowed_guild_id:
            return False
        profile = await self._fetch_profile(guild_id, user_id)
        return bool(profile.equipped_character)

    async def _resolve_equipped_character(
        self,
        profile: GachaProfile,
        guild_id: int,
        user_id: int,
    ) -> Optional[Tuple[GachaCharacterDef, Optional[GachaOutfitDef], Mapping[str, str]]]:
        if not profile.equipped_character:
            return None

        character_def = self._lookup_character(profile.equipped_character)
        if character_def is None:
            logger.warning("Equipped character %s no longer available.", profile.equipped_character)
            profile.equipped_character = None
            profile.equipped_outfit = None
            await self._update_profile(profile)
            return None

        owned_map = await self._list_owned_outfits(guild_id, user_id, character_def.display_name)
        owned_for_character = owned_map.get(character_def.display_name, {})

        selected_combo = None
        if profile.equipped_outfit:
            selected_combo = self._lookup_outfit(character_def, profile.equipped_outfit)

        if (selected_combo is None or (selected_combo.key not in character_def.outfits)) and character_def.outfits:
            selected_combo = next(iter(character_def.outfits.values()))
            profile.equipped_outfit = selected_combo.key
            await self._update_profile(profile)

        if owned_for_character and selected_combo and selected_combo.key not in owned_for_character:
            def _sort_key(candidate: str) -> str:
                combo = character_def.outfits.get(candidate)
                return combo.label.lower() if combo else candidate.lower()

            replacement_key = sorted(owned_for_character.keys(), key=_sort_key)[0]
            selected_combo = character_def.outfits.get(replacement_key, selected_combo)
            profile.equipped_outfit = replacement_key
            await self._update_profile(profile)

        return character_def, selected_combo, owned_for_character

    def _rarity_border(self, character: GachaCharacterDef, *, allow_common_fallback: bool) -> Optional[str]:
        rarity = (character.rarity or "common").lower()
        border = RARITY_BORDER_MAP.get(rarity)
        if allow_common_fallback and not border:
            return "common"
        return border

    async def _maybe_award_message_reward(self, profile: GachaProfile, content: str) -> bool:
        text = content.strip()
        if not text or not self._qualifies_for_reward(text):
            return False
        profile.rudy_coins += self.coin_earn_per_message
        await self._update_profile(profile)
        return True

    def _qualifies_for_reward(self, content: str) -> bool:
        words = [word for word in re.split(r"\s+", content.strip()) if word]
        return len(words) >= self.min_words_for_reward

    def _calculate_star_rating_from_owned(
        self,
        character: GachaCharacterDef,
        owned_for_character: Mapping[str, str],
    ) -> int:
        owned_keys = set(owned_for_character.keys())
        total_outfits = len(character.outfits)
        rating = 0
        if total_outfits > 0:
            owned_count = sum(1 for key in character.outfits if key in owned_keys)
            if owned_count >= total_outfits:
                rating = 3
            elif owned_count >= math.ceil(total_outfits / 2):
                rating = 2
            elif owned_count > 0:
                rating = 1
        rarity = (character.rarity or "common").lower()
        rarity_floor = RARITY_STAR_FLOOR.get(rarity, 1)
        return max(rating, rarity_floor)

    # Rolling helpers --------------------------------------------------

    def _rarity_weight(self, rarity: str, profile: GachaProfile) -> float:
        base = float(self._rarity_weights.get(rarity, 1.0))
        if base <= 0:
            base = 1.0
        if profile.boost_rolls_remaining > 0 and rarity in self._boost_target_rarities:
            multiplier = 1.0 + max(profile.boost_bonus, 0.0)
            return base * multiplier
        return base

    def _weighted_choice(
        self,
        items: Sequence,
        *,
        rarity_getter: Callable[[object], str],
        profile: GachaProfile,
    ):

    
        # No items, no roll.
        if not items:
            return None

        # 1) Figure out which rarities are actually present in THIS roll
        #    (because the pool might not have every rarity every time).
        #    For each rarity, start from the base config weight (e.g. 70/25/5).
        rarities_in_pool: dict[str, float] = {}
        for item in items:
            rarity = (rarity_getter(item) or "common").lower()
            base_weight = float(self._rarity_weights.get(rarity, 1.0))
            rarities_in_pool[rarity] = base_weight

        # Keep a copy to compare against later.
        base_total = sum(rarities_in_pool.values())

        # Start with boosted = base
        boosted_rarities = dict(rarities_in_pool)

        # 2) Apply a flat boost to target rarities, if the player has boosted rolls.
        #    We treat profile.boost_bonus as "add this many points to each boosted rarity".
        if profile.boost_rolls_remaining > 0:
            flat_boost = max(float(profile.boost_bonus), 0.0)  # e.g. 25.0, not 0.25
            if flat_boost > 0.0:
                for rarity in boosted_rarities:
                    if rarity in self._boost_target_rarities:
                        boosted_rarities[rarity] = boosted_rarities[rarity] + flat_boost

                # 3) If adding boosts made the total bigger than the original base total,
                #    we need to *pull that excess out of common first* (your requirement).
                boosted_total = sum(boosted_rarities.values())
                overflow = boosted_total - base_total
                if overflow > 0:
                    # Drain "common" first, if we have it.
                    if "common" in boosted_rarities:
                        drain = min(overflow, boosted_rarities["common"])
                        boosted_rarities["common"] -= drain
                        overflow -= drain

                    # If there's STILL overflow, drain it evenly from NON-boosted rarities.
                    if overflow > 0:
                        non_boosted = [
                            r for r in boosted_rarities
                            if r not in self._boost_target_rarities
                        ]
                        # (common might already be 0 here)
                        if non_boosted:
                            share = overflow / len(non_boosted)
                            for r in non_boosted:
                                boosted_rarities[r] = max(0.0, boosted_rarities[r] - share)
                        # If we still somehow have overflow, we just live with small drift.

        # 4) Now convert the per-rarity numbers back into per-item weights.
        #    Every item of the same rarity gets the same final weight.
        weights: list[float] = []
        for item in items:
            rarity = (rarity_getter(item) or "common").lower()
            weight = boosted_rarities.get(rarity, 1.0)
            weights.append(weight)

        total = sum(weights)

        # 🪶 Debug log: show all computed weights per rarity and per item
        _roll_logger.info(
            "[Gacha] Final boosted_rarities: %s | Per-item weights: %s | Total weight: %.2f",
            boosted_rarities,
            dict(zip([getattr(i, 'display_name', str(i)) for i in items], weights)),
            total,
        )

        # Safety: if something went sideways, fall back to uniform.
        if total <= 0:
            return random.choice(items)

        # 5) Standard roulette-wheel selection with the (now redistributed) weights.
        pick = random.uniform(0, total)

        # 🧭 Debug log: display the random number selected and its range
        _roll_logger.info(
            "[Gacha] Random pick: %.3f (range: 0–%.3f)", pick, total
        )
        
        cumulative = 0.0
        for item, weight in zip(items, weights):
            cumulative += weight
            if pick <= cumulative:
                # 🧩 Debug log: display which item was chosen and its stats
                _roll_logger.info(
                    "[Gacha] Selected item: %s | Rarity: %s | Weight: %.2f | "
                    "Cumulative: %.2f / %.2f | Pick: %.2f",
                    getattr(item, 'display_name', str(item)),
                    (rarity_getter(item) or 'common').lower(),
                    weight,
                    cumulative,
                    total,
                    pick,
                )
                return item

        # Fallback for floating point edge cases.
        return items[-1]


    def _consume_boost(self, profile: GachaProfile) -> None:
        if profile.boost_rolls_remaining > 0:
            profile.boost_rolls_remaining -= 1
            if profile.boost_rolls_remaining <= 0:
                profile.boost_rolls_remaining = 0
                profile.boost_bonus = 0.0

    def _chunk_lines(self, lines: Sequence[str], *, limit: int = 900) -> List[str]:
        buckets: List[str] = []
        current: List[str] = []
        current_len = 0
        for line in lines:
            if current_len + len(line) + 1 > limit and current:
                buckets.append("\n".join(current))
                current = []
                current_len = 0
            current.append(line)
            current_len += len(line) + 1
        if current:
            buckets.append("\n".join(current))
        return buckets

    def _clone_embed(self, embed: discord.Embed) -> discord.Embed:
        """Create a shallow copy of an embed with shared metadata."""
        clone = discord.Embed(
            title=embed.title,
            description=embed.description,
            colour=embed.colour,
            timestamp=embed.timestamp,
        )
        if embed.author:
            clone.set_author(name=embed.author.name, url=embed.author.url, icon_url=embed.author.icon_url)
        if embed.footer:
            clone.set_footer(text=embed.footer.text, icon_url=embed.footer.icon_url)
        if embed.thumbnail:
            clone.set_thumbnail(url=embed.thumbnail.url)
        if embed.image:
            clone.set_image(url=embed.image.url)
        return clone

    def _iter_field_chunks(
        self,
        name: str,
        value: str,
        inline: bool,
    ) -> Iterable[Tuple[str, str, bool]]:
        if len(value) <= EMBED_FIELD_VALUE_LIMIT:
            yield name, value, inline
            return
        start = 0
        part = 0
        total = len(value)
        while start < total:
            end = start + EMBED_FIELD_VALUE_LIMIT
            chunk_value = value[start:end]
            chunk_name = name if part == 0 else f"{name} (cont. {part + 1})"
            yield chunk_name, chunk_value, inline
            start = end
            part += 1

    def _paginate_embed_fields(
        self,
        base_embed: discord.Embed,
        fields: Sequence[Tuple[str, str, bool]],
    ) -> List[discord.Embed]:
        pages: List[discord.Embed] = []
        current = self._clone_embed(base_embed)
        current_length = len(current)
        for name, value, inline in fields:
            for chunk_name, chunk_value, chunk_inline in self._iter_field_chunks(name, value, inline):
                chunk_len = len(chunk_name) + len(chunk_value)
                if current.fields and (
                    len(current.fields) >= EMBED_MAX_FIELDS or current_length + chunk_len > EMBED_TOTAL_CHAR_LIMIT
                ):
                    pages.append(current)
                    current = self._clone_embed(base_embed)
                    current_length = len(current)
                current.add_field(name=chunk_name, value=chunk_value, inline=chunk_inline)
                current_length += chunk_len
        if current.fields or not pages:
            pages.append(current)
        return pages

    # Starter pack & rolls ---------------------------------------------

    async def _grant_starter_pack(self, message: discord.Message, profile: GachaProfile) -> None:
        starter_character = await self._starter_roll_character()
        if starter_character is None:
            logger.warning("Starter roll failed; catalog empty.")
            return
        starter_outfit = await self._starter_roll_outfit(starter_character)
        new_profile = profile
        new_profile.rudy_coins += self.starting_coins
        new_profile.starter_granted = True
        new_profile.equipped_character = starter_character.display_name
        new_profile.equipped_outfit = starter_outfit.key if starter_outfit else None

        await self._add_character(
            message.guild.id,
            message.author.id,
            starter_character.display_name,
            starter_character.rarity,
        )
        if starter_outfit:
            await self._add_outfit(
                message.guild.id,
                message.author.id,
                starter_character.display_name,
                starter_outfit.key,
                starter_outfit.rarity,
            )
        await self._update_profile(new_profile)
        # Profile is updated in-place by _grant_starter_pack (new_profile = profile creates a reference)
        # No need to refetch - the profile parameter already has all updates

        rarity_title = (starter_character.rarity or "common").title()
        announcement_lines = [
            f":sparkles: {message.author.mention} enters the game with a **{rarity_title}** character: **{starter_character.display_name}**!",
        ]
        if starter_outfit:
            announcement_lines.append(f"They start with the `{starter_outfit.label}` combo.")
        announcement_lines.append(f"(+{self.starting_coins} Rudy coins)")
        announcement_lines.append("Use `!help` for commands and next steps.")
        announcement = "\n".join(announcement_lines)

        owned_map = await self._list_owned_outfits(
            message.guild.id,
            message.author.id,
            starter_character.display_name,
        )
        owned_for_character = owned_map.get(starter_character.display_name, {})
        star_count = self._calculate_star_rating_from_owned(starter_character, owned_for_character)

        outfit_label = None
        if starter_outfit:
            outfit_label = starter_outfit.label or starter_outfit.outfit or starter_outfit.key

        embed_description = f"{message.author.display_name} joins the game!"
        await self._send_avatar_announcement(
            message.channel,
            announcement,
            user_id=message.author.id,
            author_name=message.author.display_name,
            character=starter_character,
            outfit_key=starter_outfit.key if starter_outfit else None,
            description=embed_description,
            title="New Challenger!",
            star_count=star_count,
            rudy_balance=profile.rudy_coins,
            frog_balance=profile.frog_coins,
            outfit_label=outfit_label,
            pose_override=starter_outfit.pose if starter_outfit else None,
            outfit_override=starter_outfit.outfit if starter_outfit else None,
        )

    async def _starter_roll_character(self) -> Optional[GachaCharacterDef]:
        candidates = [
            char
            for char in self._available_characters()
            if char.rarity in self._starter_rarities
            and (not self._starter_characters or char.display_name in self._starter_characters)
        ]
        if not candidates:
            candidates = list(self._available_characters())
        return random.choice(candidates) if candidates else None

    async def _starter_roll_outfit(self, character: GachaCharacterDef) -> Optional[GachaOutfitDef]:
        available = list(character.outfits.values())
        if not available:
            return None
        override = self._starter_outfits.get(character.display_name.lower())
        if override:
            normalized_override = [_normalize_key(item) for item in override]
            filtered = [
                outfit
                for outfit in available
                if outfit.outfit in override
                or _normalize_key(outfit.outfit) in normalized_override
            ]
            if filtered:
                available = filtered
        filtered = [out for out in available if out.rarity in self._starter_rarities]
        if filtered:
            available = filtered
        return random.choice(available)

    async def _perform_character_roll(
        self,
        guild_id: int,
        user_id: int,
        profile: GachaProfile,
    ) -> Tuple[GachaCharacterDef, Optional[GachaOutfitDef], bool, bool]:
        # Get the list of all characters available in the gacha catalog.
        catalog = self._available_characters()

        # Choose one character at random, weighted by rarity probabilities.
        # Rarer characters have lower base odds unless boosted by the user's profile.
        choice = self._weighted_choice(catalog, rarity_getter=lambda item: item.rarity, profile=profile)
        if choice is None:
            raise RuntimeError("No characters available for gacha roll.")

        # Get all outfits the player currently owns for this chosen character.
        owned_map = await self._list_owned_outfits(guild_id, user_id, choice.display_name)
        owned_for_character = owned_map.get(choice.display_name, {})

        # Select one outfit to award with this character roll.
        # Prefers unowned outfits and defaults to a “common” one if available.
        awarded_outfit = self._select_awarded_outfit(choice, owned_for_character)

        # Try adding the character to the player’s collection.
        # Returns True if it’s new, False if the player already had it.
        is_new_character = await self._add_character(guild_id, user_id, choice.display_name, choice.rarity)
        if not is_new_character:
            # Duplicate character → reward frog coins as compensation.
            profile.frog_coins += self.duplicate_character_frog

        outfit_new = False
        if awarded_outfit is not None:
            # Try adding the outfit to the player’s inventory.
            # Returns True if it’s new, False if it’s a duplicate.
            outfit_new = await self._add_outfit(
                guild_id,
                user_id,
                choice.display_name,
                awarded_outfit.key,
                awarded_outfit.rarity,
            )

        # Consume a “boosted roll” if the player had an active frog boost.
        self._consume_boost(profile)

        # Update the player’s profile in the database with new coins/boosts etc.
        await self._update_profile(profile)

        # Return:
        #   - the character rolled
        #   - the outfit awarded
        #   - whether the character was new
        #   - whether the outfit was new
        return choice, awarded_outfit, is_new_character, outfit_new

    async def _perform_outfit_roll(
        self,
        guild_id: int,
        user_id: int,
        profile: GachaProfile,
    ) -> Tuple[GachaCharacterDef, GachaOutfitDef, bool]:
        entries: List[Tuple[GachaCharacterDef, GachaOutfitDef]] = []
        for character in self._available_characters():
            for outfit in character.outfits.values():
                entries.append((character, outfit))
        if not entries:
            raise RuntimeError("No outfits available for gacha roll.")

        selected = self._weighted_choice(entries, rarity_getter=lambda item: item[1].rarity, profile=profile)
        if selected is None:
            raise RuntimeError("Outfit roll failed.")
        character, outfit = selected

        is_new = await self._add_outfit(
            guild_id,
            user_id,
            character.display_name,
            outfit.key,
            outfit.rarity,
        )
        if not is_new:
            profile.frog_coins += self.duplicate_outfit_frog

        self._consume_boost(profile)
        await self._update_profile(profile)
        return character, outfit, is_new

    # Command plumbing -------------------------------------------------

    def _validate_gacha_channel(self, ctx: commands.Context) -> bool:
        return (
            isinstance(ctx.channel, discord.TextChannel)
            and ctx.channel.id == self.channel_id
            and self._is_authorized_guild(ctx.guild)
        )

    def _register_command(self, command: commands.Command) -> None:
        existing = self.bot.get_command(command.name)
        if existing:
            self.bot.remove_command(existing.name)
        self.bot.add_command(command)

    def register_commands(self) -> None:
        existing_help = self.bot.get_command("help")
        if existing_help:
            self.bot.remove_command(existing_help.name)

        @commands.command(name="help")
        async def gacha_help(ctx: commands.Context) -> None:
            await self.command_help(ctx)

        @commands.command(name="gacha")
        async def gacha_status(ctx: commands.Context) -> None:
            await self.command_show_inventory(ctx)

        @commands.command(name="changeto")
        async def gacha_changeto(ctx: commands.Context, *, selection: str = "") -> None:
            await self.command_change_to(ctx, selection.strip())

        @commands.command(name="unequip")
        async def gacha_unequip(ctx: commands.Context) -> None:
            await self.command_unequip(ctx)

        @commands.command(name="frogtrade")
        async def gacha_frogtrade(ctx: commands.Context, amount: Optional[int] = None) -> None:
            await self.command_frogtrade(ctx, amount)

        @commands.command(name="givecoins")
        async def gacha_givecoins(
            ctx: commands.Context,
            member: Optional[discord.Member] = None,
            amount: Optional[int] = None,
        ) -> None:
            await self.command_give_coins(ctx, member, amount)

        @commands.command(name="givecharacter")
        async def gacha_givecharacter(
            ctx: commands.Context,
            member: Optional[discord.Member] = None,
            *,  # allow multi-word names
            character_name: str = "",
        ) -> None:
            await self.command_give_character(ctx, member, character_name)

        @commands.command(name="frogboost")
        async def gacha_frogboost(ctx: commands.Context) -> None:
            await self.command_frogboost(ctx)

        @commands.command(name="roster")
        async def gacha_roster(ctx: commands.Context) -> None:
            await self.command_roster(ctx)

        @commands.command(name="gachareset")
        async def gacha_reset(ctx: commands.Context) -> None:
            await self.command_reset(ctx)

        self._register_command(gacha_help)
        self._register_command(gacha_status)
        self._register_command(gacha_changeto)
        self._register_command(gacha_unequip)
        self._register_command(gacha_frogtrade)
        self._register_command(gacha_givecoins)
        self._register_command(gacha_givecharacter)
        self._register_command(gacha_frogboost)
        self._register_command(gacha_roster)
        self._register_command(gacha_reset)

    async def command_help(self, ctx: commands.Context) -> None:
        channel_mention = f"<#{self.channel_id}>"
        lines = [
            "**Gacha Commands**",
            "Use these in the gacha channel unless a command notes DM support.",
            "",
            "- `!help` - Show this list.",
            f"- `!gacha` - View your unlocked characters/outfits (works in DMs or {channel_mention}).",
            "- `!changeto <character> [outfit]` - Equip a character and optional outfit (channel only).",
            "- `!unequip` - Go back to your normal self.",
            "- `!roll character` - Spend Rudy coins to roll a new character (includes a default outfit).",
            "- `!roll outfit` - Roll a new outfit (any character).",
            "- `!frogtrade <amount>` - Convert frog coins into Rudy coins.",
            "- `!frogboost` - Spend frog coins to boost rare pull odds for a few rolls.",
            "- `!givecoins @user [amount]` - (Admins) Grant Rudy coins to a player.",
            "- `!givecharacter @user <name>` - (Admins) Grant a character plus a common outfit if available.",
        ]
        if self._rp_forum_post_id:
            rp_channel = f"<#{self._rp_forum_post_id}>"
            lines.extend(
                [
                    "",
                    "**RP Thread Commands**",
                    f"Use these inside {rp_channel} (only the assigned DM/owner has access).",
                    "- `!dm` — Show or assign the Dungeon Master for the RP thread.",
                    "- `!n <text>` — Narrator shortcut for `!say narrator <text>`.",
                    "- `!b <text>` — Syn's Ball shortcut for `!say ball <text>`.",
                    "- `!r <target> [forced]` — Reroll a player (seeding their first TF if needed).",
                    "- `!rename <target> <new name>` — Override how a player's VN panel name appears.",
                ]
            )
        message = "\n".join(lines)
        try:
            await ctx.message.delete()
        except discord.HTTPException:
            pass
        try:
            await ctx.author.send(message)
        except discord.Forbidden:
            if ctx.guild:
                await ctx.send("I couldn't DM you the help text. Please enable DMs so I can message you.", delete_after=10)
            else:
                await ctx.send(message)

    async def command_roster(self, ctx: commands.Context) -> None:
        """DM a roster of all characters, showing owned entries first."""
        in_dm = ctx.guild is None
        channel = await self._ensure_gacha_channel()
        if channel is None:
            await ctx.reply("I can't reach the gacha channel right now. Please try again later.", mention_author=False)
            return

        guild = ctx.guild or channel.guild

        profile = await self._fetch_profile(guild.id, ctx.author.id)
        owned_map = await self._list_owned_characters(guild.id, ctx.author.id)
        owned_names = sorted(owned_map.keys(), key=str.lower)
        owned_lines: List[str] = [
            f"{name} ({owned_map[name].title()})" for name in owned_names
        ]
        if not owned_lines:
            owned_lines.append("*You haven't unlocked any characters yet.*")

        owned_lower = {name.lower() for name in owned_names}
        # Build a stable, deduplicated list of catalog characters.
        catalog_defs = sorted(self._catalog.values(), key=lambda item: item.display_name.lower())
        unowned_lines: List[str] = []
        seen_unowned: set[str] = set()
        for entry in catalog_defs:
            normalized = entry.display_name.lower()
            if normalized in owned_lower or normalized in seen_unowned:
                continue
            seen_unowned.add(normalized)
            rarity_label = entry.rarity.title() if entry.rarity else "Common"
            unowned_lines.append(f"{entry.display_name} ({rarity_label})")
        if not unowned_lines:
            unowned_lines.append("*All available characters unlocked!*")

        embed = discord.Embed(
            title="Character Roster",
            description=(
                f"Rudy Coins: **{profile.rudy_coins}**\n"
                f"Frog Coins: **{profile.frog_coins}**"
            ),
            color=0x2980B9,
            timestamp=utc_now(),
        )
        fields: List[Tuple[str, str, bool]] = []

        for index, chunk in enumerate(self._chunk_lines(owned_lines), start=1):
            title = "Owned Characters" if index == 1 else f"Owned Characters (cont. {index})"
            fields.append((title, chunk, False))

        for index, chunk in enumerate(self._chunk_lines(unowned_lines), start=1):
            title = "Unowned Characters" if index == 1 else f"Unowned Characters (cont. {index})"
            fields.append((title, chunk, False))

        pages = self._paginate_embed_fields(embed, fields)

        try:
            await ctx.message.delete()
        except discord.HTTPException:
            pass

        try:
            send_fn = ctx.send if in_dm else ctx.author.send
            for page in pages:
                await send_fn(embed=page)
        except discord.Forbidden:
            if ctx.guild:
                await ctx.reply(
                    "I couldn't DM you your roster. Please enable DMs from server members so I can message you.",
                    mention_author=False,
                )
            else:
                await ctx.send("I couldn't deliver the roster DM. Please check your privacy settings.")

    async def command_show_inventory(self, ctx: commands.Context) -> None:
        in_dm = ctx.guild is None
        channel = await self._ensure_gacha_channel()
        if channel is None:
            await ctx.reply("I can't reach the gacha channel right now. Please try again later.", mention_author=False)
            return

        guild = ctx.guild or channel.guild

        profile = await self._fetch_profile(guild.id, ctx.author.id)
        owned_characters = await self._list_owned_characters(guild.id, ctx.author.id)
        owned_outfits = await self._list_owned_outfits(guild.id, ctx.author.id)
        embed = discord.Embed(
            title="Gacha Inventory",
            color=0x9B59B6,
            timestamp=utc_now(),
        )
        fields: List[Tuple[str, str, bool]] = []
        fields.append(
            (
                "Balances",
                (
                    f"Rudy Coins: **{profile.rudy_coins}**\n"
                    f"Frog Coins: **{profile.frog_coins}**\n"
                    f"Boost Rolls Remaining: **{profile.boost_rolls_remaining}**"
                ),
                False,
            )
        )
        equipped_label = "None"
        if profile.equipped_character:
            char_def = self._lookup_character(profile.equipped_character)
            if char_def and profile.equipped_outfit:
                combo = char_def.outfits.get(profile.equipped_outfit)
                if combo:
                    equipped_label = combo.label
                else:
                    equipped_label = profile.equipped_outfit
        fields.append(
            (
                "Equipped",
                f"{profile.equipped_character or 'None'} — {equipped_label}",
                False,
            )
        )
        fields.append(
            (
                "How to Play",
                (
                    f"• Use `!gacha` here in <#{self.channel_id}> to refresh this DM.\n"
                    f"• Equip a form with `!changeto <character> [pose/outfit]` (run it in <#{self.channel_id}>).\n"
                    "• Use `!unequip` any time (here or via DM) to return to your normal appearance.\n"
                    "• Equipped forms protect you from random TF rolls in the normal channel until you `!unequip`.\n"
                    "• Try `!roll character` or `!roll outfit` in this channel or DM me; DM rolls will be announced back in the gacha channel.\n"
                    f"• Chat as your equipped character in <#{self.channel_id}> with 3+ word sentences to earn {self.coin_earn_per_message} Rudy coins each time."
                ),
                False,
            )
        )

        lines: List[str] = []
        for character_name, rarity in owned_characters.items():
            outfits = owned_outfits.get(character_name, {})
            char_def = self._lookup_character(character_name)
            labels: List[str] = []
            for key, outfit_rarity in outfits.items():
                combo = char_def.outfits.get(key) if char_def else None
                combo_label = combo.label if combo else key
                labels.append(f"{combo_label} [{outfit_rarity.title()}]")
            outfit_list = ", ".join(sorted(labels, key=str.lower)) if labels else "None"
            lines.append(f"**{character_name}** ({rarity.title()}) — {outfit_list}")

        if not lines:
            lines.append("*No characters unlocked yet.*")

        for index, chunk in enumerate(self._chunk_lines(lines), start=1):
            title = "Characters" if index == 1 else f"Characters (cont. {index})"
            fields.append((title, chunk, False))

        pages = self._paginate_embed_fields(embed, fields)

        try:
            await ctx.message.delete()
        except discord.HTTPException:
            pass

        try:
            send_fn = ctx.send if in_dm else ctx.author.send
            for page in pages:
                await send_fn(embed=page)
        except discord.Forbidden:
            await ctx.reply(
                "I couldn't send you a DM. Please enable DMs from server members to receive your inventory.",
                mention_author=False,
            )

    async def command_change_to(self, ctx: commands.Context, selection: str) -> None:
        in_dm = ctx.guild is None
        if not in_dm and not self._validate_gacha_channel(ctx):
            await ctx.reply(
                f"Use this command inside <#{self.channel_id}> or in a DM with me.",
                mention_author=False,
            )
            return

        channel = await self._ensure_gacha_channel()
        if channel is None:
            await ctx.reply("I can't reach the gacha channel right now. Please try again later.", mention_author=False)
            return

        guild = ctx.guild or channel.guild
        if not selection:
            await ctx.reply("Usage: `!changeto <character name> [outfit]`", mention_author=False)
            return

        profile = await self._fetch_profile(guild.id, ctx.author.id)
        owned_characters = await self._list_owned_characters(guild.id, ctx.author.id)
        owned_outfits = await self._list_owned_outfits(guild.id, ctx.author.id)

        character_name, outfit_query = self._parse_change_selection(selection)
        if character_name is None:
            await ctx.reply("I couldn't match that character name.", mention_author=False)
            return
        character_def = self._lookup_character(character_name)
        if character_def is None:
            await ctx.reply("That character isn't available right now.", mention_author=False)
            return
        display_name = character_def.display_name

        if display_name not in owned_characters:
            await ctx.reply(f"You haven't unlocked **{display_name}** yet.", mention_author=False)
            return

        owned_for_character = owned_outfits.get(display_name, {})
        selected_key: Optional[str] = None

        if outfit_query:
            selected_key = self._match_outfit_query(character_def, outfit_query)
            if selected_key is None:
                options = ", ".join(sorted(combo.label for combo in character_def.outfits.values()))
                await ctx.reply(
                    "I couldn't match that pose/outfit combo. Try one of: "
                    f"{options or 'no outfits available'}",
                    mention_author=False,
                )
                return
            if selected_key not in owned_for_character:
                await ctx.reply(
                    f"You don't own the `{character_def.outfits[selected_key].label}` combo for **{display_name}**.",
                    mention_author=False,
                )
                return
        else:
            if owned_for_character:
                if (
                    profile.equipped_character == display_name
                    and profile.equipped_outfit in owned_for_character
                ):
                    selected_key = profile.equipped_outfit
                else:
                    def _sort_key(candidate: str) -> str:
                        combo = character_def.outfits.get(candidate)
                        return combo.label.lower() if combo else candidate.lower()

                    selected_key = sorted(owned_for_character.keys(), key=_sort_key)[0]

        profile.equipped_character = display_name
        profile.equipped_outfit = selected_key
        await self._update_profile(profile)

        combo_label = (
            character_def.outfits[selected_key].label
            if selected_key and selected_key in character_def.outfits
            else "default"
        )
        response_target = ctx
        if in_dm:
            await channel.send(
                f"{ctx.author.mention} equipped **{display_name}** with `{combo_label}`."
            )
        await response_target.reply(
            f"Equipped **{display_name}** with `{combo_label}`.",
            mention_author=False,
        )

    async def command_unequip(self, ctx: commands.Context) -> None:
        in_dm = ctx.guild is None
        if not in_dm and not self._validate_gacha_channel(ctx):
            await ctx.reply(
                f"Use this command inside <#{self.channel_id}> or in a DM with me.",
                mention_author=False,
            )
            return

        channel = await self._ensure_gacha_channel()
        if channel is None:
            await ctx.reply("I can't reach the gacha channel right now. Please try again later.", mention_author=False)
            return

        guild = ctx.guild or channel.guild
        profile = await self._fetch_profile(guild.id, ctx.author.id)
        if not profile.equipped_character:
            await ctx.reply("You are not currently transformed.", mention_author=False)
            return

        profile.equipped_character = None
        profile.equipped_outfit = None
        await self._update_profile(profile)

        farewell = "You revert to your original self."
        await ctx.reply(farewell, mention_author=False)
        if in_dm:
            await channel.send(f"{ctx.author.mention} reverted to their original self.")

    async def command_roll(self, ctx: commands.Context, roll_type: str, _extra: str) -> None:
        in_dm = ctx.guild is None
        if not in_dm and not self._validate_gacha_channel(ctx):
            await ctx.reply(
                f"Rolls must happen in <#{self.channel_id}> or in a DM with me.",
                mention_author=False,
            )
            return

        channel = await self._ensure_gacha_channel()
        if channel is None:
            await ctx.reply("I can't reach the gacha channel right now. Please try again later.", mention_author=False)
            return

        guild = ctx.guild or channel.guild

        roll_type = roll_type or ""
        if roll_type not in {"character", "outfit"}:
            await ctx.reply("Usage: `!roll character` or `!roll outfit`", mention_author=False)
            return

        profile = await self._fetch_profile(guild.id, ctx.author.id)
        if roll_type == "character":
            if profile.rudy_coins < self.character_roll_cost:
                await ctx.reply("Not enough Rudy coins for a character roll.", mention_author=False)
                return
            profile.rudy_coins -= self.character_roll_cost
            choice, awarded_outfit, char_new, outfit_new = await self._perform_character_roll(
                guild.id, ctx.author.id, profile
            )
            outfit_label = None
            if awarded_outfit:
                outfit_label = awarded_outfit.label or awarded_outfit.outfit or awarded_outfit.key

            panel_text = f"Won a {choice.rarity.title()} character: {choice.display_name}!"
            if outfit_label:
                panel_text += f"\nUnlocked outfit: `{outfit_label}`"

            result_message = (
                f":game_die: {ctx.author.mention} rolled a {choice.rarity.title()} character: **{choice.display_name}**!"
            )
            extras: List[str] = []
            if outfit_label:
                if outfit_new:
                    extras.append(f"outfit `{outfit_label}`")
                else:
                    extras.append(f"outfit `{outfit_label}` (duplicate)")
            if not char_new:
                extras.append(
                    f"+{self.duplicate_character_frog} frog coin{'s' if self.duplicate_character_frog != 1 else ''} (duplicate)"
                )
            if extras:
                result_message += " " + "; ".join(extras)

            owned_map = await self._list_owned_outfits(guild.id, ctx.author.id, choice.display_name)
            owned_for_character = owned_map.get(choice.display_name, {})
            star_count = self._calculate_star_rating_from_owned(choice, owned_for_character)
            panel_outfit_key = awarded_outfit.key if awarded_outfit else None
            panel_pose_override = awarded_outfit.pose if awarded_outfit else None
            panel_outfit_override = awarded_outfit.outfit if awarded_outfit else None
            embed_kwargs = dict(
                user_id=ctx.author.id,
                author_name=ctx.author.display_name,
                character=choice,
                outfit_key=panel_outfit_key,
                description=panel_text,
                title="Character Roll",
                star_count=star_count,
                rudy_balance=profile.rudy_coins,
                frog_balance=profile.frog_coins,
                outfit_label=outfit_label,
                pose_override=panel_pose_override,
                outfit_override=panel_outfit_override,
            )
            if in_dm:
                await self._send_avatar_announcement(ctx, result_message, **embed_kwargs)
                await self._send_avatar_announcement(channel, result_message, **embed_kwargs)
            else:
                await self._send_avatar_announcement(ctx, result_message, **embed_kwargs)
        else:
            if profile.rudy_coins < self.outfit_roll_cost:
                await ctx.reply("Not enough Rudy coins for an outfit roll.", mention_author=False)
                return
            profile.rudy_coins -= self.outfit_roll_cost
            character_def, outfit_choice, outfit_new = await self._perform_outfit_roll(
                guild.id, ctx.author.id, profile
            )
            outfit_label = outfit_choice.label or outfit_choice.outfit or outfit_choice.key
            panel_text = f"Won `{outfit_label}` for {character_def.display_name}!"
            result_message = (
                f":game_die: {ctx.author.mention} rolled `{outfit_label}` for **{character_def.display_name}**!"
            )
            extras: List[str] = []
            if not outfit_new:
                extras.append(
                    f"+{self.duplicate_outfit_frog} frog coin{'s' if self.duplicate_outfit_frog != 1 else ''} (duplicate outfit)"
                )
            if extras:
                result_message += " " + "; ".join(extras)
            owned_map = await self._list_owned_outfits(guild.id, ctx.author.id, character_def.display_name)
            owned_for_character = owned_map.get(character_def.display_name, {})
            star_count = self._calculate_star_rating_from_owned(character_def, owned_for_character)
            panel_pose_override = outfit_choice.pose
            panel_outfit_override = outfit_choice.outfit
            embed_kwargs = dict(
                user_id=ctx.author.id,
                author_name=ctx.author.display_name,
                character=character_def,
                outfit_key=outfit_choice.key,
                description=panel_text,
                title="Outfit Roll",
                star_count=star_count,
                rudy_balance=profile.rudy_coins,
                frog_balance=profile.frog_coins,
                outfit_label=outfit_label,
                pose_override=panel_pose_override,
                outfit_override=panel_outfit_override,
            )
            if in_dm:
                await self._send_avatar_announcement(ctx, result_message, **embed_kwargs)
                await self._send_avatar_announcement(channel, result_message, **embed_kwargs)
            else:
                await self._send_avatar_announcement(ctx, result_message, **embed_kwargs)

    async def command_frogtrade(self, ctx: commands.Context, amount: Optional[int]) -> None:
        if ctx.guild is None or not self._validate_gacha_channel(ctx):
            await ctx.reply(
                f"Frog trades must happen in <#{self.channel_id}>.",
                mention_author=False,
            )
            return
        qty = amount or 1
        if qty <= 0:
            await ctx.reply("Trade amount must be positive.", mention_author=False)
            return

        profile = await self._fetch_profile(ctx.guild.id, ctx.author.id)
        if profile.frog_coins < qty:
            await ctx.reply("You don't have that many frog coins.", mention_author=False)
            return
        profile.frog_coins -= qty
        gained = qty * self.frog_to_rudy_rate
        profile.rudy_coins += gained
        await self._update_profile(profile)
        await ctx.reply(
            f"Converted {qty} frog coin{'s' if qty != 1 else ''} into {gained} Rudy coins.",
            mention_author=False,
        )

    async def command_frogboost(self, ctx: commands.Context) -> None:
        if ctx.guild is None or not self._validate_gacha_channel(ctx):
            await ctx.reply(
                f"Use this command inside <#{self.channel_id}>.",
                mention_author=False,
            )
            return
        profile = await self._fetch_profile(ctx.guild.id, ctx.author.id)
        if profile.frog_coins < self.frog_boost_cost:
            await ctx.reply("Not enough frog coins for a boost.", mention_author=False)
            return
        profile.frog_coins -= self.frog_boost_cost
        profile.boost_rolls_remaining += self.frog_boost_rolls
        profile.boost_bonus += max(self.frog_boost_bonus, 0.0)
        await self._update_profile(profile)
        await ctx.reply(
            f"Boost activated! Next {self.frog_boost_rolls} roll{'s' if self.frog_boost_rolls != 1 else ''} gain +{int(self.frog_boost_bonus * 100)}% rare odds.",
            mention_author=False,
        )

    async def command_give_coins(
        self,
        ctx: commands.Context,
        member: Optional[discord.Member],
        amount: Optional[int],
    ) -> None:
        if ctx.guild is None:
            await ctx.reply("Run this command inside the server.", mention_author=False)
            return
        if member is None:
            await ctx.reply("Usage: `!givecoins @user [amount]`", mention_author=False)
            return
        invoker = ctx.author if isinstance(ctx.author, discord.Member) else None
        if invoker is None or not is_admin(invoker):
            await ctx.reply("Only a server administrator can give Rudy coins manually.", mention_author=False)
            return
        if member.bot:
            await ctx.reply("Bots don't need Rudy coins.", mention_author=False)
            return

        qty = amount if amount is not None else self.character_roll_cost
        if qty <= 0:
            await ctx.reply("The coin amount must be a positive number.", mention_author=False)
            return

        profile = await self._fetch_profile(ctx.guild.id, member.id)
        profile.rudy_coins += qty
        await self._update_profile(profile)

        await ctx.reply(
            f"Granted {qty} Rudy coins to {member.mention}.",
            mention_author=False,
        )
        if member.id != ctx.author.id:
            try:
                await member.send(
                    f"You received {qty} Rudy coins from a server admin in {ctx.guild.name}."
                )
            except discord.Forbidden:
                pass

    async def command_give_character(
        self,
        ctx: commands.Context,
        member: Optional[discord.Member],
        character_query: str,
    ) -> None:
        if ctx.guild is None:
            await ctx.reply("Run this command inside the server.", mention_author=False)
            return
        if member is None or not character_query.strip():
            await ctx.reply("Usage: `!givecharacter @user <character name>`", mention_author=False)
            return
        invoker = ctx.author if isinstance(ctx.author, discord.Member) else None
        if invoker is None or not is_admin(invoker):
            await ctx.reply("Only a server administrator can grant characters manually.", mention_author=False)
            return
        if member.bot:
            await ctx.reply("Bots don't need gacha characters.", mention_author=False)
            return

        character = self._lookup_character(character_query.strip())
        if character is None:
            await ctx.reply(f"I couldn't find a character matching `{character_query}`.", mention_author=False)
            return

        added_character = await self._add_character(
            ctx.guild.id,
            member.id,
            character.display_name,
            character.rarity,
        )

        awarded_outfit: Optional[GachaOutfitDef] = None
        common_outfits = [
            outfit
            for outfit in character.outfits.values()
            if (outfit.rarity or "common").lower() == "common"
        ]
        if common_outfits:
            owned_map = await self._list_owned_outfits(ctx.guild.id, member.id, character.display_name)
            owned_for_character = owned_map.get(character.display_name, {})
            common_outfits.sort(key=lambda combo: combo.label.lower())
            for outfit in common_outfits:
                if outfit.key in owned_for_character:
                    continue
                outfit_added = await self._add_outfit(
                    ctx.guild.id,
                    member.id,
                    character.display_name,
                    outfit.key,
                    outfit.rarity,
                )
                if outfit_added:
                    awarded_outfit = outfit
                    break

        lines = []
        rarity_label = (character.rarity or "common").title()
        if added_character:
            lines.append(
                f"Granted **{character.display_name}** ({rarity_label}) to {member.mention}."
            )
        else:
            lines.append(
                f"{member.mention} already owns **{character.display_name}**, but the request was processed."
            )
        if awarded_outfit:
            lines.append(f"Also granted outfit **{awarded_outfit.label}**.")
        elif common_outfits:
            lines.append("They already own all common outfits for that character.")
        else:
            lines.append("That character has no common outfits to grant.")

        await ctx.reply("\n".join(lines), mention_author=False)
        if member.id != ctx.author.id:
            try:
                await member.send(
                    f"You received **{character.display_name}** from an admin in {ctx.guild.name}."
                )
            except discord.Forbidden:
                pass

    async def command_reset(self, ctx: commands.Context) -> None:
        if ctx.guild is None:
            await ctx.reply("This command can only be used inside the server.", mention_author=False)
            return
        if ctx.author.id != ctx.guild.owner_id:
            await ctx.reply("Only the server owner can reset the gacha data.", mention_author=False)
            return

        try:
            await ctx.message.delete()
        except discord.HTTPException:
            pass

        channel = await self._ensure_gacha_channel()
        purged = 0
        if channel is not None:
            def not_pinned(message: discord.Message) -> bool:
                return not message.pinned

            try:
                while True:
                    batch = await channel.purge(limit=100, check=not_pinned)
                    purged += len(batch)
                    if len(batch) < 100:
                        break
                    await asyncio.sleep(1)
            except discord.HTTPException:
                pass

        await self._reset_database()

        self._spam_tracker.clear()

        summary = (
            f"Gacha data reset complete.\n"
            f"- Messages purged: {purged}\n"
            "- Database recreated."
        )
        try:
            await ctx.author.send(summary)
        except discord.Forbidden:
            await ctx.send(summary, mention_author=False)

    # Selection parsing ------------------------------------------------

    def _parse_change_selection(self, selection: str) -> Tuple[Optional[str], Optional[str]]:
        selection = selection.strip()
        if not selection:
            return None, None
        candidates = sorted((char.display_name for char in self._catalog.values()), key=len, reverse=True)
        lower = selection.lower()
        for candidate in candidates:
            cand_lower = candidate.lower()
            if lower == cand_lower:
                return candidate, None
            if lower.startswith(cand_lower + " "):
                remainder = selection[len(candidate) :].strip()
                return candidate, remainder if remainder else None
        normalized = _normalize_key(selection)
        for candidate in candidates:
            if _normalize_key(candidate) == normalized:
                return candidate, None
        tokens = selection.split()
        if tokens:
            first_word = tokens[0].lower()
            remainder = selection[len(tokens[0]) :].strip()
            first_matches = [
                candidate for candidate in candidates if candidate.lower().split()[0] == first_word
            ]
            if len(first_matches) == 1:
                return first_matches[0], remainder if remainder else None
        # Allow slug search (folder name).
        for char in self._catalog.values():
            if normalized == char.slug:
                remainder = ""
                return char.display_name, remainder if remainder else None
        return None, None


def setup_gacha_mode(
    bot: commands.Bot,
    *,
    character_pool: Sequence[TFCharacter],
    relay_fn: Callable[..., asyncio.Future],
) -> GachaManager:
    """Factory used by bot.py to bootstrap gacha mode."""
    config_path = path_from_env("TFBOT_GACHA_CONFIG")
    manager = GachaManager(
        bot=bot,
        character_pool=character_pool,
        relay_fn=relay_fn,
        config_path=config_path,
    )
    manager.register_commands()
    return manager


__all__ = ["GachaManager", "setup_gacha_mode"]
