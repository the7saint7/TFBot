import asyncio
import importlib.util
import io
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import traceback
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from functools import lru_cache, wraps
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING, Union

import aiohttp
import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv

load_dotenv()

try:
    import yaml
except ImportError:
    yaml = None

try:
    import PIL  # type: ignore
except ImportError:
    PIL = None

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(
    level=os.getenv("TFBOT_LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tfbot")
if PIL is not None:
    logging.getLogger("PIL").setLevel(logging.ERROR)


def _resolve_vn_cache_dir() -> Optional[Path]:
    cache_setting = os.getenv("TFBOT_VN_CACHE_DIR", "vn_cache").strip()
    if not cache_setting:
        return None
    cache_path = Path(cache_setting)
    if not cache_path.is_absolute():
        return (BASE_DIR / cache_path).resolve()
    return cache_path.resolve()


def _clear_vn_cache_directory(startup_logger: logging.Logger) -> None:
    cache_dir = _resolve_vn_cache_dir()
    if not cache_dir or not cache_dir.exists():
        return
    try:
        shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        startup_logger.info("VN cache cleared at %s.", cache_dir)
    except OSError as exc:
        startup_logger.warning("Unable to clear VN cache at %s: %s", cache_dir, exc)


def _read_git_head_sha(
    git_executable: str, repo_dir: Path, startup_logger: logging.Logger
) -> Optional[str]:
    try:
        result = subprocess.run(
            [git_executable, "-C", str(repo_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, OSError) as exc:
        startup_logger.debug("Unable to read git HEAD for %s: %s", repo_dir, exc)
        return None


def _detect_upstream_ref(
    git_executable: str, repo_dir: Path, startup_logger: logging.Logger
) -> Optional[str]:
    commands = [
        [
            git_executable,
            "-C",
            str(repo_dir),
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            "@{u}",
        ],
        [
            git_executable,
            "-C",
            str(repo_dir),
            "symbolic-ref",
            "refs/remotes/origin/HEAD",
        ],
    ]
    for cmd in commands:
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            ref = result.stdout.strip()
            if not ref:
                continue
            if ref.startswith("refs/remotes/"):
                ref = ref.split("refs/remotes/", 1)[1]
            startup_logger.debug("Detected upstream ref %s via `%s`.", ref, " ".join(cmd))
            return ref
        except subprocess.CalledProcessError as exc:
            startup_logger.debug("Unable to detect upstream ref via `%s`: %s", " ".join(cmd), exc.stderr or exc.stdout)
    return None


def _hard_sync_existing_repo(
    git_executable: str, repo_dir: Path, startup_logger: logging.Logger
) -> bool:
    upstream_ref = _detect_upstream_ref(git_executable, repo_dir, startup_logger)
    if upstream_ref is None:
        startup_logger.warning(
            "Cannot determine upstream branch for %s; skipping hard reset.",
            repo_dir,
        )
        return False
    commands = [
        [git_executable, "-C", str(repo_dir), "fetch", "--all", "--tags", "--prune"],
        [git_executable, "-C", str(repo_dir), "reset", "--hard", upstream_ref],
        [git_executable, "-C", str(repo_dir), "clean", "-fd"],
    ]
    for cmd in commands:
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            output = exc.stderr.strip() or exc.stdout.strip() or str(exc)
            startup_logger.warning(
                "Failed to run `%s` while syncing characters repo: %s",
                " ".join(cmd),
                output,
            )
            return False
    startup_logger.info(
        "Characters repo hard-reset to %s via fetch/reset/clean.",
        upstream_ref,
    )
    return True


def _sync_character_repo() -> Optional[Path]:
    repo_url = os.getenv("TFBOT_CHARACTERS_REPO", "").strip()
    if not repo_url:
        return None
    repo_dir_setting = os.getenv("TFBOT_CHARACTERS_REPO_DIR", "characters_repo").strip()
    repo_dir_setting = repo_dir_setting or "characters_repo"
    repo_dir = Path(repo_dir_setting)
    if not repo_dir.is_absolute():
        repo_dir = (BASE_DIR / repo_dir).resolve()
    subdir_setting = os.getenv("TFBOT_CHARACTERS_REPO_SUBDIR", "characters").strip()
    target_subdir = Path(subdir_setting) if subdir_setting else None
    startup_logger = logging.getLogger("tfbot.startup")
    git_executable = shutil.which("git")
    if git_executable is None:
        startup_logger.warning(
            "TFBOT_CHARACTERS_REPO is set but git is not available on PATH; skipping sprite sync."
        )
        return None
    try:
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        startup_logger.warning("Unable to prepare directory %s for character repo: %s", repo_dir, exc)
        return None

    repo_git_dir = repo_dir / ".git"
    if repo_dir.exists() and not repo_git_dir.exists():
        startup_logger.warning(
            "TFBOT_CHARACTERS_REPO destination %s exists but is not a git repository; skipping sync.",
            repo_dir,
        )
    else:
        old_head = None
        if repo_git_dir.exists():
            action = "sync"
            old_head = _read_git_head_sha(git_executable, repo_dir, startup_logger)
            synced = _hard_sync_existing_repo(git_executable, repo_dir, startup_logger)
            if not synced:
                startup_logger.warning(
                    "Characters repo at %s could not be synced; deleting and recloning.",
                    repo_dir,
                )
                try:
                    shutil.rmtree(repo_dir)
                except OSError as exc:
                    startup_logger.warning("Failed to remove %s for reclone: %s", repo_dir, exc)
                    return None
                action = "clone"
        if not repo_git_dir.exists():
            cmd = [git_executable, "clone", repo_url, str(repo_dir)]
            action = "clone"
        if action == "clone":
            startup_logger.info("Characters repo clone starting via `%s`.", " ".join(cmd))
            try:
                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                output = result.stdout.strip() or result.stderr.strip()
                if output:
                    startup_logger.info("Characters repo cloned via `%s`: %s", " ".join(cmd), output)
                else:
                    startup_logger.info("Characters repo cloned via `%s`.", " ".join(cmd))
            except subprocess.CalledProcessError as exc:
                output = exc.stderr.strip() or exc.stdout.strip() or str(exc)
                startup_logger.warning(
                    "Failed to clone characters repo %s (cmd=%s): %s",
                    repo_url,
                    " ".join(cmd),
                    output,
                )
                return None
        updated_assets = action == "clone"
        if action == "sync":
            new_head = _read_git_head_sha(git_executable, repo_dir, startup_logger)
            if old_head and new_head:
                updated_assets = old_head != new_head
            else:
                updated_assets = bool(new_head and not old_head)
        if updated_assets:
            _clear_vn_cache_directory(startup_logger)

    characters_dir = repo_dir
    if target_subdir:
        characters_dir = repo_dir / target_subdir
    if not characters_dir.exists():
        startup_logger.warning(
            "Characters repo synced, but %s does not exist inside %s.",
            target_subdir.as_posix() if target_subdir else ".",
            repo_dir,
        )
        return None
    return characters_dir


_characters_repo_path = _sync_character_repo()
if _characters_repo_path:
    os.environ["TFBOT_VN_ASSET_ROOT"] = str(_characters_repo_path)


def _resolve_character_faces_root() -> Optional[Path]:
    repo_dir_setting = os.getenv("TFBOT_CHARACTERS_REPO_DIR", "characters_repo").strip() or "characters_repo"
    repo_dir = Path(repo_dir_setting)
    if not repo_dir.is_absolute():
        repo_dir = (BASE_DIR / repo_dir).resolve()
    candidates: list[Path] = []
    if repo_dir.exists():
        candidates.append(repo_dir / "faces")
    fallback_repo = (BASE_DIR / "characters_repo").resolve()
    if fallback_repo.exists():
        candidates.append(fallback_repo / "faces")
    fallback_faces = (BASE_DIR / "faces").resolve()
    if fallback_faces.exists():
        candidates.append(fallback_faces)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


CHARACTER_FACES_ROOT = _resolve_character_faces_root()

try:
    from tf_characters import TF_CHARACTERS as _DEFAULT_CHARACTER_DATA
except ModuleNotFoundError:
    module_path = BASE_DIR / "tf_characters.py"
    if not module_path.exists():
        raise
    spec = importlib.util.spec_from_file_location("tf_characters", module_path)
    if spec is None or spec.loader is None:
        raise
    module = importlib.util.module_from_spec(spec)
    sys.modules["tf_characters"] = module
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    data = getattr(module, "TF_CHARACTERS", None)
    if not isinstance(data, list):
        raise RuntimeError("tf_characters.py does not define a TF_CHARACTERS list.")
    _DEFAULT_CHARACTER_DATA = data
from ai_rewriter import AI_REWRITE_ENABLED, rewrite_message_for_character
from tfbot.models import (
    OutfitAsset,
    ReplyContext,
    TFCharacter,
    TransformationState,
    TransformKey,
)
import tfbot.state as tf_state
from tfbot.state import (
    active_transformations,
    configure_state,
    find_active_transformation,
    get_last_reroll_timestamp,
    increment_tf_stats,
    load_reroll_cooldowns_from_disk,
    load_states_from_disk,
    load_stats_from_disk,
    persist_states,
    persist_stats,
    record_reroll_timestamp,
    reroll_cooldowns,
    tf_stats,
    revert_tasks,
    state_key,
)
from tfbot.swaps import SwapTransition, ensure_form_owner, unswap_chain
from tfbot.legacy_embed import build_legacy_embed
from tfbot.history import publish_history_snapshot
from tfbot.panels import (
    VN_BACKGROUND_ROOT,
    VN_BACKGROUND_DEFAULT_RELATIVE,
    VN_AVATAR_MODE,
    VN_AVATAR_SCALE,
    VN_ASSET_ROOT,
    apply_mention_placeholders,
    compose_game_avatar,
    fetch_avatar_bytes,
    get_accessory_states,
    get_selected_background_path,
    get_selected_outfit_name,
    get_selected_pose_outfit,
    list_character_accessories,
    set_accessory_state,
    list_available_outfits,
    list_background_choices,
    list_pose_outfits,
    parse_discord_formatting,
    prepare_custom_emoji_images,
    prepare_panel_mentions,
    prepare_reply_snippet,
    render_vn_panel,
    set_character_directory_overrides,
    set_selected_background,
    set_selected_outfit_name,
    set_selected_pose_outfit,
    strip_urls,
    toggle_accessory_state,
    vn_outfit_selection,
    persist_outfit_selections,
)
from tfbot.roleplay import RoleplayCog, add_roleplay_cog
from tfbot.interactions import InteractionContextAdapter
from tfbot.utils import (
    float_from_env,
    int_from_env,
    is_admin,
    member_profile_name,
    normalize_pose_name,
    path_from_env,
    utc_now,
)

if TYPE_CHECKING:
    from tfbot.gacha import GachaProfile
    from tfbot.games import GameBoardManager


BOT_MODE = os.getenv("TFBOT_MODE", "classic").lower()
TF_CHANNEL_ID = int_from_env("TFBOT_CHANNEL_ID", 0)
GACHA_CHANNEL_ID = int_from_env("TFBOT_GACHA_CHANNEL_ID", 0)
GACHA_ENABLED = GACHA_CHANNEL_ID > 0
CLASSIC_ENABLED = BOT_MODE != "gacha" and TF_CHANNEL_ID > 0

if BOT_MODE == "gacha" and not GACHA_ENABLED:
    raise RuntimeError("TFBOT_GACHA_CHANNEL_ID is required when running in gacha mode.")

CHARACTER_INFO_FORUM_CHANNEL_ID = int_from_env("TF_CHARACTER_INFO_FORUM_CHANNEL_ID", 0)
CHARACTER_INFO_FORUM_ENABLED = CHARACTER_INFO_FORUM_CHANNEL_ID > 0
CHARACTER_INFO_FORUM_ACTION_DELAY = 0.75
if not CLASSIC_ENABLED and not GACHA_ENABLED:
    raise RuntimeError("Configure at least TFBOT_CHANNEL_ID or TFBOT_GACHA_CHANNEL_ID.")
TF_HISTORY_CHANNEL_ID = int_from_env("TFBOT_HISTORY_CHANNEL_ID", 1432196317722972262)
TF_ARCHIVE_CHANNEL_ID = int_from_env("TFBOT_ARCHIVE_CHANNEL_ID", 0)
TFBOT_NAME = os.getenv("TFBOT_NAME", "Bot").strip() or "Bot"
TFBOT_VERSION = os.getenv("TFBOT_VERSION", "1.0.0").strip() or "1.0.0"
TF_STATE_FILE = Path(os.getenv("TFBOT_STATE_FILE", "tf_state.json"))
TF_STATS_FILE = Path(os.getenv("TFBOT_STATS_FILE", "tf_stats.json"))
TF_REROLL_FILE = Path(os.getenv("TFBOT_REROLL_FILE", "tf_reroll.json"))
ROLEPLAY_FORUM_POST_ID = int_from_env("TFBOT_RP_FORUM_POST_ID", 0)
ROLEPLAY_STATE_FILE = Path(os.getenv("TFBOT_RP_STATE_FILE", "rp_forum_state.json"))
GAME_FORUM_CHANNEL_ID = int_from_env("TFBOT_GAME_FORUM_CHANNEL_ID", 0)
GAME_DM_CHANNEL_ID = int_from_env("TFBOT_GAME_DM_CHANNEL_ID", 0)
GAME_CONFIG_FILE = Path(os.getenv("TFBOT_GAME_CONFIG_FILE", "games/game_config.json"))
GAME_ASSETS_DIR = path_from_env("TFBOT_GAME_ASSETS_DIR") or Path("games/assets")
GAME_BOARD_ENABLED = GAME_FORUM_CHANNEL_ID > 0
MESSAGE_STYLE = os.getenv(
    "TFBOT_MESSAGE_STYLE",
    "vn" if GACHA_ENABLED else "classic",
).lower()
_DEFAULT_INANIMATE_MINUTES = (10,)
_DEFAULT_SPECIAL_MINUTES = (60,)
_DEFAULT_GENERIC_MINUTES = (600,)


def _parse_duration_minutes(raw_value: Optional[str], fallback: Sequence[int]) -> Tuple[int, ...]:
    if not raw_value:
        return tuple(fallback)
    parsed: List[int] = []
    for token in re.split(r"[;,]", raw_value):
        stripped = token.strip()
        if not stripped:
            continue
        try:
            minutes_value = int(float(stripped))
        except ValueError:
            logger.warning("Ignoring invalid duration value '%s' in %s", stripped, raw_value)
            continue
        if minutes_value <= 0:
            logger.warning("Ignoring non-positive duration value '%s' in %s", stripped, raw_value)
            continue
        parsed.append(minutes_value)
    return tuple(parsed or fallback)


def _duration_options_from_env(var_name: str, fallback: Sequence[int], legacy_var: Optional[str] = None) -> Tuple[int, ...]:
    raw_value = os.getenv(var_name)
    if raw_value is None and legacy_var:
        raw_value = os.getenv(legacy_var)
    return _parse_duration_minutes(raw_value, fallback)


TF_INANIMATE_DURATION_OPTIONS = _duration_options_from_env(
    "TF_INANIMATE_DURATION",
    fallback=_DEFAULT_INANIMATE_MINUTES,
    legacy_var="TF_INNANIMATE_DURATION",
)
TF_SPECIAL_DURATION_OPTIONS = _duration_options_from_env(
    "TF_SPECIAL_DURATION",
    fallback=_DEFAULT_SPECIAL_MINUTES,
)
TF_GENERIC_DURATION_OPTIONS = _duration_options_from_env(
    "TF_GENERIC_DURATION",
    fallback=_DEFAULT_GENERIC_MINUTES,
)


def _format_duration_label_from_minutes(minutes: int) -> str:
    total_minutes = max(int(minutes), 1)
    days, remainder = divmod(total_minutes, 1440)
    hours, minutes_only = divmod(remainder, 60)
    parts: List[str] = []
    if days:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    if hours:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes_only:
        parts.append(f"{minutes_only} minute{'s' if minutes_only != 1 else ''}")
    if not parts:
        parts.append("less than a minute")
    return " ".join(parts)


def _random_duration_from_options(options: Sequence[int]) -> Tuple[str, timedelta]:
    minutes = random.choice(options)
    return _format_duration_label_from_minutes(minutes), timedelta(minutes=minutes)


def _choose_reroll_duration(is_inanimate: bool, character_token: str) -> Tuple[str, timedelta]:
    if is_inanimate:
        options = TF_INANIMATE_DURATION_OPTIONS
    elif _is_special_reroll_name(character_token):
        options = TF_SPECIAL_DURATION_OPTIONS
    else:
        options = TF_GENERIC_DURATION_OPTIONS
    return _random_duration_from_options(options)
REQUIRED_GUILD_PERMISSIONS = {
    "send_messages": "Send Messages (needed to respond in channels)",
    "embed_links": "Embed Links (history channel logging)",
}
MAGIC_EMOJI_NAME = os.getenv("TFBOT_MAGIC_EMOJI_NAME", "magic_emoji")
MAGIC_EMOJI_CACHE: Dict[int, str] = {}
def _parse_special_form_names(raw: str) -> tuple[str, ...]:
    tokens = [token.strip() for token in re.split(r"[;,]", raw or "") if token.strip()]
    if not tokens:
        return ("ball", "narrator")
    return tuple(tokens)


SPECIAL_REROLL_FORMS = _parse_special_form_names(os.getenv("TFBOT_SPECIAL_FORMS", "Ball,Narrator"))
ADMIN_ONLY_RANDOM_FORMS = ("syn", "circe")
CHARACTER_AUTOCOMPLETE_LIMIT = 25
OUTFIT_AUTOCOMPLETE_LIMIT = 25
ACCESSORY_AUTOCOMPLETE_LIMIT = 25
CHARACTER_DIRECTORY_CACHE_TTL = 120.0  # seconds


def _normalize_folder_token(value: Optional[str]) -> str:
    if not value:
        return ""
    normalized = value.strip().replace("\\", "/").strip("/").lower()
    return normalized


def _folder_lookup_tokens(value: Optional[str]) -> set[str]:
    """Return normalized folder tokens for matching user input against character folders."""
    normalized = _normalize_folder_token(value)
    if not normalized:
        return set()
    variants = {normalized}
    compact = normalized.replace(" ", "_").replace("-", "_")
    if compact:
        variants.add(compact)
    return variants


def _normalize_special_token(value: Optional[str]) -> str:
    normalized = _normalize_folder_token(value)
    if "/" in normalized:
        normalized = normalized.split("/")[-1]
    return normalized


SPECIAL_REROLL_TOKENS = {
    token for token in (_normalize_special_token(item) for item in SPECIAL_REROLL_FORMS) if token
}
CHARACTER_DIRECTORY_CACHE_TTL = 120.0  # seconds


def _format_human_list(items: Sequence[str]) -> str:
    cleaned = [entry.strip() for entry in items if str(entry).strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} or {cleaned[1]}"
    return ", ".join(cleaned[:-1]) + f", or {cleaned[-1]}"


SPECIAL_FORMS_DISPLAY = _format_human_list(SPECIAL_REROLL_FORMS)
SPECIAL_FORM_TARGET = SPECIAL_FORMS_DISPLAY or "the configured special forms"
SPECIAL_FORM_SUBJECT = (
    f"{SPECIAL_FORMS_DISPLAY} characters" if SPECIAL_FORMS_DISPLAY else "the configured special forms"
)
SPECIAL_FORM_CAPITALIZED = SPECIAL_FORM_TARGET[:1].upper() + SPECIAL_FORM_TARGET[1:] if SPECIAL_FORM_TARGET else "Special forms"
PRIVILEGED_FORM_TOKENS = set(SPECIAL_REROLL_TOKENS)


def _privileged_requirement_message(action: str) -> str:
    if SPECIAL_FORMS_DISPLAY:
        return f"Only admins or {SPECIAL_FORM_SUBJECT} can {action}."
    return f"Only admins can {action}."

def _parse_featured_weight_map(raw: str) -> Dict[str, float]:
    """Parse comma/semicolon separated token=weight entries."""
    weights: Dict[str, float] = {}
    if not raw:
        return weights
    for chunk in re.split(r"[;,]", raw):
        if not chunk:
            continue
        if "=" in chunk:
            token, weight_raw = chunk.split("=", 1)
        elif ":" in chunk:
            token, weight_raw = chunk.split(":", 1)
        else:
            continue
        normalized_token = token.strip().lower()
        if not normalized_token:
            continue
        try:
            weight = float(weight_raw.strip())
        except ValueError:
            continue
        if weight > 0:
            weights[normalized_token] = weight
    return weights


FEATURED_TF_WEIGHTS = _parse_featured_weight_map(os.getenv("TFBOT_FEATURED_TF_WEIGHTS", ""))


configure_state(state_file=TF_STATE_FILE, stats_file=TF_STATS_FILE, reroll_file=TF_REROLL_FILE)
tf_stats.update(load_stats_from_disk())
reroll_cooldowns.update(load_reroll_cooldowns_from_disk())

INANIMATE_DATA_FILE = Path(os.getenv("TFBOT_INANIMATE_FILE", "tf_inanimate.json")).expanduser()
INANIMATE_TF_CHANCE = float(os.getenv("TFBOT_INANIMATE_CHANCE", "0"))
ADMIN_PROTECTION_ENABLED = os.getenv("TFBOT_ADMIN_PROTECTION_ENABLED", "false").lower() in ("true", "1", "yes", "on")
SPECIAL_CHARACTERS_ADMIN_ONLY = os.getenv("TFBOT_SPECIAL_CHARACTERS_ADMIN_ONLY", "true").lower() in ("true", "1", "yes", "on")
INANIMATE_ENABLED = os.getenv("TFBOT_INANIMATE_ENABLED", "true").lower() in ("true", "1", "yes", "on")
BLOCK_INANIMATE_EXCEPT_SPECIAL = os.getenv("TFBOT_BLOCK_INANIMATE_EXCEPT_SPECIAL", "false").lower() in ("true", "1", "yes", "on")


def _is_special_reroll_name(name: str) -> bool:
    normalized = _normalize_special_token(name)
    if not normalized:
        return False
    return normalized in SPECIAL_REROLL_TOKENS


def _has_special_reroll_access(state: Optional[TransformationState]) -> bool:
    if state is None:
        return False
    token = state.character_folder or state.character_name
    return _is_special_reroll_name(token)


def _state_folder_token(state: TransformationState) -> str:
    if not state:
        return ""
    if state.character_folder:
        return _normalize_folder_token(state.character_folder)
    lookup = CHARACTER_BY_NAME.get((state.character_name or "").strip().lower())
    if lookup and lookup.folder:
        return _normalize_folder_token(lookup.folder)
    return _normalize_folder_token(state.character_name)


def _state_matches_folder(state: TransformationState, folder_name: str) -> bool:
    normalized = _normalize_folder_token(folder_name)
    if not normalized:
        return False
    return _state_folder_token(state) == normalized


def _state_has_privileged_access(state: Optional[TransformationState]) -> bool:
    if not state or not PRIVILEGED_FORM_TOKENS:
        return False
    folder_token = _state_folder_token(state)
    if folder_token and folder_token in PRIVILEGED_FORM_TOKENS:
        return True
    name_token = _normalize_special_token(state.character_name)
    return bool(name_token and name_token in PRIVILEGED_FORM_TOKENS)


def _find_character_by_folder(folder_name: str) -> Optional[TFCharacter]:
    normalized = _normalize_folder_token(folder_name)
    if not normalized:
        return None
    return CHARACTER_BY_FOLDER.get(normalized)


def _character_directory_root() -> Optional[Path]:
    if VN_ASSET_ROOT:
        return VN_ASSET_ROOT
    fallback = BASE_DIR / "characters"
    if fallback.exists():
        return fallback
    return None


def _list_character_directory_names(refresh: bool = False) -> Sequence[str]:
    global _CHARACTER_DIRECTORY_CACHE, _CHARACTER_DIRECTORY_CACHE_EXPIRY
    now = time.monotonic()
    if not refresh and _CHARACTER_DIRECTORY_CACHE and now < _CHARACTER_DIRECTORY_CACHE_EXPIRY:
        return _CHARACTER_DIRECTORY_CACHE

    root = _character_directory_root()
    names: list[str] = []
    if root and root.exists():
        try:
            filtered: list[str] = []
            for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
                if not child.is_dir():
                    continue
                normalized = _normalize_folder_token(child.name)
                if normalized not in ALLOWED_CHARACTER_FOLDERS:
                    continue
                filtered.append(child.name)
            names = filtered
        except OSError as exc:
            logger.warning("Failed to read character directories from %s: %s", root, exc)
            names = []
    _CHARACTER_DIRECTORY_CACHE = names
    _CHARACTER_DIRECTORY_CACHE_EXPIRY = now + CHARACTER_DIRECTORY_CACHE_TTL
    return names


def _autocomplete_character_names(
    query: str,
    guild: Optional[discord.Guild],
) -> Sequence[str]:
    normalized = (query or "").strip().lower()
    seen: set[str] = set()
    results: list[str] = []

    for name in _list_character_directory_names():
        lowered = name.lower()
        if normalized and normalized not in lowered:
            continue
        if name in seen:
            continue
        results.append(name)
        seen.add(name)
        if len(results) >= CHARACTER_AUTOCOMPLETE_LIMIT:
            break
    return results


async def _character_name_autocomplete(
    interaction: discord.Interaction,
    current: str,
) -> list[app_commands.Choice[str]]:
    guild = interaction.guild
    matches = _autocomplete_character_names(current, guild)
    return [app_commands.Choice(name=name, value=name) for name in matches]


async def _outfit_autocomplete(
    interaction: discord.Interaction,
    current: str,
) -> list[app_commands.Choice[str]]:
    await ensure_state_restored()
    guild = interaction.guild
    actor = interaction.user
    if guild is None or actor is None:
        return []
    state = find_active_transformation(actor.id, guild.id)
    if state is None or not state.character_name:
        return []
    pose_outfits = list_pose_outfits(state.character_name)
    if not pose_outfits:
        return []
    normalized_query = (current or "").strip().lower()
    choices: list[app_commands.Choice[str]] = []
    for pose, options in sorted(pose_outfits.items(), key=lambda item: item[0].lower()):
        pose_token = pose.strip()
        for option in sorted(options, key=lambda value: value.lower()):
            option_label = option.strip()
            if not option_label:
                continue
            value_parts = [part for part in (pose_token, option_label) if part]
            if not value_parts:
                continue
            value = " ".join(value_parts)
            match_source = value.lower()
            if normalized_query and normalized_query not in match_source:
                continue
            label_pose = pose_token or "auto"
            label = f"{label_pose} - {option_label}"
            choices.append(app_commands.Choice(name=label[:100], value=value[:100]))
            if len(choices) >= OUTFIT_AUTOCOMPLETE_LIMIT:
                return choices
    return choices


async def _accessory_autocomplete(
    interaction: discord.Interaction,
    current: str,
) -> list[app_commands.Choice[str]]:
    await ensure_state_restored()
    guild = interaction.guild
    actor = interaction.user
    if guild is None or actor is None:
        return []
    state = find_active_transformation(actor.id, guild.id)
    if state is None or not state.character_name:
        return []
    accessories = list_character_accessories(state.character_name)
    if not accessories:
        return []
    guild_channel = interaction.channel if isinstance(interaction.channel, discord.abc.GuildChannel) else None
    selection_scope = _selection_scope_for_channel(guild_channel)
    accessory_states = get_accessory_states(state.character_name, scope=selection_scope)
    normalized_query = (current or "").strip().lower()
    choices: list[app_commands.Choice[str]] = []
    
    # Add "clearall" option if query matches
    if not normalized_query or "clearall" in normalized_query or "clear" in normalized_query or "all" in normalized_query:
        choices.append(app_commands.Choice(name="clearall - Reset all accessories to off", value="clearall"))
    
    for key, label in sorted(accessories.items(), key=lambda item: item[1].lower() if item[1] else item[0]):
        display_label = label or key
        match_source = f"{display_label} {key}".lower()
        if normalized_query and normalized_query not in match_source:
            continue
        status = accessory_states.get(key, "off")
        choice_label = f"{display_label} ({status})"
        choices.append(app_commands.Choice(name=choice_label[:100], value=key[:100]))
        if len(choices) >= ACCESSORY_AUTOCOMPLETE_LIMIT:
            break
    return choices


def _find_inanimate_form_by_token(token: str) -> Optional[Dict[str, object]]:
    normalized = (token or "").strip()
    if not normalized:
        return None
    normalized = normalized.lower()
    for entry in INANIMATE_FORMS:
        name_raw = str(entry.get("name", "")).strip()
        if not name_raw:
            continue
        if name_raw.lower() == normalized:
            return entry
    return None


def _resolve_roleplay_cog(channel: Optional[discord.abc.GuildChannel]) -> tuple[Optional[RoleplayCog], Optional[str]]:
    if ROLEPLAY_COG is None:
        return None, "Roleplay commands are not configured on this bot."
    if channel is None or not ROLEPLAY_COG.is_roleplay_post(channel):
        return None, "Use this command inside the RP forum post."
    return ROLEPLAY_COG, None


def _build_roleplay_state(
    character: TFCharacter, actor: discord.Member, guild: Optional[discord.Guild]
) -> TransformationState:
    now = utc_now()
    guild_id = 0
    if guild is not None:
        guild_id = guild.id
    elif actor.guild:
        guild_id = actor.guild.id
    state = TransformationState(
        user_id=actor.id,
        guild_id=guild_id,
        character_name=character.name,
        character_folder=character.folder,
        character_avatar_path=character.avatar_path,
        character_message=character.message,
        original_nick=actor.nick,
        started_at=now,
        expires_at=now + timedelta(hours=1),
        duration_label="roleplay",
        avatar_applied=False,
        original_display_name=member_profile_name(actor),
        is_inanimate=False,
        inanimate_responses=tuple(),
    )
    ensure_form_owner(state)
    return state


def _build_inanimate_roleplay_state(
    entry: Dict[str, object], actor: discord.Member, guild: Optional[discord.Guild]
) -> TransformationState:
    now = utc_now()
    guild_id = actor.guild.id if actor.guild else 0
    if guild is not None:
        guild_id = guild.id
    responses_raw = entry.get("responses") or []
    responses: Tuple[str, ...]
    if isinstance(responses_raw, (list, tuple)):
        responses = tuple(str(item).strip() for item in responses_raw if str(item).strip())
    else:
        responses = tuple()
    if not responses:
        message = str(entry.get("message") or "").strip()
        responses = (message,) if message else tuple()
    state = TransformationState(
        user_id=actor.id,
        guild_id=guild_id,
        character_name=str(entry.get("name") or "Mysterious Relic"),
        character_folder=None,
        character_avatar_path=str(entry.get("avatar_path") or ""),
        character_message=str(entry.get("message") or ""),
        original_nick=actor.nick,
        started_at=now,
        expires_at=now + timedelta(hours=1),
        duration_label="roleplay",
        avatar_applied=False,
        original_display_name=member_profile_name(actor),
        is_inanimate=True,
        inanimate_responses=responses,
    )
    ensure_form_owner(state)
    return state


def _build_placeholder_state(member: discord.Member, guild: discord.Guild) -> TransformationState:
    now = utc_now()
    state = TransformationState(
        user_id=member.id,
        guild_id=guild.id,
        character_name="",
        character_folder=None,
        character_avatar_path="",
        character_message="",
        original_nick=member.nick,
        original_display_name=member_profile_name(member),
        started_at=now,
        expires_at=now,
        duration_label="",
        avatar_applied=False,
        is_inanimate=False,
        inanimate_responses=tuple(),
    )
    ensure_form_owner(state)
    return state


def _token_active(token: str) -> bool:
    normalized = _normalize_folder_token(token)
    if not normalized:
        return False
    for state in active_transformations.values():
        if _state_folder_token(state) == normalized:
            return True
    return False


def _character_weight(character: TFCharacter) -> float:
    if not FEATURED_TF_WEIGHTS:
        return 1.0
    weight = 1.0
    for token, bonus in FEATURED_TF_WEIGHTS.items():
        if bonus <= 0:
            continue
        folder_token = _normalize_folder_token(character.folder)
        if folder_token and folder_token == _normalize_folder_token(token) and not _token_active(token):
            weight *= bonus
    return max(weight, 0.0)


def _select_weighted_character(characters: Sequence[TFCharacter]) -> TFCharacter:
    if not characters:
        raise ValueError("Character pool is empty.")
    weights = [_character_weight(character) for character in characters]
    total = sum(weights)
    if total <= 0:
        return random.choice(list(characters))
    threshold = random.random() * total
    accumulator = 0.0
    for character, weight in zip(characters, weights):
        accumulator += weight
        if threshold <= accumulator:
            return character
    return characters[-1]


def _actor_has_narrator_power(member: Optional[discord.Member]) -> bool:
    if member is None or member.guild is None:
        return False
    state = find_active_transformation(member.id, member.guild.id)
    if not state:
        return False
    return _state_has_privileged_access(state)


def _extract_user_id_from_token(token: str) -> Optional[int]:
    cleaned = (token or "").strip()
    if not cleaned:
        return None
    mention_match = re.fullmatch(r"<@!?(\d+)>", cleaned)
    if mention_match:
        try:
            return int(mention_match.group(1))
        except (TypeError, ValueError):
            return None
    if cleaned.isdigit():
        try:
            return int(cleaned)
        except ValueError:
            return None
    return None


def _find_state_by_folder(guild: discord.Guild, token: str) -> Optional[TransformationState]:
    normalized_value = (token or "").strip()
    if not normalized_value:
        return None
    user_id = _extract_user_id_from_token(normalized_value)
    if user_id is not None:
        state = active_transformations.get(state_key(guild.id, user_id))
        if state:
            return state
    folder_token = _normalize_folder_token(normalized_value)
    if not folder_token:
        return None
    for state in active_transformations.values():
        if state.guild_id != guild.id:
            continue
        if _state_folder_token(state) == folder_token:
            return state
    return None


def _is_admin_only_random_name(name: str) -> bool:
    normalized = _normalize_folder_token(name)
    return bool(normalized and normalized in {token.lower() for token in ADMIN_ONLY_RANDOM_FORMS})


def _format_special_reroll_hint(character_label: str, folder_token: Optional[str] = None) -> Optional[str]:
    token = folder_token or character_label
    if not _is_special_reroll_name(token):
        return None
    admin_restriction_text = ""
    if ADMIN_PROTECTION_ENABLED:
        admin_restriction_text = f"- You can't target admins or turn someone into {SPECIAL_FORM_TARGET}.\n"
    return (
        "```diff\n"
        f"- {character_label} perk unlocked! Use `/reroll who_member:<target>` to reroll a non-admin or add `to_character:<folder>` to pick the form.\n"
        f"{admin_restriction_text}"
        "```"
    )


def _default_inanimate_forms() -> Tuple[Dict[str, object], ...]:
    return (
        {
            "name": "Bewitched Pumpkin",
            "avatar_path": "avatars/inanimate/pumpkin.png",
            "message": "A carved grin flickers to life as candlelight dances from within.",
            "responses": [
                "*Your carved grin flickers with eerie candlelight.*",
                "*Seeds tumble out as you wobble helplessly on the table.*",
                "*The wind whistles through your hollow interior.*",
            ],
        },
        {
            "name": "Haunted Locker",
            "avatar_path": "avatars/inanimate/locker.png",
            "message": "Metal hinges groan, and a chill seeps through with every creak.",
            "responses": [
                "*The locker door creaks open with a metallic groan.*",
                "*A stack of dusty textbooks rattles inside.*",
                "*Someone scribbled 'boo' across your dented surface.*",
            ],
        },
        {
            "name": "Sentient Broom",
            "avatar_path": "avatars/inanimate/broom.png",
            "message": "Bristles rustle to life, eager to sweep the nearest floor.",
            "responses": [
                "*Bristles rustle as you sweep across the floor on your own.*",
                "*You lean dramatically against the wall, awaiting orders.*",
                "*A chill runs down your handleâ€”if you still had a spine.*",
            ],
        },
    )


def _load_inanimate_forms_from_gacha() -> Tuple[Dict[str, object], ...]:
    config_path = path_from_env("TFBOT_GACHA_CONFIG") or Path("gacha_config.json")
    if not config_path.exists():
        return ()
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse gacha config %s (%s); skipping inanimate import.", config_path, exc)
        return ()
    characters = payload.get("characters")
    if not isinstance(characters, dict):
        return ()
    forms: list[Dict[str, object]] = []
    for entry in characters.values():
        if not isinstance(entry, dict) or not entry.get("inanimate"):
            continue
        name = str(entry.get("display_name") or entry.get("name") or "").strip()
        message = str(entry.get("message") or "").strip()
        avatar_path = str(entry.get("avatar_path") or "").strip()
        if not name or not message:
            continue
        responses_field = entry.get("responses") or []
        if isinstance(responses_field, list):
            responses = [str(item).strip() for item in responses_field if str(item).strip()]
        else:
            responses = []
        if not responses:
            responses = [message]
        forms.append(
            {
                "name": name,
                "avatar_path": avatar_path,
                "message": message,
                "responses": responses,
            }
        )
    return tuple(forms)


def _load_inanimate_forms() -> Tuple[Dict[str, object], ...]:
    from_gacha = _load_inanimate_forms_from_gacha()
    if from_gacha:
        return from_gacha
    if not INANIMATE_DATA_FILE.exists():
        logger.info("Inanimate TF file %s not found; using defaults.", INANIMATE_DATA_FILE)
        return _default_inanimate_forms()
    try:
        payload = json.loads(INANIMATE_DATA_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse %s (%s); using defaults.", INANIMATE_DATA_FILE, exc)
        return _default_inanimate_forms()
    if not isinstance(payload, list):
        logger.warning("Inanimate TF file %s did not contain a list; using defaults.", INANIMATE_DATA_FILE)
        return _default_inanimate_forms()

    forms: list[Dict[str, object]] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        avatar_path = str(entry.get("avatar_path") or "").strip()
        message = str(entry.get("message") or "").strip()
        responses_field = entry.get("responses") or []
        if not name or not message:
            logger.debug("Skipping inanimate entry missing required fields: %s", entry)
            continue
        if isinstance(responses_field, list):
            responses = [str(item).strip() for item in responses_field if str(item).strip()]
        else:
            responses = []
        if not responses:
            responses = [message]
        forms.append(
            {
                "name": name,
                "avatar_path": avatar_path,
                "message": message,
                "responses": responses,
            }
        )
    if not forms:
        logger.warning("No valid inanimate forms loaded from %s; using defaults.", INANIMATE_DATA_FILE)
        return _default_inanimate_forms()
    return tuple(forms)


INANIMATE_FORMS = _load_inanimate_forms()

CHARACTER_DATA_FILE_SETTING = os.getenv("TFBOT_CHARACTERS_FILE", "").strip()
_CHARACTER_AVATAR_ROOT_SETTING = os.getenv("TFBOT_AVATAR_ROOT", "").strip()

_history_refresh_task: Optional[asyncio.Task] = None
_history_refresh_lock = asyncio.Lock()
_history_refresh_seq = 0


def schedule_history_refresh(delay: float = 0.2) -> None:
    """Debounce history snapshot updates."""
    if not CLASSIC_ENABLED:
        return
    global _history_refresh_task, _history_refresh_seq  # pylint: disable=global-statement

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.get_event_loop()

    _history_refresh_seq += 1
    sequence_id = _history_refresh_seq

    if _history_refresh_task and not _history_refresh_task.done():
        _history_refresh_task.cancel()

    async def runner(expected_seq: int) -> None:
        try:
            if delay:
                await asyncio.sleep(delay)
            async with _history_refresh_lock:
                if expected_seq != _history_refresh_seq:
                    return
                await publish_history_snapshot(
                    bot,
                    active_transformations,
                    tf_stats,
                    CHARACTER_POOL,
                    current_history_channel_id(),
                )
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Failed to refresh history snapshot: %s", exc)

    _history_refresh_task = loop.create_task(runner(sequence_id))


def _resolve_avatar_root() -> Optional[Path]:
    if not _CHARACTER_AVATAR_ROOT_SETTING:
        return None
    root = Path(_CHARACTER_AVATAR_ROOT_SETTING)
    if not root.is_absolute():
        root = (BASE_DIR / root).resolve()
    return root


_CHARACTER_DATASET_LOADS = 0
_CHARACTER_DATASET_STACKS: list[tuple[int, str]] = []


def _load_character_dataset() -> Sequence[Dict[str, str]]:
    global _CHARACTER_DATASET_LOADS
    _CHARACTER_DATASET_LOADS += 1
    logger.info("Character dataset load invoked (count=%s).", _CHARACTER_DATASET_LOADS)
    if _CHARACTER_DATASET_LOADS > 1:
        stack_snippet = "".join(traceback.format_stack(limit=8))
        _CHARACTER_DATASET_STACKS.append((_CHARACTER_DATASET_LOADS, stack_snippet))
    if CHARACTER_DATA_FILE_SETTING:
        dataset_path = Path(CHARACTER_DATA_FILE_SETTING).expanduser()
        if not dataset_path.is_absolute():
            dataset_path = (BASE_DIR / dataset_path).resolve()
        default_path = (BASE_DIR / "tf_characters.py").resolve()
        if dataset_path == default_path:
            logger.debug("TFBOT_CHARACTERS_FILE points to default tf_characters.py; using built-in dataset.")
            return _DEFAULT_CHARACTER_DATA
        if dataset_path.exists():
            suffix = dataset_path.suffix.lower()
            if suffix == ".py":
                try:
                    spec = importlib.util.spec_from_file_location(
                        "_tf_character_override",
                        dataset_path,
                    )
                    module = importlib.util.module_from_spec(spec) if spec and spec.loader else None
                    if module and spec and spec.loader:
                        spec.loader.exec_module(module)  # type: ignore[attr-defined]
                        data = getattr(module, "TF_CHARACTERS", None)
                        if isinstance(data, list):
                            return data
                        logger.warning("TF_CHARACTERS missing or invalid in %s; using default.", dataset_path)
                    else:
                        logger.warning("Unable to load character module from %s; using default.", dataset_path)
                except Exception as exc:  # pylint: disable=broad-except
                    logger.warning("Failed to import character dataset %s: %s. Using default.", dataset_path, exc)
            else:
                try:
                    data = json.loads(dataset_path.read_text(encoding="utf-8"))
                    if isinstance(data, list):
                        return data
                    logger.warning("Character dataset %s is not a list; using default.", dataset_path)
                except json.JSONDecodeError as exc:
                    logger.warning("Failed to parse character dataset %s: %s. Using default.", dataset_path, exc)
        else:
            logger.warning("Character dataset %s not found; falling back to default.", dataset_path)
    return _DEFAULT_CHARACTER_DATA


CHARACTER_AVATAR_ROOT = _resolve_avatar_root()

def _get_magic_emoji(guild: Optional[discord.Guild]) -> str:
    if guild is None or guild.id is None:
        return f":{MAGIC_EMOJI_NAME}:"
    cached = MAGIC_EMOJI_CACHE.get(guild.id)
    if cached:
        return cached
    emoji = discord.utils.get(guild.emojis, name=MAGIC_EMOJI_NAME)
    if emoji:
        MAGIC_EMOJI_CACHE[guild.id] = str(emoji)
    else:
        MAGIC_EMOJI_CACHE[guild.id] = f":{MAGIC_EMOJI_NAME}:"
    return MAGIC_EMOJI_CACHE[guild.id]


def _build_character_pool(
    source: Sequence[Dict[str, str]],
    avatar_root: Optional[Path] = None,
) -> Sequence[TFCharacter]:
    pool: list[TFCharacter] = []
    for entry in source:
        try:
            avatar_path = str(entry.get("avatar_path", "")).strip()
            if (
                avatar_root
                and avatar_path
                and not avatar_path.startswith(("http://", "https://"))
            ):
                candidate = Path(avatar_path)
                if not candidate.is_absolute():
                    avatar_path = str((avatar_root / candidate).resolve())
                else:
                    avatar_path = str(candidate)
            folder_name = str(entry.get("folder", "")).strip()
            if not folder_name or folder_name.upper() == "TODO":
                logger.warning("Skipping character %s: missing folder assignment.", entry.get("name", "Unnamed"))
                continue
            # Replace {BOT_NAME} placeholder in messages with actual bot name
            message_text = str(entry.get("message", ""))
            if "{BOT_NAME}" in message_text:
                from tf_characters import BOT_NAME
                message_text = message_text.replace("{BOT_NAME}", BOT_NAME)
            
            pool.append(
                TFCharacter(
                    name=entry["name"],
                    avatar_path=avatar_path,
                    message=message_text,
                    folder=folder_name,
                )
            )
        except KeyError as exc:
            logger.warning("Skipping character entry missing %s", exc)
    if not pool:
        raise RuntimeError("TF character dataset is empty. Populate tf_characters.py.")
    return pool


if _CHARACTER_DATASET_STACKS:
    for count, stack in _CHARACTER_DATASET_STACKS:
        logger.warning("Character dataset reload #%s stack:\n%s", count, stack)
    _CHARACTER_DATASET_STACKS.clear()

_CHARACTER_DATASET = _load_character_dataset()
CHARACTER_POOL = _build_character_pool(_CHARACTER_DATASET, CHARACTER_AVATAR_ROOT)
CHARACTER_BY_NAME: Dict[str, TFCharacter] = {
    character.name.strip().lower(): character for character in CHARACTER_POOL
}
CHARACTER_BY_FOLDER: Dict[str, TFCharacter] = {}
for character in CHARACTER_POOL:
    folder_token = _normalize_folder_token(character.folder or character.name)
    if folder_token and folder_token not in CHARACTER_BY_FOLDER:
        CHARACTER_BY_FOLDER[folder_token] = character
ALLOWED_CHARACTER_FOLDERS: set[str] = set(CHARACTER_BY_FOLDER.keys())
_CHARACTER_FOLDER_OVERRIDES = {
    character.name.strip().lower(): character.folder.strip()
    for character in CHARACTER_POOL
    if character.folder and character.folder.strip()
}
set_character_directory_overrides(_CHARACTER_FOLDER_OVERRIDES)
GACHA_MANAGER = None
_CHARACTER_DIRECTORY_CACHE: list[str] = []
_CHARACTER_DIRECTORY_CACHE_EXPIRY: float = 0.0
_CHARACTER_INFO_FORUM_LOCK = asyncio.Lock()
_THREAD_TITLE_PATTERN = re.compile(r"^(?P<name>.+?)\s*\((?P<folder>.+?)\)$")

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
if not DISCORD_TOKEN:
    raise RuntimeError("Missing DISCORD_TOKEN. Set it in your environment or .env file.")

TF_CHANCE = float(os.getenv("TFBOT_CHANCE", "0.10"))
TF_CHANCE = max(0.0, min(1.0, TF_CHANCE))

intents = discord.Intents.default()
intents.message_content = True
intents.members = True


class TFBot(commands.Bot):
    async def setup_hook(self) -> None:
        await setup_bot_extensions()


bot = TFBot(command_prefix=os.getenv("TFBOT_PREFIX", "!"), intents=intents)
ROLEPLAY_COG: Optional[RoleplayCog] = None
GAME_BOARD_MANAGER: Optional["GameBoardManager"] = None
_SYNCED_APP_COMMAND_GUILDS: set[int] = set()


async def setup_bot_extensions() -> None:
    global ROLEPLAY_COG
    if ROLEPLAY_FORUM_POST_ID > 0:
        ROLEPLAY_COG = await add_roleplay_cog(
            bot,
            forum_post_id=ROLEPLAY_FORUM_POST_ID,
            state_file=ROLEPLAY_STATE_FILE,
        )
    try:
        await bot.tree.sync()
    except discord.HTTPException as exc:
        logger.warning("Failed to sync application commands: %s", exc)
    await _sync_application_commands_for_known_guilds()


def _known_command_guild_ids() -> set[int]:
    guild_ids = {guild.id for guild in bot.guilds}
    possible_channels = [
        TF_CHANNEL_ID,
        GACHA_CHANNEL_ID,
        TF_HISTORY_CHANNEL_ID,
        TF_ARCHIVE_CHANNEL_ID,
    ]
    for channel_id in possible_channels:
        channel = bot.get_channel(channel_id)
        if isinstance(channel, discord.abc.GuildChannel):
            guild_ids.add(channel.guild.id)
    return {gid for gid in guild_ids if gid}


async def _sync_application_commands_for_known_guilds(extra_guild_ids: Optional[Iterable[int]] = None) -> None:
    guild_ids = _known_command_guild_ids()
    if extra_guild_ids:
        guild_ids.update(extra_guild_ids)
    for guild_id in guild_ids:
        if guild_id in _SYNCED_APP_COMMAND_GUILDS:
            continue
        try:
            await bot.tree.sync(guild=discord.Object(id=guild_id))
            _SYNCED_APP_COMMAND_GUILDS.add(guild_id)
            logger.info("Synced application commands for guild %s", guild_id)
        except discord.HTTPException as exc:
            logger.warning("Failed to sync application commands for guild %s: %s", guild_id, exc)


@bot.event
async def on_guild_join(guild: discord.Guild):
    await _sync_application_commands_for_known_guilds(extra_guild_ids=[guild.id])


def _resolve_accessory_key_input(accessory_name: str, accessories: Mapping[str, str]) -> Optional[str]:
    normalized = (accessory_name or "").strip().lower()
    if not normalized:
        return None
    compact = normalized.replace(" ", "").replace("_", "").replace("/", "").replace("-", "")
    if normalized in accessories:
        return normalized
    for key in accessories.keys():
        key_compact = key.replace(" ", "").replace("_", "").replace("/", "").replace("-", "")
        if normalized == key or (compact and compact == key_compact):
            return key
    for key, label in accessories.items():
        label_value = (label or "").strip().lower()
        if not label_value:
            continue
        label_compact = label_value.replace(" ", "").replace("_", "").replace("/", "").replace("-", "")
        if normalized == label_value or (compact and compact == label_compact):
            return key
    return None


def _selection_scope_for_channel(channel: Optional[discord.abc.GuildChannel]) -> Optional[str]:
    """Get selection scope for a channel. Returns scope string that includes guild ID for isolation."""
    if channel is None:
        return None
    
    guild_id = channel.guild.id if hasattr(channel, 'guild') and channel.guild else None
    
    # Check for roleplay post first
    if ROLEPLAY_COG is not None and ROLEPLAY_COG.is_roleplay_post(channel):
        if guild_id:
            return f"rp:{guild_id}"
        return "rp"
    
    # For regular channels, include guild ID to isolate instances
    if guild_id:
        return f"guild:{guild_id}"
    
    return None


class GuardedHelpCommand(commands.DefaultHelpCommand):
    """Custom help command that can ignore specific channels."""

    def __init__(self, blocked_channel_ids: set[int], **options):
        super().__init__(**options)
        self.blocked_channel_ids = blocked_channel_ids

    async def _should_block(self) -> bool:
        ctx = self.context
        if ctx is None or not self.blocked_channel_ids:
            return False
        channel_id = getattr(ctx.channel, "id", None)
        if channel_id in self.blocked_channel_ids:
            try:
                await ctx.message.delete()
            except discord.HTTPException:
                pass
            return True
        return False

    async def send_bot_help(self, mapping):
        if await self._should_block():
            return
        await super().send_bot_help(mapping)

    async def send_cog_help(self, cog):
        if await self._should_block():
            return
        await super().send_cog_help(cog)

    async def send_group_help(self, group):
        if await self._should_block():
            return
        await super().send_group_help(group)

    async def send_command_help(self, command):
        if await self._should_block():
            return
        await super().send_command_help(command)

    async def send_error_message(self, error):
        if await self._should_block():
            return
        await super().send_error_message(error)


blocked_help_channels: set[int] = set()
if GACHA_ENABLED:
    blocked_help_channels.add(GACHA_CHANNEL_ID)
if blocked_help_channels:
    bot.help_command = GuardedHelpCommand(blocked_help_channels, verify_checks=False)
else:
    bot.help_command = commands.DefaultHelpCommand()


async def ensure_state_restored() -> None:
    if tf_state.STATE_RESTORED:
        return
    states = load_states_from_disk()
    now = utc_now()
    for state in states:
        if not state.character_folder:
            lookup = CHARACTER_BY_NAME.get((state.character_name or "").strip().lower())
            if lookup and lookup.folder:
                state.character_folder = lookup.folder
        key = state_key(state.guild_id, state.user_id)
        active_transformations[key] = state
        remaining = max((state.expires_at - now).total_seconds(), 0)
        if remaining <= 0:
            await revert_transformation(state, expired=True)
        else:
            revert_tasks[key] = asyncio.create_task(_schedule_revert(state, remaining))
            logger.info(
                "Restored TF for user %s in guild %s (expires in %.0fs)",
                state.user_id,
                state.guild_id,
                remaining,
            )
    tf_state.STATE_RESTORED = True


async def _schedule_revert(state: TransformationState, delay: float) -> None:
    try:
        await asyncio.sleep(delay)
        await revert_transformation(state, expired=True)
    except asyncio.CancelledError:
        logger.debug("Revert task for user %s cancelled", state.user_id)
    except Exception:
        logger.exception("Unexpected error while reverting TF for user %s", state.user_id)


async def _announce_swap_cascade(
    guild_id: int,
    transitions: Sequence[SwapTransition],
    *,
    reason: str,
    channel: Optional[discord.abc.Messageable] = None,
) -> None:
    guild = bot.get_guild(guild_id)
    lines: List[str] = []
    for transition in transitions:
        member_name = None
        if guild:
            member = guild.get_member(transition.user_id)
            if member:
                member_name = member.display_name
        member_name = member_name or f"User {transition.user_id}"
        lines.append(f"- {member_name}: {transition.before_form} -> {transition.after_form}")
    summary = "\n".join(lines) if lines else "No participants."
    description = f"{reason}\n{summary}"
    await send_history_message("Swap Chain Reset", description)
    if channel:
        try:
            await channel.send(
                f"Swap chain reset: {reason}\n{summary}",
                allowed_mentions=discord.AllowedMentions.none(),
            )
        except discord.HTTPException as exc:
            logger.warning(
                "Failed to announce swap cascade in channel %s: %s",
                getattr(channel, "id", "unknown"),
                exc,
            )


async def _unswap_and_announce(
    guild_id: int,
    trigger_user_id: int,
    *,
    reason: str,
    channel: Optional[discord.abc.Messageable] = None,
) -> bool:
    transitions = unswap_chain(guild_id, trigger_user_id)
    if not transitions:
        return False
    await _announce_swap_cascade(guild_id, transitions, reason=reason, channel=channel)
    return True


async def revert_transformation(state: TransformationState, *, expired: bool) -> None:
    key = state_key(state.guild_id, state.user_id)
    current = active_transformations.get(key)
    if current is None or current.expires_at != state.expires_at:
        return

    guild, member = await fetch_member(state.guild_id, state.user_id)
    reason = "TF expired" if expired else "TF reverted"
    if member:
        if not state.original_display_name:
            state.original_display_name = member_profile_name(member)
    else:
        logger.warning("Could not locate member %s in guild %s to revert TF", state.user_id, state.guild_id)

    await _unswap_and_announce(
        state.guild_id,
        state.user_id,
        reason=f"{member.display_name if member else f'User {state.user_id}'}: {reason}",
    )

    task = revert_tasks.pop(key, None)
    if task:
        task.cancel()
    active_transformations.pop(key, None)
    persist_states()

    schedule_history_refresh()


async def fetch_member(guild_id: int, user_id: int) -> Tuple[Optional[discord.Guild], Optional[discord.Member]]:
    guild = bot.get_guild(guild_id)
    if guild is None:
        try:
            guild = await bot.fetch_guild(guild_id)
        except discord.HTTPException as exc:
            logger.warning("Unable to fetch guild %s: %s", guild_id, exc)
            return None, None
    member = guild.get_member(user_id)
    if member is None:
        try:
            member = await guild.fetch_member(user_id)
        except discord.HTTPException as exc:
            logger.warning("Unable to fetch member %s in guild %s: %s", user_id, guild_id, exc)
            return guild, None
    return guild, member


BASE_DIR = Path(__file__).resolve().parent


def _load_character_context() -> Dict[str, str]:
    context_path = BASE_DIR / "data" / "character_context.json"
    if not context_path.exists():
        return {}
    try:
        return json.loads(context_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load character context dataset: %s", exc)
        return {}


CHARACTER_CONTEXT = _load_character_context()

_REPLY_LOG_SETTING = os.getenv("TFBOT_REPLY_LOG", "transform_replies.json").strip()
if _REPLY_LOG_SETTING:
    _reply_path = Path(_REPLY_LOG_SETTING)
    if not _reply_path.is_absolute():
        REPLY_LOG_FILE = (BASE_DIR / _reply_path).resolve()
    else:
        REPLY_LOG_FILE = _reply_path.resolve()
else:
    REPLY_LOG_FILE = (BASE_DIR / "transform_replies.json").resolve()


def _load_reply_log() -> Dict[int, ReplyContext]:
    if not REPLY_LOG_FILE.exists():
        return {}
    try:
        raw = json.loads(REPLY_LOG_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read reply log %s: %s", REPLY_LOG_FILE, exc)
        return {}
    result: Dict[int, ReplyContext] = {}
    for key, value in raw.items():
        try:
            message_id = int(key)
            author = value.get("author", "Unknown")
            text = value.get("text", "")
        except Exception:  # pylint: disable=broad-except
            continue
        if text:
            result[message_id] = ReplyContext(author=author, text=text)
    return result


def _persist_reply_log() -> None:
    try:
        REPLY_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            str(message_id): {"author": ctx.author, "text": ctx.text}
            for message_id, ctx in TRANSFORM_MESSAGE_LOG.items()
        }
        REPLY_LOG_FILE.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("Failed to persist reply log %s: %s", REPLY_LOG_FILE, exc)


TRANSFORM_MESSAGE_LOG: Dict[int, ReplyContext] = _load_reply_log()


def _register_relay_message(message_id: int, author: str, text: str) -> None:
    if not text:
        return
    TRANSFORM_MESSAGE_LOG[message_id] = ReplyContext(author=author, text=text)
    if len(TRANSFORM_MESSAGE_LOG) > 500:
        for key in list(TRANSFORM_MESSAGE_LOG.keys())[:100]:
            TRANSFORM_MESSAGE_LOG.pop(key, None)
    _persist_reply_log()
    logger.debug("Reply log registered id=%s author=%s text=%s", message_id, author, text[:120])


async def _resolve_reply_context(message: discord.Message) -> Optional[ReplyContext]:
    reference = message.reference
    if not reference or not reference.message_id:
        return None

    cached = TRANSFORM_MESSAGE_LOG.get(reference.message_id)
    resolved_msg = reference.resolved
    target_msg: Optional[discord.Message] = None

    if isinstance(resolved_msg, discord.Message):
        target_msg = resolved_msg
    else:
        try:
            target_msg = await message.channel.fetch_message(reference.message_id)  # type: ignore[arg-type]
        except discord.HTTPException as exc:
            logger.debug("Unable to fetch referenced message %s: %s", reference.message_id, exc)
            target_msg = None

    if target_msg is None:
        if cached:
            logger.debug("Reply context resolved from cache for %s", reference.message_id)
        return cached

    author = getattr(target_msg.author, "display_name", None) or getattr(
        target_msg.author, "name", "Unknown"
    )

    content = (target_msg.content or "").strip()
    if content:
        context = ReplyContext(author=author, text=content)
        TRANSFORM_MESSAGE_LOG[reference.message_id] = context
        _persist_reply_log()
        logger.debug("Reply context resolved from message %s", reference.message_id)
        return context

    if cached:
        return cached

    if target_msg.embeds:
        embed = target_msg.embeds[0]
        if embed.description:
            context = ReplyContext(author=author, text=embed.description.strip())
            TRANSFORM_MESSAGE_LOG[reference.message_id] = context
            _persist_reply_log()
            logger.debug("Reply context resolved from embed %s", reference.message_id)
            return context

    return None


async def relay_transformed_message(
    message: discord.Message,
    state: TransformationState,
    *,
    reference: Optional[discord.MessageReference] = None,
    gacha_stars: Optional[int] = None,
    gacha_outfit: Optional[str] = None,
    gacha_pose: Optional[str] = None,
    gacha_rudy: Optional[int] = None,
    gacha_frog: Optional[int] = None,
    gacha_border: Optional[str] = None,
) -> bool:
    guild = message.guild
    if guild is None:
        return False

    selection_scope = _selection_scope_for_channel(message.channel)

    cleaned_content = message.content.strip()
    original_content = cleaned_content
    generated_inanimate_response = False
    has_links = False
    character_name_normalized = (state.character_name or "").strip().lower()
    is_ball_override = state.is_inanimate and character_name_normalized == "ball"
    behaves_like_character = not state.is_inanimate or is_ball_override
    if state.is_inanimate and not is_ball_override:
        options = state.inanimate_responses or (
            "You emit a faint, spooky rattle.",
        )
        base_response = random.choice(options)
        spoiler_line = ""
        if original_content:
            sanitized_original, has_links = strip_urls(original_content)
            if sanitized_original:
                sanitized_original = discord.utils.escape_mentions(sanitized_original)
                sanitized_original = discord.utils.escape_markdown(sanitized_original)
            if sanitized_original:
                spoiler_line = f"\n||*{sanitized_original}*||"
        cleaned_content = f"{base_response}{spoiler_line}"
        generated_inanimate_response = True
    reply_context = await _resolve_reply_context(message)
    if (
        AI_REWRITE_ENABLED
        and cleaned_content
        and not cleaned_content.startswith(str(bot.command_prefix))
        and behaves_like_character
    ):
        context_snippet = CHARACTER_CONTEXT.get(state.character_name) or state.character_message
        rewritten = await rewrite_message_for_character(
            original_text=cleaned_content,
            character_name=state.character_name,
            character_context=context_snippet,
            user_name=message.author.display_name,
        )
        if rewritten and rewritten.strip():
            logger.debug(
                "AI rewrite applied for %s: %s -> %s",
                state.character_name,
                cleaned_content[:120],
                rewritten[:120],
            )
            cleaned_content = rewritten.strip()
    if cleaned_content and behaves_like_character:
        cleaned_content, has_links = strip_urls(cleaned_content)
        cleaned_content = cleaned_content.strip()

    if cleaned_content:
        mention_ready_text, mention_lookup, has_mentions = prepare_panel_mentions(message, cleaned_content)
        if has_mentions:
            cleaned_content = apply_mention_placeholders(mention_ready_text, mention_lookup)

    description = cleaned_content if cleaned_content else "*no message content*"
    formatted_segments = parse_discord_formatting(cleaned_content) if cleaned_content else None
    custom_emoji_images: Dict[str, "Image.Image"] = {}

    files: list[discord.File] = []
    payload: dict = {}

    if MESSAGE_STYLE == "vn" and not state.is_inanimate:
        if formatted_segments is None:
            formatted_segments = parse_discord_formatting(cleaned_content)
        custom_emoji_images = await prepare_custom_emoji_images(message, formatted_segments)
        if reply_context:
            logger.debug(
                "Replying panel: %s -> %s snippet=%s",
                state.character_name,
                reply_context.author,
                reply_context.text[:120],
            )
        character_display_name = _resolve_character_display_name(message.guild, message.author.id, state)
        vn_file = render_vn_panel(
            state=state,
            message_content=cleaned_content,
            character_display_name=character_display_name,
            original_name=message.author.display_name,
            attachment_id=str(message.id),
            formatted_segments=formatted_segments,
            custom_emoji_images=custom_emoji_images,
            reply_context=reply_context,
            selection_scope=selection_scope,
            gacha_star_count=gacha_stars,
            gacha_outfit_override=gacha_outfit,
            gacha_pose_override=gacha_pose,
            gacha_rudy=gacha_rudy,
            gacha_frog=gacha_frog,
            gacha_border=gacha_border,
        )
        if vn_file:
            files.append(vn_file)
        else:
            logger.debug("VN panel rendering unavailable; using classic embed.")

    if not files:
        embed, avatar_file = await build_legacy_embed(state, description)
        if avatar_file:
            files.append(avatar_file)
        payload["embed"] = embed


    has_attachments = bool(message.attachments)
    preserve_original = has_attachments or has_links
    deleted = False
    if not preserve_original:
        deleted = True
        try:
            await archive_original_message(message)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to archive message %s: %s", message.id, exc)
        try:
            await message.delete()
        except discord.Forbidden:
            deleted = False
            logger.debug(
                "Missing permission to delete message %s for TF relay in channel %s",
                message.id,
                message.channel.id,
            )
        except discord.HTTPException as exc:
            deleted = False
            logger.warning("Failed to delete message %s: %s", message.id, exc)

    if not deleted and "embed" in payload:
        payload["embed"].set_footer(text="Grant Manage Messages so TF relay can replace posts.")

    send_kwargs: Dict[str, object] = {}
    send_kwargs.update(payload)
    if reference:
        if isinstance(reference, discord.Message):
            reference = reference.to_reference(fail_if_not_exists=False)
        send_kwargs["reference"] = reference
        send_kwargs["mention_author"] = False
    if files:
        send_kwargs["files"] = files

    sent_message: Optional[discord.Message] = None
    try:
        sent_message = await message.channel.send(**send_kwargs)
    except discord.HTTPException as exc:
        logger.warning("Failed to relay TF message %s: %s", message.id, exc)
        return False

    if sent_message and cleaned_content:
        _register_relay_message(sent_message.id, state.character_name, cleaned_content)

    if has_attachments and not has_links:
        placeholder = "\u200b"
        if message.content != placeholder:
            try:
                await message.edit(content=placeholder, attachments=message.attachments, suppress=True)
            except discord.HTTPException as exc:
                logger.debug("Unable to clear attachment message %s: %s", message.id, exc)

    return True


if GACHA_ENABLED:
    from tfbot.gacha import setup_gacha_mode

    GACHA_MANAGER = setup_gacha_mode(
        bot,
        character_pool=CHARACTER_POOL,
        relay_fn=relay_transformed_message,
    )

# Help command function - defined before registration
async def prefix_help_game_command(ctx: commands.Context, *, command: Optional[str] = None) -> None:
    """Show available player commands - game version, gacha, or default help."""
    # Check if this is a game thread first
    if GAME_BOARD_MANAGER and isinstance(ctx.channel, discord.Thread) and GAME_BOARD_MANAGER.is_game_thread(ctx.channel):
        await GAME_BOARD_MANAGER.command_help(ctx)
        return
    
    # Check if we're in gacha channel and gacha is enabled
    if GACHA_MANAGER and isinstance(ctx.channel, discord.TextChannel) and ctx.channel.id == GACHA_MANAGER.channel_id:
        await GACHA_MANAGER.command_help(ctx)
        return
    
    # Fall back to default help command system
    if bot.help_command:
        bot.help_command.context = ctx
        try:
            if command:
                cmd = bot.get_command(command)
                if cmd:
                    await bot.help_command.send_command_help(cmd)
                else:
                    await bot.help_command.send_error_message(f"Command '{command}' not found.")
            else:
                await bot.help_command.send_bot_help(bot.cogs)
        finally:
            bot.help_command.context = None

# Register help command conditionally (after gacha registers its commands)
# Our help command handles game threads, gacha channels, and default help
def _register_help_command():
    """Register unified help command that handles game threads, gacha, and default help."""
    # Remove any existing help command (from gacha or default)
    existing_help = bot.get_command("help")
    if existing_help:
        bot.remove_command(existing_help.name)
    
    # Register our unified help command
    help_cmd = commands.command(name="help")(prefix_help_game_command)
    help_cmd = commands.guild_only()(help_cmd)
    bot.add_command(help_cmd)

if GAME_BOARD_ENABLED:
    from tfbot.games import GameBoardManager

    GAME_BOARD_MANAGER = GameBoardManager(
        bot=bot,
        config_path=GAME_CONFIG_FILE,
        assets_dir=GAME_ASSETS_DIR,
    )

# Register help command after gacha (if enabled) has registered its commands
# This ensures our unified help command replaces any existing help command
_register_help_command()

async def send_history_message(title: str, description: str) -> None:
    channel = bot.get_channel(current_history_channel_id())
    if channel is None:
        try:
            channel = await bot.fetch_channel(current_history_channel_id())
        except discord.HTTPException as exc:
            logger.warning("Cannot send history message, channel lookup failed: %s", exc)
            return
    embed = discord.Embed(
        title=title,
        description=description,
        color=0x9B59B6 if title == "TF Applied" else 0x546E7A,
        timestamp=utc_now(),
    )
    try:
        await channel.send(embed=embed, allowed_mentions=discord.AllowedMentions.none())
    except discord.HTTPException as exc:
        logger.warning("Failed to send history message: %s", exc)
    schedule_history_refresh()


async def _resolve_archive_channel(preferred_guild: Optional[discord.Guild]) -> Optional[discord.abc.Messageable]:
    if TF_ARCHIVE_CHANNEL_ID <= 0:
        return None

    archive_channel = None
    if preferred_guild is not None:
        archive_channel = preferred_guild.get_channel(TF_ARCHIVE_CHANNEL_ID)
    if archive_channel is None:
        archive_channel = bot.get_channel(TF_ARCHIVE_CHANNEL_ID)
    if archive_channel is None:
        try:
            archive_channel = await bot.fetch_channel(TF_ARCHIVE_CHANNEL_ID)
        except (discord.Forbidden, discord.HTTPException) as exc:
            logger.warning("Cannot access archive channel %s: %s", TF_ARCHIVE_CHANNEL_ID, exc)
            return None
    if archive_channel is None or not hasattr(archive_channel, "send"):
        logger.debug("Archive channel %s unavailable or not messageable.", TF_ARCHIVE_CHANNEL_ID)
        return None
    return archive_channel


async def archive_original_message(message: discord.Message) -> None:
    archive_channel = await _resolve_archive_channel(message.guild)
    if archive_channel is None:
        return

    created_at = message.created_at
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)

    author_name = getattr(message.author, "display_name", str(message.author))
    author_id = getattr(message.author, "id", "unknown")
    channel_value = getattr(message.channel, "mention", f"#{getattr(message.channel, 'id', 'unknown')}")
    jump_link = getattr(message, "jump_url", None)

    embed = discord.Embed(
        title="Archived TF Message",
        description=message.content or "*no message content*",
        color=0x546E7A,
        timestamp=created_at,
    )
    embed.add_field(name="Author", value=f"{author_name} (`{author_id}`)", inline=False)
    embed.add_field(name="Channel", value=str(channel_value), inline=False)
    embed.add_field(name="Message ID", value=str(message.id), inline=False)
    if jump_link:
        embed.add_field(name="Jump Link", value=jump_link, inline=False)

    avatar_asset = getattr(message.author, "display_avatar", None)
    avatar_url = getattr(avatar_asset, "url", None)
    if avatar_url:
        embed.set_thumbnail(url=avatar_url)

    attachments = [attachment.url for attachment in message.attachments]
    if attachments:
        embed.add_field(name="Attachments", value="\n".join(attachments), inline=False)

    try:
        await archive_channel.send(embed=embed, allowed_mentions=discord.AllowedMentions.none())
    except discord.Forbidden as exc:
        logger.warning("Forbidden to send archive message for %s: %s", message.id, exc)
    except discord.HTTPException as exc:
        logger.warning("Failed to send archive message for %s: %s", message.id, exc)


def _format_interaction_option_values(options: Optional[Sequence[Mapping[str, Any]]]) -> str:
    if not options:
        return ""
    parts: list[str] = []
    for option in options:
        name = str(option.get("name", ""))
        opt_type = option.get("type")
        nested = option.get("options")
        if opt_type in (1, 2) and nested:
            nested_str = _format_interaction_option_values(nested)
            segment = f"{name} {nested_str}".strip()
            if segment:
                parts.append(segment)
            continue
        value = option.get("value")
        if isinstance(value, str):
            value_str = value
        elif value is None:
            value_str = "None"
        else:
            value_str = str(value)
        if name:
            parts.append(f"{name}={value_str}")
        else:
            parts.append(value_str)
    return " ".join(parts)


def _format_interaction_invocation(interaction: discord.Interaction, command: app_commands.Command) -> str:
    base_name = getattr(command, "qualified_name", getattr(command, "name", "unknown"))
    prefix = f"/{base_name}"
    data = interaction.data or {}
    option_str = _format_interaction_option_values(data.get("options")) if isinstance(data, Mapping) else ""
    target_id = ""
    if isinstance(data, Mapping):
        raw_target = data.get("target_id")
        if raw_target:
            target_id = f" target_id={raw_target}"
    if option_str:
        return f"{prefix} {option_str}{target_id}".strip()
    return f"{prefix}{target_id}".strip()


async def log_command_archive_entry(
    *,
    actor: Union[discord.Member, discord.User],
    channel: Optional[discord.abc.Messageable],
    command_display: str,
    invocation_text: Optional[str],
    source: str,
    guild: Optional[discord.Guild],
    invocation_id: Optional[str],
) -> None:
    archive_channel = await _resolve_archive_channel(guild)
    if archive_channel is None:
        return

    timestamp = utc_now()
    actor_name = getattr(actor, "display_name", getattr(actor, "name", "Unknown"))
    actor_id = getattr(actor, "id", "unknown")
    if channel is None:
        channel_value = "DM"
    else:
        channel_value = getattr(channel, "mention", None) or f"#{getattr(channel, 'id', 'unknown')}"

    description = invocation_text.strip() if invocation_text else "*no invocation text available*"
    embed = discord.Embed(
        title="Command Executed",
        description=description,
        color=0x1ABC9C,
        timestamp=timestamp,
    )
    embed.add_field(name="Author", value=f"{actor_name} (`{actor_id}`)", inline=False)
    embed.add_field(name="Channel", value=str(channel_value), inline=False)
    embed.add_field(name="Command", value=command_display or "unknown", inline=False)
    embed.add_field(name="Source", value=source, inline=True)
    if invocation_id:
        embed.add_field(name="Invocation ID", value=str(invocation_id), inline=True)
    avatar_asset = getattr(actor, "display_avatar", None)
    avatar_url = getattr(avatar_asset, "url", None)
    if avatar_url:
        embed.set_thumbnail(url=avatar_url)

    try:
        await archive_channel.send(embed=embed, allowed_mentions=discord.AllowedMentions.none())
    except discord.Forbidden as exc:
        logger.warning("Forbidden to send command archive entry: %s", exc)
    except discord.HTTPException as exc:
        logger.warning("Failed to send command archive entry: %s", exc)


def _format_character_message(
    template: str,
    original_name: str,
    mention: str,
    duration: str,
    character_name: str,
) -> str:
    context = {
        "member": original_name,
        "original_name": original_name,
        "mention": mention,
        "character": character_name,
        "duration": duration,
    }

    try:
        unique_segment = template.format(**context).strip() if template else ""
    except KeyError:
        unique_segment = template.strip() if template else ""

    if unique_segment:
        lead_line = f"{original_name} {unique_segment}".strip()
    else:
        lead_line = f"{original_name} feels a strange energy swirling..."

    summary_line = f"{original_name} becomes **{character_name}** for {duration}!"
    return f"{lead_line}\n{summary_line}"


def _resolve_character_display_name(
    guild: Optional[discord.Guild],
    member_id: int,
    state: TransformationState,
) -> str:
    if ROLEPLAY_COG and guild:
        try:
            override_display = ROLEPLAY_COG.resolve_display_name(guild.id, member_id)
        except Exception:  # pragma: no cover - defensive
            override_display = None
        if override_display:
            return override_display
    if state.identity_display_name:
        return state.identity_display_name
    return state.character_name


async def handle_transformation(message: discord.Message) -> Optional[TransformationState]:
    if not message.guild:
        logger.debug("Skipping TF outside of guild context.")
        return None

    await ensure_state_restored()

    member = message.guild.get_member(message.author.id)
    if member is None:
        try:
            member = await message.guild.fetch_member(message.author.id)
        except discord.HTTPException as exc:
            logger.warning("Failed to fetch member %s: %s", message.author.id, exc)
            return

    key = state_key(message.guild.id, member.id)
    if key in active_transformations:
        logger.debug("User %s already transformed; skipping.", member.id)
        return None

    # Admin protection: Non-admins can't transform admins when protection is enabled
    # When protection is ON: Only admins can transform admins (via reroll, not normal rolls)
    # When protection is OFF: Anyone can be transformed by normal message triggers
    if ADMIN_PROTECTION_ENABLED:
        member_is_admin = is_admin(member)
        if member_is_admin:
            # Admins can only be transformed by other admins (via reroll commands), not by their own messages
            # This blocks normal daily rolls from triggering on admins
            logger.debug("Admin protection enabled: blocking normal TF trigger for admin user %s (only admin rerolls allowed)", member.id)
            return None

    used_characters = {state.character_name for state in active_transformations.values()}
    available_characters = [
        character for character in CHARACTER_POOL if character.name not in used_characters
    ]
    if not is_admin(member):
        available_characters = [
            character
            for character in available_characters
            if not _is_admin_only_random_name(character.name)
            # Toggleable restriction: Only admins can get Ball/Narrator (if enabled)
            and (not SPECIAL_CHARACTERS_ADMIN_ONLY or not _is_special_reroll_name(character.folder or character.name))
        ]
    character: Optional[TFCharacter] = None

    inanimate_form = None
    if INANIMATE_FORMS and random.random() <= INANIMATE_TF_CHANCE:
        available_inanimate = list(INANIMATE_FORMS)
        # Toggleable restriction: Only admins can get Ball/Narrator inanimate forms (if enabled)
        if SPECIAL_CHARACTERS_ADMIN_ONLY and not is_admin(member):
            available_inanimate = [
                form for form in available_inanimate
                if not _is_special_reroll_name(str(form.get("name", "")))
            ]
        # Toggleable restriction: When disabled, only allow Ball/Narrator inanimate forms
        if not INANIMATE_ENABLED:
            available_inanimate = [
                form for form in available_inanimate
                if _is_special_reroll_name(str(form.get("name", "")))
            ]
        if available_inanimate:
            inanimate_form = random.choice(available_inanimate)

    if inanimate_form is None and not available_characters:
        logger.info("No available TF characters; skipping message %s", message.id)
        return None

    if inanimate_form is not None:
        selected_name = str(inanimate_form.get("name") or "").strip() or "Mystery Relic"
        character_avatar_path = str(inanimate_form.get("avatar_path") or "").strip()
        character_message = str(inanimate_form.get("message") or "").strip()
        responses_raw = inanimate_form.get("responses") or []
        if isinstance(responses_raw, (list, tuple)):
            inanimate_responses = tuple(
                str(item).strip() for item in responses_raw if str(item).strip()
            )
        else:
            inanimate_responses = tuple()
        if not character_message:
            character_message = "You feel unsettlingly still."
        if not inanimate_responses:
            inanimate_responses = (character_message,)
        duration_label, duration_delta = _choose_reroll_duration(True, selected_name)
    else:
        character = _select_weighted_character(available_characters)
        selected_name = character.name
        character_avatar_path = character.avatar_path
        character_message = character.message
        inanimate_responses = tuple()
        character_token = character.folder or selected_name
        duration_label, duration_delta = _choose_reroll_duration(False, character_token)
    now = utc_now()
    expires_at = now + duration_delta
    original_nick = member.nick
    profile_name = member_profile_name(member)

    state = TransformationState(
        user_id=member.id,
        guild_id=message.guild.id,
        character_name=selected_name,
        character_folder=character.folder if character else None,
        character_avatar_path=character_avatar_path,
        character_message=character_message,
        original_nick=original_nick,
        started_at=now,
        expires_at=expires_at,
        duration_label=duration_label,
        avatar_applied=False,
        original_display_name=profile_name,
        is_inanimate=inanimate_form is not None,
        inanimate_responses=inanimate_responses,
    )
    ensure_form_owner(state)
    active_transformations[key] = state
    persist_states()

    delay = max((expires_at - now).total_seconds(), 0)
    revert_tasks[key] = asyncio.create_task(_schedule_revert(state, delay))

    logger.info(
        "TF applied to user %s (%s) for %s (expires at %s)",
        member.id,
        selected_name,
        duration_label,
        expires_at.isoformat(),
    )

    if character is not None:
        increment_tf_stats(message.guild.id, member.id, character.name)

    await send_history_message(
        "TF Applied",
        f"Original Name: **{member.name}**\nCharacter: **{selected_name}**\nDuration: {duration_label}.",
    )

    original_name = profile_name
    response_text = _format_character_message(
        character_message,
        original_name,
        member.mention,
        duration_label,
        selected_name,
    )
    special_hint = _format_special_reroll_hint(selected_name, character.folder if character else None)
    if special_hint:
        response_text = f"{response_text}\n{special_hint}"
    emoji_prefix = _get_magic_emoji(message.guild)
    response_text = f"{emoji_prefix} {response_text}"
    await message.reply(response_text, mention_author=False)

    reply_reference: Optional[discord.MessageReference] = (
        message.to_reference(fail_if_not_exists=False) if message.reference else None
    )
    await relay_transformed_message(message, state, reference=reply_reference)

    return state


async def log_guild_permissions() -> None:
    me = bot.user
    for guild in bot.guilds:
        member = guild.me
        if member is None:
            try:
                member = await guild.fetch_member(me.id) if me else None
            except discord.HTTPException as exc:
                logger.warning("Could not fetch self member in guild %s: %s", guild.id, exc)
                continue
        perms = member.guild_permissions
        missing = []
        for attr, reason in REQUIRED_GUILD_PERMISSIONS.items():
            if not getattr(perms, attr, False):
                missing.append(f"{attr.replace('_', ' ')} ({reason})")
        if missing:
            logger.warning(
                "Guild '%s' (%s) missing permissions: %s",
                guild.name,
                guild.id,
                "; ".join(missing),
            )
        else:
            logger.info("Guild '%s' (%s) has all required permissions.", guild.name, guild.id)


async def log_channel_access() -> None:
    for guild in bot.guilds:
        me = guild.me
        if me is None:
            continue
        channel_ids = set()
        if CLASSIC_ENABLED:
            channel_ids.add(TF_CHANNEL_ID)
        history_channel_id = current_history_channel_id()
        if history_channel_id:
            channel_ids.add(history_channel_id)
        if GACHA_MANAGER is not None:
            channel_ids.add(GACHA_MANAGER.channel_id)
        channel_ids.discard(0)
        for channel_id in channel_ids:
            channel = guild.get_channel(channel_id)
            if channel is None:
                continue
            perms = channel.permissions_for(me)
            if not perms.view_channel or not perms.read_message_history:
                logger.warning(
                    "Channel %s in guild '%s' missing read permissions (view=%s, history=%s)",
                    channel_id,
                    guild.name,
                    perms.view_channel,
                    perms.read_message_history,
                )
            else:
                logger.info(
                    "Channel %s in guild '%s' readable; send=%s, mentionable=%s",
                    channel_id,
                    guild.name,
                    perms.send_messages,
                    perms.mention_everyone,
                )


def _is_allowed_command_channel(
    channel: Optional[Union[discord.abc.GuildChannel, discord.Thread]]
) -> bool:
    if channel is None:
        return False
    if ROLEPLAY_COG and ROLEPLAY_COG.is_roleplay_post(channel):
        return True
    if (
        GACHA_CHANNEL_ID
        and isinstance(channel, discord.TextChannel)
        and channel.id == GACHA_CHANNEL_ID
    ):
        return True
    if TF_CHANNEL_ID and isinstance(channel, discord.TextChannel) and channel.id == TF_CHANNEL_ID:
        return True
    if isinstance(channel, discord.Thread):
        if GAME_BOARD_MANAGER and GAME_BOARD_MANAGER.is_game_thread(channel):
            return True
        parent = channel.parent
        if parent and TF_CHANNEL_ID and parent.id == TF_CHANNEL_ID:
            return True
        if parent and GACHA_CHANNEL_ID and parent.id == GACHA_CHANNEL_ID:
            return True
    return False


def _command_channel_error_message() -> str:
    hints: List[str] = []
    if TF_CHANNEL_ID:
        hints.append(f"<#{TF_CHANNEL_ID}>")
    if GACHA_CHANNEL_ID and GACHA_CHANNEL_ID != TF_CHANNEL_ID:
        hints.append(f"<#{GACHA_CHANNEL_ID}>")
    if GAME_BOARD_MANAGER:
        hints.append("an active game thread")
    if ROLEPLAY_COG:
        hints.append("an RP forum thread")
    if not hints:
        return "Rolls are only supported in configured TF or game channels."
    if len(hints) == 1:
        return f"Rolls are only supported in {hints[0]}."
    return f"Rolls are only supported in {', '.join(hints[:-1])}, or {hints[-1]}."


def _describe_channel(channel: Optional[Union[discord.abc.GuildChannel, discord.Thread]]) -> str:
    if channel is None:
        return "channel=None"
    parent = getattr(channel, "parent", None)
    parent_desc = f"{getattr(parent, 'name', None)}#{getattr(parent, 'id', None)}" if parent else "None"
    return f"{channel.__class__.__name__}(name={getattr(channel, 'name', None)}, id={getattr(channel, 'id', None)}, parent={parent_desc})"


def _allowed_instance_channel_ids() -> set[int]:
    allowed: set[int] = set()
    if TF_CHANNEL_ID > 0:
        allowed.add(TF_CHANNEL_ID)
    rp_forum_post_id = ROLEPLAY_COG.forum_post_id if ROLEPLAY_COG else 0
    if rp_forum_post_id:
        allowed.add(rp_forum_post_id)
    if GACHA_CHANNEL_ID > 0:
        allowed.add(GACHA_CHANNEL_ID)
    return allowed


def _channel_matches_allowed(
    channel: Optional[Union[discord.abc.GuildChannel, discord.Thread]],
    allowed_ids: set[int],
) -> bool:
    if not allowed_ids or channel is None:
        return False
    channel_id = getattr(channel, "id", None)
    if channel_id in allowed_ids:
        return True
    parent = getattr(channel, "parent", None)
    parent_id = getattr(parent, "id", None)
    if parent_id in allowed_ids:
        return True
    return False


def _extract_command_channel(
    channel_obj: Any,
) -> Optional[Union[discord.abc.GuildChannel, discord.Thread]]:
    if isinstance(channel_obj, (discord.abc.GuildChannel, discord.Thread)):
        return channel_obj
    return None


async def _ensure_command_channel_for_ctx(ctx: commands.Context) -> bool:
    channel = _extract_command_channel(ctx.channel)
    if _is_allowed_command_channel(channel):
        return True
    logger.warning(
        "Command guard blocked %s for %s in %s (allowed TF=%s, gacha=%s)",
        getattr(ctx.command, "qualified_name", getattr(ctx.command, "name", "unknown")),
        getattr(ctx.author, "id", "unknown"),
        _describe_channel(channel),
        TF_CHANNEL_ID,
        GACHA_CHANNEL_ID,
    )
    await ctx.reply(_command_channel_error_message(), mention_author=False)
    return False


async def _ensure_command_channel_for_interaction(interaction: discord.Interaction) -> bool:
    channel = _extract_command_channel(interaction.channel)
    if _is_allowed_command_channel(channel):
        return True
    logger.warning(
        "Interaction guard blocked %s for %s in %s (allowed TF=%s, gacha=%s)",
        getattr(interaction.command, "qualified_name", getattr(interaction.command, "name", "unknown")),
        getattr(interaction.user, "id", "unknown"),
        _describe_channel(channel),
        TF_CHANNEL_ID,
        GACHA_CHANNEL_ID,
    )
    message = _command_channel_error_message()
    if interaction.response.is_done():
        await interaction.followup.send(message, ephemeral=True)
    else:
        await interaction.response.send_message(message, ephemeral=True)
    return False


def guard_prefix_command_channel(func):
    @wraps(func)
    async def wrapper(ctx: commands.Context, *args, **kwargs):
        if not await _ensure_command_channel_for_ctx(ctx):
            return None
        return await func(ctx, *args, **kwargs)

    return wrapper


def guard_slash_command_channel(func):
    @wraps(func)
    async def wrapper(interaction: discord.Interaction, *args, **kwargs):
        if not await _ensure_command_channel_for_interaction(interaction):
            return None
        return await func(interaction, *args, **kwargs)

    return wrapper


async def _safe_defer_interaction(
    interaction: discord.Interaction,
    *,
    thinking: bool = True,
) -> bool:
    """Attempt to defer an interaction, tolerating already-expired interactions."""
    if interaction.response.is_done():
        return True
    try:
        await interaction.response.defer(thinking=thinking)
        return True
    except discord.NotFound as exc:
        logger.warning(
            "Interaction %s expired before defer could be sent: %s",
            getattr(interaction, "id", "unknown"),
            exc,
        )
        return False


def _find_character_face_path(folder_name: Optional[str]) -> Optional[Path]:
    if not folder_name or CHARACTER_FACES_ROOT is None:
        return None
    normalized = folder_name.strip()
    candidate_names = [normalized]
    lowered = normalized.lower()
    if lowered != normalized:
        candidate_names.append(lowered)
    checked: set[Path] = set()
    for candidate in candidate_names:
        folder_dir = (CHARACTER_FACES_ROOT / candidate).resolve()
        if folder_dir in checked or not folder_dir.exists():
            continue
        checked.add(folder_dir)
        variant_dirs = sorted(
            (entry for entry in folder_dir.iterdir() if entry.is_dir()),
            key=lambda path: path.name.lower(),
        )
        for variant_dir in variant_dirs:
            face_candidate = variant_dir / "face.png"
            if face_candidate.exists():
                return face_candidate
            png_candidates = sorted(variant_dir.glob("*.png"))
            if png_candidates:
                return png_candidates[0]
        direct_pngs = sorted(folder_dir.glob("*.png"))
        if direct_pngs:
            return direct_pngs[0]
    return None


def _render_pose_outfit_tree(pose_map: Mapping[str, Sequence[str]]) -> str:
    if not pose_map:
        return "- No pose or outfit data found."
    lines: list[str] = []
    for pose_name in sorted(pose_map.keys()):
        outfits = pose_map[pose_name]
        lines.append(f"- **{pose_name}**")
        if outfits:
            for outfit_name in sorted(outfits):
                lines.append(f"  - {outfit_name}")
        else:
            lines.append("  - (no outfits listed)")
    return "\n".join(lines)


def _build_character_info_post(character: TFCharacter) -> Tuple[str, Optional[Path]]:
    folder_token = (character.folder or character.name).strip()
    pose_map = list_pose_outfits(character.name)
    tree_text = _render_pose_outfit_tree(pose_map)
    lines = [
        f"Folder: `{folder_token}`",
        "",
        "**Poses & Outfits**",
        tree_text,
    ]
    content = "\n".join(lines).strip()
    face_path = _find_character_face_path(folder_token)
    return content, face_path


def _character_from_thread_title(title: str) -> Optional[TFCharacter]:
    normalized = title.strip()
    if not normalized:
        return None
    match = _THREAD_TITLE_PATTERN.match(normalized)
    lookup_names: list[str] = []
    folder_candidate = None
    if match:
        folder_candidate = _normalize_folder_token(match.group("folder"))
        name_candidate = match.group("name").strip().lower()
        if name_candidate:
            lookup_names.append(name_candidate)
    else:
        lookup_names.append(normalized.lower())
    for key in lookup_names:
        character = CHARACTER_BY_NAME.get(key)
        if character:
            return character
    if folder_candidate:
        return CHARACTER_BY_FOLDER.get(folder_candidate)
    return None


async def _iter_forum_threads(channel: discord.ForumChannel, forum_logger: logging.Logger):
    seen: set[int] = set()
    for thread in channel.threads:
        if thread.id in seen:
            continue
        seen.add(thread.id)
        yield thread

    try:
        async for archived in channel.archived_threads(limit=None):
            if archived.id in seen:
                continue
            seen.add(archived.id)
            yield archived
    except discord.HTTPException as exc:
        forum_logger.warning(
            "Failed to fetch archived threads for forum %s: %s",
            channel.id,
            exc,
        )


async def _purge_character_forum(channel: discord.ForumChannel, forum_logger: logging.Logger) -> int:
    deleted = 0
    async for thread in _iter_forum_threads(channel, forum_logger):
        try:
            await thread.delete()
            deleted += 1
        except discord.HTTPException as exc:
            forum_logger.warning("Failed to delete thread %s in forum %s: %s", thread.id, channel.id, exc)
        finally:
            await asyncio.sleep(CHARACTER_INFO_FORUM_ACTION_DELAY)
    return deleted


async def _create_character_forum_posts(channel: discord.ForumChannel, forum_logger: logging.Logger) -> int:
    created = 0
    for character in sorted(CHARACTER_POOL, key=lambda char: char.name.lower(), reverse=True):
        folder_token = (character.folder or character.name).strip()
        title = f"{character.name} ({folder_token})"
        content, face_path = _build_character_info_post(character)
        kwargs: dict[str, Any] = {
            "name": title,
            "content": content or "No pose or outfit data found.",
        }
        file_handle = None
        face_file = None
        if face_path:
            try:
                file_handle = face_path.open("rb")
                face_file = discord.File(file_handle, filename=f"{folder_token}_face.png")
                kwargs["file"] = face_file
            except OSError as exc:
                forum_logger.warning("Failed to open face image for %s at %s: %s", character.name, face_path, exc)
        try:
            await channel.create_thread(**kwargs)
            created += 1
        except discord.HTTPException as exc:
            forum_logger.warning("Failed to create info post for %s: %s", character.name, exc)
        finally:
            if file_handle:
                file_handle.close()
            await asyncio.sleep(CHARACTER_INFO_FORUM_ACTION_DELAY)
    return created


async def _refresh_existing_character_posts(
    channel: discord.ForumChannel, forum_logger: logging.Logger
) -> Tuple[int, int]:
    inspected = 0
    updated = 0
    async for thread in _iter_forum_threads(channel, forum_logger):
        inspected += 1
        character = _character_from_thread_title(thread.name or "")
        if character is None:
            continue
        try:
            first_message = None
            async for message in thread.history(limit=1, oldest_first=True):
                first_message = message
                break
        except discord.HTTPException as exc:
            forum_logger.warning("Failed to fetch first post for thread %s: %s", thread.id, exc)
            continue
        if first_message is None:
            continue
        expected_content, face_path = _build_character_info_post(character)
        current_content = (first_message.content or "").strip()
        needs_content = current_content != expected_content
        folder_token = (character.folder or character.name).strip()
        desired_face_name = f"{folder_token}_face.png"
        has_face_attachment = any(
            attachment.filename.lower() == desired_face_name.lower() for attachment in first_message.attachments
        )
        attachments_param: list[Union[discord.Attachment, discord.File]] = list(first_message.attachments)
        file_handle: Optional[io.BufferedReader] = None
        face_added = False
        if face_path and not has_face_attachment:
            try:
                file_handle = face_path.open("rb")
                attachments_param.append(discord.File(file_handle, filename=desired_face_name))
                face_added = True
            except OSError as exc:
                forum_logger.warning("Failed to open face image for %s at %s: %s", character.name, face_path, exc)
        edit_kwargs: dict[str, Any] = {}
        if needs_content:
            edit_kwargs["content"] = expected_content
        if face_added:
            edit_kwargs["attachments"] = attachments_param
        if not edit_kwargs:
            if file_handle:
                file_handle.close()
            continue
        try:
            await first_message.edit(**edit_kwargs)
            updated += 1
            await asyncio.sleep(CHARACTER_INFO_FORUM_ACTION_DELAY)
        except discord.HTTPException as exc:
            forum_logger.warning("Failed to refresh info post for %s (%s): %s", character.name, thread.id, exc)
        finally:
            if file_handle:
                file_handle.close()
    return inspected, updated


async def _get_character_info_forum_channel(
    forum_logger: logging.Logger,
) -> Optional[discord.ForumChannel]:
    if not CHARACTER_INFO_FORUM_ENABLED:
        return None
    await bot.wait_until_ready()
    channel = bot.get_channel(CHARACTER_INFO_FORUM_CHANNEL_ID)
    if channel is None:
        try:
            fetched_channel = await bot.fetch_channel(CHARACTER_INFO_FORUM_CHANNEL_ID)
        except discord.HTTPException as exc:
            forum_logger.warning(
                "Unable to fetch character info forum channel %s: %s",
                CHARACTER_INFO_FORUM_CHANNEL_ID,
                exc,
            )
            return None
        channel = fetched_channel
    if not isinstance(channel, discord.ForumChannel):
        forum_logger.warning(
            "Configured character info forum channel %s is not a forum. Feature disabled.",
            CHARACTER_INFO_FORUM_CHANNEL_ID,
        )
        return None
    return channel


async def _refresh_character_info_forum() -> Tuple[int, int]:
    forum_logger = logging.getLogger("tfbot.character_forum")
    channel = await _get_character_info_forum_channel(forum_logger)
    if channel is None:
        return (0, 0)
    forum_logger.info("Refreshing character info forum %s with %d characters.", channel.id, len(CHARACTER_POOL))
    deleted = await _purge_character_forum(channel, forum_logger)
    created = await _create_character_forum_posts(channel, forum_logger)
    forum_logger.info(
        "Character info forum refresh complete for channel %s (deleted=%s, created=%s).",
        channel.id,
        deleted,
        created,
    )
    return deleted, created


async def _refresh_character_info_posts_only() -> Tuple[int, int]:
    forum_logger = logging.getLogger("tfbot.character_forum")
    channel = await _get_character_info_forum_channel(forum_logger)
    if channel is None:
        return (0, 0)
    inspected, updated = await _refresh_existing_character_posts(channel, forum_logger)
    forum_logger.info(
        "Character info forum existing posts refreshed (inspected=%s, updated=%s).",
        inspected,
        updated,
    )
    return inspected, updated


def current_history_channel_id() -> int:
    return TF_HISTORY_CHANNEL_ID


@bot.event
async def on_ready():
    await ensure_state_restored()
    await _sync_application_commands_for_known_guilds()
    logger.info("Logged in as %s (id=%s)", bot.user, bot.user.id if bot.user else "unknown")
    logger.info("TF chance set to %.0f%%", TF_CHANCE * 100)
    logger.info("Message style: %s", MESSAGE_STYLE.upper())
    if TF_CHANNEL_ID > 0:
        status = "enabled" if CLASSIC_ENABLED else "disabled"
        logger.info("Primary channel (%s): %s", status, TF_CHANNEL_ID)
    elif CLASSIC_ENABLED:
        logger.warning("No primary channel configured.")
    if GACHA_MANAGER is not None:
        logger.info("Gacha channel: %s", GACHA_MANAGER.channel_id)
    await log_guild_permissions()
    await log_channel_access()
    schedule_history_refresh()
    
    # Send startup message "[Name] [version] is online"
    startup_channel_id = TF_CHANNEL_ID if TF_CHANNEL_ID > 0 else (TF_HISTORY_CHANNEL_ID if TF_HISTORY_CHANNEL_ID > 0 else 0)
    if startup_channel_id > 0:
        try:
            channel = bot.get_channel(startup_channel_id)
            if isinstance(channel, discord.TextChannel):
                await channel.send(f"{TFBOT_NAME} {TFBOT_VERSION} is online")
                logger.info("Startup message sent: %s %s is online", TFBOT_NAME, TFBOT_VERSION)
        except discord.HTTPException as exc:
            logger.warning("Failed to send startup message: %s", exc)


async def secret_reset_command(ctx: commands.Context):
    author = ctx.author
    if not isinstance(author, discord.Member):
        await ctx.reply("This command can only be used inside a server.", mention_author=False)
        return None
    if not is_admin(author):
        await ctx.reply("You lack permission to run this command.", mention_author=False)
        return None

    await ensure_state_restored()

    try:
        await ctx.message.delete()
    except discord.HTTPException:
        pass

    guild = ctx.guild
    await ctx.channel.send("Initiating full TF reset...", delete_after=5)

    states = [
        state for state in list(active_transformations.values()) if state.guild_id == guild.id
    ]
    restored = 0
    for state in states:
        await revert_transformation(state, expired=False)
        restored += 1

    await send_history_message(
        "TF Reset",
        f"Triggered by: **{author.name}**\nRestored TFs: {restored}",
    )
    await ctx.channel.send(f"TF reset completed. Restored {restored} transformations.", delete_after=10)


@bot.tree.command(name="synreset", description="Reset all active transformations in this server.")
@app_commands.guild_only()
@guard_slash_command_channel
async def slash_synreset_command(interaction: discord.Interaction) -> None:
    await interaction.response.defer(thinking=True)
    ctx = InteractionContextAdapter(interaction, bot=bot)
    await secret_reset_command(ctx)
    if not ctx.responded:
        await interaction.followup.send("TF reset completed.", ephemeral=True)


@bot.command(name="resetinfo")
async def reset_character_info_forum_command(ctx: commands.Context, confirmation: str = ""):
    author = ctx.author
    if not isinstance(author, discord.Member):
        await ctx.reply("Run this command inside a server.", mention_author=False)
        return
    if not is_admin(author):
        await ctx.reply("You lack permission to run this command.", mention_author=False)
        return
    if confirmation.strip().lower() != "confirm":
        await ctx.reply(
            "This will delete and recreate every info post. Run `!resetinfo confirm` to proceed.",
            mention_author=False,
        )
        return
    if not CHARACTER_INFO_FORUM_ENABLED:
        await ctx.reply(
            "TF_CHARACTER_INFO_FORUM_CHANNEL_ID is not configured on this bot.",
            mention_author=False,
        )
        return
    if _CHARACTER_INFO_FORUM_LOCK.locked():
        await ctx.reply("A character info refresh is already in progress. Please wait.", mention_author=False)
        return
    await ctx.reply(
        "Refreshing the character info forum... this may take a couple of minutes.",
        mention_author=False,
    )
    async with _CHARACTER_INFO_FORUM_LOCK:
        deleted, created = await _refresh_character_info_forum()
    logger.info(
        "Character info forum reset completed by %s (%s): deleted=%s created=%s",
        author.display_name,
        author.id,
        deleted,
        created,
    )
    await ctx.send(
        f"Character info forum refresh finished. Deleted {deleted} posts and created {created} posts.",
        mention_author=False,
    )


@bot.command(name="refreshinfo")
async def refresh_character_info_forum_posts_command(ctx: commands.Context):
    author = ctx.author
    if not isinstance(author, discord.Member):
        await ctx.reply("Run this command inside a server.", mention_author=False)
        return
    if not is_admin(author):
        await ctx.reply("You lack permission to run this command.", mention_author=False)
        return
    if not CHARACTER_INFO_FORUM_ENABLED:
        await ctx.reply(
            "TF_CHARACTER_INFO_FORUM_CHANNEL_ID is not configured on this bot.",
            mention_author=False,
        )
        return
    if _CHARACTER_INFO_FORUM_LOCK.locked():
        await ctx.reply("A character info refresh is already in progress. Please wait.", mention_author=False)
        return
    await ctx.reply(
        "Refreshing existing character info posts...",
        mention_author=False,
    )
    async with _CHARACTER_INFO_FORUM_LOCK:
        inspected, updated = await _refresh_character_info_posts_only()
    logger.info(
        "Character info forum refresh-only completed by %s (%s): inspected=%s updated=%s",
        author.display_name,
        author.id,
        inspected,
        updated,
    )
    await ctx.send(
        f"Character info forum refresh finished. Updated {updated} of {inspected} posts.",
        mention_author=False,
    )


async def reroll_command(ctx: commands.Context, *, args: str = ""):
    await ensure_state_restored()
    author = ctx.author
    if not isinstance(author, discord.Member):
        await ctx.reply("This command can only be used inside a server.", mention_author=False)
        return None
    
    # Check if this is a game thread - if so, apply game-specific restrictions
    game_state = None
    if GAME_BOARD_MANAGER and isinstance(ctx.channel, discord.Thread) and GAME_BOARD_MANAGER.is_game_thread(ctx.channel):
        game_state = GAME_BOARD_MANAGER._get_game_state_for_context(ctx)
        if game_state:
            # In game threads, only GM can reroll
            if not GAME_BOARD_MANAGER._is_gm(author, game_state):
                await ctx.reply("Only the GM can reroll characters in game threads.", mention_author=False)
                return None
    
    roleplay_dm_override = (
        ROLEPLAY_COG is not None
        and ROLEPLAY_COG.is_roleplay_post(ctx.channel)
        and ROLEPLAY_COG.has_control(author)
    )
    author_is_admin = is_admin(author) or roleplay_dm_override

    guild = ctx.guild
    if guild is None:
        await ctx.reply("This command can only be used inside a server.", mention_author=False)
        return None
    now = utc_now()
    author_state = find_active_transformation(author.id, guild.id)
    author_has_special_power = _has_special_reroll_access(author_state)
    author_special_label = author_state.character_name if author_state else ""
    can_force_reroll = author_is_admin or author_has_special_power
    
    # Check cooldown EARLY - before any other processing
    # If on cooldown, return immediately without processing anything
    if not (author_is_admin or author_has_special_power):
        last_reroll_at = get_last_reroll_timestamp(guild.id, author.id)
        if last_reroll_at is not None:
            cooldown_end = last_reroll_at + timedelta(hours=24)
            if cooldown_end > now:
                remaining = cooldown_end - now
                remaining_seconds = max(int(remaining.total_seconds()), 0)
                hours, remainder = divmod(remaining_seconds, 3600)
                minutes = remainder // 60
                if hours and minutes:
                    when_text = f"{hours} hour{'s' if hours != 1 else ''} and {minutes} minute{'s' if minutes != 1 else ''}"
                elif hours:
                    when_text = f"{hours} hour{'s' if hours != 1 else ''}"
                elif minutes:
                    when_text = f"{minutes} minute{'s' if minutes != 1 else ''}"
                else:
                    when_text = "less than a minute"
                await ctx.reply(
                    f"You've already used your reroll. You can reroll again in {when_text}.",
                    mention_author=False,
                )
                return  # Exit immediately - don't process anything
    
    target_member: Optional[discord.Member] = None
    target_is_admin = False
    state: Optional[TransformationState] = None

    forced_character: Optional[TFCharacter] = None
    forced_inanimate: Optional[Dict[str, object]] = None

    tokens = [token for token in args.split() if token.strip()]
    forced_token: Optional[str] = None
    forced_token_blocked = False
    target_member: Optional[discord.Member] = None
    target_is_admin = False
    state: Optional[TransformationState] = None
    placeholder_key: Optional[TransformKey] = None
    placeholder_state: Optional[TransformationState] = None
    target_selected = False

    def cleanup_placeholder() -> None:
        nonlocal placeholder_key, placeholder_state
        if placeholder_key is None or placeholder_state is None:
            return
        current = active_transformations.get(placeholder_key)
        if current is placeholder_state and not placeholder_state.character_name:
            active_transformations.pop(placeholder_key, None)
        placeholder_key = None
        placeholder_state = None

    if not tokens:
        slash_member = getattr(ctx, "_slash_target_member", None)
        slash_folder = getattr(ctx, "_slash_target_folder", None)
        if slash_member is not None:
            tokens = [slash_member.mention]
        elif slash_folder:
            tokens = [slash_folder]
        forced_override = getattr(ctx, "_slash_force_folder", None)
        if forced_override:
            forced_token = forced_override

    try:
        if tokens:
            first = tokens.pop(0)
            target_selected = True
            mention_id = _extract_user_id_from_token(first)
            if mention_id is not None:
                _, member_lookup = await fetch_member(guild.id, mention_id)
                if member_lookup is None:
                    await ctx.reply("I couldn't find that member.", mention_author=False)
                    return None
                target_member = member_lookup
                target_is_admin = is_admin(member_lookup)
                state = find_active_transformation(member_lookup.id, guild.id)
                if state is None:
                    placeholder = _build_placeholder_state(member_lookup, guild)
                    placeholder_key = state_key(guild.id, member_lookup.id)
                    placeholder_state = placeholder
                    active_transformations[placeholder_key] = placeholder
                    state = placeholder
            else:
                state = _find_state_by_folder(guild, first)
                if state is None:
                    # Allow slash args/autocomplete values that send a display name instead of a folder
                    state = _find_state_by_token(guild, first)
            if state is None:
                potential_member = discord.utils.find(
                    lambda m: m.name.lower() == first.lower()
                    or m.display_name.lower() == first.lower(),
                    guild.members,
                )
                if potential_member:
                    target_member = potential_member
                    target_is_admin = is_admin(potential_member)
                    placeholder = _build_placeholder_state(potential_member, guild)
                    placeholder_key = state_key(guild.id, potential_member.id)
                    placeholder_state = placeholder
                    active_transformations[placeholder_key] = placeholder
                    state = placeholder
                else:
                    await ctx.reply(
                        f"No active transformation uses folder `{first}`.",
                        mention_author=False,
                    )
                    return None
            if state is not None and target_member is None:
                _, member_lookup = await fetch_member(state.guild_id, state.user_id)
                target_member = member_lookup
                target_is_admin = bool(member_lookup and is_admin(member_lookup))

        if tokens:
            forced_token = tokens.pop(0)
        if tokens:
            await ctx.reply(
                "Too many arguments. Provide at most a target and optional forced folder.",
                mention_author=False,
            )
            return None

        if not target_selected:
            if not author_is_admin:
                await ctx.reply(
                    "Specify someone to reroll, e.g. `/reroll who_member:<member>` or mention the user.",
                    mention_author=False,
                )
                return None
            target_member = author
            target_is_admin = is_admin(author)
            state = author_state
            if state is None:
                await ctx.reply("You are not currently transformed.", mention_author=False)
                return None

        if forced_token is None:
            forced_override = getattr(ctx, "_slash_force_folder", None)
            if forced_override:
                forced_token = forced_override
        if forced_token and not can_force_reroll:
            forced_token = None
            forced_token_blocked = True

        if state is None and roleplay_dm_override and target_member is not None:
            placeholder = _build_placeholder_state(target_member, guild)
            placeholder_key = state_key(guild.id, target_member.id)
            placeholder_state = placeholder
            active_transformations[placeholder_key] = placeholder
            state = placeholder

        if state is None:
            await ctx.reply(
                "Unable to locate a transformation to reroll. Make sure the target is currently transformed.",
                mention_author=False,
            )
            return None
        if target_member is None:
            _, target_member = await fetch_member(state.guild_id, state.user_id)
            if target_member is None:
                await ctx.reply(
                    "Unable to locate the member transformed into this character.",
                    mention_author=False,
                )
                return None
            target_is_admin = is_admin(target_member)

        # Admin protection: Non-admins can't reroll admins (if enabled)
        if ADMIN_PROTECTION_ENABLED:
            if target_is_admin and not author_is_admin:
                await ctx.reply(
                    "Only admins can reroll other admins.",
                    mention_author=False,
                )
                return None

        # Admin protection: Ball/Narrator perks can't be used on admins (if enabled)
        if ADMIN_PROTECTION_ENABLED:
            if (
                author_has_special_power
                and not author_is_admin
                and target_is_admin
            ):
                await ctx.reply(
                    f"{SPECIAL_FORM_CAPITALIZED} perks can't be used on admins.",
                    mention_author=False,
                )
                return None

        if target_member.id == author.id and not (author_is_admin or author_has_special_power):
            await ctx.reply(
                "You can't use your own reroll. Ask another player or admin.",
                mention_author=False,
            )
            return None

        if forced_token_blocked:
            await ctx.reply(
                "The `to_character` option is disabled for regular rerolls. Choosing a random form instead.",
                mention_author=False,
            )

        if forced_token:
            forced_character = _find_character_by_folder(forced_token)
            forced_inanimate = None
            if forced_character is None:
                forced_inanimate = next(
                    (entry for entry in INANIMATE_FORMS if str(entry.get("name", "")).strip().lower() == forced_token.lower()),
                    None,
                )
            if forced_character is None and forced_inanimate is None:
                await ctx.reply(
                    f"Unknown folder `{forced_token}`. Use a valid character folder.",
                    mention_author=False,
                )
                return None
            # Toggleable restriction: Only admins or Ball/Narrator characters can force Ball/Narrator (if enabled)
            if SPECIAL_CHARACTERS_ADMIN_ONLY:
                if forced_character is not None and _is_special_reroll_name(forced_character.folder or forced_character.name):
                    if not author_is_admin and not author_has_special_power:
                        await ctx.reply(
                            f"Only admins or {SPECIAL_FORM_SUBJECT} can force someone into {SPECIAL_FORM_TARGET}.",
                            mention_author=False,
                        )
                        return None
                if forced_inanimate is not None and _is_special_reroll_name(str(forced_inanimate.get("name", ""))):
                    if not author_is_admin and not author_has_special_power:
                        await ctx.reply(
                            f"Only admins or {SPECIAL_FORM_SUBJECT} can force someone into {SPECIAL_FORM_TARGET}.",
                            mention_author=False,
                        )
                        return None
            if (
                forced_character is not None
                and _is_admin_only_random_name(forced_character.folder or forced_character.name)
                and not author_is_admin
                and not target_is_admin
            ):
                await ctx.reply(
                    f"You can only force {TFBOT_NAME} or Circe onto admins unless you're an admin yourself.",
                    mention_author=False,
                )
                return None
            # Admin protection: Ball/Narrator perks can't force into Ball/Narrator (if enabled)
            if ADMIN_PROTECTION_ENABLED:
                if (
                    forced_character is not None
                    and author_has_special_power
                    and not author_is_admin
                    and _is_special_reroll_name(forced_character.folder or forced_character.name)
                ):
                    await ctx.reply(
                        f"{SPECIAL_FORM_CAPITALIZED} perks can't force someone into {SPECIAL_FORM_TARGET}.",
                        mention_author=False,
                    )
                    return None
                if (
                    forced_inanimate is not None
                    and author_has_special_power
                    and not author_is_admin
                    and _is_special_reroll_name(str(forced_inanimate.get("name", "")))
                ):
                    await ctx.reply(
                        f"{SPECIAL_FORM_CAPITALIZED} perks can't force someone into {SPECIAL_FORM_TARGET}.",
                        mention_author=False,
                    )
                    return None

        key = state_key(guild.id, target_member.id)
        current_state = active_transformations.get(key)
        if current_state is None or current_state != state:
            await ctx.reply(
                "Unable to locate the transformation for this member.",
                mention_author=False,
            )
            return None

        await _unswap_and_announce(
            guild.id,
            target_member.id,
            reason=f"{target_member.display_name}'s swap chain reset before reroll.",
            channel=ctx.channel,
        )
        state = active_transformations.get(key)
        if state is None:
            await ctx.reply(
                "Their transformation reset during the swap cleanup. Try again if they are still transformed.",
                mention_author=False,
            )
            return None

        used_names = {
            current_state.character_name
            for current_key, current_state in active_transformations.items()
            if current_key != key
        }

        # Cooldown already checked earlier - skip duplicate check

        forced_mode = forced_character is not None or forced_inanimate is not None

        new_name: str
        new_folder: Optional[str]
        new_avatar_path: str
        new_message: str
        new_is_inanimate: bool
        new_responses: Tuple[str, ...]

        if forced_inanimate is not None:
            new_name = str(forced_inanimate.get("name") or "Mystery Relic")
            new_folder = None
            new_avatar_path = str(forced_inanimate.get("avatar_path") or "")
            new_message = str(forced_inanimate.get("message") or "You feel unsettlingly still.")
            responses_raw = forced_inanimate.get("responses") or []
            if isinstance(responses_raw, (list, tuple)):
                new_responses = tuple(str(item).strip() for item in responses_raw if str(item).strip())
            else:
                new_responses = tuple()
            if not new_responses:
                new_responses = (new_message,)
            new_is_inanimate = True
        elif forced_character is not None:
            new_name = forced_character.name
            new_folder = forced_character.folder
            new_avatar_path = forced_character.avatar_path
            new_message = forced_character.message
            new_responses = tuple()
            new_is_inanimate = False
        else:
            available_characters = [
                character
                for character in CHARACTER_POOL
                if character.name not in used_names and character.name != state.character_name
            ]
            if not target_is_admin:
                available_characters = [
                    character
                    for character in available_characters
                    if not _is_admin_only_random_name(character.name)
                    # Toggleable restriction: Only admins can reroll into Ball/Narrator (if enabled)
                    and (not SPECIAL_CHARACTERS_ADMIN_ONLY or not _is_special_reroll_name(character.folder or character.name))
                ]
            # Admin protection: Ball/Narrator perks can't reroll into Ball/Narrator (if enabled)
            if ADMIN_PROTECTION_ENABLED:
                if author_has_special_power and not author_is_admin:
                    available_characters = [
                        character
                        for character in available_characters
                        if not _is_special_reroll_name(character.folder or character.name)
                    ]
            if not available_characters:
                await ctx.reply(
                    "No alternative characters are available to reroll right now.",
                    mention_author=False,
                )
                return None
            chosen = random.choice(available_characters)
            new_name = chosen.name
            new_folder = chosen.folder
            new_avatar_path = chosen.avatar_path
            new_message = chosen.message
            new_responses = tuple()
            new_is_inanimate = False

        if new_name == state.character_name:
            await ctx.reply(
                f"They are already transformed into {new_name}.",
                mention_author=False,
            )
            return None
        if new_name in used_names:
            await ctx.reply(
                f"{new_name} is already in use by another transformation.",
                mention_author=False,
            )
            return None

        previous_character = state.character_name
        state.character_name = new_name
        state.character_folder = new_folder
        state.character_avatar_path = new_avatar_path
        state.character_message = new_message
        state.avatar_applied = False
        state.is_inanimate = new_is_inanimate
        state.inanimate_responses = new_responses
        placeholder_key = None
        placeholder_state = None

        duration_label, guaranteed_duration = _choose_reroll_duration(
            new_is_inanimate, new_folder or new_name
        )
        state.started_at = now
        state.expires_at = now + guaranteed_duration
        state.duration_label = duration_label
        existing_task = revert_tasks.get(key)
        if existing_task:
            existing_task.cancel()
        revert_tasks[key] = asyncio.create_task(
            _schedule_revert(state, guaranteed_duration.total_seconds())
        )

        persist_states()

        if not new_is_inanimate:
            increment_tf_stats(guild.id, target_member.id, new_name)
        if not (author_is_admin or author_has_special_power):
            record_reroll_timestamp(guild.id, author.id, now)

        history_details = (
            f"Triggered by: **{author.display_name}**\n"
            f"Member: **{target_member.display_name}**\n"
            f"Previous Character: **{previous_character}**\n"
            f"New Character: **{new_name}**"
        )
        if forced_mode:
            history_details += "\nReason: Forced reroll override."
        await send_history_message(
            "TF Rerolled",
            history_details,
        )

        original_name = member_profile_name(target_member)
        if forced_mode:
            custom_template = (
                f"barely has time to react before {TFBOT_NAME} swoops in with a grin and swaps them straight into {{character}}. {TFBOT_NAME} just had to spice things up."
            )
            response_text = _format_character_message(
                custom_template,
                original_name,
                target_member.mention,
                state.duration_label,
                new_name,
            )
        else:
            base_message = _format_character_message(
                new_message,
                original_name,
                target_member.mention,
                state.duration_label,
                new_name,
            )
            if author_is_admin:
                response_text = base_message
            elif author_has_special_power:
                perk_name = author_special_label or "their perk"
                response_text = (
                    f"{author.display_name} channels {perk_name} on {target_member.mention}! {base_message}"
                )
            else:
                response_text = (
                    f"{author.display_name} cashes in their reroll on {target_member.mention}! {base_message}"
                )
        # Suppress messages if this is a bulk reroll (from rerollall)
        suppress_messages = getattr(ctx, "_suppress_reroll_messages", False)
        
        if not suppress_messages:
            special_hint = _format_special_reroll_hint(new_name, new_folder)
            if special_hint:
                response_text = f"{response_text}\n{special_hint}"
            emoji_prefix = _get_magic_emoji(guild)
            try:
                await ctx.channel.send(
                    f"{emoji_prefix} {response_text}",
                    allowed_mentions=discord.AllowedMentions(users=[target_member]),
                )
            except discord.HTTPException as exc:
                logger.warning("Failed to announce reroll in channel %s: %s", ctx.channel.id, exc)

            try:
                await ctx.message.delete()
            except discord.HTTPException:
                pass

            summary_message = f"{target_member.display_name} has been rerolled into **{new_name}**."
            if forced_mode:
                summary_message += f" ({TFBOT_NAME} insisted on this one.)"
            await ctx.send(
                summary_message,
                delete_after=10,
            )
        
        # Sync game state after reroll completes (if in game thread)
        if game_state and target_member and ctx.guild and GAME_BOARD_MANAGER:
            if target_member.id in game_state.players:
                game_state.players[target_member.id].character_name = new_name
                await GAME_BOARD_MANAGER._save_game_state(game_state)
                await GAME_BOARD_MANAGER._log_action(game_state, f"{target_member.display_name} character rerolled to {new_name}")
    finally:
        cleanup_placeholder()


@bot.tree.command(name="reroll", description="Reroll an active transformation.")
@app_commands.describe(
    who_member="Member to reroll.",
    who_character="Folder of the active form to reroll.",
    to_character="Folder to force (admins or special forms only).",
)
@app_commands.autocomplete(who_character=_character_name_autocomplete, to_character=_character_name_autocomplete)
@app_commands.guild_only()
async def slash_reroll_command(
    interaction: discord.Interaction,
    who_member: Optional[discord.Member] = None,
    who_character: Optional[str] = None,
    to_character: Optional[str] = None,
) -> None:
    await interaction.response.defer(thinking=True)
    ctx = InteractionContextAdapter(interaction, bot=bot)
    ctx._slash_target_member = who_member
    ctx._slash_target_folder = who_character
    ctx._slash_force_folder = to_character
    await reroll_command(ctx, args="")
    if not ctx.responded:
        await interaction.followup.send("No reroll was performed.", ephemeral=True)


@bot.tree.command(name="rerollall", description="Reroll everyone at once (admin only).")
@app_commands.guild_only()
@guard_slash_command_channel
async def slash_rerollall_command(interaction: discord.Interaction) -> None:
    await interaction.response.defer(thinking=True)
    if not isinstance(interaction.user, discord.Member) or not is_admin(interaction.user) or not interaction.guild:
        await interaction.followup.send("Only admins can use this command.", ephemeral=True)
        return
    await ensure_state_restored()
    
    # Get members who can view this channel (those "in" the channel)
    channel_member_ids = set()
    if isinstance(interaction.channel, (discord.TextChannel, discord.Thread, discord.ForumChannel)):
        for member in interaction.guild.members:
            if interaction.channel.permissions_for(member).view_channel:
                channel_member_ids.add(member.id)
    
    rerolled_count = 0
    # Set flag to suppress individual reroll messages
    for (guild_id, user_id), state in active_transformations.items():
        if guild_id == interaction.guild.id and user_id in channel_member_ids:
            member = interaction.guild.get_member(user_id)
            if member:
                ctx = InteractionContextAdapter(interaction, bot=bot)
                ctx._slash_target_member = member
                ctx._slash_target_folder = None
                ctx._slash_force_folder = None
                ctx._suppress_reroll_messages = True
                await reroll_command(ctx, args="")
                rerolled_count += 1
    
    await interaction.followup.send("Rerolled Everyone", ephemeral=False)


@bot.tree.command(name="rerollnonadmin", description="Reroll everyone that's not admin (admin only).")
@app_commands.guild_only()
@guard_slash_command_channel
async def slash_rerollnonadmin_command(interaction: discord.Interaction) -> None:
    await interaction.response.defer(thinking=True)
    if not isinstance(interaction.user, discord.Member) or not is_admin(interaction.user) or not interaction.guild:
        await interaction.followup.send("Only admins can use this command.", ephemeral=True)
        return
    await ensure_state_restored()
    
    # Get members who can view this channel (those "in" the channel)
    channel_member_ids = set()
    if isinstance(interaction.channel, (discord.TextChannel, discord.Thread, discord.ForumChannel)):
        for member in interaction.guild.members:
            if interaction.channel.permissions_for(member).view_channel:
                channel_member_ids.add(member.id)
    
    rerolled_count = 0
    # Set flag to suppress individual reroll messages
    for (guild_id, user_id), state in active_transformations.items():
        if guild_id == interaction.guild.id and user_id in channel_member_ids:
            member = interaction.guild.get_member(user_id)
            if member and not is_admin(member):
                ctx = InteractionContextAdapter(interaction, bot=bot)
                ctx._slash_target_member = member
                ctx._slash_target_folder = None
                ctx._slash_force_folder = None
                ctx._suppress_reroll_messages = True
                await reroll_command(ctx, args="")
                rerolled_count += 1
    
    await interaction.followup.send(f"Rerolled {rerolled_count} non-admin member(s) in this channel.", ephemeral=False)


@bot.tree.command(name="dm", description="Show or assign the RP DM (use inside the RP forum thread).")
@app_commands.describe(member="Member to assign as the DM (leave blank to view current).")
@app_commands.guild_only()
@guard_slash_command_channel
async def slash_dm_command(
    interaction: discord.Interaction,
    member: Optional[discord.Member] = None,
) -> None:
    rp_cog, error = _resolve_roleplay_cog(interaction.channel)
    if error:
        await interaction.response.send_message(error, ephemeral=True)
        return
    ctx = InteractionContextAdapter(interaction, bot=bot)
    target = member.mention if member else ""
    await rp_cog.assign_dm_command(ctx, target=target)


@bot.tree.command(name="rename", description="RP: rename a participant for VN panels.")
@app_commands.describe(member="Player to rename", new_name="New VN display name")
@app_commands.guild_only()
@guard_slash_command_channel
async def slash_rename_command(
    interaction: discord.Interaction,
    member: discord.Member,
    new_name: str,
) -> None:
    rp_cog, error = _resolve_roleplay_cog(interaction.channel)
    if error:
        await interaction.response.send_message(error, ephemeral=True)
        return
    ctx = InteractionContextAdapter(interaction, bot=bot)
    if not await rp_cog._ensure_dm_actor(ctx):
        return
    await rp_cog.rename_identity_command(ctx, member=member.mention, new_name=new_name)


@bot.tree.command(name="unload", description="RP: remove a player's RP assignment/alias.")
@app_commands.describe(member="Player to unload (mention) or type 'all'. Leave blank for instructions.")
@app_commands.guild_only()
@guard_slash_command_channel
async def slash_unload_command(
    interaction: discord.Interaction,
    member: discord.Member,
) -> None:
    rp_cog, error = _resolve_roleplay_cog(interaction.channel)
    if error:
        await interaction.response.send_message(error, ephemeral=True)
        return
    ctx = InteractionContextAdapter(interaction, default_ephemeral=True, bot=bot)
    if not await rp_cog._ensure_dm_actor(ctx):
        return
    await rp_cog.unload_identity_command(ctx, member=member.mention)


@bot.tree.command(name="unloadall", description="RP: unload every participant in the RP thread (DM only).")
@app_commands.guild_only()
@guard_slash_command_channel
async def slash_unload_all_command(
    interaction: discord.Interaction,
) -> None:
    rp_cog, error = _resolve_roleplay_cog(interaction.channel)
    if error:
        await interaction.response.send_message(error, ephemeral=True)
        return
    ctx = InteractionContextAdapter(interaction, default_ephemeral=True, bot=bot)
    if not await rp_cog._ensure_dm_actor(ctx):
        return
    await rp_cog.unload_identity_command(ctx, member="all")


def _is_authorized_guild(ctx_guild: Optional[discord.Guild]) -> bool:
    if ctx_guild is None:
        return False
    channel_ids = [
        cid
        for cid in (
            TF_CHANNEL_ID,
            TF_HISTORY_CHANNEL_ID,
            TF_ARCHIVE_CHANNEL_ID,
            GACHA_CHANNEL_ID,
        )
        if cid
    ]
    if not channel_ids:
        return True
    allowed_guilds: set[int] = set()
    for channel_id in channel_ids:
        channel = bot.get_channel(channel_id)
        if channel and channel.guild:
            allowed_guilds.add(channel.guild.id)
        else:
            # Channel not found in cache; don't restrict.
            return True
    if not allowed_guilds:
        return True
    return ctx_guild.id in allowed_guilds


async def tf_stats_command(ctx: commands.Context):
    logger.debug("TF stats: starting for ctx=%s (guild=%s user=%s)", ctx, getattr(ctx.guild, "id", None), getattr(ctx.author, "id", None))
    await ensure_state_restored()
    guild_id = ctx.guild.id if ctx.guild else None
    if guild_id is None:
        logger.debug("TF stats: no guild context; aborting.")
        await ctx.reply(
            "Run this command from a server so I know which TF roster to check.",
            mention_author=False,
        )
        return False

    guild_data = tf_stats.get(str(guild_id), {})
    user_data = guild_data.get(str(ctx.author.id))
    has_stats = bool(user_data)
    if not user_data:
        user_data = {"total": 0, "characters": {}}
    logger.debug("TF stats: has_stats=%s data_keys=%s", has_stats, list(user_data.keys()))

    key = state_key(guild_id, ctx.author.id)
    current_state = active_transformations.get(key)
    logger.debug("TF stats: current_state=%s", current_state)

    if not has_stats and current_state is None:
        try:
            await ctx.author.send("You haven't experienced any transformations yet.")
        except discord.Forbidden:
            await ctx.reply(
                "I couldn't DM you. Please enable direct messages from server members.",
                mention_author=False,
                delete_after=10,
            )
        logger.debug("TF stats: no stats and no active TF; message sent.")
        if hasattr(ctx, "_responded"):
            ctx._responded = True
            if hasattr(ctx, "_responded_flag"):
                ctx._responded_flag = True
        return False

    embed = discord.Embed(
        title="Transformation Stats",
        color=0x9B59B6,
        timestamp=utc_now(),
    )
    avatar_url = ctx.author.display_avatar.url if ctx.author.display_avatar else None
    embed.set_author(
        name=ctx.author.display_name,
        icon_url=avatar_url,
    )
    embed.add_field(
        name="Total Transformations",
        value=str(user_data.get("total", 0)),
        inline=False,
    )

    characters = user_data.get("characters", {})
    if characters:
        sorted_chars = sorted(characters.items(), key=lambda item: item[1], reverse=True)
        lines = [f"- {name}: **{count}**" for name, count in sorted_chars]
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        for line in lines:
            line_len = len(line) + 1
            if current and current_len + line_len > 1000:
                chunks.append("\n".join(current))
                current = []
                current_len = 0
            current.append(line)
            current_len += line_len
        if current:
            chunks.append("\n".join(current))
        for idx, chunk in enumerate(chunks):
            name = "By Character" if idx == 0 else "\u200b"
            embed.add_field(name=name, value=chunk or "\u200b", inline=False)

    if current_state:
        remaining = max(
            (current_state.expires_at - utc_now()).total_seconds(),
            0,
        )
        minutes, seconds = divmod(int(remaining), 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            remaining_str = f"{hours}h {minutes}m {seconds}s"
        elif minutes:
            remaining_str = f"{minutes}m {seconds}s"
        else:
            remaining_str = f"{seconds}s"

        embed.add_field(
            name="Current Transformation",
            value=f"Character: **{current_state.character_name}**\nTime left: `{remaining_str}`",
            inline=False,
        )

    try:
        await ctx.author.send(embed=embed)
        logger.debug("TF stats: embed DM sent to %s", ctx.author)
        if current_state:
            pose_outfits = list_pose_outfits(current_state.character_name)
            if pose_outfits:
                selected_pose, selected_outfit = get_selected_pose_outfit(current_state.character_name)
                selected_pose_normalized = normalize_pose_name(selected_pose)
                selected_outfit_normalized = selected_outfit.lower() if selected_outfit else None
                pose_lines: list[str] = []
                for pose, options in pose_outfits.items():
                    entries: list[str] = []
                    for option in options:
                        display = option
                        if (
                            selected_outfit_normalized
                            and option.lower() == selected_outfit_normalized
                            and (
                                selected_pose_normalized is None
                                or pose.lower() == selected_pose_normalized
                            )
                        ):
                            display = f"{option} (current)"
                        entries.append(display)
                    pose_lines.append(f"{pose}: {', '.join(entries)}")
                outfit_note = (
                    f"Outfits available for {current_state.character_name}:\n"
                    + "\n".join(f"- {line}" for line in pose_lines)
                    + "\nUse `/outfit <outfit>` to pick by name or "
                    + "`/outfit <pose> <outfit>` (you can also separate with ':' or '/')."
                )
                try:
                    await ctx.author.send(outfit_note)
                except discord.Forbidden:
                    pass
        return True
    except discord.Forbidden:
        await ctx.reply(
            "I couldn't DM you. Please enable direct messages from server members.",
            mention_author=False,
            delete_after=10,
        )
        return False
    finally:
        message = getattr(ctx, "message", None)
        if message is not None:
            try:
                await message.delete()
            except Exception:
                pass


@bot.tree.command(name="tf", description="DM your transformation statistics.")
@app_commands.guild_only()
@guard_slash_command_channel
async def slash_tf_command(interaction: discord.Interaction) -> None:
    logger.debug("Slash /tf invoked by %s in guild %s channel %s", interaction.user, getattr(interaction.guild, "id", None), getattr(interaction.channel, "id", None))
    if not _is_authorized_guild(interaction.guild):
        logger.debug("Slash /tf: guild not authorized.")
        await interaction.response.send_message("This command isn't available in this guild.", ephemeral=True)
        return
    await interaction.response.defer(thinking=True, ephemeral=True)
    ctx = InteractionContextAdapter(interaction, default_ephemeral=True, bot=bot)
    logger.debug("Slash /tf: before stats responded=%s flags=%s", getattr(ctx, "responded", None), getattr(ctx, "_responded_flag", None))
    handled = await tf_stats_command(ctx)
    logger.debug("Slash /tf: after stats handled=%s responded=%s flags=%s", handled, getattr(ctx, "responded", None), getattr(ctx, "_responded_flag", None))
    if handled is False:
        return
    if not ctx.responded:
        await interaction.followup.send("Check your DMs for your stats.", ephemeral=True)
    logger.debug("TF stats: completed for %s", ctx.author)
    return True


async def background_command(ctx: commands.Context, *, selection: str = ""):
    try:
        await ctx.message.delete()
    except discord.HTTPException:
        pass

    async def send_channel_feedback(content: str, **kwargs) -> None:
        kwargs.setdefault("mention_author", False)
        reference = None
        try:
            reference = ctx.message.to_reference(fail_if_not_exists=False)
        except (AttributeError, discord.HTTPException):
            reference = None
        if reference is not None:
            kwargs.setdefault("reference", reference)
        await ctx.send(content, **kwargs)

    await ensure_state_restored()

    if VN_BACKGROUND_ROOT is None:
        try:
            await ctx.author.send("Backgrounds are not configured on this bot.")
        except discord.Forbidden:
            if ctx.guild:
                await ctx.send("I couldn't DM you. Please enable direct messages.", delete_after=10)
        return

    choices = list_background_choices()
    if not choices:
        try:
            await ctx.author.send("No background images were found in the configured directory.")
        except discord.Forbidden:
            if ctx.guild:
                await ctx.send("I couldn't DM you. Please enable direct messages.", delete_after=10)
        return

    selection = selection.strip()
    if not selection:
        lines: list[str] = []
        for idx, path in enumerate(choices, start=1):
            try:
                relative = path.resolve().relative_to(VN_BACKGROUND_ROOT.resolve())
                display = relative.as_posix()
            except ValueError:
                display = str(path)
            lines.append(f"{idx}: {display}")

        chunks: list[str] = []
        current: list[str] = []
        length = 0
        for line in lines:
            if length + len(line) + 1 > 1900 and current:
                chunks.append("\n".join(current))
                current = []
                length = 0
            current.append(line)
            length += len(line) + 1
        if current:
            chunks.append("\n".join(current))

        default_display = (
            VN_BACKGROUND_DEFAULT_RELATIVE.as_posix()
            if VN_BACKGROUND_DEFAULT_RELATIVE
            else "system default"
        )
        instructions = (
            "Use `/bg <number>` to apply that background to your VN panel.\n"
            "Example: `/bg 45` selects option 45 from the list.\n"
            f"The default background is `{default_display}`."
        )

        try:
            for chunk in chunks:
                await ctx.author.send(f"```\n{chunk}\n```")
            await ctx.author.send(instructions)
        except discord.Forbidden:
            if ctx.guild:
                await ctx.send("I couldn't DM you. Please enable direct messages, then rerun `/bg`.", delete_after=10)
            return

        return

    actor_member = ctx.author if isinstance(ctx.author, discord.Member) else None
    can_target_others = (
        ctx.guild is not None
        and actor_member is not None
        and (is_admin(actor_member) or _actor_has_narrator_power(actor_member))
    )
    selection = selection.strip()
    target_spec: Optional[str] = None
    if " " in selection:
        number_part, target_part = selection.split(None, 1)
        selection = number_part
        target_spec = target_part.strip() or None

    try:
        index = int(selection)
    except ValueError:
        try:
            await ctx.author.send(f"`{selection}` isn't a valid background number. Use `/bg` with no arguments to see the list.")
        except discord.Forbidden:
            if ctx.guild:
                await ctx.send("I couldn't DM you. Please enable direct messages.", delete_after=10)
        return

    if index < 1 or index > len(choices):
        try:
            await ctx.author.send(f"Background number must be between 1 and {len(choices)}.")
        except discord.Forbidden:
            if ctx.guild:
                await ctx.send("I couldn't DM you. Please enable direct messages.", delete_after=10)
        return

    selected_path = choices[index - 1]
    try:
        relative = selected_path.resolve().relative_to(VN_BACKGROUND_ROOT.resolve())
        display = relative.as_posix()
    except ValueError:
        display = str(selected_path)

    if target_spec:
        if ctx.guild is None or actor_member is None:
            await send_channel_feedback("Targeted background changes can only be used inside a server channel.")
            return
        if not can_target_others:
            await send_channel_feedback(_privileged_requirement_message("set backgrounds for other characters"))
            return
        target_lower = target_spec.lower()
        if target_lower == "all":
            targets = [
                state
                for state in active_transformations.values()
                if state.guild_id == ctx.guild.id and not state.is_inanimate
            ]
            if not targets:
                await send_channel_feedback("No active characters are available to update right now.")
                return
            failures = 0
            for state in targets:
                if not set_selected_background(state.user_id, selected_path):
                    failures += 1
            schedule_history_refresh()
            updated = len(targets) - failures
            await send_channel_feedback(
                f"Background set to `{display}` for {updated} character{'s' if updated != 1 else ''}."
            )
            return

        target_state = _find_state_by_folder(ctx.guild, target_spec)
        if target_state is None:
            rp_narrator_target = (
                ROLEPLAY_COG is not None
                and ROLEPLAY_COG.is_roleplay_post(ctx.channel)
                and _normalize_folder_token(target_spec) == "narrator"
            )
            if rp_narrator_target:
                dm_user_id = ROLEPLAY_COG.dm_user_id
                if not dm_user_id:
                    await send_channel_feedback("Assign a DM before setting the narrator's background.")
                    return
                dm_member = ctx.guild.get_member(dm_user_id)
                if dm_member is None:
                    await send_channel_feedback("I couldn't find the assigned DM in this server.")
                    return
                if not set_selected_background(dm_user_id, selected_path):
                    await send_channel_feedback("Unable to update the narrator's background right now.")
                    return
                schedule_history_refresh()
                await send_channel_feedback(f"Narrator background set to `{display}`.")
                return
            await send_channel_feedback(f"Couldn't find a transformed character matching `{target_spec}`.")
            return
        if target_state.is_inanimate:
            await send_channel_feedback(
                f"{target_state.character_name} is inanimate and can't use VN backgrounds."
            )
            return
        if not set_selected_background(target_state.user_id, selected_path):
            await send_channel_feedback("Unable to update that background right now.")
            return
        schedule_history_refresh()
        await send_channel_feedback(f"Background for {target_state.character_name} set to `{display}`.")
        return

    if not set_selected_background(ctx.author.id, selected_path):
        try:
            await ctx.author.send("Unable to update your background at this time.")
        except discord.Forbidden:
            if ctx.guild:
                await ctx.send("I couldn't DM you. Please enable direct messages.", delete_after=10)
        return

    try:
        await ctx.author.send(f"Background set to `{display}`.")
    except discord.Forbidden:
        if ctx.guild:
            await ctx.send("I couldn't DM you. Please enable direct messages.", delete_after=10)
    schedule_history_refresh()


@bot.tree.command(name="bg", description="Select or manage VN backgrounds.")
@app_commands.describe(selection="Background number (append a folder or member to target them).")
@app_commands.guild_only()
@guard_slash_command_channel
async def slash_bg_command(
    interaction: discord.Interaction,
    selection: Optional[str] = None,
) -> None:
    await interaction.response.defer(thinking=True)
    ctx = InteractionContextAdapter(interaction, bot=bot)
    await background_command(ctx, selection=selection or "")
    if not ctx.responded:
        await interaction.followup.send("Check your DMs for the background list.", ephemeral=True)


async def outfit_command(ctx: commands.Context, *, outfit_name: str = ""):
    outfit_name = outfit_name.strip()
    if not outfit_name:
        message = "Usage: /outfit <outfit>` or `/outfit <pose> <outfit>`"
        if ctx.guild:
            await ctx.reply(message, mention_author=False)
        else:
            await ctx.send(message)
        return

    await ensure_state_restored()

    selection_scope = _selection_scope_for_channel(
        ctx.channel if isinstance(ctx.channel, discord.abc.GuildChannel) else None
    )

    guild_id = ctx.guild.id if ctx.guild else None
    actor_member = ctx.author if isinstance(ctx.author, discord.Member) else None
    can_target_others = (
        ctx.guild is not None
        and actor_member is not None
        and (is_admin(actor_member) or _actor_has_narrator_power(actor_member))
    )
    target_state: Optional[TransformationState] = None
    if can_target_others and ctx.guild and " " in outfit_name:
        tokens = [token for token in outfit_name.split() if token.strip()]
        def _assign_target(index: int) -> bool:
            nonlocal outfit_name, target_state
            if index < 0 or index >= len(tokens):
                return False
            candidate = tokens[index]
            matched_state = _find_state_by_folder(ctx.guild, candidate)
            if not matched_state:
                return False
            target_state = matched_state
            remaining = tokens[:index] + tokens[index + 1 :]
            outfit_name = " ".join(remaining).strip()
            return True
        if len(tokens) >= 2:
            mention_index = next(
                (
                    idx
                    for idx, token in enumerate(tokens)
                    if _extract_user_id_from_token(token) is not None
                ),
                None,
            )
            if mention_index is not None and _assign_target(mention_index):
                pass
            elif _assign_target(len(tokens) - 1):
                pass
    if target_state and target_state.is_inanimate:
        if ctx.guild:
            await ctx.reply(f"{target_state.character_name} is inanimate and can't change outfits.", mention_author=False)
        else:
            await ctx.send(f"{target_state.character_name} is inanimate and can't change outfits.")
        return
    state = target_state or find_active_transformation(ctx.author.id, guild_id)
    if not state:
        fallback_state = find_active_transformation(ctx.author.id)
        if fallback_state and ctx.guild and fallback_state.guild_id != guild_id:
            target_guild = bot.get_guild(fallback_state.guild_id)
            guild_name = target_guild.name if target_guild else f"server {fallback_state.guild_id}"
            message = (
                "You're transformed right now, but in a different server. "
                f"Use this command in **{guild_name}** to change that outfit."
            )
        else:
            message = "You need to be transformed to change outfits."
        if ctx.guild:
            await ctx.reply(message, mention_author=False)
        else:
            await ctx.send(message)
        return

    pose_outfits = list_pose_outfits(state.character_name)
    if not pose_outfits:
        message = f"No outfits are available for {state.character_name}."
        if ctx.guild:
            await ctx.reply(message, mention_author=False)
        else:
            await ctx.send(message)
        return

    parsed_pose: Optional[str] = None
    parsed_outfit: Optional[str] = None

    for separator in (":", "/"):
        if separator in outfit_name:
            left, right = outfit_name.split(separator, 1)
            parsed_pose = left.strip()
            parsed_outfit = right.strip()
            break

    if parsed_outfit is None:
        parts = outfit_name.split()
        if len(parts) >= 2:
            parsed_pose = parts[0].strip()
            parsed_outfit = " ".join(parts[1:]).strip()
        else:
            parsed_outfit = outfit_name

    if not parsed_outfit:
        message = "Please provide the outfit to select. Example: `/outfit cheer` or `/outfit b cheer`."
        if ctx.guild:
            await ctx.reply(message, mention_author=False)
        else:
            await ctx.send(message)
        return

    if parsed_pose:
        normalized_pose = normalize_pose_name(parsed_pose)
        known_poses = {pose.lower() for pose in pose_outfits.keys()}
        if normalized_pose not in known_poses:
            message = (
                f"Unknown pose `{parsed_pose}`. Available poses: {', '.join(pose_outfits.keys())}."
            )
            if ctx.guild:
                await ctx.reply(message, mention_author=False)
            else:
                await ctx.send(message)
            return
    else:
        normalized_pose = None

    if not set_selected_pose_outfit(
        state.character_name,
        parsed_pose if normalized_pose else None,
        parsed_outfit,
        scope=selection_scope,
    ):
        pose_lines = []
        for pose, options in pose_outfits.items():
            pose_lines.append(f"{pose}: {', '.join(options)}")
        message = (
            f"Unable to update outfit. Available options:\n"
            + "\n".join(f"- {line}" for line in pose_lines)
        )
        if ctx.guild:
            await ctx.reply(message, mention_author=False)
        else:
            await ctx.send(message)
        return

    selected_pose, selected_outfit = get_selected_pose_outfit(
        state.character_name,
        scope=selection_scope,
    )
    pose_label = selected_pose or "auto"
    outfit_label = selected_outfit or parsed_outfit
    confirmation = (
        f"Outfit for {state.character_name} set to `{outfit_label}` (pose `{pose_label}`). "
        "Future messages will use this combination."
    )
    schedule_history_refresh()
    if ctx.guild:
        await ctx.reply(confirmation, mention_author=False)
    else:
        await ctx.send(confirmation)


@bot.tree.command(name="outfit", description="Select an outfit (optionally include pose).")
@app_commands.describe(outfit="Provide the outfit or `pose outfit`. Admins may append a target folder.")
@app_commands.autocomplete(outfit=_outfit_autocomplete)
@app_commands.guild_only()
@guard_slash_command_channel
async def slash_outfit_command(
    interaction: discord.Interaction,
    outfit: str,
) -> None:
    await interaction.response.defer(thinking=True)
    ctx = InteractionContextAdapter(interaction, bot=bot)
    await outfit_command(ctx, outfit_name=outfit or "")
    if not ctx.responded:
        await interaction.followup.send("No outfit change was applied.", ephemeral=True)


async def accessories_command(ctx: commands.Context, *, accessory_name: str = ""):
    accessory_name = accessory_name.strip()

    await ensure_state_restored()

    selection_scope = _selection_scope_for_channel(
        ctx.channel if isinstance(ctx.channel, discord.abc.GuildChannel) else None
    )
    guild_id = ctx.guild.id if ctx.guild else None
    actor_member = ctx.author if isinstance(ctx.author, discord.Member) else None
    can_target_others = (
        ctx.guild is not None
        and actor_member is not None
        and (is_admin(actor_member) or _actor_has_narrator_power(actor_member))
    )
    target_state: Optional[TransformationState] = None
    if accessory_name and can_target_others and ctx.guild and " " in accessory_name:
        tokens = [token for token in accessory_name.split() if token.strip()]
        def _assign_target(index: int) -> bool:
            nonlocal accessory_name, target_state
            if index < 0 or index >= len(tokens):
                return False
            candidate = tokens[index]
            matched_state = _find_state_by_folder(ctx.guild, candidate)
            if not matched_state:
                return False
            target_state = matched_state
            remaining = tokens[:index] + tokens[index + 1 :]
            accessory_name = " ".join(remaining).strip()
            return True
        if len(tokens) >= 2:
            mention_index = next(
                (
                    idx
                    for idx, token in enumerate(tokens)
                    if _extract_user_id_from_token(token) is not None
                ),
                None,
            )
            if mention_index is not None and _assign_target(mention_index):
                pass
            elif _assign_target(len(tokens) - 1):
                pass
    if target_state and target_state.is_inanimate:
        message = f"{target_state.character_name} is inanimate and can't change accessories."
        if ctx.guild:
            await ctx.reply(message, mention_author=False)
        else:
            await ctx.send(message)
        return
    state = target_state or find_active_transformation(ctx.author.id, guild_id)
    if not state:
        fallback_state = find_active_transformation(ctx.author.id)
        if fallback_state and ctx.guild and fallback_state.guild_id != guild_id:
            target_guild = bot.get_guild(fallback_state.guild_id)
            guild_name = target_guild.name if target_guild else f"server {fallback_state.guild_id}"
            message = (
                "You're transformed right now, but in a different server. "
                f"Use this command in **{guild_name}** to change those accessories."
            )
        else:
            message = "You need to be transformed to manage accessories."
        if ctx.guild:
            await ctx.reply(message, mention_author=False)
        else:
            await ctx.send(message)
        return

    accessories = list_character_accessories(state.character_name)
    if not accessories:
        message = f"No accessories are available for {state.character_name}."
        if ctx.guild:
            await ctx.reply(message, mention_author=False)
        else:
            await ctx.send(message)
        return

    accessory_states = get_accessory_states(state.character_name, scope=selection_scope)

    # Handle "clearall" command - completely erase accessory data, not just set to "off"
    normalized_accessory_name = (accessory_name or "").strip().lower()
    if normalized_accessory_name in ("clearall", "clear all", "reset", "nuke"):
        # Completely remove all accessory data for this character
        from tfbot.panels import resolve_character_directory, _selection_store_key
        
        directory, _ = resolve_character_directory(state.character_name)
        if directory is None:
            message = f"Unable to clear accessories for {state.character_name}."
            if ctx.guild:
                await ctx.reply(message, mention_author=False)
            else:
                await ctx.send(message)
            return
        
        # Get count before clearing for confirmation message
        all_accessories = list_character_accessories(state.character_name)
        cleared_count = len(all_accessories) if all_accessories else 0
        
        # Clear accessories from ALL scopes for this character (not just current scope)
        # This ensures we completely nuke the sprite layer regardless of scope
        from tfbot.panels import _selection_lookup_keys, _normalize_selection_scope
        
        base_key = directory.name.lower()
        # Clear from default scope
        default_store_key = base_key
        # Clear from current scope if different
        current_store_key = _selection_store_key(directory, selection_scope)
        # Also check for RP scope if it exists
        rp_store_key = f"rp:{base_key}"
        
        # List of all possible keys for this character
        keys_to_clear = {default_store_key, current_store_key, rp_store_key}
        
        # Remove accessories from all keys
        for key in keys_to_clear:
            entry = vn_outfit_selection.get(key)
            if isinstance(entry, dict):
                # Remove the entire "accessories" dictionary from the entry
                entry.pop("accessories", None)
                # If the entry is now empty or only has empty values, remove it entirely
                if not entry:
                    vn_outfit_selection.pop(key, None)
                else:
                    # Add a flag to suppress outfit-level accessories
                    entry["suppress_outfit_accessories"] = True
                    # Keep other data (like outfit, pose) but remove accessories completely
                    vn_outfit_selection[key] = entry
            elif entry is not None:
                # Entry exists but isn't a dict - preserve it but ensure no accessory data
                target_entry = {}
                if isinstance(entry, str):
                    stripped = entry.strip()
                    if stripped:
                        target_entry["outfit"] = stripped
                else:
                    stripped = str(entry).strip()
                    if stripped:
                        target_entry["outfit"] = stripped
                # Add a flag to suppress outfit-level accessories
                target_entry["suppress_outfit_accessories"] = True
                # Explicitly do NOT include "accessories" in the new entry
                vn_outfit_selection[key] = target_entry
        
        # Persist the changes
        persist_outfit_selections()
        
        # Clear avatar cache to force regeneration without any accessory layers
        # Clear both in-memory and disk cache
        from tfbot.panels import compose_game_avatar, VN_CACHE_DIR
        compose_game_avatar.cache_clear()
        
        # Also clear disk cache for this character specifically
        if VN_CACHE_DIR and directory:
            try:
                import shutil
                character_cache_dir = VN_CACHE_DIR / directory.name.lower()
                if character_cache_dir.exists() and character_cache_dir.is_dir():
                    shutil.rmtree(character_cache_dir, ignore_errors=True)
                    logger.info("Cleared disk cache for character: %s", directory.name)
            except Exception as exc:
                logger.warning("Failed to clear disk cache for character %s: %s", directory.name, exc)
        
        schedule_history_refresh()
        
        confirmation = (
            f"All {cleared_count} accessory{'ies' if cleared_count != 1 else 'y'} for {state.character_name} "
            f"have been completely removed from storage. The sprite layer has been nuked. "
            f"Future VN panels will show no accessories."
        )
        if ctx.guild:
            await ctx.reply(confirmation, mention_author=False)
        else:
            await ctx.send(confirmation)
        return

    if not accessory_name:
        lines = []
        for key, label in sorted(accessories.items(), key=lambda item: item[1].lower() if item[1] else item[0]):
            status = accessory_states.get(key, "off")
            display = label or key
            lines.append(f"- {display}: {status}")
        message = (
            f"Accessories for {state.character_name} (scope `{selection_scope or 'default'}`):\n"
            + "\n".join(lines)
        )
        if ctx.guild:
            await ctx.reply(message, mention_author=False)
        else:
            await ctx.send(message)
        return

    resolved_key = _resolve_accessory_key_input(accessory_name, accessories)
    if not resolved_key:
        available_labels = ", ".join(sorted((label or key) for key, label in accessories.items()))
        message = (
            f"Unknown accessory `{accessory_name}`. "
            f"Available options: {available_labels or 'none'}."
        )
        if ctx.guild:
            await ctx.reply(message, mention_author=False)
        else:
            await ctx.send(message)
        return

    new_state = toggle_accessory_state(
        state.character_name,
        resolved_key,
        scope=selection_scope,
    )
    if new_state is None:
        message = "Unable to update that accessory. Please try again."
        if ctx.guild:
            await ctx.reply(message, mention_author=False)
        else:
            await ctx.send(message)
        return

    schedule_history_refresh()
    label = accessories.get(resolved_key) or resolved_key
    confirmation = (
        f"Accessory `{label}` for {state.character_name} set to `{new_state}`. "
        "Future VN panels will use this state."
    )
    if ctx.guild:
        await ctx.reply(confirmation, mention_author=False)
    else:
        await ctx.send(confirmation)


@bot.tree.command(name="accessories", description="List or toggle VN accessories.")
@app_commands.describe(accessory="Select an accessory to toggle. Leave blank to list them.")
@app_commands.autocomplete(accessory=_accessory_autocomplete)
@app_commands.guild_only()
@guard_slash_command_channel
async def slash_accessories_command(
    interaction: discord.Interaction,
    accessory: Optional[str] = None,
) -> None:
    await interaction.response.defer(thinking=True)
    ctx = InteractionContextAdapter(interaction, bot=bot)
    await accessories_command(ctx, accessory_name=accessory or "")
    if not ctx.responded:
        await interaction.followup.send("No accessory change was applied.", ephemeral=True)


async def _handle_slash_say(
    interaction: discord.Interaction,
    character: str,
    text: str,
    *,
    enforce_permissions: bool = True,
) -> None:
    await ensure_state_restored()

    guild = interaction.guild
    actor = interaction.user
    if guild is None or not isinstance(actor, discord.Member):
        await interaction.response.send_message(
            "Use this command from within a server channel.",
            ephemeral=True,
        )
        return

    target_name = (character or "").strip()
    cleaned_content = (text or "").strip()
    if not target_name:
        await interaction.response.send_message("Choose which character should speak.", ephemeral=True)
        return
    if not cleaned_content:
        await interaction.response.send_message("Please provide what the character should say.", ephemeral=True)
        return

    guild_channel = interaction.channel if isinstance(interaction.channel, discord.abc.GuildChannel) else None
    selection_scope = _selection_scope_for_channel(guild_channel)

    if enforce_permissions:
        can_use_command = is_admin(actor) or _actor_has_narrator_power(actor)
        if not can_use_command and ROLEPLAY_COG is not None and guild_channel and ROLEPLAY_COG.is_roleplay_post(guild_channel):
            can_use_command = ROLEPLAY_COG.has_control(actor)
        if not can_use_command:
            await interaction.response.send_message(_privileged_requirement_message("use this command"), ephemeral=True)
            return

    directory_lookup = {name.lower(): name for name in _list_character_directory_names()}
    normalized_target = target_name.strip()
    directory_choice = directory_lookup.get(normalized_target.lower())

    target_state = None
    if directory_choice is None:
        target_state = _find_state_by_folder(guild, target_name)
    if target_state is None:
        folder_character = None
        if directory_choice is not None:
            folder_character = _find_character_by_folder(directory_choice)
        if folder_character is None:
            folder_character = _find_character_by_folder(target_name)
        if folder_character is not None:
            target_state = _build_roleplay_state(folder_character, actor, guild)
        elif directory_choice is not None:
            target_state = _build_roleplay_state(
                TFCharacter(
                    name=directory_choice,
                    avatar_path="",
                    message="",
                    folder=directory_choice,
                ),
                actor,
                guild,
            )
        else:
            inanimate_entry = _find_inanimate_form_by_token(target_name)
            if inanimate_entry is not None:
                target_state = _build_inanimate_roleplay_state(inanimate_entry, actor, guild)
    if target_state is None:
        await interaction.response.send_message(
            f"Couldn't find a character or active TF matching `{target_name}`.",
            ephemeral=True,
        )
        return

    if (
        ROLEPLAY_COG
        and guild_channel
        and ROLEPLAY_COG.is_roleplay_post(guild_channel)
        and ROLEPLAY_COG.dm_user_id
        and _state_matches_folder(target_state, "narrator")
    ):
        target_state.user_id = ROLEPLAY_COG.dm_user_id

    is_ball_character = target_state.is_inanimate and _state_matches_folder(target_state, "ball")
    if target_state.is_inanimate and not is_ball_character:
        await interaction.response.send_message(f"{target_state.character_name} can't speak right now.", ephemeral=True)
        return

    if not interaction.response.is_done():
        await interaction.response.defer(thinking=True)

    reply_context: Optional[ReplyContext] = None
    cleaned_content = cleaned_content.strip()
    if AI_REWRITE_ENABLED and cleaned_content and not cleaned_content.startswith(str(bot.command_prefix)):
        context_snippet = CHARACTER_CONTEXT.get(target_state.character_name) or target_state.character_message
        rewritten = await rewrite_message_for_character(
            original_text=cleaned_content,
            character_name=target_state.character_name,
            character_context=context_snippet,
            user_name=actor.display_name,
        )
        if rewritten and rewritten.strip():
            cleaned_content = rewritten.strip()

    cleaned_content, _ = strip_urls(cleaned_content)
    cleaned_content = cleaned_content.strip()
    if not cleaned_content:
        await interaction.followup.send("There's nothing for the character to say after filtering that text.", ephemeral=True)
        return

    _, member = await fetch_member(target_state.guild_id, target_state.user_id)
    original_name = (
        member.display_name
        if isinstance(member, discord.Member)
        else target_state.original_display_name
        or f"User {target_state.user_id}"
    )

    formatted_segments = parse_discord_formatting(cleaned_content) if cleaned_content else []
    emoji_source = SimpleNamespace(guild=guild)
    custom_emoji_images = await prepare_custom_emoji_images(emoji_source, formatted_segments)

    files: list[discord.File] = []
    payload: dict = {}
    character_display_name = _resolve_character_display_name(guild, actor.id, target_state)

    if MESSAGE_STYLE == "vn" and not target_state.is_inanimate:
        vn_file = render_vn_panel(
            state=target_state,
            message_content=cleaned_content,
            character_display_name=character_display_name,
            original_name=original_name,
            attachment_id=str(interaction.id),
            formatted_segments=formatted_segments,
            custom_emoji_images=custom_emoji_images,
            reply_context=reply_context,
            selection_scope=selection_scope,
        )
        if vn_file:
            files.append(vn_file)

    description = cleaned_content if cleaned_content else "*no message content*"
    if not files:
        embed, avatar_file = await build_legacy_embed(target_state, description)
        if avatar_file:
            files.append(avatar_file)
        payload["embed"] = embed

    send_kwargs: Dict[str, object] = {}
    send_kwargs.update(payload)
    if files:
        send_kwargs["files"] = files
    send_kwargs["allowed_mentions"] = discord.AllowedMentions.none()

    try:
        sent_message = await interaction.followup.send(**send_kwargs, wait=True)
    except discord.HTTPException as exc:
        logger.warning("Failed to send slash say panel: %s", exc)
        await interaction.followup.send("Couldn't deliver that panel.", ephemeral=True)
        return

    if sent_message and cleaned_content:
        _register_relay_message(sent_message.id, target_state.character_name, cleaned_content)


@bot.tree.command(name="say", description="Have a character deliver a line in the TF channel.")
@app_commands.describe(
    character="Character or active TF name to speak as.",
    text="What the character should say.",
)
@app_commands.autocomplete(character=_character_name_autocomplete)
@app_commands.guild_only()
@guard_slash_command_channel
async def slash_say_command(
    interaction: discord.Interaction,
    character: str,
    text: str,
) -> None:
    await _handle_slash_say(interaction, character, text, enforce_permissions=True)


@bot.tree.command(name="n", description="RP Narrator shortcut (RP forum DM/owner only).")
@app_commands.describe(text="What the Narrator should say.")
@app_commands.guild_only()
@guard_slash_command_channel
async def slash_narrator_shortcut(
    interaction: discord.Interaction,
    text: str,
) -> None:
    rp_cog, error = _resolve_roleplay_cog(interaction.channel)
    if error:
        await interaction.response.send_message(error, ephemeral=True)
        return
    ctx = InteractionContextAdapter(interaction, default_ephemeral=True, bot=bot)
    if not await rp_cog._ensure_dm_actor(ctx):
        return
    await _handle_slash_say(interaction, "narrator", text, enforce_permissions=False)


@bot.tree.command(name="b", description=f"RP {TFBOT_NAME}'s Ball shortcut (RP forum DM/owner only).")
@app_commands.describe(text=f"What {TFBOT_NAME}'s Ball should say.")
@app_commands.guild_only()
@guard_slash_command_channel
async def slash_ball_shortcut(
    interaction: discord.Interaction,
    text: str,
) -> None:
    rp_cog, error = _resolve_roleplay_cog(interaction.channel)
    if error:
        await interaction.response.send_message(error, ephemeral=True)
        return
    ctx = InteractionContextAdapter(interaction, default_ephemeral=True, bot=bot)
    if not await rp_cog._ensure_dm_actor(ctx):
        return
    await _handle_slash_say(interaction, "ball", text, enforce_permissions=False)


@bot.event
async def on_command_completion(ctx: commands.Context) -> None:
    command = ctx.command
    message = getattr(ctx, "message", None)
    if command is None or message is None:
        return
    try:
        await log_command_archive_entry(
            actor=ctx.author,
            channel=ctx.channel,
            command_display=command.qualified_name,
            invocation_text=message.content,
            source="prefix",
            guild=ctx.guild,
            invocation_id=str(message.id),
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.debug("Failed to log prefix command completion: %s", exc)


@bot.event
async def on_app_command_completion(interaction: discord.Interaction, command: app_commands.Command) -> None:
    user = interaction.user
    if user is None:
        return
    command_name = getattr(command, "qualified_name", getattr(command, "name", "unknown"))
    try:
        invocation_text = _format_interaction_invocation(interaction, command)
    except Exception as exc:  # pragma: no cover - fallback to simple label
        logger.debug("Unable to format interaction command %s: %s", command_name, exc)
        invocation_text = f"/{command_name}"
    try:
        await log_command_archive_entry(
            actor=user,
            channel=interaction.channel,
            command_display=command_name,
            invocation_text=invocation_text,
            source="slash",
            guild=interaction.guild,
            invocation_id=str(interaction.id),
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.debug("Failed to log slash command completion: %s", exc)

@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return None

    logger.info(
        "Message %s from %s in channel %s",
        message.id,
        message.author.id,
        getattr(message.channel, "id", "dm"),
    )

    command_invoked = False
    allowed_ids = _allowed_instance_channel_ids()
    channel_allowed = (
        not message.guild
        or not allowed_ids
        or _channel_matches_allowed(message.channel, allowed_ids)
    )

    ctx = await bot.get_context(message)
    if ctx.command:
        if channel_allowed:
            command_invoked = True
            logger.debug(
                "Invoking command %s by %s in channel %s",
                ctx.command.qualified_name,
                message.author.id,
                getattr(message.channel, "id", None),
            )
            await bot.invoke(ctx)
        else:
            logger.debug(
                "Ignoring command %s by %s in disallowed channel %s",
                ctx.command.qualified_name,
                message.author.id,
                getattr(message.channel, "id", None),
            )
    elif message.content.startswith(str(bot.command_prefix)):
        logger.debug(
            "Command-like message ignored (ctx.command missing) content=%s author=%s channel=%s",
            message.content,
            message.author.id,
            getattr(message.channel, "id", None),
        )

    if command_invoked:
        return None

    # Check for game thread FIRST (before other systems)
    is_game_thread = (
        GAME_BOARD_MANAGER is not None
        and isinstance(message.channel, discord.Thread)
        and GAME_BOARD_MANAGER.is_game_thread(message.channel)
    )
    if is_game_thread:
        handled = await GAME_BOARD_MANAGER.handle_message(message, command_invoked=command_invoked)
        if handled:
            return None

    is_gacha_channel = (
        GACHA_MANAGER is not None
        and isinstance(message.channel, discord.TextChannel)
        and message.guild is not None
        and message.channel.id == GACHA_MANAGER.channel_id
    )
    if is_gacha_channel:
        await GACHA_MANAGER.handle_message(message, command_invoked=command_invoked)
        return None

    if not CLASSIC_ENABLED:
        return None

    channel_id = getattr(message.channel, "id", None)
    is_roleplay_forum_post = ROLEPLAY_COG is not None and ROLEPLAY_COG.is_roleplay_post(message.channel)
    is_admin_user = is_admin(message.author)
    if message.guild and message.guild.owner_id == message.author.id and not is_roleplay_forum_post:
        logger.debug("Ignoring message %s from server owner %s", message.id, message.author.id)
        return None
    if message.guild and allowed_ids and not _channel_matches_allowed(message.channel, allowed_ids):
        logger.info(
            "Skipping message %s: channel %s is not in the monitored TF/RP channels %s.",
            message.id,
            channel_id,
            ", ".join(str(cid) for cid in sorted(allowed_ids)),
        )
        return None

    profile: Optional["GachaProfile"] = None
    gacha_equipped = False
    gacha_handled = False
    if not is_roleplay_forum_post and GACHA_MANAGER is not None and message.guild:
        profile = await GACHA_MANAGER.ensure_profile(message.guild.id, message.author.id)
        allowed = await GACHA_MANAGER.enforce_spam_policy(message, profile=profile)
        if not allowed:
            return None
        await GACHA_MANAGER.award_message_reward(message, profile=profile)
        gacha_equipped = bool(profile.equipped_character)
        if gacha_equipped:
            gacha_handled = await GACHA_MANAGER.relay_classic_message(
                message,
                profile=profile,
            )
            if gacha_handled:
                return None

    if message.guild and not gacha_equipped:
        key = state_key(message.guild.id, message.author.id)
        state = active_transformations.get(key)
        if state:
            reply_reference: Optional[discord.MessageReference] = (
                message.to_reference(fail_if_not_exists=False) if message.reference else None
            )
            await relay_transformed_message(message, state, reference=reply_reference)
            return None

    if message.guild and GACHA_MANAGER is not None and gacha_equipped:
        logger.debug(
            "Skipping TF roll for user %s in guild %s: gacha character equipped.",
            message.author.id,
            message.guild.id,
        )
        return None

    if is_roleplay_forum_post:
        return None

    logger.info(
        "Message intercepted (admin=%s): user %s in channel %s",
        is_admin_user,
        message.author.id,
        channel_id,
    )

    roll = random.random()
    logger.debug("Roll for message %s is %.4f vs threshold %.4f", message.id, roll, TF_CHANCE)
    if roll <= TF_CHANCE:
        state = await handle_transformation(message)
        if state:
            logger.debug("TF triggered for message %s in channel %s", message.id, channel_id)


# ============================================================================
# 3.5 PREFIX COMMAND SYSTEM - Added to coexist with 4.0 slash commands
# These functions and commands are from 3.5 and work alongside 4.0's system
# ============================================================================

# Helper functions for 3.5 token-based matching system
def _token_variants(token: str) -> set[str]:
    normalized_token = (token or "").strip().lower()
    if not normalized_token:
        return set()
    variants = {normalized_token}

    def _add(value: str) -> None:
        cleaned = value.strip()
        if cleaned:
            variants.add(cleaned)

    if "_" in normalized_token:
        _add(normalized_token.replace("_", " "))
        _add(normalized_token.replace("_", ""))
        _add(normalized_token.split("_", 1)[0])
    if "-" in normalized_token:
        _add(normalized_token.replace("-", " "))
        _add(normalized_token.replace("-", ""))
        _add(normalized_token.split("-", 1)[0])
    return variants


def _name_matches_token(name: str, token: str) -> bool:
    name_normalized = (name or "").strip().lower()
    if not name_normalized:
        return False
    first_token = name_normalized.split(" ", 1)[0]
    variants = _token_variants(token)
    if not variants:
        return False
    for variant in variants:
        if variant == name_normalized or variant == first_token:
            return True
    return False


def _character_matches_token(character: TFCharacter, token: str) -> bool:
    variants = _token_variants(token)
    if not variants:
        return False
    folder_name_raw = (character.folder or "").strip()
    if folder_name_raw:
        folder_normalized = folder_name_raw.replace("\\", "/").strip("/").lower()
        folder_candidates = {folder_normalized}
        last_segment = folder_normalized.rsplit("/", 1)[-1]
        folder_candidates.add(last_segment)
        for folder_candidate in folder_candidates:
            if folder_candidate in variants:
                return True
    name_normalized = character.name.lower()
    first_token = name_normalized.split(" ", 1)[0]
    for variant in variants:
        if variant == name_normalized or variant == first_token:
            return True
    return False


def _find_state_by_token(guild: discord.Guild, token: str) -> Optional[TransformationState]:
    """3.5 token-based state finder - finds state by character name, user mention, or display name."""
    normalized = (token or "").strip()
    if not normalized:
        return None
    user_id = _extract_user_id_from_token(normalized)
    if user_id is not None:
        state = active_transformations.get(state_key(guild.id, user_id))
        if state:
            return state
    folder_tokens = _folder_lookup_tokens(normalized)
    if folder_tokens:
        for state in active_transformations.values():
            if state.guild_id != guild.id:
                continue
            folder_token = _state_folder_token(state)
            if folder_token and folder_token in folder_tokens:
                return state
    token_variants = _token_variants(normalized)
    for state in active_transformations.values():
        if state.guild_id != guild.id:
            continue
        if _name_matches_token(state.character_name, normalized):
            return state
        character_entry = CHARACTER_BY_NAME.get(state.character_name.strip().lower())
        if character_entry and _character_matches_token(character_entry, normalized):
            return state
        member_obj = guild.get_member(state.user_id)
        if member_obj:
            profile = member_profile_name(member_obj).lower()
            profile_first = profile.split(" ", 1)[0]
            display = member_obj.display_name.lower()
            display_first = display.split(" ", 1)[0]
            username = member_obj.name.lower()
            username_first = username.split(" ", 1)[0]
            if (
                profile in token_variants
                or profile_first in token_variants
                or display in token_variants
                or display_first in token_variants
                or username in token_variants
                or username_first in token_variants
            ):
                return state
    return None


def _find_character_by_token(token: str) -> Optional[TFCharacter]:
    """3.5 token-based character finder."""
    normalized = (token or "").strip()
    if not normalized:
        return None
    for folder_token in _folder_lookup_tokens(normalized):
        match = CHARACTER_BY_FOLDER.get(folder_token)
        if match:
            return match
    for character in CHARACTER_POOL:
        if _character_matches_token(character, normalized):
            return character
    return None


def _has_special_reroll_access(state: Optional[TransformationState]) -> bool:
    """Check if a state has special reroll access (configured special forms)."""
    if state is None:
        return False
    # Use folder OR name to match the earlier definition at line 483
    token = state.character_folder or state.character_name
    return _is_special_reroll_name(token)


def _format_special_reroll_hint_35(character_name: str) -> Optional[str]:
    """3.5 version of special reroll hint formatter."""
    if not _is_special_reroll_name(character_name):
        return None
    return (
        "```diff\n"
        f"- {character_name} perk unlocked! Use `!reroll character` for a random swap or `!reroll character character` to force a form.\n"
        "```"
    )


# 3.5 PREFIX COMMANDS - Exact implementations from 3.5
@bot.command(name="synreset", hidden=True)
@commands.guild_only()
@guard_prefix_command_channel
async def prefix_synreset_35(ctx: commands.Context):
    """3.5 version of synreset command."""
    author = ctx.author
    if not isinstance(author, discord.Member):
        await ctx.reply("This command can only be used inside a server.", mention_author=False)
        return None
    if not is_admin(author):
        await ctx.reply("You lack permission to run this command.", mention_author=False)
        return None

    await ensure_state_restored()

    try:
        await ctx.message.delete()
    except discord.HTTPException:
        pass

    guild = ctx.guild
    await ctx.channel.send("Initiating full TF reset...", delete_after=5)

    states = [
        state for state in list(active_transformations.values()) if state.guild_id == guild.id
    ]
    restored = 0
    for state in states:
        await revert_transformation(state, expired=False)
        restored += 1

    await send_history_message(
        "TF Reset",
        f"Triggered by: **{author.name}**\nRestored TFs: {restored}",
    )
    await ctx.channel.send(f"TF reset completed. Restored {restored} transformations.", delete_after=10)


def _actor_has_narrator_power_35(member: Optional[discord.Member]) -> bool:
    """3.5 version of narrator power check using token matching."""
    if member is None or member.guild is None:
        return False
    state = find_active_transformation(member.id, member.guild.id)
    if not state:
        return False
    if _state_has_privileged_access(state):
        return True
    if not PRIVILEGED_FORM_TOKENS:
        return False
    return any(_name_matches_token(state.character_name, token) for token in PRIVILEGED_FORM_TOKENS)


@bot.command(name="reroll")
@commands.guild_only()
async def prefix_reroll_35(ctx: commands.Context, *, args: str = ""):
    """3.5 version of reroll command."""
    author = ctx.author
    if not isinstance(author, discord.Member):
        await ctx.reply("This command can only be used inside a server.", mention_author=False)
        return None
    
    # Check if this is a game thread - if so, apply game-specific restrictions
    game_state = None
    if GAME_BOARD_MANAGER and isinstance(ctx.channel, discord.Thread) and GAME_BOARD_MANAGER.is_game_thread(ctx.channel):
        game_state = GAME_BOARD_MANAGER._get_game_state_for_context(ctx)
        if game_state:
            # In game threads, only GM can reroll
            if not GAME_BOARD_MANAGER._is_gm(author, game_state):
                await ctx.reply("Only the GM can reroll characters in game threads.", mention_author=False)
                return None
    
    author_is_admin = is_admin(author)

    guild = ctx.guild
    if guild is None:
        await ctx.reply("This command can only be used inside a server.", mention_author=False)
        return None
    now = utc_now()
    author_state = find_active_transformation(author.id, guild.id)
    author_has_special_reroll = _has_special_reroll_access(author_state)
    can_force_reroll = author_is_admin or author_has_special_reroll
    
    # Check cooldown EARLY - before any other processing
    # If on cooldown, return immediately without processing query
    if not author_is_admin and not author_has_special_reroll:
        last_reroll_at = get_last_reroll_timestamp(guild.id, author.id)
        if last_reroll_at is not None:
            cooldown_end = last_reroll_at + timedelta(hours=24)
            if cooldown_end > now:
                remaining = cooldown_end - now
                remaining_seconds = max(int(remaining.total_seconds()), 0)
                hours, remainder = divmod(remaining_seconds, 3600)
                minutes = remainder // 60
                if hours and minutes:
                    when_text = f"{hours} hour{'s' if hours != 1 else ''} and {minutes} minute{'s' if minutes != 1 else ''}"
                elif hours:
                    when_text = f"{hours} hour{'s' if hours != 1 else ''}"
                elif minutes:
                    when_text = f"{minutes} minute{'s' if minutes != 1 else ''}"
                else:
                    when_text = "less than a minute"
                await ctx.reply(
                    f"You've already used your reroll. You can reroll again in {when_text}.",
                    mention_author=False,
                )
                return  # Exit immediately - don't process query at all
    
    # Only process query if not on cooldown
    target_member: Optional[discord.Member] = None
    target_is_admin = False
    state: Optional[TransformationState] = None

    forced_character: Optional[TFCharacter] = None
    forced_inanimate: Optional[Dict[str, object]] = None

    query = args.strip()
    forced_token: Optional[str] = None
    if query:
        parts = query.split()
        first_token_input = parts[0]
        first_token = first_token_input.lower()
        first_token_variants = _token_variants(first_token_input)
        # DON'T check can_force_reroll here - wait until after state is resolved
        if len(parts) > 1:
            forced_token = parts[1].lower()

        def _state_matches(state: TransformationState) -> bool:
            if state.guild_id != guild.id:
                return False
            if _name_matches_token(state.character_name, first_token_input):
                return True
            character_entry = CHARACTER_BY_NAME.get(state.character_name.strip().lower())
            if character_entry and _character_matches_token(character_entry, first_token_input):
                return True
            member_obj = guild.get_member(state.user_id)
            if member_obj:
                profile = member_profile_name(member_obj).lower()
                profile_first = profile.split(" ", 1)[0]
                if profile in first_token_variants or profile_first in first_token_variants:
                    return True
            return False

        matching_states = [s for s in active_transformations.values() if _state_matches(s)]
        target_member = None
        if matching_states:
            state = matching_states[0]
            _, target_member = await fetch_member(state.guild_id, state.user_id)
        else:
            caller_state = author_state
            if caller_state:
                if _name_matches_token(caller_state.character_name, first_token_input):
                    state = caller_state
                    target_member = author
                else:
                    entry = CHARACTER_BY_NAME.get(caller_state.character_name.strip().lower())
                    if entry and _character_matches_token(entry, first_token_input):
                        state = caller_state
                        target_member = author
            if state is None:
                member_candidate = discord.utils.find(
                    lambda m: (
                        m.display_name.lower() in first_token_variants
                        or m.display_name.split(" ", 1)[0].lower() in first_token_variants
                        or m.name.lower() in first_token_variants
                        or m.name.split(" ", 1)[0].lower() in first_token_variants
                    ),
                    guild.members if guild else [],
                )
                if member_candidate:
                    state = find_active_transformation(member_candidate.id, guild.id)
                    target_member = member_candidate
                else:
                    state = None
            if state is None:
                await ctx.reply(
                    f"No active transformation found for `{first_token_input}`.",
                    mention_author=False,
                )
                return None
            if target_member is None:
                _, target_member = await fetch_member(state.guild_id, state.user_id)
        
        # NOW that state is fully resolved, re-check author_state and permissions
        # This ensures we have the most up-to-date state with folder populated
        author_state = find_active_transformation(author.id, guild.id)
        author_has_special_reroll = _has_special_reroll_access(author_state)
        can_force_reroll = author_is_admin or author_has_special_reroll
        
        # NOW check if they can force reroll (after state is resolved)
        if forced_token and not can_force_reroll:
            await ctx.reply(
                _privileged_requirement_message("force a reroll into a specific form"),
                mention_author=False,
            )
            return None
        
        if target_member is None:
            await ctx.reply(
                f"Unable to locate the member transformed into {state.character_name}.",
                mention_author=False,
            )
            return None

        target_is_admin = isinstance(target_member, discord.Member) and is_admin(target_member)

        # Admin protection: Non-admins can't reroll admins (if enabled)
        if ADMIN_PROTECTION_ENABLED:
            if target_is_admin and not author_is_admin:
                await ctx.reply(
                    "Only admins can reroll other admins.",
                    mention_author=False,
                )
                return None

        if (
            not author_is_admin
            and target_member.id == author.id
            and not author_has_special_reroll
        ):
            await ctx.reply(
                "You can't use your reroll on yourself. Ask another player to reroll you instead.",
                mention_author=False,
            )
            return None

        if forced_token:
            forced_character = _find_character_by_folder(forced_token)
            if forced_character is None:
                forced_character = next(
                    (character for character in CHARACTER_POOL if _character_matches_token(character, forced_token)),
                    None,
                )
            if forced_character is None:
                forced_inanimate = next(
                    (
                        entry
                        for entry in INANIMATE_FORMS
                        if str(entry.get("name", "")).split(" ", 1)[0].lower() == forced_token
                        or str(entry.get("name", "")).lower() == forced_token
                    ),
                    None,
                )
            if forced_character is None and forced_inanimate is None:
                await ctx.reply(
                    f"Unknown target `{forced_token}`. Provide a valid first name.",
                    mention_author=False,
                )
                return None
            # Toggleable restriction: Only admins or Ball/Narrator characters can force Ball/Narrator (if enabled)
            if SPECIAL_CHARACTERS_ADMIN_ONLY:
                if forced_character is not None and _is_special_reroll_name(forced_character.name):
                    if not author_is_admin and not author_has_special_reroll:
                        await ctx.reply(
                            f"Only admins or {SPECIAL_FORM_SUBJECT} can force someone into {SPECIAL_FORM_TARGET}.",
                            mention_author=False,
                        )
                        return None
                if forced_inanimate is not None and _is_special_reroll_name(str(forced_inanimate.get("name", ""))):
                    if not author_is_admin and not author_has_special_reroll:
                        await ctx.reply(
                            f"Only admins or {SPECIAL_FORM_SUBJECT} can force someone into {SPECIAL_FORM_TARGET}.",
                            mention_author=False,
                        )
                        return None
            forced_special_name = None
            if forced_character is not None and _is_special_reroll_name(forced_character.name):
                forced_special_name = forced_character.name
            elif forced_inanimate is not None and _is_special_reroll_name(
                str(forced_inanimate.get("name", ""))
            ):
                forced_special_name = str(forced_inanimate.get("name", ""))
            if (
                forced_character is not None
                and _is_admin_only_random_name(forced_character.name)
                and not author_is_admin
                and not target_is_admin
            ):
                await ctx.reply(
                    f"You can only force {TFBOT_NAME} or Circe onto admins unless you're an admin yourself.",
                    mention_author=False,
                )
                return None
            # Admin protection: Ball/Narrator perks can't force into Ball/Narrator (if enabled)
            if ADMIN_PROTECTION_ENABLED:
                if forced_special_name and author_has_special_reroll and not author_is_admin:
                    await ctx.reply(
                        f"{SPECIAL_FORM_CAPITALIZED} TFs can't force others into {SPECIAL_FORM_TARGET}. Ask an admin or owner.",
                        mention_author=False,
                    )
                    return None
    else:
        if not author_is_admin:
            await ctx.reply(
                "You need to specify someone to reroll, e.g. `!reroll charname`.",
                mention_author=False,
            )
            return None
        target_member = author
        key = state_key(guild.id, target_member.id)
        state = author_state
        if state is None:
            await ctx.reply(
                "You are not currently transformed; nothing to reroll.",
                mention_author=False,
            )
            return None
        target_is_admin = isinstance(target_member, discord.Member) and is_admin(target_member)

    # Admin protection: Non-admins can't reroll admins (if enabled)
    if ADMIN_PROTECTION_ENABLED:
        if target_is_admin and not author_is_admin:
            await ctx.reply(
                "Only admins can reroll other admins.",
                mention_author=False,
            )
            return None

    key = state_key(guild.id, target_member.id)
    current_state = active_transformations.get(key)
    if current_state is None or current_state != state:
        await ctx.reply(
            "Unable to locate the transformation for this member.",
            mention_author=False,
        )
        return None

    await _unswap_and_announce(
        guild.id,
        target_member.id,
        reason=f"{target_member.display_name}'s swap chain reset before reroll.",
        channel=ctx.channel,
    )
    state = active_transformations.get(key)
    if state is None:
        await ctx.reply(
            "Their transformation reset during the swap cleanup. Try again if they are still transformed.",
            mention_author=False,
        )
        return None

    used_names = {
        current_state.character_name
        for current_key, current_state in active_transformations.items()
        if current_key != key
    }

    # Cooldown already checked earlier - skip duplicate check

    forced_mode = forced_character is not None or forced_inanimate is not None

    new_name: str
    new_folder: Optional[str]
    new_avatar_path: str
    new_message: str
    new_is_inanimate: bool
    new_responses: Tuple[str, ...]

    if forced_inanimate is not None:
        new_name = str(forced_inanimate.get("name") or "Mystery Relic")
        new_folder = None
        new_avatar_path = str(forced_inanimate.get("avatar_path") or "")
        new_message = str(forced_inanimate.get("message") or "You feel unsettlingly still.")
        responses_raw = forced_inanimate.get("responses") or []
        if isinstance(responses_raw, (list, tuple)):
            new_responses = tuple(str(item).strip() for item in responses_raw if str(item).strip())
        else:
            new_responses = tuple()
        if not new_responses:
            new_responses = (new_message,)
        new_is_inanimate = True
    elif forced_character is not None:
        new_name = forced_character.name
        new_folder = forced_character.folder
        new_avatar_path = forced_character.avatar_path
        new_message = forced_character.message
        new_responses = tuple()
        new_is_inanimate = False
    else:
        available_characters = [
            character
            for character in CHARACTER_POOL
            if character.name not in used_names and character.name != state.character_name
        ]
        if not target_is_admin:
            available_characters = [
                character
                for character in available_characters
                if not _is_admin_only_random_name(character.name)
                # Toggleable restriction: Only admins can reroll into Ball/Narrator (if enabled)
                and (not SPECIAL_CHARACTERS_ADMIN_ONLY or not _is_special_reroll_name(character.name))
            ]
        # Admin protection: Ball/Narrator perks can't reroll into Ball/Narrator (if enabled)
        if ADMIN_PROTECTION_ENABLED:
            if author_has_special_reroll and not author_is_admin:
                available_characters = [
                    character
                    for character in available_characters
                    if not _is_special_reroll_name(character.name)
                ]
        if not available_characters:
            await ctx.reply(
                "No alternative characters are available to reroll right now.",
                mention_author=False,
            )
            return None
        chosen = random.choice(available_characters)
        new_name = chosen.name
        new_folder = chosen.folder
        new_avatar_path = chosen.avatar_path
        new_message = chosen.message
        new_responses = tuple()
        new_is_inanimate = False

    if new_name == state.character_name:
        await ctx.reply(
            f"They are already transformed into {new_name}.",
            mention_author=False,
        )
        return None
    if new_name in used_names:
        await ctx.reply(
            f"{new_name} is already in use by another transformation.",
            mention_author=False,
        )
        return None

    previous_character = state.character_name
    state.character_name = new_name
    state.character_folder = new_folder
    state.character_avatar_path = new_avatar_path
    state.character_message = new_message
    state.avatar_applied = False
    state.is_inanimate = new_is_inanimate
    state.inanimate_responses = new_responses

    character_token = new_folder or new_name
    duration_label, guaranteed_duration = _choose_reroll_duration(new_is_inanimate, character_token)
    state.duration_label = duration_label
    
    state.started_at = now
    state.expires_at = now + guaranteed_duration
    existing_task = revert_tasks.get(key)
    if existing_task:
        existing_task.cancel()
    revert_tasks[key] = asyncio.create_task(
        _schedule_revert(state, guaranteed_duration.total_seconds())
    )

    persist_states()

    if not new_is_inanimate:
        increment_tf_stats(guild.id, target_member.id, new_name)
    if not author_is_admin and not author_has_special_reroll:
        record_reroll_timestamp(guild.id, author.id, now)

    history_details = (
        f"Triggered by: **{author.display_name}**\n"
        f"Member: **{target_member.display_name}**\n"
        f"Previous Character: **{previous_character}**\n"
        f"New Character: **{new_name}**"
    )
    if forced_mode:
        history_details += "\nReason: Forced reroll override."
    await send_history_message(
        "TF Rerolled",
        history_details,
    )

    original_name = member_profile_name(target_member)
    if forced_mode:
        custom_template = (
            "barely has time to react before The BunniBot swoops in and swaps them straight into {character}. BunniBot just had to spice things up."
        )
        response_text = _format_character_message(
            custom_template,
            original_name,
            target_member.mention,
            state.duration_label,
            new_name,
        )
    else:
        base_message = _format_character_message(
            new_message,
            original_name,
            target_member.mention,
            state.duration_label,
            new_name,
        )
        if author_is_admin:
            response_text = base_message
        else:
            response_text = (
                f"{author.display_name} cashes in their reroll on {target_member.mention}! {base_message}"
            )
    special_hint = _format_special_reroll_hint_35(new_name)
    if special_hint:
        response_text = f"{response_text}\n{special_hint}"
    emoji_prefix = _get_magic_emoji(guild)
    try:
        await ctx.channel.send(
            f"{emoji_prefix} {response_text}",
            allowed_mentions=discord.AllowedMentions(users=[target_member]),
        )
    except discord.HTTPException as exc:
        logger.warning("Failed to announce reroll in channel %s: %s", ctx.channel.id, exc)

    try:
        await ctx.message.delete()
    except discord.HTTPException:
        pass

    summary_message = f"{target_member.display_name} has been rerolled into **{new_name}**."
    if forced_mode:
        summary_message += " (BunniBot insisted on this one.)"
    await ctx.send(
        summary_message,
        delete_after=10,
    )
    
    # Sync game state after reroll completes (if in game thread)
    if game_state and target_member and ctx.guild and GAME_BOARD_MANAGER:
        if target_member.id in game_state.players:
            game_state.players[target_member.id].character_name = new_name
            await GAME_BOARD_MANAGER._save_game_state(game_state)
            await GAME_BOARD_MANAGER._log_action(game_state, f"{target_member.display_name} character rerolled to {new_name}")


async def _handle_pillow_command(ctx: commands.Context, target: str) -> None:
    author = ctx.author
    guild = ctx.guild
    if guild is None or not isinstance(author, discord.Member):
        logger.info("Undopillow blocked: non-guild context user=%s", getattr(author, "id", None))
        await ctx.reply(
            "This command can only be used inside a server.",
            mention_author=False,
        )
        return

    await ensure_state_restored()

    author_state = find_active_transformation(author.id, guild.id)
    author_is_admin = is_admin(author)
    author_has_special = _has_special_reroll_access(author_state)
    if not (author_is_admin or author_has_special):
        logger.info(
            "Undopillow blocked: insufficient privileges user=%s admin=%s special=%s",
            author.id,
            author_is_admin,
            author_has_special,
        )
        await ctx.reply(
            "Only admins or privileged forms can use this command.",
            mention_author=False,
        )
        return

    search_token = target.strip()
    slash_member = getattr(ctx, "_slash_target_member", None)
    slash_folder = getattr(ctx, "_slash_target_folder", None)
    if not search_token:
        if slash_member is not None:
            search_token = slash_member.mention
        elif slash_folder:
            search_token = slash_folder.strip()

    target_state: Optional[TransformationState]
    target_member: Optional[discord.Member] = None

    if search_token:
        target_state = _find_state_by_token(guild, search_token)
        if target_state:
            target_member = guild.get_member(target_state.user_id)
            if target_member is None:
                _, target_member = await fetch_member(guild.id, target_state.user_id)
    else:
        target_state = active_transformations.get(state_key(guild.id, author.id))
        target_member = author

    if target_state is None:
        logger.info(
            "Undopillow ignored: no active TF for token=%s user=%s",
            search_token or "<self>",
            author.id,
        )
        await ctx.reply(
            "I couldn't find an active transformation for that target.",
            mention_author=False,
        )
        return

    if target_state.is_inanimate:
        await ctx.reply(
            "That form is already an object—there's nowhere to wrap a pillow cover!",
            mention_author=False,
        )
        return

    if target_member is None:
        _, target_member = await fetch_member(guild.id, target_state.user_id)
        if target_member is None:
            await ctx.reply(
                "I couldn't look up that member.",
                mention_author=False,
            )
            return

    if target_state.is_pillow:
        await ctx.reply(
            f"{target_member.display_name} is already a cuddly pillow.",
            mention_author=False,
        )
        return

    if target_member.id != author.id and not author_is_admin:
        await ctx.reply(
            "Only admins can pillow other people.",
            mention_author=False,
        )
        return

    target_state.is_pillow = True
    persist_states()

    duration_hint = target_state.duration_label or "the rest of their TF"
    await ctx.reply(
        f"{author.display_name} stuffs {target_member.display_name} into a body pillowcase! "
        f"They'll stay a pillow for {duration_hint}.",
        mention_author=False,
    )
    await send_history_message(
        "TF Modifier",
        f"Pillow form applied to **{target_member.display_name}** ({target_state.character_name}) "
        f"by **{author.display_name}**.",
    )


async def _handle_undopillow_command(ctx: commands.Context, target: str) -> None:
    author = ctx.author
    guild = ctx.guild
    if guild is None or not isinstance(author, discord.Member):
        await ctx.reply(
            "This command can only be used inside a server.",
            mention_author=False,
        )
        return

    await ensure_state_restored()

    author_state = find_active_transformation(author.id, guild.id)
    author_is_admin = is_admin(author)
    author_has_special = _has_special_reroll_access(author_state)
    if not (author_is_admin or author_has_special):
        await ctx.reply(
            "Only admins or privileged forms can use this command.",
            mention_author=False,
        )
        return

    search_token = target.strip()
    slash_member = getattr(ctx, "_slash_target_member", None)
    slash_folder = getattr(ctx, "_slash_target_folder", None)
    if not search_token:
        if slash_member is not None:
            search_token = slash_member.mention
        elif slash_folder:
            search_token = slash_folder.strip()

    target_state: Optional[TransformationState]
    target_member: Optional[discord.Member] = None

    if search_token:
        target_state = _find_state_by_token(guild, search_token)
        if target_state:
            target_member = guild.get_member(target_state.user_id)
            if target_member is None:
                _, target_member = await fetch_member(guild.id, target_state.user_id)
    else:
        target_state = active_transformations.get(state_key(guild.id, author.id))
        target_member = author

    if target_state is None:
        await ctx.reply(
            "I couldn't find an active transformation for that target.",
            mention_author=False,
        )
        return

    if target_member is None:
        _, target_member = await fetch_member(guild.id, target_state.user_id)
        if target_member is None:
            logger.info(
                "Undopillow blocked: member lookup failed guild=%s target=%s",
                guild.id,
                target_state.user_id,
            )
            await ctx.reply(
                "I couldn't look up that member.",
                mention_author=False,
            )
            return

    if target_member.id != author.id and not author_is_admin:
        logger.info(
            "Undopillow blocked: non-admin tried to affect others actor=%s target=%s",
            author.id,
            target_member.id,
        )
        await ctx.reply(
            "Only admins can undo pillow forms on other people.",
            mention_author=False,
        )
        return

    if not target_state.is_pillow:
        logger.info(
            "Undopillow ignored: target not pillow actor=%s target=%s",
            author.id,
            target_member.id,
        )
        await ctx.reply(
            f"{target_member.display_name} isn't a pillow right now.",
            mention_author=False,
        )
        return

    target_state.is_pillow = False
    persist_states()

    await ctx.reply(
        f"{author.display_name} lets {target_member.display_name} stretch out of their pillowcase.",
        mention_author=False,
    )
    logger.info(
        "Undopillow applied: actor=%s target=%s character=%s",
        author.id,
        target_member.id,
        target_state.character_name,
    )
    await send_history_message(
        "TF Modifier",
        f"Pillow form removed from **{target_member.display_name}** ({target_state.character_name}) "
        f"by **{author.display_name}**.",
    )


@bot.command(name="pillow")
@commands.guild_only()
@guard_prefix_command_channel
async def pillow_command(ctx: commands.Context, *, target: str = ""):
    await _handle_pillow_command(ctx, target)


@bot.tree.command(name="pillow", description="Wrap a transformed member into a body pillow.")
@app_commands.describe(
    who_member="Member to pillow (defaults to yourself).",
    who_character="Folder or character token to target instead.",
)
@app_commands.autocomplete(who_character=_character_name_autocomplete)
@app_commands.guild_only()
@guard_slash_command_channel
async def slash_pillow_command(
    interaction: discord.Interaction,
    who_member: Optional[discord.Member] = None,
    who_character: Optional[str] = None,
) -> None:
    await interaction.response.defer(thinking=True)
    ctx = InteractionContextAdapter(interaction, bot=bot)
    ctx._slash_target_member = who_member
    ctx._slash_target_folder = who_character
    await _handle_pillow_command(ctx, "")
    if not ctx.responded:
        await interaction.followup.send("No pillow transformation was applied.", ephemeral=True)


@bot.command(name="undopillow")
@commands.guild_only()
@guard_prefix_command_channel
async def undopillow_command(ctx: commands.Context, *, target: str = ""):
    await _handle_undopillow_command(ctx, target)


@bot.tree.command(name="undopillow", description="Remove a body pillow modifier from a transformed member.")
@app_commands.describe(
    who_member="Member to restore (defaults to yourself).",
    who_character="Folder or character token to target instead.",
)
@app_commands.autocomplete(who_character=_character_name_autocomplete)
@app_commands.guild_only()
@guard_slash_command_channel
async def slash_undopillow_command(
    interaction: discord.Interaction,
    who_member: Optional[discord.Member] = None,
    who_character: Optional[str] = None,
) -> None:
    await interaction.response.defer(thinking=True)
    ctx = InteractionContextAdapter(interaction, bot=bot)
    ctx._slash_target_member = who_member
    ctx._slash_target_folder = who_character
    await _handle_undopillow_command(ctx, "")
    if not ctx.responded:
        await interaction.followup.send("No pillow modifier was removed.", ephemeral=True)


@bot.command(name="tf", aliases=["TF"])
@guard_prefix_command_channel
async def prefix_tf_35(ctx: commands.Context):
    """3.5 version of tf stats command."""
    guild_id = ctx.guild.id if ctx.guild else None
    if guild_id is None:
        await ctx.reply(
            "Run this command from a server so I know which TF roster to check.",
            mention_author=False,
        )
        return None

    guild_data = tf_stats.get(str(guild_id), {})
    user_data = guild_data.get(str(ctx.author.id))

    if not user_data:
        try:
            await ctx.message.delete()
        except discord.HTTPException:
            pass
        try:
            await ctx.author.send(
                "You haven't experienced any transformations yet."
            )
        except discord.Forbidden:
            await ctx.reply(
                "I couldn't DM you. Please enable direct messages from server members.",
                mention_author=False,
                delete_after=10,
            )
        return None

    embed = discord.Embed(
        title="Transformation Stats",
        color=0x9B59B6,
        timestamp=utc_now(),
    )
    avatar_url = (
        ctx.author.display_avatar.url if ctx.author.display_avatar else None
    )
    embed.set_author(
        name=ctx.author.display_name,
        icon_url=avatar_url,
    )
    embed.add_field(
        name="Total Transformations",
        value=str(user_data.get("total", 0)),
        inline=False,
    )

    characters = user_data.get("characters", {})
    if characters:
        sorted_chars = sorted(characters.items(), key=lambda item: item[1], reverse=True)
        lines = [f"- {name}: **{count}**" for name, count in sorted_chars]
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        for line in lines:
            line_len = len(line) + 1
            if current and current_len + line_len > 1000:
                chunks.append("\n".join(current))
                current = []
                current_len = 0
            current.append(line)
            current_len += line_len
        if current:
            chunks.append("\n".join(current))
        for idx, chunk in enumerate(chunks):
            name = "By Character" if idx == 0 else "\u200b"
            embed.add_field(name=name, value=chunk or "\u200b", inline=False)

    key = state_key(guild_id, ctx.author.id)
    current_state = active_transformations.get(key)
    if current_state:
        remaining = max(
            (current_state.expires_at - utc_now()).total_seconds(),
            0,
        )
        minutes, seconds = divmod(int(remaining), 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            remaining_str = f"{hours}h {minutes}m {seconds}s"
        elif minutes:
            remaining_str = f"{minutes}m {seconds}s"
        else:
            remaining_str = f"{seconds}s"

        embed.add_field(
            name="Current Transformation",
            value=f"Character: **{current_state.character_name}**\nTime left: `{remaining_str}`",
            inline=False,
        )

    try:
        await ctx.author.send(embed=embed)
        if current_state:
            pose_outfits = list_pose_outfits(current_state.character_name)
            if pose_outfits:
                selected_pose, selected_outfit = get_selected_pose_outfit(current_state.character_name)
                selected_pose_normalized = normalize_pose_name(selected_pose)
                selected_outfit_normalized = (
                    selected_outfit.lower() if selected_outfit else None
                )
                pose_lines: list[str] = []
                for pose, options in pose_outfits.items():
                    entries: list[str] = []
                    for option in options:
                        display = option
                        if (
                            selected_outfit_normalized
                            and option.lower() == selected_outfit_normalized
                            and (
                                selected_pose_normalized is None
                                or pose.lower() == selected_pose_normalized
                            )
                        ):
                            display = f"{option} (current)"
                        entries.append(display)
                    pose_lines.append(f"{pose}: {', '.join(entries)}")
                outfit_note = (
                    f"Outfits available for {current_state.character_name}:\n"
                    + "\n".join(f"- {line}" for line in pose_lines)
                    + "\nUse `!outfit <outfit>` to pick by name or "
                    + "`!outfit <pose> <outfit>` (you can also separate with ':' or '/')."
                )
                try:
                    await ctx.author.send(outfit_note)
                except discord.Forbidden:
                    pass
    except discord.Forbidden:
        await ctx.reply(
            "I couldn't DM you. Please enable direct messages from server members.",
            mention_author=False,
            delete_after=10,
        )
    finally:
        try:
            await ctx.message.delete()
        except discord.HTTPException:
            pass


@bot.command(name="rerollall")
@commands.guild_only()
@guard_prefix_command_channel
async def prefix_rerollall_command(ctx: commands.Context) -> None:
    if not isinstance(ctx.author, discord.Member) or not is_admin(ctx.author) or not ctx.guild:
        await ctx.reply("Only admins can use this command.", mention_author=False)
        return
    await ensure_state_restored()
    
    # Get members who can view this channel (those "in" the channel)
    channel_member_ids = set()
    if isinstance(ctx.channel, (discord.TextChannel, discord.Thread, discord.ForumChannel)):
        for member in ctx.guild.members:
            if ctx.channel.permissions_for(member).view_channel:
                channel_member_ids.add(member.id)
    
    rerolled_count = 0
    # Set flag to suppress individual reroll messages
    ctx._suppress_reroll_messages = True
    try:
        for (guild_id, user_id), state in active_transformations.items():
            if guild_id == ctx.guild.id and user_id in channel_member_ids:
                member = ctx.guild.get_member(user_id)
                if member:
                    await reroll_command(ctx, args=member.mention)
                    rerolled_count += 1
    finally:
        # Clean up the flag
        if hasattr(ctx, "_suppress_reroll_messages"):
            del ctx._suppress_reroll_messages
    
    await ctx.reply("Rerolled Everyone", mention_author=False)


@bot.command(name="rerollnonadmin")
@commands.guild_only()
@guard_prefix_command_channel
async def prefix_rerollnonadmin_command(ctx: commands.Context) -> None:
    if not isinstance(ctx.author, discord.Member) or not is_admin(ctx.author) or not ctx.guild:
        await ctx.reply("Only admins can use this command.", mention_author=False)
        return
    await ensure_state_restored()
    
    # Get members who can view this channel (those "in" the channel)
    channel_member_ids = set()
    if isinstance(ctx.channel, (discord.TextChannel, discord.Thread, discord.ForumChannel)):
        for member in ctx.guild.members:
            if ctx.channel.permissions_for(member).view_channel:
                channel_member_ids.add(member.id)
    
    rerolled_count = 0
    # Set flag to suppress individual reroll messages
    ctx._suppress_reroll_messages = True
    try:
        for (guild_id, user_id), state in active_transformations.items():
            if guild_id == ctx.guild.id and user_id in channel_member_ids:
                member = ctx.guild.get_member(user_id)
                if member and not is_admin(member):
                    await reroll_command(ctx, args=member.mention)
                    rerolled_count += 1
    finally:
        # Clean up the flag
        if hasattr(ctx, "_suppress_reroll_messages"):
            del ctx._suppress_reroll_messages
    
    await ctx.reply(f"Rerolled {rerolled_count} non-admin member(s) in this channel.", mention_author=False)


# Game Board Commands
@bot.command(name="startgame")
@commands.guild_only()
@guard_prefix_command_channel
async def prefix_startgame_command(ctx: commands.Context, *, game_type: str = "") -> None:
    """Start a new game (GM only)."""
    if GAME_BOARD_MANAGER:
        await GAME_BOARD_MANAGER.command_startgame(ctx, game_type=game_type)


@bot.command(name="endgame")
@commands.guild_only()
@guard_prefix_command_channel
async def prefix_endgame_command(ctx: commands.Context) -> None:
    """End the current game (GM only)."""
    if GAME_BOARD_MANAGER:
        await GAME_BOARD_MANAGER.command_endgame(ctx)


@bot.command(name="listgames")
@commands.guild_only()
@guard_prefix_command_channel
async def prefix_listgames_command(ctx: commands.Context) -> None:
    """List available games."""
    if GAME_BOARD_MANAGER:
        await GAME_BOARD_MANAGER.command_listgames(ctx)


@bot.command(name="addplayer")
@commands.guild_only()
@guard_prefix_command_channel
async def prefix_addplayer_command(ctx: commands.Context, member: Optional[discord.Member] = None) -> None:
    """Add a player to the game (GM only)."""
    if GAME_BOARD_MANAGER:
        await GAME_BOARD_MANAGER.command_addplayer(ctx, member=member)


@bot.command(name="removeplayer")
@commands.guild_only()
@guard_prefix_command_channel
async def prefix_removeplayer_command(ctx: commands.Context, member: Optional[discord.Member] = None) -> None:
    """Remove a player from the game (GM only)."""
    if GAME_BOARD_MANAGER:
        await GAME_BOARD_MANAGER.command_removeplayer(ctx, member=member)


@bot.command(name="assign")
@commands.guild_only()
@guard_prefix_command_channel
async def prefix_assign_command(ctx: commands.Context, member: Optional[discord.Member] = None, *, character_name: str = "") -> None:
    """Assign a character to a player (GM only)."""
    if GAME_BOARD_MANAGER:
        await GAME_BOARD_MANAGER.command_assign(ctx, member=member, character_name=character_name)


@bot.command(name="swap")
@commands.guild_only()
@guard_prefix_command_channel
async def prefix_swap_command(ctx: commands.Context, *, args: str = "") -> None:
    """Swap characters between two players or characters.
    
    Usage: !swap character1 character2
    Or: !swap @user1 @user2
    """
    await ensure_state_restored()
    
    # Check if this is a game thread - if so, delegate to gameboard manager
    if GAME_BOARD_MANAGER and isinstance(ctx.channel, discord.Thread) and GAME_BOARD_MANAGER.is_game_thread(ctx.channel):
        # Parse arguments for gameboard mode (expects @user1 @user2)
        tokens = [token for token in args.split() if token.strip()]
        member1 = None
        member2 = None
        
        if tokens:
            user_id1 = _extract_user_id_from_token(tokens[0])
            if user_id1 and ctx.guild:
                member1 = ctx.guild.get_member(user_id1)
        if len(tokens) > 1:
            user_id2 = _extract_user_id_from_token(tokens[1])
            if user_id2 and ctx.guild:
                member2 = ctx.guild.get_member(user_id2)
        
        if member1 and member2:
            await GAME_BOARD_MANAGER.command_swap(ctx, member1=member1, member2=member2)
        else:
            await ctx.reply("Usage: `!swap @user1 @user2`", mention_author=False)
        return
    
    # Normal VN mode swap - swap transformations between two characters/users
    if not isinstance(ctx.author, discord.Member) or not ctx.guild:
        await ctx.reply("This command can only be used inside a server.", mention_author=False)
        return
    
    # Check permission: admin, ball, or narrator only
    author_is_admin = is_admin(ctx.author)
    author_state = find_active_transformation(ctx.author.id, ctx.guild.id)
    author_has_special_access = _has_special_reroll_access(author_state)
    
    if not author_is_admin and not author_has_special_access:
        await ctx.reply(_privileged_requirement_message("use this command"), mention_author=False)
        return
    
    tokens = [token for token in args.split() if token.strip()]
    if len(tokens) < 2:
        await ctx.reply("Usage: `!swap character1 character2` or `!swap @user1 @user2`", mention_author=False)
        return
    
    token1 = tokens[0]
    token2 = tokens[1]
    
    # Find state1 - can be by character name/folder or user mention
    # Use _find_state_by_token which is more comprehensive (searches by name, folder, mention, display name)
    state1 = None
    user1_id = None
    if ctx.guild:
        state1 = _find_state_by_token(ctx.guild, token1)
        if state1:
            user1_id = state1.user_id
    
    # Find state2 - can be by character name/folder or user mention
    # Use _find_state_by_token which is more comprehensive (searches by name, folder, mention, display name)
    state2 = None
    user2_id = None
    if ctx.guild:
        state2 = _find_state_by_token(ctx.guild, token2)
        if state2:
            user2_id = state2.user_id
    
    if not state1 or not state2:
        await ctx.reply("Could not find both active transformations. Both characters/users must be currently transformed.", mention_author=False)
        return
    
    if not user1_id or not user2_id:
        await ctx.reply("Could not determine users for swap.", mention_author=False)
        return
    
    if user1_id == user2_id:
        await ctx.reply("Cannot swap a character with itself.", mention_author=False)
        return
    
    # Get members for display
    member1 = ctx.guild.get_member(user1_id)
    member2 = ctx.guild.get_member(user2_id)
    
    if not member1 or not member2:
        await ctx.reply("Could not find both members.", mention_author=False)
        return
    
    # Swap character data between the two states
    # Keep user-specific fields (user_id, guild_id, started_at, expires_at, original_nick, original_display_name, avatar_applied)
    # Swap character-specific fields (character_name, character_folder, character_avatar_path, character_message, is_inanimate, inanimate_responses)
    
    char1_name = state1.character_name
    char1_folder = state1.character_folder
    char1_avatar = state1.character_avatar_path
    char1_message = state1.character_message
    char1_inanimate = state1.is_inanimate
    char1_responses = state1.inanimate_responses
    
    char2_name = state2.character_name
    char2_folder = state2.character_folder
    char2_avatar = state2.character_avatar_path
    char2_message = state2.character_message
    char2_inanimate = state2.is_inanimate
    char2_responses = state2.inanimate_responses
    
    owner1 = state1.form_owner_user_id or user1_id
    owner2 = state2.form_owner_user_id or user2_id
    identity1 = state1.identity_display_name or state1.character_name
    identity2 = state2.identity_display_name or state2.character_name

    # Create new states with swapped character data
    new_state1 = TransformationState(
        user_id=user1_id,
        guild_id=state1.guild_id,
        character_name=char2_name,
        character_folder=char2_folder,
        character_avatar_path=char2_avatar,
        character_message=char2_message,
        original_nick=state1.original_nick,
        started_at=state1.started_at,
        expires_at=state1.expires_at,
        duration_label=state1.duration_label,
        avatar_applied=state1.avatar_applied,
        original_display_name=state1.original_display_name,
        is_inanimate=char2_inanimate,
        inanimate_responses=char2_responses,
        form_owner_user_id=owner2,
        identity_display_name=identity1,
        is_pillow=state1.is_pillow,
    )

    new_state2 = TransformationState(
        user_id=user2_id,
        guild_id=state2.guild_id,
        character_name=char1_name,
        character_folder=char1_folder,
        character_avatar_path=char1_avatar,
        character_message=char1_message,
        original_nick=state2.original_nick,
        started_at=state2.started_at,
        expires_at=state2.expires_at,
        duration_label=state2.duration_label,
        avatar_applied=state2.avatar_applied,
        original_display_name=state2.original_display_name,
        is_inanimate=char1_inanimate,
        inanimate_responses=char1_responses,
        form_owner_user_id=owner1,
        identity_display_name=identity2,
        is_pillow=state2.is_pillow,
    )
    
    # Update active_transformations
    key1 = state_key(ctx.guild.id, user1_id)
    key2 = state_key(ctx.guild.id, user2_id)
    active_transformations[key1] = new_state1
    active_transformations[key2] = new_state2
    
    # Update revert tasks if they exist
    if key1 in revert_tasks:
        # Cancel old task
        revert_tasks[key1].cancel()
        # Create new task with same expiration
        delay1 = max((new_state1.expires_at - utc_now()).total_seconds(), 0)
        revert_tasks[key1] = asyncio.create_task(_schedule_revert(new_state1, delay1))
    
    if key2 in revert_tasks:
        # Cancel old task
        revert_tasks[key2].cancel()
        # Create new task with same expiration
        delay2 = max((new_state2.expires_at - utc_now()).total_seconds(), 0)
        revert_tasks[key2] = asyncio.create_task(_schedule_revert(new_state2, delay2))
    
    # Persist states
    persist_states()
    
    # Send confirmation message
    author_name = ctx.author.display_name
    character1_name = member1.display_name
    character2_name = member2.display_name
    
    await ctx.reply(
        f"Thanks to {author_name}, {character1_name} swapped bodies with {character2_name}.",
        mention_author=False,
    )


@bot.tree.command(name="swap", description="Swap characters between two players or characters (admins or special-form characters only).")
@app_commands.describe(
    character1="First character/user to swap (character name or @mention).",
    character2="Second character/user to swap (character name or @mention).",
)
@app_commands.guild_only()
@guard_slash_command_channel
async def slash_swap_command(
    interaction: discord.Interaction,
    character1: str,
    character2: str,
) -> None:
    """Swap characters between two players or characters."""
    if not await _safe_defer_interaction(interaction, thinking=True):
        return
    
    # Check if this is a game thread - if so, delegate to gameboard manager
    if GAME_BOARD_MANAGER and isinstance(interaction.channel, discord.Thread) and GAME_BOARD_MANAGER.is_game_thread(interaction.channel):
        # For gameboard mode, we expect user mentions
        if not interaction.guild:
            await interaction.followup.send("This command can only be used inside a server.", ephemeral=True)
            return
        
        user_id1 = _extract_user_id_from_token(character1)
        user_id2 = _extract_user_id_from_token(character2)
        
        if not user_id1 or not user_id2:
            await interaction.followup.send("Usage: `/swap @user1 @user2`", ephemeral=True)
            return
        
        member1 = interaction.guild.get_member(user_id1)
        member2 = interaction.guild.get_member(user_id2)
        
        if member1 and member2:
            ctx = InteractionContextAdapter(interaction, bot=bot)
            await GAME_BOARD_MANAGER.command_swap(ctx, member1=member1, member2=member2)
        else:
            await interaction.followup.send("Could not find both members.", ephemeral=True)
        return
    
    # Normal VN mode - use the prefix command logic
    if not isinstance(interaction.user, discord.Member) or not interaction.guild:
        await interaction.followup.send("This command can only be used inside a server.", ephemeral=True)
        return
    
    # Check permission: admin, ball, or narrator only
    author_is_admin = is_admin(interaction.user)
    author_state = find_active_transformation(interaction.user.id, interaction.guild.id)
    author_has_special_access = _has_special_reroll_access(author_state)
    
    if not author_is_admin and not author_has_special_access:
        await interaction.followup.send(_privileged_requirement_message("use this command"), ephemeral=True)
        return
    
    # Use the same logic as prefix command but adapted for slash
    await ensure_state_restored()
    
    # Find state1 and state2 using the same comprehensive search
    state1 = None
    user1_id = None
    if interaction.guild:
        state1 = _find_state_by_token(interaction.guild, character1)
        if state1:
            user1_id = state1.user_id
    
    state2 = None
    user2_id = None
    if interaction.guild:
        state2 = _find_state_by_token(interaction.guild, character2)
        if state2:
            user2_id = state2.user_id
    
    if not state1 or not state2:
        await interaction.followup.send("Could not find both active transformations. Both characters/users must be currently transformed.", ephemeral=True)
        return
    
    if not user1_id or not user2_id:
        await interaction.followup.send("Could not determine users for swap.", ephemeral=True)
        return
    
    if user1_id == user2_id:
        await interaction.followup.send("Cannot swap a character with itself.", ephemeral=True)
        return
    
    # Get members for display
    member1 = interaction.guild.get_member(user1_id)
    member2 = interaction.guild.get_member(user2_id)
    
    if not member1 or not member2:
        await interaction.followup.send("Could not find both members.", ephemeral=True)
        return
    
    # Swap character data between the two states
    char1_name = state1.character_name
    char1_folder = state1.character_folder
    char1_avatar = state1.character_avatar_path
    char1_message = state1.character_message
    char1_inanimate = state1.is_inanimate
    char1_responses = state1.inanimate_responses
    
    char2_name = state2.character_name
    char2_folder = state2.character_folder
    char2_avatar = state2.character_avatar_path
    char2_message = state2.character_message
    char2_inanimate = state2.is_inanimate
    char2_responses = state2.inanimate_responses
    
    owner1 = state1.form_owner_user_id or user1_id
    owner2 = state2.form_owner_user_id or user2_id
    identity1 = state1.identity_display_name or state1.character_name
    identity2 = state2.identity_display_name or state2.character_name
    
    # Create new states with swapped character data
    new_state1 = TransformationState(
        user_id=user1_id,
        guild_id=state1.guild_id,
        character_name=char2_name,
        character_folder=char2_folder,
        character_avatar_path=char2_avatar,
        character_message=char2_message,
        original_nick=state1.original_nick,
        started_at=state1.started_at,
        expires_at=state1.expires_at,
        duration_label=state1.duration_label,
        avatar_applied=state1.avatar_applied,
        original_display_name=state1.original_display_name,
        is_inanimate=char2_inanimate,
        inanimate_responses=char2_responses,
        form_owner_user_id=owner2,
        identity_display_name=identity1,
        is_pillow=state1.is_pillow,
    )

    new_state2 = TransformationState(
        user_id=user2_id,
        guild_id=state2.guild_id,
        character_name=char1_name,
        character_folder=char1_folder,
        character_avatar_path=char1_avatar,
        character_message=char1_message,
        original_nick=state2.original_nick,
        started_at=state2.started_at,
        expires_at=state2.expires_at,
        duration_label=state2.duration_label,
        avatar_applied=state2.avatar_applied,
        original_display_name=state2.original_display_name,
        is_inanimate=char1_inanimate,
        inanimate_responses=char1_responses,
        form_owner_user_id=owner1,
        identity_display_name=identity2,
        is_pillow=state2.is_pillow,
    )
    
    # Update active_transformations
    key1 = state_key(interaction.guild.id, user1_id)
    key2 = state_key(interaction.guild.id, user2_id)
    active_transformations[key1] = new_state1
    active_transformations[key2] = new_state2
    
    # Update revert tasks if they exist
    if key1 in revert_tasks:
        revert_tasks[key1].cancel()
        delay1 = max((new_state1.expires_at - utc_now()).total_seconds(), 0)
        revert_tasks[key1] = asyncio.create_task(_schedule_revert(new_state1, delay1))
    
    if key2 in revert_tasks:
        revert_tasks[key2].cancel()
        delay2 = max((new_state2.expires_at - utc_now()).total_seconds(), 0)
        revert_tasks[key2] = asyncio.create_task(_schedule_revert(new_state2, delay2))
    
    # Persist states
    persist_states()
    
    # Send confirmation message
    author_name = interaction.user.display_name
    character1_name = member1.display_name
    character2_name = member2.display_name
    
    await interaction.followup.send(
        f"Thanks to {author_name}, {character1_name} swapped bodies with {character2_name}.",
        ephemeral=False,
    )


@bot.command(name="movetoken")
@commands.guild_only()
@guard_prefix_command_channel
async def prefix_movetoken_command(ctx: commands.Context, member: Optional[discord.Member] = None, *, position: str = "") -> None:
    """Move a player's token (GM only)."""
    if GAME_BOARD_MANAGER:
        await GAME_BOARD_MANAGER.command_movetoken(ctx, member=member, position=position)


@bot.command(name="dice")
@commands.guild_only()
@guard_prefix_command_channel
async def prefix_dice_command(ctx: commands.Context) -> None:
    """Roll dice (player command)."""
    if GAME_BOARD_MANAGER:
        await GAME_BOARD_MANAGER.command_dice(ctx)


@bot.command(name="roll")
@guard_prefix_command_channel
async def prefix_roll_command(ctx: commands.Context, *, args: str = "") -> None:
    """Route roll requests to the right subsystem."""
    if GACHA_MANAGER is not None:
        in_gacha_channel = ctx.guild is None or (
            isinstance(ctx.channel, discord.TextChannel) and ctx.channel.id == GACHA_MANAGER.channel_id
        )
        if in_gacha_channel:
            roll_type = ""
            extra = ""
            if args:
                parts = args.split()
                roll_type = parts[0].lower()
                extra = " ".join(parts[1:]).strip()
            await GACHA_MANAGER.command_roll(ctx, roll_type, extra)
            return

    if (
        GAME_BOARD_MANAGER
        and isinstance(ctx.channel, discord.Thread)
        and GAME_BOARD_MANAGER.is_game_thread(ctx.channel)
    ):
        await GAME_BOARD_MANAGER.command_dice(ctx)
        return

    await ctx.reply(
        "Rolls are only supported in the gacha channel or inside active game threads.",
        mention_author=False,
    )


@bot.command(name="rules")
@commands.guild_only()
@guard_prefix_command_channel
async def prefix_rules_command(ctx: commands.Context) -> None:
    """Show game rules (player command)."""
    if GAME_BOARD_MANAGER:
        await GAME_BOARD_MANAGER.command_rules(ctx)




@bot.command(name="savegame")
@commands.guild_only()
@guard_prefix_command_channel
async def prefix_savegame_command(ctx: commands.Context) -> None:
    """Save the current game state (GM only)."""
    if GAME_BOARD_MANAGER:
        await GAME_BOARD_MANAGER.command_savegame(ctx)


@bot.command(name="loadgame")
@commands.guild_only()
@guard_prefix_command_channel
async def prefix_loadgame_command(ctx: commands.Context, *, state_file: str = "") -> None:
    """Load a saved game state (GM only)."""
    if GAME_BOARD_MANAGER:
        await GAME_BOARD_MANAGER.command_loadgame(ctx, state_file=state_file)


@bot.command(name="bg")
async def prefix_bg_35(ctx: commands.Context, *, selection: str = ""):
    """3.5 version of bg command."""
    # Check if this is a game thread first - completely isolate from global VN system
    if GAME_BOARD_MANAGER and isinstance(ctx.channel, discord.Thread) and GAME_BOARD_MANAGER.is_game_thread(ctx.channel):
        # Game thread - use game-specific bg command (isolated)
        if not selection.strip():
            # Show game-specific background list
            await GAME_BOARD_MANAGER.command_bg_list(ctx)
            return
        # Parse game bg command: !bg @user <id> or !bg all <id>
        parts = selection.strip().split(maxsplit=1)
        if len(parts) == 2:
            target_str, bg_id = parts
            # Try to parse as member mention or "all"
            member = None
            if target_str.lower() != "all":
                # Try to extract member from mention
                from discord.ext.commands import MemberConverter
                try:
                    member = await MemberConverter().convert(ctx, target_str)
                except:
                    pass
            # If member is None, it means "all" was passed
            await GAME_BOARD_MANAGER.command_bg(ctx, target=member, bg_id=bg_id)
            return
        else:
            # Invalid format
            await ctx.reply("Usage: `!bg @user <id>` or `!bg all <id>`. Use `!bg` to list available backgrounds.", mention_author=False)
            return
    
    try:
        await ctx.message.delete()
    except discord.HTTPException:
        pass

    await ensure_state_restored()

    if VN_BACKGROUND_ROOT is None:
        try:
            await ctx.author.send("Backgrounds are not configured on this bot.")
        except discord.Forbidden:
            if ctx.guild:
                await ctx.send("I couldn't DM you. Please enable direct messages.", delete_after=10)
        return

    choices = list_background_choices()
    if not choices:
        try:
            await ctx.author.send("No background images were found in the configured directory.")
        except discord.Forbidden:
            if ctx.guild:
                await ctx.send("I couldn't DM you. Please enable direct messages.", delete_after=10)
        return

    selection = selection.strip()
    if not selection:
        lines: list[str] = []
        for idx, path in enumerate(choices, start=1):
            try:
                relative = path.resolve().relative_to(VN_BACKGROUND_ROOT.resolve())
                display = relative.as_posix()
            except ValueError:
                display = str(path)
            lines.append(f"{idx}: {display}")

        chunks: list[str] = []
        current: list[str] = []
        length = 0
        for line in lines:
            if length + len(line) + 1 > 1900 and current:
                chunks.append("\n".join(current))
                current = []
                length = 0
            current.append(line)
            length += len(line) + 1
        if current:
            chunks.append("\n".join(current))

        default_display = (
            VN_BACKGROUND_DEFAULT_RELATIVE.as_posix()
            if VN_BACKGROUND_DEFAULT_RELATIVE
            else "system default"
        )
        instructions = (
            "Use `!bg <number>` to apply that background to your VN panel.\n"
            "Example: `!bg 45` selects option 45 from the list.\n"
            f"The default background is `{default_display}`."
        )

        try:
            for chunk in chunks:
                await ctx.author.send(f"```\n{chunk}\n```")
            await ctx.author.send(instructions)
        except discord.Forbidden:
            if ctx.guild:
                await ctx.send("I couldn't DM you. Please enable direct messages, then rerun `!bg`.", delete_after=10)
            return

        return

    actor_member = ctx.author if isinstance(ctx.author, discord.Member) else None
    can_target_others = (
        ctx.guild is not None
        and actor_member is not None
        and (is_admin(actor_member) or _actor_has_narrator_power_35(actor_member))
    )
    selection = selection.strip()
    target_spec: Optional[str] = None
    if " " in selection:
        number_part, target_part = selection.split(None, 1)
        selection = number_part
        target_spec = target_part.strip() or None

    try:
        index = int(selection)
    except ValueError:
        try:
            await ctx.author.send(f"`{selection}` isn't a valid background number. Use `!bg` with no arguments to see the list.")
        except discord.Forbidden:
            if ctx.guild:
                await ctx.send("I couldn't DM you. Please enable direct messages.", delete_after=10)
        return

    if index < 1 or index > len(choices):
        try:
            await ctx.author.send(f"Background number must be between 1 and {len(choices)}.")
        except discord.Forbidden:
            if ctx.guild:
                await ctx.send("I couldn't DM you. Please enable direct messages.", delete_after=10)
        return

    selected_path = choices[index - 1]
    try:
        relative = selected_path.resolve().relative_to(VN_BACKGROUND_ROOT.resolve())
        display = relative.as_posix()
    except ValueError:
        display = str(selected_path)

    if target_spec:
        if ctx.guild is None or actor_member is None:
            await ctx.reply("Targeted background changes can only be used inside a server channel.", mention_author=False)
            return
        if not can_target_others:
            await ctx.reply(_privileged_requirement_message("set backgrounds for other characters"), mention_author=False)
            return
        target_lower = target_spec.lower()
        if target_lower == "all":
            targets = [
                state
                for state in active_transformations.values()
                if state.guild_id == ctx.guild.id and not state.is_inanimate
            ]
            if not targets:
                await ctx.reply("No active characters are available to update right now.", mention_author=False)
                return
            failures = 0
            for state in targets:
                if not set_selected_background(state.user_id, selected_path):
                    failures += 1
            schedule_history_refresh()
            updated = len(targets) - failures
            await ctx.reply(
                f"Background set to `{display}` for {updated} character{'s' if updated != 1 else ''}.",
                mention_author=False,
            )
            return

        target_state = _find_state_by_token(ctx.guild, target_spec)
        if target_state is None:
            await ctx.reply(f"Couldn't find a transformed character matching `{target_spec}`.", mention_author=False)
            return
        if target_state.is_inanimate:
            await ctx.reply(f"{target_state.character_name} is inanimate and can't use VN backgrounds.", mention_author=False)
            return
        if not set_selected_background(target_state.user_id, selected_path):
            await ctx.reply("Unable to update that background right now.", mention_author=False)
            return
        schedule_history_refresh()
        await ctx.reply(
            f"Background for {target_state.character_name} set to `{display}`.",
            mention_author=False,
        )
        return

    if not set_selected_background(ctx.author.id, selected_path):
        try:
            await ctx.author.send("Unable to update your background at this time.")
        except discord.Forbidden:
            if ctx.guild:
                await ctx.send("I couldn't DM you. Please enable direct messages.", delete_after=10)
        return

    try:
        await ctx.author.send(f"Background set to `{display}`.")
    except discord.Forbidden:
        if ctx.guild:
            await ctx.send("I couldn't DM you. Please enable direct messages.", delete_after=10)


@bot.command(name="outfit")
@guard_prefix_command_channel
async def prefix_outfit_35(ctx: commands.Context, *, outfit_name: str = ""):
    """3.5 version of outfit command."""
    # Check if this is a game thread first - completely isolate from global VN system
    if GAME_BOARD_MANAGER and isinstance(ctx.channel, discord.Thread) and GAME_BOARD_MANAGER.is_game_thread(ctx.channel):
        # Game thread - use game-specific outfit command (isolated)
        if not outfit_name.strip():
            # Show list
            await GAME_BOARD_MANAGER.command_outfit_list(ctx)
            return
        # Parse game outfit command: !outfit @user <outfit>
        parts = outfit_name.strip().split(maxsplit=1)
        if len(parts) == 2:
            target_str, outfit = parts
            # Try to extract member from mention
            from discord.ext.commands import MemberConverter
            try:
                member = await MemberConverter().convert(ctx, target_str)
                await GAME_BOARD_MANAGER.command_outfit(ctx, member=member, outfit_name=outfit)
                return
            except:
                pass
        # If parsing failed, try without member (show list)
        await GAME_BOARD_MANAGER.command_outfit(ctx, member=None, outfit_name=outfit_name)
        return
    
    outfit_name = outfit_name.strip()
    if not outfit_name:
        message = "Usage: `!outfit <outfit>` or `!outfit <pose> <outfit>`"
        if ctx.guild:
            await ctx.reply(message, mention_author=False)
        else:
            await ctx.send(message)
        return

    await ensure_state_restored()

    guild_id = ctx.guild.id if ctx.guild else None
    actor_member = ctx.author if isinstance(ctx.author, discord.Member) else None
    can_target_others = (
        ctx.guild is not None
        and actor_member is not None
        and (is_admin(actor_member) or _actor_has_narrator_power_35(actor_member))
    )
    target_state: Optional[TransformationState] = None
    if can_target_others and ctx.guild and " " in outfit_name:
        base_value, candidate = outfit_name.rsplit(" ", 1)
        candidate = candidate.strip()
        if candidate:
            matched_state = _find_state_by_token(ctx.guild, candidate)
            if matched_state:
                target_state = matched_state
                outfit_name = base_value.strip()
    if target_state and target_state.is_inanimate:
        if ctx.guild:
            await ctx.reply(f"{target_state.character_name} is inanimate and can't change outfits.", mention_author=False)
        else:
            await ctx.send(f"{target_state.character_name} is inanimate and can't change outfits.")
        return
    state = target_state or find_active_transformation(ctx.author.id, guild_id)
    if not state:
        fallback_state = find_active_transformation(ctx.author.id)
        if fallback_state and ctx.guild and fallback_state.guild_id != guild_id:
            target_guild = bot.get_guild(fallback_state.guild_id)
            guild_name = target_guild.name if target_guild else f"server {fallback_state.guild_id}"
            message = (
                "You're transformed right now, but in a different server. "
                f"Use this command in **{guild_name}** to change that outfit."
            )
        else:
            message = "You need to be transformed to change outfits."
        if ctx.guild:
            await ctx.reply(message, mention_author=False)
        else:
            await ctx.send(message)
        return

    pose_outfits = list_pose_outfits(state.character_name)
    if not pose_outfits:
        message = f"No outfits are available for {state.character_name}."
        if ctx.guild:
            await ctx.reply(message, mention_author=False)
        else:
            await ctx.send(message)
        return

    parsed_pose: Optional[str] = None
    parsed_outfit: Optional[str] = None

    for separator in (":", "/"):
        if separator in outfit_name:
            left, right = outfit_name.split(separator, 1)
            parsed_pose = left.strip()
            parsed_outfit = right.strip()
            break

    if parsed_outfit is None:
        parts = outfit_name.split()
        if len(parts) >= 2:
            parsed_pose = parts[0].strip()
            parsed_outfit = " ".join(parts[1:]).strip()
        else:
            parsed_outfit = outfit_name

    if not parsed_outfit:
        message = "Please provide the outfit to select. Example: `!outfit cheer` or `!outfit b cheer`."
        if ctx.guild:
            await ctx.reply(message, mention_author=False)
        else:
            await ctx.send(message)
        return

    if parsed_pose:
        normalized_pose = normalize_pose_name(parsed_pose)
        known_poses = {pose.lower() for pose in pose_outfits.keys()}
        if normalized_pose not in known_poses:
            message = (
                f"Unknown pose `{parsed_pose}`. Available poses: {', '.join(pose_outfits.keys())}."
            )
            if ctx.guild:
                await ctx.reply(message, mention_author=False)
            else:
                await ctx.send(message)
            return
    else:
        normalized_pose = None

    if not set_selected_pose_outfit(state.character_name, parsed_pose if normalized_pose else None, parsed_outfit):
        pose_lines = []
        for pose, options in pose_outfits.items():
            pose_lines.append(f"{pose}: {', '.join(options)}")
        message = (
            f"Unable to update outfit. Available options:\n"
            + "\n".join(f"- {line}" for line in pose_lines)
        )
        if ctx.guild:
            await ctx.reply(message, mention_author=False)
        else:
            await ctx.send(message)
        return

    selected_pose, selected_outfit = get_selected_pose_outfit(state.character_name)
    pose_label = selected_pose or "auto"
    outfit_label = selected_outfit or parsed_outfit
    confirmation = (
        f"Outfit for {state.character_name} set to `{outfit_label}` (pose `{pose_label}`). "
        "Future messages will use this combination."
    )
    schedule_history_refresh()
    if ctx.guild:
        await ctx.reply(confirmation, mention_author=False)
    else:
        await ctx.send(confirmation)


@bot.command(name="say")
@commands.guild_only()
@guard_prefix_command_channel
async def prefix_say_35(ctx: commands.Context, *, args: str = ""):
    """3.5 version of say command."""
    await ensure_state_restored()

    actor = ctx.author
    if not isinstance(actor, discord.Member):
        await ctx.reply("This command can only be used inside a server.", mention_author=False)
        return
    can_use_command = is_admin(actor) or _actor_has_narrator_power_35(actor)
    if not can_use_command:
        await ctx.reply(_privileged_requirement_message("use `!say`"), mention_author=False)
        return

    args = args.strip()
    if not args or " " not in args:
        await ctx.reply("Usage: `!say <character> <text>`", mention_author=False)
        return

    target_token, text = args.split(None, 1)
    target_state = _find_state_by_token(ctx.guild, target_token)
    if target_state is None:
        character = _find_character_by_token(target_token)
        if character is None:
            await ctx.reply(
                f"Couldn't find a character or active TF matching `{target_token}`.",
                mention_author=False,
            )
            return
        target_state = _build_roleplay_state(character, actor, ctx.guild)
    if target_state.is_inanimate:
        await ctx.reply(f"{target_state.character_name} can't speak right now.", mention_author=False)
        return

    cleaned_content = text.strip()
    if not cleaned_content:
        await ctx.reply("Please provide what the character should say.", mention_author=False)
        return

    _, member = await fetch_member(target_state.guild_id, target_state.user_id)
    original_name = (
        member.display_name
        if isinstance(member, discord.Member)
        else target_state.original_display_name
        or f"User {target_state.user_id}"
    )

    reply_context = await _resolve_reply_context(ctx.message)

    if AI_REWRITE_ENABLED and not cleaned_content.startswith(str(bot.command_prefix)):
        context_snippet = CHARACTER_CONTEXT.get(target_state.character_name) or target_state.character_message
        rewritten = await rewrite_message_for_character(
            original_text=cleaned_content,
            character_name=target_state.character_name,
            character_context=context_snippet,
            user_name=original_name,
        )
        if rewritten and rewritten.strip():
            cleaned_content = rewritten.strip()

    cleaned_content, _ = strip_urls(cleaned_content)
    cleaned_content = cleaned_content.strip()
    formatted_segments = parse_discord_formatting(cleaned_content) if cleaned_content else []
    custom_emoji_images = await prepare_custom_emoji_images(ctx.message, formatted_segments)

    files: list[discord.File] = []
    payload: dict = {}
    if MESSAGE_STYLE == "vn":
        character_display_name = _resolve_character_display_name(ctx.guild, target_state.user_id, target_state)
        vn_file = render_vn_panel(
            state=target_state,
            message_content=cleaned_content,
            character_display_name=character_display_name,
            original_name=original_name,
            attachment_id=str(ctx.message.id),
            formatted_segments=formatted_segments,
            custom_emoji_images=custom_emoji_images,
            reply_context=reply_context,
        )
        if vn_file:
            files.append(vn_file)

    description = cleaned_content if cleaned_content else "*no message content*"
    if not files:
        embed, avatar_file = await build_legacy_embed(target_state, description)
        if avatar_file:
            files.append(avatar_file)
        payload["embed"] = embed

    send_kwargs: Dict[str, object] = {}
    send_kwargs.update(payload)
    if files:
        send_kwargs["files"] = files
    message_reference = ctx.message.reference
    if message_reference:
        if isinstance(message_reference, discord.Message):
            message_reference = message_reference.to_reference(fail_if_not_exists=False)
        send_kwargs["reference"] = message_reference
    send_kwargs["allowed_mentions"] = discord.AllowedMentions.none()

    try:
        sent_message = await ctx.send(**send_kwargs)
    except discord.HTTPException as exc:
        logger.warning("Failed to send say panel: %s", exc)
        await ctx.reply("Couldn't deliver that panel.", mention_author=False)
        return

    if sent_message and cleaned_content:
        _register_relay_message(sent_message.id, target_state.character_name, cleaned_content)

    try:
        await ctx.message.delete()
    except discord.HTTPException:
        pass


def main():
    bot.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()
