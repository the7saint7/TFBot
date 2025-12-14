import os
import json
import importlib.util
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("tfbot")

# Bot name from environment, defaults to "Syn" for backwards compatibility
BOT_NAME = os.getenv("TFBOT_NAME", "Syn").strip() or "Syn"

# Helper function to replace bot name placeholder in messages
def _format_message(message: str) -> str:
    """Replace {BOT_NAME} placeholder with actual bot name."""
    return message.replace("{BOT_NAME}", BOT_NAME)

_MODULE_DIR = Path(__file__).resolve().parent


def _resolve_characters_repo_root() -> Optional[Path]:
    """Locate the shared characters_repo directory (required)."""
    repo_dir_setting = os.getenv("TFBOT_CHARACTERS_REPO_DIR", "characters_repo").strip() or "characters_repo"
    repo_dir = Path(repo_dir_setting)
    if not repo_dir.is_absolute():
        repo_dir = (_MODULE_DIR / repo_dir).resolve()
    return repo_dir if repo_dir.exists() else None


_CHARACTERS_REPO_ROOT = _resolve_characters_repo_root()
if _CHARACTERS_REPO_ROOT is None:
    message = (
        "characters_repo directory not found. Ensure the shared repository exists "
        "and TFBOT_CHARACTERS_REPO_DIR points to it."
    )
    logger.critical(message)
    raise SystemExit(message)

_config_path = _CHARACTERS_REPO_ROOT / "tf_characters.json"
_packs_dir = _CHARACTERS_REPO_ROOT / "packs"

if _config_path.exists():
    logger.info("Character pack configuration: %s", _config_path)
else:
    logger.warning("Character pack configuration missing at %s", _config_path)

if _packs_dir.exists():
    logger.info("Character packs directory: %s", _packs_dir)
else:
    logger.warning("Character packs directory missing at %s", _packs_dir)

# Shared cache for loaded characters so repeated imports don't reprocess packs
_PACK_CACHE_KEY = f"tf_characters::{BOT_NAME}"
TF_CHARACTERS: list[dict] = []

if _PACK_CACHE_KEY in globals():
    cached_chars = globals().get(_PACK_CACHE_KEY)
    if isinstance(cached_chars, list) and cached_chars:
        TF_CHARACTERS = cached_chars.copy()
else:
    globals()[_PACK_CACHE_KEY] = TF_CHARACTERS

configured_files = set()

if _config_path.exists():
    try:
        with open(_config_path, 'r', encoding='utf-8') as f:
            pack_configs = json.load(f)
        
        logger.info("Loading character packs for bot: %s", BOT_NAME)
        
        # Load enabled packs based on bot name
        for pack_config in pack_configs:
            pack_name = pack_config.get("name", "Unknown")
            pack_file = pack_config.get("file")
            enable_BunniBot = pack_config.get("enable_BunniBot", False)
            enable_VNBot = pack_config.get("enable_VNBot", False)
            
            if pack_file:
                configured_files.add(pack_file)
            
            # Check if pack is enabled for current bot
            should_load = False
            if BOT_NAME == "BunniBot" and enable_BunniBot:
                should_load = True
            elif BOT_NAME == "VNBot" and enable_VNBot:
                should_load = True
            
            # Log pack status
            status = "ENABLED" if should_load else "DISABLED"
            logger.info("  Pack '%s' (%s): %s", pack_name, pack_file, status)
            
            if should_load and pack_file:
                if not _packs_dir.exists():
                    logger.warning("    Packs directory missing; cannot load %s.", pack_file)
                    continue

                # Try to load from packs directory
                pack_path = _packs_dir / f"{pack_file}.py"
                if pack_path.exists():
                    try:
                        spec = importlib.util.spec_from_file_location(pack_file, pack_path)
                        if spec and spec.loader:
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                            pack_chars = getattr(module, "TF_CHARACTERS", None)
                            if isinstance(pack_chars, list):
                                TF_CHARACTERS.extend(pack_chars)
                                logger.info("    Loaded %d characters from %s", len(pack_chars), pack_file)
                    except Exception as exc:
                        logger.warning("Failed to load pack %s: %s", pack_file, exc)
                else:
                    # Try JSON file for Inanimate
                    pack_json_path = _packs_dir / f"{pack_file}.json"
                    if pack_json_path.exists():
                        try:
                            with open(pack_json_path, 'r', encoding='utf-8') as f:
                                pack_chars = json.load(f)
                                if isinstance(pack_chars, list):
                                    TF_CHARACTERS.extend(pack_chars)
                                    logger.info("    Loaded %d characters from %s", len(pack_chars), pack_file)
                        except Exception as exc:
                            logger.warning("Failed to load pack %s: %s", pack_file, exc)
                    else:
                        logger.warning("Pack file %s not found in %s", pack_file, _packs_dir)
    except Exception as exc:
        logger.warning("Failed to load pack config from tf_characters.json: %s", exc)

# Auto-detect new pack files not in config
if _packs_dir.exists():
    detected_new_packs = []
    for filename in os.listdir(_packs_dir):
        if filename.startswith("characters_") and (filename.endswith(".py") or filename.endswith(".json")):
            pack_name = filename.replace(".py", "").replace(".json", "")
            if pack_name not in configured_files:
                detected_new_packs.append(pack_name)
    
    if detected_new_packs:
        logger.info("Detected %d new pack file(s) not in config (disabled by default):", len(detected_new_packs))
        for pack_name in detected_new_packs:
            logger.info("  - %s (add to tf_characters.json to enable)", pack_name)
    else:
        logger.info("No new pack files detected")

logger.info("Total characters loaded: %d", len(TF_CHARACTERS))
