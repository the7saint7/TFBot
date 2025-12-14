import os
import json
import importlib.util
import logging
from pathlib import Path

logger = logging.getLogger("tfbot")

# Bot name from environment, defaults to "Syn" for backwards compatibility
BOT_NAME = os.getenv("TFBOT_NAME", "Syn").strip() or "Syn"

# Helper function to replace bot name placeholder in messages
def _format_message(message: str) -> str:
    """Replace {BOT_NAME} placeholder with actual bot name."""
    return message.replace("{BOT_NAME}", BOT_NAME)

TF_CHARACTERS = []

# Load pack configuration from JSON
_config_path = Path(__file__).parent / "tf_characters.json"
_packs_dir = Path(__file__).parent / "packs"
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
