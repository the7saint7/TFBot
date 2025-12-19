"""
Game Pack Loader

Loads game-specific packs dynamically. Each game pack is self-contained
and implements game-specific rules and logic.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from types import ModuleType
from typing import Dict, Optional

logger = logging.getLogger("tfbot.game_packs")


class GamePack:
    """Wrapper for a loaded game pack module."""
    
    def __init__(self, module: ModuleType, game_type: str):
        self.module = module
        self.game_type = game_type
        self.name = getattr(module, '__name__', game_type)
    
    def has_function(self, func_name: str) -> bool:
        """Check if pack has a specific function."""
        return hasattr(self.module, func_name) and callable(getattr(self.module, func_name))
    
    def call(self, func_name: str, *args, **kwargs):
        """Call a function from the pack, returning None if it doesn't exist."""
        if not self.has_function(func_name):
            return None
        return getattr(self.module, func_name)(*args, **kwargs)


_loaded_packs: Dict[str, GamePack] = {}


def load_game_pack(game_type: str, packs_dir: Optional[Path] = None) -> Optional[GamePack]:
    """
    Load a game pack module.
    
    Args:
        game_type: The game type identifier (e.g., "snakes_ladders")
        packs_dir: Directory containing pack files (default: games/packs/)
    
    Returns:
        GamePack instance or None if not found
    """
    if game_type in _loaded_packs:
        return _loaded_packs[game_type]
    
    if packs_dir is None:
        # Default to games/packs/ relative to this file
        packs_dir = Path(__file__).parent.parent.parent / "games" / "packs"
    
    pack_file = packs_dir / f"{game_type}.py"
    
    if not pack_file.exists():
        logger.debug("Game pack not found: %s", pack_file)
        return None
    
    try:
        spec = importlib.util.spec_from_file_location(f"game_pack_{game_type}", pack_file)
        if spec is None or spec.loader is None:
            logger.warning("Failed to create spec for game pack: %s", pack_file)
            return None
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        pack = GamePack(module, game_type)
        _loaded_packs[game_type] = pack
        
        logger.info("Loaded game pack: %s from %s", game_type, pack_file)
        return pack
    
    except Exception as exc:
        logger.error("Failed to load game pack %s: %s", pack_file, exc, exc_info=True)
        return None


def get_game_pack(game_type: str, packs_dir: Optional[Path] = None) -> Optional[GamePack]:
    """Get a game pack, loading it if necessary."""
    return load_game_pack(game_type, packs_dir)


def clear_pack_cache() -> None:
    """Clear the pack cache (useful for reloading)."""
    _loaded_packs.clear()

