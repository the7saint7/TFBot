#!/usr/bin/env python
"""Utility to sync gacha_config.json with available character assets."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore[assignment]

logger = logging.getLogger("update_gacha_config")

COMMON_RARITY = "common"
DEFAULT_STRUCTURE = {
    "rarities": {"common": 70, "rare": 25, "ultra": 5},
    "starter_rarities": ["common"],
    "starter_characters": [],
    "starter_outfits": {},
    "frog_boost_bonus": 0.15,
    "frog_boost_rolls": 2,
    "frog_boost_targets": ["rare", "ultra"],
    "characters": {},
}


def parse_args() -> argparse.Namespace:
    if load_dotenv:
        load_dotenv()
    parser = argparse.ArgumentParser(
        description="Scan VN character assets and seed/update gacha_config.json with new entries.",
    )
    parser.add_argument(
        "--characters-root",
        type=Path,
        help="Path to the character asset root (defaults to TFBOT_VN_ASSET_ROOT or game/images/characters).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("gacha_config.json"),
        help="Path to the gacha config file to create/update (default: gacha_config.json).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report differences without writing to the config file.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Increase logging verbosity.",
    )
    return parser.parse_args()


def resolve_characters_root(explicit: Optional[Path]) -> Path:
    if explicit:
        root = explicit.expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Characters root {root} does not exist.")
        return root

    env_game_root = os.getenv("TFBOT_VN_GAME_ROOT")
    if env_game_root:
        candidate = Path(env_game_root).expanduser().resolve() / "game" / "images" / "characters"
        if candidate.exists():
            return candidate

    env_asset_root = os.getenv("TFBOT_VN_ASSET_ROOT")
    if env_asset_root:
        root = Path(env_asset_root).expanduser().resolve()
        if root.exists():
            return root

    fallback = Path("game") / "images" / "characters"
    if fallback.exists():
        return fallback.resolve()

    raise FileNotFoundError(
        "Unable to resolve characters root. "
        "Provide --characters-root or set TFBOT_VN_GAME_ROOT / TFBOT_VN_ASSET_ROOT."
    )


def load_config(config_path: Path) -> MutableMapping[str, object]:
    if not config_path.exists():
        return copy.deepcopy(DEFAULT_STRUCTURE)
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse {config_path}: {exc}") from exc
    if not isinstance(data, MutableMapping):
        raise ValueError(f"Gacha config {config_path} must be a JSON object.")
    for key, value in DEFAULT_STRUCTURE.items():
        data.setdefault(key, value if not isinstance(value, dict) else dict(value))
    characters = data.setdefault("characters", {})
    if not isinstance(characters, MutableMapping):
        raise ValueError("`characters` entry in config must be an object.")
    return data


def read_display_name(character_dir: Path) -> str:
    config_path = character_dir / "character.yml"
    if yaml and config_path.exists():
        try:
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if isinstance(payload, Mapping):
                display_name = payload.get("display_name")
                if isinstance(display_name, str) and display_name.strip():
                    return display_name.strip()
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("Failed to parse %s: %s", config_path, exc)
    # Fallback: title-case folder name with spaces on hyphen/underscore.
    token = character_dir.name.replace("_", " ").replace("-", " ")
    return token.title()


def discover_pose_outfits(character_dir: Path) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    for pose_dir in sorted(
        (child for child in character_dir.iterdir() if child.is_dir()),
        key=lambda p: p.name.lower(),
    ):
        outfits_dir = pose_dir / "outfits"
        if not outfits_dir.exists():
            continue
        outfits: set[str] = set()
        for entry in outfits_dir.iterdir():
            if entry.is_file() and entry.suffix.lower() == ".png":
                outfits.add(entry.stem)
        for entry in outfits_dir.iterdir():
            if entry.is_dir():
                if entry.name.lower().startswith("acc"):
                    continue
                outfits.add(entry.name)
        if outfits:
            result[pose_dir.name] = sorted(outfits, key=lambda val: val.lower())
    return result


def ensure_character_entry(
    config_characters: MutableMapping[str, object],
    slug: str,
    display_name: str,
) -> MutableMapping[str, object]:
    entry = config_characters.get(slug)
    if not isinstance(entry, MutableMapping):
        entry = {
            "display_name": display_name,
            "rarity": COMMON_RARITY,
            "poses": {},
        }
        config_characters[slug] = entry
    else:
        entry.setdefault("display_name", display_name)
        entry.setdefault("rarity", COMMON_RARITY)
        entry.setdefault("poses", {})
    return entry


def ensure_pose_entry(character_entry: MutableMapping[str, object], pose: str) -> MutableMapping[str, object]:
    poses = character_entry.setdefault("poses", {})
    if not isinstance(poses, MutableMapping):
        poses = {}
        character_entry["poses"] = poses
    pose_entry = poses.get(pose)
    if not isinstance(pose_entry, MutableMapping):
        pose_entry = {"outfits": {}}
        poses[pose] = pose_entry
    else:
        existing_outfits = pose_entry.get("outfits")
        if not isinstance(existing_outfits, MutableMapping):
            pose_entry["outfits"] = {}
        if "rarity" in pose_entry:
            pose_entry.pop("rarity", None)
    return pose_entry


def ensure_outfit_entry(outfits: MutableMapping[str, object], outfit: str) -> bool:
    existing = outfits.get(outfit)
    if isinstance(existing, MutableMapping):
        existing.setdefault("rarity", COMMON_RARITY)
        return False
    if isinstance(existing, str):
        # Normalize string shorthand to object form.
        outfits[outfit] = {"rarity": existing}
        return False
    outfits[outfit] = {"rarity": COMMON_RARITY}
    return True


def apply_rarity_defaults(pose_outfits: MutableMapping[str, object]) -> None:
    """Assign default rarities: uniform > casual > all common."""
    if not pose_outfits:
        return
    normalized = {name.lower(): name for name in pose_outfits.keys()}
    priority = None
    for candidate in ("uniform", "casual"):
        if candidate in normalized:
            priority = normalized[candidate]
            break
    if priority is None:
        return  # everything already common
    for name, meta in pose_outfits.items():
        if not isinstance(meta, MutableMapping):
            meta = {"rarity": str(meta)}
            pose_outfits[name] = meta
        if name == priority:
            meta["rarity"] = COMMON_RARITY
        else:
            meta.setdefault("rarity", "rare")
            if meta["rarity"].lower() == COMMON_RARITY and name != priority:
                meta["rarity"] = "rare"


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")

    characters_root = resolve_characters_root(args.characters_root)
    config = load_config(args.config)
    characters_section: MutableMapping[str, object] = config["characters"]  # type: ignore[assignment]

    new_characters: List[str] = []
    new_outfits: List[str] = []

    for character_dir in sorted(characters_root.iterdir(), key=lambda p: p.name.lower()):
        if not character_dir.is_dir():
            continue
        slug = character_dir.name
        display_name = read_display_name(character_dir)
        pose_map = discover_pose_outfits(character_dir)
        if not pose_map:
            continue
        was_present = isinstance(characters_section.get(slug), MutableMapping)
        entry = ensure_character_entry(characters_section, slug, display_name)
        if not was_present:
            new_characters.append(slug)
        poses_entry = entry.setdefault("poses", {})
        if not isinstance(poses_entry, MutableMapping):
            poses_entry = {}
            entry["poses"] = poses_entry

        for pose, outfits in pose_map.items():
            pose_entry = ensure_pose_entry(entry, pose)
            pose_outfits = pose_entry["outfits"]  # type: ignore[index]
            if not isinstance(pose_outfits, MutableMapping):
                pose_outfits = {}
                pose_entry["outfits"] = pose_outfits
            for outfit in outfits:
                created = ensure_outfit_entry(pose_outfits, outfit)
                if created:
                    new_outfits.append(f"{slug}:{pose}:{outfit}")
            apply_rarity_defaults(pose_outfits)

    if args.dry_run:
        logger.info("Dry run complete: %d new characters, %d new pose/outfit combos detected.", len(new_characters), len(new_outfits))
        return

    args.config.parent.mkdir(parents=True, exist_ok=True)
    args.config.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info(
        "Updated %s (%d new characters, %d new pose/outfit combos).",
        args.config,
        len(new_characters),
        len(new_outfits),
    )


if __name__ == "__main__":
    main()
