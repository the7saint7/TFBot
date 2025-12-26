#!/usr/bin/env python3
"""Sync gacha_config.json with the VNBot character roster."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None  # type: ignore[assignment]

import yaml

DEFAULT_RARITY = "rare"
IMAGE_SUFFIXES = {".png", ".webp", ".jpg", ".jpeg"}


def _normalize_key(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum() or ch == "_")


def _load_tf_characters(module_path: Path) -> Sequence[Mapping[str, str]]:
    spec = importlib.util.spec_from_file_location("tf_characters", module_path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Unable to import tf_characters from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["tf_characters"] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    data = getattr(module, "TF_CHARACTERS", None)
    if not isinstance(data, list):
        raise RuntimeError("tf_characters module does not expose TF_CHARACTERS list.")
    return data


def _determine_slug(folder: str, existing: MutableMapping[str, dict], display_name: str) -> str:
    """Use folder name as canonical slug so each folder maps to a single gacha entry."""
    folder_token = folder.strip()
    if folder_token:
        return folder_token
    fallback = _normalize_key(display_name)
    if fallback:
        return fallback
    suffix = 1
    slug = "character"
    while slug in existing:
        suffix += 1
        slug = f"character{suffix}"
    return slug


def _resolve_characters_root(repo_dir: Path, subdir: str) -> Path:
    base = repo_dir
    if not base.is_absolute():
        base = (Path.cwd() / base).resolve()
    if subdir:
        base = base / subdir
    return base


def _load_character_yaml(character_dir: Path) -> Mapping[str, object]:
    yml_path = character_dir / "character.yml"
    if not yml_path.exists():
        return {}
    try:
        data = yaml.safe_load(yml_path.read_text(encoding="utf-8")) or {}
        if isinstance(data, Mapping):
            return data
    except yaml.YAMLError as exc:  # pragma: no cover
        print(f"[WARN] Failed to parse {yml_path}: {exc}")
    return {}


def _discover_pose_names(character_dir: Path, metadata: Mapping[str, object]) -> List[str]:
    pose_names: List[str] = []
    pose_section = metadata.get("poses")
    if isinstance(pose_section, Mapping):
        pose_names.extend(str(key) for key in pose_section.keys())
    for child in sorted(character_dir.iterdir()):
        if child.is_dir() and len(child.name) == 1 and child.name.isalpha():
            pose_names.append(child.name)
    seen = set()
    unique: List[str] = []
    for pose in pose_names:
        if pose not in seen:
            seen.add(pose)
            unique.append(pose)
    return unique


def _discover_outfits(pose_dir: Path) -> List[str]:
    outfits: List[str] = []
    outfits_dir = pose_dir / "outfits"
    if not outfits_dir.exists():
        return outfits
    for file_path in sorted(outfits_dir.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_SUFFIXES:
            outfits.append(file_path.stem)
    return outfits


@dataclass
class CharacterAssets:
    slug: str
    display_name: str
    folder: str
    poses: Dict[str, List[str]]


def _collect_character_assets(
    dataset: Sequence[Mapping[str, str]],
    characters_root: Path,
    config_characters: MutableMapping[str, dict],
) -> List[CharacterAssets]:
    assets: List[CharacterAssets] = []
    for entry in dataset:
        folder = entry.get("folder")
        name = entry.get("name")
        if not folder or not name:
            continue
        character_dir = characters_root / folder
        if not character_dir.exists():
            continue

        metadata = _load_character_yaml(character_dir)
        display_name = metadata.get("display_name") if isinstance(metadata, Mapping) else None
        if not isinstance(display_name, str) or not display_name.strip():
            display_name = name

        slug = _determine_slug(folder, config_characters, display_name)
        pose_names = _discover_pose_names(character_dir, metadata)
        pose_map: Dict[str, List[str]] = {}
        for pose in pose_names:
            pose_dir = character_dir / pose
            if not pose_dir.exists():
                continue
            outfits = _discover_outfits(pose_dir)
            if outfits:
                pose_map[pose] = outfits
        if not pose_map:
            continue
        assets.append(CharacterAssets(slug=slug, display_name=display_name, folder=folder, poses=pose_map))
    return assets


def _find_existing_character_entry(
    characters: MutableMapping[str, object],
    slug: str,
) -> tuple[str, Optional[MutableMapping[str, object]]]:
    entry = characters.get(slug)
    if isinstance(entry, MutableMapping):
        return slug, entry
    for key, value in characters.items():
        if _normalize_key(key) == slug and isinstance(value, MutableMapping):
            return key, value
    return slug, None


def _update_config(
    config: MutableMapping[str, object],
    assets: Sequence[CharacterAssets],
) -> Dict[str, int]:
    characters = config.setdefault("characters", {})
    if not isinstance(characters, MutableMapping):
        raise RuntimeError("gacha_config['characters'] must be a mapping.")

    stats = defaultdict(int)
    for asset in assets:
        _, entry = _find_existing_character_entry(characters, asset.slug)
        if entry is None:
            entry = {}
            characters[asset.slug] = entry
            stats["characters_added"] += 1

        if "display_name" not in entry:
            entry["display_name"] = asset.display_name
        if "rarity" not in entry:
            entry["rarity"] = DEFAULT_RARITY

        poses_section = entry.setdefault("poses", {})
        if not isinstance(poses_section, MutableMapping):
            poses_section = {}
            entry["poses"] = poses_section
        for pose_name, outfits in asset.poses.items():
            pose_entry = poses_section.setdefault(pose_name, {})
            if not isinstance(pose_entry, MutableMapping):
                pose_entry = {}
                poses_section[pose_name] = pose_entry

            outfit_section = pose_entry.setdefault("outfits", {})
            if not isinstance(outfit_section, MutableMapping):
                outfit_section = {}
                pose_entry["outfits"] = outfit_section

            for outfit in outfits:
                if outfit not in outfit_section:
                    outfit_section[outfit] = {"rarity": DEFAULT_RARITY}
                    stats["outfits_added"] += 1
    return stats


def _write_config(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update gacha_config.json with VNBot characters.")
    parser.add_argument(
        "--config",
        type=Path,
        default=os.getenv("TFBOT_GACHA_CONFIG", "gacha_config.json"),
        help="Path to gacha_config.json (defaults to TFBOT_GACHA_CONFIG or gacha_config.json).",
    )
    parser.add_argument(
        "--characters-module",
        type=Path,
        default=Path("tf_characters.py"),
        help="Path to tf_characters.py (defaults to project root file).",
    )
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=Path(os.getenv("TFBOT_CHARACTERS_REPO_DIR", "characters_repo")),
        help="Root directory of the synced characters repository.",
    )
    parser.add_argument(
        "--repo-subdir",
        type=str,
        default=os.getenv("TFBOT_CHARACTERS_REPO_SUBDIR", "characters"),
        help="Subdirectory inside the characters repo that holds character folders.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show planned changes without writing the file.")
    return parser.parse_args()


def main() -> int:
    if load_dotenv:
        load_dotenv()

    args = parse_args()
    config_path = args.config if args.config.is_absolute() else (Path.cwd() / args.config).resolve()
    module_path = args.characters_module
    if not module_path.is_absolute():
        module_path = (Path.cwd() / module_path).resolve()

    if not config_path.exists():
        print(f"[ERROR] Config file {config_path} not found.", file=sys.stderr)
        return 1

    dataset = _load_tf_characters(module_path)

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"[ERROR] Failed to parse {config_path}: {exc}", file=sys.stderr)
        return 1

    characters_root = _resolve_characters_root(args.repo_dir, args.repo_subdir)
    assets = _collect_character_assets(dataset, characters_root, config.get("characters", {}))
    stats = _update_config(config, assets)

    print(
        f"Collected {len(assets)} characters for {os.getenv('TFBOT_NAME', 'VNBot')} "
        f"({stats['characters_added']} new entries, {stats['outfits_added']} outfits added)."
    )

    if args.dry_run:
        print("[DRY-RUN] Skipping write.")
        return 0

    _write_config(config_path, config)
    print(f"Wrote updates to {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
