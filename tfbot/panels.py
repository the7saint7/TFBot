"""Visual novel panel rendering and helpers."""

from __future__ import annotations

import io
import json
import logging
import os
import random
import re
from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import aiohttp
import discord

from tfbot.models import OutfitAsset, ReplyContext, TransformationState
from tfbot.utils import float_from_env, normalize_pose_name, path_from_env, utc_now

logger = logging.getLogger("tfbot.panels")

PACKAGE_DIR = Path(__file__).resolve().parent
BASE_DIR = PACKAGE_DIR.parent

VN_BASE_IMAGE = Path(os.getenv("TFBOT_VN_BASE", "vn_assets/vn_base.png"))
VN_FONT_PATH = os.getenv("TFBOT_VN_FONT", "").strip()

VN_FONT_REGULAR_PATH = (
    path_from_env("TFBOT_VN_FONT_REGULAR")
    or (Path(VN_FONT_PATH).expanduser() if VN_FONT_PATH else None)
)
VN_FONT_BOLD_PATH = path_from_env("TFBOT_VN_FONT_BOLD")
VN_FONT_ITALIC_PATH = path_from_env("TFBOT_VN_FONT_ITALIC")
VN_FONT_BOLD_ITALIC_PATH = path_from_env("TFBOT_VN_FONT_BOLD_ITALIC")

_FONT_STYLE_PATHS: Dict[str, Optional[Path]] = {
    "regular": VN_FONT_REGULAR_PATH,
    "bold": VN_FONT_BOLD_PATH,
    "italic": VN_FONT_ITALIC_PATH,
    "bold_italic": VN_FONT_BOLD_ITALIC_PATH,
}

VN_NAME_FONT_SIZE = int(os.getenv("TFBOT_VN_NAME_SIZE", "34"))
VN_TEXT_FONT_SIZE = int(os.getenv("TFBOT_VN_TEXT_SIZE", "26"))
VN_GAME_ROOT = (
    Path(os.getenv("TFBOT_VN_GAME_ROOT", "")).expanduser().resolve()
    if os.getenv("TFBOT_VN_GAME_ROOT")
    else None
)
VN_ASSET_ROOT = VN_GAME_ROOT / "game" / "images" / "characters" if VN_GAME_ROOT else None
VN_DEFAULT_OUTFIT = os.getenv("TFBOT_VN_OUTFIT", "casual.png")
VN_DEFAULT_FACE = os.getenv("TFBOT_VN_FACE", "0.png")
VN_AVATAR_MODE = os.getenv("TFBOT_VN_AVATAR_MODE", "game").lower()
VN_AVATAR_SCALE = max(0.1, float_from_env("TFBOT_VN_AVATAR_SCALE", 1.0))
_VN_BG_ROOT_SETTING = os.getenv("TFBOT_VN_BG_ROOT", "").strip()
_VN_BG_DEFAULT_SETTING = os.getenv("TFBOT_VN_BG_DEFAULT", "school/cafeteria.png").strip()
VN_BACKGROUND_DEFAULT_RELATIVE = Path(_VN_BG_DEFAULT_SETTING) if _VN_BG_DEFAULT_SETTING else None

if _VN_BG_ROOT_SETTING:
    candidate_bg_root = Path(_VN_BG_ROOT_SETTING).expanduser()
    VN_BACKGROUND_ROOT = candidate_bg_root if candidate_bg_root.exists() else None
elif VN_GAME_ROOT:
    candidate_bg_root = VN_GAME_ROOT / "game" / "images" / "bg"
    VN_BACKGROUND_ROOT = candidate_bg_root if candidate_bg_root.exists() else None
else:
    VN_BACKGROUND_ROOT = None

if VN_BACKGROUND_ROOT and VN_BACKGROUND_DEFAULT_RELATIVE:
    VN_BACKGROUND_DEFAULT_PATH = (VN_BACKGROUND_ROOT / VN_BACKGROUND_DEFAULT_RELATIVE).resolve()
    if not VN_BACKGROUND_DEFAULT_PATH.exists():
        logger.warning(
            "VN background: default background %s does not exist under %s",
            VN_BACKGROUND_DEFAULT_RELATIVE,
            VN_BACKGROUND_ROOT,
        )
        VN_BACKGROUND_DEFAULT_PATH = None
else:
    VN_BACKGROUND_DEFAULT_PATH = None

_BG_SELECTION_FILE_SETTING = os.getenv("TFBOT_VN_BG_SELECTIONS", "tf_backgrounds.json").strip()
VN_BACKGROUND_SELECTION_FILE = (
    Path(_BG_SELECTION_FILE_SETTING) if _BG_SELECTION_FILE_SETTING else None
)
VN_NAME_DEFAULT_COLOR: Tuple[int, int, int, int] = (255, 220, 180, 255)
_VN_CACHE_DIR_SETTING = os.getenv("TFBOT_VN_CACHE_DIR", "vn_cache").strip()
if _VN_CACHE_DIR_SETTING:
    _vn_cache_path = Path(_VN_CACHE_DIR_SETTING)
    if not _vn_cache_path.is_absolute():
        VN_CACHE_DIR = (BASE_DIR / _vn_cache_path).resolve()
    else:
        VN_CACHE_DIR = _vn_cache_path.resolve()
else:
    VN_CACHE_DIR = None

VN_SELECTION_FILE = Path(os.getenv("TFBOT_VN_SELECTIONS", "tf_outfits.json"))

vn_outfit_selection: Dict[str, Dict[str, str]] = {}
background_selections: Dict[str, str] = {}
_vn_config_cache: Dict[str, Dict] = {}
_VN_BACKGROUND_IMAGES: list[Path] = []


def load_outfit_selections() -> Dict[str, Dict[str, str]]:
    if not VN_SELECTION_FILE.exists():
        return {}
    try:
        data = json.loads(VN_SELECTION_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            normalized: Dict[str, Dict[str, str]] = {}
            for key, value in data.items():
                entry: Dict[str, str] = {}
                pose_value: Optional[str] = None
                outfit_value: Optional[str] = None
                if isinstance(value, dict):
                    pose_raw = value.get("pose")
                    outfit_raw = value.get("outfit") or value.get("name")
                    if isinstance(pose_raw, str):
                        pose_value = pose_raw.strip()
                    elif pose_raw is not None:
                        pose_value = str(pose_raw).strip()
                    if isinstance(outfit_raw, str):
                        outfit_value = outfit_raw.strip()
                    elif outfit_raw is not None:
                        outfit_value = str(outfit_raw).strip()
                elif isinstance(value, str):
                    outfit_value = value.strip()
                elif value is not None:
                    outfit_value = str(value).strip()

                if outfit_value:
                    if pose_value:
                        entry["pose"] = pose_value
                    entry["outfit"] = outfit_value
                    normalized[str(key).lower()] = entry
            return normalized
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse %s: %s", VN_SELECTION_FILE, exc)
    return {}


def persist_outfit_selections() -> None:
    try:
        VN_SELECTION_FILE.parent.mkdir(parents=True, exist_ok=True)
        VN_SELECTION_FILE.write_text(json.dumps(vn_outfit_selection, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning("Failed to persist outfit selections: %s", exc)


def load_background_selections() -> Dict[str, str]:
    if VN_BACKGROUND_SELECTION_FILE is None or not VN_BACKGROUND_SELECTION_FILE.exists():
        return {}
    try:
        data = json.loads(VN_BACKGROUND_SELECTION_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            normalized: Dict[str, str] = {}
            for key, value in data.items():
                if not isinstance(value, str):
                    value = str(value)
                normalized[str(key)] = value.strip()
            return normalized
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse %s: %s", VN_BACKGROUND_SELECTION_FILE, exc)
    return {}


def persist_background_selections() -> None:
    if VN_BACKGROUND_SELECTION_FILE is None:
        return
    try:
        VN_BACKGROUND_SELECTION_FILE.parent.mkdir(parents=True, exist_ok=True)
        VN_BACKGROUND_SELECTION_FILE.write_text(
            json.dumps(background_selections, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("Failed to persist background selections: %s", exc)


vn_outfit_selection = load_outfit_selections()
background_selections = load_background_selections()


def _relative_background_path(path: Path) -> Optional[str]:
    if VN_BACKGROUND_ROOT is None:
        return None
    try:
        relative = path.resolve().relative_to(VN_BACKGROUND_ROOT.resolve())
    except ValueError:
        return None
    return relative.as_posix()


def _load_background_images() -> Sequence[Path]:
    global _VN_BACKGROUND_IMAGES
    if _VN_BACKGROUND_IMAGES:
        return _VN_BACKGROUND_IMAGES
    if VN_BACKGROUND_ROOT and VN_BACKGROUND_ROOT.exists():
        try:
            _VN_BACKGROUND_IMAGES = sorted(
                path for path in VN_BACKGROUND_ROOT.rglob("*.png") if path.is_file()
            )
        except OSError as exc:
            logger.warning("VN background: failed to scan directory %s: %s", VN_BACKGROUND_ROOT, exc)
            _VN_BACKGROUND_IMAGES = []
    else:
        _VN_BACKGROUND_IMAGES = []
    return _VN_BACKGROUND_IMAGES


def list_background_choices() -> Sequence[Path]:
    return list(_load_background_images())


def get_selected_background_path(user_id: int) -> Optional[Path]:
    if VN_BACKGROUND_ROOT is None:
        return None
    key = str(user_id)
    selected = background_selections.get(key)
    if selected:
        candidate = (VN_BACKGROUND_ROOT / selected).resolve()
        if candidate.exists():
            return candidate
        logger.warning("VN background: stored selection %s missing for user %s", selected, user_id)
        background_selections.pop(key, None)
        persist_background_selections()
    if VN_BACKGROUND_DEFAULT_PATH and VN_BACKGROUND_DEFAULT_PATH.exists():
        return VN_BACKGROUND_DEFAULT_PATH
    backgrounds = _load_background_images()
    if backgrounds:
        return backgrounds[0]
    return None


def set_selected_background(user_id: int, background_path: Path) -> bool:
    if VN_BACKGROUND_ROOT is None:
        return False
    relative = _relative_background_path(background_path)
    if not relative:
        return False
    background_selections[str(user_id)] = relative
    persist_background_selections()
    return True


def compose_background_layer(
    panel_size: Tuple[int, int],
    background_path: Optional[Path],
) -> Optional["Image.Image"]:
    backgrounds = _load_background_images()
    if background_path is None or not background_path.exists():
        if not backgrounds:
            return None
        background_path = random.choice(backgrounds)
    try:
        from PIL import Image, ImageOps

        with Image.open(background_path) as background_image:
            fitted = ImageOps.fit(
                background_image.convert("RGBA"),
                panel_size,
                Image.LANCZOS,
                centering=(0.5, 0.5),
            )
    except OSError as exc:
        logger.warning("VN background: failed to load %s: %s", background_path, exc)
        try:
            _VN_BACKGROUND_IMAGES.remove(background_path)
        except ValueError:
            pass
        return None
    layer = Image.new("RGBA", panel_size, (0, 0, 0, 0))
    layer.paste(fitted, (0, 0), fitted)
    return layer


def _default_accessory_layer(accessory_dir: Path) -> Optional[Path]:
    if not accessory_dir.is_dir():
        return None
    pngs = sorted(p for p in accessory_dir.rglob("*.png") if p.is_file())
    if not pngs:
        return None
    for candidate in pngs:
        if candidate.stem.lower() == "on":
            return candidate
    for candidate in pngs:
        if "on" in candidate.stem.lower():
            return candidate
    for candidate in pngs:
        parents = [parent.name.lower() for parent in candidate.parents]
        if "on" in parents:
            return candidate
    return pngs[0]


def _collect_accessory_layers(entry: Path, include_all: bool = False) -> list[Tuple[int, Path]]:
    accessories: list[Tuple[int, Path]] = []
    seen_paths: set[Path] = set()

    def _add_layer(path: Path, source_name: str) -> None:
        normalized_name = source_name.lower()
        order = 0
        if "-" in normalized_name:
            suffix = normalized_name.rsplit("-", 1)[-1]
            if suffix.lstrip("-").isdigit():
                try:
                    order = int(suffix)
                except ValueError:
                    order = 0
        if path not in seen_paths:
            accessories.append((order, path))
            seen_paths.add(path)

    for child in sorted(
        (
            child
            for child in entry.iterdir()
            if include_all or child.name.lower().startswith("acc")
        ),
        key=lambda p: p.name.lower(),
    ):
        if child.is_dir():
            layer = _default_accessory_layer(child)
            if layer:
                _add_layer(layer, child.name)
        elif child.is_file() and child.suffix.lower() == ".png":
            _add_layer(child, child.stem)

    if not accessories and entry.is_dir() and (include_all or entry.name.lower().startswith("acc")):
        layer = _default_accessory_layer(entry)
        if layer:
            _add_layer(layer, entry.name)

    accessories.sort(key=lambda item: item[0])
    return accessories


def _discover_outfit_assets(variant_dir: Path) -> Dict[str, OutfitAsset]:
    assets: Dict[str, OutfitAsset] = {}
    outfits_dir = variant_dir / "outfits"
    if not outfits_dir.exists():
        return assets
    entries = sorted(outfits_dir.iterdir(), key=lambda p: p.name.lower())

    for entry in entries:
        if entry.is_file() and entry.suffix.lower() == ".png":
            name = entry.stem
            assets[name.lower()] = OutfitAsset(name=name, base_path=entry, accessory_layers=())

    for entry in entries:
        if entry.is_dir() and not entry.name.lower().startswith("acc"):
            primary = entry / f"{entry.name}.png"
            if not primary.exists():
                primary = next((p for p in entry.glob("*.png")), None)
            if not primary:
                primary = next((p for p in entry.rglob("*.png")), None)
            if not primary:
                continue
            accessories = []
            accessories.extend(_collect_accessory_layers(entry, include_all=True))
            assets[entry.name.lower()] = OutfitAsset(
                name=entry.name,
                base_path=primary,
                accessory_layers=tuple(layer for _, layer in accessories),
            )

    for entry in entries:
        if entry.is_dir() and entry.name.lower().startswith("acc"):
            global_layers = _collect_accessory_layers(entry)
            if global_layers:
                for key, asset in list(assets.items()):
                    combined = list(asset.accessory_layers)
                    combined.extend(layer for _, layer in global_layers)
                    assets[key] = OutfitAsset(
                        name=asset.name,
                        base_path=asset.base_path,
                        accessory_layers=tuple(combined),
                    )
    return assets


def _load_character_config(character_dir: Path) -> Dict:
    key = character_dir.resolve().as_posix()
    cached = _vn_config_cache.get(key)
    if cached is not None:
        return cached
    config_path = character_dir / "character.yml"
    result: Dict = {}
    if config_path.exists():
        try:
            import yaml

            result = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("VN sprite: failed to parse %s: %s", config_path, exc)
            result = {}
    _vn_config_cache[key] = result
    return result


def _ordered_variant_dirs(character_dir: Path, config: Dict) -> Sequence[Path]:
    variants = {child.name.lower(): child for child in character_dir.iterdir() if child.is_dir()}
    if not variants:
        return []
    preferred_names: list[str] = []
    poses = config.get("poses")
    if isinstance(poses, dict):
        preferred_names.extend(name.lower() for name in poses.keys())
    preferred_names.append("a")
    preferred_names.extend(sorted(variants.keys()))

    ordered: list[Path] = []
    seen: set[str] = set()
    for name in preferred_names:
        if name in variants and name not in seen:
            ordered.append(variants[name])
            seen.add(name)
    return ordered


def _candidate_character_keys(raw_name: str) -> Sequence[str]:
    name = raw_name.strip()
    if not name:
        return []
    first_token = name.split(" ", 1)[0]
    if not first_token:
        return []
    candidates: list[str] = []

    def _add(token: str) -> None:
        value = token.strip()
        if value and value not in candidates:
            candidates.append(value)

    primary = first_token.lower()
    _add(primary)

    stripped = primary.replace('"', "").replace("'", "")
    _add(stripped)

    hyphen_base = stripped.split("-", 1)[0]
    _add(hyphen_base)

    hyphen_removed = stripped.replace("-", "")
    _add(hyphen_removed)

    underscore_variant = stripped.replace("-", "_")
    _add(underscore_variant)

    return candidates


def resolve_character_directory(character_name: str) -> Tuple[Optional[Path], Sequence[str]]:
    if VN_ASSET_ROOT is None:
        return None, []
    attempted: list[str] = []
    for key in _candidate_character_keys(character_name):
        candidate = VN_ASSET_ROOT / key
        attempted.append(candidate.name)
        if candidate.exists():
            return candidate, attempted
    return None, attempted


def list_pose_outfits(character_name: str) -> Dict[str, list[str]]:
    directory, attempted = resolve_character_directory(character_name)
    if directory is None:
        logger.debug("VN sprite: cannot list outfits for %s (tried %s)", character_name, attempted)
        return {}
    config = _load_character_config(directory)
    variant_dirs = _ordered_variant_dirs(directory, config)
    if not variant_dirs:
        return {}
    pose_map: Dict[str, list[str]] = {}
    for variant_dir in variant_dirs:
        assets = _discover_outfit_assets(variant_dir)
        if assets:
            pose_map[variant_dir.name] = sorted(asset.name for asset in assets.values())
    return pose_map


def list_available_outfits(character_name: str) -> Sequence[str]:
    pose_map = list_pose_outfits(character_name)
    outfits: set[str] = set()
    for options in pose_map.values():
        outfits.update(options)
    return sorted(outfits)


def get_selected_outfit_for_dir(directory: Path) -> Optional[str]:
    _, outfit = get_selected_pose_outfit_for_dir(directory)
    return outfit


def get_selected_pose_outfit_for_dir(directory: Path) -> Tuple[Optional[str], Optional[str]]:
    entry = vn_outfit_selection.get(directory.name.lower())
    if not entry:
        return None, None
    pose: Optional[str] = None
    outfit: Optional[str] = None
    if isinstance(entry, dict):
        pose_raw = entry.get("pose")
        outfit_raw = entry.get("outfit") or entry.get("name")
        if isinstance(pose_raw, str):
            pose = pose_raw.strip() or None
        elif pose_raw is not None:
            pose = str(pose_raw).strip() or None
        if isinstance(outfit_raw, str):
            outfit = outfit_raw.strip() or None
        elif outfit_raw is not None:
            outfit = str(outfit_raw).strip() or None
    elif isinstance(entry, str):
        outfit = entry.strip() or None
    else:
        outfit = str(entry).strip() or None
    return pose, outfit


def get_selected_outfit_name(character_name: str) -> Optional[str]:
    _, outfit = get_selected_pose_outfit(character_name)
    return outfit


def get_selected_pose_outfit(character_name: str) -> Tuple[Optional[str], Optional[str]]:
    directory, _ = resolve_character_directory(character_name)
    if directory is None:
        return None, None
    return get_selected_pose_outfit_for_dir(directory)


def set_selected_outfit_name(character_name: str, outfit_name: str) -> bool:
    return set_selected_pose_outfit(character_name, None, outfit_name)


def set_selected_pose_outfit(
    character_name: str,
    pose_name: Optional[str],
    outfit_name: str,
) -> bool:
    directory, attempted = resolve_character_directory(character_name)
    if directory is None:
        logger.debug("VN sprite: cannot set outfit for %s (tried %s)", character_name, attempted)
        return False
    pose_outfits = list_pose_outfits(character_name)
    if not pose_outfits:
        return False

    normalized_outfit = outfit_name.strip()
    if not normalized_outfit:
        return False
    normalized_pose = normalize_pose_name(pose_name)

    matched_pose: Optional[str] = None
    matched_outfit: Optional[str] = None

    for pose, outfits in pose_outfits.items():
        outfit_lookup = {option.lower(): option for option in outfits}
        option = outfit_lookup.get(normalized_outfit.lower())
        if not option:
            continue
        if normalized_pose is None or pose.lower() == normalized_pose:
            matched_pose = pose
            matched_outfit = option
            break

    if matched_outfit is None:
        if normalized_pose:
            logger.debug(
                "VN sprite: pose %s not available for outfit %s (character %s)",
                normalized_pose,
                outfit_name,
                character_name,
            )
            return False
        for pose, outfits in pose_outfits.items():
            outfit_lookup = {option.lower(): option for option in outfits}
            option = outfit_lookup.get(normalized_outfit.lower())
            if option:
                matched_pose = pose
                matched_outfit = option
                break

    if matched_outfit is None or matched_pose is None:
        return False

    key = directory.name.lower()
    vn_outfit_selection[key] = {"pose": matched_pose, "outfit": matched_outfit}
    persist_outfit_selections()
    compose_game_avatar.cache_clear()
    logger.info(
        "VN sprite: outfit override for %s set to pose %s outfit %s",
        directory.name,
        matched_pose,
        matched_outfit,
    )
    return True


@lru_cache(maxsize=128)
def compose_game_avatar(character_name: str) -> Optional["Image.Image"]:
    if VN_ASSET_ROOT is None:
        return None
    try:
        from PIL import Image
    except ImportError:
        return None

    character_dir, attempted = resolve_character_directory(character_name)
    if character_dir is None:
        logger.debug("VN sprite: missing character directory for %s (tried %s)", character_name, attempted)
        return None

    config = _load_character_config(character_dir)
    variant_dirs = _ordered_variant_dirs(character_dir, config)
    if not variant_dirs:
        logger.debug("VN sprite: no variant directory found for %s", character_dir.name)
        return None

    preferred_pose, preferred_outfit = get_selected_pose_outfit_for_dir(character_dir)
    if preferred_outfit:
        logger.debug(
            "VN sprite: preferred outfit override for %s is pose %s outfit %s",
            character_dir.name,
            preferred_pose or "auto",
            preferred_outfit,
        )

    variant_dir, outfit_asset = _select_outfit_path(
        variant_dirs,
        config,
        preferred_outfit,
        preferred_pose,
    )
    if not variant_dir or not outfit_asset:
        logger.warning(
            "VN sprite: outfit not found for %s (variant %s)",
            character_dir.name,
            variant_dir.name if variant_dir else "unknown",
        )
        return None

    outfit_path = outfit_asset.base_path
    face_path = _select_face_path(variant_dir)

    cache_file: Optional[Path] = None
    if VN_CACHE_DIR:
        face_token = "noface"
        if face_path and face_path.exists():
            face_token = face_path.stem.lower()
        accessory_token = "noacc"
        if outfit_asset.accessory_layers:
            accessory_token = "-".join(layer.stem.lower() for layer in outfit_asset.accessory_layers)
        cache_dir = VN_CACHE_DIR / character_dir.name.lower() / variant_dir.name.lower()
        cache_file = cache_dir / f"{outfit_path.stem.lower()}__{face_token}__{accessory_token}.png"
        if cache_file.exists():
            try:
                cached = Image.open(cache_file).convert("RGBA")
                logger.debug("VN sprite: loaded cached avatar %s", cache_file)
                return cached
            except OSError as exc:
                logger.warning("VN sprite: failed to load cached avatar %s: %s (rebuilding)", cache_file, exc)

    try:
        outfit_image = Image.open(outfit_path).convert("RGBA")
    except OSError as exc:
        logger.warning("Failed to load outfit %s: %s", outfit_path, exc)
        return None

    for layer_path in outfit_asset.accessory_layers:
        if not layer_path.exists():
            continue
        try:
            layer_image = Image.open(layer_path).convert("RGBA")
            outfit_image.paste(layer_image, (0, 0), layer_image)
        except OSError as exc:
            logger.warning("Failed to load accessory %s: %s", layer_path, exc)

    if face_path and face_path.exists():
        try:
            face_image = Image.open(face_path).convert("RGBA")
            outfit_image.paste(face_image, (0, 0), face_image)
        except OSError as exc:
            logger.warning("Failed to load face %s: %s", face_path, exc)
    else:
        logger.warning(
            "VN sprite: face image missing for %s (variant %s, searched %s)",
            character_dir.name,
            variant_dir.name,
            face_path,
        )

    pose_metadata = _get_pose_metadata(config, variant_dir.name)
    facing = str(pose_metadata.get("facing") or config.get("facing") or "left").lower()
    logger.debug("VN sprite: pose %s facing=%s", variant_dir.name, facing)
    if facing != "right":
        from PIL import ImageOps

        outfit_image = ImageOps.mirror(outfit_image)
        logger.debug("VN sprite: sprite mirrored for pose %s", variant_dir.name)

    outfit_image = _crop_transparent_vertical(outfit_image)

    if VN_CACHE_DIR and cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            outfit_image.save(cache_file, format="PNG")
            logger.debug("VN sprite: cached avatar %s", cache_file)
        except OSError as exc:
            logger.warning("VN sprite: unable to cache avatar %s: %s", cache_file, exc)

    return outfit_image


def _get_pose_metadata(config: Dict, pose_name: str) -> Dict:
    poses = config.get("poses")
    if isinstance(poses, dict):
        value = poses.get(pose_name.lower())
        if isinstance(value, dict):
            return value
    return {}


def _select_outfit_path(
    variant_dirs: Sequence[Path],
    config: Dict,
    preferred: Optional[str] = None,
    preferred_pose: Optional[str] = None,
) -> Tuple[Optional[Path], Optional[OutfitAsset]]:
    if not variant_dirs:
        return None, None

    variant_outfits: dict[Path, Dict[str, OutfitAsset]] = {}
    for variant_dir in variant_dirs:
        assets = _discover_outfit_assets(variant_dir)
        if not assets:
            continue
        variant_outfits[variant_dir] = assets

    if not variant_outfits:
        logger.warning(
            "VN sprite: no outfits discovered for variants %s",
            ", ".join(v.name for v in variant_dirs),
        )
        return None, None

    search_variants = [variant for variant in variant_dirs if variant in variant_outfits]
    if not search_variants:
        return None, None

    normalized_pose = normalize_pose_name(preferred_pose)
    if normalized_pose:
        search_variants.sort(key=lambda var: 0 if var.name.lower() == normalized_pose else 1)

    candidates: list[str] = []
    if preferred:
        candidates.append(preferred)
    default_outfit = config.get("default_outfit")
    if isinstance(default_outfit, str):
        candidates.append(default_outfit)
    if VN_DEFAULT_OUTFIT:
        candidates.append(VN_DEFAULT_OUTFIT)

    normalized_targets: list[str] = []
    for name in candidates:
        name = name.strip()
        if not name:
            continue
        lower = name.lower()
        if lower.endswith(".png"):
            normalized_targets.append(lower)
            normalized_targets.append(lower.rstrip(".png"))
        else:
            normalized_targets.append(lower)

    for target in normalized_targets:
        for variant_dir in search_variants:
            assets = variant_outfits.get(variant_dir, {})
            asset = assets.get(target)
            if asset is None:
                asset = assets.get(target.rstrip(".png"))
            if asset:
                logger.debug(
                    "VN sprite: using outfit %s for variant %s",
                    asset.base_path.name,
                    variant_dir.name,
                )
                return variant_dir, asset

    for variant_dir in search_variants:
        assets = variant_outfits.get(variant_dir, {})
        if assets:
            first_asset = next(iter(sorted(assets.values(), key=lambda a: a.name.lower())))
            logger.debug(
                "VN sprite: defaulting to first outfit %s for variant %s",
                first_asset.base_path.name,
                variant_dir.name,
            )
            return variant_dir, first_asset

    return None, None


def _select_face_path(variant_dir: Path) -> Optional[Path]:
    faces_dir = variant_dir / "faces"
    if not faces_dir.exists():
        logger.warning("VN sprite: faces directory missing at %s", faces_dir)
        return None
    groups = [d for d in faces_dir.iterdir() if d.is_dir()]
    if not groups:
        logger.warning("VN sprite: no face groups found in %s", faces_dir)
        return None

    group_preference = ["face", "neutral", "default"]
    selected_group = None
    for candidate in group_preference:
        for group in groups:
            if group.name.lower() == candidate:
                selected_group = group
                break
        if selected_group:
            break
    if selected_group is None:
        selected_group = groups[0]

    faces = sorted(
        (face_path for face_path in selected_group.glob("*.png") if face_path.is_file()),
        key=lambda p: p.name.lower(),
    )
    if not faces:
        logger.warning("VN sprite: no faces found in group %s", selected_group)
        return None
    preferred_face = VN_DEFAULT_FACE.lower()
    for face in faces:
        if face.name.lower() == preferred_face:
            return face
    return faces[0]


def _parse_hex_color(raw_color: str) -> Optional[Tuple[int, int, int, int]]:
    if not raw_color:
        return None
    value = raw_color.strip()
    if not value:
        return None
    if value.lower().startswith("0x"):
        value = value[2:]
    if value.startswith("#"):
        value = value[1:]
    if len(value) not in {6, 8}:
        return None
    try:
        components = [int(value[i : i + 2], 16) for i in range(0, len(value), 2)]
    except ValueError:
        return None
    if len(components) == 3:
        components.append(255)
    if len(components) != 4:
        return None
    r, g, b, a = components[:4]
    return r, g, b, a


def resolve_character_name_color(character_name: str) -> Tuple[int, int, int, int]:
    if not character_name:
        return VN_NAME_DEFAULT_COLOR
    directory, _ = resolve_character_directory(character_name)
    if directory is None:
        return VN_NAME_DEFAULT_COLOR
    config = _load_character_config(directory)
    raw_color = config.get("name_color") or config.get("text_color")
    if isinstance(raw_color, str):
        parsed = _parse_hex_color(raw_color)
        if parsed:
            return parsed
    if isinstance(raw_color, Sequence) and not isinstance(raw_color, (str, bytes, bytearray)):
        components = list(raw_color)
        if 3 <= len(components) <= 4 and all(isinstance(c, (int, float)) for c in components):
            channel_values = [max(0, min(255, int(c))) for c in components[:4]]
            if len(channel_values) == 3:
                channel_values.append(255)
            if len(channel_values) == 4:
                r, g, b, a = channel_values
                return r, g, b, a
    return VN_NAME_DEFAULT_COLOR


URL_RE = re.compile(r"https?://\S+")
CUSTOM_EMOJI_RE = re.compile(r"<(a?):([a-zA-Z0-9_]{2,}):(\d+)>")
MENTION_TOKEN_RE = re.compile(r"<(@!?|@&|#)(\d+)>")
MENTION_PLACEHOLDER_PATTERN = re.compile(r"\uFFF0([a-z]+):(\d+)\uFFF1")
MENTION_COLORS = {
    "user": (114, 137, 218, 255),
    "role": (89, 102, 242, 255),
    "channel": (67, 181, 129, 255),
}


def strip_urls(text: str) -> Tuple[str, bool]:
    found = [False]

    def repl(match: re.Match[str]) -> str:
        found[0] = True
        return ""

    stripped = URL_RE.sub(repl, text)
    if not found[0]:
        return text, False
    normalized = " ".join(stripped.split())
    return normalized.strip(), True


def prepare_panel_mentions(
    message: discord.Message,
    text: str,
) -> Tuple[str, Dict[str, dict], bool]:
    if not text:
        return text, {}, False

    guild = message.guild
    if guild is None:
        return text, {}, False

    user_lookup = {member.id: member for member in getattr(message, "mentions", [])}
    role_lookup = {role.id: role for role in getattr(message, "role_mentions", [])}
    channel_lookup = {
        channel.id: channel for channel in getattr(message, "channel_mentions", [])
    }

    parts: list[str] = []
    last_index = 0
    mention_lookup: Dict[str, dict] = {}
    has_mentions = False

    for match in MENTION_TOKEN_RE.finditer(text):
        parts.append(text[last_index:match.start()])
        token = match.group(1)
        try:
            target_id = int(match.group(2))
        except (TypeError, ValueError):
            parts.append(match.group(0))
            last_index = match.end()
            continue

        mention_type = "user"
        display = f"@{target_id}"
        text_color = (255, 255, 255, 255)

        if token == "@&":
            mention_type = "role"
            role_obj = role_lookup.get(target_id) or guild.get_role(target_id)
            if role_obj:
                display = f"@{role_obj.name}"
        elif token == "#":
            mention_type = "channel"
            channel_obj = channel_lookup.get(target_id) or guild.get_channel(target_id)
            if channel_obj:
                display = f"#{channel_obj.name}"
            else:
                display = f"#{target_id}"
        else:
            member = user_lookup.get(target_id) or guild.get_member(target_id)
            display_name = member.display_name if member else str(target_id)
            display = f"@{display_name}"

        placeholder = f"\uFFF0{mention_type}:{target_id}\uFFF1"
        if placeholder not in mention_lookup:
            mention_lookup[placeholder] = {
                "type": mention_type,
                "id": target_id,
                "display": display,
                "bg_color": MENTION_COLORS.get(mention_type, MENTION_COLORS["user"]),
                "text_color": text_color,
            }
        parts.append(placeholder)
        last_index = match.end()
        has_mentions = True

    parts.append(text[last_index:])
    return "".join(parts), mention_lookup, has_mentions


def apply_mention_placeholders(text: str, mention_lookup: Mapping[str, dict]) -> str:
    if not text or not mention_lookup:
        return text

    def repl(match: re.Match[str]) -> str:
        placeholder = match.group(0)
        meta = mention_lookup.get(placeholder)
        if meta:
            return meta.get("display", "")
        mention_kind = match.group(1)
        identifier = match.group(2)
        if mention_kind == "channel":
            return f"#{identifier}"
        return f"@{identifier}"

    return MENTION_PLACEHOLDER_PATTERN.sub(repl, text)


def _load_vn_font(size: int, style: str = "regular"):
    try:
        from PIL import ImageFont
    except ImportError:
        return None

    style = (style or "regular").lower()
    style_sequence = {
        "regular": ["regular"],
        "bold": ["bold", "regular"],
        "italic": ["italic", "regular"],
        "bold_italic": ["bold_italic", "bold", "italic", "regular"],
    }.get(style, ["regular"])

    attempted = set()
    for style_key in style_sequence:
        candidate = _FONT_STYLE_PATHS.get(style_key)
        if not candidate or candidate in attempted:
            continue
        candidate = candidate.expanduser()
        attempted.add(candidate)
        if candidate.exists():
            try:
                logger.debug("VN sprite: loading font %s (size=%s style=%s)", candidate, size, style_key)
                return ImageFont.truetype(str(candidate), size=size)
            except OSError as exc:
                logger.warning("Failed to load VN font %s: %s", candidate, exc)
        else:
            logger.debug("VN sprite: font candidate missing -> %s", candidate)
    logger.warning("VN sprite: falling back to default font (size=%s style=%s)", size, style)
    return ImageFont.load_default()


@lru_cache(maxsize=64)
def _load_emoji_font(size: int):
    try:
        from PIL import ImageFont
    except ImportError:
        return None
    emoji_path = path_from_env("TFBOT_VN_EMOJI_FONT") or Path(
        os.getenv("TFBOT_VN_EMOJI_FONT", "fonts/NotoEmoji-VariableFont_wght.ttf")
    ).expanduser()
    if not emoji_path.exists():
        logger.debug("VN sprite: emoji font missing -> %s", emoji_path)
        return None
    try:
        return ImageFont.truetype(str(emoji_path), size=size)
    except OSError as exc:
        logger.warning("Failed to load emoji font %s: %s", emoji_path, exc)
        return None


def _is_emoji_char(ch: str) -> bool:
    code = ord(ch)
    return code >= 0x1F000 or 0x2600 <= code <= 0x27BF or 0x1F300 <= code <= 0x1FAFF


def parse_discord_formatting(text: str) -> Sequence[dict]:
    segments: list[dict] = []
    bold = italic = strike = False
    buffer: list[str] = []

    def emit_buffer() -> None:
        if buffer:
            segments.append(
                {
                    "text": "".join(buffer),
                    "bold": bold,
                    "italic": italic,
                    "strike": strike,
                    "emoji": False,
                }
            )
            buffer.clear()

    length = len(text)
    i = 0
    while i < length:
        ch = text[i]
        if text.startswith("**", i):
            emit_buffer()
            bold = not bold
            i += 2
            continue
        if text.startswith("~~", i):
            emit_buffer()
            strike = not strike
            i += 2
            continue
        if text.startswith("__", i):
            emit_buffer()
            italic = not italic
            i += 2
            continue
        if ch in ("*", "_"):
            emit_buffer()
            italic = not italic
            i += 1
            continue
        if ch == "\r":
            i += 1
            continue
        if ch == "\n":
            emit_buffer()
            segments.append(
                {
                    "text": "\n",
                    "bold": bold,
                    "italic": italic,
                    "strike": strike,
                    "emoji": False,
                    "newline": True,
                }
            )
            i += 1
            continue
        if ch == "<":
            match = CUSTOM_EMOJI_RE.match(text, i)
            if match:
                emit_buffer()
                animated = match.group(1) == "a"
                name = match.group(2)
                emoji_id = int(match.group(3))
                key = f"{emoji_id}{'a' if animated else ''}"
                segments.append(
                    {
                        "text": f":{name}:",
                        "bold": bold,
                        "italic": italic,
                        "strike": strike,
                        "emoji": False,
                        "custom_emoji": {
                            "name": name,
                            "id": emoji_id,
                            "animated": animated,
                            "key": key,
                        },
                    }
                )
                i = match.end()
                continue
        if _is_emoji_char(ch):
            emit_buffer()
            cluster_chars = [ch]
            i += 1
            while i < length:
                nxt = text[i]
                if nxt == "\r":
                    i += 1
                    continue
                if nxt == "\n":
                    break
                code = ord(nxt)
                if nxt == "\u200d":
                    cluster_chars.append(nxt)
                    i += 1
                    if i < length:
                        cluster_chars.append(text[i])
                    continue
                if 0x1F3FB <= code <= 0x1F3FF:
                    cluster_chars.append(nxt)
                    i += 1
                    continue
                break
            cluster = "".join(cluster_chars)
            segments.append(
                {
                    "text": cluster,
                    "bold": bold,
                    "italic": italic,
                    "strike": strike,
                    "emoji": True,
                }
            )
            continue
        buffer.append(ch)
        i += 1

    emit_buffer()
    return segments


def _select_font_for_segment(segment: Dict, base_font):
    size = getattr(base_font, "size", VN_TEXT_FONT_SIZE)
    style = "regular"
    if segment.get("bold") and segment.get("italic"):
        style = "bold_italic"
    elif segment.get("bold"):
        style = "bold"
    elif segment.get("italic"):
        style = "italic"

    font = _load_vn_font(size + (2 if segment.get("bold") else 0), style=style)
    if segment.get("emoji"):
        emoji_font = _load_emoji_font(getattr(font, "size", size))
        if emoji_font:
            font = emoji_font
    return font


def _fit_text_segments(draw, segments: Sequence[dict], starting_font, max_width: int, max_height: int):
    try:
        from PIL import ImageFont
    except ImportError:
        return [], starting_font

    start_size = int(getattr(starting_font, "size", VN_TEXT_FONT_SIZE))
    min_size = max(8, int(start_size * 0.5))
    min_size = min(start_size, min_size)

    chosen_lines: Sequence[Sequence[dict]] = []
    chosen_font = starting_font

    for size in range(start_size, min_size - 1, -1):
        base_font = _load_vn_font(size)
        lines = layout_formatted_text(draw, segments, base_font, max_width)
        total_height = _measure_layout_height(lines, getattr(base_font, "size", size))
        chosen_lines = lines
        chosen_font = base_font
        if total_height <= max_height:
            break
    return chosen_lines, chosen_font


def _font_line_height(font) -> int:
    try:
        bbox = font.getbbox("Ag")
        return max(1, bbox[3] - bbox[1])
    except Exception:  # pylint: disable=broad-except
        return getattr(font, "size", VN_TEXT_FONT_SIZE)


def prepare_reply_snippet(text: str, limit: int = 180) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "..."


def _wrap_plain_text(draw, text: str, font, max_width: int) -> Sequence[str]:
    if not text:
        return []
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        width = draw.textlength(candidate, font=font)
        if current and width > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _truncate_text_to_width(draw, text: str, font, max_width: int) -> str:
    if draw.textlength(text, font=font) <= max_width:
        return text
    ellipsis_width = draw.textlength("…", font=font)
    available = max_width - ellipsis_width
    if available <= 0:
        return "…"
    result = ""
    for ch in text:
        width = draw.textlength(result + ch, font=font)
        if width > available:
            break
        result += ch
    return result.rstrip() + "…"


def _tokenize_for_layout(text: str, is_emoji: bool) -> Sequence[str]:
    if not text:
        return []
    if is_emoji:
        return [text]
    tokens: list[str] = []
    idx = 0
    length = len(text)
    while idx < length:
        start = idx
        if text[idx].isspace():
            while idx < length and text[idx].isspace():
                idx += 1
        else:
            while idx < length and not text[idx].isspace():
                idx += 1
        tokens.append(text[start:idx])
    return tokens


def _split_token_to_width(token: str, font, max_width: int, draw) -> Sequence[str]:
    if not token:
        return []
    if max_width <= 0:
        return [token]
    pieces: list[str] = []
    current = ""
    for ch in token:
        trial = current + ch
        width = draw.textlength(trial, font=font)
        if current and width > max_width:
            pieces.append(current)
            current = ch
        elif not current and width > max_width:
            pieces.append(ch)
            current = ""
        else:
            current = trial
    if current:
        pieces.append(current)
    return pieces


def layout_formatted_text(draw, segments: Sequence[dict], base_font, max_width: int) -> Sequence[Sequence[dict]]:
    result: list[list[dict]] = []
    line: list[dict] = []
    current_x = 0
    baseline_height = getattr(base_font, "size", VN_TEXT_FONT_SIZE)

    def push_line(force_blank: bool = False) -> None:
        nonlocal line, current_x
        if line:
            result.append(line)
        elif force_blank:
            result.append([])
        line = []
        current_x = 0

    for segment in segments:
        if segment.get("newline"):
            push_line(force_blank=True)
            continue

        custom_meta = segment.get("custom_emoji")
        if custom_meta:
            emoji_size = getattr(base_font, "size", baseline_height)
            if current_x > 0 and current_x + emoji_size > max_width:
                push_line()
            line.append(
                {
                    "text": segment.get("text", ""),
                    "font": base_font,
                    "strike": False,
                    "bold": segment.get("bold"),
                    "italic": segment.get("italic"),
                    "emoji": False,
                    "custom_emoji": custom_meta,
                    "color": (240, 240, 240, 255),
                    "width": emoji_size,
                    "height": emoji_size,
                    "fallback_text": segment.get("text", ""),
                }
            )
            current_x += emoji_size
            continue

        text = segment.get("text", "")
        if not text:
            continue

        segment_font = _select_font_for_segment(segment, base_font)
        tokens = deque(_tokenize_for_layout(text, segment.get("emoji")))

        while tokens:
            token = tokens.popleft()
            if token == "":
                continue

            if token.isspace() and not line:
                continue

            width = draw.textlength(token, font=segment_font)
            if width > max_width and len(token) > 1 and not segment.get("emoji"):
                splits = _split_token_to_width(token, segment_font, max_width, draw)
                if len(splits) > 1:
                    for part in reversed(splits):
                        tokens.appendleft(part)
                    continue

            if current_x > 0 and current_x + width > max_width:
                push_line()
                if token.isspace():
                    continue
                width = draw.textlength(token, font=segment_font)

            if current_x == 0 and token.isspace():
                continue

            height = getattr(segment_font, "size", baseline_height)
            bbox = None
            try:
                if hasattr(segment_font, "getbbox"):
                    bbox = segment_font.getbbox(token)
                else:
                    bbox = draw.textbbox((0, 0), token, font=segment_font)
            except Exception:  # pylint: disable=broad-except
                bbox = None
            if bbox:
                height = max(height, bbox[3] - bbox[1])

            line.append(
                {
                    "text": token,
                    "font": segment_font,
                    "strike": segment.get("strike"),
                    "bold": segment.get("bold"),
                    "italic": segment.get("italic"),
                    "emoji": segment.get("emoji"),
                    "color": (240, 240, 240, 255),
                    "width": width,
                    "height": height,
                }
            )
            current_x += width

    push_line()
    return result


def _measure_layout_height(lines: Sequence[Sequence[dict]], base_line_height: int) -> int:
    total = 0
    line_spacing = 6
    for line in lines:
        if not line:
            total += base_line_height + line_spacing
            continue
        line_height = base_line_height
        for segment in line:
            seg_height = segment.get("height")
            if seg_height:
                line_height = max(line_height, seg_height)
        total += line_height + line_spacing
    if total > 0:
        total -= line_spacing
    return total


def _crop_transparent_top(image: "Image.Image") -> "Image.Image":
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    alpha = image.getchannel("A")
    bbox = alpha.getbbox()
    if not bbox:
        return image
    _, top, _, bottom = bbox
    height = bottom - top
    if top <= 0 or height <= 0:
        return image
    bottom = min(image.height, max(top + height, top + 1))
    cropped = image.crop((0, top, image.width, bottom))
    logger.debug("VN sprite: trimmed transparent top (%s -> %s)", image.size, cropped.size)
    return cropped


def _crop_transparent_left(image: "Image.Image") -> "Image.Image":
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    if image.width <= 1:
        return image
    alpha = image.getchannel("A")
    pixels = alpha.load()
    left = 0
    right = image.width
    while left < image.width:
        column = [pixels[left, y] for y in range(image.height)]
        if any(column):
            break
        left += 1
    if left >= right - 1:
        return image
    cropped = image.crop((left, 0, right, image.height))
    logger.debug("VN sprite: trimmed transparent left (%s -> %s)", image.size, cropped.size)
    return cropped


def _crop_transparent_vertical(image: "Image.Image") -> "Image.Image":
    cropped = _crop_transparent_top(image)
    cropped = _crop_transparent_left(cropped)
    return cropped


def render_vn_panel(
    *,
    state: TransformationState,
    message_content: str,
    character_display_name: str,
    original_name: str,
    attachment_id: Optional[str],
    formatted_segments: Sequence[dict],
    custom_emoji_images: Dict[str, "Image.Image"],
    reply_context: Optional[ReplyContext],
) -> Optional[discord.File]:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None

    if not VN_BASE_IMAGE.exists():
        logger.warning("VN panel: base image missing at %s", VN_BASE_IMAGE)
        return None

    try:
        base = Image.open(VN_BASE_IMAGE).convert("RGBA")
    except OSError as exc:
        logger.warning("VN panel: failed to open base image %s: %s", VN_BASE_IMAGE, exc)
        return None

    name_box = (180, 10, 193, 26)
    text_box = (188, 80, 765, 250)
    name_padding = 10
    text_padding = 12
    avatar_box = (0, 4, 220, 250)
    avatar_width = avatar_box[2] - avatar_box[0]
    avatar_height = avatar_box[3] - avatar_box[1]

    draw = ImageDraw.Draw(base)

    avatar_image = compose_game_avatar(state.character_name)
    if avatar_image is not None:
        cropped = _crop_transparent_vertical(avatar_image)
        base_scale = max(VN_AVATAR_SCALE, 0.01)
        if base_scale != 1.0:
            scaled_width = max(1, int(cropped.width * base_scale))
            scaled_height = max(1, int(cropped.height * base_scale))
            cropped = cropped.resize((scaled_width, scaled_height), Image.LANCZOS)

        fit_scale = min(
            avatar_width / cropped.width if cropped.width else 1.0,
            avatar_height / cropped.height if cropped.height else 1.0,
            1.0,
        )
        if fit_scale < 1.0:
            scaled_width = max(1, int(cropped.width * fit_scale))
            scaled_height = max(1, int(cropped.height * fit_scale))
            cropped = cropped.resize((scaled_width, scaled_height), Image.LANCZOS)

        canvas = Image.new("RGBA", (avatar_width, avatar_height), (0, 0, 0, 0))
        offset_x = max(0, (avatar_width - cropped.width) // 2)
        offset_y = max(0, (avatar_height - cropped.height) // 2)
        canvas.paste(cropped, (offset_x, offset_y), cropped)

        pos_x = avatar_box[0]
        pos_y = avatar_box[1]
        base.paste(canvas, (pos_x, pos_y), canvas)
        logger.debug(
            "VN sprite: pasted avatar for %s at (%s, %s) size %s (base_scale=%s fit_scale=%s)",
            state.character_name,
            pos_x,
            pos_y,
            canvas.size,
            base_scale,
            fit_scale,
        )
    else:
        logger.warning(
            "VN sprite: no avatar rendered for %s (mode=%s)",
            state.character_name,
            VN_AVATAR_MODE,
        )

    name_text = character_display_name
    name_x = name_box[0] + name_padding
    name_y = name_box[1] + name_padding
    name_font = _load_vn_font(VN_NAME_FONT_SIZE, style="bold")
    name_color = resolve_character_name_color(state.character_name)
    draw.text((name_x, name_y), name_text, fill=name_color, font=name_font)

    working_content = message_content.strip()
    if not working_content:
        working_content = f"{original_name} remains quietly transformed..."

    background_path = get_selected_background_path(state.user_id)
    background_layer = compose_background_layer((base.width, base.height), background_path)
    if background_layer:
        base = Image.alpha_composite(background_layer, base)
        draw = ImageDraw.Draw(base)

    base_text_font = _load_vn_font(VN_TEXT_FONT_SIZE)
    max_width = text_box[2] - text_box[0] - text_padding * 2
    total_height = text_box[3] - text_box[1] - text_padding * 2

    reply_font = None
    reply_line: Optional[str] = None
    reply_block_height = 0

    if reply_context and reply_context.text:
        reply_font = _load_vn_font(max(10, VN_TEXT_FONT_SIZE - 10))
        label_text = f"Replying to {reply_context.author}: "
        snippet_text = prepare_reply_snippet(reply_context.text).replace("\n", " ").strip()
        available_width = max_width - draw.textlength(label_text, font=reply_font)
        truncated_snippet = _truncate_text_to_width(draw, snippet_text, reply_font, available_width)
        reply_line = f"{label_text}{truncated_snippet}"
        reply_block_height = _font_line_height(reply_font) + 6
        logger.debug(
            "Rendering reply context for %s -> %s: %s",
            state.character_name,
            reply_context.author,
            snippet_text,
        )

    available_height = max(total_height - reply_block_height, _font_line_height(base_text_font) * 2)

    segments = list(formatted_segments) if formatted_segments else []
    if not segments:
        segments = list(parse_discord_formatting(working_content))
    lines, text_font = _fit_text_segments(draw, segments, base_text_font, max_width, available_height)
    text_y = text_box[1] + text_padding

    if reply_font and reply_line:
        reply_fill = (190, 190, 190, 255)
        text_x = text_box[0] + text_padding
        draw.text((text_x, text_y), reply_line, fill=reply_fill, font=reply_font)
        text_y += _font_line_height(reply_font) + 6

    base_line_height = getattr(text_font, "size", VN_TEXT_FONT_SIZE)
    for line in lines:
        if not line:
            text_y += base_line_height + 6
            continue
        text_x = text_box[0] + text_padding
        max_height = 0
        for segment in line:
            fill = segment.get("color", (240, 240, 240, 255))
            font_segment = segment["font"]
            text_segment = segment.get("text", "")
            width = segment.get("width", 0)
            height = segment.get("height") or getattr(font_segment, "size", base_line_height)
            custom_meta = segment.get("custom_emoji")
            if custom_meta:
                key = custom_meta.get("key")
                emoji_img = None
                if custom_emoji_images and key:
                    emoji_img = custom_emoji_images.get(key)
                if emoji_img is not None:
                    emoji_render = emoji_img.copy()
                    target_w = int(width) or base_line_height
                    target_h = int(height) or base_line_height
                    emoji_render.thumbnail((target_w, target_h), Image.LANCZOS)
                    offset_y = text_y + max(0, base_line_height - emoji_render.height)
                    base.paste(emoji_render, (int(text_x), int(offset_y)), emoji_render)
                else:
                    fallback = segment.get("fallback_text") or text_segment or custom_meta.get("name") or ""
                    if fallback:
                        draw.text((text_x, text_y), fallback, fill=fill, font=text_font)
                max_height = max(max_height, height)
                text_x += width
                continue
            if text_segment:
                draw.text((text_x, text_y), text_segment, fill=fill, font=font_segment)
                if segment.get("strike"):
                    strike_y = text_y + height / 2
                    draw.line(
                        (text_x, strike_y, text_x + width, strike_y),
                        fill=fill,
                        width=max(1, int(height / 10)),
                    )
            max_height = max(max_height, height)
            text_x += width
        text_y += max_height + 6
    output = io.BytesIO()
    base.save(output, format="PNG")
    output.seek(0)
    unique_fragment = attachment_id or str(int(utc_now().timestamp() * 1000))
    filename = f"tf-panel-{state.user_id}-{unique_fragment}.png"
    return discord.File(fp=output, filename=filename)


async def fetch_avatar_bytes(path_or_url: str) -> Optional[bytes]:
    if not path_or_url:
        return None

    if path_or_url.startswith(("http://", "https://")):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(path_or_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        logger.warning("Avatar fetch failed (%s): %s", resp.status, path_or_url)
                        return None
                    return await resp.read()
        except aiohttp.ClientError as exc:
            logger.warning("Avatar fetch error for %s: %s", path_or_url, exc)
            return None

    local_path = Path(path_or_url)
    if not local_path.is_absolute():
        local_path = BASE_DIR / local_path
    local_path = local_path.resolve()
    if not local_path.exists():
        logger.warning("Avatar file not found: %s", local_path)
        return None
    try:
        return local_path.read_bytes()
    except OSError as exc:
        logger.warning("Failed to read avatar file %s: %s", local_path, exc)
        return None


async def prepare_custom_emoji_images(
    message: discord.Message,
    segments: Sequence[dict],
) -> Dict[str, "Image.Image"]:
    try:
        from PIL import Image
    except ImportError:
        return {}

    needed: Dict[str, dict] = {}
    for segment in segments:
        meta = segment.get("custom_emoji")
        if not meta:
            continue
        key = meta.get("key")
        if key and key not in needed:
            needed[key] = meta
    if not needed:
        return {}

    cache_dir: Optional[Path] = None
    if VN_CACHE_DIR:
        cache_dir = VN_CACHE_DIR / "__emojis__"
        cache_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, "Image.Image"] = {}
    for key, meta in needed.items():
        emoji_id = meta.get("id")
        if emoji_id is None:
            continue
        cache_path = cache_dir / f"{emoji_id}.png" if cache_dir else None
        image_obj = None
        if cache_path and cache_path.exists():
            try:
                image_obj = Image.open(cache_path).convert("RGBA")
            except OSError as exc:
                logger.warning("VN emoji cache read failed (%s): %s", cache_path, exc)
                image_obj = None
        if image_obj is None:
            emoji_url = None
            if message.guild:
                emoji_obj = discord.utils.get(message.guild.emojis, id=int(emoji_id))
                if emoji_obj:
                    emoji_url = str(emoji_obj.url)
            if emoji_url is None:
                ext = "gif" if meta.get("animated") else "png"
                emoji_url = f"https://cdn.discordapp.com/emojis/{emoji_id}.{ext}?quality=lossless"
            data = await fetch_avatar_bytes(emoji_url)
            if not data:
                continue
            try:
                image_obj = Image.open(io.BytesIO(data))
                if getattr(image_obj, "is_animated", False):
                    image_obj.seek(0)
                image_obj = image_obj.convert("RGBA")
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("VN sprite: failed to decode emoji %s: %s", emoji_id, exc)
                continue
            if cache_path:
                try:
                    image_obj.save(cache_path, format="PNG")
                except OSError as exc:
                    logger.warning("VN sprite: unable to store emoji cache %s: %s", cache_path, exc)
        if image_obj:
            results[key] = image_obj
    return results


__all__ = [
    "VN_AVATAR_MODE",
    "VN_AVATAR_SCALE",
    "VN_BASE_IMAGE",
    "VN_NAME_DEFAULT_COLOR",
    "VN_BACKGROUND_ROOT",
    "vn_outfit_selection",
    "background_selections",
    "load_outfit_selections",
    "persist_outfit_selections",
    "load_background_selections",
    "persist_background_selections",
    "list_background_choices",
    "get_selected_background_path",
    "set_selected_background",
    "compose_background_layer",
    "compose_game_avatar",
    "resolve_character_directory",
    "resolve_character_name_color",
    "list_pose_outfits",
    "list_available_outfits",
    "get_selected_outfit_name",
    "get_selected_pose_outfit",
    "set_selected_outfit_name",
    "set_selected_pose_outfit",
    "strip_urls",
    "prepare_panel_mentions",
    "apply_mention_placeholders",
    "prepare_reply_snippet",
    "parse_discord_formatting",
    "layout_formatted_text",
    "render_vn_panel",
    "fetch_avatar_bytes",
    "prepare_custom_emoji_images",
]



