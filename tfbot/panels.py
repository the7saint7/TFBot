"""Visual novel panel rendering and helpers."""

from __future__ import annotations

import io
import json
import logging
import os
import queue
import random
import re
import shutil
import subprocess
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

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
_VN_ASSET_ROOT_SETTING = os.getenv("TFBOT_VN_ASSET_ROOT", "").strip()
if _VN_ASSET_ROOT_SETTING:
    candidate_asset_root = Path(_VN_ASSET_ROOT_SETTING).expanduser()
    if candidate_asset_root.exists():
        VN_ASSET_ROOT = candidate_asset_root.resolve()
    else:
        VN_ASSET_ROOT = None
        logger.warning("VN asset root %s does not exist.", candidate_asset_root)
elif VN_GAME_ROOT:
    candidate_asset_root = VN_GAME_ROOT / "game" / "images" / "characters"
    VN_ASSET_ROOT = candidate_asset_root if candidate_asset_root.exists() else None
else:
    VN_ASSET_ROOT = None
VN_DEFAULT_OUTFIT = os.getenv("TFBOT_VN_OUTFIT", "casual.png")
VN_DEFAULT_FACE = os.getenv("TFBOT_VN_FACE", "0.png")
VN_AVATAR_MODE = os.getenv("TFBOT_VN_AVATAR_MODE", "game").lower()
VN_AVATAR_SCALE = max(0.1, float_from_env("TFBOT_VN_AVATAR_SCALE", 1.0))
_VN_BG_ROOT_SETTING = os.getenv("TFBOT_VN_BG_ROOT", "").strip()
_VN_BG_DEFAULT_SETTING = os.getenv("TFBOT_VN_BG_DEFAULT", "school/cafeteria.png").strip()
VN_BACKGROUND_DEFAULT_RELATIVE = Path(_VN_BG_DEFAULT_SETTING) if _VN_BG_DEFAULT_SETTING else None

if _VN_BG_ROOT_SETTING:
    candidate_bg_root = Path(_VN_BG_ROOT_SETTING).expanduser()
    VN_BACKGROUND_ROOT = candidate_bg_root.resolve() if candidate_bg_root.exists() else None
elif VN_GAME_ROOT:
    candidate_bg_root = VN_GAME_ROOT / "game" / "images" / "bg"
    VN_BACKGROUND_ROOT = candidate_bg_root.resolve() if candidate_bg_root.exists() else None
else:
    VN_BACKGROUND_ROOT = None

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

# Face cache directory for pre-cached detected faces
# Must be in the characters_repo git repository
def _resolve_characters_repo_root() -> Optional[Path]:
    """
    Find the characters_repo git repository root.
    Returns None if characters_repo is not configured or not found.
    """
    repo_url = os.getenv("TFBOT_CHARACTERS_REPO", "").strip()
    if not repo_url:
        return None
    
    repo_dir_setting = os.getenv("TFBOT_CHARACTERS_REPO_DIR", "characters_repo").strip()
    repo_dir_setting = repo_dir_setting or "characters_repo"
    repo_dir = Path(repo_dir_setting)
    if not repo_dir.is_absolute():
        repo_dir = (BASE_DIR / repo_dir).resolve()
    
    # Verify it's a git repository
    repo_git_dir = repo_dir / ".git"
    if not repo_dir.exists() or not repo_git_dir.exists():
        return None
    
    return repo_dir

def _get_face_cache_dir() -> Optional[Path]:
    """
    Get the face cache directory in characters_repo.
    Returns None if characters_repo is not found (face detection should be aborted).
    """
    characters_repo_root = _resolve_characters_repo_root()
    if characters_repo_root:
        # Place faces folder at the same level as characters folder in git repo
        return characters_repo_root / "faces"
    return None


def _get_background_root() -> Optional[Path]:
    """Return the preferred background root, preferring characters_repo/bg when available."""
    git_repo_root = _resolve_git_repo_root()
    if git_repo_root:
        repo_bg_root = (git_repo_root / "bg").resolve()
        if repo_bg_root.exists():
            return repo_bg_root
    if VN_BACKGROUND_ROOT and VN_BACKGROUND_ROOT.exists():
        return VN_BACKGROUND_ROOT.resolve()
    return None


@lru_cache(maxsize=4)
def _compute_default_background_path(root_str: Optional[str]) -> Optional[Path]:
    if root_str is None or VN_BACKGROUND_DEFAULT_RELATIVE is None:
        return None
    root = Path(root_str)
    candidate = (root / VN_BACKGROUND_DEFAULT_RELATIVE).resolve()
    if candidate.exists():
        return candidate
    logger.warning(
        "VN background: default background %s does not exist under %s",
        VN_BACKGROUND_DEFAULT_RELATIVE,
        root,
    )
    return None


def _get_background_default_path() -> Optional[Path]:
    background_root = _get_background_root()
    root_str = str(background_root.resolve()) if background_root else None
    return _compute_default_background_path(root_str)


# Face cache directory is resolved dynamically via _get_face_cache_dir()

# Face detection margin (hardcoded, not from .env)
FACE_DETECTION_MARGIN = 0.2  # 20% margin around detected face

# Face detection model path
_FACE_MODEL_PATH_SETTING = os.getenv("TFBOT_FACE_MODEL_PATH", "model/ssd_anime_face_detect.pth").strip()
if _FACE_MODEL_PATH_SETTING:
    _face_model_path = Path(_FACE_MODEL_PATH_SETTING)
    if not _face_model_path.is_absolute():
        FACE_MODEL_PATH = (BASE_DIR / _face_model_path).resolve()
    else:
        FACE_MODEL_PATH = _face_model_path.resolve()
else:
    FACE_MODEL_PATH = (BASE_DIR / "model" / "ssd_anime_face_detect.pth").resolve()

_FACE_GIT_BATCH_WINDOW = max(0.0, float_from_env("TFBOT_FACE_GIT_BATCH_WINDOW", 1.0))
_FACE_GIT_WORKER: Optional[threading.Thread] = None
_FACE_GIT_WORKER_LOCK = threading.Lock()


@dataclass(frozen=True)
class FaceGitOperation:
    git_repo_root: Path
    cache_file: Path
    character_name: str
    variant_name: str


_FACE_GIT_QUEUE: "queue.Queue[FaceGitOperation]" = queue.Queue()


def _ensure_face_git_worker() -> None:
    """Start the git worker thread if it is not already running."""
    global _FACE_GIT_WORKER
    with _FACE_GIT_WORKER_LOCK:
        if _FACE_GIT_WORKER and _FACE_GIT_WORKER.is_alive():
            return
        _FACE_GIT_WORKER = threading.Thread(
            target=_face_git_worker_loop,
            name="face-git-worker",
            daemon=True,
        )
        _FACE_GIT_WORKER.start()


def _enqueue_face_git_operation(
    git_repo_root: Path,
    cache_file: Path,
    character_name: str,
    variant_name: str,
) -> None:
    """Queue a git add/commit/push operation for a cached face file."""
    if not git_repo_root.exists():
        return
    _ensure_face_git_worker()
    _FACE_GIT_QUEUE.put(
        FaceGitOperation(
            git_repo_root=git_repo_root,
            cache_file=cache_file,
            character_name=character_name,
            variant_name=variant_name,
        )
    )


def _face_git_worker_loop() -> None:
    """Worker loop that batches git operations to avoid concurrent pushes."""
    batch: List[FaceGitOperation] = []
    while True:
        operation = _FACE_GIT_QUEUE.get()
        batch.append(operation)
        if _FACE_GIT_BATCH_WINDOW > 0:
            while True:
                try:
                    next_op = _FACE_GIT_QUEUE.get(timeout=_FACE_GIT_BATCH_WINDOW)
                    batch.append(next_op)
                except queue.Empty:
                    break
        try:
            _process_face_git_batch(batch)
        finally:
            for _ in batch:
                _FACE_GIT_QUEUE.task_done()
            batch.clear()


def _process_face_git_batch(batch: Sequence[FaceGitOperation]) -> None:
    """Run git operations for all queued face caches."""
    if not batch:
        return
    git_executable = shutil.which("git")
    if not git_executable:
        logger.debug("git executable not found; skipping %d face git operations", len(batch))
        return

    ops_by_repo: Dict[Path, List[FaceGitOperation]] = {}
    for operation in batch:
        if not operation.git_repo_root.exists():
            logger.debug("characters repo missing for git operation: %s", operation.git_repo_root)
            continue
        ops_by_repo.setdefault(operation.git_repo_root, []).append(operation)

    for repo_root, operations in ops_by_repo.items():
        try:
            _run_face_git_batch_for_repo(git_executable, repo_root, operations)
        except subprocess.CalledProcessError as exc:
            logger.warning(
                "Git operation failed for face cache batch in %s: %s",
                repo_root,
                exc.stderr.strip() or exc.stdout.strip() or str(exc),
            )
        except Exception as exc:
            logger.warning("Unexpected error processing face git batch for %s: %s", repo_root, exc, exc_info=True)


def _run_face_git_batch_for_repo(
    git_executable: str,
    repo_root: Path,
    operations: Sequence[FaceGitOperation],
) -> None:
    """Stage, commit, and push the queued face cache changes for a repository."""
    staged: List[Tuple[FaceGitOperation, Path]] = []
    seen_paths: Set[Path] = set()
    for operation in operations:
        if not operation.cache_file.exists():
            logger.debug("Skipping missing face cache file %s", operation.cache_file)
            continue
        try:
            rel_path = operation.cache_file.relative_to(repo_root)
        except ValueError:
            logger.debug("Cache file %s not inside repo %s", operation.cache_file, repo_root)
            continue
        if rel_path in seen_paths:
            continue
        seen_paths.add(rel_path)
        staged.append((operation, rel_path))

    if not staged:
        return

    for _, rel_path in staged:
        cmd = [git_executable, "-C", str(repo_root), "add", str(rel_path)]
        subprocess.run(cmd, capture_output=True, text=True, check=True)

    if len(staged) == 1:
        op = staged[0][0]
        commit_msg = f"Add face cache for {op.character_name}/{op.variant_name}"
    else:
        descriptors = ", ".join(f"{op.character_name}/{op.variant_name}" for op, _ in staged[:5])
        if len(staged) > 5:
            descriptors = f"{descriptors} +{len(staged) - 5} more"
        commit_msg = f"Add face caches for {descriptors}"

    cmd = [git_executable, "-C", str(repo_root), "commit", "-m", commit_msg]
    subprocess.run(cmd, capture_output=True, text=True, check=True)

    cmd = [git_executable, "-C", str(repo_root), "push"]
    subprocess.run(cmd, capture_output=True, text=True, check=True)

    logger.info(
        "Pushed %d face cache change(s) to remote for %s",
        len(staged),
        repo_root,
    )


VN_SELECTION_FILE = Path(os.getenv("TFBOT_VN_SELECTIONS", "tf_outfits.json"))
_VN_LAYOUT_FILE_SETTING = os.getenv("TFBOT_VN_LAYOUTS", "vn_layouts.json").strip()
VN_LAYOUT_FILE = Path(_VN_LAYOUT_FILE_SETTING) if _VN_LAYOUT_FILE_SETTING else None

SPRITE_IMAGE_SUFFIXES: Tuple[str, ...] = (".png", ".webp")

_CHARACTER_FOLDER_OVERRIDES: Dict[str, str] = {}


def set_character_directory_overrides(mapping: Mapping[str, str]) -> None:
    _CHARACTER_FOLDER_OVERRIDES.clear()
    for key, value in mapping.items():
        normalized_key = (key or "").strip().lower()
        folder_name = (value or "").strip()
        if not normalized_key or not folder_name:
            continue
        _CHARACTER_FOLDER_OVERRIDES[normalized_key] = folder_name


def _is_supported_sprite(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SPRITE_IMAGE_SUFFIXES


def _strip_sprite_suffix(value: str) -> str:
    lower = value.lower()
    for suffix in SPRITE_IMAGE_SUFFIXES:
        if lower.endswith(suffix):
            return lower[: -len(suffix)]
    return lower


def _iter_sprite_files(directory: Path, recursive: bool = False):
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    for path in iterator:
        if _is_supported_sprite(path):
            yield path


def _gather_sprite_files(directory: Path, recursive: bool = False) -> list[Path]:
    return sorted(_iter_sprite_files(directory, recursive=recursive), key=lambda p: p.as_posix().lower())


def _find_named_sprite_file(directory: Path, stem: str) -> Optional[Path]:
    for suffix in SPRITE_IMAGE_SUFFIXES:
        candidate = directory / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _find_any_sprite_file(directory: Path, recursive: bool = False) -> Optional[Path]:
    files = _gather_sprite_files(directory, recursive=recursive)
    if files:
        return files[0]
    return None


def _match_outfit_asset(
    assets: Mapping[str, OutfitAsset],
    target_name: str,
) -> Optional[OutfitAsset]:
    normalized = target_name.strip().lower()
    if not normalized:
        return None
    lookup_keys = [normalized]
    stripped = _strip_sprite_suffix(normalized)
    if stripped and stripped not in lookup_keys:
        lookup_keys.append(stripped)
    for key in lookup_keys:
        asset = assets.get(key)
        if asset:
            return asset
    return None


def _normalize_layout_key(value: str) -> str:
    if not value:
        return ""
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _load_panel_layout_overrides() -> Dict[str, Dict]:
    if VN_LAYOUT_FILE is None or not VN_LAYOUT_FILE.exists():
        return {}
    try:
        data = json.loads(VN_LAYOUT_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("VN panel: failed to parse %s: %s", VN_LAYOUT_FILE, exc)
        return {}
    if not isinstance(data, dict):
        logger.warning("VN panel: layout file %s must contain an object.", VN_LAYOUT_FILE)
        return {}
    normalized: Dict[str, Dict] = {}
    for key, value in data.items():
        if not isinstance(value, dict):
            continue
        normalized_key = _normalize_layout_key(str(key))
        if normalized_key:
            normalized[normalized_key] = value
    return normalized


_PANEL_LAYOUT_OVERRIDES: Dict[str, Dict] = _load_panel_layout_overrides()


def resolve_panel_layout(character_name: str) -> Optional[Dict]:
    key = _normalize_layout_key(character_name)
    if not key:
        return None
    return _PANEL_LAYOUT_OVERRIDES.get(key)

vn_outfit_selection: Dict[str, Dict[str, object]] = {}
background_selections: Dict[str, str] = {}
_vn_config_cache: Dict[str, Dict] = {}
_VN_BACKGROUND_IMAGES: list[Path] = []
_VN_BACKGROUND_IMAGES_ROOT: Optional[Path] = None

@lru_cache(maxsize=1)
def _get_overlay_assets() -> Dict[str, "Image.Image"]:
    """Load optional overlay icons (rudy/frog/star/border)."""
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        return {}

    base_dir = VN_BASE_IMAGE.parent
    asset_files = {
        "rudy": "rudy.png",
        "frog": "frog.png",
        "star": "star.png",
        "border_gold": "border_gold.png",
        "border_epic": "border_epic.png",
        "border_common": "border_common.png",
        "border_ultra": "border_ultra.png",
    }
    assets: Dict[str, "Image.Image"] = {}
    for key, filename in asset_files.items():
        path = base_dir / filename
        if not path.exists():
            continue
        try:
            with Image.open(path) as img:
                assets[key] = img.convert("RGBA")
        except OSError as exc:
            logger.warning("VN panel: failed to load overlay asset %s: %s", path, exc)
    return assets


@lru_cache(maxsize=2)
def _character_directory_index(root: str) -> Dict[str, Path]:
    base = Path(root)
    if not base.exists():
        return {}
    try:
        return {
            child.name.lower(): child
            for child in base.iterdir()
            if child.is_dir()
        }
    except OSError as exc:
        logger.warning("VN sprite: failed to index character directories under %s: %s", base, exc)
        return {}


def load_outfit_selections() -> Dict[str, Dict[str, object]]:
    if not VN_SELECTION_FILE.exists():
        return {}
    try:
        data = json.loads(VN_SELECTION_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            normalized: Dict[str, Dict[str, object]] = {}
            for key, value in data.items():
                entry: Dict[str, object] = {}
                pose_value: Optional[str] = None
                outfit_value: Optional[str] = None
                accessories_value: Dict[str, str] = {}
                if isinstance(value, dict):
                    # Preserve all fields from the dict, including suppress_outfit_accessories
                    # First, copy all fields to preserve flags and other metadata
                    entry.update(value)
                    
                    pose_raw = value.get("pose")
                    outfit_raw = value.get("outfit") or value.get("name")
                    accessories_raw = value.get("accessories")
                    if isinstance(pose_raw, str):
                        pose_value = pose_raw.strip()
                    elif pose_raw is not None:
                        pose_value = str(pose_raw).strip()
                    if isinstance(outfit_raw, str):
                        outfit_value = outfit_raw.strip()
                    elif outfit_raw is not None:
                        outfit_value = str(outfit_raw).strip()
                    if isinstance(accessories_raw, Mapping):
                        # Rebuild accessories dict with normalized keys
                        accessories_value = {}
                        for acc_key, acc_value in accessories_raw.items():
                            normalized_key = str(acc_key).strip().lower()
                            if not normalized_key:
                                continue
                            normalized_value = str(acc_value).strip().lower()
                            if normalized_value == "on":
                                accessories_value[normalized_key] = "on"
                        # Update the entry with normalized accessories (or remove if empty)
                        if accessories_value:
                            entry["accessories"] = accessories_value
                        elif "accessories" in entry:
                            entry.pop("accessories")
                    
                    # Normalize pose and outfit values
                    if pose_value:
                        entry["pose"] = pose_value
                    elif "pose" in entry and not entry["pose"]:
                        entry.pop("pose")
                    if outfit_value:
                        entry["outfit"] = outfit_value
                    elif "outfit" not in entry and "name" in entry:
                        entry["outfit"] = entry["name"]
                elif isinstance(value, str):
                    outfit_value = value.strip()
                    if outfit_value:
                        entry["outfit"] = outfit_value
                elif value is not None:
                    outfit_value = str(value).strip()
                    if outfit_value:
                        entry["outfit"] = outfit_value

                # Only add entry if it has at least an outfit or other meaningful data (like suppress_outfit_accessories)
                # Preserve entries that have the suppress flag even if they don't have an outfit
                if entry and (outfit_value or len(entry) > 0):
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
    background_root = _get_background_root()
    if background_root is None:
        return None
    try:
        relative = path.resolve().relative_to(background_root.resolve())
    except ValueError:
        return None
    return relative.as_posix()


def _load_background_images() -> Sequence[Path]:
    global _VN_BACKGROUND_IMAGES, _VN_BACKGROUND_IMAGES_ROOT
    background_root = _get_background_root()
    if background_root and background_root.exists():
        if _VN_BACKGROUND_IMAGES and _VN_BACKGROUND_IMAGES_ROOT == background_root:
            return _VN_BACKGROUND_IMAGES
        try:
            _VN_BACKGROUND_IMAGES = sorted(
                path for path in background_root.rglob("*.png") if path.is_file()
            )
            _VN_BACKGROUND_IMAGES_ROOT = background_root
        except OSError as exc:
            logger.warning("VN background: failed to scan directory %s: %s", background_root, exc)
            _VN_BACKGROUND_IMAGES = []
            _VN_BACKGROUND_IMAGES_ROOT = None
    else:
        _VN_BACKGROUND_IMAGES = []
        _VN_BACKGROUND_IMAGES_ROOT = None
    return _VN_BACKGROUND_IMAGES


def list_background_choices() -> Sequence[Path]:
    return list(_load_background_images())


def get_selected_background_path(user_id: int) -> Optional[Path]:
    background_root = _get_background_root()
    if background_root is None:
        return None
    key = str(user_id)
    selected = background_selections.get(key)
    if selected:
        candidate = (background_root / selected).resolve()
        if candidate.exists():
            return candidate
        logger.warning("VN background: stored selection %s missing for user %s", selected, user_id)
        background_selections.pop(key, None)
        persist_background_selections()
    default_path = _get_background_default_path()
    if default_path and default_path.exists():
        return default_path
    backgrounds = _load_background_images()
    if backgrounds:
        return backgrounds[0]
    return None


def set_selected_background(user_id: int, background_path: Path) -> bool:
    background_root = _get_background_root()
    if background_root is None:
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
    sprite_files = _gather_sprite_files(accessory_dir, recursive=True)
    if not sprite_files:
        return None

    def _find_preferred(keyword: str) -> Optional[Path]:
        lowered = keyword.lower()
        for candidate in sprite_files:
            if candidate.stem.lower() == lowered:
                return candidate
        for candidate in sprite_files:
            if lowered in candidate.stem.lower():
                return candidate
        for candidate in sprite_files:
            parents = [parent.name.lower() for parent in candidate.parents]
            if lowered in parents:
                return candidate
        return None

    preferred = _find_preferred("off") or _find_preferred("on")
    if preferred:
        return preferred
    return sprite_files[0]


def _accessory_order_from_name(source_name: str) -> int:
    normalized_name = source_name.lower()
    order = 0
    if "-" in normalized_name:
        suffix = normalized_name.rsplit("-", 1)[-1]
        if suffix.lstrip("-").isdigit():
            try:
                order = int(suffix)
            except ValueError:
                order = 0
    return order


def _collect_accessory_layers(entry: Path, include_all: bool = False) -> list[Tuple[int, Path]]:
    accessories: list[Tuple[int, Path]] = []
    seen_paths: set[Path] = set()

    def _add_layer(path: Path, source_name: str) -> None:
        order = _accessory_order_from_name(source_name)
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
        elif _is_supported_sprite(child):
            _add_layer(child, child.stem)

    if not accessories and entry.is_dir() and (include_all or entry.name.lower().startswith("acc")):
        layer = _default_accessory_layer(entry)
        if layer:
            _add_layer(layer, entry.name)

    accessories.sort(key=lambda item: item[0])
    return accessories


@dataclass(frozen=True)
class _AccessoryDefinition:
    key: str
    label: str
    order: int
    on_layer: Optional[Path]
    off_layer: Optional[Path]


_ACCESSORY_CACHE: Dict[str, Dict[str, _AccessoryDefinition]] = {}


def _normalize_accessory_key(relative_path: Path) -> str:
    return relative_path.as_posix().lower()


def _format_accessory_label(relative_path: Path) -> str:
    parts: list[str] = []
    for part in relative_path.parts:
        cleaned = part
        if cleaned.lower().startswith("acc_"):
            cleaned = cleaned[4:]
        cleaned = cleaned.replace("_", " ").strip()
        if cleaned:
            parts.append(cleaned)
    if not parts:
        return relative_path.name
    return " / ".join(parts)


def _find_accessory_layer(accessory_dir: Path, target_state: str) -> Optional[Path]:
    sprite_files = _gather_sprite_files(accessory_dir, recursive=True)
    if not sprite_files:
        return None
    normalized = target_state.strip().lower()
    if not normalized:
        return None
    for candidate in sprite_files:
        if candidate.stem.lower() == normalized:
            return candidate
    for candidate in sprite_files:
        if normalized in candidate.stem.lower():
            return candidate
    for candidate in sprite_files:
        parents = [parent.name.lower() for parent in candidate.parents]
        if normalized in parents:
            return candidate
    return None


def _discover_variant_accessories(variant_dir: Path) -> Dict[str, _AccessoryDefinition]:
    cache_key = variant_dir.resolve().as_posix()
    cached = _ACCESSORY_CACHE.get(cache_key)
    if cached is not None:
        return cached
    outfits_dir = variant_dir / "outfits"
    if not outfits_dir.exists():
        _ACCESSORY_CACHE[cache_key] = {}
        return {}
    definitions: Dict[str, _AccessoryDefinition] = {}
    try:
        entries = sorted(outfits_dir.rglob("*"), key=lambda p: p.as_posix().lower())
    except OSError as exc:
        logger.warning("VN sprite: failed to scan accessories for %s: %s", variant_dir, exc)
        _ACCESSORY_CACHE[cache_key] = {}
        return {}
    for entry in entries:
        if not entry.is_dir():
            continue
        if not entry.name.lower().startswith("acc"):
            continue
        try:
            relative = entry.relative_to(outfits_dir)
        except ValueError:
            continue
        key = _normalize_accessory_key(relative)
        if key in definitions:
            continue
        label = _format_accessory_label(relative) or entry.name
        order = _accessory_order_from_name(entry.name)
        on_layer = _find_accessory_layer(entry, "on")
        off_layer = _find_accessory_layer(entry, "off")
        definitions[key] = _AccessoryDefinition(
            key=key,
            label=label,
            order=order,
            on_layer=on_layer,
            off_layer=off_layer,
        )
    _ACCESSORY_CACHE[cache_key] = definitions
    return definitions


def _discover_outfit_assets(variant_dir: Path) -> Dict[str, OutfitAsset]:
    assets: Dict[str, OutfitAsset] = {}
    outfits_dir = variant_dir / "outfits"
    if not outfits_dir.exists():
        return assets
    entries = sorted(outfits_dir.iterdir(), key=lambda p: p.name.lower())

    for entry in entries:
        if _is_supported_sprite(entry):
            name = entry.stem
            assets[name.lower()] = OutfitAsset(name=name, base_path=entry, accessory_layers=())

    for entry in entries:
        if entry.is_dir() and not entry.name.lower().startswith("acc"):
            primary = _find_named_sprite_file(entry, entry.name)
            if not primary:
                primary = _find_any_sprite_file(entry)
            if not primary:
                primary = _find_any_sprite_file(entry, recursive=True)
            if not primary:
                continue
            accessories = []
            accessories.extend(_collect_accessory_layers(entry, include_all=True))
            assets[entry.name.lower()] = OutfitAsset(
                name=entry.name,
                base_path=primary,
                accessory_layers=tuple(layer for _, layer in accessories),
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
    directory_index: Optional[Dict[str, Path]] = None
    normalized_name = character_name.strip().lower()
    override_folder = _CHARACTER_FOLDER_OVERRIDES.get(normalized_name)
    if override_folder:
        override_path = Path(override_folder)
        if not override_path.is_absolute():
            override_path = VN_ASSET_ROOT / override_path
        attempted.append(override_path.name)
        if override_path.exists():
            return override_path, attempted
        if directory_index is None:
            directory_index = _character_directory_index(str(VN_ASSET_ROOT))
        alternate = directory_index.get(override_folder.lower()) if directory_index else None
        if alternate is not None and alternate.exists():
            attempted.append(alternate.name)
            return alternate, attempted
    for key in _candidate_character_keys(character_name):
        candidate = VN_ASSET_ROOT / key
        attempted.append(candidate.name)
        if candidate.exists():
            return candidate, attempted
        if directory_index is None:
            directory_index = _character_directory_index(str(VN_ASSET_ROOT))
        alternate = directory_index.get(key.lower()) if directory_index else None
        if alternate is not None and alternate.exists():
            attempted.append(alternate.name)
            return alternate, attempted
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


def _list_accessories_for_dir(directory: Path) -> Dict[str, str]:
    config = _load_character_config(directory)
    variant_dirs = _ordered_variant_dirs(directory, config)
    if not variant_dirs:
        return {}
    accessories: Dict[str, str] = {}
    for variant_dir in variant_dirs:
        definitions = _discover_variant_accessories(variant_dir)
        for key, definition in definitions.items():
            if key not in accessories:
                accessories[key] = definition.label or definition.key
    return accessories


def list_character_accessories(character_name: str) -> Dict[str, str]:
    directory, _ = resolve_character_directory(character_name)
    if directory is None:
        return {}
    return _list_accessories_for_dir(directory)


def _stored_accessory_states_for_dir(
    directory: Path,
    scope: Optional[str],
) -> Dict[str, str]:
    for lookup_key in _selection_lookup_keys(directory, scope):
        entry = vn_outfit_selection.get(lookup_key)
        if isinstance(entry, dict):
            raw_accessories = entry.get("accessories")
            if isinstance(raw_accessories, Mapping):
                normalized: Dict[str, str] = {}
                for key, value in raw_accessories.items():
                    normalized_key = str(key).strip().lower()
                    if not normalized_key:
                        continue
                    normalized_value = str(value).strip().lower()
                    if normalized_value == "on":
                        normalized[normalized_key] = "on"
                return normalized
    return {}


def get_accessory_states_for_dir(
    directory: Path,
    scope: Optional[str] = None,
) -> Dict[str, str]:
    accessories = _list_accessories_for_dir(directory)
    if not accessories:
        return {}
    stored_states = _stored_accessory_states_for_dir(directory, scope)
    states: Dict[str, str] = {}
    for key in accessories.keys():
        states[key] = stored_states.get(key, "off")
    return states


def get_accessory_states(
    character_name: str,
    scope: Optional[str] = None,
) -> Dict[str, str]:
    directory, _ = resolve_character_directory(character_name)
    if directory is None:
        return {}
    return get_accessory_states_for_dir(directory, scope=scope)


def _set_accessory_state_for_dir(
    directory: Path,
    accessory_key: str,
    enabled: bool,
    *,
    scope: Optional[str],
) -> bool:
    normalized_key = accessory_key.strip().lower()
    if not normalized_key:
        return False
    store_key = _selection_store_key(directory, scope)
    entry = vn_outfit_selection.get(store_key)
    if isinstance(entry, dict):
        target_entry = entry
    else:
        target_entry = {}
        if isinstance(entry, str):
            stripped = entry.strip()
            if stripped:
                target_entry["outfit"] = stripped
        elif entry is not None:
            stripped = str(entry).strip()
            if stripped:
                target_entry["outfit"] = stripped
        vn_outfit_selection[store_key] = target_entry
    accessories = target_entry.get("accessories")
    if not isinstance(accessories, dict):
        accessories = {}
        target_entry["accessories"] = accessories
    if enabled:
        accessories[normalized_key] = "on"
    else:
        accessories.pop(normalized_key, None)
        if not accessories:
            target_entry.pop("accessories", None)
    persist_outfit_selections()
    compose_game_avatar.cache_clear()
    logger.info(
        "VN sprite: accessory %s for %s scope %s set to %s",
        normalized_key,
        directory.name,
        _normalize_selection_scope(scope),
        "on" if enabled else "off",
    )
    return True


def set_accessory_state(
    character_name: str,
    accessory_key: str,
    enabled: bool,
    *,
    scope: Optional[str] = None,
) -> bool:
    directory, attempted = resolve_character_directory(character_name)
    if directory is None:
        logger.debug("VN sprite: cannot set accessory for %s (tried %s)", character_name, attempted)
        return False
    accessories = _list_accessories_for_dir(directory)
    normalized_key = accessory_key.strip().lower()
    valid_key = None
    if normalized_key in accessories:
        valid_key = normalized_key
    else:
        for key in accessories.keys():
            if key.lower() == normalized_key:
                valid_key = key
                break
    if valid_key is None:
        return False
    return _set_accessory_state_for_dir(directory, valid_key, enabled, scope=scope)


def toggle_accessory_state(
    character_name: str,
    accessory_key: str,
    *,
    scope: Optional[str] = None,
) -> Optional[str]:
    directory, attempted = resolve_character_directory(character_name)
    if directory is None:
        logger.debug("VN sprite: cannot toggle accessory for %s (tried %s)", character_name, attempted)
        return None
    accessories = _list_accessories_for_dir(directory)
    if not accessories:
        return None
    normalized_key = accessory_key.strip().lower()
    canonical_key = None
    if normalized_key in accessories:
        canonical_key = normalized_key
    else:
        for key in accessories.keys():
            if key.lower() == normalized_key:
                canonical_key = key
                break
    if canonical_key is None:
        return None
    states = get_accessory_states_for_dir(directory, scope=scope)
    current_state = states.get(canonical_key, "off")
    enabled = current_state != "on"
    if not _set_accessory_state_for_dir(directory, canonical_key, enabled, scope=scope):
        return None
    return "on" if enabled else "off"


def _resolve_optional_accessory_layers(
    character_dir: Path,
    variant_dir: Path,
    scope: Optional[str],
) -> list[Path]:
    states = get_accessory_states_for_dir(character_dir, scope=scope)
    if not states:
        return []
    variant_accessories = _discover_variant_accessories(variant_dir)
    if not variant_accessories:
        return []
    layers: list[Tuple[int, Path]] = []
    for key, state in states.items():
        definition = variant_accessories.get(key)
        if not definition:
            continue
        target_layer: Optional[Path]
        if state == "on":
            target_layer = definition.on_layer or definition.off_layer
        else:
            target_layer = definition.off_layer
        if target_layer:
            layers.append((definition.order, target_layer))
    layers.sort(key=lambda item: item[0])
    return [path for _, path in layers]


def _normalize_selection_scope(scope: Optional[str]) -> str:
    if scope is None:
        return "default"
    normalized = scope.strip().lower()
    return normalized or "default"


def _selection_lookup_keys(directory: Path, scope: Optional[str]) -> list[str]:
    base = directory.name.lower()
    normalized = _normalize_selection_scope(scope)
    keys: list[str] = []
    if normalized != "default":
        keys.append(f"{normalized}:{base}")
    keys.append(base)
    return keys


def _selection_store_key(directory: Path, scope: Optional[str]) -> str:
    base = directory.name.lower()
    normalized = _normalize_selection_scope(scope)
    if normalized == "default":
        return base
    return f"{normalized}:{base}"


def get_selected_outfit_for_dir(directory: Path, scope: Optional[str] = None) -> Optional[str]:
    _, outfit = get_selected_pose_outfit_for_dir(directory, scope=scope)
    return outfit


def get_selected_pose_outfit_for_dir(
    directory: Path,
    scope: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    entry = None
    for key in _selection_lookup_keys(directory, scope):
        entry = vn_outfit_selection.get(key)
        if entry:
            break
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


def get_selected_outfit_name(character_name: str, scope: Optional[str] = None) -> Optional[str]:
    _, outfit = get_selected_pose_outfit(character_name, scope=scope)
    return outfit


def get_selected_pose_outfit(
    character_name: str,
    scope: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    directory, _ = resolve_character_directory(character_name)
    if directory is None:
        return None, None
    return get_selected_pose_outfit_for_dir(directory, scope=scope)


def set_selected_outfit_name(
    character_name: str,
    outfit_name: str,
    scope: Optional[str] = None,
) -> bool:
    return set_selected_pose_outfit(character_name, None, outfit_name, scope=scope)


def set_selected_pose_outfit(
    character_name: str,
    pose_name: Optional[str],
    outfit_name: str,
    *,
    scope: Optional[str] = None,
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

    store_key = _selection_store_key(directory, scope)
    existing_entry = vn_outfit_selection.get(store_key)
    accessory_entry: Optional[Dict[str, str]] = None
    if isinstance(existing_entry, dict):
        existing_accessories = existing_entry.get("accessories")
        if isinstance(existing_accessories, Mapping):
            accessory_entry = {
                str(key).strip().lower(): "on"
                for key, value in existing_accessories.items()
                if str(value).strip().lower() == "on"
            }
    new_entry: Dict[str, object] = {"pose": matched_pose, "outfit": matched_outfit}
    if accessory_entry:
        new_entry["accessories"] = accessory_entry
    vn_outfit_selection[store_key] = new_entry
    persist_outfit_selections()
    compose_game_avatar.cache_clear()
    logger.info(
        "VN sprite: outfit override for %s set to pose %s outfit %s",
        directory.name,
        matched_pose,
        matched_outfit,
    )
    return True


_COMPOSE_AVATAR_CACHE: "OrderedDict[Tuple[str, Optional[str], Optional[str], str], 'Image.Image']" = OrderedDict()
_COMPOSE_AVATAR_CACHE_LIMIT = 512


def compose_game_avatar(
    character_name: str,
    pose_override: Optional[str] = None,
    outfit_override: Optional[str] = None,
    selection_scope: Optional[str] = None,
) -> Optional["Image.Image"]:
    scope_key = _normalize_selection_scope(selection_scope)
    cache_key = (character_name, pose_override, outfit_override, scope_key)
    cached = _COMPOSE_AVATAR_CACHE.get(cache_key)
    if cached is not None:
        _COMPOSE_AVATAR_CACHE.move_to_end(cache_key)
        return cached
    image = _compose_game_avatar_uncached(
        character_name,
        pose_override,
        outfit_override,
        selection_scope=scope_key,
    )
    if image is not None:
        _COMPOSE_AVATAR_CACHE[cache_key] = image
        _COMPOSE_AVATAR_CACHE.move_to_end(cache_key)
        if len(_COMPOSE_AVATAR_CACHE) > _COMPOSE_AVATAR_CACHE_LIMIT:
            _COMPOSE_AVATAR_CACHE.popitem(last=False)
    return image


def _check_face_exists_in_remote(
    git_repo_root: Path,
    face_relative_path: Path,
) -> bool:
    """
    Check if a face file exists in the remote git repository.
    
    Args:
        git_repo_root: Git repository root directory
        face_relative_path: Relative path from repo root (e.g., faces/character/variant/face.png)
        
    Returns:
        True if file exists in remote, False otherwise
    """
    git_executable = shutil.which("git")
    if not git_executable:
        return False
    
    try:
        # Fetch latest refs from remote (lightweight operation)
        fetch_cmd = [git_executable, "-C", str(git_repo_root), "fetch", "--quiet"]
        subprocess.run(fetch_cmd, capture_output=True, check=False, timeout=30)
        
        # Check if file exists in remote branch (default to origin/main or origin/master)
        # Try common branch names
        for branch in ["origin/main", "origin/master", "origin/HEAD"]:
            ls_cmd = [
                git_executable,
                "-C",
                str(git_repo_root),
                "ls-tree",
                "--name-only",
                branch,
                str(face_relative_path),
            ]
            result = subprocess.run(ls_cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                return True
        
        return False
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("Failed to check remote for face %s: %s", face_relative_path, exc)
        return False


def _sync_faces_from_remote(git_repo_root: Path) -> None:
    """
    Pull latest changes from remote for the faces directory.
    
    Args:
        git_repo_root: Git repository root directory
    """
    git_executable = shutil.which("git")
    if not git_executable:
        return
    
    try:
        # Pull latest changes (only fast-forward)
        pull_cmd = [git_executable, "-C", str(git_repo_root), "pull", "--ff-only", "--quiet"]
        result = subprocess.run(pull_cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            logger.debug("Synced faces from remote")
        else:
            logger.debug("Face sync pull returned non-zero: %s", result.stderr.strip())
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("Failed to sync faces from remote: %s", exc)


def _cache_character_face_background(
    character_dir: Path,
    variant_dir: Path,
    avatar_image: "Image.Image",
    git_repo_root: Optional[Path],
) -> None:
    """
    Background thread function to cache face and commit to git.
    
    Args:
        character_dir: Character directory path
        variant_dir: Variant directory path
        avatar_image: Fully composed avatar PIL Image (copy for thread safety)
        git_repo_root: Git repository root directory (characters_repo), or None if not found
    """
    if not FACE_MODEL_PATH.exists():
        return
    
    face_cache_dir = _get_face_cache_dir()
    if face_cache_dir is None:
        logger.warning(
            "Face detection background task aborted: characters_repo not found for %s/%s",
            character_dir.name,
            variant_dir.name,
        )
        return
    
    try:
        from PIL import Image
        
        # Create cache path: faces/character_name/variant_name/face.png
        cache_dir = face_cache_dir / character_dir.name.lower() / variant_dir.name.lower()
        cache_file = cache_dir / "face.png"
        
        # Check if already cached locally
        if cache_file.exists():
            logger.debug("Face already cached locally for %s/%s", character_dir.name, variant_dir.name)
            return
        
        # Check if exists in remote git repository
        if git_repo_root and git_repo_root.exists():
            face_relative_path = cache_file.relative_to(git_repo_root)
            if _check_face_exists_in_remote(git_repo_root, face_relative_path):
                logger.debug("Face exists in remote for %s/%s, pulling...", character_dir.name, variant_dir.name)
                # Pull the file from remote
                _sync_faces_from_remote(git_repo_root)
                # Check again after pull
                if cache_file.exists():
                    logger.debug("Face pulled from remote for %s/%s", character_dir.name, variant_dir.name)
                    return
        
        # Import face detection
        from tfbot.face_detection import detect_faces_in_pil_image
        
        # Detect faces in the avatar image
        faces = detect_faces_in_pil_image(avatar_image, FACE_MODEL_PATH)
        
        if faces is None or len(faces) == 0:
            logger.debug("No face detected for %s/%s", character_dir.name, variant_dir.name)
            return
        
        # Get the first (highest confidence) face
        face = faces[0]
        xmin = int(face[0])
        ymin = int(face[1])
        xmax = int(face[2])
        ymax = int(face[3])
        score = face[4]
        
        logger.debug(
            "Face detected for %s/%s: bbox=(%d,%d,%d,%d) score=%.3f",
            character_dir.name,
            variant_dir.name,
            xmin,
            ymin,
            xmax,
            ymax,
            score,
        )
        
        # Calculate margin
        face_width = xmax - xmin
        face_height = ymax - ymin
        margin_x = int(face_width * FACE_DETECTION_MARGIN)
        margin_y = int(face_height * FACE_DETECTION_MARGIN)
        
        # Expand bounding box with margin, clamped to image bounds
        img_width, img_height = avatar_image.size
        crop_xmin = max(0, xmin - margin_x)
        crop_ymin = max(0, ymin - margin_y)
        crop_xmax = min(img_width, xmax + margin_x)
        crop_ymax = min(img_height, ymax + margin_y)
        
        # Crop face region with margin
        face_crop = avatar_image.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax))
        
        # Resize to max 250x250 if larger, maintaining aspect ratio
        MAX_FACE_SIZE = 250
        face_width, face_height = face_crop.size
        if face_width > MAX_FACE_SIZE or face_height > MAX_FACE_SIZE:
            # Calculate new size maintaining aspect ratio
            if face_width > face_height:
                new_width = MAX_FACE_SIZE
                new_height = int(face_height * (MAX_FACE_SIZE / face_width))
            else:
                new_height = MAX_FACE_SIZE
                new_width = int(face_width * (MAX_FACE_SIZE / face_height))
            face_crop = face_crop.resize((new_width, new_height), Image.LANCZOS)
            logger.debug(
                "Resized face for %s/%s from %dx%d to %dx%d",
                character_dir.name,
                variant_dir.name,
                face_width,
                face_height,
                new_width,
                new_height,
            )
        
        # Save cached face
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            face_crop.save(cache_file, format="PNG")
            logger.info("Cached face for %s/%s to %s", character_dir.name, variant_dir.name, cache_file)
        except OSError as exc:
            logger.warning("Failed to cache face for %s/%s: %s", character_dir.name, variant_dir.name, exc)
            return
        
        # Git operations if in git repo (queued to avoid concurrent pushes)
        if git_repo_root and git_repo_root.exists():
            _enqueue_face_git_operation(
                git_repo_root=git_repo_root,
                cache_file=cache_file,
                character_name=character_dir.name,
                variant_name=variant_dir.name,
            )

    except ImportError:
        logger.debug("Face detection not available (missing dependencies)")
    except Exception as exc:
        logger.warning("Error caching face for %s/%s: %s", character_dir.name, variant_dir.name, exc, exc_info=True)


# Periodic sync state
_last_face_sync_time: float = 0.0
_face_sync_interval: float = 86400.0  # 24 hours in seconds
_face_sync_lock = threading.Lock()


def _periodic_face_sync() -> None:
    """Periodic background task to sync faces from remote git repository."""
    global _last_face_sync_time
    
    git_repo_root = _resolve_characters_repo_root()
    if not git_repo_root or not git_repo_root.exists():
        return
    
    current_time = time.time()
    with _face_sync_lock:
        # Check if enough time has passed since last sync
        if current_time - _last_face_sync_time < _face_sync_interval:
            return
        _last_face_sync_time = current_time
    
    logger.debug("Running periodic face sync from remote")
    _sync_faces_from_remote(git_repo_root)


def _cache_character_face(
    character_dir: Path,
    variant_dir: Path,
    avatar_image: "Image.Image",
) -> None:
    """
    Cache the detected face from a character's avatar image in a background thread.
    Checks if face is already cached locally and remotely, and if not, launches background thread to detect, save, and commit to git.
    
    Args:
        character_dir: Character directory path
        variant_dir: Variant directory path
        avatar_image: Fully composed avatar PIL Image
    """
    if not FACE_MODEL_PATH.exists():
        return
    
    # Check if characters_repo is available - abort if not
    face_cache_dir = _get_face_cache_dir()
    if face_cache_dir is None:
        logger.warning(
            "Face detection aborted: characters_repo git repository not found. "
            "Set TFBOT_CHARACTERS_REPO and ensure characters_repo directory exists with .git folder. "
            "Face caching requires the characters_repo to be configured and synced."
        )
        return
    
    git_repo_root = _resolve_characters_repo_root()
    
    # Create cache path: faces/character_name/variant_name/face.png
    cache_dir = face_cache_dir / character_dir.name.lower() / variant_dir.name.lower()
    cache_file = cache_dir / "face.png"
    
    # Check if already cached locally
    if cache_file.exists():
        logger.debug("Face already cached locally for %s/%s", character_dir.name, variant_dir.name)
        return
    
    # Check remote if in git repo (quick check before launching thread)
    if git_repo_root and git_repo_root.exists():
        face_relative_path = cache_file.relative_to(git_repo_root)
        if _check_face_exists_in_remote(git_repo_root, face_relative_path):
            logger.debug("Face exists in remote for %s/%s, syncing...", character_dir.name, variant_dir.name)
            # Sync from remote in background
            sync_thread = threading.Thread(
                target=_sync_faces_from_remote,
                args=(git_repo_root,),
                daemon=True,
                name=f"face-sync-{character_dir.name}-{variant_dir.name}",
            )
            sync_thread.start()
            return
    
    # Trigger periodic sync check (non-blocking)
    try:
        sync_check_thread = threading.Thread(
            target=_periodic_face_sync,
            daemon=True,
            name="face-sync-check",
        )
        sync_check_thread.start()
    except Exception:
        pass  # Ignore errors in sync check
    
    # Make a copy of the image for thread safety
    try:
        from PIL import Image
        avatar_copy = avatar_image.copy()
    except Exception:
        logger.warning("Failed to copy avatar image for face caching")
        return
    
    # Launch background thread to detect and cache face
    thread = threading.Thread(
        target=_cache_character_face_background,
        args=(character_dir, variant_dir, avatar_copy, git_repo_root),
        daemon=True,
        name=f"face-cache-{character_dir.name}-{variant_dir.name}",
    )
    thread.start()
    logger.debug("Launched background thread to cache face for %s/%s", character_dir.name, variant_dir.name)


def _compose_game_avatar_uncached(
    character_name: str,
    pose_override: Optional[str] = None,
    outfit_override: Optional[str] = None,
    *,
    selection_scope: Optional[str] = None,
) -> Optional["Image.Image"]:
    layout = resolve_panel_layout(character_name)
    if layout and layout.get("disable_avatar"):
        return None
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

    preferred_pose, preferred_outfit = get_selected_pose_outfit_for_dir(
        character_dir,
        scope=selection_scope,
    )
    if pose_override:
        cleaned_pose = pose_override.strip()
        if cleaned_pose:
            preferred_pose = cleaned_pose
    if outfit_override:
        cleaned_outfit = outfit_override.strip()
        if cleaned_outfit:
            preferred_outfit = cleaned_outfit
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

    pose_metadata = _get_pose_metadata(config, variant_dir.name)
    sprite_height_limit = _resolve_image_height(config, pose_metadata)

    outfit_path = outfit_asset.base_path
    face_path = _select_face_path(variant_dir)

    optional_layers = _resolve_optional_accessory_layers(
        character_dir,
        variant_dir,
        selection_scope,
    )
    
    # Check if outfit-level accessories should be suppressed (from clearall command)
    suppress_outfit_accessories = False
    for lookup_key in _selection_lookup_keys(character_dir, selection_scope):
        entry = vn_outfit_selection.get(lookup_key)
        if isinstance(entry, dict) and entry.get("suppress_outfit_accessories"):
            suppress_outfit_accessories = True
            break
    
    combined_accessory_layers = []
    if not suppress_outfit_accessories:
        combined_accessory_layers = list(outfit_asset.accessory_layers)
    combined_accessory_layers.extend(optional_layers)

    cache_file: Optional[Path] = None
    if VN_CACHE_DIR:
        face_token = "noface"
        if face_path and face_path.exists():
            face_token = face_path.stem.lower()
        accessory_token = "noacc"
        if combined_accessory_layers:
            accessory_token = "-".join(layer.stem.lower() for layer in combined_accessory_layers)
        height_token = f"h{sprite_height_limit}" if sprite_height_limit else "auto"
        cache_dir = VN_CACHE_DIR / character_dir.name.lower() / variant_dir.name.lower()
        cache_file = cache_dir / f"{outfit_path.stem.lower()}__{face_token}__{accessory_token}__{height_token}.png"
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

    for layer_path in combined_accessory_layers:
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
    if sprite_height_limit:
        limit = min(sprite_height_limit, outfit_image.height)
        if limit > 0 and limit < outfit_image.height:
            outfit_image = outfit_image.crop((0, 0, outfit_image.width, limit))

    # Cache detected face if not already cached
    _cache_character_face(character_dir, variant_dir, outfit_image)

    if VN_CACHE_DIR and cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            outfit_image.save(cache_file, format="PNG")
            logger.debug("VN sprite: cached avatar %s", cache_file)
        except OSError as exc:
            logger.warning("VN sprite: unable to cache avatar %s: %s", cache_file, exc)

    return outfit_image


def _clear_compose_game_avatar_cache() -> None:
    _COMPOSE_AVATAR_CACHE.clear()
    # Also clear disk cache to force regeneration when accessories change
    if VN_CACHE_DIR and VN_CACHE_DIR.exists():
        try:
            import shutil
            for cache_dir in VN_CACHE_DIR.iterdir():
                if cache_dir.is_dir():
                    shutil.rmtree(cache_dir, ignore_errors=True)
            logger.debug("VN sprite: cleared disk cache directory %s", VN_CACHE_DIR)
        except Exception as exc:
            logger.warning("VN sprite: failed to clear disk cache %s: %s", VN_CACHE_DIR, exc)


compose_game_avatar.cache_clear = _clear_compose_game_avatar_cache  # type: ignore[attr-defined]


def _get_pose_metadata(config: Dict, pose_name: str) -> Dict:
    poses = config.get("poses")
    if isinstance(poses, dict):
        value = poses.get(pose_name.lower())
        if isinstance(value, dict):
            return value
    return {}


def _resolve_image_height(config: Dict, pose_metadata: Dict) -> Optional[int]:
    raw_value = pose_metadata.get("image_height")
    if raw_value is None:
        raw_value = config.get("image_height")
    if raw_value is None:
        return None
    try:
        value = int(str(raw_value).strip())
    except (ValueError, TypeError, AttributeError):
        return None
    return value if value > 0 else None


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
        cleaned = name.strip().lower()
        if cleaned and cleaned not in normalized_targets:
            normalized_targets.append(cleaned)

    for target in normalized_targets:
        for variant_dir in search_variants:
            assets = variant_outfits.get(variant_dir, {})
            asset = _match_outfit_asset(assets, target)
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
        (face_path for face_path in selected_group.iterdir() if _is_supported_sprite(face_path)),
        key=lambda p: p.name.lower(),
    )
    if not faces:
        logger.warning("VN sprite: no faces found in group %s", selected_group)
        return None
    preferred_face = (VN_DEFAULT_FACE or "").strip().lower()
    preferred_face_stem = _strip_sprite_suffix(preferred_face) if preferred_face else ""
    for face in faces:
        if face.name.lower() == preferred_face:
            return face
    if preferred_face_stem:
        for face in faces:
            if face.stem.lower() == preferred_face_stem:
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


def _is_single_emoji_message(segments: Sequence[dict]) -> bool:
    """Return True if the formatted segments equate to a lone emoji/custom emoji."""
    emoji_segment_found = False
    for segment in segments:
        if segment.get("newline"):
            continue
        is_emoji = bool(segment.get("emoji") or segment.get("custom_emoji"))
        text = (segment.get("text") or "").strip()
        if not text and not is_emoji:
            continue
        if is_emoji and not emoji_segment_found:
            emoji_segment_found = True
            continue
        return False
    return emoji_segment_found


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
    return image.crop((0, top, image.width, bottom))


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
    return image.crop((left, 0, right, image.height))


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
    selection_scope: Optional[str] = None,
    gacha_star_count: Optional[int] = None,
    gacha_rudy: Optional[int] = None,
    gacha_frog: Optional[int] = None,
    gacha_outfit_override: Optional[str] = None,
    gacha_pose_override: Optional[str] = None,
    gacha_border: Optional[str] = None,
) -> Optional[discord.File]:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None

    layout = resolve_panel_layout(state.character_name) or {}

    base_image_path = VN_BASE_IMAGE
    base_override = layout.get("base_image")
    if isinstance(base_override, str) and base_override.strip():
        candidate = Path(base_override.strip())
        if not candidate.is_absolute():
            from_base = (BASE_DIR / candidate).resolve()
            from_assets = (VN_BASE_IMAGE.parent / candidate).resolve()
            candidate = from_base if from_base.exists() else from_assets
        if candidate.exists():
            base_image_path = candidate
        else:
            logger.warning(
                "VN panel: custom base image %s missing for %s",
                candidate,
                state.character_name,
            )

    if not base_image_path.exists():
        logger.warning("VN panel: base image missing at %s", base_image_path)
        return None

    try:
        with Image.open(base_image_path) as base_image:
            base = base_image.convert("RGBA")
    except OSError as exc:
        logger.warning("VN panel: failed to open base image %s: %s", base_image_path, exc)
        return None

    assets = _get_overlay_assets()
    border_img = None
    if gacha_border:
        border_key = f"border_{gacha_border.lower()}"
        border_src = assets.get(border_key)
        if border_src is not None:
            border_img = border_src.resize(base.size, Image.LANCZOS)

    def _resolve_box(setting: Optional[Sequence[int]], fallback: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        if isinstance(setting, (list, tuple)) and len(setting) == 4:
            try:
                return tuple(int(value) for value in setting)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                pass
        return fallback

    def _resolve_int(setting: Optional[object], fallback: int) -> int:
        try:
            return int(setting)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return fallback

    name_box = _resolve_box(layout.get("name_box"), (180, 10, 193, 26))
    text_box = _resolve_box(layout.get("text_box"), (188, 80, 765, 250))
    name_padding = _resolve_int(layout.get("name_padding"), 10)
    text_padding = _resolve_int(layout.get("text_padding"), 12)

    avatar_box: Optional[Tuple[int, int, int, int]] = (0, 4, 220, 250)
    if layout.get("disable_avatar"):
        avatar_box = None
    elif "avatar_box" in layout:
        value = layout.get("avatar_box")
        if value is None:
            avatar_box = None
        else:
            avatar_box = _resolve_box(value, (0, 4, 220, 250))

    avatar_width = avatar_height = 0
    if avatar_box:
        avatar_width = avatar_box[2] - avatar_box[0]
        avatar_height = avatar_box[3] - avatar_box[1]

    draw = ImageDraw.Draw(base)

    avatar_image = None
    if avatar_box:
        avatar_image = compose_game_avatar(
            state.character_name,
            pose_override=gacha_pose_override,
            outfit_override=gacha_outfit_override,
        )
    if avatar_image is not None and avatar_box:
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
        # align sprite bottom with panel bottom; keep centered horizontally
        offset_y = max(0, avatar_height - cropped.height)
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
    elif avatar_box:
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

    overlay_rows: list[dict] = []
    OVERLAY_PADDING = 6
    ROW_SPACING = 4
    TEXT_GAP = 6
    COIN_GROUP_GAP = 14
    STAR_GAP = 4

    def _resize_icon(source: Optional["Image.Image"], target_height: int) -> Optional["Image.Image"]:
        if source is None:
            return None
        icon = source.copy()
        if icon.height != target_height:
            width = max(1, int(icon.width * target_height / icon.height))
            icon = icon.resize((width, target_height), Image.LANCZOS)
        return icon

    def _finalize_row(segments: list[dict]) -> None:
        if not segments:
            return
        row_height = max(seg["height"] for seg in segments)
        row_width = 0
        for seg in segments:
            row_width += seg["width"]
            row_width += seg.get("gap_after", 0)
        overlay_rows.append(
            {
                "segments": segments,
                "width": row_width,
                "height": row_height,
            }
        )

    info_font = _load_vn_font(max(14, VN_TEXT_FONT_SIZE - 8), style="bold")
    coin_data = [
        (key, amount) for key, amount in (("rudy", gacha_rudy), ("frog", gacha_frog)) if amount is not None
    ]
    if coin_data:
        coin_segments: list[dict] = []
        for idx, (icon_key, amount) in enumerate(coin_data):
            amount_text = str(amount)
            text_width = draw.textlength(amount_text, font=info_font)
            text_height = _font_line_height(info_font)
            icon_img = _resize_icon(assets.get(icon_key), 22)
            if icon_img is not None:
                coin_segments.append(
                    {
                        "type": "icon",
                        "image": icon_img,
                        "width": icon_img.width,
                        "height": icon_img.height,
                        "gap_after": TEXT_GAP,
                    }
                )
            coin_segments.append(
                {
                    "type": "text",
                    "text": amount_text,
                    "font": info_font,
                    "width": text_width,
                    "height": text_height,
                    "fill": (220, 220, 220, 255),
                    "gap_after": COIN_GROUP_GAP if idx < len(coin_data) - 1 else 0,
                }
            )
        if coin_segments:
            coin_segments[-1]["gap_after"] = 0
            _finalize_row(coin_segments)

    if gacha_star_count:
        star_count = max(0, min(int(gacha_star_count), 3))
        if star_count > 0:
            star_icon_src = _resize_icon(assets.get("star"), 20)
            star_segments: list[dict] = []
            if star_icon_src is not None:
                for idx in range(star_count):
                    star_segments.append(
                        {
                            "type": "icon",
                            "image": star_icon_src.copy(),
                            "width": star_icon_src.width,
                            "height": star_icon_src.height,
                            "gap_after": STAR_GAP if idx < star_count - 1 else 0,
                        }
                    )
            else:
                star_font = _load_vn_font(max(18, VN_NAME_FONT_SIZE - 6), style="bold")
                star_text = "★" * star_count
                star_segments.append(
                    {
                        "type": "text",
                        "text": star_text,
                        "font": star_font,
                        "width": draw.textlength(star_text, font=star_font),
                        "height": _font_line_height(star_font),
                        "fill": (255, 215, 0, 255),
                        "gap_after": 0,
                    }
                )
            if star_segments:
                _finalize_row(star_segments)

    if overlay_rows:
        overlay_width = max(row["width"] for row in overlay_rows) + OVERLAY_PADDING * 2
        overlay_height = sum(row["height"] for row in overlay_rows)
        if len(overlay_rows) > 1:
            overlay_height += ROW_SPACING * (len(overlay_rows) - 1)
        overlay_height += OVERLAY_PADDING * 2
        overlay_right = base.width - 10
        overlay_left = max(name_box[0], overlay_right - overlay_width)
        overlay_top = name_box[1]
        overlay_bottom = overlay_top + overlay_height

        draw.rectangle((overlay_left, overlay_top, overlay_right, overlay_bottom), fill=(20, 20, 20, 210))

        cursor_y = overlay_top + OVERLAY_PADDING
        for row_index, row in enumerate(overlay_rows):
            row_x = overlay_right - OVERLAY_PADDING - row["width"]
            for segment in row["segments"]:
                gap = segment.get("gap_after", 0)
                if segment["type"] == "icon":
                    icon_img = segment["image"]
                    offset_y = cursor_y + max(0, (row["height"] - icon_img.height) // 2.2)
                    base.paste(icon_img, (int(row_x), int(offset_y)), icon_img)
                    row_x += icon_img.width + gap
                else:
                    text_height = segment["height"]
                    text_y = cursor_y + max(0, (row["height"] - text_height) // 2)
                    draw.text(
                        (row_x, text_y),
                        segment["text"],
                        font=segment["font"],
                        fill=segment.get("fill", (220, 220, 220, 255)),
                    )
                    row_x += segment["width"] + gap
            cursor_y += row["height"]
            if row_index < len(overlay_rows) - 1:
                cursor_y += ROW_SPACING

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
    big_emoji_mode = _is_single_emoji_message(segments)
    emoji_target_size: Optional[int] = None
    if big_emoji_mode:
        target_size = int(min(max_width, available_height) * 0.9)
        if target_size < VN_TEXT_FONT_SIZE:
            target_size = VN_TEXT_FONT_SIZE
        emoji_target_size = target_size
        base_text_font = _load_vn_font(target_size)
    lines, text_font = _fit_text_segments(draw, segments, base_text_font, max_width, available_height)

    vertical_offset = 0
    if big_emoji_mode:
        block_height = emoji_target_size or getattr(text_font, "size", VN_TEXT_FONT_SIZE)
        vertical_offset = max(0, (available_height - block_height) // 2)
    text_y = text_box[1] + text_padding + vertical_offset

    if reply_font and reply_line:
        reply_fill = (190, 190, 190, 255)
        text_x = text_box[0] + text_padding
        draw.text((text_x, text_y), reply_line, fill=reply_fill, font=reply_font)
        text_y += _font_line_height(reply_font) + 6

    base_line_height = getattr(text_font, "size", VN_TEXT_FONT_SIZE)
    line_spacing = 6
    for line in lines:
        if not line:
            text_y += base_line_height + line_spacing
            continue
        line_height = base_line_height
        line_width = 0
        for segment in line:
            seg_height = segment.get("height")
            if big_emoji_mode and emoji_target_size and segment.get("custom_emoji"):
                seg_height = emoji_target_size
            if seg_height:
                line_height = max(line_height, seg_height)
            segment_width = segment.get("width", 0)
            if big_emoji_mode and emoji_target_size and segment.get("custom_emoji"):
                segment_width = emoji_target_size
            line_width += segment_width

        if big_emoji_mode:
            text_x = text_box[0] + text_padding + max(0, (max_width - line_width) // 2)
        else:
            text_x = text_box[0] + text_padding

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
                target_w = int(width) or base_line_height
                target_h = int(height) or base_line_height
                if big_emoji_mode and emoji_target_size:
                    target_w = target_h = emoji_target_size
                advance = width or target_w
                if big_emoji_mode and emoji_target_size:
                    advance = emoji_target_size
                if emoji_img is not None:
                    emoji_render = emoji_img.copy().resize((target_w, target_h), Image.LANCZOS)
                    if big_emoji_mode:
                        offset_y = text_y + max(0, (line_height - emoji_render.height) // 2)
                    else:
                        offset_y = text_y + max(0, base_line_height - emoji_render.height)
                    base.paste(emoji_render, (int(text_x), int(offset_y)), emoji_render)
                else:
                    fallback = segment.get("fallback_text") or text_segment or custom_meta.get("name") or ""
                    if fallback:
                        draw_y = text_y
                        if big_emoji_mode:
                            draw_y = text_y + max(0, (line_height - height) // 2)
                        draw.text((text_x, draw_y), fallback, fill=fill, font=text_font)
                height = max(height, target_h)
                text_x += advance
                continue
            if text_segment:
                draw_y = text_y
                if big_emoji_mode:
                    draw_y = text_y + max(0, (line_height - height) // 2)
                draw.text((text_x, draw_y), text_segment, fill=fill, font=font_segment)
                if segment.get("strike"):
                    strike_y = draw_y + height / 2
                    draw.line(
                        (text_x, strike_y, text_x + width, strike_y),
                        fill=fill,
                        width=max(1, int(height / 10)),
                    )
            text_x += width
        text_y += line_height + line_spacing
    if border_img is not None:
        base = Image.alpha_composite(base, border_img)

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
    "resolve_panel_layout",
    "compose_game_avatar",
    "resolve_character_directory",
    "resolve_character_name_color",
    "list_pose_outfits",
    "list_available_outfits",
    "list_character_accessories",
    "get_accessory_states",
    "set_accessory_state",
    "toggle_accessory_state",
    "get_selected_outfit_name",
    "get_selected_pose_outfit",
    "set_selected_outfit_name",
    "set_selected_pose_outfit",
    "set_character_directory_overrides",
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
