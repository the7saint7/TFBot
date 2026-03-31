"""Visual novel panel rendering and helpers."""

from __future__ import annotations

import io
import json
import logging
import math
import os
import queue
import random
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

import aiohttp
import discord

from tfbot.models import OutfitAsset, ReplyContext, TransformationState
from tfbot.utils import float_from_env, int_from_env, normalize_pose_name, path_from_env, utc_now
from .animation_perf_log import log_event as log_animation_perf_event
from tfbot.transition_constants import (
    APPEARANCE_GIF_ANIMATION_FPS,
    APPEARANCE_GIF_CROSSFADE_MS,
    APPEARANCE_GIF_FINAL_HOLD_MS,
    APPEARANCE_GIF_INITIAL_HOLD_MS,
    APPEARANCE_GIF_MAX_FRAMES,
    APPEARANCE_TRANSITION_PANEL_SIZE,
    BG_GIF_ANIMATION_FPS,
    BG_GIF_FINAL_HOLD_MS,
    BG_GIF_INITIAL_HOLD_MS,
    BG_GIF_MAX_FRAMES,
    BG_GIF_TRAVEL_MS,
    BG_TRANSITION_PANEL_SIZE,
    COLOR_PARITY_ENABLED,
    COLOR_PARITY_EXCLUDED_LABELS,
    COLOR_PARITY_LABELS,
    DEVICE_GIF_ADAPTIVE_COLOR_STEP,
    DEVICE_GIF_ADAPTIVE_MAX_ATTEMPTS,
    DEVICE_GIF_ADAPTIVE_MIN_COLORS,
    DEVICE_GIF_INCLUDE_PARTICLES,
    DEVICE_GIF_TARGET_BYTES,
    DEVICE_GIF_USE_SHARED_PALETTE,
    DEVICE_REROLL_GIF_ANIMATION_FPS,
    DEVICE_REROLL_GIF_EFFECT_MS,
    DEVICE_REROLL_GIF_FINAL_HOLD_MS,
    DEVICE_REROLL_GIF_INITIAL_HOLD_MS,
    DEVICE_REROLL_GIF_MAX_FRAMES,
    DEVICE_REROLL_INCLUDE_PARTICLES,
    DEVICE_SWAP_GIF_ANIMATION_FPS,
    DEVICE_SWAP_GIF_EFFECT_MS,
    DEVICE_SWAP_GIF_FINAL_HOLD_MS,
    DEVICE_SWAP_GIF_INITIAL_HOLD_MS,
    DEVICE_SWAP_GIF_MAX_FRAMES,
    DEVICE_SWAP_PARTICLE_ALPHA,
    DEVICE_SWAP_PARTICLE_COUNT,
    DEVICE_SWAP_PARTICLE_GRID,
    DEVICE_SWAP_WASH_ALPHA,
    GIF_ADAPTIVE_COLOR_STEP,
    GIF_ADAPTIVE_COLOR_STEP_COLOR_PARITY,
    GIF_ADAPTIVE_MAX_ATTEMPTS,
    GIF_ADAPTIVE_MIN_COLORS,
    GIF_ADAPTIVE_MIN_COLORS_COLOR_PARITY,
    GIF_COLORS,
    GIF_DITHER_MODE,
    GIF_POST_OPTIMIZER,
    GIF_POST_OPTIMIZER_LOSSY,
    GIF_POST_OPTIMIZER_TIMEOUT_MS,
    GIF_PREFILTER_RESAMPLE,
    GIF_PREFILTER_SCALE,
    GIF_QUANTIZE_METHOD,
    GIF_SHARED_PALETTE,
    GIF_SHARED_PALETTE_SAMPLES,
    GIF_TARGET_BYTES,
    GIF_TARGET_BYTES_COLOR_PARITY,
    MASS_REROLL_BACKGROUND_NUMBER,
    MASS_REROLL_GIF_ANIMATION_FPS,
    MASS_REROLL_GIF_FINAL_HOLD_MS,
    MASS_REROLL_GIF_INITIAL_HOLD_MS,
    MASS_REROLL_GIF_MAX_FRAMES,
    MASS_REROLL_GIF_TRANSITION_MS,
    MASS_REROLL_STAGE_COUNT,
    MASS_SWAP_BACKGROUND_NUMBER,
    MASS_SWAP_GIF_ANIMATION_FPS,
    MASS_SWAP_GIF_EXIT_MS,
    MASS_SWAP_GIF_FINAL_HOLD_MS,
    MASS_SWAP_GIF_MAX_FRAMES,
    MASS_SWAP_GIF_WANDER_MS,
    MASS_SWAP_GHOST_ALPHA,
    MASS_SWAP_GHOST_MAX_HEIGHT,
    REROLL_GIF_ANIMATION_FPS,
    REROLL_GIF_BOTH_SILHOUETTES_HOLD_MS,
    REROLL_GIF_FINAL_HOLD_MS,
    REROLL_GIF_MAX_FRAMES,
    REROLL_GIF_NEW_OVERLAY_ALPHA,
    REROLL_GIF_NEW_REVEAL_MS,
    REROLL_GIF_NEW_SILHOUETTE_HOLD_MS,
    REROLL_GIF_OLD_SILHOUETTE_FADE_MS,
    REROLL_GIF_OLD_SILHOUETTE_HOLD_MS,
    REROLL_GIF_OLD_TO_SILHOUETTE_MS,
    REROLL_GIF_ORIGINAL_HOLD_MS,
    REROLL_PANEL_SIZE,
    SWAP_GIF_ANIMATION_FPS,
    SWAP_GIF_FINAL_HOLD_MS,
    SWAP_GIF_GHOST_ALPHA,
    SWAP_GIF_GHOST_APPEAR_MS,
    SWAP_GIF_GHOST_DISSOLVE_MS,
    SWAP_GIF_GHOST_END_SCALE,
    SWAP_GIF_GHOST_OFFSET_X,
    SWAP_GIF_INITIAL_HOLD_MS,
    SWAP_GIF_MAX_FRAMES,
    SWAP_GIF_TRAVEL_MS,
    TRANSITION_ALLOW_GIF_PRIMARY,
    TRANSITION_FALLBACK_FORMAT,
    TRANSITION_PRIMARY_FORMAT,
    WEBP_ALPHA_QUALITY,
    WEBP_ALPHA_QUALITY_COLOR_PARITY,
    WEBP_CALIBRATION_BACKEND,
    WEBP_FAST_MAX_ATTEMPTS,
    WEBP_FAST_METHOD,
    WEBP_FAST_MIN_QUALITY,
    WEBP_FAST_TRANSITION_LABELS,
    WEBP_LOSSLESS,
    WEBP_MAX_ATTEMPTS,
    WEBP_MAX_ATTEMPTS_MASS,
    WEBP_MAX_OVERRUN_RATIO,
    WEBP_MAX_OVERRUN_RATIO_COLOR_PARITY,
    WEBP_MAX_OVERRUN_RATIO_DEVICE,
    WEBP_METHOD,
    WEBP_MIN_QUALITY,
    WEBP_MIN_QUALITY_COLOR_PARITY,
    WEBP_MIN_QUALITY_MASS,
    WEBP_QUALITY,
    WEBP_QUALITY_GUARDRAILS,
    WEBP_QUALITY_STEP,
    WEBP_QUALITY_STEP_MASS,
    WEBP_TARGET_BYTES,
    WEBP_TARGET_BYTES_COLOR_PARITY,
    WEBP_TARGET_BYTES_DEVICE,
    WEBP_TARGET_BYTES_MASS,
    WEBP_TARGET_BYTES_STANDARD,
    WEBP_TARGET_HARD_RATIO,
    WEBP_TARGET_HARD_RATIO_MASS,
    WEBP_TARGET_SOFT_RATIO,
    TRANSITION_ENCODE_MAX_TOTAL_FRAMES,
    TRANSITION_GIF_FALLBACK,
    WEBP_ANIMATED_HARD_MAX_BYTES,
    WEBP_TARGET_SOFT_RATIO_MASS,
)

logger = logging.getLogger("tfbot.panels")

# Animated transitions encode as WebP first; GIF is fallback only (see _encode_transition_payload).
if WEBP_CALIBRATION_BACKEND not in {"off", ""}:
    logger.info(
        "WEBP calibration backend '%s' requested; runtime path remains Pillow-first unless explicitly wired.",
        WEBP_CALIBRATION_BACKEND,
    )

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
VN_BIG_EMOJI_MAX_PX = max(VN_TEXT_FONT_SIZE, int_from_env("TFBOT_VN_BIG_EMOJI_MAX_PX", 128))
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
VN_MASK_GUIDE_ENABLED = os.getenv("TFBOT_VN_MASK_GUIDE_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
_VN_MASK_GUIDE_PATH_SETTING = os.getenv("TFBOT_VN_MASK_GUIDE_PATH", "vn_assets/vn_mask.webp").strip()
VN_MASK_GUIDE_THRESHOLD = max(1, min(255, int_from_env("TFBOT_VN_MASK_GUIDE_THRESHOLD", 200)))
VN_MASK_GUIDE_DEBUG = os.getenv("TFBOT_VN_MASK_GUIDE_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}
VN_MASK_VISIBLE_ALPHA_THRESHOLD = max(0, min(254, int_from_env("TFBOT_VN_MASK_VISIBLE_ALPHA_THRESHOLD", 12)))
# Tune this single value during live testing if edges look too soft/hard.
SPRITE_EDGE_AA_STRENGTH = 0.16
if _VN_MASK_GUIDE_PATH_SETTING:
    _vn_mask_guide_path = Path(_VN_MASK_GUIDE_PATH_SETTING)
    if not _vn_mask_guide_path.is_absolute():
        VN_MASK_GUIDE_PATH = (BASE_DIR / _vn_mask_guide_path).resolve()
    else:
        VN_MASK_GUIDE_PATH = _vn_mask_guide_path.resolve()
else:
    VN_MASK_GUIDE_PATH = None
_VN_BG_ROOT_SETTING = os.getenv("TFBOT_VN_BG_ROOT", "").strip()
_VN_BG_DEFAULT_SETTING = os.getenv("TFBOT_VN_BG_DEFAULT", "school/cafeteria.png").strip()
VN_BACKGROUND_DEFAULT_RELATIVE = Path(_VN_BG_DEFAULT_SETTING) if _VN_BG_DEFAULT_SETTING else None

_PILLOW_IMAGE_SETTING = os.getenv("TFBOT_PILLOW_IMAGE", "vn_assets/pillow.png").strip()
_pillow_candidate = Path(_PILLOW_IMAGE_SETTING) if _PILLOW_IMAGE_SETTING else Path("vn_assets/pillow.png")
if not _pillow_candidate.is_absolute():
    PILLOW_IMAGE_PATH = (BASE_DIR / _pillow_candidate).resolve()
else:
    PILLOW_IMAGE_PATH = _pillow_candidate.resolve()
_PILLOW_POINT_TOP_LEFT = (37.0, 38.0)
_PILLOW_POINT_TOP_RIGHT = (310.0, 12.0)
_PILLOW_POINT_BOTTOM_LEFT = (88.0, 600.0)
_PILLOW_POINT_BOTTOM_RIGHT = (360.0, 575.0)
PILLOW_SURFACE_POINTS: Tuple[Tuple[float, float], ...] = (
    _PILLOW_POINT_TOP_LEFT,
    _PILLOW_POINT_TOP_RIGHT,
    _PILLOW_POINT_BOTTOM_LEFT,
    _PILLOW_POINT_BOTTOM_RIGHT,
)
_PILLOW_WARP_POINTS: Tuple[Tuple[float, float], ...] = (
    _PILLOW_POINT_TOP_LEFT,
    _PILLOW_POINT_TOP_RIGHT,
    _PILLOW_POINT_BOTTOM_RIGHT,
    _PILLOW_POINT_BOTTOM_LEFT,
)


def _pillow_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


_PILLOW_TARGET_WIDTH = max(
    1,
    max(
        _pillow_distance(_PILLOW_POINT_TOP_LEFT, _PILLOW_POINT_TOP_RIGHT),
        _pillow_distance(_PILLOW_POINT_BOTTOM_LEFT, _PILLOW_POINT_BOTTOM_RIGHT),
    ),
)
_PILLOW_TARGET_HEIGHT = max(
    1,
    max(
        _pillow_distance(_PILLOW_POINT_TOP_LEFT, _PILLOW_POINT_BOTTOM_LEFT),
        _pillow_distance(_PILLOW_POINT_TOP_RIGHT, _PILLOW_POINT_BOTTOM_RIGHT),
    ),
)

if _VN_BG_ROOT_SETTING:
    candidate_bg_root = Path(_VN_BG_ROOT_SETTING).expanduser()
    VN_BACKGROUND_ROOT = candidate_bg_root.resolve() if candidate_bg_root.exists() else None
elif VN_GAME_ROOT:
    candidate_bg_root = VN_GAME_ROOT / "game" / "images" / "bg"
    VN_BACKGROUND_ROOT = candidate_bg_root.resolve() if candidate_bg_root.exists() else None
else:
    VN_BACKGROUND_ROOT = None

_BG_SELECTION_FILE_SETTING = os.getenv("TFBOT_VN_BG_SELECTIONS", "vn_states/tf_backgrounds.json").strip()
if _BG_SELECTION_FILE_SETTING:
    _bg_path = Path(_BG_SELECTION_FILE_SETTING)
    if _bg_path.is_absolute():
        VN_BACKGROUND_SELECTION_FILE = _bg_path.resolve()
    else:
        VN_BACKGROUND_SELECTION_FILE = (BASE_DIR / _bg_path).resolve()
else:
    VN_BACKGROUND_SELECTION_FILE = None
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

_BG_LAYER_CACHE_LIMIT = max(16, int(os.environ.get("TFBOT_BG_LAYER_CACHE_LIMIT", "96")))
_BG_LAYER_CACHE: "OrderedDict[Tuple[str, int, int, int], Image.Image]" = OrderedDict()
_BG_LAYER_CACHE_LOCK = threading.Lock()

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
    characters_repo_root = _resolve_characters_repo_root()
    if characters_repo_root:
        repo_bg_root = (characters_repo_root / "bg").resolve()
        if repo_bg_root.exists():
            return repo_bg_root
    if VN_BACKGROUND_ROOT and VN_BACKGROUND_ROOT.exists():
        return VN_BACKGROUND_ROOT.resolve()
    return None


def get_background_root() -> Optional[Path]:
    """Public accessor for the effective VN background root."""
    return _get_background_root()


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


@dataclass(frozen=True)
class FaceGitSyncOperation:
    """Sentinel operation for periodic sync from remote."""
    git_repo_root: Path


_FACE_GIT_QUEUE: "queue.Queue[Union[FaceGitOperation, FaceGitSyncOperation]]" = queue.Queue()


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
        
        # Handle sync operations immediately (don't batch)
        if isinstance(operation, FaceGitSyncOperation):
            try:
                _process_face_git_sync(operation)
            finally:
                _FACE_GIT_QUEUE.task_done()
            continue
        
        # Batch regular push operations
        batch.append(operation)
        if _FACE_GIT_BATCH_WINDOW > 0:
            while True:
                try:
                    next_op = _FACE_GIT_QUEUE.get(timeout=_FACE_GIT_BATCH_WINDOW)
                    # If we get a sync operation, put it back and process our batch first
                    if isinstance(next_op, FaceGitSyncOperation):
                        _FACE_GIT_QUEUE.put(next_op)
                        _FACE_GIT_QUEUE.task_done()
                        break
                    batch.append(next_op)
                except queue.Empty:
                    break
        try:
            _process_face_git_batch(batch)
        finally:
            for _ in batch:
                _FACE_GIT_QUEUE.task_done()
            batch.clear()


def _process_face_git_sync(operation: FaceGitSyncOperation) -> None:
    """Pull latest changes from remote repository."""
    git_executable = shutil.which("git")
    if not git_executable:
        return
    
    repo_root = operation.git_repo_root
    if not repo_root.exists():
        return
    
    try:
        # Pull with rebase and autostash to handle any local changes
        pull_cmd = [git_executable, "-C", str(repo_root), "pull", "--rebase", "--autostash", "--quiet"]
        result = subprocess.run(pull_cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            logger.info("Synced faces from remote for %s", repo_root)
        else:
            logger.debug("Face sync pull returned non-zero: %s", result.stderr.strip())
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("Failed to sync faces from remote for %s: %s", repo_root, exc)


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

    # Pull with rebase before pushing to handle concurrent bot updates
    try:
        pull_cmd = [git_executable, "-C", str(repo_root), "pull", "--rebase", "--autostash"]
        pull_result = subprocess.run(pull_cmd, capture_output=True, text=True, timeout=60)
        if pull_result.returncode != 0:
            logger.warning(
                "Face git pull --rebase failed for %s: %s",
                repo_root,
                pull_result.stderr.strip(),
            )
            # Try to abort the rebase and continue without pushing
            subprocess.run(
                [git_executable, "-C", str(repo_root), "rebase", "--abort"],
                capture_output=True,
                text=True,
            )
            return
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("Face git pull timed out or failed for %s: %s", repo_root, exc)
        return

    cmd = [git_executable, "-C", str(repo_root), "push"]
    push_result = subprocess.run(cmd, capture_output=True, text=True)
    if push_result.returncode != 0:
        logger.warning(
            "Face git push failed for %s: %s",
            repo_root,
            push_result.stderr.strip(),
        )
        return

    logger.info(
        "Pushed %d face cache change(s) to remote for %s",
        len(staged),
        repo_root,
    )


_vn_selection_setting = os.getenv("TFBOT_VN_SELECTIONS", "vn_states/tf_outfits.json").strip()
if _vn_selection_setting:
    _vn_selection_path = Path(_vn_selection_setting)
    if _vn_selection_path.is_absolute():
        VN_SELECTION_FILE = _vn_selection_path.resolve()
    else:
        VN_SELECTION_FILE = (BASE_DIR / _vn_selection_path).resolve()
else:
    VN_SELECTION_FILE = None
_VN_LAYOUT_FILE_SETTING = os.getenv("TFBOT_VN_LAYOUTS", "vn_layouts.json").strip()
if _VN_LAYOUT_FILE_SETTING:
    layout_path = Path(_VN_LAYOUT_FILE_SETTING)
    # Resolve relative to BASE_DIR if not absolute
    if not layout_path.is_absolute():
        VN_LAYOUT_FILE = (BASE_DIR / layout_path).resolve()
    else:
        VN_LAYOUT_FILE = layout_path.resolve()
else:
    VN_LAYOUT_FILE = None

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


VN_COMPOSE_DEBUG_DIR = (BASE_DIR / "vn_debug" / "compose").resolve()
VN_COMPOSE_DEBUG_CHARACTERS: Set[str] = {
    _normalize_layout_key("kiyoshiShortstack"),
    _normalize_layout_key("kiyoshiShortStack"),
    _normalize_layout_key("Kiyoshishortstack (Treats of Summer)"),
}


def _compose_debug_enabled(character_name: str) -> bool:
    return _normalize_layout_key(character_name) in VN_COMPOSE_DEBUG_CHARACTERS


def _create_compose_debug_dir(
    character_name: str,
    *,
    variant_name: str,
    outfit_name: str,
) -> Path:
    safe_character = _normalize_layout_key(character_name) or "unknown"
    safe_variant = _normalize_layout_key(variant_name) or "variant"
    safe_outfit = _normalize_layout_key(outfit_name) or "outfit"
    stamp = f"{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns() % 1_000_000:06d}"
    debug_dir = VN_COMPOSE_DEBUG_DIR / safe_character / f"{stamp}-{safe_variant}-{safe_outfit}"
    debug_dir.mkdir(parents=True, exist_ok=True)
    return debug_dir


def _write_compose_debug_metadata(debug_dir: Path, lines: Sequence[str]) -> None:
    try:
        (debug_dir / "metadata.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError as exc:
        logger.warning("VN sprite debug: failed to write metadata in %s: %s", debug_dir, exc)


def _dump_compose_debug_image(debug_dir: Optional[Path], step_name: str, image: Optional["Image.Image"]) -> None:
    if debug_dir is None or image is None:
        return
    try:
        image.save(debug_dir / f"{step_name}.png", format="PNG")
    except OSError as exc:
        logger.warning("VN sprite debug: failed to save %s in %s: %s", step_name, debug_dir, exc)


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


@lru_cache(maxsize=4)
def _load_vn_mask_guide_image(
    panel_size: Tuple[int, int],
    mask_path: str,
    threshold: int,
) -> Optional["Image.Image"]:
    if not mask_path:
        return None
    try:
        from PIL import Image
    except ImportError:
        return None
    mask_file = Path(mask_path)
    if not mask_file.exists():
        logger.warning("VN mask guide missing at %s; using default avatar placement.", mask_file)
        return None
    try:
        with Image.open(mask_file) as mask_src:
            mask = mask_src.convert("L")
    except OSError as exc:
        logger.warning("VN mask guide failed to load at %s: %s", mask_file, exc)
        return None
    if mask.size != panel_size:
        logger.warning(
            "VN mask guide size mismatch %s (expected %s); using default avatar placement.",
            mask.size,
            panel_size,
        )
        return None
    binary = mask.point(lambda px: 255 if px >= threshold else 0, mode="L")
    return binary


@lru_cache(maxsize=64)
def _resolve_vn_mask_allowed_rows(
    panel_size: Tuple[int, int],
    box: Tuple[int, int, int, int],
) -> Optional[Tuple[Optional[Tuple[int, int]], ...]]:
    if not VN_MASK_GUIDE_ENABLED or VN_MASK_GUIDE_PATH is None:
        return None
    mask = _load_vn_mask_guide_image(panel_size, str(VN_MASK_GUIDE_PATH), VN_MASK_GUIDE_THRESHOLD)
    if mask is None:
        return None
    panel_width, panel_height = panel_size
    x0 = max(0, min(panel_width, int(box[0])))
    y0 = max(0, min(panel_height, int(box[1])))
    x1 = max(x0, min(panel_width, int(box[2])))
    y1 = max(y0, min(panel_height, int(box[3])))
    if x1 <= x0 or y1 <= y0:
        return None
    px = mask.load()
    rows: List[Optional[Tuple[int, int]]] = []
    any_row = False
    for y in range(y0, y1):
        row_left: Optional[int] = None
        row_right: Optional[int] = None
        for x in range(x0, x1):
            if px[x, y] > 0:
                if row_left is None:
                    row_left = x - x0
                row_right = (x - x0) + 1
        if row_left is None or row_right is None:
            rows.append(None)
            continue
        rows.append((row_left, row_right))
        any_row = True
    if not any_row:
        return None
    return tuple(rows)

vn_outfit_selection: Dict[str, Dict[str, object]] = {}
background_selections: Dict[str, str] = {}
_vn_config_cache: Dict[str, Dict] = {}
_VN_BACKGROUND_IMAGES: list[Path] = []
_VN_BACKGROUND_IMAGES_ROOT: Optional[Path] = None
_DEVICE_HOLDER_USER_IDS_BY_GUILD: Dict[int, int] = {}


def set_device_holder_user_ids_by_guild(mapping: Mapping[int, int]) -> None:
    _DEVICE_HOLDER_USER_IDS_BY_GUILD.clear()
    for guild_id, user_id in mapping.items():
        try:
            _DEVICE_HOLDER_USER_IDS_BY_GUILD[int(guild_id)] = int(user_id)
        except (TypeError, ValueError):
            continue

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
        "remote": "remote.png",
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
    cache_key: Optional[Tuple[str, int, int, int]] = None
    if background_path is not None and background_path.exists():
        try:
            st = background_path.stat()
            cache_key = (str(background_path.resolve()), st.st_mtime_ns, int(panel_size[0]), int(panel_size[1]))
        except OSError:
            cache_key = None
    if cache_key is not None:
        with _BG_LAYER_CACHE_LOCK:
            cached = _BG_LAYER_CACHE.get(cache_key)
            if cached is not None:
                _BG_LAYER_CACHE.move_to_end(cache_key)
                return cached.copy()
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
    if cache_key is not None:
        with _BG_LAYER_CACHE_LOCK:
            _BG_LAYER_CACHE[cache_key] = layer.copy()
            _BG_LAYER_CACHE.move_to_end(cache_key)
            while len(_BG_LAYER_CACHE) > _BG_LAYER_CACHE_LIMIT:
                _BG_LAYER_CACHE.popitem(last=False)
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


def seed_vn_selection_from_scopes(
    character_name: str,
    *,
    from_scope: Optional[str],
    to_scope: Optional[str],
    persist: bool = True,
) -> bool:
    """Copy effective pose, outfit, and accessory-on states from from_scope into to_scope.

    Used when a user becomes a clone so their VN selection is keyed under the clone scope
    instead of falling through to the shared character fallback.
    """
    directory, _attempted = resolve_character_directory(character_name)
    if directory is None:
        return False
    pose_outfits = list_pose_outfits(character_name)
    if not pose_outfits:
        return False
    src_pose, src_outfit = get_selected_pose_outfit(character_name, scope=from_scope)
    if not src_pose or not src_outfit:
        return False
    pose_lookup = {p.lower(): p for p in pose_outfits.keys()}
    if src_pose.lower() not in pose_lookup:
        return False
    canonical_pose = pose_lookup[src_pose.lower()]
    outfit_options = pose_outfits.get(canonical_pose) or []
    outfit_lookup = {o.lower(): o for o in outfit_options}
    if src_outfit.lower() not in outfit_lookup:
        return False
    canonical_outfit = outfit_lookup[src_outfit.lower()]

    accessory_states = get_accessory_states(character_name, scope=from_scope)
    accessory_entry: Optional[Dict[str, str]] = None
    if accessory_states:
        on_keys = {k.lower(): "on" for k, v in accessory_states.items() if str(v).strip().lower() == "on"}
        if on_keys:
            accessory_entry = on_keys

    new_entry: Dict[str, object] = {"pose": canonical_pose, "outfit": canonical_outfit}
    if accessory_entry:
        new_entry["accessories"] = accessory_entry

    store_key = _selection_store_key(directory, to_scope)
    vn_outfit_selection[store_key] = new_entry
    if persist:
        persist_outfit_selections()
        compose_game_avatar.cache_clear()
        _PILLOW_AVATAR_CACHE.clear()
        logger.info(
            "VN sprite: seeded selection for %s from scope to clone scope (pose=%s outfit=%s)",
            directory.name,
            canonical_pose,
            canonical_outfit,
        )
    return True


_COMPOSE_AVATAR_CACHE: "OrderedDict[Tuple[str, Optional[str], Optional[str], str], 'Image.Image']" = OrderedDict()
_COMPOSE_AVATAR_CACHE_LIMIT = 512
_PILLOW_AVATAR_CACHE: "OrderedDict[Tuple[str, str, str, Optional[str], Optional[str], str], 'Image.Image']" = OrderedDict()
_PILLOW_AVATAR_CACHE_LIMIT = 256


def compose_game_avatar(
    character_name: str,
    pose_override: Optional[str] = None,
    outfit_override: Optional[str] = None,
    selection_scope: Optional[str] = None,
) -> Optional["Image.Image"]:
    scope_key = _normalize_selection_scope(selection_scope)
    cache_key = (character_name, pose_override, outfit_override, scope_key)
    debug_enabled = _compose_debug_enabled(character_name)
    if not debug_enabled:
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
    if image is not None and not debug_enabled:
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
    Queue a sync operation to pull latest changes from remote.
    
    Args:
        git_repo_root: Git repository root directory
    """
    if not git_repo_root.exists():
        return
    
    _ensure_face_git_worker()
    _FACE_GIT_QUEUE.put(FaceGitSyncOperation(git_repo_root=git_repo_root))


def _cache_character_face_background(
    character_dir: Path,
    variant_dir: Path,
    avatar_image: "Image.Image",
    git_repo_root: Optional[Path],
    force: bool = False,
) -> None:
    """
    Background thread function to cache face and commit to git.
    
    Args:
        character_dir: Character directory path
        variant_dir: Variant directory path
        avatar_image: Fully composed avatar PIL Image (copy for thread safety)
        git_repo_root: Git repository root directory (characters_repo), or None if not found
        force: If True, skip existence checks and always regenerate face (default: False)
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
        
        # Skip existence checks if force=True (always regenerate)
        if not force:
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
        else:
            logger.debug("Force mode: regenerating face for %s/%s even if it exists", character_dir.name, variant_dir.name)
        
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


# Periodic sync state (convert hours to seconds)
_FACE_SYNC_INTERVAL_HOURS = max(0.016, float_from_env("TFBOT_FACE_SYNC_HOURS", 3.0))  # Default 3 hours (min 1 minute)
_FACE_SYNC_INTERVAL = _FACE_SYNC_INTERVAL_HOURS * 3600.0  # Convert to seconds
_face_sync_scheduler: Optional[threading.Thread] = None
_face_sync_scheduler_lock = threading.Lock()


def _ensure_face_sync_scheduler() -> None:
    """Start the periodic sync scheduler if not already running."""
    global _face_sync_scheduler
    with _face_sync_scheduler_lock:
        if _face_sync_scheduler and _face_sync_scheduler.is_alive():
            return
        _face_sync_scheduler = threading.Thread(
            target=_face_sync_scheduler_loop,
            name="face-sync-scheduler",
            daemon=True,
        )
        _face_sync_scheduler.start()


def _face_sync_scheduler_loop() -> None:
    """Background loop that periodically triggers face sync from remote."""
    git_repo_root = _resolve_characters_repo_root()
    if not git_repo_root or not git_repo_root.exists():
        logger.warning("Cannot start face sync scheduler: no git repo found")
        return
    
    logger.info(
        "Face sync scheduler started (interval: %.1f hours)",
        _FACE_SYNC_INTERVAL_HOURS,
    )
    
    while True:
        try:
            time.sleep(_FACE_SYNC_INTERVAL)
            logger.debug("Triggering periodic face sync from remote")
            _sync_faces_from_remote(git_repo_root)
        except Exception as exc:
            logger.warning("Error in face sync scheduler: %s", exc, exc_info=True)


def _cache_character_face(
    character_dir: Path,
    variant_dir: Path,
    avatar_image: "Image.Image",
    force: bool = False,
) -> None:
    """
    Cache the detected face from a character's avatar image in a background thread.
    Checks if face is already cached locally and remotely, and if not, launches background thread to detect, save, and commit to git.
    
    Args:
        character_dir: Character directory path
        variant_dir: Variant directory path
        avatar_image: Fully composed avatar PIL Image
        force: If True, skip existence checks and always regenerate the face (default: False)
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
    
    # Skip existence checks if force=True (always regenerate)
    if not force:
        # Check if already cached locally
        if cache_file.exists():
            logger.debug("Face already cached locally for %s/%s", character_dir.name, variant_dir.name)
            return
        
        # Check remote if in git repo (quick check before launching thread)
        if git_repo_root and git_repo_root.exists():
            face_relative_path = cache_file.relative_to(git_repo_root)
            if _check_face_exists_in_remote(git_repo_root, face_relative_path):
                logger.debug("Face exists in remote for %s/%s, syncing...", character_dir.name, variant_dir.name)
                # Queue sync operation through worker
                _sync_faces_from_remote(git_repo_root)
                return
    else:
        logger.debug("Force mode: regenerating face for %s/%s even if it exists", character_dir.name, variant_dir.name)
    
    # Ensure periodic sync scheduler is running
    try:
        _ensure_face_sync_scheduler()
    except Exception:
        pass  # Ignore errors in scheduler startup
    
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
        args=(character_dir, variant_dir, avatar_copy, git_repo_root, force),
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

    debug_dir: Optional[Path] = None
    if _compose_debug_enabled(character_name):
        debug_dir = _create_compose_debug_dir(
            character_name,
            variant_name=variant_dir.name,
            outfit_name=outfit_asset.base_path.stem,
        )

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

    if debug_dir is not None:
        metadata_lines = [
            f"character_name={character_name}",
            f"character_dir={character_dir}",
            f"variant_dir={variant_dir}",
            f"selected_pose={preferred_pose or 'auto'}",
            f"selected_outfit={preferred_outfit or 'auto'}",
            f"resolved_outfit={outfit_path}",
            f"resolved_face={face_path}",
            f"selection_scope={selection_scope or ''}",
            f"sprite_height_limit={sprite_height_limit or 'auto'}",
            f"suppress_outfit_accessories={suppress_outfit_accessories}",
        ]
        if combined_accessory_layers:
            metadata_lines.extend(
                [f"accessory_layer_{index + 1}={layer_path}" for index, layer_path in enumerate(combined_accessory_layers)]
            )
        _write_compose_debug_metadata(debug_dir, metadata_lines)

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
                if debug_dir is None:
                    return cached
                _dump_compose_debug_image(debug_dir, "00_cached_avatar_reference", cached)
            except OSError as exc:
                logger.error(
                    "VN sprite: failed to load cached avatar %s (rebuilding)",
                    cache_file,
                    exc_info=(type(exc), exc, exc.__traceback__),
                )

    try:
        outfit_image = Image.open(outfit_path).convert("RGBA")
    except OSError as exc:
        logger.error(
            "Failed to load outfit %s",
            outfit_path,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return None
    _dump_compose_debug_image(debug_dir, "01_outfit_base", outfit_image)

    for layer_index, layer_path in enumerate(combined_accessory_layers):
        if not layer_path.exists():
            continue
        try:
            layer_image = Image.open(layer_path).convert("RGBA")
            _dump_compose_debug_image(
                debug_dir,
                f"02_accessory_source_{layer_index + 1:02d}_{_normalize_layout_key(layer_path.stem)}",
                layer_image,
            )
            outfit_image.paste(layer_image, (0, 0), layer_image)
            _dump_compose_debug_image(
                debug_dir,
                f"03_after_accessory_{layer_index + 1:02d}_{_normalize_layout_key(layer_path.stem)}",
                outfit_image,
            )
        except OSError as exc:
            logger.error(
                "Failed to load accessory %s",
                layer_path,
                exc_info=(type(exc), exc, exc.__traceback__),
            )

    if face_path and face_path.exists():
        try:
            face_image = Image.open(face_path).convert("RGBA")
            _dump_compose_debug_image(debug_dir, "04_face_source", face_image)
            outfit_image = _alpha_composite_same_canvas(outfit_image, face_image)
            _dump_compose_debug_image(debug_dir, "05_after_face", outfit_image)
        except OSError as exc:
            logger.error(
                "Failed to load face %s",
                face_path,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
    else:
        logger.error(
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
        _dump_compose_debug_image(debug_dir, "06_after_mirror", outfit_image)
    else:
        _dump_compose_debug_image(debug_dir, "06_no_mirror", outfit_image)

    outfit_image = _crop_transparent_vertical(outfit_image)
    _dump_compose_debug_image(debug_dir, "07_after_vertical_crop", outfit_image)
    if sprite_height_limit:
        limit = min(sprite_height_limit, outfit_image.height)
        if limit > 0 and limit < outfit_image.height:
            outfit_image = outfit_image.crop((0, 0, outfit_image.width, limit))
            _dump_compose_debug_image(debug_dir, "08_after_height_limit", outfit_image)

    # Cache detected face if not already cached
    _cache_character_face(character_dir, variant_dir, outfit_image)

    if VN_CACHE_DIR and cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            outfit_image.save(cache_file, format="PNG")
            logger.debug("VN sprite: cached avatar %s", cache_file)
        except OSError as exc:
            logger.error(
                "VN sprite: unable to cache avatar %s",
                cache_file,
                exc_info=(type(exc), exc, exc.__traceback__),
            )

    _dump_compose_debug_image(debug_dir, "99_final_avatar", outfit_image)
    return outfit_image


def _clear_compose_game_avatar_cache() -> None:
    _COMPOSE_AVATAR_CACHE.clear()
    _PILLOW_AVATAR_CACHE.clear()
    # Also clear disk cache to force regeneration when accessories change
    if VN_CACHE_DIR and VN_CACHE_DIR.exists():
        try:
            import shutil
            for cache_dir in VN_CACHE_DIR.iterdir():
                if cache_dir.is_dir():
                    shutil.rmtree(cache_dir, ignore_errors=True)
            logger.debug("VN sprite: cleared disk cache directory %s", VN_CACHE_DIR)
        except Exception as exc:
            logger.error(
                "VN sprite: failed to clear disk cache %s",
                VN_CACHE_DIR,
                exc_info=(type(exc), exc, exc.__traceback__),
            )


compose_game_avatar.cache_clear = _clear_compose_game_avatar_cache  # type: ignore[attr-defined]


def _load_state_avatar_from_path(path: str) -> Optional["Image.Image"]:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = (BASE_DIR / candidate).resolve()
    if not candidate.exists():
        logger.warning("State avatar image missing at %s", candidate)
        return None
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(candidate) as avatar_file:
            return avatar_file.convert("RGBA")
    except OSError as exc:
        logger.warning("Failed to load avatar image %s: %s", candidate, exc)
        return None


@lru_cache(maxsize=1)
def _load_pillow_assets() -> Optional[Tuple["Image.Image", "Image.Image", "Image.Image"]]:
    if not PILLOW_IMAGE_PATH.exists():
        logger.warning("Pillow base image missing at %s", PILLOW_IMAGE_PATH)
        return None
    try:
        from PIL import Image, ImageDraw, ImageOps
    except ImportError:
        return None
    try:
        with Image.open(PILLOW_IMAGE_PATH) as pillow_source:
            pillow_base = pillow_source.convert("RGBA")
    except OSError as exc:
        logger.warning("Failed to read pillow base %s: %s", PILLOW_IMAGE_PATH, exc)
        return None
    mask = Image.new("L", pillow_base.size, 0)
    ImageDraw.Draw(mask).polygon(PILLOW_SURFACE_POINTS, fill=255)
    shading_source = ImageOps.autocontrast(pillow_base.convert("L"))
    shading = Image.new("L", pillow_base.size, 255)
    shading.paste(shading_source, mask=mask)
    return pillow_base, shading, mask


def _prepare_pillow_sprite(image: "Image.Image") -> Optional["Image.Image"]:
    try:
        from PIL import Image
    except ImportError:
        return None
    sprite = _crop_transparent_vertical(image)
    if sprite.width <= 0 or sprite.height <= 0:
        return None
    sprite = sprite.convert("RGBA")
    scale = min(_PILLOW_TARGET_WIDTH / sprite.width, _PILLOW_TARGET_HEIGHT / sprite.height)
    if scale <= 0:
        return None
    if not math.isclose(scale, 1.0, rel_tol=0.02):
        scaled_width = max(1, int(sprite.width * scale))
        scaled_height = max(1, int(sprite.height * scale))
        sprite = sprite.resize((scaled_width, scaled_height), Image.LANCZOS)
    return sprite


def _basic_sprite_projection(
    sprite: "Image.Image",
    canvas_size: Tuple[int, int],
) -> Optional["Image.Image"]:
    try:
        from PIL import Image, ImageChops, ImageDraw
    except ImportError:
        return None
    canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    min_x = int(min(point[0] for point in PILLOW_SURFACE_POINTS))
    min_y = int(min(point[1] for point in PILLOW_SURFACE_POINTS))
    max_x = int(max(point[0] for point in PILLOW_SURFACE_POINTS))
    max_y = int(max(point[1] for point in PILLOW_SURFACE_POINTS))
    width = max(1, max_x - min_x)
    height = max(1, max_y - min_y)
    resized = sprite.resize((width, height), Image.LANCZOS)
    canvas.paste(resized, (min_x, min_y), resized)
    mask = Image.new("L", canvas_size, 0)
    ImageDraw.Draw(mask).polygon(PILLOW_SURFACE_POINTS, fill=255)
    alpha = canvas.getchannel("A")
    alpha = ImageChops.multiply(alpha, mask)
    canvas.putalpha(alpha)
    return canvas


def _warp_sprite_to_pillow(sprite: "Image.Image", canvas_size: Tuple[int, int]) -> Optional["Image.Image"]:
    try:
        import numpy as np  # type: ignore
        import cv2  # type: ignore
    except ImportError:
        return _basic_sprite_projection(sprite, canvas_size)
    try:
        from PIL import Image
    except ImportError:
        return None
    src = np.float32([[0, 0], [sprite.width, 0], [sprite.width, sprite.height], [0, sprite.height]])
    dst = np.float32(_PILLOW_WARP_POINTS)
    sprite_array = np.array(sprite.convert("RGBA"))
    width, height = canvas_size
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        sprite_array,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return Image.fromarray(warped, mode="RGBA")


def _apply_pillow_shading(
    warped: "Image.Image",
    shading_map: Optional["Image.Image"],
) -> "Image.Image":
    if shading_map is None:
        return warped
    try:
        from PIL import Image, ImageChops
    except ImportError:
        return warped
    softened = ImageChops.blend(Image.new("L", shading_map.size, 255), shading_map, 0.35)
    shading_rgb = Image.merge("RGB", (softened, softened, softened))
    shaded_rgb = ImageChops.multiply(warped.convert("RGB"), shading_rgb)
    shaded = Image.merge("RGBA", (*shaded_rgb.split(), warped.getchannel("A")))
    return shaded


def _compose_pillow_avatar(avatar_image: "Image.Image") -> Optional["Image.Image"]:
    assets = _load_pillow_assets()
    if assets is None:
        return None
    pillow_base, shading_map, _ = assets
    sprite = _prepare_pillow_sprite(avatar_image)
    if sprite is None:
        return None
    warped = _warp_sprite_to_pillow(sprite, pillow_base.size)
    if warped is None:
        warped = _basic_sprite_projection(sprite, pillow_base.size)
    if warped is None:
        return None
    shaded = _apply_pillow_shading(warped, shading_map)
    alpha = shaded.getchannel("A").point(lambda value: int(value * 0.95))
    shaded.putalpha(alpha)
    pillow_canvas = pillow_base.copy()
    pillow_canvas.alpha_composite(shaded)
    return pillow_canvas


def compose_state_avatar_image(
    state: TransformationState,
    pose_override: Optional[str] = None,
    outfit_override: Optional[str] = None,
    selection_scope: Optional[str] = None,
) -> Optional["Image.Image"]:
    scope_key = _normalize_selection_scope(selection_scope)
    avatar = compose_game_avatar(
        state.character_name,
        pose_override=pose_override,
        outfit_override=outfit_override,
        selection_scope=scope_key,
    )
    fallback_identity = (state.character_avatar_path or "").strip()
    if avatar is None and fallback_identity:
        avatar = _load_state_avatar_from_path(fallback_identity)
    if avatar is None:
        return None
    if not getattr(state, "is_pillow", False):
        return avatar
    folder_token = (state.character_folder or "").strip().lower()
    cache_key = (
        state.character_name.strip().lower(),
        folder_token,
        fallback_identity,
        pose_override,
        outfit_override,
        scope_key,
    )
    cached = _PILLOW_AVATAR_CACHE.get(cache_key)
    if cached is not None:
        _PILLOW_AVATAR_CACHE.move_to_end(cache_key)
        return cached
    pillow_variant = _compose_pillow_avatar(avatar)
    if pillow_variant is None:
        return avatar
    _PILLOW_AVATAR_CACHE[cache_key] = pillow_variant
    _PILLOW_AVATAR_CACHE.move_to_end(cache_key)
    if len(_PILLOW_AVATAR_CACHE) > _PILLOW_AVATAR_CACHE_LIMIT:
        _PILLOW_AVATAR_CACHE.popitem(last=False)
    return pillow_variant


def _compose_swap_identity_avatar_image(
    state: Optional[TransformationState],
    *,
    selection_scope: Optional[str] = None,
) -> Optional["Image.Image"]:
    if state is None:
        return None
    scope_key = _normalize_selection_scope(selection_scope)
    identity_name = (state.identity_display_name or "").strip() or state.character_name
    avatar = compose_game_avatar(identity_name, selection_scope=scope_key)
    if avatar is not None:
        return avatar
    return compose_state_avatar_image(state, selection_scope=scope_key)


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


def _alpha_composite_same_canvas(
    base_image: "Image.Image",
    overlay_image: "Image.Image",
) -> "Image.Image":
    try:
        from PIL import Image
    except ImportError:
        return base_image
    base_rgba = base_image.convert("RGBA")
    overlay_rgba = overlay_image.convert("RGBA")
    if overlay_rgba.size != base_rgba.size:
        overlay_canvas = Image.new("RGBA", base_rgba.size, (0, 0, 0, 0))
        overlay_canvas.alpha_composite(overlay_rgba)
        overlay_rgba = overlay_canvas
    return Image.alpha_composite(base_rgba, overlay_rgba)


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


def _crop_horizontal_visible_bounds(image: "Image.Image", *, alpha_threshold: int = 0) -> "Image.Image":
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    if image.width <= 1:
        return image
    alpha = image.getchannel("A")
    pixels = alpha.load()
    left = 0
    right = image.width - 1
    while left <= right:
        found = False
        for y in range(image.height):
            if int(pixels[left, y]) > alpha_threshold:
                found = True
                break
        if found:
            break
        left += 1
    while right >= left:
        found = False
        for y in range(image.height):
            if int(pixels[right, y]) > alpha_threshold:
                found = True
                break
        if found:
            break
        right -= 1
    if left >= image.width or right < left:
        return image
    return image.crop((left, 0, right + 1, image.height))


def _visible_row_bounds(
    image: "Image.Image",
    *,
    alpha_threshold: int = 0,
) -> List[Optional[Tuple[int, int]]]:
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    alpha = image.getchannel("A")
    px = alpha.load()
    rows: List[Optional[Tuple[int, int]]] = []
    for y in range(image.height):
        left: Optional[int] = None
        right: Optional[int] = None
        for x in range(image.width):
            if int(px[x, y]) > alpha_threshold:
                if left is None:
                    left = x
                right = x + 1
        if left is None or right is None:
            rows.append(None)
        else:
            rows.append((left, right))
    return rows


def _apply_sprite_edge_antialias(image: "Image.Image", *, strength: float) -> "Image.Image":
    if strength <= 0.0:
        return image
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    try:
        from PIL import Image, ImageChops, ImageFilter
    except ImportError:
        return image
    alpha = image.getchannel("A")
    edge_outer = alpha.filter(ImageFilter.MaxFilter(3))
    edge_inner = alpha.filter(ImageFilter.MinFilter(3))
    edge_mask = ImageChops.subtract(edge_outer, edge_inner)
    if edge_mask.getbbox() is None:
        return image
    softened_alpha = alpha.filter(ImageFilter.GaussianBlur(radius=1.0))
    mixed_alpha = Image.blend(alpha, softened_alpha, max(0.0, min(strength, 1.0)))
    final_alpha = Image.composite(mixed_alpha, alpha, edge_mask)
    out = image.copy()
    out.putalpha(final_alpha)
    return out


def _paste_avatar_into_box(
    base: "Image.Image",
    avatar_image: "Image.Image",
    box: Tuple[int, int, int, int],
    *,
    fit_width: bool = True,
    clip_to_box: bool = True,
    use_mask_guide: bool = False,
) -> bool:
    from PIL import Image

    if avatar_image.mode != "RGBA":
        avatar_image = avatar_image.convert("RGBA")

    box_width = max(0, box[2] - box[0])
    box_height = max(0, box[3] - box[1])
    if box_width <= 0 or box_height <= 0:
        return False

    cropped = _crop_transparent_vertical(avatar_image)
    base_scale = max(VN_AVATAR_SCALE, 0.01)
    if base_scale != 1.0:
        scaled_width = max(1, int(cropped.width * base_scale))
        scaled_height = max(1, int(cropped.height * base_scale))
        cropped = cropped.resize((scaled_width, scaled_height), Image.LANCZOS)

    if fit_width:
        fit_scale = min(
            box_width / cropped.width if cropped.width else 1.0,
            box_height / cropped.height if cropped.height else 1.0,
            1.0,
        )
    else:
        fit_scale = min(
            box_height / cropped.height if cropped.height else 1.0,
            1.0,
        )
    if fit_scale < 1.0:
        scaled_width = max(1, int(cropped.width * fit_scale))
        scaled_height = max(1, int(cropped.height * fit_scale))
        cropped = cropped.resize((scaled_width, scaled_height), Image.LANCZOS)

    cropped = _apply_sprite_edge_antialias(cropped, strength=SPRITE_EDGE_AA_STRENGTH)

    if use_mask_guide and VN_MASK_GUIDE_ENABLED:
        # Ignore faint/empty horizontal margins for placement so right-snap
        # calculations track visible pixels only.
        cropped = _crop_horizontal_visible_bounds(
            cropped,
            alpha_threshold=VN_MASK_VISIBLE_ALPHA_THRESHOLD,
        )

    offset_x = box[0] + ((box_width - cropped.width) // 2)
    offset_y = box[1] + max(0, box_height - cropped.height)
    local_x = (box_width - cropped.width) // 2
    if use_mask_guide and VN_MASK_GUIDE_ENABLED:
        row_bounds = _resolve_vn_mask_allowed_rows(base.size, box)
        if row_bounds is not None and len(row_bounds) == box_height:
            sprite_rows = _visible_row_bounds(cropped, alpha_threshold=VN_MASK_VISIBLE_ALPHA_THRESHOLD)
            canvas_offset_y = max(0, box_height - cropped.height)
            clip_left = 0
            for _ in range(3):
                left_limit = -10_000
                right_limit = 10_000
                has_constraints = False
                for row_idx, row in enumerate(sprite_rows):
                    if row is None:
                        continue
                    box_y = canvas_offset_y + row_idx
                    if box_y < 0 or box_y >= box_height:
                        continue
                    allowed = row_bounds[box_y]
                    if allowed is None:
                        continue
                    vis_left, vis_right = row
                    mask_left, mask_right = allowed
                    left_limit = max(left_limit, mask_left - vis_left)
                    right_limit = min(right_limit, mask_right - vis_right)
                    has_constraints = True
                if not has_constraints:
                    break
                local_x = right_limit  # right-snap always
                if local_x >= left_limit:
                    break
                # Too wide for at least one constrained row -> clip left only.
                clip_needed = left_limit - local_x
                if clip_needed <= 0:
                    break
                clip_left += int(math.ceil(clip_needed))
                if clip_left >= cropped.width:
                    return False
                cropped = cropped.crop((clip_left, 0, cropped.width, cropped.height))
                sprite_rows = _visible_row_bounds(cropped, alpha_threshold=VN_MASK_VISIBLE_ALPHA_THRESHOLD)
                clip_left = 0
            if VN_MASK_GUIDE_DEBUG:
                logger.debug(
                    "VN mask row-fit: box=%s sprite=%sx%s local_x=%s",
                    box,
                    cropped.width,
                    cropped.height,
                    local_x,
                )
        offset_x = box[0] + local_x
    if clip_to_box:
        canvas = Image.new("RGBA", (box_width, box_height), (0, 0, 0, 0))
        canvas_offset_x = max(0, local_x)
        canvas_offset_y = max(0, box_height - cropped.height)
        clipped = cropped
        if local_x < 0:
            src_x0 = min(-local_x, cropped.width)
            if src_x0 >= cropped.width:
                return False
            clipped = cropped.crop((src_x0, 0, cropped.width, cropped.height))
        canvas.paste(clipped, (canvas_offset_x, canvas_offset_y), clipped)
        base.paste(canvas, (box[0], box[1]), canvas)
    else:
        base.paste(cropped, (offset_x, offset_y), cropped)
    return True


def _compose_avatar_layer(
    panel_size: Tuple[int, int],
    avatar_image: Optional["Image.Image"],
    box: Tuple[int, int, int, int],
    *,
    fit_width: bool = True,
    clip_to_box: bool = True,
) -> Optional["Image.Image"]:
    if avatar_image is None:
        return None
    try:
        from PIL import Image
    except ImportError:
        return None
    layer = Image.new("RGBA", panel_size, (0, 0, 0, 0))
    if not _paste_avatar_into_box(
        layer,
        avatar_image,
        box,
        fit_width=fit_width,
        clip_to_box=clip_to_box,
    ):
        return None
    return layer


def _flatten_avatar_layer(placed_layer: Optional["Image.Image"]) -> Optional["Image.Image"]:
    """Single composited RGBA used for silhouette generation (avoids repeated flatten work)."""
    if placed_layer is None:
        return None
    try:
        from PIL import Image
    except ImportError:
        return None
    flattened = Image.new("RGBA", placed_layer.size, (0, 0, 0, 0))
    flattened.alpha_composite(placed_layer.convert("RGBA"))
    return flattened


def _silhouette_from_flattened(
    flattened: "Image.Image",
    *,
    alpha_scale: float = 1.0,
) -> "Image.Image":
    from PIL import Image

    alpha = flattened.getchannel("A")
    if alpha_scale != 1.0:
        alpha = alpha.point(lambda value: max(0, min(255, int(value * alpha_scale))))
    silhouette = Image.new("RGBA", flattened.size, (0, 0, 0, 255))
    silhouette.putalpha(alpha)
    return silhouette


def _make_silhouette_layer(
    placed_layer: Optional["Image.Image"],
    *,
    alpha_scale: float = 1.0,
) -> Optional["Image.Image"]:
    if placed_layer is None:
        return None
    flattened = _flatten_avatar_layer(placed_layer)
    if flattened is None:
        return None
    return _silhouette_from_flattened(flattened, alpha_scale=alpha_scale)


def _make_outlined_silhouette_layer(
    placed_layer: Optional["Image.Image"],
    *,
    alpha_scale: float = 1.0,
    outline_alpha_scale: float = 1.0,
) -> Optional["Image.Image"]:
    base_silhouette = _make_silhouette_layer(placed_layer, alpha_scale=alpha_scale)
    if base_silhouette is None:
        return None
    try:
        from PIL import Image, ImageChops
    except ImportError:
        return base_silhouette
    alpha = base_silhouette.getchannel("A")
    if outline_alpha_scale != 1.0:
        alpha = alpha.point(lambda value: max(0, min(255, int(value * outline_alpha_scale))))
    outline_alpha = Image.new("L", base_silhouette.size, 0)
    for offset_x, offset_y in (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ):
        shifted_alpha = Image.new("L", base_silhouette.size, 0)
        shifted_alpha.paste(alpha, (offset_x, offset_y))
        outline_alpha = ImageChops.lighter(outline_alpha, shifted_alpha)
    outline = Image.new("RGBA", base_silhouette.size, (255, 255, 255, 255))
    outline.putalpha(outline_alpha)
    combined = Image.new("RGBA", base_silhouette.size, (0, 0, 0, 0))
    combined.alpha_composite(outline)
    combined.alpha_composite(base_silhouette)
    return combined


def _layer_with_alpha_scale(
    placed_layer: Optional["Image.Image"],
    *,
    alpha_scale: float = 1.0,
) -> Optional["Image.Image"]:
    if placed_layer is None:
        return None
    try:
        from PIL import Image
    except ImportError:
        return None
    layer = placed_layer.convert("RGBA")
    if alpha_scale == 1.0:
        return layer
    alpha = layer.getchannel("A")
    alpha = alpha.point(lambda value: max(0, min(255, int(value * alpha_scale))))
    layer.putalpha(alpha)
    return layer


def _make_ghost_layer(
    placed_layer: Optional["Image.Image"],
    *,
    alpha_scale: float = 1.0,
) -> Optional["Image.Image"]:
    if placed_layer is None:
        return None
    try:
        from PIL import Image, ImageOps
    except ImportError:
        return None
    layer = placed_layer.convert("RGBA")
    alpha = layer.getchannel("A")
    grayscale = ImageOps.grayscale(layer)
    ghost = Image.merge("RGBA", (grayscale, grayscale, grayscale, alpha))
    if alpha_scale != 1.0:
        ghost_alpha = ghost.getchannel("A").point(
            lambda value: max(0, min(255, int(value * alpha_scale)))
        )
        ghost.putalpha(ghost_alpha)
    return ghost


def _layer_anchor(placed_layer: Optional["Image.Image"]) -> Optional[Tuple[float, float]]:
    if placed_layer is None:
        return None
    bbox = placed_layer.getbbox()
    if bbox is None:
        return None
    center_x = (bbox[0] + bbox[2]) / 2.0
    bottom_y = float(bbox[3])
    return center_x, bottom_y


def _union_layer_bbox(*layers: Optional["Image.Image"]) -> Optional[Tuple[int, int, int, int]]:
    bbox: Optional[Tuple[int, int, int, int]] = None
    for layer in layers:
        if layer is None:
            continue
        current = layer.getbbox()
        if current is None:
            continue
        if bbox is None:
            bbox = current
            continue
        bbox = (
            min(bbox[0], current[0]),
            min(bbox[1], current[1]),
            max(bbox[2], current[2]),
            max(bbox[3], current[3]),
        )
    return bbox


def _compose_static_background_frame(
    background_layer: "Image.Image",
    *layers: Optional["Image.Image"],
) -> "Image.Image":
    frame = background_layer.copy()
    for layer in layers:
        if layer is not None:
            frame.alpha_composite(layer)
    return frame


def _build_device_particle_overlay(
    canvas_size: Tuple[int, int],
    body_regions: Sequence[Tuple[int, int, int, int]],
    *,
    progress: float,
    alpha_scale: float = 1.0,
    seed: int = 0,
) -> Optional["Image.Image"]:
    if not body_regions:
        return None
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None

    effect_strength = max(0.0, math.sin(math.pi * max(0.0, min(1.0, progress)))) * alpha_scale
    if effect_strength <= 0.0:
        return None

    overlay = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    for region_index, region in enumerate(body_regions):
        left, top, right, bottom = region
        region_width = max(1, right - left)
        region_height = max(1, bottom - top)
        effect_left = left
        effect_right = right
        effect_width = max(1, effect_right - effect_left)
        rng = random.Random((seed * 7919) + (region_index * 104729))
        glow_alpha = int(round(135 * effect_strength))
        core_alpha = int(round(DEVICE_SWAP_PARTICLE_ALPHA * effect_strength))

        # Optional soft wash under the particles; set alpha to 0 to disable entirely.
        wash_alpha = int(round(DEVICE_SWAP_WASH_ALPHA * effect_strength))
        if wash_alpha > 0:
            draw.rounded_rectangle(
                (effect_left, top, effect_right, bottom),
                radius=18,
                fill=(35, 135, 255, wash_alpha),
            )

        for particle_index in range(DEVICE_SWAP_PARTICLE_COUNT):
            px = effect_left + int(round(rng.random() * effect_width))
            py_base = top + int(round(rng.random() * region_height))
            upward_travel = region_height * (0.06 + (0.16 * rng.random()))
            px_drift = (rng.random() - 0.5) * 12.0
            py = py_base - (upward_travel * progress)
            px = int(round(px + (px_drift * progress)))
            py = int(round(py))
            if DEVICE_SWAP_PARTICLE_GRID > 1:
                px = int(round(px / DEVICE_SWAP_PARTICLE_GRID) * DEVICE_SWAP_PARTICLE_GRID)
                py = int(round(py / DEVICE_SWAP_PARTICLE_GRID) * DEVICE_SWAP_PARTICLE_GRID)
            radius = 2 + int(rng.random() * 5)
            glow_radius = radius + 3 + int(rng.random() * 4)
            phase = (particle_index / max(DEVICE_SWAP_PARTICLE_COUNT, 1)) + (rng.random() * 0.35)
            flicker = 0.55 + (0.45 * abs(math.sin((progress + phase) * math.pi * 1.5)))
            particle_core_alpha = int(round(core_alpha * flicker))
            particle_glow_alpha = int(round(glow_alpha * flicker))
            draw.ellipse(
                (px - glow_radius, py - glow_radius, px + glow_radius, py + glow_radius),
                fill=(65, 180, 255, particle_glow_alpha),
            )
            draw.ellipse(
                (px - radius, py - radius, px + radius, py + radius),
                fill=(170, 235, 255, particle_core_alpha),
            )

    return overlay


def _transform_layer(
    placed_layer: Optional["Image.Image"],
    *,
    canvas_size: Tuple[int, int],
    center_x: float,
    bottom_y: float,
    scale: float = 1.0,
    alpha_scale: float = 1.0,
    flip_horizontal: bool = False,
) -> Optional["Image.Image"]:
    if placed_layer is None:
        return None
    try:
        from PIL import Image
    except ImportError:
        return None
    bbox = placed_layer.getbbox()
    if bbox is None:
        return None
    crop = placed_layer.crop(bbox).convert("RGBA")
    if scale != 1.0:
        scaled_width = max(1, int(round(crop.width * scale)))
        scaled_height = max(1, int(round(crop.height * scale)))
        crop = crop.resize((scaled_width, scaled_height), Image.LANCZOS)
    if flip_horizontal:
        crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
    if alpha_scale != 1.0:
        alpha = crop.getchannel("A").point(
            lambda value: max(0, min(255, int(value * alpha_scale)))
        )
        crop.putalpha(alpha)
    layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    paste_x = int(round(center_x - (crop.width / 2.0)))
    paste_y = int(round(bottom_y - crop.height))
    layer.paste(crop, (paste_x, paste_y), crop)
    return layer


def _ease_in_out(progress: float) -> float:
    clamped = max(0.0, min(1.0, progress))
    return 0.5 - (0.5 * math.cos(math.pi * clamped))


def _frame_count_for_duration(duration_ms: int) -> int:
    seconds = max(0.0, duration_ms / 1000.0)
    frames = max(2, int(round(seconds * REROLL_GIF_ANIMATION_FPS)))
    if REROLL_GIF_MAX_FRAMES > 0:
        frames = min(frames, max(2, REROLL_GIF_MAX_FRAMES))
    return frames


def _cap_frames(frames: int, cap: int) -> int:
    if cap <= 0:
        return max(2, frames)
    return max(2, min(frames, cap))



def _thin_animation_frames(
    frames: list["Image.Image"],
    durations: list[int],
    max_total: int,
) -> tuple[list["Image.Image"], list[int]]:
    n = len(frames)
    if max_total <= 0 or n <= max_total:
        return list(frames), list(durations)
    picked: list[int] = [0, n - 1]
    inner_budget = max_total - 2
    if inner_budget > 0 and n > 2:
        step = (n - 1) / (inner_budget + 1)
        for i in range(1, inner_budget + 1):
            idx = int(round(i * step))
            idx = max(1, min(n - 2, idx))
            if idx not in picked:
                picked.append(idx)
        picked.sort()
    while len(picked) > max_total:
        best_i = 1
        best_cost = 10**18
        for i in range(1, len(picked) - 1):
            a, b = picked[i - 1], picked[i + 1]
            cost = sum(durations[t] for t in range(a, b) if t < len(durations))
            if cost < best_cost:
                best_cost = cost
                best_i = i
        picked.pop(best_i)
    new_frames: list["Image.Image"] = []
    new_durations: list[int] = []
    for i, start in enumerate(picked):
        new_frames.append(frames[start])
        end = picked[i + 1] if i + 1 < len(picked) else n
        seg = sum(durations[t] for t in range(start, end) if t < len(durations))
        new_durations.append(max(1, seg))
    return new_frames, new_durations


def _gif_frame_from_rgba(frame: "Image.Image") -> "Image.Image":
    try:
        from PIL import Image
    except ImportError:
        return frame
    rgba_frame = frame.convert("RGBA")
    if GIF_PREFILTER_SCALE > 1:
        try:
            resample_lut = {
                "nearest": Image.Resampling.NEAREST,
                "bilinear": Image.Resampling.BILINEAR,
                "bicubic": Image.Resampling.BICUBIC,
                "lanczos": Image.Resampling.LANCZOS,
            }
            resample = resample_lut.get(GIF_PREFILTER_RESAMPLE, Image.Resampling.LANCZOS)
            enlarged = rgba_frame.resize(
                (rgba_frame.width * GIF_PREFILTER_SCALE, rgba_frame.height * GIF_PREFILTER_SCALE),
                resample=resample,
            )
            rgba_frame = enlarged.resize((rgba_frame.width, rgba_frame.height), resample=resample)
        except Exception:
            pass
    return rgba_frame


def _quantize_frame(
    frame: "Image.Image",
    palette_source: Optional["Image.Image"] = None,
    *,
    colors_override: Optional[int] = None,
    conservative: bool = False,
) -> "Image.Image":
    from PIL import Image

    method_lut = {
        "fastoctree": Image.FASTOCTREE,
        "mediancut": Image.MEDIANCUT,
    }
    dither_lut = {
        "none": Image.Dither.NONE,
        "floyd": Image.Dither.FLOYDSTEINBERG,
    }
    method = method_lut.get(GIF_QUANTIZE_METHOD, Image.FASTOCTREE)
    dither = dither_lut.get(GIF_DITHER_MODE, Image.Dither.NONE)
    colors = max(32, min(256, int(colors_override if colors_override is not None else GIF_COLORS)))
    if conservative:
        method = Image.FASTOCTREE
        dither = Image.Dither.NONE
        colors = min(colors, 96)
    if palette_source is not None:
        palette_image = palette_source if palette_source.mode == "P" else palette_source.convert("P")
        return frame.convert("RGB").quantize(palette=palette_image, dither=dither)
    return frame.convert("RGBA").quantize(
        colors=colors,
        method=method,
        dither=dither,
    )


def _palette_sample_indices(total: int, sample_count: int) -> list[int]:
    if total <= 0:
        return []
    picks = {0, total - 1, total // 2}
    if sample_count <= 3:
        return sorted(picks)
    span = max(1, sample_count - 1)
    for i in range(sample_count):
        idx = int(round((total - 1) * (i / span)))
        picks.add(idx)
    return sorted(picks)


def _build_shared_palette_source(
    frames: list["Image.Image"],
    *,
    sample_count: int,
    colors: int,
) -> Optional["Image.Image"]:
    if not frames:
        return None
    from PIL import Image

    indices = _palette_sample_indices(len(frames), sample_count)
    if not indices:
        return None
    width, height = frames[0].size
    canvas = Image.new("RGB", (width, height * len(indices)), (0, 0, 0))
    for row, idx in enumerate(indices):
        sample = frames[idx].convert("RGB")
        canvas.paste(sample, (0, row * height))
    return _quantize_frame(canvas, colors_override=colors)


def _dedupe_gif_frames(frames: list["Image.Image"], durations: list[int]) -> tuple[list["Image.Image"], list[int]]:
    if not frames:
        return frames, durations
    merged_frames: list["Image.Image"] = [frames[0]]
    merged_durations: list[int] = [max(0, int(durations[0])) if durations else 0]
    prev = frames[0]
    prev_bytes = prev.tobytes()
    prev_palette = tuple(prev.getpalette() or [])
    for idx in range(1, len(frames)):
        frame = frames[idx]
        duration = max(0, int(durations[idx])) if idx < len(durations) else 0
        frame_bytes = frame.tobytes()
        frame_palette = tuple(frame.getpalette() or [])
        if frame_bytes == prev_bytes and frame_palette == prev_palette:
            merged_durations[-1] += duration
            continue
        merged_frames.append(frame)
        merged_durations.append(duration)
        prev_bytes = frame_bytes
        prev_palette = frame_palette
    return merged_frames, merged_durations


def _frame_has_partial_alpha(frame: "Image.Image") -> bool:
    rgba = frame.convert("RGBA")
    alpha = rgba.getchannel("A")
    minimum, maximum = alpha.getextrema()
    if minimum >= 255 or maximum <= 0:
        return False
    if minimum == 0 and maximum == 255:
        histogram = alpha.histogram()
        return any(count > 0 for count in histogram[1:255])
    return True


def _prepare_gif_frames(
    frames: list["Image.Image"],
    durations: list[int],
    *,
    use_shared_palette: bool,
    dedupe: bool,
    colors_override: Optional[int] = None,
) -> tuple[list["Image.Image"], list[int]]:
    if not frames:
        return frames, durations
    rgba_frames = [_gif_frame_from_rgba(frame) for frame in frames]
    colors = max(32, min(256, int(colors_override if colors_override is not None else GIF_COLORS)))
    has_partial_alpha = any(_frame_has_partial_alpha(frame) for frame in rgba_frames)
    use_shared_palette_effective = use_shared_palette and GIF_SHARED_PALETTE and not has_partial_alpha
    if use_shared_palette_effective:
        try:
            palette_source = _build_shared_palette_source(
                rgba_frames,
                sample_count=GIF_SHARED_PALETTE_SAMPLES,
                colors=colors,
            )
            if palette_source is None:
                quantized_frames = [_quantize_frame(frame, colors_override=colors) for frame in rgba_frames]
            else:
                quantized_frames = [
                    _quantize_frame(frame, palette_source=palette_source, colors_override=colors)
                    for frame in rgba_frames
                ]
        except Exception:
            try:
                quantized_frames = [_quantize_frame(frame, colors_override=colors) for frame in rgba_frames]
            except Exception:
                quantized_frames = [_quantize_frame(frame, colors_override=colors, conservative=True) for frame in rgba_frames]
    else:
        try:
            quantized_frames = [_quantize_frame(frame, colors_override=colors) for frame in rgba_frames]
        except Exception:
            quantized_frames = [_quantize_frame(frame, colors_override=colors, conservative=True) for frame in rgba_frames]
    normalized_durations = [max(0, int(value)) for value in durations]
    if dedupe:
        quantized_frames, normalized_durations = _dedupe_gif_frames(quantized_frames, normalized_durations)
    return quantized_frames, normalized_durations


def _maybe_run_gifsicle(payload: bytes, *, colors_override: Optional[int] = None) -> bytes:
    if GIF_POST_OPTIMIZER != "gifsicle":
        return payload
    try:
        with tempfile.TemporaryDirectory(prefix="tfbot-gifopt-") as tmp_dir:
            in_path = Path(tmp_dir) / "in.gif"
            out_path = Path(tmp_dir) / "out.gif"
            in_path.write_bytes(payload)
            colors = max(32, min(256, int(colors_override if colors_override is not None else GIF_COLORS)))
            options = ["-O3", "--careful", f"--colors={colors}"]
            if GIF_POST_OPTIMIZER_LOSSY > 0:
                options.append(f"--lossy={GIF_POST_OPTIMIZER_LOSSY}")
            try:
                from pygifsicle import optimize as pygifsicle_optimize  # type: ignore

                pygifsicle_optimize(str(in_path), str(out_path), options=options)
            except Exception:
                gifsicle_bin = shutil.which("gifsicle")
                if not gifsicle_bin:
                    return payload
                cmd = [gifsicle_bin, *options, str(in_path), "-o", str(out_path)]
                completed = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=max(0.1, GIF_POST_OPTIMIZER_TIMEOUT_MS / 1000.0),
                    check=False,
                )
                if completed.returncode != 0:
                    return payload
            if not out_path.exists():
                return payload
            optimized = out_path.read_bytes()
            if optimized and len(optimized) < len(payload):
                return optimized
    except Exception:
        return payload
    return payload


def _encode_gif_payload(
    frames: list["Image.Image"],
    durations: list[int],
    *,
    loop: int,
    optimize: bool,
    disposal: int,
    shared_palette: bool,
    dedupe: bool,
    allow_post_opt: bool,
    profile: str = "default",
    label: Optional[str] = None,
    target_bytes_override: Optional[int] = None,
    adaptive_min_colors_override: Optional[int] = None,
    adaptive_color_step_override: Optional[int] = None,
    adaptive_max_attempts_override: Optional[int] = None,
) -> Optional[bytes]:
    target_bytes = max(0, int(target_bytes_override if target_bytes_override is not None else GIF_TARGET_BYTES))
    min_colors = min(
        GIF_COLORS,
        max(32, int(adaptive_min_colors_override if adaptive_min_colors_override is not None else GIF_ADAPTIVE_MIN_COLORS)),
    )
    attempts = max(1, int(adaptive_max_attempts_override if adaptive_max_attempts_override is not None else GIF_ADAPTIVE_MAX_ATTEMPTS))
    step = max(4, int(adaptive_color_step_override if adaptive_color_step_override is not None else GIF_ADAPTIVE_COLOR_STEP))

    colors_to_try: list[int] = [GIF_COLORS]
    next_colors = GIF_COLORS
    for _ in range(attempts - 1):
        next_colors = max(min_colors, next_colors - step)
        if next_colors == colors_to_try[-1]:
            break
        colors_to_try.append(next_colors)

    best_payload: Optional[bytes] = None
    best_colors = GIF_COLORS
    attempts_used = 0
    for idx, colors in enumerate(colors_to_try, start=1):
        attempts_used = idx
        prepared_frames, prepared_durations = _prepare_gif_frames(
            frames,
            durations,
            use_shared_palette=shared_palette,
            dedupe=dedupe,
            colors_override=colors,
        )
        if not prepared_frames:
            return None
        output = io.BytesIO()
        prepared_frames[0].save(
            output,
            format="GIF",
            save_all=True,
            append_images=prepared_frames[1:],
            duration=prepared_durations,
            loop=loop,
            optimize=optimize,
            disposal=disposal,
        )
        payload = output.getvalue()
        if allow_post_opt:
            payload = _maybe_run_gifsicle(payload, colors_override=colors)
        if best_payload is None or len(payload) < len(best_payload):
            best_payload = payload
            best_colors = colors
        if target_bytes > 0 and len(payload) <= target_bytes:
            break
        if target_bytes <= 0:
            break

    if best_payload is None:
        return None

    log_animation_perf_event(
        "gif_encode_final",
        profile=profile,
        label=label or "unknown",
        bytes=len(best_payload),
        colors=best_colors,
        dither=GIF_DITHER_MODE,
        shared_palette=int(bool(shared_palette and GIF_SHARED_PALETTE)),
        dedupe=int(bool(dedupe)),
        post_opt=int(bool(allow_post_opt and GIF_POST_OPTIMIZER == "gifsicle")),
        target_bytes=target_bytes if target_bytes > 0 else "off",
        attempts_used=attempts_used,
        attempts_planned=len(colors_to_try),
    )
    return best_payload


def _normalize_transition_format(value: str, *, default: str = "gif") -> str:
    normalized = (value or "").strip().lower()
    if normalized in {"webp", "gif"}:
        return normalized
    return default


def _filename_with_ext(filename: str, ext: str) -> str:
    suffix = f".{ext.lstrip('.').lower()}"
    stem = Path(filename or "transition").stem
    if not stem:
        stem = "transition"
    return f"{stem}{suffix}"


def _encode_webp_payload(
    frames: list["Image.Image"],
    durations: list[int],
    *,
    loop: int,
    dedupe: bool,
    profile: str = "default",
    label: Optional[str] = None,
    target_bytes_override: Optional[int] = None,
    quality_override: Optional[int] = None,
    method_override: Optional[int] = None,
    min_quality_override: Optional[int] = None,
    quality_step_override: Optional[int] = None,
    max_attempts_override: Optional[int] = None,
    alpha_quality_override: Optional[int] = None,
) -> Optional[bytes]:
    if not frames:
        return None
    target_bytes = max(0, int(target_bytes_override if target_bytes_override is not None else WEBP_TARGET_BYTES))
    method = max(0, min(6, int(method_override if method_override is not None else WEBP_METHOD)))
    alpha_quality = max(1, min(100, int(alpha_quality_override if alpha_quality_override is not None else WEBP_ALPHA_QUALITY)))
    start_quality = max(1, min(100, int(quality_override if quality_override is not None else WEBP_QUALITY)))
    quality_values: list[int] = [start_quality]
    min_quality = max(25, min(100, int(min_quality_override if min_quality_override is not None else WEBP_MIN_QUALITY)))
    quality_step = max(1, min(20, int(quality_step_override if quality_step_override is not None else WEBP_QUALITY_STEP)))
    max_attempts = max(1, min(8, int(max_attempts_override if max_attempts_override is not None else WEBP_MAX_ATTEMPTS)))
    if not WEBP_LOSSLESS:
        next_quality = start_quality
        for _ in range(max_attempts - 1):
            next_quality = max(min_quality, next_quality - quality_step)
            if next_quality == quality_values[-1]:
                break
            quality_values.append(next_quality)

    rgba_frames = [_gif_frame_from_rgba(frame).convert("RGBA") for frame in frames]
    normalized_durations = [max(0, int(value)) for value in durations]
    if dedupe:
        deduped_frames: list["Image.Image"] = [rgba_frames[0]]
        deduped_durations: list[int] = [normalized_durations[0] if normalized_durations else 0]
        prev = rgba_frames[0].tobytes()
        for idx in range(1, len(rgba_frames)):
            cur_img = rgba_frames[idx]
            duration = normalized_durations[idx] if idx < len(normalized_durations) else 0
            if cur_img is deduped_frames[-1]:
                deduped_durations[-1] += duration
                continue
            current = cur_img.tobytes()
            if current == prev:
                deduped_durations[-1] += duration
            else:
                deduped_frames.append(rgba_frames[idx])
                deduped_durations.append(duration)
                prev = current
        rgba_frames = deduped_frames
        normalized_durations = deduped_durations

    selected_payload: Optional[bytes] = None
    selected_quality = quality_values[0]
    smallest_payload: Optional[bytes] = None
    smallest_quality = quality_values[0]
    first_payload: Optional[bytes] = None
    first_quality = quality_values[0]
    soft_ratio = WEBP_TARGET_SOFT_RATIO_MASS if profile == "mass" else WEBP_TARGET_SOFT_RATIO
    hard_ratio = WEBP_TARGET_HARD_RATIO_MASS if profile == "mass" else WEBP_TARGET_HARD_RATIO
    apply_color_parity = _is_color_parity_transition(label, profile=profile)
    if profile == "device":
        max_overrun_ratio = WEBP_MAX_OVERRUN_RATIO_DEVICE
    else:
        max_overrun_ratio = WEBP_MAX_OVERRUN_RATIO_COLOR_PARITY if apply_color_parity else WEBP_MAX_OVERRUN_RATIO
    soft_target = max(0, int(round(target_bytes * soft_ratio))) if target_bytes > 0 else 0
    hard_target = max(soft_target, int(round(target_bytes * hard_ratio))) if target_bytes > 0 else 0
    max_overrun_target = max(hard_target, int(round(target_bytes * max_overrun_ratio))) if target_bytes > 0 else 0
    decision = "smallest"
    attempts_used = 0
    for idx, quality in enumerate(quality_values, start=1):
        attempts_used = idx
        output = io.BytesIO()
        _append = rgba_frames[1:] if len(rgba_frames) > 1 else []
        rgba_frames[0].save(
            output,
            format="WEBP",
            save_all=bool(_append),
            append_images=_append,
            duration=normalized_durations,
            loop=max(0, int(loop)),
            quality=quality,
            method=method,
            lossless=WEBP_LOSSLESS,
            minimize_size=True,
            alpha_quality=alpha_quality,
            exact=False,
        )
        payload = output.getvalue()
        if first_payload is None:
            first_payload = payload
            first_quality = quality
        if smallest_payload is None or len(payload) < len(smallest_payload):
            smallest_payload = payload
            smallest_quality = quality

        if WEBP_QUALITY_GUARDRAILS and target_bytes > 0:
            if len(payload) <= soft_target:
                selected_payload = payload
                selected_quality = quality
                decision = "soft_target"
                break
            if len(payload) <= hard_target and selected_payload is None:
                selected_payload = payload
                selected_quality = quality
                decision = "hard_target"
        elif target_bytes > 0 and len(payload) <= target_bytes:
            selected_payload = payload
            selected_quality = quality
            decision = "target"
            break
        elif target_bytes <= 0:
            selected_payload = payload
            selected_quality = quality
            decision = "quality_first"
            break

    if selected_payload is None:
        if WEBP_QUALITY_GUARDRAILS and target_bytes > 0 and selected_payload is None:
            if first_payload is not None and hard_target > 0 and len(first_payload) <= hard_target:
                selected_payload = first_payload
                selected_quality = first_quality
                decision = "first_under_hard_cap"
            elif first_payload is not None and max_overrun_target > 0 and len(first_payload) <= max_overrun_target:
                selected_payload = first_payload
                selected_quality = first_quality
                decision = "first_under_overrun_cap"
        if selected_payload is None:
            selected_payload = smallest_payload
            selected_quality = smallest_quality
            decision = "smallest"

    if selected_payload is None:
        return None
    hard_cap = max(0, int(WEBP_ANIMATED_HARD_MAX_BYTES))
    if hard_cap > 0 and len(selected_payload) > hard_cap:
        q_cap = max(38, int(selected_quality))
        best_blob: Optional[bytes] = selected_payload
        capped_ok = False
        for _ in range(12):
            q_cap = max(38, q_cap - 5)
            out_try = io.BytesIO()
            ap = rgba_frames[1:] if len(rgba_frames) > 1 else []
            rgba_frames[0].save(
                out_try,
                format="WEBP",
                save_all=bool(ap),
                append_images=ap,
                duration=normalized_durations,
                loop=max(0, int(loop)),
                quality=q_cap,
                method=min(method, WEBP_FAST_METHOD),
                lossless=WEBP_LOSSLESS,
                minimize_size=True,
                alpha_quality=alpha_quality,
                exact=False,
            )
            blob = out_try.getvalue()
            if len(blob) < len(best_blob):
                best_blob = blob
            if len(blob) <= hard_cap:
                selected_payload = blob
                selected_quality = q_cap
                decision = "hard_cap_quality"
                capped_ok = True
                break
        if not capped_ok:
            thin_f = list(rgba_frames)
            thin_d = list(normalized_durations)
            for _ in range(4):
                if len(thin_f) <= 8:
                    break
                cap_n = max(8, (len(thin_f) * 2) // 3)
                thin_f, thin_d = _thin_animation_frames(thin_f, thin_d, cap_n)
                out_try = io.BytesIO()
                ap2 = thin_f[1:] if len(thin_f) > 1 else []
                thin_f[0].save(
                    out_try,
                    format="WEBP",
                    save_all=bool(ap2),
                    append_images=ap2,
                    duration=thin_d,
                    loop=max(0, int(loop)),
                    quality=max(38, min(q_cap, 68)),
                    method=min(method, WEBP_FAST_METHOD),
                    lossless=WEBP_LOSSLESS,
                    minimize_size=True,
                    alpha_quality=alpha_quality,
                    exact=False,
                )
                blob = out_try.getvalue()
                if len(blob) < len(best_blob):
                    best_blob = blob
                if len(blob) <= hard_cap:
                    selected_payload = blob
                    selected_quality = max(38, min(q_cap, 68))
                    decision = "hard_cap_thin"
                    capped_ok = True
                    break
        if not capped_ok:
            selected_payload = best_blob
            decision = "hard_cap_best_effort"
        if len(selected_payload) > hard_cap:
            decision = "hard_cap_rejected"
            logger.warning(
                "WebP transition exceeds hard cap (%d > %d bytes, label=%s); rejecting (caller may use static fallback)",
                len(selected_payload),
                hard_cap,
                label or "unknown",
            )
    log_animation_perf_event(
        "webp_encode_final",
        profile=profile,
        label=label or "unknown",
        bytes=len(selected_payload),
        quality=selected_quality,
        method=method,
        lossless=int(WEBP_LOSSLESS),
        dedupe=int(bool(dedupe)),
        target_bytes=target_bytes if target_bytes > 0 else "off",
        attempts_used=attempts_used,
        attempts_planned=len(quality_values),
        min_quality=min_quality,
        quality_step=quality_step,
        soft_target=soft_target if soft_target > 0 else "off",
        hard_target=hard_target if hard_target > 0 else "off",
        max_overrun_target=max_overrun_target if max_overrun_target > 0 else "off",
        max_overrun_ratio=max_overrun_ratio,
        alpha_quality=alpha_quality,
        quality_guardrails=int(WEBP_QUALITY_GUARDRAILS),
        decision=decision,
    )
    if decision == "hard_cap_rejected":
        return None
    return selected_payload


def _is_color_parity_transition(label: Optional[str], *, profile: str) -> bool:
    if not COLOR_PARITY_ENABLED:
        return False
    normalized = (label or "").strip().lower()
    if not normalized:
        return False
    if profile == "mass":
        return False
    if normalized in COLOR_PARITY_EXCLUDED_LABELS:
        return False
    return normalized in COLOR_PARITY_LABELS


def _encode_transition_payload(
    frames: list["Image.Image"],
    durations: list[int],
    *,
    loop: int,
    optimize: bool,
    disposal: int,
    shared_palette: bool,
    dedupe: bool,
    allow_post_opt: bool,
    profile: str = "default",
    label: Optional[str] = None,
    gif_target_bytes_override: Optional[int] = None,
    gif_adaptive_min_colors_override: Optional[int] = None,
    gif_adaptive_color_step_override: Optional[int] = None,
    gif_adaptive_max_attempts_override: Optional[int] = None,
    webp_target_bytes_override: Optional[int] = None,
    webp_min_quality_override: Optional[int] = None,
    webp_quality_step_override: Optional[int] = None,
    webp_max_attempts_override: Optional[int] = None,
) -> tuple[Optional[bytes], str, bool]:
    apply_color_parity = _is_color_parity_transition(label, profile=profile)
    shared_palette_effective = shared_palette
    gif_target_bytes_effective = gif_target_bytes_override
    gif_min_colors_effective = gif_adaptive_min_colors_override
    gif_color_step_effective = gif_adaptive_color_step_override
    webp_target_bytes_effective = webp_target_bytes_override
    webp_min_quality_effective = webp_min_quality_override
    webp_alpha_quality_effective: Optional[int] = None
    if apply_color_parity:
        shared_palette_effective = False
        gif_target_bytes_effective = max(
            int(gif_target_bytes_override) if gif_target_bytes_override is not None else GIF_TARGET_BYTES,
            GIF_TARGET_BYTES_COLOR_PARITY,
        )
        gif_min_colors_effective = max(
            int(gif_adaptive_min_colors_override)
            if gif_adaptive_min_colors_override is not None
            else GIF_ADAPTIVE_MIN_COLORS,
            GIF_ADAPTIVE_MIN_COLORS_COLOR_PARITY,
        )
        gif_color_step_effective = min(
            int(gif_adaptive_color_step_override)
            if gif_adaptive_color_step_override is not None
            else GIF_ADAPTIVE_COLOR_STEP,
            GIF_ADAPTIVE_COLOR_STEP_COLOR_PARITY,
        )
        webp_target_bytes_effective = max(
            int(webp_target_bytes_override) if webp_target_bytes_override is not None else WEBP_TARGET_BYTES_STANDARD,
            WEBP_TARGET_BYTES_COLOR_PARITY,
        )
        webp_min_quality_effective = max(
            int(webp_min_quality_override) if webp_min_quality_override is not None else WEBP_MIN_QUALITY,
            WEBP_MIN_QUALITY_COLOR_PARITY,
        )
        webp_alpha_quality_effective = WEBP_ALPHA_QUALITY_COLOR_PARITY

    norm_label = (label or "").strip().lower()
    webp_method_override: Optional[int] = None
    webp_max_eff = webp_max_attempts_override
    webp_min_eff = webp_min_quality_effective
    if norm_label in WEBP_FAST_TRANSITION_LABELS:
        webp_method_override = WEBP_FAST_METHOD
        webp_max_eff = WEBP_FAST_MAX_ATTEMPTS
        base_mq = (
            int(webp_min_quality_effective)
            if webp_min_quality_effective is not None
            else WEBP_MIN_QUALITY
        )
        webp_min_eff = min(base_mq, WEBP_FAST_MIN_QUALITY)

    primary = _normalize_transition_format(TRANSITION_PRIMARY_FORMAT, default="webp")
    if primary == "gif":
        if not TRANSITION_ALLOW_GIF_PRIMARY:
            logger.warning(
                "TRANSITION_PRIMARY_FORMAT=gif is ignored; transitions encode WebP first. "
                "Set TRANSITION_ALLOW_GIF_PRIMARY=True in tfbot/transition_constants.py to use GIF as primary."
            )
            primary = "webp"
    fallback = _normalize_transition_format(TRANSITION_FALLBACK_FORMAT, default="gif")
    frames, durations = _thin_animation_frames(list(frames), list(durations), TRANSITION_ENCODE_MAX_TOTAL_FRAMES)

    # Always try animated WebP first for every transition label; GIF runs only if enabled and WebP fails.
    payload = _encode_webp_payload(
        frames,
        durations,
        loop=loop,
        dedupe=dedupe,
        profile=profile,
        label=label,
        target_bytes_override=webp_target_bytes_effective,
        min_quality_override=webp_min_eff,
        quality_step_override=webp_quality_step_override,
        max_attempts_override=webp_max_eff,
        alpha_quality_override=webp_alpha_quality_effective,
        method_override=webp_method_override,
    )
    if payload is not None:
        webp_method_logged = (
            webp_method_override if webp_method_override is not None else WEBP_METHOD
        )
        log_animation_perf_event(
            "transition_encode_selected",
            profile=profile,
            label=label or "unknown",
            format="webp",
            selected_format="webp",
            primary_format=primary,
            fallback_format=fallback,
            fallback_used=0,
            color_parity=int(apply_color_parity),
            bytes=len(payload),
            webp_quality=WEBP_QUALITY,
            webp_method=webp_method_logged,
            webp_target_bytes=webp_target_bytes_effective if webp_target_bytes_effective is not None else WEBP_TARGET_BYTES,
            gif_colors=GIF_COLORS,
            gif_target_bytes=gif_target_bytes_effective if gif_target_bytes_effective is not None else GIF_TARGET_BYTES,
        )
        return payload, "webp", False

    if TRANSITION_GIF_FALLBACK:
        gif_payload = _encode_gif_payload(
            frames,
            durations,
            loop=loop,
            optimize=optimize,
            disposal=disposal,
            shared_palette=shared_palette_effective,
            dedupe=dedupe,
            allow_post_opt=False,
            profile=profile,
            label=label,
            target_bytes_override=gif_target_bytes_effective,
            adaptive_min_colors_override=gif_min_colors_effective,
            adaptive_color_step_override=gif_color_step_effective,
            adaptive_max_attempts_override=gif_adaptive_max_attempts_override,
        )
        if gif_payload is not None:
            log_animation_perf_event(
                "transition_encode_selected",
                profile=profile,
                label=label or "unknown",
                format="gif",
                selected_format="gif",
                primary_format=primary,
                fallback_format=fallback,
                fallback_used=1,
                color_parity=int(apply_color_parity),
                bytes=len(gif_payload),
                webp_quality=WEBP_QUALITY,
                webp_method=webp_method_override if webp_method_override is not None else WEBP_METHOD,
                webp_target_bytes=webp_target_bytes_effective if webp_target_bytes_effective is not None else WEBP_TARGET_BYTES,
                gif_colors=GIF_COLORS,
                gif_target_bytes=gif_target_bytes_effective if gif_target_bytes_effective is not None else GIF_TARGET_BYTES,
            )
            return gif_payload, "gif", True

    log_animation_perf_event(
        "transition_encode_selected",
        profile=profile,
        label=label or "unknown",
        format="none",
        selected_format="none",
        primary_format=primary,
        fallback_format=fallback,
        fallback_used=0,
        color_parity=int(apply_color_parity),
        bytes=0,
        webp_quality=WEBP_QUALITY,
        webp_method=webp_method_override if webp_method_override is not None else WEBP_METHOD,
        webp_target_bytes=webp_target_bytes_effective if webp_target_bytes_effective is not None else WEBP_TARGET_BYTES,
        gif_colors=GIF_COLORS,
        gif_target_bytes=gif_target_bytes_effective if gif_target_bytes_effective is not None else GIF_TARGET_BYTES,
    )
    return None, "webp", False


def _build_transition_file(payload: bytes, *, filename: str, output_format: str) -> discord.File:
    output = io.BytesIO(payload)
    output.seek(0)
    return discord.File(fp=output, filename=_filename_with_ext(filename, output_format))


def render_tf_split_panel(
    *,
    left_state: Optional[TransformationState],
    right_state: Optional[TransformationState],
    background_user_id: int,
    right_background_user_id: Optional[int] = None,
    center_symbol_name: Optional[str] = None,
    filename: str = "tf_transition.png",
    left_avatar_image: Optional["Image.Image"] = None,
    right_avatar_image: Optional["Image.Image"] = None,
    left_selection_scope: Optional[str] = None,
    right_selection_scope: Optional[str] = None,
    panel_size: Optional[Tuple[int, int]] = None,
    fit_avatar_width: bool = False,
    duplicate_left_background: bool = True,
) -> Optional[discord.File]:
    try:
        from PIL import Image
    except ImportError:
        return None

    rendered = _build_tf_split_image(
        left_state=left_state,
        right_state=right_state,
        background_user_id=background_user_id,
        right_background_user_id=right_background_user_id,
        center_symbol_name=center_symbol_name,
        left_avatar_image=left_avatar_image,
        right_avatar_image=right_avatar_image,
        left_selection_scope=left_selection_scope,
        right_selection_scope=right_selection_scope,
        panel_size=panel_size,
        fit_avatar_width=fit_avatar_width,
        duplicate_left_background=duplicate_left_background,
    )
    if rendered is None:
        return None

    output = io.BytesIO()
    rendered.save(output, format="PNG")
    output.seek(0)
    return discord.File(fp=output, filename=filename)


def _build_tf_split_image(
    *,
    left_state: Optional[TransformationState],
    right_state: Optional[TransformationState],
    background_user_id: int,
    right_background_user_id: Optional[int] = None,
    center_symbol_name: Optional[str] = None,
    left_avatar_image: Optional["Image.Image"] = None,
    right_avatar_image: Optional["Image.Image"] = None,
    left_selection_scope: Optional[str] = None,
    right_selection_scope: Optional[str] = None,
    panel_size: Optional[Tuple[int, int]] = None,
    fit_avatar_width: bool = True,
    duplicate_left_background: bool = True,
) -> Optional["Image.Image"]:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None

    resolved_panel_size = panel_size
    if resolved_panel_size is None and VN_BASE_IMAGE.exists():
        try:
            with Image.open(VN_BASE_IMAGE) as base_image:
                resolved_panel_size = base_image.size
        except OSError:
            resolved_panel_size = (800, 250)
    if resolved_panel_size is None:
        resolved_panel_size = (800, 250)

    width, height = resolved_panel_size
    half_width = width // 2
    base = Image.new("RGBA", resolved_panel_size, (0, 0, 0, 0))

    left_background_path = get_selected_background_path(background_user_id)
    if duplicate_left_background:
        background_layer = compose_background_layer(resolved_panel_size, left_background_path)
        if background_layer is not None:
            left_crop = background_layer.crop((0, 0, half_width, height))
            base.paste(left_crop, (0, 0), left_crop)
            base.paste(left_crop, (half_width, 0), left_crop)
    else:
        left_background_layer = compose_background_layer((half_width, height), left_background_path)
        if left_background_layer is not None:
            base.paste(left_background_layer, (0, 0), left_background_layer)
        right_background_path = get_selected_background_path(
            right_background_user_id if right_background_user_id is not None else background_user_id
        )
        right_background_layer = compose_background_layer((half_width, height), right_background_path)
        if right_background_layer is not None:
            base.paste(right_background_layer, (half_width, 0), right_background_layer)

    left_box = (20, 4, half_width - 20, height)
    right_box = (half_width + 20, 4, width - 20, height)

    if left_avatar_image is None and left_state is not None and left_state.character_name.strip():
        left_avatar_image = compose_state_avatar_image(
            left_state,
            selection_scope=left_selection_scope,
        )
    if right_avatar_image is None and right_state is not None and right_state.character_name.strip():
        right_avatar_image = compose_state_avatar_image(
            right_state,
            selection_scope=right_selection_scope,
        )

    if left_avatar_image is not None:
        _paste_avatar_into_box(
            base,
            left_avatar_image,
            left_box,
            fit_width=fit_avatar_width,
            clip_to_box=fit_avatar_width,
        )
    if right_avatar_image is not None:
        _paste_avatar_into_box(
            base,
            right_avatar_image,
            right_box,
            fit_width=fit_avatar_width,
            clip_to_box=fit_avatar_width,
        )

    draw = ImageDraw.Draw(base)
    divider_left = max(0, half_width - 2)
    divider_right = min(width, half_width + 2)
    draw.rectangle((divider_left, 0, divider_right, height), fill=(0, 0, 0, 255))

    if center_symbol_name:
        symbol_path = Path(center_symbol_name)
        if not symbol_path.is_absolute():
            symbol_path = (BASE_DIR / "vn_assets" / symbol_path).resolve()
        if symbol_path.exists():
            try:
                with Image.open(symbol_path) as symbol_image:
                    symbol = symbol_image.convert("RGBA")
                if symbol.width == width and symbol.height == height:
                    symbol_x = 0
                    symbol_y = 0
                else:
                    symbol_x = max(0, (width - symbol.width) // 2)
                    symbol_y = max(0, (height - symbol.height) // 2)
                base.paste(symbol, (symbol_x, symbol_y), symbol)
            except OSError as exc:
                logger.warning("TF split panel: failed to load center symbol %s: %s", symbol_path, exc)

    return base


def render_tf_split_panel_gif(
    *,
    left_state: Optional[TransformationState],
    right_state: Optional[TransformationState],
    background_user_id: int,
    center_symbol_name: Optional[str] = None,
    filename: str = "tf_transition.gif",
    left_avatar_image: Optional["Image.Image"] = None,
    right_avatar_image: Optional["Image.Image"] = None,
    left_selection_scope: Optional[str] = None,
    right_selection_scope: Optional[str] = None,
    original_hold_ms: int = REROLL_GIF_ORIGINAL_HOLD_MS,
    old_to_silhouette_ms: int = REROLL_GIF_OLD_TO_SILHOUETTE_MS,
    old_silhouette_hold_ms: int = REROLL_GIF_OLD_SILHOUETTE_HOLD_MS,
    both_silhouettes_hold_ms: int = REROLL_GIF_BOTH_SILHOUETTES_HOLD_MS,
    old_silhouette_fade_ms: int = REROLL_GIF_OLD_SILHOUETTE_FADE_MS,
    new_silhouette_hold_ms: int = REROLL_GIF_NEW_SILHOUETTE_HOLD_MS,
    new_reveal_ms: int = REROLL_GIF_NEW_REVEAL_MS,
    final_hold_ms: int = REROLL_GIF_FINAL_HOLD_MS,
    panel_size: Tuple[int, int] = REROLL_PANEL_SIZE,
    fit_avatar_width: bool = False,
) -> Optional[discord.File]:
    try:
        from PIL import Image
    except ImportError:
        return None

    width, height = panel_size
    base = Image.new("RGBA", panel_size, (0, 0, 0, 0))
    background_path = get_selected_background_path(background_user_id)
    background_layer = compose_background_layer(panel_size, background_path)
    if background_layer is not None:
        base = background_layer.copy()

    left_box = (20, 4, width - 20, height)

    if left_avatar_image is None and left_state is not None and left_state.character_name.strip():
        left_avatar_image = compose_state_avatar_image(
            left_state,
            selection_scope=left_selection_scope,
        )
    if right_avatar_image is None and right_state is not None and right_state.character_name.strip():
        right_avatar_image = compose_state_avatar_image(
            right_state,
            selection_scope=right_selection_scope,
        )

    old_avatar_layer = _compose_avatar_layer(
        panel_size,
        left_avatar_image,
        left_box,
        fit_width=fit_avatar_width,
        clip_to_box=fit_avatar_width,
    )
    new_avatar_layer = _compose_avatar_layer(
        panel_size,
        right_avatar_image,
        left_box,
        fit_width=fit_avatar_width,
        clip_to_box=fit_avatar_width,
    )
    flat_old = _flatten_avatar_layer(old_avatar_layer)
    flat_new = _flatten_avatar_layer(new_avatar_layer)
    old_silhouette_layer = _silhouette_from_flattened(flat_old, alpha_scale=1.0) if flat_old else None
    new_silhouette_layer = (
        _silhouette_from_flattened(flat_new, alpha_scale=REROLL_GIF_NEW_OVERLAY_ALPHA / 255.0)
        if flat_new
        else None
    )
    new_silhouette_full_layer = _silhouette_from_flattened(flat_new, alpha_scale=1.0) if flat_new else None
    center_symbol_layer: Optional[Image.Image] = None
    if center_symbol_name:
        symbol_path = Path(center_symbol_name)
        if not symbol_path.is_absolute():
            symbol_path = (BASE_DIR / "vn_assets" / symbol_path).resolve()
        if symbol_path.exists():
            try:
                with Image.open(symbol_path) as symbol_image:
                    center_symbol_layer = symbol_image.convert("RGBA")
            except OSError:
                center_symbol_layer = None

    if old_avatar_layer is None and new_avatar_layer is None:
        return None

    def _composite_frame(*layers: Optional["Image.Image"]) -> "Image.Image":
        frame = _compose_static_background_frame(base, *layers)
        if center_symbol_layer is not None:
            if center_symbol_layer.size == frame.size:
                frame.alpha_composite(center_symbol_layer)
            else:
                symbol_x = max(0, (frame.width - center_symbol_layer.width) // 2)
                symbol_y = max(0, (frame.height - center_symbol_layer.height) // 2)
                frame.alpha_composite(center_symbol_layer, (symbol_x, symbol_y))
        return frame

    frames: list[Image.Image] = []
    durations: list[int] = []

    frames.append(_gif_frame_from_rgba(_composite_frame(old_avatar_layer)))
    durations.append(max(0, int(original_hold_ms)))

    old_to_silhouette_frames = _frame_count_for_duration(old_to_silhouette_ms)
    old_to_silhouette_frame_ms = max(20, int(round(old_to_silhouette_ms / old_to_silhouette_frames)))
    for frame_index in range(1, old_to_silhouette_frames):
        progress = frame_index / (old_to_silhouette_frames - 1)
        fading_old = _layer_with_alpha_scale(old_avatar_layer, alpha_scale=1.0 - progress)
        growing_silhouette = (
            _silhouette_from_flattened(flat_old, alpha_scale=progress) if flat_old is not None else None
        )
        frames.append(_gif_frame_from_rgba(_composite_frame(fading_old, growing_silhouette)))
        durations.append(old_to_silhouette_frame_ms)

    frames.append(_gif_frame_from_rgba(_composite_frame(old_silhouette_layer)))
    durations.append(max(0, int(old_silhouette_hold_ms)))

    frames.append(_gif_frame_from_rgba(_composite_frame(old_silhouette_layer, new_silhouette_layer)))
    durations.append(max(0, int(both_silhouettes_hold_ms)))

    old_fade_frames = _frame_count_for_duration(old_silhouette_fade_ms)
    old_fade_frame_ms = max(20, int(round(old_silhouette_fade_ms / old_fade_frames)))
    for frame_index in range(1, old_fade_frames):
        progress = frame_index / (old_fade_frames - 1)
        fading_old_silhouette = (
            _silhouette_from_flattened(flat_old, alpha_scale=(1.0 - progress)) if flat_old is not None else None
        )
        frames.append(_gif_frame_from_rgba(_composite_frame(fading_old_silhouette, new_silhouette_full_layer)))
        durations.append(old_fade_frame_ms)

    frames.append(_gif_frame_from_rgba(_composite_frame(new_silhouette_full_layer)))
    durations.append(max(0, int(new_silhouette_hold_ms)))

    new_reveal_frames = _frame_count_for_duration(new_reveal_ms)
    new_reveal_frame_ms = max(20, int(round(new_reveal_ms / new_reveal_frames)))
    for frame_index in range(1, new_reveal_frames):
        progress = frame_index / (new_reveal_frames - 1)
        fading_new_silhouette = (
            _silhouette_from_flattened(flat_new, alpha_scale=(1.0 - progress)) if flat_new is not None else None
        )
        frames.append(_gif_frame_from_rgba(_composite_frame(new_avatar_layer, fading_new_silhouette)))
        durations.append(new_reveal_frame_ms)

    frames.append(_gif_frame_from_rgba(_composite_frame(new_avatar_layer)))
    durations.append(max(0, int(final_hold_ms)))

    payload, output_format, _ = _encode_transition_payload(
        frames,
        durations,
        loop=0,
        optimize=True,
        disposal=2,
        shared_palette=True,
        dedupe=True,
        allow_post_opt=True,
        profile="standard",
        label="reroll_transition",
        webp_target_bytes_override=WEBP_TARGET_BYTES_STANDARD,
    )
    if payload is None:
        return None
    return _build_transition_file(payload, filename=filename, output_format=output_format)


def render_swap_transition_panel(
    *,
    left_state: Optional[TransformationState],
    right_state: Optional[TransformationState],
    left_background_user_id: int,
    right_background_user_id: int,
    filename: str = "swap_transition.png",
    center_symbol_name: Optional[str] = None,
    left_avatar_image: Optional["Image.Image"] = None,
    right_avatar_image: Optional["Image.Image"] = None,
    left_selection_scope: Optional[str] = None,
    right_selection_scope: Optional[str] = None,
    panel_size: Optional[Tuple[int, int]] = None,
    fit_avatar_width: bool = True,
) -> Optional[discord.File]:
    return render_tf_split_panel(
        left_state=left_state,
        right_state=right_state,
        background_user_id=left_background_user_id,
        right_background_user_id=right_background_user_id,
        center_symbol_name=center_symbol_name,
        filename=filename,
        left_avatar_image=left_avatar_image,
        right_avatar_image=right_avatar_image,
        left_selection_scope=left_selection_scope,
        right_selection_scope=right_selection_scope,
        panel_size=panel_size,
        fit_avatar_width=fit_avatar_width,
        duplicate_left_background=False,
    )


def render_swap_transition_panel_gif(
    *,
    before_left_state: Optional[TransformationState],
    before_right_state: Optional[TransformationState],
    after_left_state: Optional[TransformationState],
    after_right_state: Optional[TransformationState],
    left_background_user_id: int,
    right_background_user_id: int,
    filename: str = "swap_transition.gif",
    before_left_avatar_image: Optional["Image.Image"] = None,
    before_right_avatar_image: Optional["Image.Image"] = None,
    after_left_avatar_image: Optional["Image.Image"] = None,
    after_right_avatar_image: Optional["Image.Image"] = None,
    left_before_selection_scope: Optional[str] = None,
    right_before_selection_scope: Optional[str] = None,
    left_after_selection_scope: Optional[str] = None,
    right_after_selection_scope: Optional[str] = None,
    panel_size: Tuple[int, int] = (800, 250),
    fit_avatar_width: bool = True,
    initial_hold_ms: int = SWAP_GIF_INITIAL_HOLD_MS,
    ghost_appear_ms: int = SWAP_GIF_GHOST_APPEAR_MS,
    travel_ms: int = SWAP_GIF_TRAVEL_MS,
    ghost_dissolve_ms: int = SWAP_GIF_GHOST_DISSOLVE_MS,
    final_hold_ms: int = SWAP_GIF_FINAL_HOLD_MS,
) -> Optional[discord.File]:
    try:
        from PIL import Image
    except ImportError:
        return None

    width, height = panel_size
    half_width = width // 2
    left_box = (20, 4, half_width - 20, height)
    right_box = (half_width + 20, 4, width - 20, height)

    background_base = _build_tf_split_image(
        left_state=None,
        right_state=None,
        background_user_id=left_background_user_id,
        right_background_user_id=right_background_user_id,
        panel_size=panel_size,
        duplicate_left_background=False,
    )
    if background_base is None:
        return None

    if before_left_avatar_image is None and before_left_state is not None and before_left_state.character_name.strip():
        before_left_avatar_image = compose_state_avatar_image(
            before_left_state,
            selection_scope=left_before_selection_scope,
        )
    if before_right_avatar_image is None and before_right_state is not None and before_right_state.character_name.strip():
        before_right_avatar_image = compose_state_avatar_image(
            before_right_state,
            selection_scope=right_before_selection_scope,
        )
    if after_left_avatar_image is None and after_left_state is not None and after_left_state.character_name.strip():
        after_left_avatar_image = compose_state_avatar_image(
            after_left_state,
            selection_scope=left_after_selection_scope,
        )
    if after_right_avatar_image is None and after_right_state is not None and after_right_state.character_name.strip():
        after_right_avatar_image = compose_state_avatar_image(
            after_right_state,
            selection_scope=right_after_selection_scope,
        )

    old_left_layer = _compose_avatar_layer(
        panel_size,
        before_left_avatar_image,
        left_box,
        fit_width=fit_avatar_width,
        clip_to_box=fit_avatar_width,
    )
    old_right_layer = _compose_avatar_layer(
        panel_size,
        before_right_avatar_image,
        right_box,
        fit_width=fit_avatar_width,
        clip_to_box=fit_avatar_width,
    )
    new_left_layer = _compose_avatar_layer(
        panel_size,
        after_left_avatar_image,
        left_box,
        fit_width=fit_avatar_width,
        clip_to_box=fit_avatar_width,
    )
    new_right_layer = _compose_avatar_layer(
        panel_size,
        after_right_avatar_image,
        right_box,
        fit_width=fit_avatar_width,
        clip_to_box=fit_avatar_width,
    )

    if all(layer is None for layer in (old_left_layer, old_right_layer, new_left_layer, new_right_layer)):
        return None

    before_left_ghost_avatar_image = _compose_swap_identity_avatar_image(
        before_left_state,
        selection_scope=left_before_selection_scope,
    )
    before_right_ghost_avatar_image = _compose_swap_identity_avatar_image(
        before_right_state,
        selection_scope=right_before_selection_scope,
    )
    old_left_ghost_layer = _compose_avatar_layer(
        panel_size,
        before_left_ghost_avatar_image,
        left_box,
        fit_width=fit_avatar_width,
        clip_to_box=fit_avatar_width,
    )
    old_right_ghost_layer = _compose_avatar_layer(
        panel_size,
        before_right_ghost_avatar_image,
        right_box,
        fit_width=fit_avatar_width,
        clip_to_box=fit_avatar_width,
    )

    old_left_ghost = _make_ghost_layer(old_left_ghost_layer, alpha_scale=SWAP_GIF_GHOST_ALPHA / 255.0)
    old_right_ghost = _make_ghost_layer(old_right_ghost_layer, alpha_scale=SWAP_GIF_GHOST_ALPHA / 255.0)

    old_left_anchor = _layer_anchor(old_left_layer)
    old_right_anchor = _layer_anchor(old_right_layer)
    new_left_anchor = _layer_anchor(new_left_layer)
    new_right_anchor = _layer_anchor(new_right_layer)

    def _composite_frame(*layers: Optional["Image.Image"]) -> "Image.Image":
        frame = background_base.copy()
        for layer in layers:
            if layer is not None:
                frame.alpha_composite(layer)
        return frame

    frames: list[Image.Image] = []
    durations: list[int] = []

    frames.append(_gif_frame_from_rgba(_composite_frame(old_left_layer, old_right_layer)))
    durations.append(max(0, int(initial_hold_ms)))

    ghost_appear_frames = _cap_frames(max(2, int(round((max(0, ghost_appear_ms) / 1000.0) * SWAP_GIF_ANIMATION_FPS))), SWAP_GIF_MAX_FRAMES)
    ghost_appear_frame_ms = max(20, int(round(ghost_appear_ms / ghost_appear_frames))) if ghost_appear_ms > 0 else 20
    for frame_index in range(1, ghost_appear_frames):
        progress = _ease_in_out(frame_index / (ghost_appear_frames - 1))
        left_ghost_frame = None
        if old_left_ghost is not None and old_left_anchor is not None:
            left_ghost_frame = _transform_layer(
                old_left_ghost,
                canvas_size=panel_size,
                center_x=old_left_anchor[0] + (SWAP_GIF_GHOST_OFFSET_X * progress),
                bottom_y=old_left_anchor[1],
                scale=1.0,
                alpha_scale=progress,
            )
        right_ghost_frame = None
        if old_right_ghost is not None and old_right_anchor is not None:
            right_ghost_frame = _transform_layer(
                old_right_ghost,
                canvas_size=panel_size,
                center_x=old_right_anchor[0] - (SWAP_GIF_GHOST_OFFSET_X * progress),
                bottom_y=old_right_anchor[1],
                scale=1.0,
                alpha_scale=progress,
            )
        frames.append(_gif_frame_from_rgba(_composite_frame(old_left_layer, old_right_layer, left_ghost_frame, right_ghost_frame)))
        durations.append(ghost_appear_frame_ms)

    travel_frames = _cap_frames(max(2, int(round((max(0, travel_ms) / 1000.0) * SWAP_GIF_ANIMATION_FPS))), SWAP_GIF_MAX_FRAMES)
    travel_frame_ms = max(20, int(round(travel_ms / travel_frames))) if travel_ms > 0 else 20
    left_start_center = old_left_anchor[0] + SWAP_GIF_GHOST_OFFSET_X if old_left_anchor is not None else None
    left_start_bottom = old_left_anchor[1] if old_left_anchor is not None else None
    right_start_center = old_right_anchor[0] - SWAP_GIF_GHOST_OFFSET_X if old_right_anchor is not None else None
    right_start_bottom = old_right_anchor[1] if old_right_anchor is not None else None
    for frame_index in range(travel_frames):
        raw_progress = frame_index / (travel_frames - 1)
        progress = _ease_in_out(raw_progress)
        current_left_body = old_left_layer
        current_right_body = old_right_layer
        left_ghost_frame = None
        if old_left_ghost is not None and left_start_center is not None and left_start_bottom is not None and old_right_anchor is not None:
            left_ghost_frame = _transform_layer(
                old_left_ghost,
                canvas_size=panel_size,
                center_x=(left_start_center * (1.0 - progress)) + (old_right_anchor[0] * progress),
                bottom_y=(left_start_bottom * (1.0 - progress)) + (old_right_anchor[1] * progress),
                scale=1.0,
                alpha_scale=1.0,
            )
        right_ghost_frame = None
        if old_right_ghost is not None and right_start_center is not None and right_start_bottom is not None and old_left_anchor is not None:
            right_ghost_frame = _transform_layer(
                old_right_ghost,
                canvas_size=panel_size,
                center_x=(right_start_center * (1.0 - progress)) + (old_left_anchor[0] * progress),
                bottom_y=(right_start_bottom * (1.0 - progress)) + (old_left_anchor[1] * progress),
                scale=1.0,
                alpha_scale=1.0,
            )
        frames.append(_gif_frame_from_rgba(_composite_frame(current_left_body, current_right_body, left_ghost_frame, right_ghost_frame)))
        durations.append(travel_frame_ms)

    dissolve_frames = _cap_frames(max(2, int(round((max(0, ghost_dissolve_ms) / 1000.0) * SWAP_GIF_ANIMATION_FPS))), SWAP_GIF_MAX_FRAMES)
    dissolve_frame_ms = max(20, int(round(ghost_dissolve_ms / dissolve_frames))) if ghost_dissolve_ms > 0 else 20
    for frame_index in range(dissolve_frames):
        raw_progress = frame_index / (dissolve_frames - 1)
        progress = _ease_in_out(raw_progress)
        left_ghost_frame = None
        if old_left_ghost is not None and old_right_anchor is not None:
            left_ghost_frame = _transform_layer(
                old_left_ghost,
                canvas_size=panel_size,
                center_x=old_right_anchor[0],
                bottom_y=old_right_anchor[1],
                scale=1.0 - ((1.0 - SWAP_GIF_GHOST_END_SCALE) * progress),
                alpha_scale=1.0 - progress,
            )
        right_ghost_frame = None
        if old_right_ghost is not None and old_left_anchor is not None:
            right_ghost_frame = _transform_layer(
                old_right_ghost,
                canvas_size=panel_size,
                center_x=old_left_anchor[0],
                bottom_y=old_left_anchor[1],
                scale=1.0 - ((1.0 - SWAP_GIF_GHOST_END_SCALE) * progress),
                alpha_scale=1.0 - progress,
            )
        frames.append(_gif_frame_from_rgba(_composite_frame(old_left_layer, old_right_layer, left_ghost_frame, right_ghost_frame)))
        durations.append(dissolve_frame_ms)

    frames.append(_gif_frame_from_rgba(_composite_frame(old_left_layer, old_right_layer)))
    durations.append(max(0, int(final_hold_ms)))

    payload, output_format, _ = _encode_transition_payload(
        frames,
        durations,
        loop=1,
        optimize=True,
        disposal=2,
        shared_palette=DEVICE_GIF_USE_SHARED_PALETTE,
        dedupe=True,
        allow_post_opt=True,
        profile="device",
        label="device_swap_transition",
        gif_target_bytes_override=DEVICE_GIF_TARGET_BYTES,
        gif_adaptive_min_colors_override=DEVICE_GIF_ADAPTIVE_MIN_COLORS,
        gif_adaptive_color_step_override=DEVICE_GIF_ADAPTIVE_COLOR_STEP,
        gif_adaptive_max_attempts_override=DEVICE_GIF_ADAPTIVE_MAX_ATTEMPTS,
        webp_target_bytes_override=WEBP_TARGET_BYTES_DEVICE,
    )
    if payload is None:
        return None
    return _build_transition_file(payload, filename=filename, output_format=output_format)


def render_device_swap_transition_panel_gif(
    *,
    before_left_state: Optional[TransformationState],
    before_right_state: Optional[TransformationState],
    after_left_state: Optional[TransformationState],
    after_right_state: Optional[TransformationState],
    left_background_user_id: int,
    right_background_user_id: int,
    filename: str = "device_swap_transition.gif",
    before_left_avatar_image: Optional["Image.Image"] = None,
    before_right_avatar_image: Optional["Image.Image"] = None,
    after_left_avatar_image: Optional["Image.Image"] = None,
    after_right_avatar_image: Optional["Image.Image"] = None,
    left_before_selection_scope: Optional[str] = None,
    right_before_selection_scope: Optional[str] = None,
    left_after_selection_scope: Optional[str] = None,
    right_after_selection_scope: Optional[str] = None,
    panel_size: Tuple[int, int] = (800, 250),
    fit_avatar_width: bool = True,
    initial_hold_ms: int = DEVICE_SWAP_GIF_INITIAL_HOLD_MS,
    effect_ms: int = DEVICE_SWAP_GIF_EFFECT_MS,
    final_hold_ms: int = DEVICE_SWAP_GIF_FINAL_HOLD_MS,
) -> Optional[discord.File]:
    try:
        from PIL import Image
    except ImportError:
        return None

    width, height = panel_size
    half_width = width // 2
    left_box = (20, 4, half_width - 20, height)
    right_box = (half_width + 20, 4, width - 20, height)

    background_base = _build_tf_split_image(
        left_state=None,
        right_state=None,
        background_user_id=left_background_user_id,
        right_background_user_id=right_background_user_id,
        panel_size=panel_size,
        duplicate_left_background=False,
    )
    if background_base is None:
        return None

    if before_left_avatar_image is None and before_left_state is not None and before_left_state.character_name.strip():
        before_left_avatar_image = compose_state_avatar_image(
            before_left_state,
            selection_scope=left_before_selection_scope,
        )
    if before_right_avatar_image is None and before_right_state is not None and before_right_state.character_name.strip():
        before_right_avatar_image = compose_state_avatar_image(
            before_right_state,
            selection_scope=right_before_selection_scope,
        )
    if after_left_avatar_image is None and after_left_state is not None and after_left_state.character_name.strip():
        after_left_avatar_image = compose_state_avatar_image(
            after_left_state,
            selection_scope=left_after_selection_scope,
        )
    if after_right_avatar_image is None and after_right_state is not None and after_right_state.character_name.strip():
        after_right_avatar_image = compose_state_avatar_image(
            after_right_state,
            selection_scope=right_after_selection_scope,
        )

    old_left_layer = _compose_avatar_layer(
        panel_size,
        before_left_avatar_image,
        left_box,
        fit_width=fit_avatar_width,
        clip_to_box=fit_avatar_width,
    )
    old_right_layer = _compose_avatar_layer(
        panel_size,
        before_right_avatar_image,
        right_box,
        fit_width=fit_avatar_width,
        clip_to_box=fit_avatar_width,
    )
    new_left_layer = _compose_avatar_layer(
        panel_size,
        after_left_avatar_image,
        left_box,
        fit_width=fit_avatar_width,
        clip_to_box=fit_avatar_width,
    )
    new_right_layer = _compose_avatar_layer(
        panel_size,
        after_right_avatar_image,
        right_box,
        fit_width=fit_avatar_width,
        clip_to_box=fit_avatar_width,
    )

    if all(layer is None for layer in (old_left_layer, old_right_layer, new_left_layer, new_right_layer)):
        return None

    before_left_ghost_avatar_image = _compose_swap_identity_avatar_image(
        before_left_state,
        selection_scope=left_before_selection_scope,
    )
    before_right_ghost_avatar_image = _compose_swap_identity_avatar_image(
        before_right_state,
        selection_scope=right_before_selection_scope,
    )
    old_left_ghost_layer = _compose_avatar_layer(
        panel_size,
        before_left_ghost_avatar_image,
        left_box,
        fit_width=fit_avatar_width,
        clip_to_box=fit_avatar_width,
    )
    old_right_ghost_layer = _compose_avatar_layer(
        panel_size,
        before_right_ghost_avatar_image,
        right_box,
        fit_width=fit_avatar_width,
        clip_to_box=fit_avatar_width,
    )
    old_left_ghost = _make_ghost_layer(old_left_ghost_layer, alpha_scale=SWAP_GIF_GHOST_ALPHA / 255.0)
    old_right_ghost = _make_ghost_layer(old_right_ghost_layer, alpha_scale=SWAP_GIF_GHOST_ALPHA / 255.0)

    left_region = _union_layer_bbox(old_left_layer, new_left_layer)
    right_region = _union_layer_bbox(old_right_layer, new_right_layer)
    body_regions = [region for region in (left_region, right_region) if region is not None]
    old_left_anchor = _layer_anchor(old_left_layer)
    old_right_anchor = _layer_anchor(old_right_layer)

    def _composite_frame(
        *layers: Optional["Image.Image"],
        effect_progress: float,
        effect_alpha: float,
    ) -> "Image.Image":
        frame = background_base.copy()
        for layer in layers:
            if layer is not None:
                frame.alpha_composite(layer)
        if DEVICE_GIF_INCLUDE_PARTICLES:
            particle_overlay = _build_device_particle_overlay(
                panel_size,
                body_regions,
                progress=effect_progress,
                alpha_scale=effect_alpha,
                seed=(
                    left_background_user_id
                    ^ (right_background_user_id << 1)
                    ^ ((before_left_state.user_id if before_left_state else 0) << 2)
                    ^ ((before_right_state.user_id if before_right_state else 0) << 3)
                ),
            )
            if particle_overlay is not None:
                frame.alpha_composite(particle_overlay)
        return frame

    frames: list[Image.Image] = []
    durations: list[int] = []

    frames.append(_gif_frame_from_rgba(_composite_frame(old_left_layer, old_right_layer, effect_progress=0.0, effect_alpha=0.0)))
    durations.append(max(0, int(initial_hold_ms)))

    ghost_total = SWAP_GIF_GHOST_APPEAR_MS + SWAP_GIF_TRAVEL_MS + SWAP_GIF_GHOST_DISSOLVE_MS
    ghost_appear_ms = int(round(effect_ms * (SWAP_GIF_GHOST_APPEAR_MS / ghost_total))) if ghost_total else 0
    travel_ms = int(round(effect_ms * (SWAP_GIF_TRAVEL_MS / ghost_total))) if ghost_total else 0
    ghost_dissolve_ms = max(0, int(effect_ms) - ghost_appear_ms - travel_ms)
    total_effect_ms = max(1, ghost_appear_ms + travel_ms + ghost_dissolve_ms)
    elapsed_ms = 0

    ghost_appear_frames = _cap_frames(max(2, int(round((max(0, ghost_appear_ms) / 1000.0) * DEVICE_SWAP_GIF_ANIMATION_FPS))), DEVICE_SWAP_GIF_MAX_FRAMES)
    ghost_appear_frame_ms = max(20, int(round(ghost_appear_ms / ghost_appear_frames))) if ghost_appear_ms > 0 else 20
    for frame_index in range(1, ghost_appear_frames):
        raw_progress = frame_index / (ghost_appear_frames - 1)
        progress = _ease_in_out(raw_progress)
        elapsed_ms += ghost_appear_frame_ms
        effect_progress = min(1.0, elapsed_ms / total_effect_ms)
        frames.append(
            _gif_frame_from_rgba(
                _composite_frame(
                    old_left_layer,
                    old_right_layer,
                    _transform_layer(
                        old_left_ghost,
                        canvas_size=panel_size,
                        center_x=old_left_anchor[0] + (SWAP_GIF_GHOST_OFFSET_X * progress),
                        bottom_y=old_left_anchor[1],
                        scale=1.0,
                        alpha_scale=progress,
                    ) if old_left_ghost is not None and old_left_anchor is not None else None,
                    _transform_layer(
                        old_right_ghost,
                        canvas_size=panel_size,
                        center_x=old_right_anchor[0] - (SWAP_GIF_GHOST_OFFSET_X * progress),
                        bottom_y=old_right_anchor[1],
                        scale=1.0,
                        alpha_scale=progress,
                    ) if old_right_ghost is not None and old_right_anchor is not None else None,
                    effect_progress=effect_progress,
                    effect_alpha=1.0,
                )
            )
        )
        durations.append(ghost_appear_frame_ms)

    travel_frames = _cap_frames(max(2, int(round((max(0, travel_ms) / 1000.0) * DEVICE_SWAP_GIF_ANIMATION_FPS))), DEVICE_SWAP_GIF_MAX_FRAMES)
    travel_frame_ms = max(20, int(round(travel_ms / travel_frames))) if travel_ms > 0 else 20
    left_start_center = old_left_anchor[0] + SWAP_GIF_GHOST_OFFSET_X if old_left_anchor is not None else None
    left_start_bottom = old_left_anchor[1] if old_left_anchor is not None else None
    right_start_center = old_right_anchor[0] - SWAP_GIF_GHOST_OFFSET_X if old_right_anchor is not None else None
    right_start_bottom = old_right_anchor[1] if old_right_anchor is not None else None
    for frame_index in range(travel_frames):
        raw_progress = frame_index / (travel_frames - 1)
        progress = _ease_in_out(raw_progress)
        elapsed_ms += travel_frame_ms
        effect_progress = min(1.0, elapsed_ms / total_effect_ms)
        frames.append(
            _gif_frame_from_rgba(
                _composite_frame(
                    old_left_layer,
                    old_right_layer,
                    _transform_layer(
                        old_left_ghost,
                        canvas_size=panel_size,
                        center_x=(left_start_center * (1.0 - progress)) + (old_right_anchor[0] * progress),
                        bottom_y=(left_start_bottom * (1.0 - progress)) + (old_right_anchor[1] * progress),
                        scale=1.0,
                        alpha_scale=1.0,
                    ) if old_left_ghost is not None and left_start_center is not None and left_start_bottom is not None and old_right_anchor is not None else None,
                    _transform_layer(
                        old_right_ghost,
                        canvas_size=panel_size,
                        center_x=(right_start_center * (1.0 - progress)) + (old_left_anchor[0] * progress),
                        bottom_y=(right_start_bottom * (1.0 - progress)) + (old_left_anchor[1] * progress),
                        scale=1.0,
                        alpha_scale=1.0,
                    ) if old_right_ghost is not None and right_start_center is not None and right_start_bottom is not None and old_left_anchor is not None else None,
                    effect_progress=effect_progress,
                    effect_alpha=1.0,
                )
            )
        )
        durations.append(travel_frame_ms)

    dissolve_frames = _cap_frames(max(2, int(round((max(0, ghost_dissolve_ms) / 1000.0) * DEVICE_SWAP_GIF_ANIMATION_FPS))), DEVICE_SWAP_GIF_MAX_FRAMES)
    dissolve_frame_ms = max(20, int(round(ghost_dissolve_ms / dissolve_frames))) if ghost_dissolve_ms > 0 else 20
    for frame_index in range(dissolve_frames):
        raw_progress = frame_index / (dissolve_frames - 1)
        progress = _ease_in_out(raw_progress)
        elapsed_ms += dissolve_frame_ms
        effect_progress = min(1.0, elapsed_ms / total_effect_ms)
        effect_alpha = max(0.0, 1.0 - progress)
        frames.append(
            _gif_frame_from_rgba(
                _composite_frame(
                    old_left_layer,
                    old_right_layer,
                    _transform_layer(
                        old_left_ghost,
                        canvas_size=panel_size,
                        center_x=old_right_anchor[0],
                        bottom_y=old_right_anchor[1],
                        scale=1.0 - ((1.0 - SWAP_GIF_GHOST_END_SCALE) * progress),
                        alpha_scale=1.0 - progress,
                    ) if old_left_ghost is not None and old_right_anchor is not None else None,
                    _transform_layer(
                        old_right_ghost,
                        canvas_size=panel_size,
                        center_x=old_left_anchor[0],
                        bottom_y=old_left_anchor[1],
                        scale=1.0 - ((1.0 - SWAP_GIF_GHOST_END_SCALE) * progress),
                        alpha_scale=1.0 - progress,
                    ) if old_right_ghost is not None and old_left_anchor is not None else None,
                    effect_progress=effect_progress,
                    effect_alpha=effect_alpha,
                )
            )
        )
        durations.append(dissolve_frame_ms)

    frames.append(_gif_frame_from_rgba(_composite_frame(old_left_layer, old_right_layer, effect_progress=1.0, effect_alpha=0.0)))
    durations.append(max(0, int(final_hold_ms)))

    payload, output_format, _ = _encode_transition_payload(
        frames,
        durations,
        loop=1,
        optimize=True,
        disposal=2,
        shared_palette=DEVICE_GIF_USE_SHARED_PALETTE,
        dedupe=True,
        allow_post_opt=True,
        profile="device",
        label="device_reroll_transition",
        gif_target_bytes_override=DEVICE_GIF_TARGET_BYTES,
        gif_adaptive_min_colors_override=DEVICE_GIF_ADAPTIVE_MIN_COLORS,
        gif_adaptive_color_step_override=DEVICE_GIF_ADAPTIVE_COLOR_STEP,
        gif_adaptive_max_attempts_override=DEVICE_GIF_ADAPTIVE_MAX_ATTEMPTS,
        webp_target_bytes_override=WEBP_TARGET_BYTES_DEVICE,
    )
    if payload is None:
        return None
    return _build_transition_file(payload, filename=filename, output_format=output_format)


def render_device_reroll_transition_panel_gif(
    *,
    previous_state: Optional[TransformationState],
    current_state: Optional[TransformationState],
    background_user_id: int,
    filename: str = "device_reroll_transition.gif",
    previous_avatar_image: Optional["Image.Image"] = None,
    current_avatar_image: Optional["Image.Image"] = None,
    previous_selection_scope: Optional[str] = None,
    current_selection_scope: Optional[str] = None,
    panel_size: Tuple[int, int] = REROLL_PANEL_SIZE,
    fit_avatar_width: bool = True,
    initial_hold_ms: int = DEVICE_REROLL_GIF_INITIAL_HOLD_MS,
    effect_ms: int = DEVICE_REROLL_GIF_EFFECT_MS,
    final_hold_ms: int = DEVICE_REROLL_GIF_FINAL_HOLD_MS,
    include_particles: bool = DEVICE_REROLL_INCLUDE_PARTICLES,
    animation_fps: int = DEVICE_REROLL_GIF_ANIMATION_FPS,
    max_frames: int = DEVICE_REROLL_GIF_MAX_FRAMES,
    webp_target_bytes_override: Optional[int] = None,
    gif_target_bytes_override: Optional[int] = None,
    transition_label: str = "device_reroll_transition",
) -> Optional[discord.File]:
    try:
        from PIL import Image
    except ImportError:
        return None

    width, height = panel_size
    avatar_box = (20, 4, width - 20, height)

    base = Image.new("RGBA", panel_size, (0, 0, 0, 0))
    background_path = get_selected_background_path(background_user_id)
    background_layer = compose_background_layer(panel_size, background_path)
    if background_layer is not None:
        base = background_layer.copy()

    if previous_avatar_image is None and previous_state is not None and previous_state.character_name.strip():
        previous_avatar_image = compose_state_avatar_image(
            previous_state,
            selection_scope=previous_selection_scope,
        )
    if current_avatar_image is None and current_state is not None and current_state.character_name.strip():
        current_avatar_image = compose_state_avatar_image(
            current_state,
            selection_scope=current_selection_scope,
        )

    old_avatar_layer = _compose_avatar_layer(
        panel_size,
        previous_avatar_image,
        avatar_box,
        fit_width=fit_avatar_width,
        clip_to_box=fit_avatar_width,
    )
    new_avatar_layer = _compose_avatar_layer(
        panel_size,
        current_avatar_image,
        avatar_box,
        fit_width=fit_avatar_width,
        clip_to_box=fit_avatar_width,
    )
    if old_avatar_layer is None and new_avatar_layer is None:
        return None

    old_silhouette_layer = _make_silhouette_layer(old_avatar_layer, alpha_scale=1.0)
    new_silhouette_layer = _make_silhouette_layer(new_avatar_layer, alpha_scale=1.0)
    body_region = _union_layer_bbox(old_avatar_layer, new_avatar_layer)
    body_regions = [body_region] if body_region is not None else []

    def _composite_frame(
        old_sprite_alpha: float,
        old_silhouette_alpha: float,
        new_silhouette_alpha: float,
        new_sprite_alpha: float,
        effect_progress: float,
        effect_alpha: float,
    ) -> "Image.Image":
        composited_layers = (
            _layer_with_alpha_scale(old_avatar_layer, alpha_scale=old_sprite_alpha),
            _layer_with_alpha_scale(old_silhouette_layer, alpha_scale=old_silhouette_alpha),
            _layer_with_alpha_scale(new_silhouette_layer, alpha_scale=new_silhouette_alpha),
            _layer_with_alpha_scale(new_avatar_layer, alpha_scale=new_sprite_alpha),
        )
        frame = _compose_static_background_frame(base, *composited_layers)
        if include_particles:
            particle_overlay = _build_device_particle_overlay(
                panel_size,
                body_regions,
                progress=effect_progress,
                alpha_scale=effect_alpha,
                seed=(background_user_id ^ ((previous_state.user_id if previous_state else 0) << 2) ^ ((current_state.user_id if current_state else 0) << 3)),
            )
            if particle_overlay is not None:
                frame.alpha_composite(particle_overlay)
        return frame

    frames: list[Image.Image] = []
    durations: list[int] = []

    frames.append(_gif_frame_from_rgba(_composite_frame(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)))
    durations.append(max(0, int(initial_hold_ms)))

    effect_frames = _cap_frames(max(2, int(round((max(0, effect_ms) / 1000.0) * max(6, animation_fps)))), max_frames)
    effect_frame_ms = max(20, int(round(effect_ms / effect_frames))) if effect_ms > 0 else 20
    for frame_index in range(1, effect_frames):
        progress = _ease_in_out(frame_index / (effect_frames - 1))
        if progress < (1.0 / 3.0):
            local_progress = progress / (1.0 / 3.0)
            old_sprite_alpha = max(0.0, 1.0 - local_progress)
            old_silhouette_alpha = min(1.0, local_progress)
            new_silhouette_alpha = 0.0
            new_sprite_alpha = 0.0
        elif progress < (2.0 / 3.0):
            local_progress = (progress - (1.0 / 3.0)) / (1.0 / 3.0)
            old_sprite_alpha = 0.0
            old_silhouette_alpha = max(0.0, 1.0 - local_progress)
            new_silhouette_alpha = min(1.0, local_progress)
            new_sprite_alpha = 0.0
        else:
            local_progress = (progress - (2.0 / 3.0)) / (1.0 / 3.0)
            old_sprite_alpha = 0.0
            old_silhouette_alpha = 0.0
            new_silhouette_alpha = max(0.0, 1.0 - local_progress)
            new_sprite_alpha = min(1.0, local_progress)
        effect_alpha = 0.35 + (0.65 * math.sin(math.pi * progress))
        frames.append(
            _gif_frame_from_rgba(
                _composite_frame(
                    old_sprite_alpha,
                    old_silhouette_alpha,
                    new_silhouette_alpha,
                    new_sprite_alpha,
                    progress,
                    effect_alpha,
                )
            )
        )
        durations.append(effect_frame_ms)

    frames.append(_gif_frame_from_rgba(_composite_frame(0.0, 0.0, 0.0, 1.0, 1.0, 0.0)))
    durations.append(max(0, int(final_hold_ms)))

    payload, output_format, _ = _encode_transition_payload(
        frames,
        durations,
        loop=1,
        optimize=True,
        disposal=2,
        shared_palette=DEVICE_GIF_USE_SHARED_PALETTE,
        dedupe=True,
        allow_post_opt=True,
        profile="device",
        label=transition_label,
        gif_target_bytes_override=gif_target_bytes_override if gif_target_bytes_override is not None else DEVICE_GIF_TARGET_BYTES,
        gif_adaptive_min_colors_override=DEVICE_GIF_ADAPTIVE_MIN_COLORS,
        gif_adaptive_color_step_override=DEVICE_GIF_ADAPTIVE_COLOR_STEP,
        gif_adaptive_max_attempts_override=DEVICE_GIF_ADAPTIVE_MAX_ATTEMPTS,
        webp_target_bytes_override=webp_target_bytes_override if webp_target_bytes_override is not None else WEBP_TARGET_BYTES_DEVICE,
    )
    if payload is None:
        return None
    return _build_transition_file(payload, filename=filename, output_format=output_format)


def render_clone_transition_panel_gif(
    *,
    source_state: Optional[TransformationState],
    before_target_state: Optional[TransformationState],
    after_target_state: Optional[TransformationState],
    source_background_user_id: int,
    target_background_user_id: int,
    filename: str = "clone_transition.gif",
    source_avatar_image: Optional["Image.Image"] = None,
    before_target_avatar_image: Optional["Image.Image"] = None,
    after_target_avatar_image: Optional["Image.Image"] = None,
    source_selection_scope: Optional[str] = None,
    before_target_selection_scope: Optional[str] = None,
    after_target_selection_scope: Optional[str] = None,
    panel_size: Tuple[int, int] = (800, 250),
    fit_avatar_width: bool = True,
    initial_hold_ms: int = DEVICE_SWAP_GIF_INITIAL_HOLD_MS,
    effect_ms: int = DEVICE_SWAP_GIF_EFFECT_MS,
    final_hold_ms: int = DEVICE_SWAP_GIF_FINAL_HOLD_MS,
    include_particles: bool = DEVICE_GIF_INCLUDE_PARTICLES,
) -> Optional[discord.File]:
    try:
        from PIL import Image
    except ImportError:
        return None

    width, height = panel_size
    half_width = width // 2
    left_box = (20, 4, half_width - 20, height)
    right_box = (half_width + 20, 4, width - 20, height)

    background_base = _build_tf_split_image(
        left_state=None,
        right_state=None,
        background_user_id=source_background_user_id,
        right_background_user_id=target_background_user_id,
        panel_size=panel_size,
        duplicate_left_background=False,
    )
    if background_base is None:
        return None

    if source_avatar_image is None and source_state is not None and source_state.character_name.strip():
        source_avatar_image = compose_state_avatar_image(
            source_state,
            selection_scope=source_selection_scope,
        )
    if before_target_avatar_image is None and before_target_state is not None and before_target_state.character_name.strip():
        before_target_avatar_image = compose_state_avatar_image(
            before_target_state,
            selection_scope=before_target_selection_scope,
        )
    if after_target_avatar_image is None and after_target_state is not None and after_target_state.character_name.strip():
        after_target_avatar_image = compose_state_avatar_image(
            after_target_state,
            selection_scope=after_target_selection_scope,
        )

    source_layer = _compose_avatar_layer(
        panel_size,
        source_avatar_image,
        left_box,
        fit_width=fit_avatar_width,
        clip_to_box=fit_avatar_width,
    )
    before_target_layer = _compose_avatar_layer(
        panel_size,
        before_target_avatar_image,
        right_box,
        fit_width=fit_avatar_width,
        clip_to_box=fit_avatar_width,
    )
    after_target_layer = _compose_avatar_layer(
        panel_size,
        after_target_avatar_image,
        right_box,
        fit_width=fit_avatar_width,
        clip_to_box=fit_avatar_width,
    )

    if all(layer is None for layer in (source_layer, before_target_layer, after_target_layer)):
        return None

    before_target_silhouette = _make_silhouette_layer(before_target_layer, alpha_scale=1.0)
    after_target_silhouette = _make_silhouette_layer(after_target_layer, alpha_scale=1.0)
    left_region = _union_layer_bbox(source_layer, source_layer)
    right_region = _union_layer_bbox(before_target_layer, after_target_layer)
    body_regions = [region for region in (left_region, right_region) if region is not None]

    def _composite_frame(
        source_alpha: float,
        before_target_alpha: float,
        before_target_silhouette_alpha: float,
        after_target_silhouette_alpha: float,
        after_target_alpha: float,
        effect_progress: float,
        effect_alpha: float,
    ) -> "Image.Image":
        composited_layers = (
            _layer_with_alpha_scale(source_layer, alpha_scale=source_alpha),
            _layer_with_alpha_scale(before_target_layer, alpha_scale=before_target_alpha),
            _layer_with_alpha_scale(before_target_silhouette, alpha_scale=before_target_silhouette_alpha),
            _layer_with_alpha_scale(after_target_silhouette, alpha_scale=after_target_silhouette_alpha),
            _layer_with_alpha_scale(after_target_layer, alpha_scale=after_target_alpha),
        )
        frame = _compose_static_background_frame(background_base, *composited_layers)
        if include_particles:
            particle_overlay = _build_device_particle_overlay(
                panel_size,
                body_regions,
                progress=effect_progress,
                alpha_scale=effect_alpha,
                seed=(
                    source_background_user_id
                    ^ (target_background_user_id << 1)
                    ^ ((source_state.user_id if source_state else 0) << 2)
                    ^ ((before_target_state.user_id if before_target_state else 0) << 3)
                ),
            )
            if particle_overlay is not None:
                frame.alpha_composite(particle_overlay)
        return frame

    frames: list[Image.Image] = []
    durations: list[int] = []

    frames.append(_gif_frame_from_rgba(_composite_frame(1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)))
    durations.append(max(0, int(initial_hold_ms)))

    effect_frames = _cap_frames(max(2, int(round((max(0, effect_ms) / 1000.0) * DEVICE_SWAP_GIF_ANIMATION_FPS))), DEVICE_SWAP_GIF_MAX_FRAMES)
    effect_frame_ms = max(20, int(round(effect_ms / effect_frames))) if effect_ms > 0 else 20
    for frame_index in range(1, effect_frames):
        progress = _ease_in_out(frame_index / (effect_frames - 1))
        if progress < (1.0 / 3.0):
            local_progress = progress / (1.0 / 3.0)
            before_target_alpha = max(0.0, 1.0 - local_progress)
            before_target_silhouette_alpha = min(1.0, local_progress)
            after_target_silhouette_alpha = 0.0
            after_target_alpha = 0.0
        elif progress < (2.0 / 3.0):
            local_progress = (progress - (1.0 / 3.0)) / (1.0 / 3.0)
            before_target_alpha = 0.0
            before_target_silhouette_alpha = max(0.0, 1.0 - local_progress)
            after_target_silhouette_alpha = min(1.0, local_progress)
            after_target_alpha = 0.0
        else:
            local_progress = (progress - (2.0 / 3.0)) / (1.0 / 3.0)
            before_target_alpha = 0.0
            before_target_silhouette_alpha = 0.0
            after_target_silhouette_alpha = max(0.0, 1.0 - local_progress)
            after_target_alpha = min(1.0, local_progress)
        effect_alpha = max(0.0, 1.0 - max(0.0, (progress - 0.7) / 0.3))
        frames.append(
            _gif_frame_from_rgba(
                _composite_frame(
                    1.0,
                    before_target_alpha,
                    before_target_silhouette_alpha,
                    after_target_silhouette_alpha,
                    after_target_alpha,
                    progress,
                    effect_alpha,
                )
            )
        )
        durations.append(effect_frame_ms)

    frames.append(_gif_frame_from_rgba(_composite_frame(1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0)))
    durations.append(max(0, int(final_hold_ms)))

    payload, output_format, _ = _encode_transition_payload(
        frames,
        durations,
        loop=1,
        optimize=True,
        disposal=2,
        shared_palette=True,
        dedupe=True,
        allow_post_opt=True,
        profile="standard",
        label="swap_transition",
        webp_target_bytes_override=WEBP_TARGET_BYTES_STANDARD,
    )
    if payload is None:
        return None
    return _build_transition_file(payload, filename=filename, output_format=output_format)


def render_mass_swap_transition_panel_gif(
    *,
    filename: str = "mass_swap_transition.webp",
    panel_size: Tuple[int, int] = (800, 250),
    background_number: int = MASS_SWAP_BACKGROUND_NUMBER,
    wander_ms: int = MASS_SWAP_GIF_WANDER_MS,
    exit_ms: int = MASS_SWAP_GIF_EXIT_MS,
    final_hold_ms: int = MASS_SWAP_GIF_FINAL_HOLD_MS,
) -> Optional[discord.File]:
    try:
        from PIL import Image
    except ImportError:
        return None

    width, height = panel_size
    backgrounds = list_background_choices()
    background_path: Optional[Path] = None
    if backgrounds:
        background_index = max(0, min(len(backgrounds) - 1, background_number - 1))
        background_path = backgrounds[background_index]
    background_base = compose_background_layer(panel_size, background_path)
    if background_base is None:
        return None

    ghost_specs = [
        {"name": "John Davis", "track_y": float(height), "start_x": 90.0},
        {"name": "Katrina Morgan", "track_y": float(height), "start_x": 170.0},
        {"name": "Kiyoshi Honda", "track_y": float(height), "start_x": 250.0},
        {"name": "Kyoko Hano", "track_y": float(height), "start_x": 330.0},
        {"name": "Allison Stein", "track_y": float(height), "start_x": 410.0},
        {"name": "Sayaka Sato", "track_y": float(height), "start_x": 490.0},
        {"name": "Irene Virelles", "track_y": float(height), "start_x": 570.0},
        {"name": "Claus Hawkins", "track_y": float(height), "start_x": 650.0},
        {"name": "Zoey Chambers", "track_y": float(height), "start_x": 710.0},
    ]

    prepared_ghosts: list[dict[str, object]] = []
    call_seed = random.randrange(1, 1_000_000_000)
    random.Random(call_seed).shuffle(ghost_specs)
    depth_min = 0.84
    depth_max = 1.12
    depth_span = max(len(ghost_specs) - 1, 1)
    for spec_index, spec in enumerate(ghost_specs):
        avatar_image = compose_game_avatar(str(spec["name"]))
        if avatar_image is None:
            continue
        avatar_rgba = avatar_image.convert("RGBA")
        if avatar_rgba.height > MASS_SWAP_GHOST_MAX_HEIGHT:
            fit_scale = MASS_SWAP_GHOST_MAX_HEIGHT / max(avatar_rgba.height, 1)
            fitted_width = max(1, int(round(avatar_rgba.width * fit_scale)))
            fitted_height = max(1, int(round(avatar_rgba.height * fit_scale)))
            avatar_rgba = avatar_rgba.resize((fitted_width, fitted_height), Image.LANCZOS)
        depth_scale = depth_min + (((depth_max - depth_min) * spec_index) / depth_span)
        if depth_scale != 1.0:
            scaled_width = max(1, int(round(avatar_rgba.width * depth_scale)))
            scaled_height = max(1, int(round(avatar_rgba.height * depth_scale)))
            avatar_rgba = avatar_rgba.resize((scaled_width, scaled_height), Image.LANCZOS)
        ghost_layer = _make_ghost_layer(avatar_rgba, alpha_scale=MASS_SWAP_GHOST_ALPHA / 255.0)
        if ghost_layer is None:
            continue
        rng = random.Random(call_seed + (spec_index * 9973))
        waypoints: list[Tuple[float, float]] = []
        start_x = float(spec["start_x"])
        track_y = float(spec["track_y"])
        current_x = start_x
        waypoints.append((current_x, track_y))
        waypoint_count = rng.randint(3, 6)
        for _ in range(waypoint_count):
            horizontal_step = rng.uniform(70.0, 210.0)
            step_direction = rng.choice((-1.0, 1.0))
            current_x = max(90.0, min(width - 90.0, current_x + (horizontal_step * step_direction)))
            waypoints.append((current_x, track_y))
        edge = rng.choice(("left", "right"))
        if edge == "left":
            exit_point = (-220.0, track_y)
        else:
            exit_point = (width + 220.0, track_y)
        wander_progress_offset = rng.uniform(0.0, 0.22)
        wander_progress_scale = rng.uniform(0.62, 0.92)
        wander_end_progress = min(1.0, wander_progress_offset + wander_progress_scale)
        prepared_ghosts.append(
            {
                "layer": ghost_layer,
                "waypoints": waypoints,
                "exit_point": exit_point,
                "phase_offset": rng.random(),
                "speed_scale": rng.uniform(0.82, 1.18),
                "wiggle_amplitude": rng.uniform(10.0, 34.0),
                "wiggle_cycles": rng.uniform(1.0, 2.8),
                "wander_progress_offset": wander_progress_offset,
                "wander_progress_scale": wander_progress_scale,
                "wander_end_progress": wander_end_progress,
            }
        )

    if not prepared_ghosts:
        return None

    def _position_for_progress(waypoints: Sequence[Tuple[float, float]], progress: float) -> Tuple[float, float]:
        if len(waypoints) == 1:
            return waypoints[0]
        segment_count = max(1, len(waypoints) - 1)
        scaled = max(0.0, min(0.999999, progress)) * segment_count
        segment_index = min(segment_count - 1, int(math.floor(scaled)))
        local_progress = scaled - segment_index
        start_x, start_y = waypoints[segment_index]
        end_x, end_y = waypoints[segment_index + 1]
        eased = _ease_in_out(local_progress)
        return (
            (start_x * (1.0 - eased)) + (end_x * eased),
            (start_y * (1.0 - eased)) + (end_y * eased),
        )

    def _composite_frame(progress: float, *, exiting: bool = False) -> "Image.Image":
        frame = background_base.copy()
        for ghost in prepared_ghosts:
            ghost_layer = ghost["layer"]
            waypoints = ghost["waypoints"]
            exit_point = ghost["exit_point"]
            phase_offset = float(ghost["phase_offset"])
            speed_scale = float(ghost["speed_scale"])
            wiggle_amplitude = float(ghost["wiggle_amplitude"])
            wiggle_cycles = float(ghost["wiggle_cycles"])
            wander_progress_offset = float(ghost["wander_progress_offset"])
            wander_progress_scale = float(ghost["wander_progress_scale"])
            wander_end_progress = float(ghost["wander_end_progress"])
            local_progress = progress
            if not exiting:
                local_progress = min(
                    1.0,
                    wander_progress_offset + (progress * wander_progress_scale * speed_scale),
                )
            current_x, current_y = _position_for_progress(waypoints, local_progress)
            travel_direction_x = 1.0
            if not exiting:
                current_x += math.sin((local_progress + phase_offset) * math.pi * 2.0 * wiggle_cycles) * wiggle_amplitude
                current_x = max(70.0, min(width - 70.0, current_x))
                lookahead_progress = min(1.0, local_progress + 0.02)
                next_x, _ = _position_for_progress(waypoints, lookahead_progress)
                next_x += math.sin((lookahead_progress + phase_offset) * math.pi * 2.0 * wiggle_cycles) * wiggle_amplitude
                travel_direction_x = next_x - current_x
            if exiting:
                last_x, last_y = _position_for_progress(waypoints, wander_end_progress)
                eased = _ease_in_out(progress)
                current_x = (last_x * (1.0 - eased)) + (float(exit_point[0]) * eased)
                current_y = (last_y * (1.0 - eased)) + (float(exit_point[1]) * eased)
                travel_direction_x = float(exit_point[0]) - last_x
            ghost_frame = _transform_layer(
                ghost_layer,  # type: ignore[arg-type]
                canvas_size=panel_size,
                center_x=current_x,
                bottom_y=current_y,
                scale=1.0,
                alpha_scale=1.0 - (0.18 * progress if exiting else 0.0),
                flip_horizontal=travel_direction_x < 0.0,
            )
            if ghost_frame is not None:
                frame.alpha_composite(ghost_frame)
        return frame

    frames: list[Image.Image] = []
    durations: list[int] = []

    wander_frames = _cap_frames(
        max(2, int(round((max(0, wander_ms) / 1000.0) * MASS_SWAP_GIF_ANIMATION_FPS))),
        MASS_SWAP_GIF_MAX_FRAMES,
    )
    wander_frame_ms = max(20, int(round(wander_ms / wander_frames))) if wander_ms > 0 else 20
    for frame_index in range(wander_frames):
        progress = frame_index / (wander_frames - 1)
        frames.append(_gif_frame_from_rgba(_composite_frame(progress, exiting=False)))
        durations.append(wander_frame_ms)

    exit_frames = _cap_frames(
        max(2, int(round((max(0, exit_ms) / 1000.0) * MASS_SWAP_GIF_ANIMATION_FPS))),
        MASS_SWAP_GIF_MAX_FRAMES,
    )
    exit_frame_ms = max(20, int(round(exit_ms / exit_frames))) if exit_ms > 0 else 20
    for frame_index in range(1, exit_frames):
        progress = frame_index / (exit_frames - 1)
        frames.append(_gif_frame_from_rgba(_composite_frame(progress, exiting=True)))
        durations.append(exit_frame_ms)

    frames.append(_gif_frame_from_rgba(background_base.copy()))
    durations.append(max(0, int(final_hold_ms)))

    payload, output_format, _ = _encode_transition_payload(
        frames,
        durations,
        loop=1,
        optimize=False,
        disposal=2,
        shared_palette=False,
        dedupe=True,
        allow_post_opt=False,
        profile="mass",
        label="mass_swap_transition",
        gif_target_bytes_override=GIF_TARGET_BYTES,
        webp_target_bytes_override=WEBP_TARGET_BYTES_MASS,
        webp_min_quality_override=WEBP_MIN_QUALITY_MASS,
        webp_quality_step_override=WEBP_QUALITY_STEP_MASS,
        webp_max_attempts_override=WEBP_MAX_ATTEMPTS_MASS,
    )
    if payload is None:
        return None
    return _build_transition_file(payload, filename=filename, output_format=output_format)


def render_mass_reroll_transition_panel_gif(
    *,
    filename: str = "mass_reroll_transition.webp",
    panel_size: Tuple[int, int] = (800, 250),
    background_number: int = MASS_REROLL_BACKGROUND_NUMBER,
    initial_hold_ms: int = MASS_REROLL_GIF_INITIAL_HOLD_MS,
    transition_ms: int = MASS_REROLL_GIF_TRANSITION_MS,
    final_hold_ms: int = MASS_REROLL_GIF_FINAL_HOLD_MS,
    stage_count: int = MASS_REROLL_STAGE_COUNT,
) -> Optional[discord.File]:
    try:
        from PIL import Image
    except ImportError:
        return None

    width, height = panel_size
    backgrounds = list_background_choices()
    background_path: Optional[Path] = None
    if backgrounds:
        background_index = max(0, min(len(backgrounds) - 1, background_number - 1))
        background_path = backgrounds[background_index]
    background_base = compose_background_layer(panel_size, background_path)
    if background_base is None:
        return None

    silhouette_specs = [
        {"name": "John Davis", "track_y": float(height), "start_x": 90.0},
        {"name": "Katrina Morgan", "track_y": float(height), "start_x": 170.0},
        {"name": "Kiyoshi Honda", "track_y": float(height), "start_x": 250.0},
        {"name": "Kyoko Hano", "track_y": float(height), "start_x": 330.0},
        {"name": "Allison Stein", "track_y": float(height), "start_x": 410.0},
        {"name": "Sayaka Sato", "track_y": float(height), "start_x": 490.0},
        {"name": "Irene Virelles", "track_y": float(height), "start_x": 570.0},
        {"name": "Claus Hawkins", "track_y": float(height), "start_x": 650.0},
        {"name": "Zoey Chambers", "track_y": float(height), "start_x": 710.0},
    ]

    call_seed = random.randrange(1, 1_000_000_000)
    random.Random(call_seed).shuffle(silhouette_specs)
    available_names = [str(spec["name"]) for spec in silhouette_specs]
    depth_min = 0.84
    depth_max = 1.12
    depth_span = max(len(silhouette_specs) - 1, 1)
    rotation_candidates = [step for step in range(1, len(available_names)) if math.gcd(step, len(available_names)) == 1]
    if not rotation_candidates:
        rotation_candidates = [1]
    rotation_step = random.Random(call_seed + 41).choice(rotation_candidates)
    prepared_silhouettes: list[dict[str, object]] = []

    for spec_index, spec in enumerate(silhouette_specs):
        base_name = str(spec["name"])
        stage_names = [base_name]
        for stage_offset in range(1, max(1, int(stage_count)) + 1):
            rotated_index = (spec_index + (stage_offset * rotation_step)) % len(available_names)
            stage_names.append(available_names[rotated_index])

        silhouette_layers: list["Image.Image"] = []
        depth_scale = depth_min + (((depth_max - depth_min) * spec_index) / depth_span)
        for stage_name in stage_names:
            avatar_image = compose_game_avatar(stage_name)
            if avatar_image is None:
                silhouette_layers = []
                break
            avatar_rgba = avatar_image.convert("RGBA")
            if avatar_rgba.height > MASS_SWAP_GHOST_MAX_HEIGHT:
                fit_scale = MASS_SWAP_GHOST_MAX_HEIGHT / max(avatar_rgba.height, 1)
                fitted_width = max(1, int(round(avatar_rgba.width * fit_scale)))
                fitted_height = max(1, int(round(avatar_rgba.height * fit_scale)))
                avatar_rgba = avatar_rgba.resize((fitted_width, fitted_height), Image.LANCZOS)
            if depth_scale != 1.0:
                scaled_width = max(1, int(round(avatar_rgba.width * depth_scale)))
                scaled_height = max(1, int(round(avatar_rgba.height * depth_scale)))
                avatar_rgba = avatar_rgba.resize((scaled_width, scaled_height), Image.LANCZOS)
            silhouette_layer = _make_outlined_silhouette_layer(avatar_rgba, alpha_scale=1.0)
            if silhouette_layer is None:
                silhouette_layers = []
                break
            silhouette_layers.append(silhouette_layer)

        if not silhouette_layers:
            continue

        prepared_silhouettes.append(
            {
                "layers": silhouette_layers,
                "center_x": float(spec["start_x"]),
                "bottom_y": float(spec["track_y"]),
            }
        )

    if not prepared_silhouettes:
        return None

    def _composite_frame(stage_index: int, progress: float) -> "Image.Image":
        frame = background_base.copy()
        eased = _ease_in_out(progress)
        for silhouette in prepared_silhouettes:
            layers = silhouette["layers"]
            current_layer = layers[min(stage_index, len(layers) - 1)]  # type: ignore[index]
            next_layer = layers[min(stage_index + 1, len(layers) - 1)]  # type: ignore[index]
            center_x = float(silhouette["center_x"])
            bottom_y = float(silhouette["bottom_y"])
            current_frame = _transform_layer(
                current_layer,
                canvas_size=panel_size,
                center_x=center_x,
                bottom_y=bottom_y,
                scale=1.0,
                alpha_scale=1.0 - eased,
            )
            if current_frame is not None:
                frame.alpha_composite(current_frame)
            next_frame = _transform_layer(
                next_layer,
                canvas_size=panel_size,
                center_x=center_x,
                bottom_y=bottom_y,
                scale=1.0,
                alpha_scale=eased,
            )
            if next_frame is not None:
                frame.alpha_composite(next_frame)
        return frame

    frames: list[Image.Image] = []
    durations: list[int] = []

    frames.append(_gif_frame_from_rgba(_composite_frame(0, 0.0)))
    durations.append(max(0, int(initial_hold_ms)))

    transition_frames = _cap_frames(
        max(2, int(round((max(0, transition_ms) / 1000.0) * MASS_REROLL_GIF_ANIMATION_FPS))),
        MASS_REROLL_GIF_MAX_FRAMES,
    )
    transition_frame_ms = max(20, int(round(transition_ms / transition_frames))) if transition_ms > 0 else 20
    for stage_index in range(max(1, int(stage_count))):
        for frame_index in range(1, transition_frames):
            progress = frame_index / (transition_frames - 1)
            frames.append(_gif_frame_from_rgba(_composite_frame(stage_index, progress)))
            durations.append(transition_frame_ms)

    frames.append(_gif_frame_from_rgba(_composite_frame(max(0, int(stage_count) - 1), 1.0)))
    durations.append(max(0, int(final_hold_ms)))

    payload, output_format, _ = _encode_transition_payload(
        frames,
        durations,
        loop=1,
        optimize=False,
        disposal=2,
        shared_palette=False,
        dedupe=True,
        allow_post_opt=False,
        profile="mass",
        label="mass_reroll_transition",
        gif_target_bytes_override=GIF_TARGET_BYTES,
        webp_target_bytes_override=WEBP_TARGET_BYTES_MASS,
        webp_min_quality_override=WEBP_MIN_QUALITY_MASS,
        webp_quality_step_override=WEBP_QUALITY_STEP_MASS,
        webp_max_attempts_override=WEBP_MAX_ATTEMPTS_MASS,
    )
    if payload is None:
        return None
    return _build_transition_file(payload, filename=filename, output_format=output_format)


def render_background_transition_panel_gif(
    *,
    state: Optional[TransformationState],
    old_background_path: Optional[Path],
    new_background_path: Optional[Path],
    filename: str = "bg_transition.gif",
    avatar_image: Optional["Image.Image"] = None,
    selection_scope: Optional[str] = None,
    panel_size: Tuple[int, int] = BG_TRANSITION_PANEL_SIZE,
    fit_avatar_width: bool = False,
    initial_hold_ms: int = BG_GIF_INITIAL_HOLD_MS,
    travel_ms: int = BG_GIF_TRAVEL_MS,
    final_hold_ms: int = BG_GIF_FINAL_HOLD_MS,
) -> Optional[discord.File]:
    try:
        from PIL import Image
    except ImportError:
        return None

    try:
        width, height = panel_size
        source_old_path = old_background_path or new_background_path
        source_new_path = new_background_path or old_background_path
        if source_old_path is None or source_new_path is None:
            return None

        old_background_layer = compose_background_layer(panel_size, source_old_path)
        new_background_layer = compose_background_layer(panel_size, source_new_path)
        if old_background_layer is None or new_background_layer is None:
            return None

        if avatar_image is None and state is not None and state.character_name.strip():
            avatar_image = compose_state_avatar_image(state, selection_scope=selection_scope)
        avatar_box = (20, 4, width - 20, height)
        avatar_layer = _compose_avatar_layer(
            panel_size,
            avatar_image,
            avatar_box,
            fit_width=fit_avatar_width,
            clip_to_box=fit_avatar_width,
        )

        strip = Image.new("RGBA", (width * 2, height), (0, 0, 0, 0))
        strip.paste(old_background_layer, (0, 0), old_background_layer)
        strip.paste(new_background_layer, (width, 0), new_background_layer)

        def _frame_at(offset_x: int) -> "Image.Image":
            clamped_offset = max(0, min(width, offset_x))
            frame = strip.crop((clamped_offset, 0, clamped_offset + width, height)).convert("RGBA")
            if avatar_layer is not None:
                frame.alpha_composite(avatar_layer)
            return frame

        frames: list[Image.Image] = []
        durations: list[int] = []
        frames.append(_gif_frame_from_rgba(_frame_at(0)))
        durations.append(max(0, int(initial_hold_ms)))

        travel_frames = _cap_frames(max(2, int(round((max(0, travel_ms) / 1000.0) * BG_GIF_ANIMATION_FPS))), BG_GIF_MAX_FRAMES)
        travel_frame_ms = max(20, int(round(travel_ms / travel_frames))) if travel_ms > 0 else 20
        for frame_index in range(1, travel_frames):
            progress = _ease_in_out(frame_index / (travel_frames - 1))
            offset_x = int(round(width * progress))
            frames.append(_gif_frame_from_rgba(_frame_at(offset_x)))
            durations.append(travel_frame_ms)

        frames.append(_gif_frame_from_rgba(_frame_at(width)))
        durations.append(max(0, int(final_hold_ms)))

        payload, output_format, _ = _encode_transition_payload(
            frames,
            durations,
            loop=1,
            optimize=True,
            disposal=2,
            shared_palette=True,
            dedupe=True,
            allow_post_opt=True,
            profile="standard",
            label="background_transition",
            webp_target_bytes_override=WEBP_TARGET_BYTES_STANDARD,
        )
        if payload is None:
            return None
        return _build_transition_file(payload, filename=filename, output_format=output_format)
    except Exception:
        logger.exception("VN bg transition GIF failed")
        return None


def render_appearance_transition_panel_gif(
    *,
    state: Optional[TransformationState],
    background_user_id: int,
    filename: str = "appearance_transition.gif",
    before_avatar_image: Optional["Image.Image"] = None,
    after_avatar_image: Optional["Image.Image"] = None,
    selection_scope: Optional[str] = None,
    panel_size: Tuple[int, int] = APPEARANCE_TRANSITION_PANEL_SIZE,
    fit_avatar_width: bool = False,
    initial_hold_ms: int = APPEARANCE_GIF_INITIAL_HOLD_MS,
    crossfade_ms: int = APPEARANCE_GIF_CROSSFADE_MS,
    final_hold_ms: int = APPEARANCE_GIF_FINAL_HOLD_MS,
) -> Optional[discord.File]:
    try:
        from PIL import Image, ImageChops, ImageDraw
    except ImportError:
        return None

    background_path = get_selected_background_path(background_user_id)
    base = compose_background_layer(panel_size, background_path)
    if base is None:
        return None

    width, height = panel_size
    avatar_box = (20, 4, width - 20, height)

    if after_avatar_image is None and state is not None and state.character_name.strip():
        after_avatar_image = compose_state_avatar_image(state, selection_scope=selection_scope)

    before_layer = _compose_avatar_layer(
        panel_size,
        before_avatar_image,
        avatar_box,
        fit_width=fit_avatar_width,
        clip_to_box=fit_avatar_width,
    )
    after_layer = _compose_avatar_layer(
        panel_size,
        after_avatar_image,
        avatar_box,
        fit_width=fit_avatar_width,
        clip_to_box=fit_avatar_width,
    )
    if before_layer is None and after_layer is None:
        return None

    before_anchor = _layer_anchor(before_layer)
    after_anchor = _layer_anchor(after_layer)
    shared_anchor = after_anchor or before_anchor
    if shared_anchor is not None:
        if before_layer is not None:
            before_layer = _transform_layer(
                before_layer,
                canvas_size=panel_size,
                center_x=shared_anchor[0],
                bottom_y=shared_anchor[1],
            )
        if after_layer is not None:
            after_layer = _transform_layer(
                after_layer,
                canvas_size=panel_size,
                center_x=shared_anchor[0],
                bottom_y=shared_anchor[1],
            )

    def _frame_for_progress(progress: float) -> "Image.Image":
        composed_layers: list["Image.Image"] = []
        reveal_height = max(0, min(height, int(round(height * progress))))
        if before_layer is not None:
            remaining_before = Image.new("RGBA", panel_size, (0, 0, 0, 0))
            if reveal_height < height:
                bottom_slice = before_layer.crop((0, reveal_height, width, height))
                remaining_before.paste(bottom_slice, (0, reveal_height), bottom_slice)
            composed_layers.append(remaining_before)
        if after_layer is not None and reveal_height > 0:
            revealed_after = Image.new("RGBA", panel_size, (0, 0, 0, 0))
            top_slice = after_layer.crop((0, 0, width, reveal_height))
            revealed_after.paste(top_slice, (0, 0), top_slice)
            composed_layers.append(revealed_after)
        return _compose_static_background_frame(base, *composed_layers)

    frames: list[Image.Image] = []
    durations: list[int] = []

    frames.append(_gif_frame_from_rgba(_frame_for_progress(0.0)))
    durations.append(max(0, int(initial_hold_ms)))

    crossfade_frames = _cap_frames(max(2, int(round((max(0, crossfade_ms) / 1000.0) * APPEARANCE_GIF_ANIMATION_FPS))), APPEARANCE_GIF_MAX_FRAMES)
    crossfade_frame_ms = max(20, int(round(crossfade_ms / crossfade_frames))) if crossfade_ms > 0 else 20
    for frame_index in range(1, crossfade_frames):
        progress = _ease_in_out(frame_index / (crossfade_frames - 1))
        frames.append(_gif_frame_from_rgba(_frame_for_progress(progress)))
        durations.append(crossfade_frame_ms)

    frames.append(_gif_frame_from_rgba(_frame_for_progress(1.0)))
    durations.append(max(0, int(final_hold_ms)))

    payload, output_format, _ = _encode_transition_payload(
        frames,
        durations,
        loop=1,
        optimize=True,
        disposal=2,
        shared_palette=True,
        dedupe=True,
        allow_post_opt=True,
        profile="standard",
        label="appearance_transition",
        webp_target_bytes_override=WEBP_TARGET_BYTES_STANDARD,
    )
    if payload is None:
        return None
    return _build_transition_file(payload, filename=filename, output_format=output_format)


class _PanelBackgroundUnset:
    """Sentinel: use get_selected_background_path(state.user_id) in render_vn_panel."""

    __slots__ = ()


_PANEL_BACKGROUND_UNSET = _PanelBackgroundUnset()


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
    override_avatar_image: Optional["Image.Image"] = None,
    panel_background_path: Union[Optional[Path], _PanelBackgroundUnset] = _PANEL_BACKGROUND_UNSET,
) -> Optional[discord.File]:
    try:
        from PIL import Image, ImageChops, ImageDraw
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
            logger.error(
                "VN panel: custom base image %s missing for %s",
                candidate,
                state.character_name,
            )

    if not base_image_path.exists():
        logger.error("VN panel: base image missing at %s", base_image_path)
        return None

    try:
        with Image.open(base_image_path) as base_image:
            base = base_image.convert("RGBA")
    except OSError as exc:
        logger.error(
            "VN panel: failed to open base image %s",
            base_image_path,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
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
    if override_avatar_image is not None:
        avatar_image = override_avatar_image.copy()
        if avatar_image.mode != "RGBA":
            avatar_image = avatar_image.convert("RGBA")
    if avatar_box and avatar_image is None:
        avatar_image = compose_state_avatar_image(
            state,
            pose_override=gacha_pose_override,
            outfit_override=gacha_outfit_override,
            selection_scope=selection_scope,
        )
    if avatar_image is not None and avatar_box:
        _paste_avatar_into_box(base, avatar_image, avatar_box, use_mask_guide=True)
        pos_x = avatar_box[0]
        pos_y = avatar_box[1]
        logger.debug(
            "VN sprite: pasted avatar for %s at (%s, %s) size %s (base_scale=%s fit_scale=%s)",
            state.character_name,
            pos_x,
            pos_y,
            (avatar_width, avatar_height),
            max(VN_AVATAR_SCALE, 0.01),
            "auto",
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
    # Removed fallback message - empty content is allowed for image-only posts

    if panel_background_path is _PANEL_BACKGROUND_UNSET:
        background_path = get_selected_background_path(state.user_id)
    else:
        background_path = panel_background_path
    background_layer = compose_background_layer((base.width, base.height), background_path)
    if background_layer:
        base = Image.alpha_composite(background_layer, base)
        draw = ImageDraw.Draw(base)

    base_text_font = _load_vn_font(VN_TEXT_FONT_SIZE)
    # Clamp text bounds to panel size first so custom layouts (e.g., Narrator)
    # follow the same emoji sizing/centering rules as default panels.
    text_left = max(0, min(base.width - 1, text_box[0] + text_padding))
    text_top = max(0, min(base.height - 1, text_box[1] + text_padding))
    text_right = max(text_left + 1, min(base.width, text_box[2] - text_padding))
    text_bottom = max(text_top + 1, min(base.height, text_box[3] - text_padding))
    max_width = max(1, text_right - text_left)
    total_height = max(1, text_bottom - text_top)

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
        target_size = min(target_size, VN_BIG_EMOJI_MAX_PX)
        if target_size < VN_TEXT_FONT_SIZE:
            target_size = VN_TEXT_FONT_SIZE
        emoji_target_size = target_size
        base_text_font = _load_vn_font(target_size)
    lines, text_font = _fit_text_segments(draw, segments, base_text_font, max_width, available_height)

    vertical_offset = 0
    if big_emoji_mode:
        block_height = emoji_target_size or getattr(text_font, "size", VN_TEXT_FONT_SIZE)
        vertical_offset = max(0, (available_height - block_height) // 2)
    text_y = text_top + vertical_offset
    text_overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    text_overlay_draw = ImageDraw.Draw(text_overlay)

    if reply_font and reply_line:
        reply_fill = (190, 190, 190, 255)
        text_x = text_left
        text_overlay_draw.text((text_x, text_y), reply_line, fill=reply_fill, font=reply_font)
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
            text_x = text_left + max(0, (max_width - line_width) // 2)
        else:
            text_x = text_left

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
                target_w = max(1, min(int(target_w), max_width))
                target_h = max(1, min(int(target_h), available_height))
                advance = width or target_w
                if big_emoji_mode and emoji_target_size:
                    advance = emoji_target_size
                advance = max(1, min(int(advance), max_width))
                if emoji_img is not None:
                    emoji_source = emoji_img.copy()
                    if emoji_source.mode != "RGBA":
                        emoji_source = emoji_source.convert("RGBA")
                    # Ignore transparent padding so emoji sizing is consistent
                    # across custom layouts (e.g., Narrator) and default panels.
                    alpha_bbox = emoji_source.getchannel("A").getbbox()
                    if alpha_bbox:
                        emoji_source = emoji_source.crop(alpha_bbox)
                    emoji_render = emoji_source.resize((target_w, target_h), Image.LANCZOS)
                    if big_emoji_mode:
                        offset_y = text_y + max(0, (line_height - emoji_render.height) // 2)
                    else:
                        offset_y = text_y + max(0, base_line_height - emoji_render.height)
                    text_overlay.paste(emoji_render, (int(text_x), int(offset_y)), emoji_render)
                else:
                    fallback = segment.get("fallback_text") or text_segment or custom_meta.get("name") or ""
                    if fallback:
                        draw_y = text_y
                        if big_emoji_mode:
                            draw_y = text_y + max(0, (line_height - height) // 2)
                        text_overlay_draw.text((text_x, draw_y), fallback, fill=fill, font=text_font)
                height = max(height, target_h)
                text_x += advance
                continue
            if text_segment:
                draw_y = text_y
                if big_emoji_mode:
                    draw_y = text_y + max(0, (line_height - height) // 2)
                text_overlay_draw.text((text_x, draw_y), text_segment, fill=fill, font=font_segment)
                if segment.get("strike"):
                    strike_y = draw_y + height / 2
                    text_overlay_draw.line(
                        (text_x, strike_y, text_x + width, strike_y),
                        fill=fill,
                        width=max(1, int(height / 10)),
                    )
            text_x += width
        text_y += line_height + line_spacing
    clip_mask = Image.new("L", base.size, 0)
    clip_mask_draw = ImageDraw.Draw(clip_mask)
    clip_left = text_left
    clip_top = text_top
    clip_right = text_right
    clip_bottom = text_bottom
    clip_mask_draw.rectangle((clip_left, clip_top, clip_right - 1, clip_bottom - 1), fill=255)
    clipped_alpha = ImageChops.multiply(text_overlay.getchannel("A"), clip_mask)
    text_overlay.putalpha(clipped_alpha)
    base = Image.alpha_composite(base, text_overlay)
    if border_img is not None:
        base = Image.alpha_composite(base, border_img)

    if _DEVICE_HOLDER_USER_IDS_BY_GUILD.get(state.guild_id) == state.user_id:
        remote_img = assets.get("remote")
        if remote_img is not None:
            if remote_img.size == base.size:
                base = Image.alpha_composite(base, remote_img)
            else:
                remote_x = max(0, base.width - remote_img.width)
                remote_y = 0
                base.alpha_composite(remote_img, (remote_x, remote_y))

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
    if segments is None:
        return {}
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
    "get_background_root",
    "vn_outfit_selection",
    "background_selections",
    "load_outfit_selections",
    "persist_outfit_selections",
    "load_background_selections",
    "persist_background_selections",
    "list_background_choices",
    "get_selected_background_path",
    "set_selected_background",
    "set_device_holder_user_ids_by_guild",
    "compose_background_layer",
    "resolve_panel_layout",
    "compose_game_avatar",
    "compose_state_avatar_image",
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
    "seed_vn_selection_from_scopes",
    "set_character_directory_overrides",
    "strip_urls",
    "prepare_panel_mentions",
    "apply_mention_placeholders",
    "prepare_reply_snippet",
    "parse_discord_formatting",
    "layout_formatted_text",
    "render_vn_panel",
    "render_tf_split_panel",
    "render_tf_split_panel_gif",
    "render_swap_transition_panel",
    "render_swap_transition_panel_gif",
    "render_clone_transition_panel_gif",
    "render_mass_reroll_transition_panel_gif",
    "render_mass_swap_transition_panel_gif",
    "render_background_transition_panel_gif",
    "render_device_swap_transition_panel_gif",
    "render_device_reroll_transition_panel_gif",
    "render_appearance_transition_panel_gif",
    "fetch_avatar_bytes",
    "prepare_custom_emoji_images",
]
