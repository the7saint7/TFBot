"""Animated transition timing, GIF/WebP encode, and color-parity policy.

Optional env (all animated transition types share these knobs):
  TFBOT_TRANSITION_WEBP_MAX_BYTES — animated WebP byte ceiling (default 250000).
  TFBOT_TRANSITION_WEBP_MAX_BYTES_REROLL — alias for the same cap if the primary key is unset.
  TFBOT_TRANSITION_MAX_FRAMES — max frames after global thin (default 40).
  TFBOT_TRANSITION_REROLL_MAX_FRAMES — alias for max frames if the primary key is unset.
  TFBOT_TRANSITION_GIF_FALLBACK — set 1 only to allow GIF after WebP fails (default off).
Most other knobs are constants below.
"""

from __future__ import annotations

import os


def _env_int_clamped(name: str, default: int, lo: int, hi: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        v = default
    else:
        try:
            v = int(raw, 10)
        except ValueError:
            v = default
    return max(lo, min(hi, v))


def _env_int_clamped_dual(primary: str, alias: str, default: int, lo: int, hi: int) -> int:
    raw = os.environ.get(primary, "").strip()
    if not raw:
        raw = os.environ.get(alias, "").strip()
    if not raw:
        v = default
    else:
        try:
            v = int(raw, 10)
        except ValueError:
            v = default
    return max(lo, min(hi, v))


WEBP_ANIMATED_HARD_MAX_BYTES = _env_int_clamped_dual(
    "TFBOT_TRANSITION_WEBP_MAX_BYTES",
    "TFBOT_TRANSITION_WEBP_MAX_BYTES_REROLL",
    250_000,
    50_000,
    2_000_000,
)
# Slightly higher default than 24 so swap/device/bg transitions stay smoother; still env-tunable.
TRANSITION_ENCODE_MAX_TOTAL_FRAMES = _env_int_clamped_dual(
    "TFBOT_TRANSITION_MAX_FRAMES",
    "TFBOT_TRANSITION_REROLL_MAX_FRAMES",
    40,
    8,
    96,
)
TRANSITION_GIF_FALLBACK = os.environ.get("TFBOT_TRANSITION_GIF_FALLBACK", "0").strip().lower() in ("1", "true", "yes", "on")

REROLL_PANEL_SIZE = (800, 250)

# --- Standard reroll (split panel GIF) — short holds + capped frames for wall time ---
REROLL_GIF_ORIGINAL_HOLD_MS = 1080
REROLL_GIF_OLD_TO_SILHOUETTE_MS = 540
REROLL_GIF_OLD_SILHOUETTE_HOLD_MS = 720
REROLL_GIF_BOTH_SILHOUETTES_HOLD_MS = 180
REROLL_GIF_OLD_SILHOUETTE_FADE_MS = 400
REROLL_GIF_NEW_SILHOUETTE_HOLD_MS = 400
REROLL_GIF_NEW_REVEAL_MS = 540
REROLL_GIF_FINAL_HOLD_MS = 180
REROLL_GIF_ANIMATION_FPS = max(8, min(30, 12))
REROLL_GIF_NEW_OVERLAY_ALPHA = 170
REROLL_GIF_MAX_FRAMES = max(0, 20)

# --- Swap (two-character) ---
SWAP_GIF_INITIAL_HOLD_MS = 700
SWAP_GIF_GHOST_APPEAR_MS = 700
SWAP_GIF_TRAVEL_MS = 1850
SWAP_GIF_GHOST_DISSOLVE_MS = 550
SWAP_GIF_FINAL_HOLD_MS = 350
SWAP_GIF_ANIMATION_FPS = max(8, min(30, 12))
SWAP_GIF_GHOST_ALPHA = 150
SWAP_GIF_GHOST_OFFSET_X = 36
SWAP_GIF_GHOST_END_SCALE = 0.2
SWAP_GIF_MAX_FRAMES = max(0, 20)

# --- Device swap / clone (shared composite builder) ---
DEVICE_SWAP_GIF_INITIAL_HOLD_MS = 700
DEVICE_SWAP_GIF_EFFECT_MS = 5000
DEVICE_SWAP_GIF_FINAL_HOLD_MS = 500
DEVICE_SWAP_GIF_ANIMATION_FPS = max(8, min(30, 8))
DEVICE_SWAP_PARTICLE_COUNT = max(0, min(200, 14))
DEVICE_SWAP_PARTICLE_ALPHA = 210
DEVICE_SWAP_WASH_ALPHA = 0
DEVICE_SWAP_PARTICLE_GRID = max(1, min(16, 3))
DEVICE_SWAP_GIF_MAX_FRAMES = max(0, 24)

# --- Device reroll only (faster than clone/swap timeline) ---
DEVICE_REROLL_GIF_INITIAL_HOLD_MS = 550
DEVICE_REROLL_GIF_EFFECT_MS = 1900
DEVICE_REROLL_GIF_FINAL_HOLD_MS = 350
DEVICE_REROLL_GIF_ANIMATION_FPS = max(8, min(30, 12))
DEVICE_REROLL_GIF_MAX_FRAMES = max(0, 18)
DEVICE_REROLL_INCLUDE_PARTICLES = False

DEVICE_GIF_USE_SHARED_PALETTE = False
DEVICE_GIF_TARGET_BYTES = max(0, 20000)
DEVICE_GIF_ADAPTIVE_MIN_COLORS = max(32, min(256, 72))
DEVICE_GIF_ADAPTIVE_COLOR_STEP = max(4, min(64, 16))
DEVICE_GIF_ADAPTIVE_MAX_ATTEMPTS = max(1, min(8, 3))
DEVICE_GIF_INCLUDE_PARTICLES = True

# --- Background / appearance ---
BG_TRANSITION_PANEL_SIZE = (800, 250)
BG_GIF_INITIAL_HOLD_MS = 300
BG_GIF_TRAVEL_MS = 1200
BG_GIF_FINAL_HOLD_MS = 300
BG_GIF_ANIMATION_FPS = max(8, min(30, 10))
BG_GIF_MAX_FRAMES = max(0, 24)

APPEARANCE_TRANSITION_PANEL_SIZE = (800, 250)
APPEARANCE_GIF_INITIAL_HOLD_MS = 250
APPEARANCE_GIF_CROSSFADE_MS = 1400
APPEARANCE_GIF_FINAL_HOLD_MS = 250
APPEARANCE_GIF_ANIMATION_FPS = max(8, min(30, 10))
APPEARANCE_GIF_MAX_FRAMES = max(0, 24)

# --- Mass transitions ---
MASS_SWAP_BACKGROUND_NUMBER = 430
MASS_SWAP_GIF_WANDER_MS = max(200, min(15000, 1650))
MASS_SWAP_GIF_EXIT_MS = max(100, min(10000, 500))
MASS_SWAP_GIF_FINAL_HOLD_MS = max(0, min(5000, 180))
MASS_SWAP_GIF_ANIMATION_FPS = max(8, min(30, 12))
MASS_SWAP_GHOST_ALPHA = 195
MASS_SWAP_GHOST_MAX_HEIGHT = 225
MASS_SWAP_GIF_MAX_FRAMES = max(0, 20)

MASS_REROLL_BACKGROUND_NUMBER = 430
MASS_REROLL_GIF_INITIAL_HOLD_MS = max(0, min(5000, 350))
MASS_REROLL_GIF_TRANSITION_MS = max(100, min(10000, 700))
MASS_REROLL_GIF_FINAL_HOLD_MS = max(0, min(5000, 350))
MASS_REROLL_GIF_ANIMATION_FPS = max(8, min(30, 10))
MASS_REROLL_STAGE_COUNT = max(1, min(5, 2))
MASS_REROLL_GIF_MAX_FRAMES = max(0, 24)

# --- GIF quantize (keys for Pillow LUTs in panels) ---
GIF_DITHER_MODE = "floyd"
GIF_QUANTIZE_METHOD = "fastoctree"
GIF_COLORS = max(32, min(256, 128))
GIF_PREFILTER_SCALE = max(1, min(2, 1))
GIF_PREFILTER_RESAMPLE = "lanczos"
GIF_SHARED_PALETTE = True
GIF_SHARED_PALETTE_SAMPLES = max(3, min(16, 8))
GIF_POST_OPTIMIZER = "off"
GIF_POST_OPTIMIZER_TIMEOUT_MS = max(100, 1800)
GIF_POST_OPTIMIZER_LOSSY = max(0, 0)
GIF_TARGET_BYTES = max(0, 20000)
GIF_ADAPTIVE_MIN_COLORS = max(32, min(256, 80))
GIF_ADAPTIVE_COLOR_STEP = max(4, min(64, 16))
GIF_ADAPTIVE_MAX_ATTEMPTS = max(1, min(8, 3))

# --- Transition encode format ---
TRANSITION_PRIMARY_FORMAT = "webp"
TRANSITION_FALLBACK_FORMAT = "gif"
# If True, allow GIF as primary when TRANSITION_PRIMARY_FORMAT=gif (discouraged).
TRANSITION_ALLOW_GIF_PRIMARY = False

# --- WebP (method 5 = faster than 6; fewer attempts for wall time) ---
WEBP_QUALITY = max(40, min(100, 82))
WEBP_METHOD = max(0, min(6, 5))
WEBP_TARGET_BYTES = max(10_000, int(WEBP_ANIMATED_HARD_MAX_BYTES * 0.82))
WEBP_MIN_QUALITY = max(25, min(100, 74))
WEBP_QUALITY_STEP = max(2, min(20, 4))
WEBP_MAX_ATTEMPTS = max(1, min(8, 2))
WEBP_FAST_TRANSITION_LABELS = frozenset({
    "reroll_transition", "device_reroll_transition", "swap_transition", "device_swap_transition",
    "clone_transition", "body_change_transition", "background_transition", "appearance_transition",
    "mass_swap_transition", "mass_reroll_transition",
})
WEBP_FAST_METHOD = max(0, min(6, 4))
WEBP_FAST_MAX_ATTEMPTS = 2
WEBP_FAST_MIN_QUALITY = max(25, min(100, 76))
WEBP_LOSSLESS = False
WEBP_TARGET_BYTES_STANDARD = max(10_000, int(WEBP_ANIMATED_HARD_MAX_BYTES * 0.82))
WEBP_TARGET_BYTES_DEVICE = WEBP_ANIMATED_HARD_MAX_BYTES
WEBP_TARGET_BYTES_MASS = WEBP_ANIMATED_HARD_MAX_BYTES
WEBP_MIN_QUALITY_MASS = max(25, min(100, 66))
WEBP_QUALITY_STEP_MASS = max(1, min(20, 4))
WEBP_MAX_ATTEMPTS_MASS = max(1, min(8, 4))
WEBP_ALPHA_QUALITY = max(1, min(100, 98))

COLOR_PARITY_ENABLED = True
# Reroll labels omitted: faster WebP encode vs VN color parity (clone/bg/appearance still parity).
COLOR_PARITY_LABELS = frozenset(
    {
        "body_change_transition",
        "clone_transition",
        "background_transition",
        "appearance_transition",
    }
)
COLOR_PARITY_EXCLUDED_LABELS = frozenset(
    {
        "swap_transition",
        "device_swap_transition",
        "mass_swap_transition",
        "mass_reroll_transition",
    }
)

WEBP_TARGET_BYTES_COLOR_PARITY = max(50_000, int(WEBP_ANIMATED_HARD_MAX_BYTES * 0.88))
WEBP_MIN_QUALITY_COLOR_PARITY = max(WEBP_MIN_QUALITY, min(100, 78))
WEBP_ALPHA_QUALITY_COLOR_PARITY = max(WEBP_ALPHA_QUALITY, min(100, 100))
GIF_TARGET_BYTES_COLOR_PARITY = max(0, 20000)
GIF_ADAPTIVE_MIN_COLORS_COLOR_PARITY = max(GIF_ADAPTIVE_MIN_COLORS, min(256, 112))
GIF_ADAPTIVE_COLOR_STEP_COLOR_PARITY = max(2, min(64, min(GIF_ADAPTIVE_COLOR_STEP, 8)))

WEBP_QUALITY_GUARDRAILS = True
WEBP_TARGET_SOFT_RATIO = max(0.5, min(2.0, 1.00))
WEBP_TARGET_HARD_RATIO = max(WEBP_TARGET_SOFT_RATIO, min(3.0, 1.25))
WEBP_TARGET_SOFT_RATIO_MASS = max(0.5, min(2.0, WEBP_TARGET_SOFT_RATIO))
WEBP_TARGET_HARD_RATIO_MASS = max(WEBP_TARGET_SOFT_RATIO_MASS, min(3.0, WEBP_TARGET_HARD_RATIO))
WEBP_MAX_OVERRUN_RATIO = max(1.0, min(3.0, 1.35))
WEBP_MAX_OVERRUN_RATIO_COLOR_PARITY = max(WEBP_MAX_OVERRUN_RATIO, min(3.5, 1.55))
WEBP_MAX_OVERRUN_RATIO_DEVICE = max(1.0, min(2.5, 1.2))

# "off" | backend name — informational only unless wired elsewhere
WEBP_CALIBRATION_BACKEND = "off"

# --- bot.py: static PNG reroll instead of animated transition ---
REROLL_USE_STATIC_TRANSITION = False

# --- bot.py: static two-person swap panel instead of animated WebP (device swap unaffected) ---
SWAP_USE_STATIC_TRANSITION = False
