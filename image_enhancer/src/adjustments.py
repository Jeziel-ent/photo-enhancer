"""Manual post-processing adjustments (MVP 2).

Deterministic, bounded, per-pixel/local-neighborhood corrections a user
explicitly dials in via the frontend's Brightness/Contrast/Highlights/
Shadows/Saturation/Detail sliders. This is the ONE implementation used by
every place the app renders an adjusted image -- the live preview endpoint,
the single-image download, and the batch export -- so preview and export can
never drift apart (see backend/adjustment_overlay.py, which is the only
caller of `apply_adjustments`).

Unlike tonal_correction.py's automatic engine (which only ever fixes a
provably broken photo and is otherwise a no-op), every function here is a
direct, always-available creative control: at its default value (0) it is a
byte-identical no-op, and any nonzero value applies exactly the requested
adjustment, bounded so it can never blow out highlights, crush shadows past
recovery, invert hue, or produce a halo/ring (see MAX_* constants and
DETAIL_CLIP_KERNEL below).

These adjustments run strictly AFTER the automatic AI engine (Restormer +
SwinIR-M + tonal_correction.py + the F3/G7-MS/A+ detail stages) -- never
before it, and never re-invoking it. Applying them again to their own output
with all sliders at 0 is guaranteed to be a no-op, so "Reset Adjustments" in
the UI is always safe.
"""

from __future__ import annotations

import cv2
import numpy as np

# Sliders whose UI range is symmetric, [-100, 100], default 0 (0 = no change
# in either direction). `detail` is the only asymmetric one -- see below.
_SYMMETRIC_KEYS = ("brightness", "contrast", "highlights", "shadows", "saturation")

DEFAULT_ADJUSTMENTS = {
    "brightness": 0.0,
    "contrast": 0.0,
    "highlights": 0.0,
    "shadows": 0.0,
    "saturation": 0.0,
    "detail": 0.0,  # [0, 100] only -- a slider for "how much clarity to add
                     # back", not something a user would want to push negative
}

# --------------------------------------------------------------- bounds
# Every MAX_* value below is the actual effect at slider=100 (or slider=-100
# for the symmetric ones) -- i.e. the hard ceiling this module can ever move
# a pixel, regardless of the slider's own already-bounded [-100, 100] range.
MAX_BRIGHTNESS = 60.0          # levels (of 255) added/subtracted, flat
MAX_CONTRAST_GAIN = 0.5        # contrast multiplier ranges [0.5x, 1.5x]
MAX_HIGHLIGHT = 45.0           # levels, tapered onto bright tones only
MAX_SHADOW = 45.0              # levels, tapered onto dark tones only
SHADOW_WEIGHT_THRESHOLD = 110.0    # L below this can be affected by "shadows";
                                     # weight tapers to 0 at L=0's opposite end
HIGHLIGHT_WEIGHT_THRESHOLD = 145.0  # L above this can be affected by "highlights"
MAX_SATURATION_GAIN = 0.7      # HSV S multiplier ranges [0.3x, 1.7x]
MAX_DETAIL_STRENGTH = 1.4      # unsharp-mask multiplier at detail=100
DETAIL_BLUR_SIGMA = 1.0        # matches enhance.py's own A+ stage's sigma
DETAIL_CLIP_KERNEL = 5         # local min/max clip neighborhood (px) -- same
                                 # halo-prevention pattern used throughout this
                                 # codebase (enhance.py's A+/G7-MS stages,
                                 # tonal_correction.py's apply_local_tone_mapping)


def normalize_adjustments(params) -> dict:
    """Coerces an arbitrary (e.g. JSON-decoded, possibly partial or
    malformed) dict into a complete, clamped adjustments dict. Never raises:
    a missing/non-numeric/out-of-range value silently falls back to the
    default (0) or gets clamped into range, so a bad request body can never
    crash the server or produce an out-of-bounds correction."""
    out = dict(DEFAULT_ADJUSTMENTS)
    if not isinstance(params, dict):
        return out
    for key in _SYMMETRIC_KEYS:
        try:
            value = float(params.get(key, 0.0))
        except (TypeError, ValueError):
            value = 0.0
        out[key] = float(np.clip(value, -100.0, 100.0))
    try:
        detail = float(params.get("detail", 0.0))
    except (TypeError, ValueError):
        detail = 0.0
    out["detail"] = float(np.clip(detail, 0.0, 100.0))
    return out


def is_default(params) -> bool:
    """True when `params` (after normalization) is a total no-op."""
    n = normalize_adjustments(params)
    return all(abs(n[key]) < 1e-9 for key in n)


def _shadow_weight(x: np.ndarray) -> np.ndarray:
    """1.0 at L=0, tapering linearly to 0.0 by SHADOW_WEIGHT_THRESHOLD --
    "shadows" only ever touches genuinely dark tones."""
    return np.clip(1.0 - x / SHADOW_WEIGHT_THRESHOLD, 0.0, 1.0)


def _highlight_weight(x: np.ndarray) -> np.ndarray:
    """0.0 up to HIGHLIGHT_WEIGHT_THRESHOLD, tapering linearly to 1.0 by
    L=255 -- "highlights" only ever touches genuinely bright tones."""
    span = max(255.0 - HIGHLIGHT_WEIGHT_THRESHOLD, 1.0)
    return np.clip((x - HIGHLIGHT_WEIGHT_THRESHOLD) / span, 0.0, 1.0)


def _build_lut(n: dict) -> np.ndarray:
    """A single monotonic 256-entry LUT combining brightness/contrast/
    highlights/shadows -- same convention as tonal_correction.py's own
    _build_lut: a pure per-value remap (no neighborhood op), always clamped
    to [0, 255] and forced monotonic non-decreasing, so it can never invert
    tonal order or create a halo/ring at an edge."""
    x = np.arange(256, dtype=np.float64)

    contrast_gain = 1.0 + (n["contrast"] / 100.0) * MAX_CONTRAST_GAIN
    y = (x - 128.0) * contrast_gain + 128.0

    y = y + (n["brightness"] / 100.0) * MAX_BRIGHTNESS
    y = y + (n["shadows"] / 100.0) * MAX_SHADOW * _shadow_weight(x)
    y = y + (n["highlights"] / 100.0) * MAX_HIGHLIGHT * _highlight_weight(x)

    y = np.clip(y, 0.0, 255.0)
    y = np.maximum.accumulate(y)  # belt-and-braces: guarantee monotonicity
    return y.astype(np.uint8)


def _apply_saturation(bgr: np.ndarray, amount: float) -> np.ndarray:
    """HSV S-channel-only multiplicative adjustment -- H and V untouched, so
    hue and brightness are never affected by this control."""
    if amount == 0.0:
        return bgr
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    factor = 1.0 + (amount / 100.0) * MAX_SATURATION_GAIN
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _apply_detail(bgr: np.ndarray, amount: float) -> np.ndarray:
    """A conservative unsharp-mask boost on the Lab L channel, with the
    EXTRA (boosted) contribution hard-clipped to each pixel's own local
    [min, max] neighborhood -- the same halo-prevention pattern this
    codebase's own G7-MS/A+ detail stages and
    tonal_correction.apply_local_tone_mapping already use in production.
    This makes it structurally impossible for this control to invent a
    brightness level, ring, or halo that isn't already present somewhere in
    the pixel's own small neighborhood -- "improves clarity conservatively",
    never a sharpening filter effect."""
    if amount <= 0.0:
        return bgr
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_u8 = lab[:, :, 0]
    l = l_u8.astype(np.float32)
    blur = cv2.GaussianBlur(l, (0, 0), DETAIL_BLUR_SIGMA)
    raw = l - blur

    strength = (amount / 100.0) * MAX_DETAIL_STRENGTH
    k = np.ones((DETAIL_CLIP_KERNEL, DETAIL_CLIP_KERNEL), np.uint8)
    lo = cv2.erode(l_u8, k).astype(np.float32)
    hi = cv2.dilate(l_u8, k).astype(np.float32)
    boosted = np.clip(raw * strength, lo - l, hi - l)

    lab_out = lab.copy()
    lab_out[:, :, 0] = np.clip(l + boosted, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)


def apply_adjustments(bgr: np.ndarray, params=None) -> np.ndarray:
    """Applies the (normalized, bounded) manual adjustments in `params` to
    `bgr` (a BGR uint8 image), in this fixed order: contrast+brightness+
    highlights+shadows (one combined monotonic LUT on Lab L) -> saturation
    (HSV S) -> detail (Lab L, local-clipped unsharp mask). Never mutates the
    input array. A total no-op (a genuine `.copy()`, not merely
    "unchanged-looking") when every value is at its default -- this is what
    makes "Reset Adjustments" and the default-slider-state guarantee exact,
    not approximate. Deterministic: identical input always produces
    identical output, no randomness, no learned model, no external state.
    """
    if bgr is None or not isinstance(bgr, np.ndarray) or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("apply_adjustments expects a BGR image")

    n = normalize_adjustments(params)
    if all(abs(n[key]) < 1e-9 for key in n):
        return bgr.copy()

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lut = _build_lut(n)
    lab_out = lab.copy()
    lab_out[:, :, 0] = cv2.LUT(lab[:, :, 0], lut)
    out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)

    out = _apply_saturation(out, n["saturation"])
    out = _apply_detail(out, n["detail"])
    return out
