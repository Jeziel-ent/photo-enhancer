"""Adaptive tonal correction.

A conservative, deterministic, luminance-only global tone remap that
activates ONLY when a photo's histogram shows a genuine exposure/contrast
problem (underexposure, overexposure, low contrast, crushed shadows, or
clipped highlights) -- and is a total no-op otherwise.

Implemented as a single monotonic 256-entry lookup table (LUT) built from
robust histogram percentiles (never simple min/max), applied to the LAB L
channel only via cv2.LUT. Because it is a pure per-value remap (not a
spatial operation), it cannot:
  - shift color (the a/b chroma channels are never touched)
  - create halos (there is no neighborhood/gradient operation at all)
  - invent texture (no sharpening, no generative step, no new frequencies)
  - alter geometry, text, logos, or face structure (every pixel at a given
    input brightness maps to the same output brightness everywhere in the
    frame; spatial layout and edges are completely untouched)
The built LUT is also always monotonic non-decreasing and clamped to
[0, 255], so it cannot invert tonal order or push a value further into an
already-clipped region than it was.
"""

from __future__ import annotations

import cv2
import numpy as np

# --------------------------------------------------------- activation rules
# All on the standard OpenCV 8-bit LAB L scale (0-255). Percentiles (not
# min/max) so a handful of noise/sensor outlier pixels can never trigger or
# skew a decision.
UNDEREXPOSED_MEDIAN = 95.0      # median well below mid-gray...
UNDEREXPOSED_P95 = 200.0        # ...and even the brightest 5% is still dim
OVEREXPOSED_MEDIAN = 175.0      # median well above mid-gray...
OVEREXPOSED_P5 = 55.0           # ...and even the darkest 5% is fairly bright
LOW_CONTRAST_RANGE = 110.0      # robust dynamic range (p99-p1) below this
CRUSHED_SHADOW_FRAC = 0.006     # >0.6% of all pixels pinned at L<=1
CLIPPED_HIGHLIGHT_FRAC = 0.006  # >0.6% of all pixels pinned at L>=254

# ------------------------------------------------------- correction limits
# Every knob below is a hard ceiling on how far this module may ever move a
# pixel, regardless of how severe the detected issue is.
MAX_GAMMA_DEVIATION = 0.25      # gamma stays within [0.75, 1.25]
TARGET_MEDIAN = 118.0           # conventional photographic mid-gray target
MAX_STRETCH_BLEND = 0.6         # contrast stretch blended in at most 60%
MAX_STRETCH_MARGIN = 40.0       # black/white points pulled at most this far
MAX_SHADOW_LIFT = 14.0          # levels (of 255), smoothly localized near 0
MAX_HIGHLIGHT_PULL = 14.0       # levels (of 255), smoothly localized near 255
_EDGE_FALLOFF_SIGMA = 24.0      # how quickly the shadow/highlight fix fades


def _histogram_stats(l_u8: np.ndarray) -> dict:
    hist = cv2.calcHist([l_u8], [0], None, [256], [0, 256]).flatten()
    total = float(hist.sum())
    cdf = np.cumsum(hist)

    def percentile(p: float) -> float:
        return float(np.searchsorted(cdf, p * total))

    return {
        "p1": percentile(0.01),
        "p5": percentile(0.05),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "black_clip_frac": float(hist[:2].sum() / total),
        "white_clip_frac": float(hist[254:].sum() / total),
        "mean": float(l_u8.mean()),
    }


def detect_issues(stats: dict) -> list[str]:
    """Which (if any) of the five problems this histogram shows. Each rule
    requires two independent signals to agree (e.g. a dark median AND dim
    highlights) so a merely dark-toned *subject* on a normally-exposed
    frame doesn't falsely trigger a global exposure fix."""
    issues = []
    if stats["p50"] < UNDEREXPOSED_MEDIAN and stats["p95"] < UNDEREXPOSED_P95:
        issues.append("underexposed")
    if stats["p50"] > OVEREXPOSED_MEDIAN and stats["p5"] > OVEREXPOSED_P5:
        issues.append("overexposed")
    if (stats["p99"] - stats["p1"]) < LOW_CONTRAST_RANGE:
        issues.append("low_contrast")
    if stats["black_clip_frac"] > CRUSHED_SHADOW_FRAC:
        issues.append("crushed_shadows")
    if stats["white_clip_frac"] > CLIPPED_HIGHLIGHT_FRAC:
        issues.append("clipped_highlights")
    return issues


def _build_lut(stats: dict, issues: list[str]) -> tuple[np.ndarray, dict]:
    x = np.arange(256, dtype=np.float64)
    y = x.copy()
    params: dict = {}

    # Exposure: a bounded gamma curve. Preserves 0 and 255 exactly (x**gamma
    # is 0 at x=0 and 1 at x=1), so it can only reshape midtones -- it can
    # never worsen shadow/highlight clipping.
    if "underexposed" in issues or "overexposed" in issues:
        median = float(np.clip(stats["p50"], 1.0, 254.0))
        raw_gamma = np.log(TARGET_MEDIAN / 255.0) / np.log(median / 255.0)
        gamma = float(np.clip(raw_gamma, 1.0 - MAX_GAMMA_DEVIATION, 1.0 + MAX_GAMMA_DEVIATION))
        y = 255.0 * (y / 255.0) ** gamma
        params["gamma"] = gamma

    # Low contrast: a bounded, severity-scaled blend toward a percentile
    # (not min/max) black/white-point stretch. `alpha` -- and therefore how
    # much of the stretch is actually applied -- grows only with how far
    # below the threshold the image's own dynamic range falls.
    if "low_contrast" in issues:
        span = max(stats["p99"] - stats["p1"], 1.0)
        severity = float(np.clip(1.0 - span / LOW_CONTRAST_RANGE, 0.0, 1.0))
        alpha = severity * MAX_STRETCH_BLEND
        bp = max(0.0, stats["p1"] - min(MAX_STRETCH_MARGIN, stats["p1"]))
        wp = min(255.0, stats["p99"] + min(MAX_STRETCH_MARGIN, 255.0 - stats["p99"]))
        wp = max(wp, bp + 1.0)
        stretched = np.clip((y - bp) / (wp - bp) * 255.0, 0.0, 255.0)
        y = (1.0 - alpha) * y + alpha * stretched
        params.update(stretch_alpha=alpha, stretch_black_point=bp, stretch_white_point=wp)

    # Crushed shadows: a smooth, bounded lift confined to near-black values
    # only (a Gaussian falloff in VALUE space, not a spatial one) -- it
    # cannot touch midtones/highlights and cannot itself blow out anything,
    # since it is capped at MAX_SHADOW_LIFT and still passes through the
    # final monotonic clamp below.
    if "crushed_shadows" in issues:
        severity = float(np.clip(stats["black_clip_frac"] / (CRUSHED_SHADOW_FRAC * 4.0), 0.0, 1.0))
        lift = MAX_SHADOW_LIFT * severity
        y = y + lift * np.exp(-(x / _EDGE_FALLOFF_SIGMA) ** 2)
        params["shadow_lift"] = lift

    # Clipped highlights: the symmetric, equally bounded counterpart.
    if "clipped_highlights" in issues:
        severity = float(np.clip(stats["white_clip_frac"] / (CLIPPED_HIGHLIGHT_FRAC * 4.0), 0.0, 1.0))
        pull = MAX_HIGHLIGHT_PULL * severity
        y = y - pull * np.exp(-((255.0 - x) / _EDGE_FALLOFF_SIGMA) ** 2)
        params["highlight_pull"] = pull

    y = np.clip(y, 0.0, 255.0)
    y = np.maximum.accumulate(y)  # belt-and-braces: guarantee monotonicity
    return y.astype(np.uint8), params


def adaptive_tonal_correction(bgr: np.ndarray, return_meta: bool = False):
    """Applies bounded, deterministic tonal correction to `bgr` (a BGR
    uint8 image) ONLY when its luminance histogram shows a genuine
    exposure/contrast problem; returns an unmodified copy otherwise. Never
    mutates the input array. Purely a function of `bgr`'s own pixel values
    -- no randomness, no learned model, no external state -- so identical
    input always produces identical output.
    """
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("adaptive_tonal_correction expects a BGR image")

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0]
    stats = _histogram_stats(l)
    issues = detect_issues(stats)

    if not issues:
        meta = {"applied": False, "issues": [], "stats": stats, "params": {}}
        return (bgr.copy(), meta) if return_meta else bgr.copy()

    lut, params = _build_lut(stats, issues)
    l_corrected = cv2.LUT(l, lut)
    lab_out = lab.copy()
    lab_out[:, :, 0] = l_corrected
    out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)

    meta = {"applied": True, "issues": issues, "stats": stats, "params": params}
    return (out, meta) if return_meta else out
