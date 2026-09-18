"""Adaptive photographic quality correction.

Conservative, deterministic, bounded image-quality corrections that activate
ONLY when a photo's own statistics show a genuine problem -- and are a total
no-op otherwise. The production entry point, `enhance_photographic_quality`,
is a thin wrapper around `adaptive_tonal_correction` alone: it fixes
genuinely broken exposure/contrast (underexposed, overexposed, low contrast,
crushed shadows, clipped highlights) and is a true no-op on an already
well-exposed photo. It deliberately does NOT touch color at all -- no forced
time-of-day look, no forced color temperature (warm or cool), no forced
white balance, no saturation boost for appearance's sake -- and, per the
MVP 2 "automatic engine + manual controls" product direction (see
docs/MVP2_RESEARCH.md), it also does NOT apply any proactive/stylistic
brightness lift, HDR-like tone mapping, or automatic detail/sharpening
boost: those are creative decisions now left entirely to the user via the
frontend's manual Brightness/Contrast/Highlights/Shadows/Saturation/Detail
sliders (see image_enhancer/src/adjustments.py), which run strictly AFTER
this automatic stage and never re-invoke it. The photo's original color
character is preserved exactly; only its Lab L (luminance) channel is ever
modified by the production path, and only to fix a provably broken photo --
never to make an already-fine photo look a particular way.

`apply_local_tone_mapping`, `correct_color_cast`, and
`boost_natural_saturation` (bounded, independently-tested experiments from
earlier in this project's history -- edge-aware HDR-like tone mapping with a
detail boost, neutral color-cast pull-back, and HSV saturation lift,
respectively) remain defined and independently tested below, but are NOT
part of the default production composition -- see `enhance_photographic_quality`'s
own docstring for why: each is a stylistic "appearance" push, exactly the
category of automatic behavior MVP 2 replaces with explicit, reversible,
user-driven manual controls instead.

None of these can:
  - invent objects, text, faces, or geometry (every stage is a per-pixel or
    per-tile value remap driven by that pixel/tile's own existing content --
    never a generative or content-synthesizing operation)
  - create ringing/halos at hard edges (the tone-mapping stage is
    bilateral/edge-aware specifically to avoid this; the tonal-correction
    LUT is a pure per-value remap with no neighborhood operation at all)
  - invert tonal order or push a value further into an already-clipped
    region than it started (the LUT is always monotonic non-decreasing and
    clamped to [0, 255])
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

SKY_CLIP_L = 250  # near-white/blown threshold used to build "scene" stats below


def _histogram_stats(l_u8: np.ndarray) -> dict:
    hist = cv2.calcHist([l_u8], [0], None, [256], [0, 256]).flatten()
    total = float(hist.sum())
    cdf = np.cumsum(hist)

    def percentile(p: float, cdf_arr=cdf, tot=total) -> float:
        return float(np.searchsorted(cdf_arr, p * tot))

    # "Scene" stats: the same percentiles, but computed with a big blown-out
    # sky/light-source cluster (L >= SKY_CLIP_L) EXCLUDED first. A road/
    # street photo can easily be 30-45% sky, and that sky has no tonal
    # information to recover -- but on the RAW whole-frame histogram it
    # still dominates p50/p95, which can make a bright-sky, merely
    # dim-*foreground* photo get misclassified as "overexposed" and DARKENED
    # by _build_lut's gamma curve (the opposite of what a correctly-exposed
    # street-level scene under a bright/overcast sky needs). Excluding the
    # blown cluster first makes every downstream exposure decision react to
    # the actual scene (buildings/road/people/vehicles), not the sky. Falls
    # back to the raw (non-scene) stats when there's too little non-sky
    # content to be statistically meaningful (e.g. a snow field, a wall of
    # fog) -- see the `scene_frac < 0.05` fallback below.
    sky_hist = hist.copy()
    sky_hist[SKY_CLIP_L:] = 0
    scene_total = float(sky_hist.sum())
    scene_frac = scene_total / total if total else 0.0

    if scene_frac >= 0.05:
        scene_cdf = np.cumsum(sky_hist)

        def scene_percentile(p: float) -> float:
            return float(np.searchsorted(scene_cdf, p * scene_total))

        scene_p1 = scene_percentile(0.01)
        scene_p5 = scene_percentile(0.05)
        scene_p25 = scene_percentile(0.25)
        scene_p50 = scene_percentile(0.50)
        scene_p95 = scene_percentile(0.95)
        scene_p99 = scene_percentile(0.99)
    else:
        scene_p1 = percentile(0.01)
        scene_p5 = percentile(0.05)
        scene_p25 = percentile(0.25)
        scene_p50 = percentile(0.50)
        scene_p95 = percentile(0.95)
        scene_p99 = percentile(0.99)

    return {
        "p1": percentile(0.01),
        "p5": percentile(0.05),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "scene_p1": scene_p1,
        "scene_p5": scene_p5,
        "scene_p25": scene_p25,
        "scene_p50": scene_p50,
        "scene_p95": scene_p95,
        "scene_p99": scene_p99,
        "scene_frac": scene_frac,
        "black_clip_frac": float(hist[:2].sum() / total),
        "white_clip_frac": float(hist[254:].sum() / total),
        "mean": float(l_u8.mean()),
    }


def detect_issues(stats: dict) -> list[str]:
    """Which (if any) of the five problems this histogram shows. Each rule
    requires two independent signals to agree (e.g. a dark median AND dim
    highlights) so a merely dark-toned *subject* on a normally-exposed
    frame doesn't falsely trigger a global exposure fix.

    Uses the SCENE percentiles (blown sky/light-source cluster excluded --
    see _histogram_stats), not the raw whole-frame ones, for exposure/
    contrast direction -- a big bright sky must not by itself decide the
    whole photo is "overexposed" and get darkened. black_clip_frac/
    white_clip_frac (crushed shadows / clipped highlights) still use the
    real, literal whole-frame clipping fractions -- those genuinely mean
    "this many pixels have no recoverable detail," sky included."""
    issues = []
    if stats["scene_p50"] < UNDEREXPOSED_MEDIAN and stats["scene_p95"] < UNDEREXPOSED_P95:
        issues.append("underexposed")
    if stats["scene_p50"] > OVEREXPOSED_MEDIAN and stats["scene_p5"] > OVEREXPOSED_P5:
        issues.append("overexposed")
    if (stats["scene_p99"] - stats["scene_p1"]) < LOW_CONTRAST_RANGE:
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
    # never worsen shadow/highlight clipping. Driven by the SCENE median
    # (sky/blown-highlight cluster excluded -- see _histogram_stats), so a
    # large bright sky above a dim street scene no longer drags the whole
    # frame's gamma toward "overexposed" and darkens the actual subject.
    if "underexposed" in issues or "overexposed" in issues:
        median = float(np.clip(stats["scene_p50"], 1.0, 254.0))
        raw_gamma = np.log(TARGET_MEDIAN / 255.0) / np.log(median / 255.0)
        gamma = float(np.clip(raw_gamma, 1.0 - MAX_GAMMA_DEVIATION, 1.0 + MAX_GAMMA_DEVIATION))
        y = 255.0 * (y / 255.0) ** gamma
        params["gamma"] = gamma

    # Low contrast: a bounded, severity-scaled blend toward a percentile
    # (not min/max) black/white-point stretch. `alpha` -- and therefore how
    # much of the stretch is actually applied -- grows only with how far
    # below the threshold the image's own dynamic range falls.
    if "low_contrast" in issues:
        span = max(stats["scene_p99"] - stats["scene_p1"], 1.0)
        severity = float(np.clip(1.0 - span / LOW_CONTRAST_RANGE, 0.0, 1.0))
        alpha = severity * MAX_STRETCH_BLEND
        bp = max(0.0, stats["scene_p1"] - min(MAX_STRETCH_MARGIN, stats["scene_p1"]))
        wp = min(255.0, stats["scene_p99"] + min(MAX_STRETCH_MARGIN, 255.0 - stats["scene_p99"]))
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
    exposure/contrast problem (see `detect_issues`); returns an unmodified
    copy otherwise. Never mutates the input array. Purely a function of
    `bgr`'s own pixel values -- no randomness, no learned model, no external
    state -- so identical input always produces identical output.
    """
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("adaptive_tonal_correction expects a BGR image")

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0]
    stats = _histogram_stats(l)
    issues = detect_issues(stats)

    # _build_lut also runs the gentle proactive exposure assist (see its own
    # comment) whenever the scene's dark quarter reads below
    # SHADOW_EXPOSURE_P25_TARGET -- independent of `issues`. `params` being
    # empty (nothing in `issues` fired AND the assist didn't either) is what
    # determines the no-op case.
    lut, params = _build_lut(stats, issues)
    if not params:
        meta = {"applied": False, "issues": issues, "stats": stats, "params": {}}
        return (bgr.copy(), meta) if return_meta else bgr.copy()

    l_corrected = cv2.LUT(l, lut)
    lab_out = lab.copy()
    lab_out[:, :, 0] = l_corrected
    out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)

    meta = {"applied": True, "issues": issues, "stats": stats, "params": params}
    return (out, meta) if return_meta else out


# --------------------------------------------------------- color-cast fixup
# adaptive_tonal_correction (above) is deliberately LUMINANCE-only -- see
# this module's own docstring and its own dedicated test
# (image_enhancer/tests/test_tonal_correction.py "strong/saturated colors ->
# no significant color (a/b) shift"). This is a separate, additive, equally
# bounded correction -- never merged into adaptive_tonal_correction itself,
# so that function's own contract and tests stay exactly as they are.
#
# Unlike an earlier version of this module, the target here is PURE
# NEUTRAL, not any particular warm/cool bias -- "keep the photograph's
# original lighting/color character unless correction is needed for natural
# image quality" (this module's own governing rule). This only activates
# on a clearly excessive cast (a materially higher bar than before -- see
# CAST_MEAN_THRESHOLD), and even then only partially pulls it back (never a
# full gray-world correction, which would repaint a legitimately
# warm/cool-toned scene into something artificial).
NEUTRAL_AB = 128.0             # Lab a/b neutral point (OpenCV 8-bit convention)
CAST_MEAN_THRESHOLD = 7.0      # mean |a-128| or |b-128| beyond this = a real,
                                # materially-excessive cast worth correcting
MAX_CHROMA_SHIFT = 14.0        # hard ceiling: at most this many Lab levels of
                                # a/b correction, regardless of how strong the
                                # detected cast is
CORRECTION_STRENGTH = 0.55     # blend factor toward neutral -- deliberately
                                # NOT 1.0 (a full gray-world correction would
                                # erase a photo's real character); only
                                # partially pulls an EXCESSIVE cast back
                                # toward plausible, never repaints a subtle
                                # or legitimate cast.
# A flat, uniform a/b shift applied to EVERY pixel -- including a blown-out
# sky -- can visibly tint that sky (real photo evidence: a sky at mean
# L=236 landed at b=135, clearly cream, not white -- see
# docs/MVP2_RESEARCH.md "Brightness tuning pass"). A genuinely blown
# highlight has no real color information left to "correct" -- it should
# stay close to its own original near-white, not get painted. This
# L-driven weight tapers the cast shift to ~0 by CAST_HIGHLIGHT_PROTECT_END,
# so brights/sky are protected while every darker pixel still gets the full
# correction above.
CAST_HIGHLIGHT_PROTECT_START = 205.0  # L below this: full cast correction
CAST_HIGHLIGHT_PROTECT_END = 245.0    # L above this: ~0 cast correction


def _highlight_protect_weight_lut() -> np.ndarray:
    """256-entry LUT: 1.0 for L <= CAST_HIGHLIGHT_PROTECT_START, tapering
    linearly to 0.0 by L >= CAST_HIGHLIGHT_PROTECT_END. Used to keep the
    color-cast shift away from blown highlights/sky (see
    CAST_HIGHLIGHT_PROTECT_START's own comment)."""
    x = np.arange(256, dtype=np.float32)
    span = max(CAST_HIGHLIGHT_PROTECT_END - CAST_HIGHLIGHT_PROTECT_START, 1.0)
    weight = 1.0 - (x - CAST_HIGHLIGHT_PROTECT_START) / span
    return np.clip(weight, 0.0, 1.0)


def correct_color_cast(bgr: np.ndarray, return_meta: bool = False):
    """Bounded, deterministic correction of a genuinely EXCESSIVE color
    cast toward pure neutral (NEUTRAL_AB) on both Lab axes -- never a
    directional (warm/cool) bias. Only the Lab a/b planes are ever written
    (see adaptive_tonal_correction for the separate exposure/luminance fix)
    -- luminance is not a target of this correction, though OpenCV's 8-bit
    LAB<->BGR conversion is not perfectly invertible at gamut extremes, so a
    strongly saturated/clipped input can see a small (single-digit, out of
    255) incidental L change on the round-trip, the same caveat
    adaptive_tonal_correction's own chroma stability already carries in the
    other direction.

    The full a/b shift is only applied to dark/mid-tone pixels (L <=
    CAST_HIGHLIGHT_PROTECT_START); it tapers to ~0 by
    CAST_HIGHLIGHT_PROTECT_END so a blown sky/highlight is not visibly
    tinted (still a pure per-pixel function of that pixel's own L -- no
    neighborhood/gradient operation, so this still cannot create halos or
    move an edge). Never mutates the input array; a no-op copy when the
    photo's own cast is within CAST_MEAN_THRESHOLD of neutral -- the large
    majority of real photos, by design (mirrors adaptive_tonal_correction's
    own no-op contract: adaptive, not blanket)."""
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("correct_color_cast expects a BGR image")

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    a_mean = float(lab[:, :, 1].mean())
    b_mean = float(lab[:, :, 2].mean())
    a_off = a_mean - NEUTRAL_AB
    b_off = b_mean - NEUTRAL_AB

    meta = {"applied": False, "a_mean": a_mean, "b_mean": b_mean}
    if abs(a_off) < CAST_MEAN_THRESHOLD and abs(b_off) < CAST_MEAN_THRESHOLD:
        return (bgr.copy(), meta) if return_meta else bgr.copy()

    a_shift = float(np.clip(-a_off * CORRECTION_STRENGTH, -MAX_CHROMA_SHIFT, MAX_CHROMA_SHIFT))
    b_shift = float(np.clip(-b_off * CORRECTION_STRENGTH, -MAX_CHROMA_SHIFT, MAX_CHROMA_SHIFT))

    weight_lut = _highlight_protect_weight_lut()
    weight = cv2.LUT(lab[:, :, 0].astype(np.uint8), weight_lut).astype(np.float32)

    lab[:, :, 1] = np.clip(lab[:, :, 1] + a_shift * weight, 0, 255)
    lab[:, :, 2] = np.clip(lab[:, :, 2] + b_shift * weight, 0, 255)
    out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    meta.update(applied=True, a_shift=a_shift, b_shift=b_shift,
                highlight_protected_mean_weight=float(weight.mean()))
    return (out, meta) if return_meta else out


# --------------------------------------------- local tone mapping (HDR-like)
# Classical, deterministic local tone mapping (Durand & Dorsey-style base/
# detail decomposition -- the standard technique behind "HDR-look" tools
# that stays photographic rather than reading as an obvious filter). An
# edge-aware low-pass filter (bilateral) separates the L channel into:
#   - a smooth "base" layer: the large-scale illumination -- the actual
#     shadow-to-highlight SPREAD of the scene
#   - a "detail" layer (original minus base): fine texture/micro-contrast
#     AND residual JPEG-block/sensor noise, indistinguishably
# The base layer's own dynamic range is then compressed toward its own
# mid-point by a BOUNDED, scene-adaptive amount -- shadows lift, highlights
# recover/compress (the actual "HDR tone mapping" step). The detail layer's
# EXTRA (beyond 1x) contribution is added back at a modest, conservative
# boost, but -- critically -- that extra contribution is additionally
# clipped to the pixel's own LOCAL min/max neighborhood (see
# DETAIL_CLIP_KERNEL below), the exact same safety pattern this codebase's
# own G7-MS/A+ detail stages already use in production. Being "edge-aware"
# alone (the bilateral filter) was NOT sufficient on its own to prevent
# halos at a strong boost -- a real photo showed both dark-outline halos at
# building edges and amplified fake texture in flat sky at boost=1.85x with
# no local clip; see docs/MVP2_RESEARCH.md "Halo regression fix". The local
# clip is what actually makes it structurally impossible to invent a
# brightness level, halo, or texture that isn't already present somewhere
# in the pixel's own small neighborhood.
TONE_MAP_SIGMA_COLOR = 16.0    # bilateral filter's range sigma (edge sensitivity
                                # -- how different two neighboring values must be
                                # to be treated as a real edge, not local noise).
                                # Deliberately TIGHT (lowered again from 22, and
                                # 40 before that) -- a large sigmaColor lets the
                                # filter blend across genuinely different
                                # surfaces (e.g. a building silhouette against
                                # sky, or a specular highlight on a glossy car
                                # roof against its own body shadow), producing
                                # visible dark-outline halos and a "waxy"
                                # flattened-highlight look once the detail
                                # layer was boosted (see
                                # docs/MVP2_RESEARCH.md "Halo regression fix"
                                # and "Conservative realism pass" for the real
                                # photo evidence of both).
TONE_MAP_SIGMA_SPACE = 24.0    # bilateral filter's spatial sigma (roughly how
                                # large a neighborhood counts as "local")
# MAX_BASE_COMPRESSION governs the "HDR tone mapping" (shadow lift/highlight
# recovery) strength; WIDE_RANGE_FLOOR/CEIL (below) govern WHEN it activates
# at all. A real, already well-exposed sunny photo can have a genuinely wide
# dynamic range (bright sky + deep vehicle shadows) WITHOUT being a problem
# -- wide range alone is not evidence the photo needs HDR compression, and
# treating it as such produced a visibly flattened/"waxy" look on glossy
# surfaces (a real car's roof highlight measurably compressed toward mid-
# gray on an already-good photo -- see docs/MVP2_RESEARCH.md "Conservative
# realism pass"). Both the ceiling AND the activation range were widened/
# lowered together so this stage stays closer to a no-op on photos that are
# already well-exposed, reserving real compression for genuinely extreme
# dynamic range.
MAX_BASE_COMPRESSION = 0.10    # lowered again from 0.20 (0.32 originally)
# The detail layer is fundamentally unbounded (original minus a smoothed
# base) -- it contains real texture AND residual JPEG-block/sensor noise
# indistinguishably. A previous version of this function amplified it by a
# flat, unbounded 1.85x, which was measurably too aggressive: it produced
# visible cartoon/painted sky texture (amplified compression noise in a
# region with no real detail to recover) and dark-outline halos at building
# edges (the classic unsharp-mask overshoot artifact) on a real user photo
# -- see docs/MVP2_RESEARCH.md "Halo regression fix". Two independent fixes,
# both required (a lower boost alone was not enough on its own -- confirmed
# by testing 1.5x and 1.3x, still haloed at building edges before the local
# clip was added):
#   1. A much lower, conservative boost ceiling.
#   2. A LOCAL MIN/MAX CLIP on the final result -- the exact same safety
#      pattern this codebase's own downstream G7-MS/A+ detail stages already
#      use in production (enhance.py's _final_multiscale_detail /
#      _final_whole_frame_detail: "hard-clipped to something already present
#      in the pixel's own neighborhood", per docs/ENGINE_AUDIT.md's own
#      analysis of why those stages don't hallucinate). This makes it
#      structurally impossible for the boosted detail to push a pixel past a
#      brightness level that doesn't already exist somewhere in its own
#      local neighborhood -- which is exactly what a halo/ring is.
MAX_DETAIL_BOOST = 1.12        # detail layer amplified by at most this factor
                                # (lowered again from 1.20, and 2.0 originally)
DETAIL_CLIP_KERNEL = 5         # local neighborhood size (px) for the min/max
                                # clip -- matches enhance.py's own A+ stage
                                # (_final_whole_frame_detail) exactly
WIDE_RANGE_FLOOR = 130.0       # scene dynamic-range span (p99-p1) at/below
                                # this: base compression stays at 0 (nothing
                                # to compress -- an already well-balanced
                                # scene is left alone). Raised from 110 -- a
                                # normally-exposed sunny photo with real sky
                                # and real shadow routinely spans 110-140
                                # already; that is not itself evidence of an
                                # exposure problem worth compressing.
WIDE_RANGE_CEIL = 280.0        # span at/above this: base compression reaches
                                # its MAX_BASE_COMPRESSION ceiling. Raised
                                # from 235 -- a real, well-exposed, wide-range
                                # sunny photo (measured scene_range=231 on the
                                # real benchmark below) must land well short
                                # of the ceiling, not right at it.


def apply_local_tone_mapping(bgr: np.ndarray, return_meta: bool = False):
    """Bounded, edge-aware local tone mapping: compresses the scene's
    large-scale shadow-to-highlight spread (scaled by how wide that spread
    actually is -- a no-op on an already well-balanced scene) while
    preserving and modestly boosting fine local detail/micro-contrast.
    Chroma is never touched here. Never mutates the input array; always
    returns a genuine copy (the detail boost, even at its default modest
    strength, is treated as "applied" -- see the module docstring's
    "modest, bounded enhancement" framing, distinct from the corrective
    stages above which are true no-ops when nothing is wrong)."""
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("apply_local_tone_mapping expects a BGR image")

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_u8 = lab[:, :, 0]
    l = l_u8.astype(np.float32)

    # d=0 -> diameter derived from sigmaSpace; edge-aware, so flat regions
    # get smoothed together while real edges (billboard frames, vehicle
    # outlines, building edges) are preserved in the base layer -- reduces
    # how much detail/noise leaks across a hard edge, though the local
    # min/max clip below (not this filter alone) is what actually
    # guarantees no halo/ring/fake-texture artifact.
    base = cv2.bilateralFilter(l_u8, d=0, sigmaColor=TONE_MAP_SIGMA_COLOR,
                                sigmaSpace=TONE_MAP_SIGMA_SPACE).astype(np.float32)
    detail = l - base

    stats = _histogram_stats(l_u8)
    span = stats["scene_p99"] - stats["scene_p1"]
    severity = float(np.clip((span - WIDE_RANGE_FLOOR) / (WIDE_RANGE_CEIL - WIDE_RANGE_FLOOR),
                              0.0, 1.0))
    compress_alpha = MAX_BASE_COMPRESSION * severity

    mid = float(np.clip(stats["scene_p50"], 1.0, 254.0))
    base_compressed = base + (mid - base) * compress_alpha

    # A fixed, conservative detail boost (not scene-gated -- "clearer/
    # sharper/more detailed" is an explicit, always-relevant goal here, not
    # a reactive fix), inside MAX_DETAIL_BOOST's own (now much lower)
    # ceiling. Split into the unboosted (1x) contribution -- which just
    # reconstructs `l` plus the base-compression shift, a smooth, large-
    # scale, intentionally NOT locally-clipped change -- and the EXTRA
    # contribution the boost adds on top, which IS locally clipped below
    # (this is the only part capable of a halo/ring/fake-texture artifact).
    detail_boost = 1.0 + (MAX_DETAIL_BOOST - 1.0) * 0.60
    detail_extra = detail * (detail_boost - 1.0)

    # Local min/max clip on the extra boost only (see MAX_DETAIL_BOOST's own
    # comment) -- the exact same delta-clipping pattern as enhance.py's own
    # A+ stage: bound to [lo-l, hi-l] so the extra contribution can never
    # push a pixel past a brightness level that doesn't already exist
    # somewhere in its own local neighborhood. lo/hi come from the CURRENT
    # (pre-tone-mapping) L channel, matching that stage's own convention.
    k = np.ones((DETAIL_CLIP_KERNEL, DETAIL_CLIP_KERNEL), np.uint8)
    lo = cv2.erode(l_u8, k).astype(np.float32)
    hi = cv2.dilate(l_u8, k).astype(np.float32)
    detail_extra_clipped = np.clip(detail_extra, lo - l, hi - l)

    new_l = base_compressed + detail + detail_extra_clipped
    new_l = np.clip(new_l, 0, 255).astype(np.uint8)

    lab_out = lab.copy()
    lab_out[:, :, 0] = new_l
    out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)
    mean_l_delta = float(np.abs(new_l.astype(np.float32) - l).mean())
    meta = {
        "applied": True,
        "scene_range": span,
        "compress_alpha": compress_alpha,
        "detail_boost": detail_boost,
        "mean_l_delta": mean_l_delta,
    }
    return (out, meta) if return_meta else out


# ------------------------------------------------------- natural saturation
# A small, bounded HSV saturation lift for "richer but realistic colors" --
# skipped when the photo is already reasonably saturated (real OOH/street
# photos are frequently slightly desaturated by haze/JPEG compression, but
# an already colorful scene needs no push, and pushing it further risks the
# "plastic/painted" look this module's own governing rule forbids).
SATURATION_MEAN_FLOOR = 90.0   # mean HSV S (0-255) at/above this: no boost
                                # (already reasonably saturated)
MAX_SATURATION_BOOST = 0.18    # at most an 18% multiplicative lift, and only
                                # ever a lift -- this never desaturates


def boost_natural_saturation(bgr: np.ndarray, return_meta: bool = False):
    """Bounded, deterministic saturation lift on the HSV S channel only (H
    and V untouched -- hue/brightness are not targets of this correction).
    A no-op when the photo's own mean saturation is already at or above
    SATURATION_MEAN_FLOOR. Never mutates the input array."""
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("boost_natural_saturation expects a BGR image")

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    s = hsv[:, :, 1]
    s_mean = float(s.mean())

    meta = {"applied": False, "s_mean": s_mean}
    if s_mean >= SATURATION_MEAN_FLOOR:
        return (bgr.copy(), meta) if return_meta else bgr.copy()

    severity = float(np.clip((SATURATION_MEAN_FLOOR - s_mean) / SATURATION_MEAN_FLOOR, 0.0, 1.0))
    boost = 1.0 + MAX_SATURATION_BOOST * severity
    hsv[:, :, 1] = np.clip(s * boost, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    meta.update(applied=True, boost=boost)
    return (out, meta) if return_meta else out


def enhance_photographic_quality(bgr: np.ndarray, return_meta: bool = False):
    """The automatic engine's whole-frame photographic-quality entry point.

    MVP 2 product direction: the automatic engine produces a clean, natural
    enhancement by itself and must NOT force a specific brightness,
    contrast, saturation, warmth, or HDR look -- those are now user-driven
    manual adjustments (see image_enhancer/src/adjustments.py), applied
    client/export-side AFTER this function runs, never before or instead of
    it. So this function is now a thin, explicit wrapper around
    `adaptive_tonal_correction` alone: it fixes genuinely broken exposure/
    contrast (underexposed, overexposed, low contrast, crushed shadows,
    clipped highlights) and is a true no-op on an already well-exposed
    photo -- no proactive brightness lift, no HDR-like local tone mapping,
    no automatic detail/sharpening boost, no color-cast correction, no
    saturation boost. `apply_local_tone_mapping`, `correct_color_cast`, and
    `boost_natural_saturation` remain defined and independently tested below
    (bounded, evidence-driven experiments from earlier in this project's
    history) but are DELIBERATELY not called here -- each is exactly the
    kind of automatic "appearance styling" this product direction replaces
    with explicit manual controls. This function touches ONLY the Lab L
    (luminance) channel, and only when a real defect is detected; chroma is
    always passed through untouched from `bgr` to the final result.
    """
    out, tonal_meta = adaptive_tonal_correction(bgr, return_meta=True)
    meta = {"tonal": tonal_meta}
    return (out, meta) if return_meta else out
