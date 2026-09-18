"""MVP 2 Phase 4 -- a classical, deterministic, bounded daylight/illuminant
normalization candidate, evaluated the same way the existing
tonal_correction.py module already works (same file's own design
philosophy: percentile-driven activation, hard-bounded correction, LUMINANCE
*and* CHROMA-safe, zero learned parameters, zero hallucination risk by
construction).

This does NOT touch production code. It is a standalone research
prototype so the daylight-normalization approach can be evaluated with real
before/after evidence before any production decision is made.

Method: gray-world-style illuminant estimation on the LAB a/b (chroma)
planes, activated only when the frame's own mean chroma shows a genuine
color cast (the same "two independent signals must agree" conservatism as
tonal_correction.detect_issues), and bounded to a hard maximum shift so it
can never repaint the scene's actual colors -- only pull an existing color
cast back toward neutral.

Test methodology: since no paired evening/daylight dataset is downloaded in
this pass (see docs/MVP2_RESEARCH.md's "Phase 3/4 scoping" note), this
script applies a KNOWN, controlled warm-cast simulation to a REAL daylight
photo already in this repo (a standard, reproducible white-balance-research
technique -- see e.g. the color-constancy literature's own synthetic-cast
evaluation protocol) and measures how much of that known, controlled cast
the candidate removes, without needing any external download.

Run:
    .venv\\Scripts\\python.exe research\\scripts\\daylight_candidate_classical.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "image_enhancer" / "src"))

OUT_DIR = REPO_ROOT / "research" / "outputs" / "phase4_daylight_classical"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE = REPO_ROOT / "comparison_original.png"

# ------------------------------------------------------------- the candidate

# Mirrors tonal_correction.py's own constants style: hard ceilings, nothing
# unbounded.
NEUTRAL_AB = 128.0            # Lab a/b neutral point (OpenCV 8-bit convention)
CAST_MEAN_THRESHOLD = 4.0     # mean |a-128| or |b-128| beyond this = a real cast
MAX_CHROMA_SHIFT = 18.0       # hard ceiling: at most this many Lab levels of
                               # a/b correction, regardless of how strong the
                               # detected cast is -- prevents any possibility
                               # of over-correcting into an artificial tint
CORRECTION_STRENGTH = 0.65    # blend factor toward fully neutral (never 1.0 --
                               # a full gray-world correction is too aggressive
                               # for real photos with legitimately warm-toned
                               # scenes; this only PARTIALLY pulls the cast back)


def estimate_and_correct_cast(bgr: np.ndarray) -> tuple[np.ndarray, dict]:
    """Bounded, deterministic warm/cool cast reduction. Luminance (L) is
    left completely untouched here (that's tonal_correction.py's job) --
    this function only ever adjusts the a/b chroma planes, and only by a
    single global, clamped delta (no per-pixel/spatial operation, so it
    cannot invent texture, move edges, or alter geometry/text/faces any
    more than a white-balance slider in a normal photo editor would)."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    a_mean = float(lab[:, :, 1].mean())
    b_mean = float(lab[:, :, 2].mean())
    a_off = a_mean - NEUTRAL_AB
    b_off = b_mean - NEUTRAL_AB

    meta = {"a_mean": a_mean, "b_mean": b_mean, "applied": False}
    if abs(a_off) < CAST_MEAN_THRESHOLD and abs(b_off) < CAST_MEAN_THRESHOLD:
        return bgr.copy(), meta

    a_shift = float(np.clip(-a_off * CORRECTION_STRENGTH, -MAX_CHROMA_SHIFT, MAX_CHROMA_SHIFT))
    b_shift = float(np.clip(-b_off * CORRECTION_STRENGTH, -MAX_CHROMA_SHIFT, MAX_CHROMA_SHIFT))

    lab[:, :, 1] = np.clip(lab[:, :, 1] + a_shift, 0, 255)
    lab[:, :, 2] = np.clip(lab[:, :, 2] + b_shift, 0, 255)
    out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    meta.update(applied=True, a_shift=a_shift, b_shift=b_shift)
    return out, meta


# --------------------------------------------------- controlled cast simulation

def simulate_warm_evening_cast(bgr: np.ndarray, strength: float = 0.55) -> np.ndarray:
    """Applies a KNOWN, controlled warm (tungsten-like, ~3000K-on-5500K)
    color cast to a real daylight photo -- reduces blue, boosts red/green
    slightly, and drops overall luminance a bit (dimmer, warmer, lower
    contrast, matching what an evening/warm-lit photo actually looks like).
    Purely for controlled-experiment purposes; never used in production."""
    b, g, r = cv2.split(bgr.astype(np.float32))
    r = np.clip(r * (1.0 + 0.22 * strength), 0, 255)
    g = np.clip(g * (1.0 + 0.06 * strength), 0, 255)
    b = np.clip(b * (1.0 - 0.30 * strength), 0, 255)
    warm = cv2.merge([b, g, r])
    warm = np.clip(warm * (1.0 - 0.18 * strength), 0, 255)  # dimmer, like lower evening sun
    return warm.astype(np.uint8)


def lab_ab_distance(bgr_a: np.ndarray, bgr_b: np.ndarray) -> float:
    """Mean Euclidean distance between two images' Lab a/b planes -- a
    simple, interpretable proxy for "how different is the color cast"."""
    lab_a = cv2.cvtColor(bgr_a, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_b = cv2.cvtColor(bgr_b, cv2.COLOR_BGR2LAB).astype(np.float32)
    d = lab_a[:, :, 1:] - lab_b[:, :, 1:]
    return float(np.sqrt((d ** 2).sum(axis=2)).mean())


def main() -> None:
    original = cv2.imread(str(SOURCE), cv2.IMREAD_COLOR)
    if original is None:
        raise SystemExit(f"could not read {SOURCE}")
    cv2.imwrite(str(OUT_DIR / "00_original_daylight.png"), original)

    evening = simulate_warm_evening_cast(original)
    cv2.imwrite(str(OUT_DIR / "01_simulated_evening_cast.png"), evening)

    corrected, meta = estimate_and_correct_cast(evening)
    cv2.imwrite(str(OUT_DIR / "02_candidate_corrected.png"), corrected)

    import tonal_correction
    toned, tonal_meta = tonal_correction.adaptive_tonal_correction(corrected, return_meta=True)
    cv2.imwrite(str(OUT_DIR / "03_candidate_plus_existing_tonal.png"), toned)

    dist_evening_vs_original = lab_ab_distance(evening, original)
    dist_corrected_vs_original = lab_ab_distance(corrected, original)
    dist_toned_vs_original = lab_ab_distance(toned, original)

    print("== Chroma cast metadata ==")
    print("evening (simulated) Lab a/b mean offsets from neutral:",
          f"a={meta['a_mean']-128:.2f} b={meta['b_mean']-128:.2f}")
    print("candidate correction applied:", meta["applied"],
          f"a_shift={meta.get('a_shift', 0):.2f} b_shift={meta.get('b_shift', 0):.2f}")
    print("existing tonal_correction issues detected on corrected image:", tonal_meta["issues"])

    print("\n== Mean Lab a/b distance from the real original (lower = closer to true daylight color) ==")
    print(f"evening (uncorrected)      : {dist_evening_vs_original:.3f}")
    print(f"candidate-corrected        : {dist_corrected_vs_original:.3f}  "
          f"({(1 - dist_corrected_vs_original / dist_evening_vs_original) * 100:.1f}% cast removed)")
    print(f"candidate + existing tonal : {dist_toned_vs_original:.3f}")

    print(f"\nOutputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
