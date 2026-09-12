"""Validation tests for image_enhancer/src/tonal_correction.py.

Unlike test_final_pipeline.py / test_g91_integration_pipeline.py, this
module needs no GPU/model weights -- adaptive_tonal_correction is pure
NumPy/OpenCV -- so it runs in seconds.

Run with the project venv (no third-party test runner required):
    .\\.venv\\Scripts\\python.exe tests/test_tonal_correction.py

Covers:
  1. a normal/well-balanced image gets no (or near-zero) correction
  2. underexposed -> corrected
  3. overexposed -> corrected without introducing new clipping
  4. low contrast -> dynamic range improves
  5. crushed shadows -> lifted without destroying relative shadow detail
  6. clipped highlights -> pulled back without adding NEW clipping
  7. strong/saturated colors -> no significant color (a/b) shift
  8. text/signage/geometry protection -> edge map is preserved (a global,
     monotonic per-value LUT cannot move or blur an edge)
  9. deterministic output (byte-identical rerun)
  10. all six production reference images: correction is safe (bounded,
      monotonic) and matches the module's own activation decision, whatever
      it is for that image (this does not replace the real GPU pipeline
      regression suite in test_final_pipeline.py -- see that file for the
      end-to-end 3840x2160 checks).

Exit code is non-zero on the first failure.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import tonal_correction as tc  # noqa: E402

ORIGINALS = [str(ROOT / "originals" / f"{i}.jpeg") for i in range(1, 7)]
NAMES = [f"{i}.jpeg" for i in range(1, 7)]

_rng = np.random.RandomState(0)


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name
          + (f" [{detail}]" if detail else ""), flush=True)
    if not cond:
        raise SystemExit(f"REGRESSION FAILURE: {name} {detail}")


def _l_channel(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 0]


def _textured_gray(mean, noise, size=(300, 400), seed=0):
    """A synthetic image with real local structure (a rectangle + text +
    per-pixel noise), not a flat color -- so it's a meaningful stand-in for
    a photo, not a degenerate edge case."""
    rng = np.random.RandomState(seed)
    base = np.clip(rng.normal(mean, noise, size), 0, 255).astype(np.uint8)
    img = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    shade = int(np.clip(mean * 0.6, 0, 255))
    cv2.rectangle(img, (50, 50), (150, 150), (shade, shade, shade), -1)
    text_shade = int(np.clip(mean * 1.3, 0, 255))
    cv2.putText(img, "TEXT", (180, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (text_shade,) * 3, 2)
    return img


def main():
    # ---------------------------------------------------------- 1. normal
    normal = _textured_gray(mean=120, noise=45)
    out, meta = tc.adaptive_tonal_correction(normal, return_meta=True)
    check("normal-bypassed", meta["applied"] is False and meta["issues"] == [],
          f"issues={meta['issues']}")
    check("normal-unchanged", np.array_equal(out, normal))

    # ----------------------------------------------------- 2. underexposed
    dark = _textured_gray(mean=40, noise=15)
    out, meta = tc.adaptive_tonal_correction(dark, return_meta=True)
    check("underexposed-detected", "underexposed" in meta["issues"], meta["issues"])
    check("underexposed-brightened",
          float(_l_channel(out).mean()) > float(_l_channel(dark).mean()) + 10,
          f"before={_l_channel(dark).mean():.1f} after={_l_channel(out).mean():.1f}")
    gamma = meta["params"].get("gamma")
    check("underexposed-gamma-bounded",
          gamma is not None and (1.0 - tc.MAX_GAMMA_DEVIATION) <= gamma <= 1.0,
          f"gamma={gamma}")

    # ------------------------------------------------------ 3. overexposed
    bright = _textured_gray(mean=210, noise=15)
    before_white = float((_l_channel(bright) >= 254).mean())
    out, meta = tc.adaptive_tonal_correction(bright, return_meta=True)
    after_white = float((_l_channel(out) >= 254).mean())
    check("overexposed-detected", "overexposed" in meta["issues"], meta["issues"])
    check("overexposed-darkened",
          float(_l_channel(out).mean()) < float(_l_channel(bright).mean()) - 5,
          f"before={_l_channel(bright).mean():.1f} after={_l_channel(out).mean():.1f}")
    check("overexposed-no-new-clipping", after_white <= before_white + 1e-9,
          f"before_white={before_white:.4f} after_white={after_white:.4f}")

    # ----------------------------------------------------- 4. low contrast
    flat = _textured_gray(mean=120, noise=8)
    before_range = float(np.percentile(_l_channel(flat), 99) - np.percentile(_l_channel(flat), 1))
    out, meta = tc.adaptive_tonal_correction(flat, return_meta=True)
    after_range = float(np.percentile(_l_channel(out), 99) - np.percentile(_l_channel(out), 1))
    check("low-contrast-detected", "low_contrast" in meta["issues"], meta["issues"])
    check("low-contrast-improved", after_range > before_range,
          f"before={before_range:.1f} after={after_range:.1f}")
    alpha = meta["params"].get("stretch_alpha")
    check("low-contrast-alpha-bounded",
          alpha is not None and 0.0 <= alpha <= tc.MAX_STRETCH_BLEND, f"alpha={alpha}")

    # ------------------------------------------------- 5. crushed shadows
    shadow_img = _textured_gray(mean=130, noise=35, seed=1)
    shadow_img = shadow_img.copy()
    shadow_img[0:120, :] = 0
    # two distinct near-black bands to prove detail survives the lift
    shadow_img[10:40, 10:390] = 4
    shadow_img[60:90, 10:390] = 10
    out, meta = tc.adaptive_tonal_correction(shadow_img, return_meta=True)
    check("crushed-shadows-detected", "crushed_shadows" in meta["issues"], meta["issues"])
    lift = meta["params"].get("shadow_lift", 0.0)
    check("crushed-shadows-lift-bounded", 0.0 < lift <= tc.MAX_SHADOW_LIFT, f"lift={lift}")
    band_low = float(_l_channel(out)[10:40, 10:390].mean())
    band_high = float(_l_channel(out)[60:90, 10:390].mean())
    check("crushed-shadows-detail-preserved", band_high > band_low + 1.0,
          f"band(L=4)->{band_low:.2f} band(L=10)->{band_high:.2f}")

    # ----------------------------------------------- 6. clipped highlights
    hi_img = _textured_gray(mean=130, noise=35, seed=2)
    hi_img = hi_img.copy()
    hi_img[0:120, :] = 255
    before_white2 = float((_l_channel(hi_img) >= 254).mean())
    out, meta = tc.adaptive_tonal_correction(hi_img, return_meta=True)
    after_white2 = float((_l_channel(out) >= 254).mean())
    check("clipped-highlights-detected", "clipped_highlights" in meta["issues"], meta["issues"])
    check("clipped-highlights-no-new-clipping", after_white2 <= before_white2 + 1e-9,
          f"before={before_white2:.4f} after={after_white2:.4f}")
    pull = meta["params"].get("highlight_pull", 0.0)
    check("clipped-highlights-pull-bounded", 0.0 < pull <= tc.MAX_HIGHLIGHT_PULL, f"pull={pull}")

    # -------------------------------------------------------- 7. strong colors
    strong = np.zeros((300, 300, 3), np.uint8)
    strong[:, :100] = (40, 40, 220)
    strong[:, 100:200] = (40, 200, 40)
    strong[:, 200:] = (220, 40, 40)
    out, meta = tc.adaptive_tonal_correction(strong, return_meta=True)
    lab_a = cv2.cvtColor(strong, cv2.COLOR_BGR2LAB).astype(np.int16)
    lab_b = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.int16)
    a_diff = int(np.abs(lab_a[:, :, 1] - lab_b[:, :, 1]).max())
    b_diff = int(np.abs(lab_a[:, :, 2] - lab_b[:, :, 2]).max())
    check("strong-colors-no-shift", a_diff <= 3 and b_diff <= 3,
          f"a_diff={a_diff} b_diff={b_diff} (issues={meta['issues']})")

    # --------------------------------------------- 8. geometry/text/edges
    # A pure per-value LUT cannot move an edge; confirm empirically via
    # Canny on a triggered case (the crushed-shadow image, which has real
    # correction applied above).
    edges_before = cv2.Canny(shadow_img, 50, 150)
    out_shadow, _ = tc.adaptive_tonal_correction(shadow_img, return_meta=True)
    edges_after = cv2.Canny(out_shadow, 50, 150)
    union = np.count_nonzero(edges_before | edges_after)
    intersection = np.count_nonzero(edges_before & edges_after)
    iou = (intersection / union) if union else 1.0
    check("edges-preserved", iou > 0.85, f"edge IoU={iou:.3f}")

    # ---------------------------------------------------- 9. determinism
    out_a = tc.adaptive_tonal_correction(dark)
    out_b = tc.adaptive_tonal_correction(dark)
    check("deterministic", np.array_equal(out_a, out_b))

    # ------------------------------------------ 10. six production originals
    for path, name in zip(ORIGINALS, NAMES):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            check(f"reference-readable-{name}", False, "could not read fixture")
            continue
        out, meta = tc.adaptive_tonal_correction(img, return_meta=True)
        check(f"reference-shape-preserved-{name}", out.shape == img.shape)
        # Whatever the decision, it must be internally consistent and bounded.
        if meta["applied"]:
            lab_a = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.int16)
            lab_b = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.int16)
            a_diff = int(np.abs(lab_a[:, :, 1] - lab_b[:, :, 1]).max())
            b_diff = int(np.abs(lab_a[:, :, 2] - lab_b[:, :, 2]).max())
            l_diff = int(np.abs(lab_a[:, :, 0] - lab_b[:, :, 0]).max())
            check(f"reference-{name}-chroma-stable", a_diff <= 3 and b_diff <= 3,
                  f"a_diff={a_diff} b_diff={b_diff}")
            check(f"reference-{name}-luma-bounded", l_diff <= 90,
                  f"L max diff={l_diff} params={meta['params']}")
        else:
            check(f"reference-{name}-bypassed-identical", np.array_equal(out, img))

    print("\nALL TONAL CORRECTION TESTS PASSED", flush=True)


if __name__ == "__main__":
    main()
