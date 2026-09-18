"""Validation tests for image_enhancer/src/adjustments.py (MVP 2 manual
post-processing controls).

Pure NumPy/OpenCV, no GPU/model weights -- runs in seconds.

Run with the project venv (no third-party test runner required):
    .\\.venv\\Scripts\\python.exe tests/test_adjustments.py

Covers:
  1. default params (all zero) are a true no-op (byte-identical)
  2. brightness: +/- moves mean L in the expected direction, bounded
  3. contrast: +/- moves L std-dev in the expected direction, bounded
  4. highlights: only affects bright tones, dark tones stay ~untouched
  5. shadows: only affects dark tones, bright tones stay ~untouched
  6. saturation: hue is never shifted; S moves in the expected direction
  7. detail: local min/max clip -- never exceeds each pixel's own local
     neighborhood (no halo), zero at amount=0
  8. determinism (byte-identical rerun) and idempotent reset (adjust then
     reset-to-default != re-deriving the ORIGINAL, but IS a no-op relative
     to whatever it's applied to)
  9. malformed/out-of-range input is safely clamped, never crashes
  10. combined LUT (brightness+contrast+highlights+shadows) stays monotonic
      non-decreasing -- cannot invert tonal order or create a halo

Exit code is non-zero on the first failure.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import adjustments as adj  # noqa: E402


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name
          + (f" [{detail}]" if detail else ""), flush=True)
    if not cond:
        raise SystemExit(f"REGRESSION FAILURE: {name} {detail}")


def _l_channel(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)


def _textured_photo(size=(300, 400), seed=0):
    """A synthetic image with real local structure spanning shadow to
    highlight tones, not a flat color."""
    rng = np.random.RandomState(seed)
    base = np.clip(rng.normal(128, 40, size), 0, 255).astype(np.uint8)
    img = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(img, (30, 30), (120, 120), (40, 40, 40), -1)      # dark patch
    cv2.rectangle(img, (250, 180), (360, 270), (230, 230, 230), -1)  # bright patch
    cv2.putText(img, "TEXT", (140, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (160, 160, 160), 2)
    return img


def main():
    photo = _textured_photo()

    # ----------------------------------------------------------- 1. no-op
    out = adj.apply_adjustments(photo)
    check("default-is-true-noop", np.array_equal(out, photo))
    out2 = adj.apply_adjustments(photo, adj.DEFAULT_ADJUSTMENTS)
    check("explicit-default-dict-is-noop", np.array_equal(out2, photo))
    check("is_default-true-for-defaults", adj.is_default({}))
    check("is_default-true-for-explicit-zeros",
          adj.is_default({"brightness": 0, "contrast": 0, "highlights": 0,
                           "shadows": 0, "saturation": 0, "detail": 0}))

    # ------------------------------------------------------- 2. brightness
    out_up = adj.apply_adjustments(photo, {"brightness": 60})
    out_down = adj.apply_adjustments(photo, {"brightness": -60})
    mean_before = float(_l_channel(photo).mean())
    mean_up = float(_l_channel(out_up).mean())
    mean_down = float(_l_channel(out_down).mean())
    check("brightness-positive-brightens", mean_up > mean_before,
          f"before={mean_before:.1f} after={mean_up:.1f}")
    check("brightness-negative-darkens", mean_down < mean_before,
          f"before={mean_before:.1f} after={mean_down:.1f}")
    # Midtone-only patch (no clipping headroom lost) confirms the actual
    # ceiling magnitude directly.
    mid_flat = np.full((80, 80, 3), 120, np.uint8)
    mid_up = adj.apply_adjustments(mid_flat, {"brightness": 100})
    check("brightness-full-scale-matches-ceiling",
          abs(float(_l_channel(mid_up).mean()) - float(_l_channel(mid_flat).mean())
              - adj.MAX_BRIGHTNESS) < 2.0,
          f"delta={float(_l_channel(mid_up).mean()) - float(_l_channel(mid_flat).mean()):.2f} "
          f"ceiling={adj.MAX_BRIGHTNESS}")

    # --------------------------------------------------------- 3. contrast
    out_more = adj.apply_adjustments(photo, {"contrast": 100})
    out_less = adj.apply_adjustments(photo, {"contrast": -100})
    std_before = float(_l_channel(photo).std())
    std_more = float(_l_channel(out_more).std())
    std_less = float(_l_channel(out_less).std())
    check("contrast-positive-increases-spread", std_more > std_before,
          f"before={std_before:.1f} after={std_more:.1f}")
    check("contrast-negative-decreases-spread", std_less < std_before,
          f"before={std_before:.1f} after={std_less:.1f}")

    # ------------------------------------------------------- 4. highlights
    out_hi = adj.apply_adjustments(photo, {"highlights": 100})
    l_before = _l_channel(photo)
    l_hi = _l_channel(out_hi)
    dark_mask = l_before < 40
    bright_mask = l_before > 200
    dark_delta = float(np.abs(l_hi[dark_mask] - l_before[dark_mask]).mean())
    bright_delta = float((l_hi[bright_mask] - l_before[bright_mask]).mean())
    check("highlights-protects-dark-tones", dark_delta < 1.0, f"dark_delta={dark_delta:.2f}")
    check("highlights-boosts-bright-tones", bright_delta > 5.0, f"bright_delta={bright_delta:.2f}")

    out_hi_neg = adj.apply_adjustments(photo, {"highlights": -100})
    l_hi_neg = _l_channel(out_hi_neg)
    bright_delta_neg = float((l_hi_neg[bright_mask] - l_before[bright_mask]).mean())
    check("highlights-negative-recovers-bright-tones", bright_delta_neg < -5.0,
          f"bright_delta={bright_delta_neg:.2f}")

    # ---------------------------------------------------------- 5. shadows
    out_sh = adj.apply_adjustments(photo, {"shadows": 100})
    l_sh = _l_channel(out_sh)
    bright_delta_sh = float(np.abs(l_sh[bright_mask] - l_before[bright_mask]).mean())
    dark_delta_sh = float((l_sh[dark_mask] - l_before[dark_mask]).mean())
    check("shadows-protects-bright-tones", bright_delta_sh < 1.0, f"bright_delta={bright_delta_sh:.2f}")
    check("shadows-reveals-dark-tones", dark_delta_sh > 5.0, f"dark_delta={dark_delta_sh:.2f}")

    # ------------------------------------------------------- 6. saturation
    # A real, clearly-saturated color (near-gray pixels have numerically
    # unstable/meaningless hue, so this checks hue stability where it
    # actually matters -- exactly what a user would perceive as "hue").
    colorful = np.zeros((120, 160, 3), np.uint8)
    colorful[:, :] = (40, 90, 200)  # a clear, saturated orange (BGR)
    out_sat = adj.apply_adjustments(colorful, {"saturation": 80})
    hsv_before = cv2.cvtColor(colorful, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_after = cv2.cvtColor(out_sat, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue_diff = float(np.abs(hsv_before[:, :, 0].astype(np.int16)
                             - hsv_after[:, :, 0].astype(np.int16)).max())
    check("saturation-hue-unchanged", hue_diff <= 1.0, f"hue_diff={hue_diff}")
    check("saturation-positive-increases-s",
          float(hsv_after[:, :, 1].mean()) > float(hsv_before[:, :, 1].mean()))

    out_sat_neg = adj.apply_adjustments(photo, {"saturation": -80})
    hsv_neg = cv2.cvtColor(out_sat_neg, cv2.COLOR_BGR2HSV).astype(np.float32)
    check("saturation-negative-decreases-s",
          float(hsv_neg[:, :, 1].mean()) < float(hsv_before[:, :, 1].mean()))

    # ---------------------------------------------------------- 7. detail
    out_detail_zero = adj.apply_adjustments(photo, {"detail": 0})
    check("detail-zero-is-noop", np.array_equal(out_detail_zero, photo))

    out_detail = adj.apply_adjustments(photo, {"detail": 100})
    l_detail = _l_channel(out_detail)
    k = np.ones((adj.DETAIL_CLIP_KERNEL, adj.DETAIL_CLIP_KERNEL), np.uint8)
    l_u8 = _l_channel(photo).astype(np.uint8)
    lo = cv2.erode(l_u8, k).astype(np.float32)
    hi = cv2.dilate(l_u8, k).astype(np.float32)
    # Tolerance accounts for LAB<->BGR 8-bit round-trip rounding noise (the
    # same caveat this codebase's other Lab-based corrections document --
    # see tonal_correction.py's correct_color_cast docstring), not a real
    # clip violation.
    ROUND_TRIP_TOLERANCE = 3.5
    within_local_bounds = bool(np.all(l_detail >= lo - ROUND_TRIP_TOLERANCE)
                                and np.all(l_detail <= hi + ROUND_TRIP_TOLERANCE))
    check("detail-never-exceeds-local-neighborhood", within_local_bounds,
          "detail boost must be locally clipped (no halo/ring)")
    check("detail-changes-something",
          float(np.abs(l_detail - _l_channel(photo)).max()) > 0.0)

    # ------------------------------------------------- 8. determinism/reset
    out_a = adj.apply_adjustments(photo, {"brightness": 30, "detail": 40})
    out_b = adj.apply_adjustments(photo, {"brightness": 30, "detail": 40})
    check("deterministic", np.array_equal(out_a, out_b))
    reset_of_adjusted = adj.apply_adjustments(out_a, adj.DEFAULT_ADJUSTMENTS)
    check("reset-is-noop-on-adjusted-image", np.array_equal(reset_of_adjusted, out_a),
          "applying defaults to an already-adjusted image must not change it further")
    check("input-array-not-mutated", np.array_equal(photo, _textured_photo()))

    # ------------------------------------------------ 9. malformed input
    n = adj.normalize_adjustments({"brightness": "not-a-number", "contrast": 9999,
                                    "detail": -50})
    check("malformed-brightness-falls-back-to-zero", n["brightness"] == 0.0, n)
    check("out-of-range-contrast-clamped", n["contrast"] == 100.0, n)
    check("negative-detail-clamped-to-zero", n["detail"] == 0.0, n)
    check("non-dict-params-falls-back-to-defaults",
          adj.normalize_adjustments(None) == adj.DEFAULT_ADJUSTMENTS)
    out_malformed = adj.apply_adjustments(photo, {"brightness": "nope", "contrast": None})
    check("malformed-params-do-not-crash", out_malformed.shape == photo.shape)

    # -------------------------------------------------------- 10. LUT shape
    lut = adj._build_lut(adj.normalize_adjustments(
        {"brightness": 70, "contrast": -60, "highlights": 90, "shadows": -80}))
    check("lut-monotonic-non-decreasing", bool(np.all(np.diff(lut.astype(np.int16)) >= 0)),
          "combined LUT must never invert tonal order")
    check("lut-in-bounds", int(lut.min()) >= 0 and int(lut.max()) <= 255)

    print("\nALL ADJUSTMENTS TESTS PASSED", flush=True)


if __name__ == "__main__":
    main()
