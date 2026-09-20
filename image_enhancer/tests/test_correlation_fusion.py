"""Regression tests for enhance._final_correlation_fusion (the validated
replacement for F3-natural on the GPU production path -- see
image_enhancer/reports/research/correlation_fusion_final_validation/ for
the research/validation history).

Pure NumPy/OpenCV, no GPU/model weights -- runs in seconds.

Run with the project venv (no third-party test runner required):
    .\\.venv\\Scripts\\python.exe tests/test_correlation_fusion.py

Covers:
  1. output dimensions match input
  2. output is finite (no NaN/Inf) and in valid uint8 BGR range
  3. determinism (byte-identical rerun on identical input)
  4. neutral/identity behavior where mathematically expected:
       a. perfectly flat faithful+D1 -> true no-op (no local structure to
          trust a correlation against -> gate=0 everywhere)
       b. D1 with zero band-passed signal (D1 == its own blur) -> true
          no-op regardless of what faithful looks like (nothing to inject)
  5. the applied per-pixel L change never exceeds the documented envelope
     (+-_FINAL_CF_ENV) -- the hard safety bound this function promises
  6. chroma (Lab a/b channels) is never touched, only L
  7. module constants are within their own documented valid ranges
  8. production wiring: enhance.final_enhance() really calls
     _final_correlation_fusion at the "f3" stage position (verified by
     monkeypatching _final_d1_weak to skip the real GPU/model step, not
     by requiring a GPU)

Exit code is non-zero on the first failure.
"""

import sys
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import enhance  # noqa: E402

_failures = []


def check(name, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    if not condition:
        _failures.append(name)


def _lab(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)


def _real_photo_like(seed, h=256, w=384):
    """A synthetic but non-trivial BGR image: smooth gradient + real
    structured texture (not pure random noise, which wouldn't correlate
    with itself the way real photographic detail does)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = 80 + 60 * np.sin(xx / 40.0) * np.cos(yy / 55.0)
    base = np.clip(base, 0, 255)
    noise = rng.normal(0, 6, size=(h, w)).astype(np.float32)
    l_plane = np.clip(base + noise, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(l_plane, cv2.COLOR_GRAY2BGR)
    bgr[:, :, 1] = np.clip(bgr[:, :, 1].astype(np.int16) + 10, 0, 255).astype(np.uint8)
    return bgr


def main():
    faithful = _real_photo_like(1)
    d1 = _real_photo_like(2)  # a genuinely different "SR" signal, same shape

    out = enhance._final_correlation_fusion(faithful, d1)

    # 1. dimensions
    check("output shape matches input", out.shape == faithful.shape)

    # 2. finite + valid range
    check("output dtype is uint8", out.dtype == np.uint8)
    check("output has no NaN/Inf (uint8 can't represent them, decode round-trip)",
          np.all(np.isfinite(out.astype(np.float64))))
    check("output values within [0, 255]", out.min() >= 0 and out.max() <= 255)

    # 3. determinism
    out2 = enhance._final_correlation_fusion(faithful, d1)
    check("deterministic (byte-identical rerun)", np.array_equal(out, out2))

    # 4a. flat input -> true no-op
    flat_faithful = np.full((128, 128, 3), 120, dtype=np.uint8)
    flat_d1 = np.full((128, 128, 3), 200, dtype=np.uint8)  # different value, still flat
    flat_out = enhance._final_correlation_fusion(flat_faithful, flat_d1)
    check("flat (no local structure) faithful+D1 -> true no-op",
          np.array_equal(flat_out, flat_faithful))

    # 4b. D1 with zero band-passed signal -> true no-op regardless of faithful
    smooth_d1 = cv2.GaussianBlur(d1, (0, 0), enhance._FINAL_CF_BP_SIGMA * 4)
    # further blur so D1's OWN band-pass at _FINAL_CF_BP_SIGMA is ~zero
    smooth_d1 = cv2.GaussianBlur(smooth_d1, (0, 0), enhance._FINAL_CF_BP_SIGMA * 4)
    zero_bp_out = enhance._final_correlation_fusion(faithful, smooth_d1)
    l_faithful = enhance._final_lab_l(faithful)
    l_zero_bp_out = enhance._final_lab_l(zero_bp_out)
    check("D1 with ~zero band-passed signal -> ~no change to faithful's L-plane",
          float(np.abs(l_faithful - l_zero_bp_out).max()) < 1.5)

    # 5. envelope respected
    l_out = enhance._final_lab_l(out)
    max_delta = float(np.abs(l_out - _lab(faithful)[:, :, 0]).max())
    check(f"applied L delta never exceeds +-{enhance._FINAL_CF_ENV} envelope "
          f"(max observed: {max_delta:.2f})",
          max_delta <= enhance._FINAL_CF_ENV + 1e-3)

    # 6. chroma untouched (mathematically only L is modified; a small
    # tolerance accounts for uint8 quantization in the LAB<->BGR round
    # trip the function itself performs before returning)
    lab_faithful = _lab(faithful)
    lab_out = _lab(out)
    chroma_delta = float(np.abs(lab_faithful[:, :, 1:] - lab_out[:, :, 1:]).max())
    check(f"chroma (a/b channels) unchanged from faithful "
          f"(max observed delta: {chroma_delta:.2f}, uint8 round-trip tolerance)",
          chroma_delta <= 2.0)

    # 7. constants sanity
    check("_FINAL_CF_T0 in [0, 1)", 0.0 <= enhance._FINAL_CF_T0 < 1.0)
    check("_FINAL_CF_SIGMA > 0", enhance._FINAL_CF_SIGMA > 0)
    # production-pinned value (approved after controlled w=2.0 -> 3.0 experiment
    # + human visual review of FINAL_VISUAL_REVIEW.png; gate admission region is
    # w-independent, so a change here only scales already-admitted detail)
    check("_FINAL_CF_W == 3.0 (production-pinned)",
          enhance._FINAL_CF_W == 3.0)
    check("_FINAL_CF_CLIP > 0", enhance._FINAL_CF_CLIP > 0)
    check("_FINAL_CF_ENV > 0", enhance._FINAL_CF_ENV > 0)
    check("_FINAL_CF_BP_SIGMA > 0", enhance._FINAL_CF_BP_SIGMA > 0)
    check("_FINAL_CF_MIN_LOCAL_STD >= 0", enhance._FINAL_CF_MIN_LOCAL_STD >= 0)

    # 8. production wiring: final_enhance() actually calls
    # _final_correlation_fusion at the "f3" stage, without needing a GPU
    # (short-circuit the real SR model call).
    fake_faithful = _real_photo_like(3, h=64, w=64)
    fake_d1 = _real_photo_like(4, h=64, w=64)
    with mock.patch.object(enhance, "_final_d1_weak",
                            return_value=(fake_d1, 0.0, "cpu")):
        img = _real_photo_like(5, h=32, w=32)
        _, stages = enhance.final_enhance(img, target=(64, 64), return_stages=True)
    expected_f3 = enhance._final_correlation_fusion(stages["faithful"], fake_d1)
    check("final_enhance()'s 'f3' stage is produced by _final_correlation_fusion "
          "(not the old F3-natural)",
          np.array_equal(stages["f3"], expected_f3))
    check("_final_f3_natural is unchanged and still importable (kept for reference)",
          callable(enhance._final_f3_natural))

    print(f"\n{len(_failures)} failure(s)" if _failures else "\nAll checks passed.")
    return 1 if _failures else 0


if __name__ == "__main__":
    sys.exit(main())
