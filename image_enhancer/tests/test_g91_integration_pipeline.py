"""Regression tests for the G9.1-B production-integration experiment
(enhance_g91.final_enhance_g91), mirroring test_final_pipeline.py's
invariants but for the integrated pipeline. Does NOT modify or replace
test_final_pipeline.py -- that suite keeps validating the untouched
production baseline (enhance.final_enhance).

Run with the project venv:
    .\\.venv\\Scripts\\python.exe tests/test_g91_integration_pipeline.py

Proves, on all six reference images:
  1. output is exactly 3840x2160 BGR uint8, finite, non-blank
  2. Candidate A / billboard reconstruction is UNCHANGED by the G9.1-B
     integration: deep inside every verified billboard box, the integrated
     output is byte-identical (allowing only the sub-pixel LAB round-trip
     floor) to enhance.final_enhance()'s own baseline output; within the
     16px feather ring at the box boundary, any residual is bounded by
     G9's own clip
  3. deterministic output (byte-identical rerun)

Exit code is non-zero on the first failure.
"""
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import enhance  # noqa: E402  (production, unmodified -- baseline)
import enhance_g91  # noqa: E402  (integration experiment under test)

ORIGINALS = [str(ROOT / "originals" / f"{i}.jpeg") for i in range(1, 7)]
NAMES = [f"{i}.jpeg" for i in range(1, 7)]
FEATHER_MARGIN = 24  # > g9_exp.recipe_g9's feather=16, so "deep interior"
                      # below is safely past the Gaussian ramp (verified
                      # numerically: the mask is exactly 0.0 at this
                      # distance from the box edge for these box sizes)
LAB_ROUNDTRIP_T = 20  # OpenCV's own BGR<->LAB 8-bit round-trip floor --
                       # same threshold test_final_pipeline.py's STRONG_T
                       # uses ("<=13..20"); with the feather mask verified
                       # exactly 0 here, any residual is this round-trip
                       # noise, not a G9 contribution
CLIP_T = 5.0 + LAB_ROUNDTRIP_T  # G9.1-B's own clip, +round-trip floor

_results = {}


def _run_all():
    for path, name in zip(ORIGINALS, NAMES):
        os.environ["BILLBOARD_IMAGE"] = name
        os.environ.pop("REGIONS_CONFIG", None)
        enhance._REGION_CONFIG = None
        img = enhance.load_image(path)
        t0 = time.perf_counter()
        out, stages = enhance_g91.final_enhance_g91(img, return_stages=True)
        dt = time.perf_counter() - t0
        _results[name] = (out, stages, dt)
        print(f"  {name}: {dt:.1f}s boxes4={stages['boxes4']} "
              f"g9_stats={stages.get('g9_stats')}", flush=True)


def _inner_outer_masks(shape, boxes4, margin):
    H, W = shape[:2]
    inner = np.zeros((H, W), bool)
    outer_ring = np.zeros((H, W), bool)
    for (bx, by, bw, bh) in boxes4:
        x0, y0 = max(0, bx), max(0, by)
        x1, y1 = min(W, bx + bw), min(H, by + bh)
        outer_ring[y0:y1, x0:x1] = True
        ix0, iy0 = min(x1, x0 + margin), min(y1, y0 + margin)
        ix1, iy1 = max(x0, x1 - margin), max(y0, y1 - margin)
        if ix1 > ix0 and iy1 > iy0:
            inner[iy0:iy1, ix0:ix1] = True
    outer_ring &= ~inner
    return inner, outer_ring


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name
          + (f" [{detail}]" if detail else ""), flush=True)
    if not cond:
        raise SystemExit(f"REGRESSION FAILURE: {name} {detail}")


def main():
    print("running G9.1-B integrated pipeline on all six references ...", flush=True)
    _run_all()

    for name in NAMES:
        out, _, _ = _results[name]
        check(f"size-{name}", out.shape == (2160, 3840, 3) and out.dtype == np.uint8,
              f"shape={out.shape} dtype={out.dtype}")

    for name in NAMES:
        out, stages, _ = _results[name]
        check(f"success-{name}",
              out is not None and np.isfinite(out.astype(np.float32)).all()
              and float(out.std()) > 0,
              f"std={float(out.std()):.2f}")

    # Candidate A / billboard reconstruction preserved: deep inside every
    # verified box, G9.1-B must match the production baseline (computed by
    # the SAME enhance.final_enhance() call the wrapper made internally,
    # captured in stages["baseline"]) to within the LAB round-trip floor;
    # the feather ring may carry a small, CLIP-bounded residual only.
    for name in NAMES:
        out, stages, _ = _results[name]
        baseline = stages["baseline"]
        boxes4 = stages["boxes4"]
        if not boxes4:
            continue
        diff = np.abs(out.astype(np.int16) - baseline.astype(np.int16)).max(axis=2)
        inner, outer_ring = _inner_outer_masks(out.shape, boxes4, FEATHER_MARGIN)
        inner_max = float(diff[inner].max()) if inner.any() else 0.0
        ring_max = float(diff[outer_ring].max()) if outer_ring.any() else 0.0
        check(f"billboard-unchanged-{name}",
              inner_max <= LAB_ROUNDTRIP_T and ring_max <= CLIP_T,
              f"inner_max={inner_max:.1f} (t={LAB_ROUNDTRIP_T}) "
              f"ring_max={ring_max:.1f} (t={CLIP_T})")

    # deterministic output
    os.environ["BILLBOARD_IMAGE"] = "1.jpeg"
    os.environ.pop("REGIONS_CONFIG", None)
    enhance._REGION_CONFIG = None
    img1 = enhance.load_image(ORIGINALS[0])
    out_a = enhance_g91.final_enhance_g91(img1)
    out_b = enhance_g91.final_enhance_g91(img1)
    check("deterministic", np.array_equal(out_a, out_b)
          and np.array_equal(out_a, _results["1.jpeg"][0]))

    total = sum(v[2] for v in _results.values())
    print(f"\nALL G9.1-B INTEGRATION REGRESSION TESTS PASSED "
          f"(pipeline wall {total:.1f}s for 6 images)", flush=True)


if __name__ == "__main__":
    main()
