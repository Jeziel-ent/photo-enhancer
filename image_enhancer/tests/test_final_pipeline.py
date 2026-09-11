"""Regression tests for the final photographic image engine (enhance.final).

Run with the project venv (no third-party test runner required):
    .\\.venv\\Scripts\\python.exe tests/test_final_pipeline.py

Proves, on all six reference images:
  1. output is exactly 3840x2160 BGR uint8
  2. all six reference images process successfully (finite, non-blank)
  3. billboard processing stays inside the verified boxes (+feather margin)
  4. outside-box pixels are unchanged by the billboard stages (Lab floor only)
  5. no invalid regions (degenerate boxes rejected; box-less images return F3)
  6. deterministic output (byte-identical rerun)
  7. conservative native recovery fires ONLY on strongly defocused boxes
  8. existing CLI/API behaviour is preserved (METHODS/dispatch intact)

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

import enhance  # noqa: E402

ORIGINALS = [str(ROOT / "originals" / f"{i}.jpeg") for i in range(1, 7)]
NAMES = [f"{i}.jpeg" for i in range(1, 7)]
TARGET = (3840, 2160)
FEATHER_MARGIN = enhance._FINAL_FEATHER + 4  # box dilation for feather ramp
STRONG_T = 20  # above OpenCV's BGR<->Lab 8-bit round-trip floor (<=13..20)
INSIDE_T = 12  # real G1/native deltas reach the +-16/+8 clips on text edges

_results = {}


def _run_all():
    for path, name in zip(ORIGINALS, NAMES):
        os.environ["BILLBOARD_IMAGE"] = name
        img = enhance.load_image(path)
        t0 = time.perf_counter()
        out, stages = enhance.final_enhance(img, return_stages=True)
        dt = time.perf_counter() - t0
        _results[name] = (out, stages, dt)
        print(f"  {name}: {dt:.1f}s boxes4={stages['boxes4']} "
              f"sigmas={[round(s, 2) for s in stages['psf_sigmas']]} "
              f"defocus={stages['defocus_boxes']}", flush=True)


def _allowed_mask(shape, boxes4, margin):
    H, W = shape[:2]
    m = np.zeros((H, W), bool)
    for (bx, by, bw, bh) in boxes4:
        x0, y0 = max(0, bx - margin), max(0, by - margin)
        x1, y1 = min(W, bx + bw + margin), min(H, by + bh + margin)
        m[y0:y1, x0:x1] = True
    return m


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name
          + (f" [{detail}]" if detail else ""), flush=True)
    if not cond:
        raise SystemExit(f"REGRESSION FAILURE: {name} {detail}")


def main():
    print("running final pipeline on all six references ...", flush=True)
    _run_all()

    # 1. output size / type
    for name in NAMES:
        out, _, _ = _results[name]
        check(f"size-{name}", out.shape == (2160, 3840, 3) and out.dtype == np.uint8,
              f"shape={out.shape} dtype={out.dtype}")

    # 2. all six process successfully
    for name in NAMES:
        out, stages, _ = _results[name]
        check(f"success-{name}",
              out is not None and np.isfinite(out.astype(np.float32)).all()
              and float(out.std()) > 0 and len(stages["boxes4"]) >= 1,
              f"std={float(out.std()):.2f} boxes={stages['boxes4']}")

    # 3. billboard processing stays inside verified boxes
    # Compared against f3_plus (F3 + the whole-frame A+ detail pass), not raw
    # f3: A+ intentionally changes pixels everywhere by design (see enhance.py
    # module docstring), so it must not be mistaken for billboard leakage --
    # this isolates ONLY Candidate A's own board-specific compositing.
    for name in NAMES:
        out, stages, _ = _results[name]
        f3 = stages["f3_plus"]
        diff = np.abs(out.astype(np.int16) - f3.astype(np.int16)).max(axis=2)
        allowed = _allowed_mask(out.shape, stages["boxes4"], FEATHER_MARGIN)
        outside_strong = int(((diff > STRONG_T) & ~allowed).sum())
        inside_strong = int(((diff > INSIDE_T) & allowed).sum())
        check(f"confinement-{name}", outside_strong == 0 and inside_strong > 0,
              f"outside>{STRONG_T}: {outside_strong}px, "
              f"inside>{INSIDE_T}: {inside_strong}px")

    # 4. outside-box pixels unchanged by billboard stages (round-trip floor)
    for name in NAMES:
        out, stages, _ = _results[name]
        f3 = stages["f3_plus"]
        diff = np.abs(out.astype(np.int16) - f3.astype(np.int16)).max(axis=2)
        allowed = _allowed_mask(out.shape, stages["boxes4"], FEATHER_MARGIN)
        outside = diff[~allowed]
        check(f"outside-unchanged-{name}",
              float(outside.max()) <= STRONG_T and float(outside.mean()) <= 1.0,
              f"max={float(outside.max()):.1f} mean={float(outside.mean()):.3f}")

    # 5. no invalid regions
    W, H = 1374, 773
    check("clamped-negative",
          enhance._final_valid_box((-5, -5, 10, 10), W, H) == (0, 0, 10, 10))
    check("invalid-outside",
          enhance._final_valid_box((5000, 5000, 100, 100), W, H) is None)
    check("invalid-tiny", enhance._final_valid_box((0, 0, 1, 1), W, H) is None)
    check("invalid-none", enhance._final_valid_box(None, W, H) is None)
    check("invalid-str", enhance._final_valid_box("xx", W, H) is None)
    check("clamped-huge",
          enhance._final_valid_box((-50, -50, 5000, 5000), W, H) == (0, 0, W, H))
    check("valid-ok",
          enhance._final_valid_box((524, 108, 291, 220), W, H) == (524, 108, 291, 220))
    if "BILLBOARD_IMAGE" in os.environ:
        del os.environ["BILLBOARD_IMAGE"]
    rng = np.random.RandomState(0)
    synth = rng.randint(0, 256, (480, 640, 3)).astype(np.uint8)
    out_s, stages_s = enhance.final_enhance(synth, return_stages=True)
    check("nobox-size", out_s.shape == (2160, 3840, 3))
    check("nobox-identical-f3plus",
          stages_s["boxes4"] == [] and stages_s["defocus_boxes"] == []
          and np.array_equal(out_s, stages_s["f3_plus"]))

    # 6. deterministic output
    os.environ["BILLBOARD_IMAGE"] = "1.jpeg"
    img1 = enhance.load_image(ORIGINALS[0])
    out_a = enhance.final_enhance(img1)
    out_b = enhance.final_enhance(img1)
    check("deterministic", np.array_equal(out_a, out_b)
          and np.array_equal(out_a, _results["1.jpeg"][0]))

    # 7. native recovery only on strongly defocused boxes
    defocus_names = [n for n in NAMES if _results[n][1]["defocus_boxes"]]
    sig_all = {n: [round(s, 2) for s in _results[n][1]["psf_sigmas"]] for n in NAMES}
    check("defocus-only", defocus_names == ["1.jpeg"],
          f"defocus={defocus_names} sigmas={sig_all}")

    # 8. existing CLI/API behaviour preserved
    for m in ("classical", "realesrgan", "hybrid", "tiled", "billboard",
              "repair", "repair_v2", "final"):
        check(f"method-{m}", m in enhance.METHODS)
    tiny = np.zeros((64, 64, 3), np.uint8) + 128
    out_c, dt_c = enhance.enhance("classical", tiny)
    check("dispatch-classical", out_c.shape == (2160, 3840, 3) and dt_c >= 0)
    try:
        enhance.enhance("nope", tiny)
        check("dispatch-unknown", False, "expected KeyError")
    except KeyError:
        check("dispatch-unknown", True)
    try:
        enhance.final_enhance(np.zeros((8, 8, 3), np.uint8))
        check("reject-tiny", False, "expected ValueError")
    except ValueError:
        check("reject-tiny", True)

    total = sum(v[2] for v in _results.values())
    print(f"\nALL REGRESSION TESTS PASSED (pipeline wall {total:.1f}s "
          f"for 6 images)", flush=True)


if __name__ == "__main__":
    main()
