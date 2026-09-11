"""ROI / content-category analysis for the F3 t0 candidates.

For every image:
  - F3 with t0 in {0.30 baseline, 0.25, 0.20} -> G7-MS -> A+
  - Per-category ROIs:
      signs/text/buildings -> regions.json billboard boxes (scaled 4K)
      faces                -> YuNet detections (scaled 4K)
      sky/flat             -> lowest-variance tiles, top half
      road/surface         -> lowest-variance tiles, bottom half
      fine structures      -> pipeline2.detail_window crop
      high-texture         -> highest-Laplacian tiles (vehicles/buildings)
  - Per-ROI: PSNR, SSIM, mean pixel diff vs faithful
  - Ringing/overshoot proxy: % of pixels with |out-faithful| > 10 L, and the
    tail at > 20, measured over both the whole frame and each ROI, so halos
    (bilateral overshoot at edges) are separated from genuine sharpening.
  - Flat-region safety: max |out-faithful| inside sky/road tiles
    (any value > 5 there = gate leaked into flats = noise amplification).
"""

import sys
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import enhance  # noqa: E402
import quality  # noqa: E402
from f3_gate_exp.recipe import f3_gate_t0  # noqa: E402
from regression_harness import discover_benchmark_images  # noqa: E402

T0S = [0.30, 0.25, 0.20]


def variance_tiles(gray, tile=96, top_half=False, max_keep=3):
    H, W = gray.shape
    ys = range(0, H // 2, tile) if top_half else range(H // 2, H, tile)
    scored = []
    for y in ys:
        for x in range(0, W, tile):
            patch = gray[y:min(y + tile, H), x:min(x + tile, W)]
            if patch.size < 400:
                continue
            scored.append((patch.std(), x, y, min(tile, W - x), min(tile, H - y)))
    scored.sort()
    return scored[:max_keep]


def roi_stats(faithful, out, x, y, w, h):
    r = faithful[y:y + h, x:x + w]
    p = out[y:y + h, x:x + w]
    if r.size < 64 or r.shape != p.shape:
        return None
    diff = np.abs(r.astype(np.float32) - p.astype(np.float32))
    return {
        "psnr": round(float(quality.compute_psnr(r, p)), 2),
        "ssim": round(float(quality.compute_ssim(r, p)), 4),
        "pd": round(float(diff.mean()), 3),
        "pct_gt10": round(float((diff > 10).mean() * 100), 3),
        "pct_gt20": round(float((diff > 20).mean() * 100), 3),
        "max_diff": round(float(diff.max()), 1),
    }


def main():
    regions = json.loads((ROOT / "regions.json").read_text(encoding="utf-8"))
    from pro_exp import pipeline2
    from restore_exp import faces as faces_mod

    summary = {}
    for path in discover_benchmark_images():
        img = enhance.load_image(str(path))
        faithful = enhance.simple_upscale(img)
        d1, _, _ = enhance._final_d1_weak(img)

        outs = {}
        for t0 in T0S:
            f3 = f3_gate_t0(faithful, d1, t0)
            ms = enhance._final_multiscale_detail(f3)
            outs[t0] = enhance._final_whole_frame_detail(ms)

        orig = cv2.imread(str(path), cv2.IMREAD_COLOR)
        H, W = orig.shape[:2]
        s = enhance.OUT_W / W

        cats = {}
        for bi, box in enumerate(regions.get(path.name, {}).get("boxes", [])):
            cats[f"signs_bldg_{bi}"] = tuple(int(round(v * s)) for v in box)
        dets = faces_mod.detect_faces(orig)
        for fi, d in enumerate(dets[:2]):
            cats[f"face_{fi}"] = tuple(int(round(v * s)) for v in
                                       (d["x"], d["y"], d["w"], d["h"]))
        g = cv2.cvtColor(faithful, cv2.COLOR_BGR2GRAY)
        for i, (_, x, y, w, h) in enumerate(variance_tiles(g, top_half=True)):
            cats[f"sky_flat_{i}"] = (x, y, w, h)
        for i, (_, x, y, w, h) in enumerate(variance_tiles(g, top_half=False)):
            cats[f"road_{i}"] = (x, y, w, h)
        try:
            dw = pipeline2.detail_window(
                orig, regions.get(path.name, {}).get("boxes", []) +
                [(d["x"], d["y"], d["w"], d["h"]) for d in dets[:2]], win=160)
            cats["fine_struct"] = tuple(int(round(v * s)) for v in dw)
        except Exception:
            pass

        print(f"\n{'=' * 84}\n {path.name}\n{'=' * 84}")
        print(f" {'roi':<16} {'t0':<6} {'psnr':>7} {'ssim':>7} "
              f"{'pd':>6} {'%>10':>7} {'%>20':>7} {'max':>5}")
        target_idx = [0, 1, 2]  # baseline, 0.25, 0.20
        for cat, rect in sorted(cats.items()):
            x, y, w, h = rect
            for t0 in T0S:
                ms = roi_stats(faithful, outs[t0], x, y, w, h)
                if ms is None:
                    continue
                tag = "base" if t0 == 0.30 else f"{t0:.2f}"
                print(f" {cat:<16} {tag:<6} {ms['psnr']:>7} {ms['ssim']:>7} "
                      f"{ms['pd']:>6} {ms['pct_gt10']:>7} {ms['pct_gt20']:>7} "
                      f"{ms['max_diff']:>5}")

        # full-frame ringing-tail (halo proxy) + flat-region safety
        for t0 in T0S:
            diff = np.abs(faithful.astype(np.float32) - outs[t0].astype(np.float32))
            tag = "base" if t0 == 0.30 else f"{t0:.2f}"
            print(f" FULLFRAME  {tag:<6} "
                  f"pct>10={round(float((diff > 10).mean() * 100), 3):>7} "
                  f"pct>20={round(float((diff > 20).mean() * 100), 3):>7} "
                  f"max={round(float(diff.max()), 1):>5}")
            for cat, rect in [c for c in sorted(cats.items())
                              if c[0].startswith(("sky_flat", "road"))]:
                x, y, w, h = rect
                ms = roi_stats(faithful, outs[t0], x, y, w, h)
                if ms:
                    summary.setdefault((path.name, cat), {})[t0] = ms["max_diff"]

    print("\n\n=== Flat-region leak check: max |diff| in sky/road tiles (threshold >5 = leak) ===")
    for (im, cat), vals in sorted(summary.items()):
        row = f" {im:<14} {cat:<12} " + " ".join(
            f"t0={t:.2f}: {v:>4}" for t, v in sorted(vals.items()))
        print(row)


if __name__ == "__main__":
    main()