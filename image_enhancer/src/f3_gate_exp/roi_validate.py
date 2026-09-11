"""ROI / content-category artifact analysis for the F3 t0=0.25 validation.

For every image in the expanded validation set, both pipelines (t0=0.30
reference = production, t0=0.25 candidate) are run and compared inside
content-category ROIs:

  signs_text   : verified billboard boxes from regions.json (originals only)
  faces        : YuNet detections (faces.py)
  fine_struct  : detail_window crop (pipeline2.py)
  sky_flat     : lowest-variance tiles, top 60%
  road         : lowest-variance tiles, bottom 40%
  foliage      : green-dominant high-texture tiles (G dominant over R and B)
  buildings    : high-texture non-green tiles in upper band
  vehicles     : high-texture non-green tiles in lower band (road-adjacent proxy)

Per ROI we measure (candidate vs reference, both vs the faithful baseline):
  PSNR, SSIM, mean pixel diff, %>10 L, %>20 L (ringing/halo tail), max|diff|.

Artifact checks:
  A) Flat leak   : sky/road max|diff| must not rise with candidate (noise
                   amplification / gate leaking into flat regions).
  B) Ringing tail: %>20 in high-texture categories must not grow materially
                   (halos would add a deep tail).
  C) Structure    : edge_align inside each category (candidate vs faithful)
                   must not drop more than the harness tolerance.
  D) Text/faces   : signs_text and faces region max|diff| and tail must stay
                   roughly unchanged (no invented strokes, no skin texture).

The script writes ROI dumps to f3_gate_exp/validate_roi.json and prints a
per-category summary plus the worst (max-delta) image of each category.
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import enhance  # noqa: E402
import quality  # noqa: E402
from regression_harness import (  # noqa: E402
    neutralize_billboard_regions,
    PIXEL_CHANGED_THRESHOLD,
)
from f3_gate_exp.recipe import f3_gate_t0  # noqa: E402
from f3_gate_exp.validate import discover_validation_images, run_candidate  # noqa: E402

REF_T0 = 0.30
CAND_T0 = 0.25
TILE = 128


def color_green_dominance(bgr):
    b, g, r = cv2.split(bgr.astype(np.float32))
    return (g >= r + 6).mean() >= 0.25 and (g >= b + 6).mean() >= 0.25


def classify_tiles(faithful):
    """Tile-level ROIs at 4K: sky_flat / road / foliage / buildings / vehicles."""
    H, W = faithful.shape[:2]
    lab_gray = cv2.cvtColor(faithful, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sobel = cv2.Sobel(lab_gray, cv2.CV_32F, 1, 0) ** 2 + \
        cv2.Sobel(lab_gray, cv2.CV_32F, 0, 1) ** 2
    cats = defaultdict(list)
    for y in range(0, H - TILE + 1, TILE):
        for x in range(0, W - TILE + 1, TILE):
            tile = faithful[y:y + TILE, x:x + TILE]
            e = float(sobel[y:y + TILE, x:x + TILE].mean())
            low = e < (np.percentile(sobel, 55) + 1e-6)
            high = e > np.percentile(sobel, 55)
            green = color_green_dominance(tile)
            frac = y / H
            if low and frac < 0.6:
                cats["sky_flat"].append((x, y, TILE, TILE))
            elif low and frac >= 0.6:
                cats["road"].append((x, y, TILE, TILE))
            elif high and green:
                cats["foliage"].append((x, y, TILE, TILE))
            elif high and not green and frac < 0.55:
                cats["buildings"].append((x, y, TILE, TILE))
            elif high and not green and frac >= 0.55:
                cats["vehicles"].append((x, y, TILE, TILE))
    return cats


def roi_stats(faithful, out, rects):
    if not rects:
        return None
    max_psnr_l, max_ssim_l, max_pd, max_tail, max_deep, max_max = -1, -1, -1, -1, -1, -1
    n = 0
    for (x, y, w, h) in rects:
        r = faithful[y:y + h, x:x + w]
        p = out[y:y + h, x:x + w]
        if r.size < 128 or r.shape != p.shape:
            continue
        n += 1
        diff = np.abs(r.astype(np.float32) - p.astype(np.float32))
        max_psnr_l = max(max_psnr_l, float(quality.compute_psnr(r, p)))
        max_ssim_l = max(max_ssim_l, float(quality.compute_ssim(r, p)))
        max_pd = max(max_pd, float(diff.mean()))
        max_tail = max(max_tail, float((diff > 10).mean() * 100))
        max_deep = max(max_deep, float((diff > 20).mean() * 100))
        max_max = max(max_max, float(diff.max()))
    if n == 0:
        return None
    return {"psnr": round(max_psnr_l, 2), "ssim": round(max_ssim_l, 4),
            "pd": round(max_pd, 3), "pct_gt10": round(max_tail, 3),
            "pct_gt20": round(max_deep, 3), "max_diff": round(max_max, 1),
            "n_rois": n}


def edge_align_in_rects(faithful, out, rects):
    if not rects:
        return None
    vals = []
    for (x, y, w, h) in rects:
        r = faithful[y:y + h, x:x + w]
        p = out[y:y + h, x:x + w]
        if r.size < 128 or r.shape != p.shape:
            continue
        try:
            vals.append(float(quality.edge_alignment(r, p)))
        except Exception:
            continue
    if not vals:
        return None
    return round(float(np.median(vals)), 4)


def main():
    regions = json.loads((ROOT / "regions.json").read_text(encoding="utf-8"))
    from restore_exp import faces as faces_mod
    from pro_exp import pipeline2

    device = None
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images = discover_validation_images()
    cat_agg = defaultdict(lambda: {"base": [], "cand": []})
    cat_worst = {}
    per_image = []

    for path in images:
        img = enhance.load_image(str(path))
        faithful = enhance.simple_upscale(img)
        H, W = img.shape[:2]
        s = enhance.OUT_W / max(W, 1)

        cats = classify_tiles(faithful)
        # Ground-truth sign boxes (originals only; site/mumbai have none)
        for bi, box in enumerate(regions.get(path.name, {}).get("boxes", [])):
            cats["signs_text"].append(tuple(int(round(v * s)) for v in box))
        # Faces at native res, scaled to 4K
        try:
            for fi, d in enumerate(faces_mod.detect_faces(img)[:2]):
                cats["faces"].append(tuple(int(round(v * s)) for v in
                                           (d["x"], d["y"], d["w"], d["h"])))
        except Exception:
            pass
        # Fine-structure detail window
        try:
            dw = pipeline2.detail_window(
                img, regions.get(path.name, {}).get("boxes", []), win=140)
            cats["fine_struct"].append(tuple(int(round(v * s)) for v in dw))
        except Exception:
            pass
        cats = {k: v for k, v in cats.items() if v}

        with neutralize_billboard_regions():
            out_ref, _ = run_candidate(img, device, REF_T0)
            out_cand, _ = run_candidate(img, device, CAND_T0)

        row = {"image": path.name}
        for cat, rects in sorted(cats.items()):
            b = roi_stats(faithful, out_ref, rects)
            c = roi_stats(faithful, out_cand, rects)
            if b is None or c is None:
                continue
            ea_b = edge_align_in_rects(faithful, out_ref, rects)
            ea_c = edge_align_in_rects(faithful, out_cand, rects)
            rec = {
                "base": b, "cand": c,
                "ea_base": ea_b, "ea_cand": ea_c,
                "delta_pd": round(c["pd"] - b["pd"], 3),
                "delta_tail": round(c["pct_gt20"] - b["pct_gt20"], 3),
                "delta_max": round(c["max_diff"] - b["max_diff"], 1),
                "delta_ea": round((ea_c - ea_b), 4) if (ea_b is not None and ea_c is not None) else None,
            }
            row[cat] = rec
            cat_agg[cat]["base"].append(b)
            cat_agg[cat]["cand"].append(c)
            if cat not in cat_worst or abs(rec["delta_max"]) > abs(cat_worst[cat].get("delta_max", 0)):
                cat_worst[cat] = {"image": path.name, **rec}
        per_image.append(row)
        tot = int(time.monotonic()) - 0
        print(f"[{path.name:<26}] done {tot}s", flush=True)

    out = {
        "category_aggregates": {
            k: {
                "base": {m: round(float(np.median([r[m] for r in v["base"]])), 4)
                         for m in ("psnr", "ssim", "pd", "pct_gt10", "pct_gt20", "max_diff")},
                "cand": {m: round(float(np.median([r[m] for r in v["cand"]])), 4)
                         for m in ("psnr", "ssim", "pd", "pct_gt10", "pct_gt20", "max_diff")},
            } for k, v in cat_agg.items()
        },
        "worst_per_category": cat_worst,
        "per_image": per_image,
    }
    dump = ROOT / "src" / "f3_gate_exp" / "validate_roi.json"
    dump.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\n{'=' * 84}\n ROI CATEGORY SUMMARY (median over images)\n{'=' * 84}")
    print(f" {'category':<12} {'base.max':>9} {'cand.max':>9} {'dMax':>6} "
          f"{'dTail(b20)':>10} {'dEA':>7}   worst(max d) image")
    for cat, v in sorted(out["category_aggregates"].items()):
        b = v["base"]
        c = v["cand"]
        w = out["worst_per_category"].get(cat, {})
        print(f" {cat:<12} {b['max_diff']:>9} {c['max_diff']:>9} "
              f"{c['max_diff'] - b['max_diff']:>6.1f} "
              f"{w.get('delta_tail', 0):>+10.3f} "
              f"{w.get('delta_ea', 0) if w.get('delta_ea') is not None else 0:>+7.4f} "
              f"  {w.get('image', '-')}")
    print(f"\nSaved -> {dump}")


if __name__ == "__main__":
    sys.exit(main() or 0)