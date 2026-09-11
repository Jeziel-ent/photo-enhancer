"""Automated enhancement benchmark.

Runs every method on each original image, writes 3840x2160 outputs, and
collects metrics:
  - VS ORIGINAL (fidelity): PSNR, SSIM, NRMSE, histogram similarity
  - NO-REFERENCE quality: NIQE
  - timing

Outputs a CSV report + a side-by-side comparison image grid.

Fidelity is measured against the ORIGINAL (upscaled to 3840x2160 via
Lanczos as the faithful baseline) because we must preserve content — i.e.,
we do NOT want hallucinated content freely drifting from the source.
"""

import argparse
import csv
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))

from enhance import METHODS, enhance, OUT_W, OUT_H
from quality import evaluate_similarity, compute_no_reference
import enhance as enhance_mod

ROOT = Path(__file__).parent.parent
ORIGINALS = ROOT / "originals"
REFERENCES = ROOT / "references"
ENHANCED = ROOT / "enhanced"
REPORTS = ROOT / "reports"
COMPARISON = REPORTS / "comparisons"


def supported_image(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def lanczos_baseline(img, target=(OUT_W, OUT_H)):
    """Faithful upscale of the original with no restoration — used as the
    reference for content-fidelity metrics and as our 'fair' baseline."""
    return cv2.resize(img, target, interpolation=cv2.INTER_LANCZOS4)


def require_verified_regions(images, methods):
    """Method C uses the MANUALLY VERIFIED billboard boxes only (one photo may
    contain several billboards). Refuse to run the billboard benchmark while
    any image lacks a verified region — the automatic detector must never
    stand in for it."""
    if "billboard" not in methods:
        return
    cfg = enhance_mod._load_region_config()
    missing = []
    for p in images:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]
        boxes = enhance_mod._entry_boxes(cfg.get(p.name, {}))
        if not boxes:
            match = [e for e in cfg.values() if e.get("image_size") == [W, H]]
            if len(match) == 1:
                boxes = enhance_mod._entry_boxes(match[0])
        if not boxes:
            missing.append(p.name)
    if missing:
        print("Cannot benchmark method 'billboard' before region verification.")
        print("Missing verified boxes for: " + ", ".join(missing))
        print("Review/correct the boxes first:")
        print("  python src\\verify_regions.py             # view overlay montages")
        print("  python src\\verify_regions.py --mode interactive")
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="*", default=list(METHODS),
                    help="methods to run (default: all)")
    ap.add_argument("--images", nargs="*", default=None,
                    help="specific filenames to process (default: all originals)")
    ap.add_argument("--compare-ref", action="store_true",
                    help="enable comparison against references/ if present")
    args = ap.parse_args()

    images = [p for p in sorted(ORIGINALS.iterdir()) if supported_image(p)]
    if not images:
        print(f"No images found in {ORIGINALS}. Place 6 original Adinn photos here.")
        return
    if args.images:
        images = [p for p in images if p.name in args.images]

    ENHANCED.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    COMPARISON.mkdir(exist_ok=True)

    require_verified_regions(images, args.methods)
    rows = []
    for method in args.methods:
        if method not in METHODS:
            print(f"!! unknown method {method}")
            continue
        for img_p in images:
            img = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
            base = lanczos_baseline(img)  # faithful 3840x2160 of the original

            os.environ["BILLBOARD_IMAGE"] = img_p.name

            t0 = time.perf_counter()
            out, enh_time = enhance(method, img)
            wall = time.perf_counter() - t0

            out_p = ENHANCED / f"{method}_{img_p.stem}.png"
            cv2.imwrite(str(out_p), out)

            sim = evaluate_similarity(base, out)
            row = {
                "method": method,
                "image": img_p.name,
                "output": out_p.name,
                "out_w": out.shape[1],
                "out_h": out.shape[0],
                "enhance_s": round(enh_time, 3),
                "wall_s": round(wall, 3),
                **sim,
            }
            row["no_ref"] = compute_no_reference(out)
            rows.append(row)
            print(f"[{method}] {img_p.name} -> {out_p.name} "
                  f"PSNR={row['psnr']:.2f} SSIM={row['ssim']:.4f} "
                  f"sharp={row['no_ref']['sharpness']:.0f} "
                  f"({row['wall_s']}s)")

    if rows:
        # flatten no_ref columns into the dataframe
        flat = []
        for r in rows:
            nr = r.pop("no_ref")
            flat.append({**r, **nr})
        df = pd.DataFrame(flat)
        csv_path = REPORTS / "benchmark_metrics.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nReport written: {csv_path}")
        agg_cols = ["psnr", "ssim", "nrmse", "hist_sim",
                    "sharpness", "noise", "contrast", "edge_density", "wall_s"]
        subgroup = [c for c in agg_cols if c in df.columns]
        print(df.groupby("method")[subgroup].mean().round(3))


if __name__ == "__main__":
    main()
