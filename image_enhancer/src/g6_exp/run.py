"""G6 experiment runner: whole-image quality R&D on top of Candidate A.

Isolated experiment. Does NOT touch enhance.py or production. Produces, for
each candidate x test image: the full enhanced frame, billboard crops, a
side-by-side montage (Original / A / A+ / B / C / D), and quantitative
metrics (full-frame vs the faithful Lanczos upscale of the original; billboard
region vs the same-region faithful crop), plus runtime and peak VRAM.

Usage:
  .venv/Scripts/python src/g6_exp/run.py
"""
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import enhance  # noqa: E402
from quality import evaluate_similarity, compute_no_reference, compute_contrast  # noqa: E402
from restore_exp.metrics_ext import crop_rect, region_metrics  # noqa: E402
from restore_exp.plan_abc import build_montage  # noqa: E402
from g6_exp import recipe_g6  # noqa: E402

SAVE_DIR = ROOT / "enhanced" / "g6_exp"
REPORT_DIR = ROOT / "reports" / "g6_exp"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATES = {
    "A": recipe_g6.candidate_a,
    "A+": recipe_g6.candidate_a_plus,
    "B": recipe_g6.candidate_b,
    "C": recipe_g6.candidate_c,
    "D": recipe_g6.candidate_d,
}
LABELS = {
    "Original": "Original (faithful 4K)",
    "A": "A (production baseline)",
    "A+": "A+ (whole-frame local contrast)",
    "B": "B (D1 + global deblur)",
    "C": "C (SwinIR-L backbone)",
    "D": "D (B global + A board)",
}
COLORS = {
    "Original": (60, 200, 255), "A": (255, 80, 120), "A+": (120, 220, 120),
    "B": (255, 200, 40), "C": (160, 160, 255), "D": (255, 120, 255),
}

# JJ Gold + GRT boards on the real two-board test image (dreamland-ventures /
# phase1_enhancement/originals/1.jpeg -- same source photo, confirmed earlier).
TWO_BOARD_IMG = ROOT.parent / "backend" / ".deckstore" / \
    "dreamland-ventures-chennai-sivagangai-ooh-proposal" / "uploads" / \
    "dreamland-ventures-chennai-01.jpeg"
TWO_BOARDS_NATIVE = [[543, 131, 247, 171], [502, 316, 192, 110]]

# Representative subset (compute budget): the two-board photo covers the
# defocus case (native regions.json box) + JJ Gold + GRT; 3.jpeg and 4.jpeg
# add a mid-blur and a mild-blur single-board reference for range.
TEST_CASES = [
    dict(stem="two_board", path=TWO_BOARD_IMG, custom_boxes=TWO_BOARDS_NATIVE,
        note="Real JJ GOLD + GRT test photo (2 boards)."),
    dict(stem="3", path=ROOT / "originals" / "3.jpeg", custom_boxes=None,
        note="Reference 3.jpeg (repo regions.json box, mid blur)."),
    dict(stem="4", path=ROOT / "originals" / "4.jpeg", custom_boxes=None,
        note="Reference 4.jpeg (repo regions.json box, mildest blur)."),
]


def setup_regions(case):
    img = cv2.imread(str(case["path"]), cv2.IMREAD_COLOR)
    H, W = img.shape[:2]
    if case["custom_boxes"] is not None:
        regions_path = SAVE_DIR / f"{case['stem']}_regions.json"
        name = f"{case['stem']}.jpeg"
        regions_path.write_text(json.dumps(
            {name: {"boxes": case["custom_boxes"], "image_size": [W, H]}}))
        os.environ["REGIONS_CONFIG"] = str(regions_path)
        os.environ["BILLBOARD_IMAGE"] = name
    else:
        os.environ.pop("REGIONS_CONFIG", None)
        os.environ["BILLBOARD_IMAGE"] = f"{case['stem']}.jpeg"
    enhance._REGION_CONFIG = None
    return img


def run_one(fn, orig):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    t0 = time.perf_counter()
    out, stages = fn(orig)
    wall = time.perf_counter() - t0
    peak_mb = 0.0
    if torch.cuda.is_available():
        peak_mb = float(torch.cuda.max_memory_allocated() / 1048576.0)
    return out, stages, wall, peak_mb


def main():
    full_rows, bill_rows, runtime_rows = [], [], []

    for case in TEST_CASES:
        print(f"== {case['stem']} — {case['note']}", flush=True)
        orig = setup_regions(case)
        faithful = enhance.simple_upscale(orig, (3840, 2160))
        W0 = orig.shape[1]

        results = {}
        boxes4_ref = None
        for name, fn in CANDIDATES.items():
            out, stages, wall, peak_mb = run_one(fn, orig)
            results[name] = out
            if boxes4_ref is None:
                boxes4_ref = stages["boxes4"]
            cv2.imwrite(str(SAVE_DIR / f"{case['stem']}_{name}.png"), out)
            runtime_rows.append({"image": case["stem"], "candidate": name,
                                 "wall_s": round(wall, 2),
                                 "peak_vram_mb": round(peak_mb, 1)})
            print(f"   {name}: {wall:.1f}s  peak_vram={peak_mb:.0f}MB",
                 flush=True)

            m = evaluate_similarity(faithful, out)
            nr = compute_no_reference(out)
            full_rows.append({"image": case["stem"], "candidate": name, **m,
                              "sharpness": round(nr["sharpness"], 2),
                              "noise": round(nr["noise"], 3),
                              "contrast": round(nr["contrast"], 4)})

        for bi, box4 in enumerate(boxes4_ref):
            ref_crop = crop_rect(faithful, box4)
            for name, out in results.items():
                vc = crop_rect(out, box4)
                m = region_metrics(ref_crop, vc)
                if not m:
                    continue
                bill_rows.append({"image": case["stem"], "board": bi,
                                  "candidate": name, **m,
                                  "contrast": round(compute_contrast(vc), 3)})

        # montages: full frame + each board at 3x
        cols = [("Original", faithful, LABELS["Original"])] + \
            [(n, results[n], LABELS[n]) for n in CANDIDATES]
        build_montage([(img, lab, COLORS.get(n, (255, 255, 255)))
                       for n, img, lab in cols],
                      REPORT_DIR / f"g6_full_{case['stem']}.png", scale=0.3)

        for bi, box4 in enumerate(boxes4_ref):
            cols = [("Original", crop_rect(faithful, box4), LABELS["Original"])] + \
                [(n, crop_rect(results[n], box4), LABELS[n]) for n in CANDIDATES]
            build_montage([(img, lab, COLORS.get(n, (255, 255, 255)))
                           for n, img, lab in cols if img is not None],
                          REPORT_DIR / f"g6_board{bi}_{case['stem']}.png",
                          scale=2.0)

    pd.DataFrame(full_rows).to_csv(REPORT_DIR / "full_metrics.csv", index=False)
    pd.DataFrame(bill_rows).to_csv(REPORT_DIR / "billboard_metrics.csv", index=False)
    pd.DataFrame(runtime_rows).to_csv(REPORT_DIR / "runtime.csv", index=False)

    print("\nfull-frame means (no-reference + similarity-to-faithful):")
    print(pd.DataFrame(full_rows).drop(columns=["image"]).groupby(
        "candidate").mean(numeric_only=True).round(3).to_string())

    print("\nbillboard means:")
    print(pd.DataFrame(bill_rows).drop(columns=["image", "board"]).groupby(
        "candidate").mean(numeric_only=True).round(3).to_string())

    print("\nruntime/VRAM means:")
    print(pd.DataFrame(runtime_rows).drop(columns=["image"]).groupby(
        "candidate").mean(numeric_only=True).round(1).to_string())

    print("\ng6 artifacts ->", REPORT_DIR)


if __name__ == "__main__":
    main()
