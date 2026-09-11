"""G7 experiment runner: conservative post-A+ whole-image candidates R&D on
top of the G6 Candidate A / A+ baseline, plus a Real-ESRGAN diagnostic upper
bound (explicitly rejected -- see G7_EXP_REPORT.md).

Isolated experiment. Does NOT touch enhance.py, g6_exp, or production.
Produces, for each candidate x test image: the full enhanced frame,
billboard-region metrics, full-frame metrics vs the faithful Lanczos
baseline, montages, and runtime/peak-VRAM.

Writes CSVs incrementally (appends after each test case) so a partial run
(e.g. interrupted by a timeout) still leaves usable data behind.

Usage:
  .venv/Scripts/python src/g7_exp/run.py [stems...]

  With no arguments, runs the full sweep: two_board + 1..6 (all 7 test
  cases) over all 7 candidates (A, A+, G7-RL, G7-MS, G7-GF, G7-WIDE,
  G7-GAN). Pass one or more stems (e.g. "two_board" "3" "4") to run only
  those cases -- useful for resuming a partial run.
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
from g7_exp import recipe_g7  # noqa: E402

SAVE_DIR = ROOT / "enhanced" / "g7_exp"
REPORT_DIR = ROOT / "reports" / "g7_exp"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATES = {
    "A": recipe_g7.candidate_a,
    "A+": recipe_g7.candidate_a_plus,
    "G7-RL": recipe_g7.candidate_g7_rl,
    "G7-MS": recipe_g7.candidate_g7_ms,
    "G7-GF": recipe_g7.candidate_g7_gf,
    "G7-WIDE": recipe_g7.candidate_g7_wide,
    "G7-GAN": recipe_g7.candidate_g7_gan,
}
LABELS = {
    "Original": "Original (faithful 4K)",
    "A": "A (production baseline)",
    "A+": "A+ (G6 whole-frame contrast)",
    "G7-RL": "G7-RL (full-frame RL deblur)",
    "G7-MS": "G7-MS (multi-scale detail)",
    "G7-GF": "G7-GF (guided/bilateral detail)",
    "G7-WIDE": "G7-WIDE (+-20 F3 envelope)",
    "G7-GAN": "G7-GAN (Real-ESRGAN, REJECTED diagnostic)",
}
COLORS = {
    "Original": (60, 200, 255), "A": (255, 80, 120), "A+": (120, 220, 120),
    "G7-RL": (255, 200, 40), "G7-MS": (160, 160, 255), "G7-GF": (255, 120, 255),
    "G7-WIDE": (80, 220, 220), "G7-GAN": (60, 60, 220),
}

TWO_BOARD_IMG = ROOT.parent / "backend" / ".deckstore" / \
    "dreamland-ventures-chennai-sivagangai-ooh-proposal" / "uploads" / \
    "dreamland-ventures-chennai-01.jpeg"
TWO_BOARDS_NATIVE = [[543, 131, 247, 171], [502, 316, 192, 110]]

TEST_CASES = [
    dict(stem="two_board", path=TWO_BOARD_IMG, custom_boxes=TWO_BOARDS_NATIVE,
        note="Real JJ GOLD + GRT test photo (2 boards) -- primary benchmark."),
] + [
    dict(stem=str(i), path=ROOT / "originals" / f"{i}.jpeg", custom_boxes=None,
        note=f"Reference {i}.jpeg (repo regions.json box).")
    for i in range(1, 7)
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
        torch.cuda.empty_cache()
    return out, stages, wall, peak_mb


def _append_csv(rows, path, cols_order=None):
    if not rows:
        return
    df = pd.DataFrame(rows)
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


def main():
    stems_filter = set(sys.argv[1:]) or None
    full_path = REPORT_DIR / "full_metrics.csv"
    bill_path = REPORT_DIR / "billboard_metrics.csv"
    runtime_path = REPORT_DIR / "runtime.csv"

    for case in TEST_CASES:
        if stems_filter and case["stem"] not in stems_filter:
            continue
        print(f"== {case['stem']} — {case['note']}", flush=True)
        orig = setup_regions(case)
        faithful = enhance.simple_upscale(orig, (3840, 2160))

        full_rows, bill_rows, runtime_rows = [], [], []
        results = {}
        boxes4_ref = None
        for name, fn in CANDIDATES.items():
            out, stages, wall, peak_mb = run_one(fn, orig)
            results[name] = out
            if boxes4_ref is None or len(stages["boxes4"]) > len(boxes4_ref):
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

        for bi, box4 in enumerate(boxes4_ref or []):
            ref_crop = crop_rect(faithful, box4)
            for name, out in results.items():
                vc = crop_rect(out, box4)
                m = region_metrics(ref_crop, vc)
                if not m:
                    continue
                bill_rows.append({"image": case["stem"], "board": bi,
                                  "candidate": name, **m,
                                  "contrast": round(compute_contrast(vc), 3)})

        _append_csv(full_rows, full_path)
        _append_csv(bill_rows, bill_path)
        _append_csv(runtime_rows, runtime_path)

        cols = [("Original", faithful, LABELS["Original"])] + \
            [(n, results[n], LABELS[n]) for n in CANDIDATES]
        build_montage([(img, lab, COLORS.get(n, (255, 255, 255)))
                       for n, img, lab in cols],
                      REPORT_DIR / f"g7_full_{case['stem']}.png", scale=0.3)

        for bi, box4 in enumerate(boxes4_ref or []):
            cols = [("Original", crop_rect(faithful, box4), LABELS["Original"])] + \
                [(n, crop_rect(results[n], box4), LABELS[n]) for n in CANDIDATES]
            build_montage([(img, lab, COLORS.get(n, (255, 255, 255)))
                           for n, img, lab in cols if img is not None],
                          REPORT_DIR / f"g7_board{bi}_{case['stem']}.png",
                          scale=2.0)

        print(f"   -> {case['stem']} done, CSVs appended.", flush=True)

    print("\ng7 artifacts ->", REPORT_DIR)


if __name__ == "__main__":
    main()
