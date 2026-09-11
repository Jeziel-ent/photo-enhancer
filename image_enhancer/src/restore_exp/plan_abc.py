"""Part 4/5/6 orchestrator: A/B/C multi-scale benchmark, region metrics,
difference heatmaps, and visual montages for the content-aware pipeline.

Usage:
  .venv/Scripts/python src/restore_exp/plan_abc.py [--reuse enhanced/swinir]
"""

import argparse
import json
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
from quality import evaluate_similarity  # noqa: E402
from restore_exp import pipeline, perspective  # noqa: E402
from restore_exp.metrics_ext import crop_rect, diff_heatmap, region_metrics  # noqa: E402

VARIANTS = ["A", "B", "B_deblur", "C"]
LABELS = {
    "A": "A: SwinIR-M PSNR (full frame)",
    "B": "B: + conservative billboard restore",
    "B_deblur": "B_deblur: + billboard restore w/ motion-deblur",
    "C": "C: + billboard + face restore (content-aware)",
}
COLORS = {
    "A": (80, 200, 255),
    "B": (200, 160, 255),
    "B_deblur": (120, 120, 120),
    "C": (110, 255, 180),
}

OUT_DIR = ROOT / "enhanced" / "restore_exp"
REPORT_DIR = ROOT / "reports" / "restore_exp"


def build_montage(panels, out_path, scale=0.33):
    label_h = 30
    tw, th = 0, 0
    rows = []
    for img, label, color in panels:
        if img is None:
            continue
        h, w = img.shape[:2]
        rw, rh = int(w * scale), int(h * scale)
        p = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_AREA)
        p = cv2.copyMakeBorder(p, label_h, 0, 0, 0, cv2.BORDER_CONSTANT,
                               value=(20, 20, 20))
        cv2.putText(p, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    color, 2, cv2.LINE_AA)
        rows.append(p)
        tw, th = rw, rh
    if not rows:
        return None
    sep = np.full((th, tw, 3), 255, np.uint8)
    frames = []
    for i, p in enumerate(rows):
        frames.append(p)
        if i < len(rows) - 1:
            frames.append(sep)
    mont = np.vstack(frames)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), mont)
    return out_path


def crop_montage(regions, out_path, scale=1.5):
    panels = [(img, label, color) for label, img, color in regions if img is not None]
    if not panels:
        return
    build_montage(panels, out_path, scale=scale)


def existing_enhanced(stem):
    try:
        return cv2.imread(str(ROOT / "enhanced" / f"realesrgan_{stem}.png"),
                          cv2.IMREAD_COLOR)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", default="enhanced/swinir",
                    help="dir holding Path-A (swinir_m_psnr_*.png)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "montages").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "crops").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "heatmaps").mkdir(parents=True, exist_ok=True)

    images = sorted(p.stem for p in (ROOT / "originals").glob("*.jpeg"))
    assert images, "no originals found"

    full_rows, bill_rows, face_rows, face_csv, keys_rows = [], [], [], [], []
    runtime_rows = []
    dets_map = {}

    for stem in images:
        print(f"== {stem}.jpeg ==")
        orig = cv2.imread(str(ROOT / "originals" / f"{stem}.jpeg"), cv2.IMREAD_COLOR)
        faithful = enhance.simple_upscale(orig, (3840, 2160))

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        stats = {}
        t_glob = time.perf_counter()
        variants = pipeline.build_variants(stem, orig, device, args.reuse,
                                           stats=stats)
        dt = time.perf_counter() - t_glob
        peak = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0

        for var in VARIANTS:
            op = OUT_DIR / f"{var}_{stem}.png"
            cv2.imwrite(str(op), variants[var])
        stats["total_s"] = round(dt, 2)
        stats["peak_vram_mb"] = round(peak, 1)
        stats["image"] = f"{stem}.jpeg"
        runtime_rows.append(stats)

        boxes = pipeline.boxes_for(stem)
        H, W = orig.shape[:2]

        for bi, box in enumerate(boxes):
            sx, sy, sw, sh = pipeline.scale_rect(box, W)
            keys_rows.append({"image": f"{stem}.jpeg", "box": bi,
                              **perspective.analyze(orig, box)})
            for var in VARIANTS:
                m = region_metrics(crop_rect(faithful, (sx, sy, sw, sh)),
                                   crop_rect(variants[var], (sx, sy, sw, sh)))
                if m:
                    bill_rows.append({"image": f"{stem}.jpeg", "box": bi,
                                      "box_w": box[2], "box_h": box[3],
                                      "variant": var, **m})
                diff_heatmap(crop_rect(faithful, (sx, sy, sw, sh)),
                             crop_rect(variants[var], (sx, sy, sw, sh)),
                             REPORT_DIR / "heatmaps" / f"bill_{stem}_b{bi}_{var}.png",
                             scale_vis=2.0)

        dets = variants["faces"]
        dets_map[stem] = dets
        for i, d in enumerate(dets[:3]):
            face_csv.append({"image": f"{stem}.jpeg", "face": i, "x": d["x"],
                             "y": d["y"], "w": d["w"], "h": d["h"],
                             "score": round(d["score"], 3), "area": int(d["area"])})
        for fi, d in enumerate(dets[:2]):
            fx, fy, fw, fh = pipeline.scale_rect((d["x"], d["y"], d["w"], d["h"]), W)
            for var in VARIANTS:
                m = region_metrics(crop_rect(faithful, (fx, fy, fw, fh)),
                                   crop_rect(variants[var], (fx, fy, fw, fh)))
                if m:
                    face_rows.append({"image": f"{stem}.jpeg", "face": fi,
                                      "variant": var, **m})
            diff_heatmap(crop_rect(faithful, (fx, fy, fw, fh)),
                         crop_rect(variants["C"], (fx, fy, fw, fh)),
                         REPORT_DIR / "heatmaps" / f"face_{stem}_f{fi}_C.png",
                         scale_vis=2.0)

        for var in VARIANTS:
            m = evaluate_similarity(faithful, variants[var])
            full_rows.append({"image": f"{stem}.jpeg", "variant": var, **m})
        print(f"   runtime {dt:.1f}s, peak VRAM {peak:.0f} MB, "
              f"faces={len(dets)}, boxes={len(boxes)}")

    if runtime_rows:
        pd.DataFrame(runtime_rows).to_csv(REPORT_DIR / "runtime.csv", index=False)
    pd.DataFrame(full_rows).to_csv(REPORT_DIR / "full_metrics.csv", index=False)
    pd.DataFrame(bill_rows).to_csv(REPORT_DIR / "billboard_metrics.csv", index=False)
    pd.DataFrame(face_rows).to_csv(REPORT_DIR / "face_metrics.csv", index=False)
    pd.DataFrame(face_csv).to_csv(REPORT_DIR / "faces_detected.csv", index=False)
    pd.DataFrame(keys_rows).to_csv(REPORT_DIR / "keystone.csv", index=False)

    for stem in images:
        orig = cv2.imread(str(ROOT / "originals" / f"{stem}.jpeg"), cv2.IMREAD_COLOR)
        faithful = enhance.simple_upscale(orig, (3840, 2160))
        real = existing_enhanced(stem)
        repair = cv2.imread(str(ROOT / "enhanced" / f"repair_v2_{stem}.png"),
                            cv2.IMREAD_COLOR)
        v = {var: cv2.imread(str(OUT_DIR / f"{var}_{stem}.png"), cv2.IMREAD_COLOR)
             for var in VARIANTS}

        panels = [
            (faithful, "Faithful Lanczos baseline", (255, 200, 60)),
            (real, "Real-ESRGAN x4plus", (80, 255, 120)),
            (repair, "Repair V2 (conservative)", (80, 255, 120)),
            (v["A"], LABELS["A"], COLORS["A"]),
            (v["B"], LABELS["B"], COLORS["B"]),
            (v["C"], LABELS["C"], COLORS["C"]),
        ]
        build_montage(panels, REPORT_DIR / "montages" / f"cmp_{stem}.png")

        regions = [
            (f"Faithful {stem}", faithful, (255, 200, 60)),
            (f"Real-ESRGAN {stem}", real, (80, 255, 120)),
            (f"Repair V2 {stem}", repair, (80, 255, 120)),
            (f"A {stem}", v["A"], COLORS["A"]),
            (f"B {stem}", v["B"], COLORS["B"]),
            (f"B_deblur {stem}", v["B_deblur"], COLORS["B_deblur"]),
            (f"C {stem}", v["C"], COLORS["C"]),
        ]

        boxes = pipeline.boxes_for(stem)
        H, W = orig.shape[:2]
        if boxes:
            bx, by, bw, bh = pipeline.scale_rect(boxes[0], W)
            cols = [(lab, crop_rect(img, (bx, by, bw, bh)), col)
                    for lab, img, col in regions]
            crop_montage(cols, REPORT_DIR / "crops" / f"billboard_{stem}.png")

        dets = dets_map[stem]
        for fi in range(2):
            if fi < len(dets):
                dx = dets[fi]
                fx, fy, fw, fh = pipeline.scale_rect(
                    (dx["x"], dx["y"], dx["w"], dx["h"]), W)
                cols = [(lab, crop_rect(img, (fx, fy, fw, fh)), col)
                        for lab, img, col in regions]
                crop_montage(cols, REPORT_DIR / "crops" / f"face{fi}_{stem}.png")

    print("\nruntime mean:")
    print(pd.DataFrame(runtime_rows).drop(
        columns=["image"]).mean(numeric_only=True).round(3).to_string())
    print("\nrestore_exp artifacts ->", REPORT_DIR)


if __name__ == "__main__":
    main()