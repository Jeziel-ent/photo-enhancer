"""G2 experiment runner: next-generation non-generative image engine.

Compares:
  Original | F3-natural | G1-text | G2-Defocus | G2-SwinIR-M | G2-SwinIR-L | G2-Fusion

Across all 6 real originals under verified regions.json billboard boxes.
Measures full-frame, billboard, face, and runtime/VRAM metrics.
Generates full-scene, billboard, and pixel-level zoom comparisons.

Usage:
  .venv/Scripts/python src/g2_exp/run.py [--limit N]
"""

import argparse
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
from quality import compute_contrast, compute_no_reference, evaluate_similarity  # noqa: E402
from restore_exp import faces as faces_mod  # noqa: E402
from restore_exp import pipeline  # noqa: E402
from restore_exp.metrics_ext import crop_rect, region_metrics  # noqa: E402
from restore_exp.plan_abc import build_montage  # noqa: E402
from pro_exp import pipeline2  # noqa: E402
from fidelity_exp.recipe import micro_texture  # noqa: E402
from g2_exp import recipe_g2  # noqa: E402

F3_DIR = ROOT / "enhanced" / "pro_exp" / "f3_exp"
G1_DIR = ROOT / "enhanced" / "g1_exp"
SAVE_DIR = ROOT / "enhanced" / "g2_exp"
REPORT_DIR = ROOT / "reports" / "g2_exp"

ORDER = [
    "Original",
    "F3-natural",
    "G1-text",
    "G2-Defocus",
    "G2-SwinIR-M",
    "G2-SwinIR-L",
    "G2-Fusion",
]

G2_VARIANTS = [
    ("G2-Defocus", recipe_g2.g2_defocus),
    ("G2-SwinIR-M", recipe_g2.g2_swinir_m),
    ("G2-SwinIR-L", recipe_g2.g2_swinir_l),
    ("G2-Fusion", recipe_g2.g2_fusion),
]

LABELS = {
    "Original": "Original (faithful 4K)",
    "F3-natural": "F3-natural (accepted base)",
    "G1-text": "G1-text (RL deconv)",
    "G2-Defocus": "G2-Defocus (Restormer)",
    "G2-SwinIR-M": "G2-SwinIR-M (native 4x)",
    "G2-SwinIR-L": "G2-SwinIR-L (native 4x)",
    "G2-Fusion": "G2-Fusion (chroma+envelope)",
}

COLORS = {
    "Original": (60, 200, 255),
    "F3-natural": (120, 180, 255),
    "G1-text": (255, 80, 120),
    "G2-Defocus": (200, 160, 255),
    "G2-SwinIR-M": (100, 220, 180),
    "G2-SwinIR-L": (60, 220, 100),
    "G2-Fusion": (50, 255, 50),
}

FULL_SCALE = 0.30
BILLBOARD_SCALE = 2.8


def order_and_filter(cols):
    panels = [(img, lab, col) for lab, img, col in cols if img is not None]
    panels = [p for p in panels if np.ndim(p[0]) == 3 and p[0].shape[0] > 4 and p[0].shape[1] > 4]
    if not panels:
        return panels
    th, tw = panels[0][0].shape[:2]
    out = []
    for img, lab, col in panels:
        if img.shape[:2] != (th, tw):
            img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        out.append((img, lab, col))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    images = sorted(p.stem for p in (ROOT / "originals").glob("*.jpeg"))
    if args.limit > 0:
        images = images[:args.limit]

    full_rows, bill_rows, face_rows, runtime_rows = [], [], [], []
    boxes_native_map, boxes_4k_map, dets_map = {}, {}, {}

    for stem in images:
        print(f"\n== Processing {stem}.jpeg ==")
        orig = cv2.imread(str(ROOT / "originals" / f"{stem}.jpeg"), cv2.IMREAD_COLOR)
        faithful = enhance.simple_upscale(orig, (3840, 2160))
        H, W = orig.shape[:2]

        boxes = pipeline.boxes_for(stem)
        boxes_native_map[stem] = boxes
        boxes4 = [pipeline.scale_rect(b, W) for b in boxes]
        boxes_4k_map[stem] = boxes4

        dets = faces_mod.detect_faces(orig)
        dets_map[stem] = dets

        f3n = cv2.imread(str(F3_DIR / f"F3-natural_{stem}.png"), cv2.IMREAD_COLOR)
        g1t = cv2.imread(str(G1_DIR / f"G1-text_{stem}.png"), cv2.IMREAD_COLOR)
        if f3n is None:
            raise FileNotFoundError(f"F3-natural baseline missing for {stem}")
        if g1t is None:
            raise FileNotFoundError(f"G1-text baseline missing for {stem}")

        v = {
            "Original": faithful,
            "F3-natural": f3n,
            "G1-text": g1t,
        }

        # Run G2 candidates
        for name, fn in G2_VARIANTS:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
            t0 = time.perf_counter()

            v[name] = fn(orig, f3n, boxes, boxes4)
            wall_s = round(time.perf_counter() - t0, 3)
            peak_vram = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1) if torch.cuda.is_available() else 0.0

            runtime_rows.append({
                "image": f"{stem}.jpeg",
                "variant": name,
                "wall_s": wall_s,
                "peak_vram_mb": peak_vram,
            })
            cv2.imwrite(str(SAVE_DIR / f"{name}_{stem}.png"), v[name])
            print(f"   {name:14s}: {wall_s:6.3f}s | Peak VRAM: {peak_vram:6.1f} MB")

        # Full-frame metrics
        for var in ORDER:
            m = evaluate_similarity(faithful, v[var])
            nr = compute_no_reference(v[var])
            full_rows.append({
                "image": f"{stem}.jpeg",
                "variant": var,
                **m,
                "sharpness": round(nr["sharpness"], 2),
                "noise": round(nr["noise"], 3),
                "contrast": round(nr["contrast"], 4),
            })

        # Billboard metrics
        for bi, box in enumerate(boxes):
            rect4 = pipeline.scale_rect(box, W)
            refc = crop_rect(faithful, rect4)
            for var in ORDER:
                if var == "Original":
                    continue
                vc = crop_rect(v[var], rect4)
                if vc is None:
                    continue
                m = region_metrics(refc, vc)
                if not m:
                    continue
                tp = pipeline2.text_proxies(vc)
                bill_rows.append({
                    "image": f"{stem}.jpeg",
                    "box": bi,
                    "variant": var,
                    **m,
                    "contrast": round(compute_contrast(vc), 3),
                    "micro_tex": round(micro_texture(vc), 3),
                    **tp,
                })

        # Face metrics (verifying faces outside billboard boxes are untouched)
        for fi, d in enumerate(dets[:2]):
            fx, fy, fw, fh = pipeline.scale_rect((d["x"], d["y"], d["w"], d["h"]), W)
            refc = crop_rect(faithful, (fx, fy, fw, fh))
            if refc is None:
                continue
            mic_ref = micro_texture(refc)
            for var in ORDER:
                if var == "Original":
                    continue
                vc = crop_rect(v[var], (fx, fy, fw, fh))
                if vc is None:
                    continue
                m = region_metrics(refc, vc)
                if not m:
                    continue
                face_rows.append({
                    "image": f"{stem}.jpeg",
                    "face": fi,
                    "variant": var,
                    **m,
                    "micro_tex": round(micro_texture(vc), 3),
                    "micro_tex_ref": round(mic_ref, 3),
                })

    # Save CSV metrics
    df_full = pd.DataFrame(full_rows)
    df_bill = pd.DataFrame(bill_rows)
    df_face = pd.DataFrame(face_rows)
    df_run = pd.DataFrame(runtime_rows)

    df_full.to_csv(REPORT_DIR / "full_metrics.csv", index=False)
    df_bill.to_csv(REPORT_DIR / "billboard_metrics.csv", index=False)
    df_face.to_csv(REPORT_DIR / "face_metrics.csv", index=False)
    df_run.to_csv(REPORT_DIR / "runtime.csv", index=False)

    print("\n=== Generating Comparison Montages ===")
    for stem in images:
        orig = cv2.imread(str(ROOT / "originals" / f"{stem}.jpeg"), cv2.IMREAD_COLOR)
        faithful = enhance.simple_upscale(orig, (3840, 2160))
        H, W = orig.shape[:2]

        v = {
            "Original": faithful,
            "F3-natural": cv2.imread(str(F3_DIR / f"F3-natural_{stem}.png"), cv2.IMREAD_COLOR),
            "G1-text": cv2.imread(str(G1_DIR / f"G1-text_{stem}.png"), cv2.IMREAD_COLOR),
        }
        for name, _ in G2_VARIANTS:
            v[name] = cv2.imread(str(SAVE_DIR / f"{name}_{stem}.png"), cv2.IMREAD_COLOR)

        # Full-scene montage
        build_montage([(v[l], LABELS[l], COLORS[l]) for l in ORDER],
                      REPORT_DIR / f"g2_full_{stem}.png", scale=FULL_SCALE)

        # Billboard montage
        boxes = boxes_native_map[stem]
        if boxes:
            bx, by, bw, bh = pipeline.scale_rect(boxes[0], W)
            cols = [(LABELS[l], crop_rect(v[l], (bx, by, bw, bh)), COLORS[l]) for l in ORDER]
            build_montage(order_and_filter(cols),
                          REPORT_DIR / f"g2_billboard_{stem}.png",
                          scale=BILLBOARD_SCALE)

    def load_stem_variants(s):
        s_orig = cv2.imread(str(ROOT / "originals" / f"{s}.jpeg"), cv2.IMREAD_COLOR)
        s_faithful = enhance.simple_upscale(s_orig, (3840, 2160))
        res = {
            "Original": s_faithful,
            "F3-natural": cv2.imread(str(F3_DIR / f"F3-natural_{s}.png"), cv2.IMREAD_COLOR),
            "G1-text": cv2.imread(str(G1_DIR / f"G1-text_{s}.png"), cv2.IMREAD_COLOR),
        }
        for name, _ in G2_VARIANTS:
            res[name] = cv2.imread(str(SAVE_DIR / f"{name}_{s}.png"), cv2.IMREAD_COLOR)
        return res

    # Generate specialized high-zoom crops
    # 1. JJ GOLD zoom on 1.jpeg
    if "1" in images:
        v1 = load_stem_variants("1")
        b1_4k = boxes_4k_map["1"][0]
        c1 = {l: crop_rect(v1[l], b1_4k) for l in ORDER}
        # ROI for JJ GOLD / CASH FOR GOLD
        jj_cols = [(LABELS[l], c1[l][280:500, 120:680], COLORS[l]) for l in ORDER]
        build_montage(order_and_filter(jj_cols), REPORT_DIR / "g2_text_zoom_1_JJGOLD.png", scale=1.0)
        # Edge ringing check on green/red logo boundary
        edge_cols = [(LABELS[l], c1[l][220:340, 200:360], COLORS[l]) for l in ORDER]
        build_montage(order_and_filter(edge_cols), REPORT_DIR / "g2_edge_artifact_check_1.png", scale=2.0)

    # 2. No. 1 AGAIN zoom on 3.jpeg
    if "3" in images:
        v3 = load_stem_variants("3")
        b3_4k = boxes_4k_map["3"][0]
        c3 = {l: crop_rect(v3[l], b3_4k) for l in ORDER}
        no1_cols = [(LABELS[l], c3[l][160:360, 80:560], COLORS[l]) for l in ORDER]
        build_montage(order_and_filter(no1_cols), REPORT_DIR / "g2_text_zoom_3_No1Again.png", scale=1.0)
        # Small text honesty check
        small_cols = [(LABELS[l], c3[l][370:460, 70:580], COLORS[l]) for l in ORDER]
        build_montage(order_and_filter(small_cols), REPORT_DIR / "g2_smalltext_limit_3.png", scale=1.5)

    # 3. HANOBAR zoom on 5.jpeg
    if "5" in images:
        v5 = load_stem_variants("5")
        b5_4k = boxes_4k_map["5"][0]
        c5 = {l: crop_rect(v5[l], b5_4k) for l in ORDER}
        hano_cols = [(LABELS[l], c5[l][100:320, 50:570], COLORS[l]) for l in ORDER]
        build_montage(order_and_filter(hano_cols), REPORT_DIR / "g2_text_zoom_5_Hanobar.png", scale=1.0)

    print("\n=== Full-Frame Summary Means ===")
    print(df_full.drop(columns=["image"]).groupby("variant").mean(numeric_only=True)[
        ["psnr", "ssim", "hist_sim", "edge_align", "sharpness", "noise", "contrast"]
    ].round(3).to_string())

    print("\n=== Billboard Summary Means ===")
    print(df_bill.drop(columns=["image", "box"]).groupby("variant").mean(numeric_only=True)[
        ["psnr", "ssim", "edge_align", "sharp_gain_x", "te_contrast", "hp_energy", "micro_tex"]
    ].round(3).to_string())

    print("\n=== Face Summary Means ===")
    if not df_face.empty:
        print(df_face.drop(columns=["image", "face"]).groupby("variant").mean(numeric_only=True)[
            ["psnr", "ssim", "edge_align", "micro_tex", "micro_tex_ref"]
        ].round(3).to_string())

    print(f"\nG2 experiment complete! Artifacts saved to: {REPORT_DIR}")


if __name__ == "__main__":
    main()
