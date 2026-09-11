"""G1 experiment runner: billboard-text readability vs F3-natural baseline.

Isolated experiment. Does NOT touch enhance.py or main. Reuses existing
D1-weak / F3-natural outputs from prior experiments; only computes the new
G1 box-restoration variants.

Usage:
  .venv/Scripts/python src/g1_exp/run.py [--limit N]
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

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
from g1_exp import recipe_g1  # noqa: E402

D1_DIR = ROOT / "enhanced" / "pro_exp" / "d1_sweep"
F3_DIR = ROOT / "enhanced" / "pro_exp" / "f3_exp"
SAVE_DIR = ROOT / "enhanced" / "g1_exp"
REPORT_DIR = ROOT / "reports" / "g1_exp"

ORDER = ["Original", "D1-weak", "F3-natural", "G1-natural",
         "G1-text-conservative", "G1-text"]
VARIANT_NEW = ["G1-natural", "G1-text-conservative", "G1-text"]
LABELS = {
    "Original": "Original (faithful 4K)",
    "D1-weak": "D1-weak (K=0.30) baseline",
    "F3-natural": "F3-natural (accepted photographic base)",
    "G1-natural": "G1-natural (= F3-natural, control)",
    "G1-text-conservative": "G1-text-conservative (RL s1.0/6it, t0=0.45)",
    "G1-text": "G1-text (RL s1.2/12it, t0=0.30, +USM 0.25)",
}
COLORS = {
    "Original": (60, 200, 255),
    "D1-weak": (200, 160, 255),
    "F3-natural": (120, 180, 255),
    "G1-natural": (110, 255, 180),
    "G1-text-conservative": (80, 220, 220),
    "G1-text": (255, 80, 120),
}
FULL_SCALE = 0.32
BILLBOARD_SCALE = 3.0  # high-resolution billboard crops for visual inspection


def order_and_filter(cols):
    panels = [(img, lab, col) for lab, img, col in cols if img is not None]
    panels = [p for p in panels if np.ndim(p[0]) == 3 and p[0].shape[0] > 4
              and p[0].shape[1] > 4]
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
    boxes_map, dets_map = {}, {}

    for stem in images:
        print(f"== {stem}.jpeg ==")
        orig = cv2.imread(str(ROOT / "originals" / f"{stem}.jpeg"), cv2.IMREAD_COLOR)
        faithful = enhance.simple_upscale(orig, (3840, 2160))
        H, W = orig.shape[:2]

        boxes = pipeline.boxes_for(stem)
        boxes_map[stem] = boxes
        boxes4 = [pipeline.scale_rect(b, W) for b in boxes]

        dets = faces_mod.detect_faces(orig)
        dets_map[stem] = dets

        d1 = cv2.imread(str(D1_DIR / f"D1-weak_{stem}.png"), cv2.IMREAD_COLOR)
        f3n = cv2.imread(str(F3_DIR / f"F3-natural_{stem}.png"), cv2.IMREAD_COLOR)
        if d1 is None or f3n is None:
            raise FileNotFoundError("D1-weak / F3-natural baseline missing "
                                     f"for {stem} -- run f3_exp/run.py first")

        v = {"Original": faithful, "D1-weak": d1, "F3-natural": f3n}
        for name, fn in (
            ("G1-natural", recipe_g1.g1_natural),
            ("G1-text-conservative", recipe_g1.g1_text_conservative),
            ("G1-text", recipe_g1.g1_text),
        ):
            t0 = time.perf_counter()
            v[name] = fn(faithful, f3n, boxes4)
            runtime_rows.append({"image": f"{stem}.jpeg", "variant": name,
                                 "wall_s": round(time.perf_counter() - t0, 3),
                                 "peak_vram_mb": 0})
            cv2.imwrite(str(SAVE_DIR / f"{name}_{stem}.png"), v[name])
        print("   G1 nat/cons/text: " + " / ".join(
            f"{r['wall_s']:.3f}s" for r in runtime_rows[-3:]))

        for var in ORDER:
            m = evaluate_similarity(faithful, v[var])
            nr = compute_no_reference(v[var])
            full_rows.append({"image": f"{stem}.jpeg", "variant": var, **m,
                              "sharpness": round(nr["sharpness"], 2),
                              "noise": round(nr["noise"], 3),
                              "contrast": round(nr["contrast"], 4)})

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
                bill_rows.append({"image": f"{stem}.jpeg", "box": bi,
                                  "variant": var, **m,
                                  "contrast": round(compute_contrast(vc), 3),
                                  "micro_tex": round(micro_texture(vc), 3),
                                  **tp})

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
                face_rows.append({"image": f"{stem}.jpeg", "face": fi,
                                  "variant": var, **m,
                                  "micro_tex": round(micro_texture(vc), 3),
                                  "micro_tex_ref": round(mic_ref, 3)})

    pd.DataFrame(full_rows).to_csv(REPORT_DIR / "full_metrics.csv", index=False)
    pd.DataFrame(bill_rows).to_csv(REPORT_DIR / "billboard_metrics.csv", index=False)
    pd.DataFrame(face_rows).to_csv(REPORT_DIR / "face_metrics.csv", index=False)
    pd.DataFrame(runtime_rows).to_csv(REPORT_DIR / "runtime.csv", index=False)

    for stem in images:
        orig = cv2.imread(str(ROOT / "originals" / f"{stem}.jpeg"), cv2.IMREAD_COLOR)
        faithful = enhance.simple_upscale(orig, (3840, 2160))
        H, W = orig.shape[:2]
        v = {
            "Original": faithful,
            "D1-weak": cv2.imread(str(D1_DIR / f"D1-weak_{stem}.png"), cv2.IMREAD_COLOR),
            "F3-natural": cv2.imread(str(F3_DIR / f"F3-natural_{stem}.png"), cv2.IMREAD_COLOR),
        }
        for name in VARIANT_NEW:
            v[name] = cv2.imread(str(SAVE_DIR / f"{name}_{stem}.png"), cv2.IMREAD_COLOR)

        # full-scene comparison (confirms surrounding scene is untouched)
        build_montage([(v[l], LABELS[l], COLORS[l]) for l in ORDER],
                      REPORT_DIR / f"g1_full_{stem}.png", scale=FULL_SCALE)

        boxes = boxes_map[stem]
        if boxes:
            bx, by, bw, bh = pipeline.scale_rect(boxes[0], W)
            cols = [(LABELS[l], crop_rect(v[l], (bx, by, bw, bh)), COLORS[l])
                    for l in ORDER]
            build_montage(order_and_filter(cols),
                          REPORT_DIR / f"g1_billboard_{stem}.png",
                          scale=BILLBOARD_SCALE)

    print("\nfull-frame means:")
    print(pd.DataFrame(full_rows).drop(columns=["image"]).groupby(
        "variant").mean(numeric_only=True)[
        ["psnr", "ssim", "hist_sim", "edge_align", "sharpness", "noise",
         "contrast"]].round(3).to_string())

    print("\nbillboard means:")
    print(pd.DataFrame(bill_rows).drop(columns=["image", "box"]).groupby(
        "variant").mean(numeric_only=True)[
        ["psnr", "ssim", "edge_align", "sharp_gain_x", "te_contrast",
         "hp_energy", "micro_tex"]].round(3).to_string())

    print("g1_exp artifacts ->", REPORT_DIR)


if __name__ == "__main__":
    main()
