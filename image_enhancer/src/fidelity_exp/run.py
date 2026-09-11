"""Fidelity-first controlled experiments on top of the D1-weak baseline.

Variants (all CPU-only, non-generative, built from EXISTING pixels):
  E1 TexReinject  D1-weak + source-photo luminance high-frequency (alpha=0.30)
  E2 SmartUSM     D1-weak + masked luminance unsharp
  E3 Combined     D1-weak -> SmartUSM -> TexReinject

Center baseline files are RE-USED from the D1 sweep outputs
(enhanced/pro_exp/d1_sweep/D1-weak_*.png): same SwinIR-M PSNR x4 stage,
so the ONLY difference between E1/E2/E3 and the baseline is the controlled
post-recipe. GPU models are not re-loaded (VRAM = 0 for this experiment).

Usage:
  .venv/Scripts/python src/fidelity_exp/run.py [--limit N]
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
from fidelity_exp import recipe  # noqa: E402

BASE_DIR = ROOT / "enhanced" / "pro_exp" / "d1_sweep"
SAVE_DIR = ROOT / "enhanced" / "pro_exp" / "fidelity_exp"
REPORT_DIR = ROOT / "reports" / "pro_exp" / "fidelity_exp"

ALPHA = 0.30
VARIANTS = ("D1-weak", "E1", "E2", "E3")
ORDER = ["Original", "D1-weak", "E1", "E2", "E3"]
LABELS = {
    "Original": "Original (faithful 4K)",
    "D1-weak": "D1-weak (K=0.30) baseline",
    "E1": "E1 TexReinject (alpha=0.30)",
    "E2": "E2 SmartUSM",
    "E3": "E3 Combined (USM+Tex)",
}
COLORS = {
    "Original": (60, 200, 255),
    "D1-weak": (200, 160, 255),
    "E1": (110, 255, 180),
    "E2": (255, 160, 80),
    "E3": (80, 120, 255),
}
FULL_SCALE, CROP_SCALE = 0.32, 2.2
FACE_SCALE, DETAIL_SCALE = 3.0, 2.0
FACE_PAD = 0.0


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


def color_stats(img):
    stats = {}
    for ci, name in enumerate(("b", "g", "r")):
        stats[f"mean_{name}"] = round(float(cv2.mean(img)[ci]), 3)
        stats[f"std_{name}"] = round(float(img[:, :, ci].std()), 3)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="run first N images only")
    args = ap.parse_args()

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    images = sorted(p.stem for p in (ROOT / "originals").glob("*.jpeg"))
    if args.limit > 0:
        images = images[:args.limit]

    full_rows, bill_rows, face_rows, face_csv, detail_rows, runtime_rows = (
        [], [], [], [], [], [])
    dets_map, boxes_map, dw_map = {}, {}, {}

    for stem in images:
        print(f"== {stem}.jpeg ==")
        orig = cv2.imread(str(ROOT / "originals" / f"{stem}.jpeg"), cv2.IMREAD_COLOR)
        faithful = enhance.simple_upscale(orig, (3840, 2160))
        H, W = orig.shape[:2]

        dets = faces_mod.detect_faces(orig)
        dets_map[stem] = dets
        boxes = pipeline.boxes_for(stem)
        boxes_map[stem] = boxes

        base = cv2.imread(str(BASE_DIR / f"D1-weak_{stem}.png"), cv2.IMREAD_COLOR)
        if base is None:
            raise FileNotFoundError(f"baseline missing: {BASE_DIR / f'D1-weak_{stem}.png'}")

        t0 = time.perf_counter()
        v = {
            "Original": faithful,
            "D1-weak": base,
            "E1": recipe.tex_reinject(base, faithful, alpha=ALPHA),
            "E2": recipe.smart_usm(base),
            "E3": recipe.combined(base, faithful, alpha=ALPHA),
        }
        wall = time.perf_counter() - t0
        for name, img in v.items():
            if name != "Original" and name != "D1-weak":
                cv2.imwrite(str(SAVE_DIR / f"{name}_{stem}.png"), img)
        runtime_rows.append({"image": f"{stem}.jpeg", "variant": "E1+E2+E3",
                             "wall_s": round(wall, 3), "peak_vram_mb": 0})
        print(f"   post-processing {wall:.2f}s total (E1/E2/E3, CPU only)")

        for var in ORDER:
            m = evaluate_similarity(faithful, v[var])
            nr = compute_no_reference(v[var])
            full_rows.append({"image": f"{stem}.jpeg", "variant": var, **m,
                              "sharpness": round(nr["sharpness"], 2),
                              "noise": round(nr["noise"], 3),
                              "contrast": round(nr["contrast"], 4),
                              "edge_density": round(nr["edge_density"], 5),
                              **color_stats(v[var])})

        for bi, box in enumerate(boxes):
            rect4 = pipeline.scale_rect(box, W)
            refc = crop_rect(faithful, rect4)
            outr = {"image": f"{stem}.jpeg", "box": bi}
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
                row = {**outr, "variant": var, **m,
                       "contrast": round(compute_contrast(vc), 3),
                       "micro_tex": round(recipe.micro_texture(vc), 3),
                       **tp}
                bill_rows.append(row)

        for fi, d in enumerate(dets[:2]):
            face_csv.append({"image": f"{stem}.jpeg", "face": fi,
                             "x": d["x"], "y": d["y"], "w": d["w"], "h": d["h"],
                             "score": round(d["score"], 3), "area": int(d["area"])})
            fx, fy, fw, fh = pipeline.scale_rect((d["x"], d["y"], d["w"], d["h"]), W)
            refc = crop_rect(faithful, (fx, fy, fw, fh))
            if refc is None:
                continue
            hp_ref = pipeline2.text_proxies(refc)["hp_energy"] + 1e-9
            mic_ref = recipe.micro_texture(refc)
            for var in ORDER:
                if var == "Original":
                    continue
                vc = crop_rect(v[var], (fx, fy, fw, fh))
                if vc is None:
                    continue
                m = region_metrics(refc, vc)
                if not m:
                    continue
                hp_pred = pipeline2.text_proxies(vc)["hp_energy"]
                face_rows.append({"image": f"{stem}.jpeg", "face": fi,
                                  "variant": var, **m,
                                  "tex_retain_hp": round(hp_pred / hp_ref, 3),
                                  "micro_tex": round(recipe.micro_texture(vc), 3),
                                  "micro_tex_ref": round(mic_ref, 3)})

        ex = boxes + [(d["x"], d["y"], d["w"], d["h"]) for d in dets[:3]]
        dw = pipeline2.detail_window(orig, ex, win=160)
        dw_map[stem] = dw
        s = pipeline2.OUT_W / W
        rect4 = tuple(int(round(vv * s)) for vv in dw)
        refc = crop_rect(faithful, rect4)
        if refc is not None:
            mic_ref = recipe.micro_texture(refc)
            for var in ORDER:
                if var == "Original":
                    continue
                vc = crop_rect(v[var], rect4)
                if vc is None:
                    continue
                m = region_metrics(refc, vc)
                if m:
                    detail_rows.append({"image": f"{stem}.jpeg",
                                        "variant": var, **m,
                                        "micro_tex": round(recipe.micro_texture(vc), 3),
                                        "micro_tex_ref": round(mic_ref, 3)})

    pd.DataFrame(full_rows).to_csv(REPORT_DIR / "full_metrics.csv", index=False)
    pd.DataFrame(bill_rows).to_csv(REPORT_DIR / "billboard_metrics.csv", index=False)
    pd.DataFrame(face_rows).to_csv(REPORT_DIR / "face_metrics.csv", index=False)
    pd.DataFrame(face_csv).to_csv(REPORT_DIR / "faces_detected.csv", index=False)
    pd.DataFrame(detail_rows).to_csv(REPORT_DIR / "detail_metrics.csv", index=False)
    pd.DataFrame(runtime_rows).to_csv(REPORT_DIR / "runtime.csv", index=False)

    for stem in images:
        orig = cv2.imread(str(ROOT / "originals" / f"{stem}.jpeg"), cv2.IMREAD_COLOR)
        faithful = enhance.simple_upscale(orig, (3840, 2160))
        H, W = orig.shape[:2]
        v = {
            "Original": faithful,
            "D1-weak": cv2.imread(str(BASE_DIR / f"D1-weak_{stem}.png"), cv2.IMREAD_COLOR),
        }
        for name in ("E1", "E2", "E3"):
            v[name] = cv2.imread(str(SAVE_DIR / f"{name}_{stem}.png"), cv2.IMREAD_COLOR)

        panels = [(v[lab], LABELS[lab], COLORS[lab]) for lab in ORDER]
        build_montage(panels, REPORT_DIR / f"fexp_full_{stem}.png",
                      scale=FULL_SCALE)

        boxes = boxes_map[stem]
        if boxes:
            bx, by, bw, bh = pipeline.scale_rect(boxes[0], W)
            cols = [(lab, crop_rect(img, (bx, by, bw, bh)), col)
                    for lab, img, col in zip(ORDER, [v[x] for x in ORDER],
                                             [COLORS[x] for x in ORDER])]
            build_montage(order_and_filter(cols),
                          REPORT_DIR / f"fexp_billboard_{stem}.png",
                          scale=CROP_SCALE)

        dets = dets_map[stem]
        for fi, d in enumerate(dets[:2]):
            fx, fy, fw, fh = pipeline.scale_rect((d["x"], d["y"], d["w"], d["h"]), W)
            cols = [(lab, crop_rect(img, (fx, fy, fw, fh)), col)
                    for lab, img, col in zip(ORDER, [v[x] for x in ORDER],
                                             [COLORS[x] for x in ORDER])]
            build_montage(order_and_filter(cols),
                          REPORT_DIR / f"fexp_face{fi}_{stem}.png",
                          scale=FACE_SCALE)

        dw = dw_map[stem]
        s = pipeline2.OUT_W / W
        rect4 = tuple(int(round(vv * s)) for vv in dw)
        cols = [(lab, crop_rect(img, rect4), col)
                for lab, img, col in zip(ORDER, [v[x] for x in ORDER],
                                         [COLORS[x] for x in ORDER])]
        build_montage(order_and_filter(cols),
                      REPORT_DIR / f"fexp_detail_{stem}.png",
                      scale=DETAIL_SCALE)

    print("\nfull-frame means:")
    print(pd.DataFrame(full_rows).drop(columns=["image"]).groupby(
        "variant").mean(numeric_only=True)[
        ["psnr", "ssim", "hist_sim", "edge_align", "sharpness", "noise",
         "contrast"]].round(3).to_string())
    print("\nfidelity_exp artifacts ->", REPORT_DIR)


if __name__ == "__main__":
    main()