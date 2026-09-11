"""D1 parameter sweep - isolated experiment (do not touch production).

Question: at which conservative denoise strength does Restormer denoise help
the super-resolution most, WITHOUT removing real skin/texture/road detail?

Denoise strength is controlled ONLY by blending the fixed SIDD denoiser output
back into the untouched original:

    denoised_in = (1 - K) * original + K * Restormer_SIDD(original)

Small, justified steps (no aggressive variant):
    D1-weak    K = 0.30   mostly keep the original texture
    D1-medium  K = 0.65   intermediate (extra sweep point)
    D1-current K = 1.00   full SIDD denoise (the existing D1 center point)

Every variant then goes through the SAME SwinIR-M PSNR x4 stage (tile 256).

VRAM fix vs the prior ~9.6 GB shared-memory spike: the Restormer denoise runs
tiled (512 px, feathered), the torch cache is cleared between stages, and
stage-wise peak VRAM is recorded. Nothing new is generated; the only surviving
content is what the photograph already contains.

Usage:
  .venv/Scripts/python src/pro_exp/d1_sweep.py [--limit N] [--denoise-tile 512]
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
from restore_exp import pipeline, restormer, swinir_m  # noqa: E402
from restore_exp.metrics_ext import crop_rect, region_metrics  # noqa: E402
from restore_exp.plan_abc import build_montage  # noqa: E402
from pro_exp import pipeline2  # noqa: E402

K = {"D1-weak": 0.30, "D1-medium": 0.65, "D1-current": 1.00}
BASE = ["A", "D1-weak", "D1-current", "D1-medium"]
DN_TILE = 512
DN_OVERLAP = 48
SR_TILE = 256
FACE_PAD = 0.30

A_DIR = ROOT / "enhanced" / "restore_exp"
SAVE_DIR = ROOT / "enhanced" / "pro_exp" / "d1_sweep"
REPORT_DIR = ROOT / "reports" / "pro_exp" / "d1_sweep"

LABELS = {
    "Original": "Original (faithful 4K)",
    "A": "SwinIR-M baseline",
    "D1-weak": "D1-weak (K=0.30)",
    "D1-medium": "D1-medium (K=0.65)",
    "D1-current": "D1-current (K=1.00)",
}
COLORS = {
    "Original": (60, 200, 255),
    "A": (80, 200, 255),
    "D1-weak": (200, 160, 255),
    "D1-medium": (110, 255, 180),
    "D1-current": (140, 140, 255),
}
FULL_SCALE, CROP_SCALE = 0.32, 2.2
FACE_SCALE, DETAIL_SCALE = 3.0, 2.0


def denoise_blend(orig, k, dn, device, tile=DN_TILE, overlap=DN_OVERLAP):
    dn_img = pipeline2.tiled_restore(dn, orig, tile=tile, overlap=overlap)
    if k >= 1.0:
        return dn_img
    blend = (orig.astype(np.float32) * (1 - k) +
             dn_img.astype(np.float32) * k)
    return np.clip(blend, 0, 255).astype(np.uint8)


def sr_to_4k(imgsmall, device, out_w=3840, out_h=2160):
    hi = swinir_m.sr_bgr(imgsmall, device, tile=SR_TILE)
    return cv2.resize(hi, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)


def build_d1(stem, orig, k, dn, device, stats):
    t0 = time.perf_counter()
    pre = denoise_blend(orig, k, dn, device)
    stats["dn_s"] = round(time.perf_counter() - t0, 2)
    t0 = time.perf_counter()
    out = sr_to_4k(pre, device)
    stats["sr_s"] = round(time.perf_counter() - t0, 2)
    return out


def stage_peak():
    if not torch.cuda.is_available():
        return 0.0
    p = torch.cuda.max_memory_allocated() / 1048576.0
    torch.cuda.reset_peak_memory_stats()
    return p


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    images = sorted(p.stem for p in (ROOT / "originals").glob("*.jpeg"))
    if args.limit > 0:
        images = images[:args.limit]

    full_rows, bill_rows, face_rows, face_csv, detail_rows = [], [], [], [], []
    runtime_rows = []
    dets_map, boxes_map, dw_map = {}, {}, {}

    for stem in images:
        print(f"== {stem}.jpeg ==")
        orig = cv2.imread(str(ROOT / "originals" / f"{stem}.jpeg"), cv2.IMREAD_COLOR)
        faithful = enhance.simple_upscale(orig, (3840, 2160))
        H, W = orig.shape[:2]

        t0 = time.perf_counter()
        dets = faces_mod.detect_faces(orig)
        det_s = time.perf_counter() - t0
        dets_map[stem] = dets
        boxes = pipeline.boxes_for(stem)
        boxes_map[stem] = boxes

        a = cv2.imread(str(A_DIR / f"A_{stem}.png"), cv2.IMREAD_COLOR)
        if a is None:
            raise FileNotFoundError(f"Path A output missing: {A_DIR / f'A_{stem}.png'}")
        variants = {"A": a}

        dn = restormer.load("real_denoise", device)
        for name, k in K.items():
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            stats = {"image": f"{stem}.jpeg", "variant": name, "k": k}
            variants[name] = build_d1(stem, orig, k, dn, device, stats)
            stats["peak_vram_mb"] = round(stage_peak(), 1)
            stats["total_s"] = round(stats["dn_s"] + stats["sr_s"], 2)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            runtime_rows.append(stats)
            p0 = variants[name]
            cv2.imwrite(str(SAVE_DIR / f"{name}_{stem}.png"), p0)
            print(f"   {name}: dn {stats['dn_s']:.1f}s, sr {stats['sr_s']:.1f}s, "
                  f"peak {stats['peak_vram_mb']:.0f} MB")

        for var in BASE:
            m = evaluate_similarity(faithful, variants[var])
            nr = compute_no_reference(variants[var])
            row = {"image": f"{stem}.jpeg", "variant": var, **m,
                   "sharpness": round(nr["sharpness"], 2),
                   "noise": round(nr["noise"], 3),
                   "contrast": round(nr["contrast"], 4),
                   "edge_density": round(nr["edge_density"], 5),
                   **color_stats(variants[var])}
            full_rows.append(row)

        for bi, box in enumerate(boxes):
            rect4 = pipeline.scale_rect(box, W)
            refc = crop_rect(faithful, rect4)
            for var in BASE:
                vc = crop_rect(variants[var], rect4)
                m = region_metrics(refc, vc)
                if not m:
                    continue
                tp = pipeline2.text_proxies(vc)
                bill_rows.append({"image": f"{stem}.jpeg", "box": bi,
                                  "variant": var, **m,
                                  "contrast": round(compute_contrast(vc), 3),
                                  "contrast_ref": round(compute_contrast(refc), 3),
                                  **tp})

        for fi, d in enumerate(dets[:2]):
            face_csv.append({"image": f"{stem}.jpeg", "face": fi,
                             "x": d["x"], "y": d["y"], "w": d["w"], "h": d["h"],
                             "score": round(d["score"], 3), "area": int(d["area"])})
            fx, fy, fw, fh = pipeline.scale_rect((d["x"], d["y"], d["w"], d["h"]), W)
            refc = crop_rect(faithful, (fx, fy, fw, fh))
            if refc is None:
                continue
            hp_ref = pipeline2.text_proxies(refc)["hp_energy"] + 1e-9
            for var in BASE:
                vc = crop_rect(variants[var], (fx, fy, fw, fh))
                if vc is None:
                    continue
                m = region_metrics(refc, vc)
                if not m:
                    continue
                hp_pred = pipeline2.text_proxies(vc)["hp_energy"]
                face_rows.append({"image": f"{stem}.jpeg", "face": fi,
                                  "variant": var, **m,
                                  "tex_retain_hp": round(hp_pred / hp_ref, 3)})

        ex = boxes + [(d["x"], d["y"], d["w"], d["h"]) for d in dets[:3]]
        dw = pipeline2.detail_window(orig, ex, win=160)
        dw_map[stem] = dw
        s = pipeline2.OUT_W / W
        rect4 = tuple(int(round(v * s)) for v in dw)
        refc = crop_rect(faithful, rect4)
        if refc is not None:
            for var in BASE:
                m = region_metrics(refc, crop_rect(variants[var], rect4))
                if m:
                    detail_rows.append({"image": f"{stem}.jpeg",
                                        "variant": var, **m})

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
        v = {"Original": faithful,
             "A": cv2.imread(str(A_DIR / f"A_{stem}.png"), cv2.IMREAD_COLOR)}
        for name in K:
            v[name] = cv2.imread(str(SAVE_DIR / f"{name}_{stem}.png"), cv2.IMREAD_COLOR)

        panels = [(v[lab], LABELS[lab], COLORS[lab]) for lab in LABELS_ORDER]
        build_montage(panels, REPORT_DIR / f"d1_sweep_full_{stem}.png",
                      scale=FULL_SCALE)

        boxes = boxes_map[stem]
        if boxes:
            bx, by, bw, bh = pipeline.scale_rect(boxes[0], W)
            cols = [(lab, crop_rect(img, (bx, by, bw, bh)), col)
                    for lab, img, col in zip(LABELS_ORDER, [v[x] for x in LABELS_ORDER],
                                            [COLORS[x] for x in LABELS_ORDER])]
            build_montage(order_and_filter(cols),
                          REPORT_DIR / f"d1_sweep_billboard_{stem}.png",
                          scale=CROP_SCALE)

        dets = dets_map[stem]
        for fi, d in enumerate(dets[:2]):
            fx, fy, fw, fh = pipeline.scale_rect((d["x"], d["y"], d["w"], d["h"]), W)
            cols = [(lab, crop_rect(img, (fx, fy, fw, fh)), col)
                    for lab, img, col in zip(LABELS_ORDER, [v[x] for x in LABELS_ORDER],
                                            [COLORS[x] for x in LABELS_ORDER])]
            build_montage(order_and_filter(cols),
                          REPORT_DIR / f"d1_sweep_face{fi}_{stem}.png",
                          scale=FACE_SCALE)

        dw = dw_map[stem]
        s = pipeline2.OUT_W / W
        rect4 = tuple(int(round(vv * s)) for vv in dw)
        cols = [(lab, crop_rect(img, rect4), col)
                for lab, img, col in zip(LABELS_ORDER, [v[x] for x in LABELS_ORDER],
                                        [COLORS[x] for x in LABELS_ORDER])]
        build_montage(order_and_filter(cols),
                      REPORT_DIR / f"d1_sweep_detail_{stem}.png",
                      scale=DETAIL_SCALE)

    print("\nruntime (mean):")
    print(pd.DataFrame(runtime_rows).drop(columns=["image", "variant", "k", "total_s"]
                                          ).mean(numeric_only=True).round(2).to_string()
          if runtime_rows else "n/a")
    print("\nfull-frame means:")
    print(pd.DataFrame(full_rows).drop(columns=["image"]).groupby(
        "variant").mean(numeric_only=True).round(3).to_string())
    print("\nD1 sweep artifacts ->", REPORT_DIR)


LABELS_ORDER = ("Original", "A", "D1-weak", "D1-current", "D1-medium")


if __name__ == "__main__":
    main()