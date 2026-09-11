"""F2 experiment runner: multi-scale detail-fusion on the D1-weak base.

Variants (CPU-only, reusing the D1-weak SR files):
  F2-fuse  pyramid fusion (original fine + SR coarse, edge-masked, envelope)
  F2       F2-fuse + very mild local contrast
  F2-bb    F2 + slightly stronger fine-band weight + mild USM inside verified
           billboard boxes only

Compare every metric against faithfully-upscaled Original and D1-weak.

Usage:
  .venv/Scripts/python src/f2_exp/run.py [--limit N]
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
from f2_exp import recipe2  # noqa: E402

BASE_DIR = ROOT / "enhanced" / "pro_exp" / "d1_sweep"
SAVE_DIR = ROOT / "enhanced" / "pro_exp" / "f2_exp"
REPORT_DIR = ROOT / "reports" / "pro_exp" / "f2_exp"

ORDER = ["Original", "D1-weak", "F2-fuse", "F2", "F2-bb"]
VARIANT_NEW = ["F2-fuse", "F2", "F2-bb"]
LABELS = {
    "Original": "Original (faithful 4K)",
    "D1-weak": "D1-weak (K=0.30) baseline",
    "F2-fuse": "F2-fuse (pyramid fusion)",
    "F2": "F2 (+ mild local contrast)",
    "F2-bb": "F2-bb (+ billboard detail boost)",
}
COLORS = {
    "Original": (60, 200, 255),
    "D1-weak": (200, 160, 255),
    "F2-fuse": (110, 255, 180),
    "F2": (120, 180, 255),
    "F2-bb": (255, 150, 80),
}
FULL_SCALE, CROP_SCALE = 0.32, 2.2
FACE_SCALE, DETAIL_SCALE = 3.0, 2.0


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
    ap.add_argument("--limit", type=int, default=0)
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
        boxes4 = [pipeline.scale_rect(b, W) for b in boxes]

        base = cv2.imread(str(BASE_DIR / f"D1-weak_{stem}.png"), cv2.IMREAD_COLOR)
        if base is None:
            raise FileNotFoundError(f"baseline missing: {BASE_DIR / f'D1-weak_{stem}.png'}")

        v = {"Original": faithful, "D1-weak": base}
        t0 = time.perf_counter()
        v["F2-fuse"] = recipe2.f2_fuse(base, faithful)
        t_fuse = time.perf_counter() - t0
        t0 = time.perf_counter()
        v["F2"] = recipe2.f2(base, faithful)
        t_lc = time.perf_counter() - t0
        t0 = time.perf_counter()
        v["F2-bb"] = recipe2.f2_bb(base, faithful, boxes4)
        t_bb = time.perf_counter() - t0

        for name in VARIANT_NEW:
            cv2.imwrite(str(SAVE_DIR / f"{name}_{stem}.png"), v[name])
            runtime_rows.append({"image": f"{stem}.jpeg", "variant": name,
                                 "wall_s": round({"F2-fuse": t_fuse,
                                                  "F2": t_lc,
                                                  "F2-bb": t_bb}[name], 3),
                                 "peak_vram_mb": 0})
        print(f"   fuse {t_fuse:.2f}s, F2 {t_lc:.2f}s, F2-bb {t_bb:.2f}s (CPU)")

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
                bill_rows.append({**outr, "variant": var, **m,
                                  "contrast": round(compute_contrast(vc), 3),
                                  "micro_tex": round(micro_texture(vc), 3),
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
                hp_pred = pipeline2.text_proxies(vc)["hp_energy"]
                face_rows.append({"image": f"{stem}.jpeg", "face": fi,
                                  "variant": var, **m,
                                  "tex_retain_hp": round(hp_pred / hp_ref, 3),
                                  "micro_tex": round(micro_texture(vc), 3),
                                  "micro_tex_ref": round(mic_ref, 3)})

        ex = boxes + [(d["x"], d["y"], d["w"], d["h"]) for d in dets[:3]]
        dw = pipeline2.detail_window(orig, ex, win=160)
        dw_map[stem] = dw
        s = pipeline2.OUT_W / W
        rect4 = tuple(int(round(vv * s)) for vv in dw)
        refc = crop_rect(faithful, rect4)
        if refc is not None:
            mic_ref = micro_texture(refc)
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
                                        "micro_tex": round(micro_texture(vc), 3),
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
        for name in VARIANT_NEW:
            v[name] = cv2.imread(str(SAVE_DIR / f"{name}_{stem}.png"), cv2.IMREAD_COLOR)

        build_montage([(v[l], LABELS[l], COLORS[l]) for l in ORDER],
                      REPORT_DIR / f"f2_full_{stem}.png", scale=FULL_SCALE)

        boxes = boxes_map[stem]
        if boxes:
            bx, by, bw, bh = pipeline.scale_rect(boxes[0], W)
            cols = [(lab, crop_rect(v[x], (bx, by, bw, bh)), col)
                    for lab, x, col in zip(ORDER, ORDER, [COLORS[l] for l in ORDER])]
            build_montage(order_and_filter(cols),
                          REPORT_DIR / f"f2_billboard_{stem}.png", scale=CROP_SCALE)

        dets = dets_map[stem]
        for fi, d in enumerate(dets[:1]):
            fx, fy, fw, fh = pipeline.scale_rect((d["x"], d["y"], d["w"], d["h"]), W)
            cols = [(lab, crop_rect(v[x], (fx, fy, fw, fh)), col)
                    for lab, x, col in zip(ORDER, ORDER, [COLORS[l] for l in ORDER])]
            build_montage(order_and_filter(cols),
                          REPORT_DIR / f"f2_face{fi}_{stem}.png", scale=FACE_SCALE)

        dw = dw_map[stem]
        s = pipeline2.OUT_W / W
        rect4 = tuple(int(round(vv * s)) for vv in dw)
        cols = [(lab, crop_rect(v[x], rect4), col)
                for lab, x, col in zip(ORDER, ORDER, [COLORS[l] for l in ORDER])]
        build_montage(order_and_filter(cols),
                      REPORT_DIR / f"f2_detail_{stem}.png", scale=DETAIL_SCALE)

    print("\nfull-frame means:")
    print(pd.DataFrame(full_rows).drop(columns=["image"]).groupby(
        "variant").mean(numeric_only=True)[
        ["psnr", "ssim", "hist_sim", "edge_align", "sharpness", "noise",
         "contrast"]].round(3).to_string())
    print("f2_exp artifacts ->", REPORT_DIR)


if __name__ == "__main__":
    main()