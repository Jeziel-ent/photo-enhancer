"""Standalone montage/crop builder for the pro_exp experiment.

Reads the saved enhanced/pro_exp/{var}_{stem}.png outputs and region crops
and (re)creates reports/pro_exp/montages + crops. Fast (no model inference),
so it can be re-run while iterating on visualization.

Usage:
  .venv/Scripts/python src/pro_exp/montage.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import enhance  # noqa: E402
from restore_exp import faces as faces_mod  # noqa: E402
from restore_exp import pipeline  # noqa: E402
from restore_exp.metrics_ext import crop_rect  # noqa: E402
from restore_exp.plan_abc import build_montage  # noqa: E402
from pro_exp import pipeline2  # noqa: E402

FVAR = ["A", "D1", "D2", "P2", "P2b", "P3"]
REC = ["RECa", "RECb", "RECc"]
LABELS = {
    "A": "A: SwinIR-M PSNR (baseline)",
    "D1": "D1: denoise BEFORE SR",
    "D2": "D2: SR then denoise",
    "P2": "P2: A + ProLook lite",
    "P2b": "P2b: A + ProLook standard",
    "P3": "P3: ProLook + billboard (RECb)",
    "RECa": "RECa: billboard crop SwinIR",
    "RECb": "RECb: crop denoise->SwinIR",
    "RECc": "RECc: RECb + ProLook",
    "FC": "FC: face crop denoise->SwinIR->ProLook",
}
COLORS = {
    "Faithful": (60, 200, 255),
    "A": (80, 200, 255),
    "D1": (200, 160, 255),
    "D2": (120, 120, 120),
    "P2": (110, 255, 180),
    "P2b": (140, 140, 255),
    "P3": (70, 220, 220),
    "RECa": (0, 200, 200),
    "RECb": (0, 180, 255),
    "RECc": (0, 255, 0),
    "FC": (0, 180, 255),
}

OUT_DIR = ROOT / "enhanced" / "pro_exp"
REPORT_DIR = ROOT / "reports" / "pro_exp"
FULL_SCALE, BILL_SCALE, FACE_SCALE, DETAIL_SCALE = 0.32, 2.0, 3.0, 2.2


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
    (REPORT_DIR / "montages").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "crops").mkdir(parents=True, exist_ok=True)
    images = sorted(p.stem for p in (ROOT / "originals").glob("*.jpeg"))

    for stem in images:
        orig = cv2.imread(str(ROOT / "originals" / f"{stem}.jpeg"), cv2.IMREAD_COLOR)
        faithful = enhance.simple_upscale(orig, (3840, 2160))
        H, W = orig.shape[:2]
        v = {var: cv2.imread(str(OUT_DIR / f"{var}_{stem}.png"), cv2.IMREAD_COLOR)
             for var in FVAR}
        if any(img is None for img in v.values()):
            raise FileNotFoundError(f"missing full-frame outputs for {stem}")

        panels = [
            (faithful, "Faithful Lanczos", COLORS["Faithful"]),
            (v["A"], LABELS["A"], COLORS["A"]),
            (v["D1"], LABELS["D1"], COLORS["D1"]),
            (v["D2"], LABELS["D2"], COLORS["D2"]),
            (v["P2"], LABELS["P2"], COLORS["P2"]),
            (v["P2b"], LABELS["P2b"], COLORS["P2b"]),
        ]
        build_montage(panels, REPORT_DIR / "montages" / f"cmp_full_{stem}.png",
                      scale=FULL_SCALE)

        boxes = pipeline.boxes_for(stem)
        if boxes:
            bx, by, bw, bh = pipeline.scale_rect(boxes[0], W)
            rec = {mode: cv2.imread(str(REPORT_DIR / "crops"
                                        / f"rec_b0_{stem}_{mode}.png"))
                   for mode in REC}
            cols = [
                (f"Faithful {stem}", faithful, COLORS["Faithful"]),
                (f"A {stem}", v["A"], COLORS["A"]),
                (f"D1 {stem}", v["D1"], COLORS["D1"]),
                (f"D2 {stem}", v["D2"], COLORS["D2"]),
                (f"P2 {stem}", v["P2"], COLORS["P2"]),
                (f"P2b {stem}", v["P2b"], COLORS["P2b"]),
                (f"P3 {stem}", v["P3"], COLORS["P3"]),
            ]
            cols = [(lab, crop_rect(img, (bx, by, bw, bh)), col)
                    for lab, img, col in cols]
            for mode in REC:
                cols.append((f"{mode} {stem}",
                             pipeline2.to_box_size(rec[mode], (bx, by, bw, bh)),
                             COLORS[mode]))
            build_montage(order_and_filter(cols),
                          REPORT_DIR / "crops" / f"billboard_{stem}.png",
                          scale=BILL_SCALE)

        dets = faces_mod.detect_faces(orig)
        for fi, d in enumerate(dets[:2]):
            fx, fy, fw, fh = pipeline.scale_rect((d["x"], d["y"], d["w"], d["h"]), W)
            fc = cv2.imread(str(REPORT_DIR / "crops" / f"rec_face{fi}_{stem}.png"))
            cols = [
                (f"Faithful {stem}", faithful, COLORS["Faithful"]),
                (f"A {stem}", v["A"], COLORS["A"]),
                (f"D1 {stem}", v["D1"], COLORS["D1"]),
                (f"P2 {stem}", v["P2"], COLORS["P2"]),
                (f"FC {stem}", fc, COLORS["FC"]),
            ]
            cols = [(lab, crop_rect(img, (fx, fy, fw, fh)), col)
                    for lab, img, col in cols]
            build_montage(order_and_filter(cols),
                          REPORT_DIR / "crops" / f"face{fi}_{stem}.png",
                          scale=FACE_SCALE)

        ex = boxes + [(d["x"], d["y"], d["w"], d["h"]) for d in dets[:3]]
        dw = pipeline2.detail_window(orig, ex, win=160)
        s = pipeline2.OUT_W / W
        rect4 = tuple(int(round(v * s)) for v in dw)
        cols = [
            (f"Faithful {stem}", faithful, COLORS["Faithful"]),
            (f"A {stem}", v["A"], COLORS["A"]),
            (f"D1 {stem}", v["D1"], COLORS["D1"]),
            (f"P2 {stem}", v["P2"], COLORS["P2"]),
        ]
        cols = [(lab, crop_rect(img, rect4), col) for lab, img, col in cols]
        build_montage(order_and_filter(cols),
                      REPORT_DIR / "crops" / f"detail_{stem}.png",
                      scale=DETAIL_SCALE)
        print(f"{stem}: full, billboard, faces={len(dets)}, detail montages done")

    print("montages ->", REPORT_DIR)


if __name__ == "__main__":
    main()