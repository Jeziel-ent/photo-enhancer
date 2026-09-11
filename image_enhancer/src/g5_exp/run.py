"""G5 experiment runner: final classical billboard-text recovery test.

Isolated experiment. Does NOT touch enhance.py or main. Reuses existing
F3-natural / G1-text outputs; computes only the three G5 box candidates.

Usage:
  .venv/Scripts/python src/g5_exp/run.py [--limit N]
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
from restore_exp import pipeline  # noqa: E402
from restore_exp.metrics_ext import crop_rect, region_metrics  # noqa: E402
from restore_exp.plan_abc import build_montage  # noqa: E402
from pro_exp import pipeline2  # noqa: E402
from fidelity_exp.recipe import micro_texture  # noqa: E402
from g3_exp import recipe_g3  # noqa: E402
from g5_exp import recipe_g5  # noqa: E402

F3_DIR = ROOT / "enhanced" / "pro_exp" / "f3_exp"
G1_DIR = ROOT / "enhanced" / "g1_exp"
SAVE_DIR = ROOT / "enhanced" / "g5_exp"
REPORT_DIR = ROOT / "reports" / "g5_exp"

ORDER = ["Original", "F3-natural", "G1-text", "G5-Wiener", "G5-RLC", "G5"]
VARIANT_NEW = ["G5-Wiener", "G5-RLC", "G5"]
LABELS = {
    "Original": "Original (faithful 4K)",
    "F3-natural": "F3-natural (base)",
    "G1-text": "G1-text (RL s1.2/12, 4K)",
    "G5-Wiener": "G5-Wiener (true PSF + auto NSR)",
    "G5-RLC": "G5-RLC (measured PSF, tuned iters)",
    "G5": "G5 (edge-constrained local)",
}
COLORS = {
    "Original": (60, 200, 255),
    "F3-natural": (120, 180, 255),
    "G1-text": (255, 80, 120),
    "G5-Wiener": (160, 160, 255),
    "G5-RLC": (200, 255, 120),
    "G5": (255, 200, 40),
}
FULL_SCALE = 0.32
BILLBOARD_SCALE = 3.0
ZOOM_SCALE = 4.0


def prepare_boxes(orig, stem):
    out = []
    for box in pipeline.boxes_for(stem):
        bx, by, bw, bh = box
        nc = orig[by:by + bh, bx:bx + bw]
        gray = cv2.cvtColor(nc, cv2.COLOR_BGR2GRAY)
        sig = recipe_g5.estimate_psf_sigma(gray)
        nat_l = cv2.cvtColor(nc, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
        out.append((nat_l, sig, pipeline.scale_rect(box, orig.shape[1])))
    return out


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


def pick_window(box_img, kind, win_frac=0.55, step_frac=0.12):
    gray = cv2.cvtColor(box_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    H, W = gray.shape
    win = max(48, int(round(min(H, W) * win_frac)))
    step = max(4, int(round(win * step_frac)))
    best, best_r = -1.0, (0, 0, win, win)
    for y in range(0, H - win + 1, step):
        for x in range(0, W - win + 1, step):
            m = mag[y:y + win, x:x + win]
            if kind == "strong":
                s = float(m.mean())
            elif kind == "weak":
                lo = float(np.percentile(m, 55))
                hi = float(np.percentile(m, 85))
                s = float(((m >= lo) & (m <= hi)).mean())
            elif kind == "maxedge":
                s = float(m.max())
            else:
                s = float(m.mean())
            if s > best:
                best, best_r = s, (x, y, win, win)
    return best_r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    images = sorted(p.stem for p in (ROOT / "originals").glob("*.jpeg"))
    if args.limit > 0:
        images = images[:args.limit]

    full_rows, bill_rows, runtime_rows = [], [], []
    boxes_map = {}

    for stem in images:
        print(f"== {stem}.jpeg ==", flush=True)
        orig = cv2.imread(str(ROOT / "originals" / f"{stem}.jpeg"), cv2.IMREAD_COLOR)
        faithful = enhance.simple_upscale(orig, (3840, 2160))
        W = orig.shape[1]

        boxes = pipeline.boxes_for(stem)
        boxes_map[stem] = boxes
        prepped = prepare_boxes(orig, stem)
        sigdiag = " ".join(f"{s:.2f}" if s > 0 else "n/a"
                           for (_, s, _) in prepped)
        print(f"   measured PSF sigmas (native px): {sigdiag}", flush=True)

        f3n = cv2.imread(str(F3_DIR / f"F3-natural_{stem}.png"), cv2.IMREAD_COLOR)
        g1t = cv2.imread(str(G1_DIR / f"G1-text_{stem}.png"), cv2.IMREAD_COLOR)
        if f3n is None or g1t is None:
            raise FileNotFoundError(f"F3-natural / G1-text missing for {stem}")

        v = {"Original": faithful, "F3-natural": f3n, "G1-text": g1t}
        for name in VARIANT_NEW:
            t0 = time.perf_counter()
            v[name] = recipe_g5.apply_variant(f3n, prepped, name)
            runtime_rows.append({"image": f"{stem}.jpeg", "variant": name,
                                 "wall_s": round(time.perf_counter() - t0, 3),
                                 "peak_vram_mb": 0})
            cv2.imwrite(str(SAVE_DIR / f"{name}_{stem}.png"), v[name])
        print("   G5 Wiener/RLC/Local: " + " / ".join(
            f"{r['wall_s']:.3f}s" for r in runtime_rows[-3:]), flush=True)

        for var in ORDER:
            m = evaluate_similarity(faithful, v[var])
            nr = compute_no_reference(v[var])
            full_rows.append({"image": f"{stem}.jpeg", "variant": var, **m,
                              "sharpness": round(nr["sharpness"], 2),
                              "noise": round(nr["noise"], 3),
                              "contrast": round(nr["contrast"], 4)})

        for bi, (p, box) in enumerate(zip(prepped, boxes)):
            _, sig, rect4 = p
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
                vc_l = cv2.cvtColor(vc, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
                ref_l = cv2.cvtColor(refc, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
                bill_rows.append({"image": f"{stem}.jpeg", "box": bi,
                                  "psf_sigma": round(sig, 3), "variant": var,
                                  **m,
                                  "contrast": round(compute_contrast(vc), 3),
                                  "micro_tex": round(micro_texture(vc), 3),
                                  **tp,
                                  "stroke_iou": round(
                                      recipe_g3.stroke_mask_iou(ref_l, vc_l), 4),
                                  "overshoot_pct": round(
                                      100 * recipe_g3.overshoot_frac(ref_l, vc_l), 3)})

    pd.DataFrame(full_rows).to_csv(REPORT_DIR / "full_metrics.csv", index=False)
    pd.DataFrame(bill_rows).to_csv(REPORT_DIR / "billboard_metrics.csv", index=False)
    pd.DataFrame(runtime_rows).to_csv(REPORT_DIR / "runtime.csv", index=False)

    for stem in images:
        v = {"Original": faithful,
             "F3-natural": cv2.imread(str(F3_DIR / f"F3-natural_{stem}.png"),
                                      cv2.IMREAD_COLOR),
             "G1-text": cv2.imread(str(G1_DIR / f"G1-text_{stem}.png"),
                                   cv2.IMREAD_COLOR)}
        for name in VARIANT_NEW:
            v[name] = cv2.imread(str(SAVE_DIR / f"{name}_{stem}.png"),
                                 cv2.IMREAD_COLOR)
        build_montage([(v[l], LABELS[l], COLORS[l]) for l in ORDER],
                      REPORT_DIR / f"g5_full_{stem}.png", scale=FULL_SCALE)

        boxes = boxes_map[stem]
        if boxes:
            bx, by, bw, bh = pipeline.scale_rect(boxes[0], W)
            cols = [(LABELS[l], crop_rect(v[l], (bx, by, bw, bh)), COLORS[l])
                    for l in ORDER]
            build_montage(order_and_filter(cols),
                          REPORT_DIR / f"g5_billboard_{stem}.png",
                          scale=BILLBOARD_SCALE)

        if boxes:
            bx, by, bw, bh = pipeline.scale_rect(boxes[0], W)
            box0 = crop_rect(faithful, (bx, by, bw, bh))
            if box0 is not None:
                for kind, suffix in (("strong", "textzoom"),
                                     ("weak", "smalltext"),
                                     ("maxedge", "edgecheck")):
                    win = pick_window(box0, kind)
                    wx, wy, ww, wh = win
                    cols = [(LABELS[l], crop_rect(v[l], (bx + wx, by + wy, ww, wh)), COLORS[l])
                            for l in ORDER]
                    build_montage(order_and_filter(cols),
                                  REPORT_DIR / f"g5_{suffix}_{stem}.png",
                                  scale=ZOOM_SCALE)

    print("\nfull-frame means:")
    print(pd.DataFrame(full_rows).drop(columns=["image"]).groupby(
        "variant").mean(numeric_only=True)[
        ["psnr", "ssim", "hist_sim", "edge_align", "sharpness", "noise",
         "contrast"]].round(3).to_string())

    print("\nbillboard means:")
    print(pd.DataFrame(bill_rows).drop(columns=["image", "box"]).groupby(
        "variant").mean(numeric_only=True)[
        ["psnr", "ssim", "edge_align", "te_contrast", "hp_energy",
         "micro_tex", "stroke_iou", "overshoot_pct"]].round(3).to_string())

    print("\ng5 artifacts ->", REPORT_DIR)


if __name__ == "__main__":
    main()