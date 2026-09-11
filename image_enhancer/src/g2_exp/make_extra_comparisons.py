"""G2 supplemental comparisons: face + detail montages + metrics.

Reuses already-computed outputs (no re-inference, no production touch):
  Original (faithful 4K) | F3-natural | G1-text | G2-Defocus | G2-SwinIR-M | G2-SwinIR-L | G2-Fusion

Generates (in reports/g2_exp/):
  g2_face0_1.png, g2_detail_{1..6}.png, detail_metrics.csv, faces_detected.csv

Detail window = sharpest 160px window outside verified billboard + face boxes
(same helper F3 uses: pro_exp.pipeline2.detail_window), scaled to 4K.

Usage:
  .venv/Scripts/python.exe src/g2_exp/make_extra_comparisons.py
"""

import sys
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import enhance  # noqa: E402
from fidelity_exp.recipe import micro_texture  # noqa: E402
from pro_exp import pipeline2  # noqa: E402
from restore_exp import faces as faces_mod  # noqa: E402
from restore_exp.metrics_ext import crop_rect, region_metrics  # noqa: E402
from restore_exp.pipeline import boxes_for, scale_rect  # noqa: E402
from restore_exp.plan_abc import build_montage  # noqa: E402

F3_DIR = ROOT / "enhanced" / "pro_exp" / "f3_exp"
G1_DIR = ROOT / "enhanced" / "g1_exp"
SAVE_DIR = ROOT / "enhanced" / "g2_exp"
REPORT_DIR = ROOT / "reports" / "g2_exp"

ORDER = ["Original", "F3-natural", "G1-text", "G2-Defocus",
         "G2-SwinIR-M", "G2-SwinIR-L", "G2-Fusion"]

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

FACE_SCALE = 3.0
DETAIL_SCALE = 2.0


def order_and_filter(cols):
    panels = [(img, lab, col) for lab, img, col in cols if img is not None]
    panels = [p for p in panels if p[0] is not None and p[0].ndim == 3
              and p[0].shape[0] > 4 and p[0].shape[1] > 4]
    if not panels:
        return panels
    th, tw = panels[0][0].shape[:2]
    out = []
    for img, lab, col in panels:
        if img.shape[:2] != (th, tw):
            img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        out.append((img, lab, col))
    return out


def load_variants(stem):
    orig = cv2.imread(str(ROOT / "originals" / f"{stem}.jpeg"), cv2.IMREAD_COLOR)
    faithful = enhance.simple_upscale(orig, (3840, 2160))
    v = {
        "Original": faithful,
        "F3-natural": cv2.imread(str(F3_DIR / f"F3-natural_{stem}.png"), cv2.IMREAD_COLOR),
        "G1-text": cv2.imread(str(G1_DIR / f"G1-text_{stem}.png"), cv2.IMREAD_COLOR),
    }
    for name in ORDER[3:]:
        v[name] = cv2.imread(str(SAVE_DIR / f"{name}_{stem}.png"), cv2.IMREAD_COLOR)
    return orig, faithful, v


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    images = sorted(p.stem for p in (ROOT / "originals").glob("*.jpeg"))

    detail_rows, face_csv = [], []

    for stem in images:
        orig, faithful, v = load_variants(stem)
        H, W = orig.shape[:2]
        for k, img in v.items():
            if img is None:
                raise FileNotFoundError(f"missing variant {k} for {stem}")

        boxes = boxes_for(stem)
        dets = faces_mod.detect_faces(orig)
        face_csv.append({"image": f"{stem}.jpeg", "n_faces": len(dets),
                         "faces": str(dets[:2])})

        # Face montage (only stems with detections, i.e. 1.jpeg)
        for fi, d in enumerate(dets[:1]):
            fx, fy, fw, fh = scale_rect((d["x"], d["y"], d["w"], d["h"]), W)
            cols = [(LABELS[l], crop_rect(v[l], (fx, fy, fw, fh)), COLORS[l])
                    for l in ORDER]
            build_montage(order_and_filter(cols),
                          REPORT_DIR / f"g2_face{fi}_{stem}.png",
                          scale=FACE_SCALE)
            print(f"face montage -> g2_face{fi}_{stem}.png")

        # Detail window outside billboard + face boxes (F3 helper)
        ex = list(boxes) + [(d["x"], d["y"], d["w"], d["h"]) for d in dets[:3]]
        dw = pipeline2.detail_window(orig, ex, win=160)
        s = pipeline2.OUT_W / W
        rect4 = tuple(int(round(q * s)) for q in dw)
        cols = [(LABELS[l], crop_rect(v[l], rect4), COLORS[l]) for l in ORDER]
        build_montage(order_and_filter(cols),
                      REPORT_DIR / f"g2_detail_{stem}.png",
                      scale=DETAIL_SCALE)
        print(f"detail montage -> g2_detail_{stem}.png (orig-window={dw})")

        # Detail metrics vs faithful reference
        refc = crop_rect(faithful, rect4)
        mic_ref = micro_texture(refc)
        for var in ORDER:
            if var == "Original":
                continue
            vc = crop_rect(v[var], rect4)
            m = region_metrics(refc, vc)
            if not m:
                continue
            detail_rows.append({"image": f"{stem}.jpeg", "variant": var, **m,
                                "micro_tex": round(micro_texture(vc), 3),
                                "micro_tex_ref": round(mic_ref, 3)})

    pd.DataFrame(detail_rows).to_csv(REPORT_DIR / "detail_metrics.csv", index=False)
    pd.DataFrame(face_csv).to_csv(REPORT_DIR / "faces_detected.csv", index=False)
    print("\nDetail means:")
    df = pd.DataFrame(detail_rows)
    if not df.empty:
        print(df.drop(columns=["image"]).groupby("variant").mean(numeric_only=True)[
            ["psnr", "ssim", "edge_align", "micro_tex", "micro_tex_ref"]
        ].round(3).to_string())
    print(f"\nDone -> {REPORT_DIR}")


if __name__ == "__main__":
    main()
