"""Higher-zoom comparison crops for visual inspection.

Panels always: Original (faithful 4K) | D1-weak | E1 | E3.
Uses the SAME source regions as the fidelity_exp run (regions.json billboard
boxes, YuNet face on 1.jpeg, detail_window fine-texture strips). No
enhancement algorithm is modified - only existing saved outputs are
re-composited at higher zoom for human inspection.

Output: reports/pro_exp/fidelity_exp/zoom/zoom_*.png
"""

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import enhance  # noqa: E402
from restore_exp import faces as faces_mod  # noqa: E402
from restore_exp import pipeline  # noqa: E402
from restore_exp.metrics_ext import crop_rect  # noqa: E402
from restore_exp.plan_abc import build_montage  # noqa: E402
from pro_exp import pipeline2  # noqa: E402

BASE_DIR = ROOT / "enhanced" / "pro_exp" / "d1_sweep"
FE_DIR = ROOT / "enhanced" / "pro_exp" / "fidelity_exp"
OUT = ROOT / "reports" / "pro_exp" / "fidelity_exp" / "zoom"

PANELS = ["Original", "D1-weak", "E1", "E3"]
LABELS = {
    "Original": "Original (faithful 4K)",
    "D1-weak": "D1-weak (K=0.30) baseline",
    "E1": "E1 TexReinject",
    "E3": "E3 Combined (USM+Tex)",
}
COLORS = {
    "Original": (60, 200, 255),
    "D1-weak": (200, 160, 255),
    "E1": (110, 255, 180),
    "E3": (80, 120, 255),
}

CAP_H = 1100      # cap each zoomed panel height (pixels)
MIN_SCALE = 2.0   # never zoom below this


def zoom_montage(panels, out_path):
    imgs = [(img, "", (0, 0, 0)) for img, _, _ in panels if img is not None]
    imgs = [p for p in imgs if p[0].shape[0] > 4 and p[0].shape[1] > 4]
    if not imgs:
        return
    h0 = max(p[0].shape[0] for p in imgs)
    scale = MIN_SCALE
    if h0 * scale > CAP_H:
        scale = CAP_H / float(h0)
    build_montage(panels, out_path, scale=scale)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    images = sorted(p.stem for p in (ROOT / "originals").glob("*.jpeg"))

    for stem in images:
        print(f"== {stem}.jpeg ==")
        orig = cv2.imread(str(ROOT / "originals" / f"{stem}.jpeg"), cv2.IMREAD_COLOR)
        faithful = enhance.simple_upscale(orig, (3840, 2160))
        H, W = orig.shape[:2]
        v = {
            "Original": faithful,
            "D1-weak": cv2.imread(str(BASE_DIR / f"D1-weak_{stem}.png"), cv2.IMREAD_COLOR),
            "E1": cv2.imread(str(FE_DIR / f"E1_{stem}.png"), cv2.IMREAD_COLOR),
            "E3": cv2.imread(str(FE_DIR / f"E3_{stem}.png"), cv2.IMREAD_COLOR),
        }
        dets = faces_mod.detect_faces(orig)
        boxes = pipeline.boxes_for(stem)

        if boxes:
            bx, by, bw, bh = pipeline.scale_rect(boxes[0], W)
            panels = [(crop_rect(img, (bx, by, bw, bh)), LABELS[k], COLORS[k])
                      for k, img in v.items()]
            zoom_montage(panels, OUT / f"zoom_billboard_{stem}.png")

        for fi, d in enumerate(dets[:1]):
            if stem != "1" and fi == 0:
                continue
            fx, fy, fw, fh = pipeline.scale_rect((d["x"], d["y"], d["w"], d["h"]), W)
            panels = [(crop_rect(img, (fx, fy, fw, fh)), LABELS[k], COLORS[k])
                      for k, img in v.items()]
            zoom_montage(panels, OUT / f"zoom_face{fi}_{stem}.png")

        ex = boxes + [(d["x"], d["y"], d["w"], d["h"]) for d in dets[:3]]
        dw = pipeline2.detail_window(orig, ex, win=160)
        s = pipeline2.OUT_W / W
        rect4 = tuple(int(round(vv * s)) for vv in dw)
        panels = [(crop_rect(img, rect4), LABELS[k], COLORS[k])
                  for k, img in v.items()]
        zoom_montage(panels, OUT / f"zoom_detail_{stem}.png")

    files = sorted(p.name for p in OUT.glob("zoom_*.png"))
    print(f"\n{len(files)} zoom crops -> {OUT}")
    for f in files:
        print("  ", f)


if __name__ == "__main__":
    main()