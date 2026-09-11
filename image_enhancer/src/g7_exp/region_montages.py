"""G7 six-region crop montages for the two-board photo (JJ GOLD, GRT,
car/plate, road barrier, buildings, sky/background), comparing Original vs
Candidate A+ vs each G7 candidate (including the Real-ESRGAN diagnostic).

Reads the already-rendered full-frame PNGs from
phase1_enhancement/enhanced/g7_exp/two_board_<candidate>.png (written by
run.py), so it can be re-run cheaply without re-computing any candidate.

Isolated experiment. Does NOT touch enhance.py, g6_exp, or production.
"""
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import enhance  # noqa: E402
from restore_exp.plan_abc import build_montage  # noqa: E402

SAVE_DIR = ROOT / "enhanced" / "g7_exp"
REPORT_DIR = ROOT / "reports" / "g7_exp"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TWO_BOARD_IMG = ROOT.parent / "backend" / ".deckstore" / \
    "dreamland-ventures-chennai-sivagangai-ooh-proposal" / "uploads" / \
    "dreamland-ventures-chennai-01.jpeg"

# All boxes in NATIVE (1374x773) pixel coords, scaled to the 4K (3840x2160)
# output frame below (scale factor = 3840/1374 = 2.795...).
NATIVE_BOXES = {
    "jj_gold": (543, 131, 247, 171),   # given (production billboard box)
    "grt": (502, 316, 192, 110),        # given (production billboard box)
    "car_plate": (430, 520, 160, 70),   # rear license plate of the black SUV
    "barrier": (120, 460, 180, 110),    # black/white striped road barrier
    "buildings": (1180, 290, 190, 110), # construction crane + apartment blocks
    "sky": (780, 10, 280, 120),          # clear sky / clouds, no billboard edge
}

CANDIDATES = ["A", "A+", "G7-RL", "G7-MS", "G7-GF", "G7-WIDE", "G7-GAN"]
LABELS = {
    "Original": "Original (faithful 4K)",
    "A": "A (production baseline)",
    "A+": "A+ (G6 whole-frame contrast)",
    "G7-RL": "G7-RL (full-frame RL deblur)",
    "G7-MS": "G7-MS (multi-scale detail)",
    "G7-GF": "G7-GF (guided/bilateral detail)",
    "G7-WIDE": "G7-WIDE (+-20 F3 envelope)",
    "G7-GAN": "G7-GAN (Real-ESRGAN, REJECTED)",
}
COLORS = {
    "Original": (60, 200, 255), "A": (255, 80, 120), "A+": (120, 220, 120),
    "G7-RL": (255, 200, 40), "G7-MS": (160, 160, 255), "G7-GF": (255, 120, 255),
    "G7-WIDE": (80, 220, 220), "G7-GAN": (60, 60, 220),
}


def native_to_4k(box, w0):
    s = enhance.OUT_W / w0
    x, y, w, h = box
    return (int(round(x * s)), int(round(y * s)),
           int(round(w * s)), int(round(h * s)))


def crop(img, box4):
    x, y, w, h = box4
    x = max(0, min(x, img.shape[1] - 1))
    y = max(0, min(y, img.shape[0] - 1))
    w = min(w, img.shape[1] - x)
    h = min(h, img.shape[0] - y)
    return img[y:y + h, x:x + w]


def main():
    orig = cv2.imread(str(TWO_BOARD_IMG), cv2.IMREAD_COLOR)
    W0 = orig.shape[1]
    faithful = enhance.simple_upscale(orig, (enhance.OUT_W, enhance.OUT_H))

    imgs = {"Original": faithful}
    for name in CANDIDATES:
        p = SAVE_DIR / f"two_board_{name}.png"
        if not p.exists():
            print(f"WARNING: missing {p}, skipping from region montages")
            continue
        imgs[name] = cv2.imread(str(p), cv2.IMREAD_COLOR)

    for region, native_box in NATIVE_BOXES.items():
        box4 = native_to_4k(native_box, W0)
        panels = []
        for name in ["Original"] + CANDIDATES:
            if name not in imgs:
                continue
            c = crop(imgs[name], box4)
            panels.append((c, LABELS.get(name, name), COLORS.get(name, (255, 255, 255))))
        out_path = REPORT_DIR / f"g7_region_{region}.png"
        build_montage(panels, out_path, scale=3.0)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
