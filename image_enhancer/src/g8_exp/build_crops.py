"""Rebuild region crop montages from the already-saved stage PNGs (written
by run_g8.py) without re-running the GPU pipeline. Diagnostic only."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import enhance  # noqa: E402
from quality import compute_no_reference  # noqa: E402

STAGES_DIR = ROOT / "reports" / "g8_exp" / "stages"
CROPS_DIR = ROOT / "reports" / "g8_exp" / "crops"
CROPS_DIR.mkdir(parents=True, exist_ok=True)

W0 = 1374  # native width of the two-board benchmark photo

REGIONS = {
    "jj_gold": (543, 131, 247, 171),
    "grt": (502, 316, 192, 110),
    "foreground_car": (380, 470, 260, 150),
    "license_plate": (450, 540, 100, 30),
    "road_barrier": (120, 460, 180, 110),
    "buildings": (1180, 290, 190, 110),
}
BOXES4 = {
    "jj_gold": (1518, 366, 690, 478),
    "grt": (1403, 883, 537, 307),
}
BILLBOARD_CROP_FILE = {
    "jj_gold": "h_billboard_crop_0.png",
    "grt": "h_billboard_crop_1.png",
}

STAGE_ORDER = ["original", "faithful", "d1_weak", "f3", "f3_ms", "f3_plus",
              "billboard_crop", "final"]
STAGE_LABELS = {
    "original": "Original (native)",
    "faithful": "Faithful 4K",
    "d1_weak": "D1-weak",
    "f3": "F3-natural",
    "f3_ms": "G7-MS",
    "f3_plus": "A+ (whole-frame)",
    "billboard_crop": "Billboard crop (pre-composite)",
    "final": "FINAL composite",
}
STAGE_FILES = {
    "original": "a_original.png",
    "faithful": "b_faithful.png",
    "d1_weak": "c_d1_weak.png",
    "f3": "d_f3.png",
    "f3_ms": "e_f3_ms.png",
    "f3_plus": "f_f3_plus.png",
    "final": "g_final.png",
}


def native_to_4k(box, w0):
    s = enhance.OUT_W / w0
    x, y, w, h = box
    return (int(round(x * s)), int(round(y * s)),
           int(round(w * s)), int(round(h * s)))


def crop(img, box):
    if img is None:
        return None
    x, y, w, h = box
    x = max(0, min(x, img.shape[1] - 1))
    y = max(0, min(y, img.shape[0] - 1))
    w = min(w, img.shape[1] - x)
    h = min(h, img.shape[0] - y)
    return img[y:y + h, x:x + w]


def upscale_for_view(img, factor, interp):
    if img is None:
        return None
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * factor), int(h * factor)),
                      interpolation=interp)


def label_panel(img, text, color=(60, 220, 60)):
    if img is None:
        return None
    label_h = 26
    p = cv2.copyMakeBorder(img, label_h, 4, 4, 4, cv2.BORDER_CONSTANT,
                           value=(20, 20, 20))
    cv2.putText(p, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
               cv2.LINE_AA)
    return p


def build_region_montage(crops_dict, out_path, pct, interp, interp_name,
                         max_cols=4):
    panels = []
    for stage in STAGE_ORDER:
        img = crops_dict.get(stage)
        if img is None:
            continue
        up = upscale_for_view(img, pct / 100.0, interp)
        p = label_panel(up, f"{STAGE_LABELS[stage]} ({pct}% {interp_name})")
        panels.append(p)
    if not panels:
        return
    n = len(panels)
    cols = min(max_cols, n)
    rows = int(np.ceil(n / cols))
    ph = max(p.shape[0] for p in panels)
    pw = max(p.shape[1] for p in panels)
    padded = []
    for p in panels:
        pad_b = ph - p.shape[0]
        pad_r = pw - p.shape[1]
        p = cv2.copyMakeBorder(p, 0, pad_b, 0, pad_r, cv2.BORDER_CONSTANT,
                               value=(20, 20, 20))
        padded.append(p)
    while len(padded) < rows * cols:
        padded.append(np.full((ph, pw, 3), 20, np.uint8))
    grid_rows = [np.hstack(padded[r * cols:(r + 1) * cols]) for r in range(rows)]
    mont = np.vstack(grid_rows)
    cv2.imwrite(str(out_path), mont)
    print(f"wrote {out_path}  ({mont.shape[1]}x{mont.shape[0]})")


def region_metrics_table(crops_dict):
    rows = {}
    for stage in STAGE_ORDER:
        img = crops_dict.get(stage)
        if img is None:
            rows[stage] = None
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        sobel_mean = float(cv2.magnitude(gx, gy).mean())
        nr = compute_no_reference(img)
        rows[stage] = {
            "lap_var": round(lap_var, 2),
            "sobel_mean": round(sobel_mean, 2),
            "nr_sharpness": round(nr["sharpness"], 2),
            "noise": round(nr["noise"], 3),
        }
    return rows


def main():
    stage_imgs = {k: cv2.imread(str(STAGES_DIR / v), cv2.IMREAD_COLOR)
                 for k, v in STAGE_FILES.items()}
    billboard_imgs = {k: cv2.imread(str(STAGES_DIR / v), cv2.IMREAD_COLOR)
                      for k, v in BILLBOARD_CROP_FILE.items()}

    all_metrics = {}
    for region_name, native_box in REGIONS.items():
        box4 = native_to_4k(native_box, W0)
        crops_dict = {}
        crops_dict["original"] = crop(stage_imgs["original"], native_box)
        for st in ["faithful", "d1_weak", "f3", "f3_ms", "f3_plus", "final"]:
            crops_dict[st] = crop(stage_imgs[st], box4)
        crops_dict["billboard_crop"] = billboard_imgs.get(region_name)

        for pct, interp, iname in [(300, cv2.INTER_NEAREST, "nearest"),
                                   (300, cv2.INTER_LANCZOS4, "lanczos"),
                                   (400, cv2.INTER_NEAREST, "nearest"),
                                   (400, cv2.INTER_LANCZOS4, "lanczos")]:
            out_path = CROPS_DIR / f"{region_name}_{pct}pct_{iname}.png"
            build_region_montage(crops_dict, out_path, pct, interp, iname)
        all_metrics[region_name] = region_metrics_table(crops_dict)

    metrics_path = ROOT / "reports" / "g8_exp" / "stage_metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2))
    print(f"wrote {metrics_path}")

    for region_name, rows in all_metrics.items():
        print(f"\n== {region_name} ==")
        for stage in STAGE_ORDER:
            r = rows.get(stage)
            if r is None:
                continue
            print(f"  {stage:16s} lap_var={r['lap_var']:>10.1f}  "
                 f"sobel_mean={r['sobel_mean']:>7.2f}  noise={r['noise']}")


if __name__ == "__main__":
    main()
