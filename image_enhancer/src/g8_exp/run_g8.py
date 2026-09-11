"""G8 diagnostic: trace WHERE useful visual detail is lost in the current
production `final_enhance()` pipeline (see phase1_enhancement/src/enhance.py).

Pure diagnosis. Does NOT modify enhance.py or any production/Phase2/Phase3
code. Does NOT test new candidate algorithms -- only instruments/observes the
EXISTING pipeline stages via return_stages=True plus a local monkey-patch
(in THIS file only) that intercepts the native billboard-crop-before-
composite, which final_enhance() does not expose directly.

Usage:
  .venv/Scripts/python src/g8_exp/run_g8.py
"""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import enhance  # noqa: E402
from quality import compute_no_reference  # noqa: E402

IMG_PATH = ROOT.parent / "backend" / ".deckstore" / \
    "dreamland-ventures-chennai-sivagangai-ooh-proposal" / "uploads" / \
    "dreamland-ventures-chennai-01.jpeg"

STAGES_DIR = ROOT / "reports" / "g8_exp" / "stages"
CROPS_DIR = ROOT / "reports" / "g8_exp" / "crops"
SAVE_DIR = ROOT / "enhanced" / "g8_exp"
STAGES_DIR.mkdir(parents=True, exist_ok=True)
CROPS_DIR.mkdir(parents=True, exist_ok=True)
SAVE_DIR.mkdir(parents=True, exist_ok=True)

NATIVE_BOXES = [
    ("jj_gold", (543, 131, 247, 171)),
    ("grt", (502, 316, 192, 110)),
]

# Six diagnostic regions in NATIVE (1374x773) pixel coords.
REGIONS = {
    "jj_gold": (543, 131, 247, 171),
    "grt": (502, 316, 192, 110),
    "foreground_car": (380, 470, 260, 150),
    "license_plate": (450, 540, 100, 30),
    "road_barrier": (120, 460, 180, 110),
    "buildings": (1180, 290, 190, 110),
}

STAGE_ORDER = ["original", "faithful", "d1_weak", "f3", "f3_ms", "f3_plus",
              "billboard_crop", "final"]
STAGE_LABELS = {
    "original": "Original (native)",
    "faithful": "Faithful 4K (Lanczos)",
    "d1_weak": "D1-weak (Restormer+SwinIR)",
    "f3": "F3-natural (fusion)",
    "f3_ms": "G7-MS (multiscale detail)",
    "f3_plus": "A+ (whole-frame detail)",
    "billboard_crop": "Billboard crop (pre-composite)",
    "final": "FINAL composite output",
}


def setup_regions_env():
    """Point enhance.py's verified-regions lookup at the two-board boxes,
    exactly as g7_exp/run.py did for the same benchmark image."""
    img = cv2.imread(str(IMG_PATH), cv2.IMREAD_COLOR)
    H, W = img.shape[:2]
    regions_path = SAVE_DIR / "two_board_regions.json"
    name = "two_board.jpeg"
    regions_path.write_text(json.dumps(
        {name: {"boxes": [list(b) for _, b in NATIVE_BOXES],
                "image_size": [W, H]}}))
    os.environ["REGIONS_CONFIG"] = str(regions_path)
    os.environ["BILLBOARD_IMAGE"] = name
    enhance._REGION_CONFIG = None
    return img


def native_to_4k(box, w0):
    s = enhance.OUT_W / w0
    x, y, w, h = box
    return (int(round(x * s)), int(round(y * s)),
           int(round(w * s)), int(round(h * s)))


def crop(img, box):
    x, y, w, h = box
    x = max(0, min(x, img.shape[1] - 1))
    y = max(0, min(y, img.shape[0] - 1))
    w = min(w, img.shape[1] - x)
    h = min(h, img.shape[0] - y)
    return img[y:y + h, x:x + w]


def run_instrumented(img):
    """Run final_enhance() once, monkey-patching (in this script only, NOT
    in enhance.py) the internal box-compositing helper so we can capture the
    native-resolution billboard crop's result BEFORE it is feathered into
    the frame -- final_enhance()'s return_stages dict does not expose this
    intermediate on its own."""
    captured = []  # list of (board_bgr copy, box4)
    orig_composite = enhance._final_composite_box_bgr

    def capture_and_composite(frame_bgr, board_bgr, box4):
        captured.append((board_bgr.copy(), box4))
        return orig_composite(frame_bgr, board_bgr, box4)

    enhance._final_composite_box_bgr = capture_and_composite
    try:
        out, stages = enhance.final_enhance(img, return_stages=True)
    finally:
        enhance._final_composite_box_bgr = orig_composite
    return out, stages, captured


def save_stage_images(orig, out, stages, captured):
    cv2.imwrite(str(STAGES_DIR / "a_original.png"), orig)
    cv2.imwrite(str(STAGES_DIR / "b_faithful.png"), stages["faithful"])
    cv2.imwrite(str(STAGES_DIR / "c_d1_weak.png"), stages["d1_weak"])
    cv2.imwrite(str(STAGES_DIR / "d_f3.png"), stages["f3"])
    cv2.imwrite(str(STAGES_DIR / "e_f3_ms.png"), stages["f3_ms"])
    cv2.imwrite(str(STAGES_DIR / "f_f3_plus.png"), stages["f3_plus"])
    cv2.imwrite(str(STAGES_DIR / "g_final.png"), out)
    for i, (board_bgr, box4) in enumerate(captured):
        cv2.imwrite(str(STAGES_DIR / f"h_billboard_crop_{i}.png"), board_bgr)
    print(f"peak_vram_mb={stages.get('peak_vram_mb')} device={stages.get('device')}")
    print(f"boxes4={stages.get('boxes4')}")
    print(f"psf_sigmas={stages.get('psf_sigmas')}")
    print(f"defocus_boxes={stages.get('defocus_boxes')}")
    return stages.get("boxes4", [])


def build_stage_crops(region_name, native_box, orig, stages, out, captured,
                      boxes4, w0):
    """Return an ordered dict of stage_name -> native-resolution-equivalent
    BGR crop for one diagnostic region (each stage's own crop pulled at that
    stage's own resolution -- 4K stages use the 4K-scaled box, the native
    original/billboard-crop stages use native coords directly)."""
    box4 = native_to_4k(native_box, w0)
    out_crops = {}
    out_crops["original"] = crop(orig, native_box)
    out_crops["faithful"] = crop(stages["faithful"], box4)
    out_crops["d1_weak"] = crop(stages["d1_weak"], box4)
    out_crops["f3"] = crop(stages["f3"], box4)
    out_crops["f3_ms"] = crop(stages["f3_ms"], box4)
    out_crops["f3_plus"] = crop(stages["f3_plus"], box4)

    # billboard crop pre-composite: only for jj_gold / grt, and only if that
    # region's own box4 matches one of the captured composited boxes closely
    billboard_crop_img = None
    if region_name in ("jj_gold", "grt"):
        for board_bgr, cbox4 in zip([c[0] for c in captured], boxes4):
            # match by IoU-ish proximity (same box the composite used)
            if _boxes_close(cbox4, box4):
                billboard_crop_img = board_bgr
                break
    out_crops["billboard_crop"] = billboard_crop_img

    out_crops["final"] = crop(out, box4)
    return out_crops


def _boxes_close(b1, b2, tol=8):
    return all(abs(a - c) <= tol for a, c in zip(b1, b2))


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


def build_region_montage(region_name, crops_dict, out_path, pct, interp,
                         interp_name, max_cols=4):
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
    grid_rows = []
    for r in range(rows):
        row_panels = padded[r * cols:(r + 1) * cols]
        row_img = np.hstack(row_panels)
        grid_rows.append(row_img)
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
    img = setup_regions_env()
    W0 = img.shape[1]
    print(f"Loaded {IMG_PATH} shape={img.shape}")

    out, stages, captured = run_instrumented(img)
    boxes4 = save_stage_images(img, out, stages, captured)

    all_metrics = {}
    for region_name, native_box in REGIONS.items():
        crops_dict = build_stage_crops(region_name, native_box, img, stages,
                                       out, captured, boxes4, W0)
        for pct, interp, iname in [(300, cv2.INTER_NEAREST, "nearest"),
                                   (300, cv2.INTER_LANCZOS4, "lanczos"),
                                   (400, cv2.INTER_NEAREST, "nearest"),
                                   (400, cv2.INTER_LANCZOS4, "lanczos")]:
            out_path = CROPS_DIR / f"{region_name}_{pct}pct_{iname}.png"
            build_region_montage(region_name, crops_dict, out_path, pct,
                                 interp, iname)
        all_metrics[region_name] = region_metrics_table(crops_dict)

    metrics_path = ROOT / "reports" / "g8_exp" / "stage_metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2))
    print(f"wrote {metrics_path}")

    # print a compact table to stdout for the report writer
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
