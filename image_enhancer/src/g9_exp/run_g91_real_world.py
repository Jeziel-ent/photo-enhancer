"""G9.1-B FINAL real-world validation (isolated R&D, NOT production).

Thin driver on top of the existing, unmodified G9.1 harness (run_g91.py ->
run.py -> recipe_g9.py) -- no new enhancement pipeline. It only:
  1. Locks candidate B params (strength=0.30, clip=+-5, gate=0.40).
  2. Runs the real JJ GOLD + GRT benchmark image (1.jpeg, the one used in
     the G8 investigation) through baseline vs G9.1-B, using the SAME
     hand-verified ROI boxes already defined in run.py's ROIS["1.jpeg"]
     (jj_gold is the regions.json-verified box; the rest are the same
     G8/G9 hand-picked diagnostic boxes: grt, face, logo, text,
     license_plate, vehicle_edges, building_road, sky).
  3. Additionally renders side-by-side and enlarged crops for visual
     sign-off, and runs an independent second determinism pass.

Writes ONLY under phase1_enhancement/reports/g9_1_exp/real_world/.
Touches nothing in enhance.py, Phase 2, Phase 3, backend, or frontend.

Usage:
  .venv/Scripts/python phase1_enhancement/src/g9_exp/run_g91_real_world.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import enhance as prod_enhance  # noqa: E402
from g9_exp import recipe_g9  # noqa: E402
from g9_exp import run as g9run  # noqa: E402
from g9_exp import run_g91  # noqa: E402

IMAGE = "1.jpeg"
OUT_DIR = ROOT / "reports" / "g9_1_exp" / "real_world"
FULL_DIR = OUT_DIR / "full"
CROPS_DIR = OUT_DIR / "crops"
SIDEBYSIDE_DIR = OUT_DIR / "side_by_side"
ENLARGED_DIR = OUT_DIR / "enlarged"

ROI_ORDER = [
    ("jj_gold", "1. JJ GOLD billboard"),
    ("grt", "2. GRT billboard"),
    ("face", "3. JJ GOLD face"),
    ("logo", "4. logo"),
    ("text", "5. phone/text area"),
    ("license_plate", "6. license plate"),
    ("vehicle_edges", "7. vehicle edges"),
    ("building_road", "8. building/road"),
    ("sky", "9. sky/background"),
]

LABEL_BAR_H = 36


def label_bar(w, text):
    bar = np.full((LABEL_BAR_H, w, 3), 255, np.uint8)
    cv2.putText(bar, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 0), 2, cv2.LINE_AA)
    return bar


def side_by_side(base_img, g9_img, title):
    h = max(base_img.shape[0], g9_img.shape[0])
    w = base_img.shape[1] + g9_img.shape[1] + 8
    canvas = np.full((h, w, 3), 200, np.uint8)
    canvas[:base_img.shape[0], :base_img.shape[1]] = base_img
    canvas[:g9_img.shape[0], base_img.shape[1] + 8:base_img.shape[1] + 8 + g9_img.shape[1]] = g9_img
    bar = label_bar(w, f"{title}  |  LEFT=baseline  RIGHT=G9.1-B")
    return np.vstack([bar, canvas])


def enlarge(img, min_dim=320, max_scale=6):
    h, w = img.shape[:2]
    scale = min(max_scale, max(1, int(round(min_dim / max(h, w, 1)))))
    if scale <= 1:
        return img
    return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)


def main():
    for d in (FULL_DIR, CROPS_DIR, SIDEBYSIDE_DIR, ENLARGED_DIR):
        d.mkdir(parents=True, exist_ok=True)

    params = run_g91.CANDIDATES["B"]
    assert params == {"strength": 0.30, "clip": 5.0, "gate": 0.40}
    run_g91.apply_params(params)
    print(f"G9.1-B FINAL real-world validation on {IMAGE}: params={params}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    # --- Run 1 (primary result) ---
    result = run_g91.process_image(IMAGE, device, FULL_DIR, CROPS_DIR)

    # --- Determinism: two INDEPENDENT full g9_enhance runs on the same
    # original image (fresh baseline recompute each time -- not reusing
    # precomputed_baseline -- for a true end-to-end determinism check). ---
    import os
    os.environ["BILLBOARD_IMAGE"] = IMAGE
    os.environ.pop("REGIONS_CONFIG", None)
    prod_enhance._REGION_CONFIG = None
    img = cv2.imread(str(g9run.ORIGINALS / IMAGE), cv2.IMREAD_COLOR)

    run_a = recipe_g9.g9_enhance(img, device=device)
    run_b = recipe_g9.g9_enhance(img, device=device)
    d = cv2.absdiff(run_a, run_b)
    determinism = {
        "identical": bool(d.max() == 0),
        "max_abs_diff": int(d.max()),
        "nonzero_px_channels": int(np.count_nonzero(d)),
    }
    print(f"determinism (2 independent runs): {determinism}")
    result["determinism_2_independent_runs"] = determinism

    # --- Side-by-side + enlarged crops for visual sign-off ---
    W, H = prod_enhance.OUT_W, prod_enhance.OUT_H
    baseline_full = cv2.imread(str(FULL_DIR / f"{Path(IMAGE).stem}_baseline.png"))
    g9_full = cv2.imread(str(FULL_DIR / f"{Path(IMAGE).stem}_g9.png"))
    scale_disp = 1600 / baseline_full.shape[1]
    b_small = cv2.resize(baseline_full, None, fx=scale_disp, fy=scale_disp)
    g_small = cv2.resize(g9_full, None, fx=scale_disp, fy=scale_disp)
    cv2.imwrite(str(SIDEBYSIDE_DIR / "00_full_frame.png"),
                side_by_side(b_small, g_small, "FULL FRAME (downscaled for display)"))

    for roi_name, label in ROI_ORDER:
        bpath = CROPS_DIR / f"{Path(IMAGE).stem}_{roi_name}_baseline.png"
        gpath = CROPS_DIR / f"{Path(IMAGE).stem}_{roi_name}_g9.png"
        if not bpath.exists() or not gpath.exists():
            print(f"WARNING: missing crop for {roi_name}")
            continue
        bimg = cv2.imread(str(bpath))
        gimg = cv2.imread(str(gpath))
        cv2.imwrite(str(SIDEBYSIDE_DIR / f"{roi_name}.png"), side_by_side(bimg, gimg, label))
        be = enlarge(bimg)
        ge = enlarge(gimg)
        cv2.imwrite(str(ENLARGED_DIR / f"{roi_name}_baseline_enlarged.png"), be)
        cv2.imwrite(str(ENLARGED_DIR / f"{roi_name}_g9_enlarged.png"), ge)
        cv2.imwrite(str(SIDEBYSIDE_DIR / f"{roi_name}_enlarged.png"), side_by_side(be, ge, label))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(result, indent=2))
    print(f"\nwrote {OUT_DIR / 'summary.json'}")
    print(f"side-by-side images: {SIDEBYSIDE_DIR}")
    print(f"enlarged crops: {ENLARGED_DIR}")


if __name__ == "__main__":
    main()
