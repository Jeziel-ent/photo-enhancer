"""G9 benchmark harness (isolated R&D, NOT production).

Compares the CURRENT PRODUCTION BASELINE (enhance.final_enhance) against
G9 (g9_exp.recipe_g9.g9_enhance) on:
  - whole image: no-reference sharpness/noise/contrast, PSNR/SSIM/edge-
    align of G9 vs baseline (content-preservation) and of both vs the
    faithful original upscale (fidelity-to-source).
  - named crops on 1.jpeg (has both JJ GOLD and GRT boards): jj_gold, grt,
    face, logo, text, license_plate, vehicle_edges, building_road, sky.
  - the single verified billboard box on every other original (2-6.jpeg),
    for broader robustness evidence.
  - determinism: two G9 runs on the same image, byte-exact diff.

Writes phase1_enhancement/reports/g9_exp/{full,crops}/*.png + summary.json.
Never touches enhance.py, regions.json, or anything outside
phase1_enhancement/src/g9_exp and phase1_enhancement/reports/g9_exp.

Usage:
  .venv/Scripts/python phase1_enhancement/src/g9_exp/run.py
  .venv/Scripts/python phase1_enhancement/src/g9_exp/run.py --rois-only   # just dump ROI crops of the ORIGINAL for visual sanity-check, no model inference
  .venv/Scripts/python phase1_enhancement/src/g9_exp/run.py --images 1.jpeg
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import enhance as prod_enhance  # noqa: E402
from quality import evaluate_similarity, compute_no_reference  # noqa: E402
from g9_exp import recipe_g9  # noqa: E402

ORIGINALS = ROOT / "originals"
REGIONS_JSON = ROOT / "regions.json"
OUT_DIR = ROOT / "reports" / "g9_exp"
FULL_DIR = OUT_DIR / "full"
CROPS_DIR = OUT_DIR / "crops"
ROI_PREVIEW_DIR = OUT_DIR / "roi_preview"

# Native-resolution ROI boxes (x, y, w, h) on the ORIGINAL photo, hand-picked
# by visual inspection (same manual-verification policy as regions.json --
# no auto-detection). "jj_gold" reuses the already-verified regions.json
# box for 1.jpeg; the rest are G9-diagnostic crops added for this
# experiment only and are NOT written back to regions.json.
ROIS = {
    "1.jpeg": {
        "jj_gold": (524, 108, 291, 220),       # == regions.json verified box
        "grt": (520, 300, 230, 155),
        "face": (555, 145, 85, 95),            # JJ GOLD ad model's face
        "logo": (655, 195, 45, 40),            # JJ Gold icon mark
        "text": (524, 268, 291, 30),           # phone-number bar
        "license_plate": (452, 558, 90, 24),   # black SUV, "KL 46 X 9581"
        "vehicle_edges": (395, 438, 400, 235), # black SUV full silhouette
        "building_road": (1045, 255, 329, 210),
        "sky": (0, 0, 1374, 85),
    },
}


def clamp_box(box, W, H):
    x, y, w, h = box
    x = max(0, min(x, W - 1)); y = max(0, min(y, H - 1))
    w = min(w, W - x); h = min(h, H - y)
    return (x, y, w, h)


def crop(img, box):
    x, y, w, h = box
    return img[y:y + h, x:x + w]


def load_region_billboard_box(fname):
    cfg = json.loads(REGIONS_JSON.read_text(encoding="utf-8-sig"))
    entry = cfg.get(fname)
    if not entry:
        return None
    boxes = entry.get("boxes") or ([entry["box"]] if entry.get("box") else [])
    return tuple(boxes[0]) if boxes else None


def dump_roi_previews(images):
    """No model inference -- just crop the ORIGINAL at each ROI box and
    save a PNG so the boxes can be eyeballed/corrected before a full run."""
    ROI_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    for fname in images:
        img = cv2.imread(str(ORIGINALS / fname), cv2.IMREAD_COLOR)
        H, W = img.shape[:2]
        rois = dict(ROIS.get(fname, {}))
        bb = load_region_billboard_box(fname)
        if bb and "jj_gold" not in rois:
            rois["billboard"] = bb
        overlay = img.copy()
        for name, box in rois.items():
            box = clamp_box(box, W, H)
            c = crop(img, box)
            cv2.imwrite(str(ROI_PREVIEW_DIR / f"{Path(fname).stem}_{name}.png"), c)
            x, y, w, h = box
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(overlay, name, (x, max(0, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(ROI_PREVIEW_DIR / f"{Path(fname).stem}_overlay.png"), overlay)
        print(f"wrote ROI previews for {fname} -> {ROI_PREVIEW_DIR}")


def metrics_for(img):
    if img is None or img.size == 0:
        return None
    nr = compute_no_reference(img)
    return {"sharpness": round(nr["sharpness"], 2),
            "noise": round(nr["noise"], 3),
            "contrast": round(nr["contrast"], 4),
            "edge_density": round(nr["edge_density"], 4),
            "w": int(img.shape[1]), "h": int(img.shape[0])}


def pixel_diff(a, b):
    d = cv2.absdiff(a, b)
    return {"mean_abs_diff": float(d.mean()),
            "max_abs_diff": int(d.max()),
            "pct_px_changed_gt2": float((d.max(axis=2) > 2).mean() * 100.0)}


def determinism_check(img, device):
    out1 = recipe_g9.g9_enhance(img, device=device)
    out2 = recipe_g9.g9_enhance(img, device=device)
    d = cv2.absdiff(out1, out2)
    return {"identical": bool(d.max() == 0),
            "max_abs_diff": int(d.max()),
            "nonzero_px_channels": int(np.count_nonzero(d))}


def process_image(fname, device):
    os.environ["BILLBOARD_IMAGE"] = fname
    os.environ.pop("REGIONS_CONFIG", None)
    prod_enhance._REGION_CONFIG = None
    img = cv2.imread(str(ORIGINALS / fname), cv2.IMREAD_COLOR)
    H, W = img.shape[:2]
    target = (prod_enhance.OUT_W, prod_enhance.OUT_H)
    s = target[0] / W

    faithful = prod_enhance.simple_upscale(img, target)

    t0 = time.perf_counter()
    baseline, base_stages = prod_enhance.final_enhance(img, target=target, return_stages=True)
    t_base = time.perf_counter() - t0

    t0 = time.perf_counter()
    g9_out, g9_stages = recipe_g9.g9_enhance(
        img, device=device, target=target, return_stages=True,
        precomputed_baseline=(baseline, base_stages))
    t_g9 = time.perf_counter() - t0

    cv2.imwrite(str(FULL_DIR / f"{Path(fname).stem}_baseline.png"), baseline)
    cv2.imwrite(str(FULL_DIR / f"{Path(fname).stem}_g9.png"), g9_out)

    result = {
        # t_g9 is the INCREMENTAL cost only (precomputed_baseline is passed
        # in, so g9_enhance does not re-run final_enhance) -- it is already
        # directly comparable as "extra time G9 adds on top of baseline".
        "elapsed_s": {"baseline": round(t_base, 2), "g9_extra_s": round(t_g9, 2)},
        "peak_vram_mb": base_stages.get("peak_vram_mb"),
        "g9_stats": g9_stages.get("g9_stats"),
        "whole_image": {
            "baseline_vs_faithful": evaluate_similarity(faithful, baseline),
            "g9_vs_faithful": evaluate_similarity(faithful, g9_out),
            "g9_vs_baseline": evaluate_similarity(baseline, g9_out),
            "g9_vs_baseline_pixeldiff": pixel_diff(baseline, g9_out),
            "baseline_no_ref": metrics_for(baseline),
            "g9_no_ref": metrics_for(g9_out),
        },
        "crops": {},
    }

    rois = dict(ROIS.get(fname, {}))
    bb = load_region_billboard_box(fname)
    if bb and "jj_gold" not in rois:
        rois["billboard"] = bb

    for name, box in rois.items():
        vb = clamp_box(box, W, H)
        bx, by, bw, bh = vb
        box4 = (int(round(bx * s)), int(round(by * s)),
                int(round(bw * s)), int(round(bh * s)))
        base_c = crop(baseline, box4)
        g9_c = crop(g9_out, box4)
        if base_c.size == 0 or g9_c.size == 0:
            continue
        cv2.imwrite(str(CROPS_DIR / f"{Path(fname).stem}_{name}_baseline.png"), base_c)
        cv2.imwrite(str(CROPS_DIR / f"{Path(fname).stem}_{name}_g9.png"), g9_c)
        result["crops"][name] = {
            "box4": list(box4),
            "baseline_no_ref": metrics_for(base_c),
            "g9_no_ref": metrics_for(g9_c),
            "g9_vs_baseline_pixeldiff": pixel_diff(base_c, g9_c),
            "g9_vs_baseline_similarity": (evaluate_similarity(base_c, g9_c)
                                          if base_c.shape == g9_c.shape else None),
        }

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="*", default=None)
    ap.add_argument("--rois-only", action="store_true")
    args = ap.parse_args()

    all_images = sorted(p.name for p in ORIGINALS.glob("*.jpeg"))
    images = args.images if args.images else all_images

    if args.rois_only:
        dump_roi_previews(images)
        return

    FULL_DIR.mkdir(parents=True, exist_ok=True)
    CROPS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    all_results = {}
    for fname in images:
        print(f"== {fname} ==")
        res = process_image(fname, device)
        print(f"  baseline {res['elapsed_s']['baseline']}s | g9 extra "
              f"+{res['elapsed_s']['g9_extra_s']}s | peak_vram={res['peak_vram_mb']}MB")
        wi = res["whole_image"]
        print(f"  g9_vs_baseline: SSIM={wi['g9_vs_baseline']['ssim']:.4f} "
              f"PSNR={wi['g9_vs_baseline']['psnr']:.2f} "
              f"edge_align={wi['g9_vs_baseline']['edge_align']:.4f} "
              f"mean_abs_diff={wi['g9_vs_baseline_pixeldiff']['mean_abs_diff']:.3f} "
              f"sharp: base={wi['baseline_no_ref']['sharpness']:.1f} "
              f"g9={wi['g9_no_ref']['sharpness']:.1f}")
        all_results[fname] = res

    print("== determinism check (1.jpeg) ==")
    os.environ["BILLBOARD_IMAGE"] = "1.jpeg"
    os.environ.pop("REGIONS_CONFIG", None)
    prod_enhance._REGION_CONFIG = None
    det_img = cv2.imread(str(ORIGINALS / "1.jpeg"), cv2.IMREAD_COLOR)
    det = determinism_check(det_img, device)
    print(f"  {det}")
    all_results["determinism_1jpeg"] = det

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(all_results, indent=2))
    print(f"\nwrote {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
