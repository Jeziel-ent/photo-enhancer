"""G9.1-B production-integration validation driver (isolated R&D script).

Exercises the ACTUAL integration entry point (enhance_g91.final_enhance_g91)
-- not the raw recipe_g9.g9_enhance call used by the earlier tuning/real-
world scripts -- so this proves the integration plumbing itself (not just
the underlying math) on all 6 reference images plus the full 9-ROI JJ GOLD
+ GRT benchmark, with determinism and VRAM/time capture.

Reuses run.py's ROI boxes / crop / metrics helpers unmodified. Writes ONLY
under phase1_enhancement/reports/g9_1_exp/integration/. Does not touch
enhance.py, Phase 2, Phase 3, backend, or frontend.

Usage:
  .venv/Scripts/python phase1_enhancement/src/g9_exp/run_g91_integration.py
"""
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
import enhance_g91  # noqa: E402
from g9_exp import run as g9run  # noqa: E402
from quality import evaluate_similarity, compute_no_reference  # noqa: E402

OUT_DIR = ROOT / "reports" / "g9_1_exp" / "integration"
FULL_DIR = OUT_DIR / "full"
CROPS_DIR = OUT_DIR / "crops"


def metrics_for(im):
    if im is None or im.size == 0:
        return None
    nr = compute_no_reference(im)
    return {"sharpness": round(nr["sharpness"], 2), "noise": round(nr["noise"], 3),
            "contrast": round(nr["contrast"], 4), "edge_density": round(nr["edge_density"], 4),
            "w": int(im.shape[1]), "h": int(im.shape[0])}


def pixel_diff(a, b):
    d = cv2.absdiff(a, b)
    return {"mean_abs_diff": float(d.mean()), "max_abs_diff": int(d.max()),
            "pct_px_changed_gt2": float((d.max(axis=2) > 2).mean() * 100.0)}


def process_image(fname, device):
    os.environ["BILLBOARD_IMAGE"] = fname
    os.environ.pop("REGIONS_CONFIG", None)
    prod_enhance._REGION_CONFIG = None
    img = cv2.imread(str(g9run.ORIGINALS / fname), cv2.IMREAD_COLOR)
    H, W = img.shape[:2]
    target = (prod_enhance.OUT_W, prod_enhance.OUT_H)
    s = target[0] / W

    faithful = prod_enhance.simple_upscale(img, target)

    t0 = time.perf_counter()
    baseline_direct = prod_enhance.final_enhance(img, target=target)
    t_base = time.perf_counter() - t0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    integrated, stages = enhance_g91.final_enhance_g91(
        img, target=target, return_stages=True, device=device)
    t_total = time.perf_counter() - t0
    peak_vram_mb = (float(torch.cuda.max_memory_allocated() / 1048576.0)
                    if device.type == "cuda" else 0.0)

    # baseline computed INSIDE the wrapper (stages["baseline"]) must match a
    # direct, independent enhance.final_enhance() call -- proves the
    # integration doesn't silently diverge from calling production directly.
    assert np.array_equal(baseline_direct, stages["baseline"]), \
        f"{fname}: wrapper baseline diverges from direct final_enhance() call"

    cv2.imwrite(str(FULL_DIR / f"{Path(fname).stem}_baseline.png"), baseline_direct)
    cv2.imwrite(str(FULL_DIR / f"{Path(fname).stem}_g91.png"), integrated)

    result = {
        "elapsed_s": {"baseline_direct": round(t_base, 2),
                      "integrated_total": round(t_total, 2)},
        "peak_vram_mb_integrated_call": peak_vram_mb,
        "g9_stats": stages.get("g9_stats"),
        "whole_image": {
            "baseline_vs_faithful": evaluate_similarity(faithful, baseline_direct),
            "integrated_vs_faithful": evaluate_similarity(faithful, integrated),
            "integrated_vs_baseline": evaluate_similarity(baseline_direct, integrated),
            "integrated_vs_baseline_pixeldiff": pixel_diff(baseline_direct, integrated),
            "baseline_no_ref": metrics_for(baseline_direct),
            "integrated_no_ref": metrics_for(integrated),
        },
        "crops": {},
    }

    rois = dict(g9run.ROIS.get(fname, {}))
    bb = g9run.load_region_billboard_box(fname)
    if bb and "jj_gold" not in rois:
        rois["billboard"] = bb

    for name, box in rois.items():
        vb = g9run.clamp_box(box, W, H)
        bx, by, bw, bh = vb
        box4 = (int(round(bx * s)), int(round(by * s)), int(round(bw * s)), int(round(bh * s)))
        base_c = g9run.crop(baseline_direct, box4)
        int_c = g9run.crop(integrated, box4)
        if base_c.size == 0 or int_c.size == 0:
            continue
        cv2.imwrite(str(CROPS_DIR / f"{Path(fname).stem}_{name}_baseline.png"), base_c)
        cv2.imwrite(str(CROPS_DIR / f"{Path(fname).stem}_{name}_g91.png"), int_c)
        result["crops"][name] = {
            "box4": list(box4),
            "baseline_no_ref": metrics_for(base_c),
            "integrated_no_ref": metrics_for(int_c),
            "integrated_vs_baseline_pixeldiff": pixel_diff(base_c, int_c),
            "integrated_vs_baseline_similarity": (evaluate_similarity(base_c, int_c)
                                                   if base_c.shape == int_c.shape else None),
        }

    return result


def determinism_check(img, device):
    out1 = enhance_g91.final_enhance_g91(img, device=device)
    out2 = enhance_g91.final_enhance_g91(img, device=device)
    d = cv2.absdiff(out1, out2)
    return {"identical": bool(d.max() == 0), "max_abs_diff": int(d.max()),
            "nonzero_px_channels": int(np.count_nonzero(d))}


def main():
    FULL_DIR.mkdir(parents=True, exist_ok=True)
    CROPS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"G9.1-B integration validation, device={device}")

    all_images = sorted(p.name for p in g9run.ORIGINALS.glob("*.jpeg"))
    all_results = {}
    for fname in all_images:
        print(f"== {fname} ==")
        res = process_image(fname, device)
        wi = res["whole_image"]
        print(f"  baseline {res['elapsed_s']['baseline_direct']}s | integrated total "
              f"{res['elapsed_s']['integrated_total']}s | peak_vram="
              f"{res['peak_vram_mb_integrated_call']:.1f}MB")
        print(f"  integrated_vs_baseline: SSIM={wi['integrated_vs_baseline']['ssim']:.4f} "
              f"PSNR={wi['integrated_vs_baseline']['psnr']:.2f} "
              f"mean_abs_diff={wi['integrated_vs_baseline_pixeldiff']['mean_abs_diff']:.3f} "
              f"sharp: base={wi['baseline_no_ref']['sharpness']:.1f} "
              f"g91={wi['integrated_no_ref']['sharpness']:.1f}")
        all_results[fname] = res

    print("== determinism check (1.jpeg, 2 independent final_enhance_g91 calls) ==")
    os.environ["BILLBOARD_IMAGE"] = "1.jpeg"
    os.environ.pop("REGIONS_CONFIG", None)
    prod_enhance._REGION_CONFIG = None
    det_img = cv2.imread(str(g9run.ORIGINALS / "1.jpeg"), cv2.IMREAD_COLOR)
    det = determinism_check(det_img, device)
    print(f"  {det}")
    all_results["determinism_1jpeg"] = det

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(all_results, indent=2))
    print(f"\nwrote {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
