"""G9.1 controlled tuning harness (isolated R&D, NOT production).

Reuses the EXACT corrected G9 harness (run.py's process_image/ROIS/metrics
logic) and the EXACT G9 recipe (recipe_g9.g9_enhance / controlled_generative_
detail) UNMODIFIED. The only thing this script changes is the three
module-level tuning constants in recipe_g9 (G9_STRENGTH, G9_CLIP,
G9_GATE_T0), reassigned in-process for one candidate at a time -- it never
edits recipe_g9.py, run.py, or enhance.py on disk.

Writes each candidate's results to its own subtree:
  phase1_enhancement/reports/g9_1_exp/<candidate>/{full,crops}/*.png
  phase1_enhancement/reports/g9_1_exp/<candidate>/summary.json

Usage:
  .venv/Scripts/python phase1_enhancement/src/g9_exp/run_g91.py --candidate A
  .venv/Scripts/python phase1_enhancement/src/g9_exp/run_g91.py --candidate B
  .venv/Scripts/python phase1_enhancement/src/g9_exp/run_g91.py --candidate C
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
from g9_exp import recipe_g9  # noqa: E402
from g9_exp import run as g9run  # noqa: E402  (unmodified corrected harness)

CANDIDATES = {
    "A": {"strength": 0.24, "clip": 5.0, "gate": 0.40},
    "B": {"strength": 0.30, "clip": 5.0, "gate": 0.40},
    "C": {"strength": 0.30, "clip": 8.0, "gate": 0.35},
}

OUT_ROOT = ROOT / "reports" / "g9_1_exp"


def apply_params(params):
    recipe_g9.G9_STRENGTH = params["strength"]
    recipe_g9.G9_CLIP = params["clip"]
    recipe_g9.G9_GATE_T0 = params["gate"]


def process_image(fname, device, out_full, out_crops):
    """Identical logic to run.process_image, just parameterized on output dirs."""
    os.environ["BILLBOARD_IMAGE"] = fname
    os.environ.pop("REGIONS_CONFIG", None)
    prod_enhance._REGION_CONFIG = None
    img = cv2.imread(str(g9run.ORIGINALS / fname), cv2.IMREAD_COLOR)
    H, W = img.shape[:2]
    target = (prod_enhance.OUT_W, prod_enhance.OUT_H)
    s = target[0] / W

    faithful = prod_enhance.simple_upscale(img, target)

    t0 = time.perf_counter()
    baseline, base_stages = prod_enhance.final_enhance(img, target=target, return_stages=True)
    t_base = time.perf_counter() - t0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    g9_out, g9_stages = recipe_g9.g9_enhance(
        img, device=device, target=target, return_stages=True,
        precomputed_baseline=(baseline, base_stages))
    t_g9 = time.perf_counter() - t0
    peak_vram_mb = (float(torch.cuda.max_memory_allocated() / 1048576.0)
                    if device.type == "cuda" else 0.0)

    cv2.imwrite(str(out_full / f"{Path(fname).stem}_baseline.png"), baseline)
    cv2.imwrite(str(out_full / f"{Path(fname).stem}_g9.png"), g9_out)

    from quality import evaluate_similarity, compute_no_reference

    def metrics_for(im):
        if im is None or im.size == 0:
            return None
        nr = compute_no_reference(im)
        return {"sharpness": round(nr["sharpness"], 2),
                "noise": round(nr["noise"], 3),
                "contrast": round(nr["contrast"], 4),
                "edge_density": round(nr["edge_density"], 4),
                "w": int(im.shape[1]), "h": int(im.shape[0])}

    def pixel_diff(a, b):
        d = cv2.absdiff(a, b)
        return {"mean_abs_diff": float(d.mean()),
                "max_abs_diff": int(d.max()),
                "pct_px_changed_gt2": float((d.max(axis=2) > 2).mean() * 100.0)}

    result = {
        "elapsed_s": {"baseline": round(t_base, 2), "g9_extra_s": round(t_g9, 2)},
        "peak_vram_mb_incremental": peak_vram_mb,
        "base_peak_vram_mb": base_stages.get("peak_vram_mb"),
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

    rois = dict(g9run.ROIS.get(fname, {}))
    bb = g9run.load_region_billboard_box(fname)
    if bb and "jj_gold" not in rois:
        rois["billboard"] = bb

    for name, box in rois.items():
        vb = g9run.clamp_box(box, W, H)
        bx, by, bw, bh = vb
        box4 = (int(round(bx * s)), int(round(by * s)),
                int(round(bw * s)), int(round(bh * s)))
        base_c = g9run.crop(baseline, box4)
        g9_c = g9run.crop(g9_out, box4)
        if base_c.size == 0 or g9_c.size == 0:
            continue
        cv2.imwrite(str(out_crops / f"{Path(fname).stem}_{name}_baseline.png"), base_c)
        cv2.imwrite(str(out_crops / f"{Path(fname).stem}_{name}_g9.png"), g9_c)
        result["crops"][name] = {
            "box4": list(box4),
            "baseline_no_ref": metrics_for(base_c),
            "g9_no_ref": metrics_for(g9_c),
            "g9_vs_baseline_pixeldiff": pixel_diff(base_c, g9_c),
            "g9_vs_baseline_similarity": (evaluate_similarity(base_c, g9_c)
                                          if base_c.shape == g9_c.shape else None),
        }

    return result


def determinism_check(img, device):
    out1 = recipe_g9.g9_enhance(img, device=device)
    out2 = recipe_g9.g9_enhance(img, device=device)
    d = cv2.absdiff(out1, out2)
    return {"identical": bool(d.max() == 0),
            "max_abs_diff": int(d.max()),
            "nonzero_px_channels": int(np.count_nonzero(d))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, choices=list(CANDIDATES))
    ap.add_argument("--images", nargs="*", default=None)
    args = ap.parse_args()

    params = CANDIDATES[args.candidate]
    apply_params(params)
    # GAN weight cache is independent of these params, so reuse is safe and
    # does not affect correctness of the post-processing stage under test.

    out_dir = OUT_ROOT / args.candidate
    out_full = out_dir / "full"
    out_crops = out_dir / "crops"
    out_full.mkdir(parents=True, exist_ok=True)
    out_crops.mkdir(parents=True, exist_ok=True)

    all_images = sorted(p.name for p in g9run.ORIGINALS.glob("*.jpeg"))
    images = args.images if args.images else all_images

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"candidate={args.candidate} params={params} device={device}")

    all_results = {"candidate": args.candidate, "params": params}
    for fname in images:
        print(f"== {fname} ==")
        res = process_image(fname, device, out_full, out_crops)
        wi = res["whole_image"]
        print(f"  baseline {res['elapsed_s']['baseline']}s | g9 extra "
              f"+{res['elapsed_s']['g9_extra_s']}s | peak_vram_incr="
              f"{res['peak_vram_mb_incremental']:.1f}MB")
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
    det_img = cv2.imread(str(g9run.ORIGINALS / "1.jpeg"), cv2.IMREAD_COLOR)
    det = determinism_check(det_img, device)
    print(f"  {det}")
    all_results["determinism_1jpeg"] = det

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(all_results, indent=2))
    print(f"\nwrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
