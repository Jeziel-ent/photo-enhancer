"""F3 edge-gate threshold sweep.

Runs D1-weak SR once per image (expensive), then evaluates the three t0
variants (0.30 baseline / 0.25 / 0.20) through F3 + G7-MS + A+ and measures
the full metric set against the faithful baseline.

Usage:
  .venv\\Scripts\\python.exe src\\f3_gate_exp\\sweep.py
  .venv\\Scripts\\python.exe src\\f3_gate_exp\\sweep.py --images 1,5,mumbai-01
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import enhance  # noqa: E402
import quality  # noqa: E402
from f3_gate_exp.recipe import f3_gate_t0, gate_coverage  # noqa: E402

T0_VARIANTS = {"t0_030_baseline": 0.30, "t0_025": 0.25, "t0_020": 0.20}


def full_output(faithful, d1, t0):
    """F3 with given t0 -> G7-MS -> A+. Same as harness path (no boxes)."""
    f3 = f3_gate_t0(faithful, d1, t0)
    ms = enhance._final_multiscale_detail(f3)
    return enhance._final_whole_frame_detail(ms)


def sweep_one(path):
    img = enhance.load_image(str(path))
    faithful = enhance.simple_upscale(img)
    d1, peak_vram, _ = enhance._final_d1_weak(img)

    rows = []
    cov = gate_coverage(enhance._final_lab_l(faithful))
    for name, t0 in T0_VARIANTS.items():
        t_start = time.perf_counter()
        out = full_output(faithful, d1, t0)
        rt = time.perf_counter() - t_start

        sim = quality.evaluate_similarity(faithful, out)
        nr = quality.evaluate_no_reference(out)
        nr_f = quality.evaluate_no_reference(faithful)
        diff = np.abs(faithful.astype(np.float32) - out.astype(np.float32))
        rows.append({
            "variant": name, "t0": t0,
            "psnr": round(sim["psnr"], 3),
            "ssim": round(sim["ssim"], 4),
            "hist_sim": round(sim["hist_sim"], 4),
            "edge_align": round(sim["edge_align"], 4),
            "sharpness": round(nr["sharpness"], 1),
            "noise": round(nr["noise"], 4),
            "contrast": round(nr["contrast"], 4),
            "edge_density": round(nr["edge_density"], 4),
            "sharp_gain": round(nr["sharpness"] / max(nr_f["sharpness"], 1e-6), 3),
            "mean_pd": round(float(diff.mean()), 4),
            "pct_changed": round(float((diff.max(axis=2) > 2.0).mean() * 100), 3),
            "runtime_s": round(rt, 2),
            "peak_vram_mb": peak_vram,
            "cov_gt0": cov[t0]["pct_gate_gt_0.05"],
            "cov_full": cov[t0]["pct_gate_full_1.0"],
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from regression_harness import discover_benchmark_images
    images = discover_benchmark_images()
    if args.images:
        allowed = set(args.images.split(","))
        images = [p for p in images if p.stem in allowed]
    if args.limit > 0:
        images = images[:args.limit]

    print("device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print("images:", [p.name for p in images])

    for path in images:
        print(f"\n== {path.name} ==")
        for r in sweep_one(path):
            print(f"{r['variant']:<16} t0={r['t0']:.2f} "
                  f"PSNR={r['psnr']:>7} SSIM={r['ssim']:.4f} "
                  f"hist={r['hist_sim']:.4f} ealign={r['edge_align']:.4f} "
                  f"sharp={r['sharpness']:>7} noise={r['noise']:.3f} "
                  f"conf={r['contrast']:.3f} edens={r['edge_density']:.2f} "
                  f"gain={r['sharp_gain']:>6} pd={r['mean_pd']:.3f} "
                  f"pct%={r['pct_changed']:.2f} rt={r['runtime_s']:.2f}s "
                  f"cov>{0.05}={r['cov_gt0']:.1f}% full={r['cov_full']:.1f}%")
            if r["variant"] == "t0_030_baseline":
                print(f"    (baseline t0=cov above; runtime vs reported rt_s)")


if __name__ == "__main__":
    main()