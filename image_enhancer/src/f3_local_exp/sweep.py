"""Parameter sweep for the F3 local-envelope variants.

Runs D1-weak SR once per image (expensive), then evaluates every F3
variant + downstream stages (cheap) to find the working point with the
smallest fidelity regression.

Usage:
  .venv\Scripts\python.exe src\f3_local_exp\sweep.py [--limit N] [--images mm,nn]
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
from f3_local_exp.recipe import f3_self_env_tight, f3_self_env_capped, f3_hybrid, f3_local_contrast_env  # noqa: E402

VARIANTS = {
    "baseline": None,  # production flat ±14
    "self_env_k3": f3_self_env_tight,
    "self_env_k5_cap10": f3_self_env_capped,
    "hybrid_50": f3_hybrid,
    "local_contrast_env": f3_local_contrast_env,
}


def run_full_pipeline(faithful, d1, img, device, f3_fn=None):
    """Run downstream stages after F3 (unique F3 fn or production default)."""
    if f3_fn is None:
        f3 = enhance._final_f3_natural(faithful, d1)
    else:
        f3 = f3_fn(faithful, d1)
    f3_ms = enhance._final_multiscale_detail(f3)
    out = enhance._final_whole_frame_detail(f3_ms)
    # No billboard boxes (neutralized) — matches regression harness path
    return out


def sweep_one_image(path, device):
    img = enhance.load_image(str(path))
    faithful = enhance.simple_upscale(img)
    d1, _, _ = enhance._final_d1_weak(img)

    rows = []
    for name, f3_fn in VARIANTS.items():
        t0 = time.perf_counter()
        out = run_full_pipeline(faithful, d1, img, device, f3_fn)
        dt = time.perf_counter() - t0

        sim = quality.evaluate_similarity(faithful, out)
        nr = quality.evaluate_no_reference(out)
        nr_f = quality.evaluate_no_reference(faithful)
        diff = np.abs(faithful.astype(np.float32) - out.astype(np.float32))
        mean_pd = float(diff.mean())
        pct = float((diff.max(axis=2) > 2.0).mean() * 100.0)

        rows.append({
            "variant": name,
            "psnr": round(sim["psnr"], 3),
            "ssim": round(sim["ssim"], 4),
            "hist_sim": round(sim["hist_sim"], 4),
            "edge_align": round(sim["edge_align"], 4),
            "sharpness": round(nr["sharpness"], 1),
            "noise": round(nr["noise"], 3),
            "sharpness_gain": round(nr["sharpness"] / max(nr_f["sharpness"], 1e-6), 2),
            "mean_pixel_diff": round(mean_pd, 4),
            "pct_pixels_changed": round(pct, 3),
            "runtime_s": round(dt, 2),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--images", default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    from regression_harness import discover_benchmark_images
    images = discover_benchmark_images()
    if args.images:
        allowed = set(args.images.split(","))
        images = [p for p in images if p.stem in allowed]
    if args.limit > 0:
        images = images[:args.limit]

    print("images:", [p.name for p in images])

    for path in images:
        print(f"\n== {path.name} ==")
        rows = sweep_one_image(path, device)
        hdr = f"{'variant':<22}{'psnr':>8}{'ssim':>8}{'hist':>7}{'e_align':>9}"
        hdr += f"{'sharp':>9}{'noise':>7}{'gain':>7}{'pdiff':>8}{'pct%':>8}{'rt_s':>7}"
        print(hdr)
        for r in rows:
            print(f"{r['variant']:<22}{r['psnr']:>8}{r['ssim']:>8}{r['hist_sim']:>7}"
                  f"{r['edge_align']:>9}{r['sharpness']:>9}{r['noise']:>7}"
                  f"{r['sharpness_gain']:>7}{r['mean_pixel_diff']:>8}"
                  f"{r['pct_pixels_changed']:>8}{r['runtime_s']:>7}")


if __name__ == "__main__":
    main()