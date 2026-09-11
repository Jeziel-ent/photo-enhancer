"""Final-engine six-image benchmark (production `final` vs F3/G1 research).

Runs the production pipeline (enhance.final_enhance) on all six reference
images, writes 3840x2160 outputs plus full-frame and billboard metrics, and
prints a head-to-head table against the frozen research results
(F3-natural from reports/pro_exp/f3_exp, G1-text from reports/g1_exp).

Usage:
  .\\.venv\\Scripts\\python.exe src/benchmark_final.py [--limit N]

Outputs (git-ignored): reports/final/FINAL_{1..6}.png, full_metrics.csv,
billboard_metrics.csv, runtime.csv, compare.csv
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import enhance  # noqa: E402
from quality import compute_no_reference, evaluate_similarity  # noqa: E402
from restore_exp.metrics_ext import crop_rect, region_metrics  # noqa: E402
from pro_exp import pipeline2  # noqa: E402
from fidelity_exp.recipe import micro_texture  # noqa: E402

SAVE_DIR = ROOT / "reports" / "final"
F3_FULL = ROOT / "reports" / "pro_exp" / "f3_exp" / "full_metrics.csv"
F3_BILL = ROOT / "reports" / "pro_exp" / "f3_exp" / "billboard_metrics.csv"
G1_FULL = ROOT / "reports" / "g1_exp" / "full_metrics.csv"
G1_BILL = ROOT / "reports" / "g1_exp" / "billboard_metrics.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    images = sorted(p.stem for p in (ROOT / "originals").glob("*.jpeg"))
    if args.limit > 0:
        images = images[:args.limit]

    full_rows, bill_rows, runtime_rows = [], [], []
    for stem in images:
        print(f"== {stem}.jpeg ==", flush=True)
        os.environ["BILLBOARD_IMAGE"] = f"{stem}.jpeg"
        orig = enhance.load_image(str(ROOT / "originals" / f"{stem}.jpeg"))
        faithful = enhance.simple_upscale(orig, (3840, 2160))
        t0 = time.perf_counter()
        out, stages = enhance.final_enhance(orig, return_stages=True)
        wall = time.perf_counter() - t0
        enhance.save_image(out, SAVE_DIR / f"FINAL_{stem}.png")
        runtime_rows.append({"image": f"{stem}.jpeg", "variant": "final",
                             "wall_s": round(wall, 1),
                             "peak_vram_mb": round(stages["peak_vram_mb"], 1),
                             "device": stages["device"],
                             "defocus_boxes": len(stages["defocus_boxes"])})
        print(f"   final: {wall:.1f}s, peak VRAM {stages['peak_vram_mb']:.0f} MB "
              f"({stages['device']}), defocus boxes: {stages['defocus_boxes']}",
              flush=True)

        m = evaluate_similarity(faithful, out)
        nr = compute_no_reference(out)
        full_rows.append({"image": f"{stem}.jpeg", "variant": "final", **m,
                          "sharpness": round(nr["sharpness"], 2),
                          "noise": round(nr["noise"], 3),
                          "contrast": round(nr["contrast"], 4)})
        for bi, box4 in enumerate(stages["boxes4"]):
            refc = crop_rect(faithful, box4)
            vc = crop_rect(out, box4)
            if vc is None:
                continue
            bm = region_metrics(refc, vc)
            if not bm:
                continue
            tp = pipeline2.text_proxies(vc)
            bill_rows.append({"image": f"{stem}.jpeg", "box": bi,
                              "variant": "final", **bm,
                              "micro_tex": round(micro_texture(vc), 3), **tp})

    full = pd.DataFrame(full_rows)
    bill = pd.DataFrame(bill_rows)
    runt = pd.DataFrame(runtime_rows)
    full.to_csv(SAVE_DIR / "full_metrics.csv", index=False)
    bill.to_csv(SAVE_DIR / "billboard_metrics.csv", index=False)
    runt.to_csv(SAVE_DIR / "runtime.csv", index=False)

    f3f = pd.read_csv(F3_FULL)
    g1f = pd.read_csv(G1_FULL)
    f3b = pd.read_csv(F3_BILL)
    g1b = pd.read_csv(G1_BILL)

    def mean(df, variant, cols):
        sub = df[df["variant"] == variant]
        return sub[cols].mean(numeric_only=True).round(3)

    fcols = ["psnr", "ssim", "edge_align", "sharpness", "noise"]
    bcols = ["psnr", "ssim", "edge_align", "te_contrast", "hp_energy", "micro_tex"]
    cmp_full = pd.DataFrame({
        "F3-natural(research)": mean(f3f, "F3-natural", fcols),
        "G1-text(research)": mean(g1f, "G1-text", fcols),
        "final(production)": mean(full, "final", fcols),
    })
    cmp_bill = pd.DataFrame({
        "F3-natural(research)": mean(f3b, "F3-natural", bcols),
        "G1-text(research)": mean(g1b, "G1-text", bcols),
        "final(production)": mean(bill, "final", bcols),
    })
    print("\nfull-frame means (6 images):")
    print(cmp_full.to_string())
    print("\nbillboard means (6 boxes):")
    print(cmp_bill.to_string())
    print("\nruntime / VRAM:")
    print(runt.to_string(index=False))
    cmp_full.to_csv(SAVE_DIR / "compare_full.csv")
    cmp_bill.to_csv(SAVE_DIR / "compare_billboard.csv")
    print(f"\nbenchmark artifacts -> {SAVE_DIR}")


if __name__ == "__main__":
    main()
