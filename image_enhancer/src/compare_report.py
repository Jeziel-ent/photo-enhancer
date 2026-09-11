"""Generate comparison report + automated quality/fidelity checks.

For each original image and each enhancement method, we build:
  - a side-by-side visual montage (original upscale | enhanced)
  - a fidelity summary vs the faithful original upscale (content-preservation)
  - an enhancement-gain summary (no-reference sharpness/quality delta)

Automated checks emit warnings if a method drifts from the source content
(low histogram similarity / low SSIM) = violation of the preserve-only rule.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))

from enhance import METHODS, simple_upscale
from quality import evaluate_similarity, compute_no_reference

ROOT = Path(__file__).parent.parent
ORIGINALS = ROOT / "originals"
ENHANCED = ROOT / "enhanced"
REPORTS = ROOT / "reports"
COMPARISON = REPORTS / "comparisons"
OUT_W, OUT_H = 3840, 2160

# Constraints for the "preserve only" rule.
# Structural/geometric preservation is the primary guard: high SSIM means no
# content was added/removed/replaced. Tonal shift (histogram) is EXPECTED from
# legitimate contrast/color enhancement, so we only flag extreme drift that
# would indicate content alteration.
SSIM_FLOOR = 0.80      # structural similarity vs faithful upscale
PSNR_FLOOR = 20.0      # dB vs faithful upscale (very low = content changed)
HIST_SEVERE = 0.65     # extreme histogram drift = likely content changed


def supported_image(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def find_enhanced(method, image):
    for e in ENHANCED.glob(f"{method}_{image.stem}.*"):
        return e
    return None


def make_montage(orig_up, enh, out_path, method, image, scale=0.5):
    """Stack original-upscale (top) and enhanced (bottom) at scale, with labels."""
    h1, w1 = orig_up.shape[:2]
    th, tw = int(h1 * scale), int(w1 * scale)
    a = cv2.resize(orig_up, (tw, th), interpolation=cv2.INTER_AREA)
    b = cv2.resize(enh, (tw, th), interpolation=cv2.INTER_AREA)

    label_h = 30
    a_lab = cv2.copyMakeBorder(a, label_h, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    cv2.putText(a_lab, "Original (Lanczos to 3840x2160)", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 60), 2, cv2.LINE_AA)
    b_lab = cv2.copyMakeBorder(b, label_h, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    cv2.putText(b_lab, f"Enhanced [{method}] - {image}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 255, 120), 2, cv2.LINE_AA)

    sep = np.full((th + label_h, tw, 3), 255, np.uint8)
    montage = np.vstack([a_lab, sep, b_lab])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), montage)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="*", default=list(METHODS))
    args = ap.parse_args()

    ORIGINALS.mkdir(exist_ok=True)
    ENHANCED.mkdir(exist_ok=True)
    COMPARISON.mkdir(exist_ok=True)

    images = [p for p in sorted(ORIGINALS.iterdir()) if supported_image(p)]
    if not images:
        print("no originals"); return

    rows = []
    warnings = []

    for img_p in images:
        img = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
        faithful = simple_upscale(img, (OUT_W, OUT_H))  # preserve-only baseline

        for method in args.methods:
            if method not in METHODS:
                continue
            e_p = find_enhanced(method, img_p)
            if not e_p:
                warnings.append(f"missing enhanced output for {method}/{img_p.name}")
                continue
            enh = cv2.imread(str(e_p), cv2.IMREAD_COLOR)

            sim = evaluate_similarity(faithful, enh)
            nr = compute_no_reference(enh)
            nr_base = compute_no_reference(faithful)
            row = {
                "image": img_p.name,
                "method": method,
                "psnr": sim["psnr"],
                "ssim": sim["ssim"],
                "nrmse": sim["nrmse"],
                "hist_sim": sim["hist_sim"],
                "edge_align": sim["edge_align"],
                "sharp_enh": nr["sharpness"],
                "sharp_base": nr_base["sharpness"],
                "sharp_gain_x": round(nr["sharpness"] / (nr_base["sharpness"] + 1e-9), 3),
                "edge_enh": nr["edge_density"],
                "edge_base": nr_base["edge_density"],
                "edge_gain_x": round(nr["edge_density"] / (nr_base["edge_density"] + 1e-9), 3),
                "noise": nr["noise"],
            }
            rows.append(row)

            # montage
            mont = COMPARISON / f"cmp_{method}_{img_p.stem}.png"
            make_montage(faithful, enh, mont, method, img_p.name)

            # automated checks (preserve-only rule)
            checks = []
            if sim["ssim"] < SSIM_FLOOR:
                checks.append(f"SSIM {sim['ssim']:.3f}<{SSIM_FLOOR} (content/structure altered)")
            if sim["hist_sim"] < HIST_SEVERE:
                checks.append(f"hist {sim['hist_sim']:.3f}<{HIST_SEVERE} (content likely changed)")
            if sim["psnr"] < PSNR_FLOOR:
                checks.append(f"PSNR {sim['psnr']:.2f}<{PSNR_FLOOR}")
            if sim["edge_align"] < 0.9:
                checks.append(f"edge_align {sim['edge_align']:.3f}<0.9 (structure displaced)")
            if checks:
                warnings.append(f"{method}/{img_p.name} FIDELITY DRIFT: {', '.join(checks)}")

    if not rows:
        print("no data"); return

    df = pd.DataFrame(rows)
    rep = REPORTS / "fidelity_comparison.csv"
    df.to_csv(rep, index=False)
    print(f"comparison report: {rep}\n")

    agg = df.groupby("method")[["psnr", "ssim", "hist_sim", "edge_align",
                                "sharp_gain_x", "edge_gain_x", "noise"]].mean().round(3)
    print("Mean fidelity vs faithful original upscale (psnr/ssim/hist/edge_align ~1 = preserved):")
    print(agg.to_string())
    print("\nMean no-reference enhancement gain (sharp_gain_x > 1 = sharper than original):")
    print(df.groupby("method")[["sharp_gain_x", "edge_gain_x"]].mean().round(3).to_string())

    print("\nAutomated checks:")
    if warnings:
        for w in warnings:
            print("  WARN:", w)
    else:
        print("  All methods passed the preserve-only fidelity checks.")

    # montages
    monts = list(COMPARISON.glob("cmp_*.png"))
    print(f"\nVisual comparison montages: {len(monts)} generated in {COMPARISON}")


if __name__ == "__main__":
    main()
