"""Diagnostic: where does the self-envelope actually deviate from the flat
±14 cap? Measures the distribution of local [min,max] ranges across
F3-active (gate > 0) pixels on real regression images, to understand
whether a locally-derived envelope can be strictly tighter, strictly wider,
or equal to the flat cap at the pixels F3 actually touches."""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import enhance  # noqa: E402
from f3_local_exp.recipe import _edge_gate, _structure_term, _lab_l  # noqa: E402


def analyze(path, ksize=5, w_s=6.0, clip=18.0, t0=0.30):
    img = enhance.load_image(str(path))
    faithful = enhance.simple_upscale(img)
    d1, _, _ = enhance._final_d1_weak(img)

    a_l = _lab_l(faithful)
    s_l = _lab_l(d1)
    gate = _edge_gate(a_l, t0)
    sc = _structure_term(s_l, a_l)
    dL = w_s * np.clip(sc, -clip, clip) * gate

    k = np.ones((ksize, ksize), np.uint8)
    lo = cv2.erode(a_l, k)
    hi = cv2.dilate(a_l, k)
    lo_off = lo - a_l   # per-pixel lower bound (negative)
    hi_off = hi - a_l   # per-pixel upper bound (positive)

    active = gate > 0.05
    dL_active = np.abs(dL[active])

    local_range = hi_off - lo_off  # = hi - lo

    data = {
        "image": path.name,
        "ksize": ksize,
        "n_active_px": int(active.sum()),
        "pct_active": round(float(active.mean() * 100), 2),
        "local_range": {
            "pct50": round(float(np.percentile(local_range[active], 50)), 2),
            "pct90": round(float(np.percentile(local_range[active], 90)), 2),
            "pct99": round(float(np.percentile(local_range[active], 99)), 2),
            "max": round(float(local_range[active].max()), 2),
        },
        "dL": {
            "pct50_abs": round(float(np.percentile(dL_active, 50)), 2),
            "pct90_abs": round(float(np.percentile(dL_active, 90)), 2),
            "pct99_abs": round(float(np.percentile(dL_active, 99)), 2),
            "max_abs": round(float(dL_active.max()), 2),
            "mean_abs": round(float(dL_active.mean()), 2),
        },
        # Where self-envelope is STRICTLY tighter than flat ±14:
        "pct_gate_px_tighter_than_14": round(
            float(((np.abs(lo_off) < 14) | (np.abs(hi_off) < 14))[active].mean() * 100), 2),
        # Where local range allows MORE than 14 (self-env wider than flat):
        "pct_gate_px_local_range_gt_14": round(
            float((local_range[active] > 14).mean() * 100), 2),
        # Where local range allows MORE than 28 (i.e. would exceed baseline by 2x):
        "pct_gate_px_local_range_gt_28": round(
            float((local_range[active] > 28).mean() * 100), 2),
        # mean signed dL before any envelope:
        "dL_mean": round(float(dL[active].mean()), 3),
    }
    return data


def main():
    from regression_harness import discover_benchmark_images
    images = discover_benchmark_images()
    for ksize in (3, 5, 7):
        print(f"\n=== ksize={ksize} ===")
        for path in images:
            d = analyze(path, ksize=ksize)
            print(f"{d['image']:<16} active={d['pct_active']:>5.2f}%  "
                  f"local_range med={d['local_range']['pct50']:>5} p90={d['local_range']['pct90']:>5} "
                  f"p99={d['local_range']['pct99']:>5} max={d['local_range']['max']:>5}  |  "
                  f"dL med={d['dL']['pct50_abs']:>5} p90={d['dL']['pct90_abs']:>6} "
                  f"max={d['dL']['max_abs']:>6} mean={d['dL']['mean_abs']:>5}  |  "
                  f"tighter14={d['pct_gate_px_tighter_than_14']:>5}%  "
                  f"range>14={d['pct_gate_px_local_range_gt_14']:>5}%  "
                  f"range>28={d['pct_gate_px_local_range_gt_28']:>5}%")


if __name__ == "__main__":
    main()