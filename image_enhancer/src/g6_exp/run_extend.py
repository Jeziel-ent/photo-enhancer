"""G6 extension: confirm A vs A+ across the remaining reference images
(2/5/6.jpeg) not covered by the main run.py sweep, at lower cost since only
the two candidates that showed a real difference are needed here.
"""
import sys
import time
from pathlib import Path

import cv2
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import enhance  # noqa: E402
from quality import evaluate_similarity, compute_no_reference  # noqa: E402
from g6_exp import recipe_g6  # noqa: E402

SAVE_DIR = ROOT / "enhanced" / "g6_exp"
REPORT_DIR = ROOT / "reports" / "g6_exp"

CANDIDATES = {"A": recipe_g6.candidate_a, "A+": recipe_g6.candidate_a_plus}
STEMS = ["2", "5", "6"]

rows = []
for stem in STEMS:
    print(f"== {stem}.jpeg ==", flush=True)
    orig = cv2.imread(str(ROOT / "originals" / f"{stem}.jpeg"), cv2.IMREAD_COLOR)
    faithful = enhance.simple_upscale(orig, (3840, 2160))
    for name, fn in CANDIDATES.items():
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        out, stages = fn(orig)
        wall = time.perf_counter() - t0
        peak = float(torch.cuda.max_memory_allocated() / 1048576.0) if torch.cuda.is_available() else 0.0
        cv2.imwrite(str(SAVE_DIR / f"{stem}_{name}.png"), out)
        m = evaluate_similarity(faithful, out)
        nr = compute_no_reference(out)
        rows.append({"image": stem, "candidate": name, "wall_s": round(wall, 1),
                    "peak_vram_mb": round(peak, 0), **m,
                    "sharpness": round(nr["sharpness"], 2),
                    "noise": round(nr["noise"], 3)})
        print(f"   {name}: {wall:.1f}s sharpness={nr['sharpness']:.1f} "
             f"ssim={m['ssim']:.3f} hist_sim={m['hist_sim']:.3f}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(REPORT_DIR / "extend_A_vs_Aplus.csv", index=False)
print("\n", df.drop(columns=["image"]).groupby("candidate").mean(numeric_only=True).round(3).to_string())
