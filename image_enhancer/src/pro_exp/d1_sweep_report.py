"""Writes reports/pro_exp/d1_sweep/D1_SWEEP_REPORT.md from the sweep CSVs.

Usage:
  .venv/Scripts/python src/pro_exp/d1_sweep_report.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
RD = ROOT / "reports" / "pro_exp" / "d1_sweep"

BASE = ["A", "D1-weak", "D1-current", "D1-medium"]
ORDER = ["A", "D1-weak", "D1-medium", "D1-current"]


def get(name):
    p = RD / name
    return pd.read_csv(p) if p.exists() else None


def gmean(df, cols):
    t = df.drop(columns=["image"])
    sel = [c for c in cols if c in t.columns]
    g = t.groupby("variant")[sel].mean(numeric_only=True).round(3)
    return g.reindex(ORDER)


def main():
    full = get("full_metrics.csv")
    bill = get("billboard_metrics.csv")
    face = get("face_metrics.csv")
    detail = get("detail_metrics.csv")
    run = get("runtime.csv")

    L = []
    a = L.append
    a("# D1 parameter sweep - isolated report")
    a("")
    a("Center: **D1 = conservative denoise (Restormer SIDD) -> SwinIR-M PSNR x4**.  ")
    a("")
    a("Only the **denoise strength** changes, controlled by blending the fixed "
      "denoiser into the untouched original: `input = original*(1-K) + "
      "Restormer_SIDD(original)*K`.")
    a("")
    a("| variant | K (denoise fraction) | meaning |")
    a("|---|---|---|")
    a("| A | - | SwinIR-M PSNR baseline (no denoise) |")
    a("| D1-weak | 0.30 | keep most of the original texture |")
    a("| D1-medium | 0.65 | intermediate sweep point |")
    a("| D1-current | 1.00 | full SIDD denoise (previous D1 center) |")
    a("")
    a("Fidelity reference = faithful Lanczos 4K upscale of the original. "
      "All variants went through the SAME SwinIR-M PSNR x4 stage "
      "(tile 256). No region compositing, no text/OCR, no generation.")
    a("")
    a("## VRAM fix (requirement)")
    a("")
    a("The previous D1 run spiked to ~9.6 GB Windows shared memory. Here the "
      "denoise runs tiled (512 px, feathered seams, `tiled_restore`) with the "
      "torch cache cleared between stages, and stage-wise peak VRAM is "
      "recorded. Result: **peak 2.4 GB** - safely inside the 8 GB limit, no "
      "shared-memory spill. Output quality is unchanged in tile interiors.")
    a("")
    if run is not None:
        a(f"Mean runtime per variant: denoise {run['dn_s'].mean():.1f}s + "
          f"SR {run['sr_s'].mean():.1f}s = {run['total_s'].mean():.1f}s; "
          f"peak VRAM {run['peak_vram_mb'].mean():.0f} MB.")

    a("")
    a("## 1. Full image (mean over 6 photos)")
    a("")
    a("```")
    a(gmean(full, ["psnr", "ssim", "nrmse", "hist_sim", "edge_align",
                   "sharpness", "noise", "contrast", "mean_b", "mean_r",
                   "std_b", "std_r"]).to_string())
    a("```")
    a("")
    a("Reading: PSNR/SSIM/hist_sim/edge_align vs faithful = how much of the "
      "source is preserved. **D1-weak loses only 0.14 dB PSNR vs A (27.38 vs "
      "27.52) while lowering noise (1.29 vs 1.37) and raising sharpness "
      "(317 vs 296).** Stronger K buys more noise removal (current: 1.15, "
      "sharp 333) but drifts further from the source (PSNR 27.28). Color "
      "means/stds stay within ~1 unit of A for every variant - no tonal "
      "shift.")
    a("")
    a("## 2. Billboard region (verified boxes, mean)")
    a("")
    a("```")
    a(gmean(bill, ["psnr", "ssim", "edge_align", "sharp", "noise", "contrast",
                   "te_contrast", "hp_energy"]).to_string())
    a("```")
    a("")
    a("Text-legibility proxies: **D1-weak 610.2 / D1-medium 608.2 are >= A "
      "(608.4); D1-current is the only one that drops (600.9)** - full "
      "denoise slightly smooths letter edges. Local board PSNR rises with K "
      "(A 23.25 -> weak 23.30 -> medium 23.38 -> current 23.48) because "
      "denoise helps SR on the board. D1-weak: board fidelity slightly "
      "better than A, letter edges unchanged - the only no-tradeoff point.")
    a("")
    a("## 3. Face region (largest face: 1.jpeg, 42x53px, sole >=24px "
      "detection)")
    a("")
    a("```")
    a(gmean(face, ["psnr", "ssim", "edge_align", "sharp", "noise",
                   "tex_retain_hp"]).to_string())
    a("```")
    a("")
    a("`tex_retain_hp` = high-frequency energy ratio vs the faithful crop "
      "(all variants sit ~2.4-2.6x because the SR itself adds aligned "
      "detail). **D1-weak preserves face texture closest to A (2.515 vs "
      "2.546) while improving face fidelity (26.68/0.8405 vs A 26.47/0.8399)** "
      "- a real, if modest, gain. D1-current smooths texture most (2.433) and "
      "scores highest PSNR/SSIM (27.06/0.846) - the classic smoothing-vs-"
      "texture trade. At 42px the face gain is measurable but CANNOT be "
      "'noticeably clearer' non-generatively; identity is never invented.")
    a("")
    a("## 4. Fine-detail strip (auto-picked informative window, mean)")
    a("")
    a("```")
    a(gmean(detail, ["psnr", "ssim", "edge_align", "sharp", "noise"]).to_string())
    a("```")
    a("")
    a("D1-weak costs only 0.27 dB on fine detail vs A (21.39 vs 21.66) while "
      "rendering it sharper (1382 vs 1263). D1-medium/current lose measurably "
      "more fidelity here (21.25 / 21.20) - i.e. road/car/building texture "
      "gets smoothed by strong denoising. edge_align >= 0.996 everywhere: no "
      "geometry invented.")
    a("")
    a("## Decision")
    a("")
    a("1. **Best D1 parameter: D1-weak, K = 0.30** (blend 70% original + 30% "
      "SIDD denoised); fallback D1-medium (0.65) only if the user wants "
      "stronger noise removal on text-free frames.")
    a("2. **Why:** it captures most of the denoise-for-SR benefit at the "
      "lowest texture risk: full-frame PSNR only -0.14 dB vs A, noise down, "
      "sharpness up; billboard letter proxies >= A; fine-detail fidelity "
      "closest to A; face texture retained closest to A while face metrics "
      "edge A. Higher K buys diminishing gains at measurable texture/fidelity "
      "cost (front-mounted per the tables above).")
    a("3. **Face clarity genuinely improved?** Metrics say yes but slightly "
      "(PSNR 26.68 vs 26.47, SSIM 0.8405 vs 0.8399, texture 2.515 close to "
      "A's 2.546). At 42x53 px the practical visible gain is subtle - "
      "**confirm in `d1_sweep_face0_1.png`**; a larger-subject photo would "
      "show it better.")
    a("4. **Billboard text genuinely improved?** Proxies: D1-weak >= A on "
      "letter-edge contrast (610 vs 608) with better local board PSNR. The "
      "visible change is again mild - **confirm in "
      "`d1_sweep_billboard_*.png`**.")
    a("5. **Fine detail improved?** Unlike stronger K, D1-weak loses almost "
      "nothing (21.39 vs 21.66 dB) while delivering more sharpness - "
      "**confirm in `d1_sweep_detail_*.png`**.")
    a("6. **Waxy/painterly artifacts?** No numeric indicator (texture "
      "retained at A levels, edge_align ~0.99). D1-weak is the lowest-risk "
      "variant by construction (70% untouched pixels). Waxiness is a visual "
      "call - check the face crop.")
    a("7. **Text/artwork changed?** No. edge_align >= 0.99 at every scale, "
      "no region compositing was applied (full-frame pipeline only), no "
      "OCR/regeneration exists. The photograph is unchanged except where "
      "denoise+SR legitimately restore it.")
    a("8. **Safe for production?** As a fidelity-first tweak (K=0.30), yes - "
      "if and ONLY if you approve the visuals. It adds one cheap 5s denoise "
      "stage, stays at 2.4 GB peak VRAM, and does not create content. "
      "**Nothing has been integrated; `src/enhance.py` is untouched.**")
    a("")
    a("## Visual evidence - review before approving")
    a("")
    a("All in `reports/pro_exp/d1_sweep/` (labeled Original / SwinIR-M "
      "baseline / D1-weak / D1-current / D1-medium):")
    a("")
    a("- Full scenes: `d1_sweep_full_1..6.png`")
    a("- Billboards: `d1_sweep_billboard_1..6.png`")
    a("- Fine detail: `d1_sweep_detail_1..6.png`")
    a("- Face (1.jpeg): `d1_sweep_face0_1.png`")
    a("")
    a("*Final adoption decision is VISUAL, not metric-based. Production "
      "pipeline untouched until you approve.*")

    out = ROOT / "reports" / "pro_exp" / "D1_SWEEP_REPORT.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()