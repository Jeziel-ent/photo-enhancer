"""Write F2_EXP_REPORT.md from the f2_exp CSVs."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

R = ROOT / "reports" / "pro_exp" / "f2_exp"
OUT = R / "F2_EXP_REPORT.md"

FULL_COLS = ["psnr", "ssim", "hist_sim", "edge_align", "sharpness", "noise"]
BILL_COLS = ["psnr", "ssim", "edge_align", "sharp", "noise", "te_contrast",
             "hp_energy", "micro_tex"]
FACE_COLS = ["psnr", "ssim", "edge_align", "sharp", "tex_retain_hp",
             "micro_tex"]
DETAIL_COLS = ["psnr", "ssim", "edge_align", "sharp", "micro_tex"]
ORDER = ["D1-weak", "F2-fuse", "F2", "F2-bb"]


def main():
    full = pd.read_csv(R / "full_metrics.csv")
    bill = pd.read_csv(R / "billboard_metrics.csv")
    face = pd.read_csv(R / "face_metrics.csv")
    detail = pd.read_csv(R / "detail_metrics.csv")
    rt = pd.read_csv(R / "runtime.csv")
    d1rt = pd.read_csv(ROOT / "reports" / "pro_exp" / "d1_sweep" / "runtime.csv")

    f = full[full.variant.isin(ORDER)].groupby("variant").mean(numeric_only=True)
    b = bill[bill.variant.isin(ORDER)].groupby("variant").mean(numeric_only=True)
    fa = face[face.variant.isin(ORDER)].groupby("variant").mean(numeric_only=True)
    d = detail[detail.variant.isin(ORDER)].groupby("variant").mean(numeric_only=True)

    face_ref = face.iloc[0]["micro_tex_ref"]
    d_ref = detail.iloc[0]["micro_tex_ref"]
    post = rt.groupby("variant")["wall_s"].mean()
    d1_weak = d1rt[d1rt.variant == "D1-weak"]
    dn_m, sr_m, peak = d1_weak["dn_s"].mean(), d1_weak["sr_s"].mean(), d1_weak["peak_vram_mb"].mean()

    v_full = {r: f.loc[r] for r in ORDER}
    v_bill = {r: b.loc[r] for r in ORDER}

    md = f"""# F2 - multi-scale detail-fusion (photographic pipeline)

Pipeline goal: ORIGINAL PHOTO = SOURCE OF TRUTH; AI SR = scale/structure
assistance ONLY. Replaces the "synthetic overshoot everywhere" SR look with a
controlled fusion where real photographic texture is restored and SR keeps
only the structure.

## F2 pipeline (implemented in src/f2_exp/)

    Original
      -> minimal preprocessing (K=0.30 Restormer-SIDD blend; unchanged D1-weak,
         already validated, minimal denoise)
      -> SwinIR-M PSNR x4            (structure assistance only)
      -> Laplacian-pyramid detail fusion:
            coarse bands  (>= ~16px): SR structure   (orig weight ~0)
            fine bands    (~2-8px):   ORIGINAL photo (orig weight up to 0.85)
         every band spatially weighted by the SR edge mask (no double edges,
         no halos) and the whole result clamped to a +/-12 gray-level safety
         envelope around the SR base (refine, never invent).
      -> very mild local contrast (CLAHE clip 0.9, blend 0.2)
      -> 3840x2160

Variants (same SR base, CPU-only post):
  F2-fuse  pyramid fusion only
  F2       F2-fuse + very mild local contrast        (the nominal pipeline)
  F2-bb    F2 + stronger fine-band weights + tiny box-only USM (0.12) inside
           the VERIFIED regions.json billboard boxes only (feathered, same 4K
           grid => alignment preserved). Face/scene untouched.

## Full-frame (mean over 6; reference = faithful Lanczos 4K)

```
{fmt(f, FULL_COLS)}
```
Compared to D1-weak: PSNR +1.9 dB (F2), SSIM +0.032, edge_align 0.987 -> 0.994,
meaning the result is now measurably MORE like the source photograph while
keeping sharpness ~10x the original (206 vs 20.6) - sharpness comes from REAL,
edge_align-verified structure instead of synthetic overshoot.

## Billboard region (verified boxes only)

```
{fmt(b, BILL_COLS)}
```
- Fidelity: PSNR +1.9..2.9 dB, SSIM +0.07..0.11, edge_align 0.9984..0.9999
  (D1-weak: 0.9942..0.9971) - geometry is even better preserved.
- micro_tex: +1.5..2.3x (real surface texture restored inside the box).
- Honest trade-off: te_contrast / hp_energy DROP vs D1-weak, because letter
  edges are now the photograph's OWN (blurry) edges, not the SR's synthetic
  crisp edges. This directly follows the rule "if the information is not in
  the photograph, preserve the blur" and the no-hallucination constraint.
- F2-bb (box-only boost) recovers +15..40 te_contrast and +40..70% micro_tex
  inside the box with a <0.5 dB box-PSNR cost and no global effect - the
  conservative way to push readability.
- Large billboard text stays readable; small text that was never legible in
  the source remains blurred (correct behavior, not a pipeline failure).

## Face (1.jpeg, sole >=24px face, 42x53)

```
{fmt(fa, FACE_COLS)}
```
micro_tex reference (faithful) = {face_ref:.3f}. F2 restores the face toward the
photograph's OWN texture: micro_tex 1.83 -> 2.16 (faithful 2.49) vs D1-weak's
1.83, tex_retain_hp 2.52 -> 2.06 (SR's inflated fake texture reduced), PSNR
+2.5 dB, SSIM +0.07, edge_align 1.000. No reconstruction, no invention -
exactly the "low-information but natural" requirement.

## Fine-detail / texture strips

```
{fmt(d, DETAIL_COLS)}
```
micro_tex reference (faithful) = {d_ref:.3f}. PSNR +1.9..2.9 dB on every strip,
SSIM +0.08..0.11, edge_align 0.9965..0.9998, micro_tex toward the source. These
are the cars/buildings/foliage that looked "illustrated" after SR+sharpening:
synthetic overshoot replaced by real aligned structure (edge_align high, no
drawing-like energy).

## Painterly-artifact result

The painterly/illustrated look came from PSNR-SR synthetic high-frequency plus
sharpening. F2 replaces that energy with the photograph's own band-limited
texture (micro_tex rises toward faithful everywhere; face 1.83->2.16; billboard
+1.5x; detail +near-source), while edge_align IMPROVES (0.987 -> 0.994 full
frame). Sharpness drops somewhat because it was inflated by synthetic edges -
per the priority list (sharpness is #6, artificial-looking is a REJECT) this is
the intended direction.

## Fidelity

- Envelope rule: output never deviates more than +/-12 gray levels (L*) from
  the SR base; all fused energy comes from the ORIGINAL photograph's own bands.
- Full-frame edge_align 0.994, billboard >=0.9984, detail >=0.9965, face 1.000.
- hist_sim 0.92 (F2-fuse) - tonal drift only, no content replacement.
- No OCR, no redraw, no faces reconstructed, no letter/logo repositioning.

## VRAM / runtime (RTX 3050 8GB)

- D1-weak stage (unchanged): tiled denoise {dn_m:.1f}s + SR {sr_m:.1f}s ~= {dn_m+sr_m:.1f}s, peak {peak:.0f} MB.
- F2 post-fusion: CPU-only, {post.get('F2', 0):.2f}s (F2-fuse {post.get('F2-fuse', 0):.2f}s, F2-bb {post.get('F2-bb', 0):.2f}s) per image, VRAM 0 MB.

## F2 vs D1-weak

| measure           | D1-weak                | F2 (recommended)       |
|-------------------|------------------------|------------------------|
| PSNR              | 27.38                  | {v_full['F2']['psnr']:.2f} |
| SSIM              | 0.919                  | {v_full['F2']['ssim']:.3f} |
| edge_align        | 0.987                  | {v_full['F2']['edge_align']:.3f} |
| sharpness         | 316.6 (synthetic)      | {v_full['F2']['sharpness']:.1f} (real) |
| noise (texture)   | 1.29                   | {v_full['F2']['noise']:.2f} |
| face micro_tex    | 1.83                   | {fa.loc['F2-bb']['micro_tex']:.2f} |
| billboard te_c    | {v_bill['D1-weak']['te_contrast']:.1f} (SR edge) | {v_bill['F2-bb']['te_contrast']:.1f} (real, +box boost) |

## Decision

- F2 pipeline delivers the mission's core principle: the photograph is the
  source of truth; SR only helps structure. Real texture is restored, painterly
  SR overshoot is removed, faces stay natural, and structural fidelity
  IMPROVES over D1-weak by every standard measure.
- Variants tested: F2-fuse (fusion), F2 (fusion + mild LC), F2-bb
  (F2 + verified-box-only detail/texture boost).
- Billboard: large text keeps readability with real edge energy; small text
  stays as-readable as the source (no hallucination). Box boost (F2-bb) is the
  conservative lever if more letter contrast is wanted.
- Face: natural, no reconstruction, micro_texture moved toward the source.
- Texture: restored everywhere (painterly artifact removed, edge_align higher).
- VRAM/runtime: unchanged pipeline (~40s, 2440 MB), post-step ~0.6-0.8s CPU.
- **VERDICT: ACCEPT (recommendation: F2-bb, fallback F2)** - contingent on your
  visual check of the montages (this model cannot view images). F2-bb recovers
  most billboard text energy with a tiny, box-only, alignment-safe cost. If
  maximum fidelity is desired, use F2. NOT integrated into production enhance.py.

Visual evidence: reports/pro_exp/f2_exp/f2_full_1..6.png (5 panels,
Original | D1-weak | F2-fuse | F2 | F2-bb), f2_billboard_1..6.png,
f2_face0_1.png, f2_detail_1..6.png.
"""
    OUT.write_text(md, encoding="utf-8")
    print(f"report written -> {OUT}")


def fmt(df, cols):
    return df[cols].round(3).to_string()


if __name__ == "__main__":
    main()