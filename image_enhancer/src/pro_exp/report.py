"""Writes reports/pro_exp/REPORT.md from the experiment CSVs.

Usage:
  .venv/Scripts/python src/pro_exp/report.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
REPORT_DIR = ROOT / "reports" / "pro_exp"


def _mean(df, cols=None):
    if df is None or df.empty:
        return None
    d = df.drop(columns=["image"])
    num = d.select_dtypes(include="number")
    return num.mean(numeric_only=True)


def _pivot(df, value, index="variant"):
    if df is None or df.empty:
        return ""
    t = df.pivot_table(index=index, values=value, aggfunc="mean").round(3)
    t = t.sort_values(value, ascending=False)
    return t.to_string()


def main():
    full = _read("full_metrics.csv")
    bill = _read("billboard_metrics.csv")
    face = _read("face_metrics.csv")
    text = _read("text_proxies.csv")
    detail = _read("detail_metrics.csv")
    run = _read("runtime.csv")
    dets = _read("faces_detected.csv")

    lines = []
    a = lines.append

    a("# Professional Photo Enhancement - isolated experiment report")
    a("")
    a("Goal: make the real Adinn phone photos *look professionally enhanced* "
      "outside the board: clearer faces, clearer billboard/text, finer detail, "
      "more photographic quality -- while keeping the photograph as the only "
      "source of truth (ENHANCE, DO NOT GENERATE). All variants are "
      "non-generative; no GFPGAN/CodeFormer/OCR-regeneration/diffusion.")
    a("")
    a("Compare every variant against the current best non-generative SR "
      "baseline **A = SwinIR-M PSNR x4 full frame** (from the completed "
      "Restormer experiment), and both against the **faithful Lanczos** "
      "upscale of the original.")
    a("")
    a(f"Inputs: {len(full['image'].unique()) if full is not None else 0} real "
      "Adinn phone photos. Reference = faithful Lanczos 3840x2160 of each "
      "original (there is no ground truth).")
    a("")
    a("## Variants")
    a("")
    a("| id | recipe |")
    a("|---|---|")
    a("| A | SwinIR-M PSNR x4, full frame (baseline) |")
    a("| D1 | Restormer real-denoise BEFORE SwinIR-M PSNR x4 |")
    a("| D2 | SwinIR-M PSNR x4, then Restormer real-denoise on the 4K result |")
    a("| P2 | A + ProLook lite (deterministic professional polish: luminance clarity, mild USM, gentle tone, soft vibrance) |")
    a("| P2b | A + ProLook standard (stronger polish) |")
    a("| P3 | P2 full frame + feathered composite of billboard RECb crops |")
    a("| RECa/RECb/RECc | billboard crop: SwinIR direct / denoise then SwinIR / RECb + ProLook |")
    a("| FC | face crop: denoise -> SwinIR -> ProLook lite |")
    a("")
    a("## Method")
    a("")
    a("- Full-frame and region fidelity: PSNR / SSIM / NRMSE / hist_sim / "
      "edge_align vs the faithful upscale (higher = more source content "
      "preserved), plus no-reference sharpness/noise/contrast/edge_density.")
    a("- Billboards use the manually verified `regions.json` boxes; faces use "
      "YuNet detection (identification only).")
    a("- Fine-detail strip = automatically picked sharpest informative window "
      "outside billboard/face boxes.")
    a("- Text-legibility proxies (no-reference, on each variant's crop): "
      "`te_contrast` = mean gradient magnitude of the strongest 5% edge "
      "pixels; `hp_energy` = std of the 2px high-pass. Higher is sharper, but "
      "the metric cannot judge legibility - the human looks at the crops.")
    a("")

    if run is not None:
        a("## Runtime")
        a("")
        a("```")
        a(_mean(run).round(3).to_string())
        a("```")
        a("")

    if full is not None:
        a("## Full-frame fidelity vs faithful Lanczos (mean)")
        a("")
        a("```")
        a(_pivot(full, "psnr"))
        a("")
        a(_pivot(full, "ssim"))
        a("")
        a("edge_align:")
        a(_pivot(full, "edge_align"))
        a("")
        a("hist_sim:")
        a(_pivot(full, "hist_sim"))
        a("")
        a("sharpness (no-ref, higher=sharper):")
        a(_pivot(full, "sharpness"))
        a("")
        a("noise (no-ref, low=clean but zero=grain removed):")
        a(_pivot(full, "noise"))
        a("```")
        a("")

    if bill is not None:
        a("## Billboard region metrics vs faithful crop (mean per variant)")
        a("")
        a("```")
        a("PSNR:")
        a(_pivot(bill, "psnr"))
        a("")
        a("SSIM:")
        a(_pivot(bill, "ssim"))
        a("")
        a("edge_align:")
        a(_pivot(bill, "edge_align"))
        a("")
        a("sharp_gain_x (vs faithful crop, >1 = sharper):")
        a(_pivot(bill, "sharp_gain_x"))
        a("```")
        a("")

    if text is not None:
        a("## Text-legibility proxies on billboard crops (mean)")
        a("")
        a("```")
        a("te_contrast:")
        a(_pivot(text, "te_contrast"))
        a("")
        a("hp_energy:")
        a(_pivot(text, "hp_energy"))
        a("```")
        a("")

    if face is not None and not face.empty:
        a("## Face region metrics vs faithful crop (mean per variant)")
        a("")
        a("```")
        a("PSNR / SSIM:")
        a(_pivot(face, "psnr"))
        a("")
        a(_pivot(face, "ssim"))
        a("")
        a("sharp_gain_x:")
        a(_pivot(face, "sharp_gain_x"))
        a("```")
        a("")

    if dets is not None and not dets.empty:
        a("## Faces found (YuNet)")
        a("")
        a("```")
        a(dets[["image", "face", "w", "h", "score"]].to_string(index=False))
        a("```")
        a("")

    if full is not None:
        a("## Reading the table (honest, unavoidable caveats)")
        a("")
        a("- **Faces are tiny** in these photos (largest is ~42x53px in the "
          "original). A down-and-up sub-scale cannot resurrect identity "
          "detail without *inventing* it. Win condition for the face line is "
          "therefore: **no regression + preserved structure**, not more "
          "detail. Any claim of 'better face' must come from viewing the "
          "face crops.")
        a("- PSNR/SSIM reward smoothness because the reference is a blurry "
          "upscale; a higher sharp_gain mostly reflects retained texture, not "
          "invented detail (edge_align stays ~1 for all variants => no "
          "geometry is displaced).")
        a("- ProLook adjusts luminance/chroma deterministically and never "
          "adds pixels, so hist_sim/edge_align/preserve-only checks are "
          "expected to stay high.")
        a("- **Final word is visual.** Open the montages and, above all, the "
          "enlarged crops in `reports/pro_exp/crops/` and judge: faces "
          "natural, billboard text legible, fine detail finer, overall "
          "photographic quality - and nothing invented.")
        a("")
        a("*Isolated experiment. Production `src/enhance.py` untouched. No "
          "adoption until you approve the visual results.*")

    out = REPORT_DIR / "REPORT.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


def _read(name):
    p = REPORT_DIR / name
    if not p.exists():
        return None
    return pd.read_csv(p)


if __name__ == "__main__":
    main()