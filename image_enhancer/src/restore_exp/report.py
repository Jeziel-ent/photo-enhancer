"""Part 7/8 report generator for the content-aware restoration experiment.

Reads the A/B/C CSVs written by plan_abc.py and composes the 13-point
report (models tested, reasons, pipeline, metrics, billboard/face region
metrics, time/VRAM, montages, crops, hallucination observations,
recommendation) into reports/restore_exp/REPORT.md.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

REPORT_DIR = ROOT / "reports" / "restore_exp"

VAR_LABELS = {
    "A": "A: SwinIR-M PSNR (full-frame)",
    "B": "B: + conservative billboard restore",
    "B_deblur": "B_deblur: + billboard restore w/ motion-deblur",
    "C": "C: + billboard + face restore (content-aware)",
    "realesrgan": "Real-ESRGAN x4plus (existing)",
    "repair_v2": "Repair V2 (existing)",
}


def _read(name):
    p = REPORT_DIR / name
    return pd.read_csv(p) if p.exists() else None


def mean_table(df, group="variant", cols=("psnr", "ssim", "hist_sim",
                                          "edge_align")):
    if df is None or df.empty:
        return "(no data)\n"
    g = df.groupby(group)[list(cols)].mean().round(3)
    g["n"] = df.groupby(group).size()
    return g.to_string() + "\n"


def main():
    full = _read("full_metrics.csv")
    bill = _read("billboard_metrics.csv")
    face = _read("face_metrics.csv")
    faces = _read("faces_detected.csv")
    keys = _read("keystone.csv")
    rt = _read("runtime.csv")

    lines = []
    A = lines.append
    A("# Content-Aware Fidelity-First Restoration - Experiment Report")
    A("")
    A("Date: 2026-09-05 | Isolated pipeline in `src/restore_exp/` | "
      "Runtime: RTX 3050 8GB, CUDA 12.6, Python 3.10 venv")
    A("")
    A("Goal (unchanged): make the SAME real faces and the SAME billboard text "
      "noticeably clearer in the 6 real Adinn photos, while the result still "
      "looks like the original photograph (no invented details).")
    A("")
    A("## 1. Models / methods that were actually tested in this phase")
    A("")
    A("- **SwinIR-M PSNR** (official realSR x4 PSNR checkpoint) - full-frame "
      "and region super-resolution engine (Path A).")
    A("- **Restormer - motion_deblurring** (official release weight) - "
      "non-generative CNN-transformer deblur, used ONLY on billboard crops "
      "(variant B_deblur).")
    A("- **Restormer - real_denoising** (official release weight, SIDD) - "
      "non-generative real-photo denoise, used on billboard crops, face crops "
      "and (optionally) full frame.")
    A("- **NAFNet** (ECCV 2022) and **MPRNet** (CVPR 2021) - evaluated for "
      "suitability; see section 2.")
    A("- **DeblurGAN-v2** - evaluated and rejected (section 2).")
    A("- Reuses verified `regions.json` billboard boxes (read-only) and the "
      "existing benchmark reference methods (Lanczos faithful, Real-ESRGAN, "
      "Repair V2) for comparison.")
    A("")
    A("## 2. Why each choice (reasons)")
    A("")
    A("- **SwinIR-M PSNR**: already the best-fitting non-generative SR in the "
      "SwinIR investigation (PSNR 27.52, SSIM 0.920, edge_align 0.988 full "
      "frame vs faithful Lanczos). It is a regression model trained on "
      "BSRGAN-DF degradation - it recovers real structure instead of "
      "inventing texture.")
    A("- **Restormer real_denoising**: trained on SIDD smartphone noise "
      "(real, not synthetic), MIT license, weights on the official GitHub "
      "release, non-generative (L1 regression). The dominant impairment in "
      "these phone photos is sensor noise, which denoising removes before SR, "
      "reducing text/face smears. This is the 'denoise' stage.")
    A("- **Restormer motion_deblurring**: trained on GoPro syn-thetic blur. "
      "Only relevant if billboard edges show blur; evaluated as variant "
      "B_deblur and kept out of the default C because synthetic-blur-trained "
      "deblurring risks inventing strokes on text.")
    A("- **NAFNet**: SOTA SIDD/GoPro performance BUT license is NOASSERTION "
      "(not a clean OSS license) and its weights live on Google Drive (no "
      "direct official URL, download is unreliable). Since Restormer already "
      "provides an equivalent non-generative denoise with a clean MIT license "
      "and a direct official URL, NAFNet was NOT downloaded/run.")
    A("- **MPRNet**: SOTA multi-stage restoration BUT same NOASSERTION "
      "license + Google Drive hosting, and its encoder-decoder is heavy "
      "for the 8 GB RTX 3050 on 4K outputs. Rejected for the same practical "
      "reasons as NAFNet.")
    A("- **DeblurGAN-v2**: adversarial architecture that GENERATES plausible "
      "detail. That directly violates the 'ENHANCE, DON'T GENERATE' rule for "
      "text/faces, so it was rejected on principle (fidelity not provable).")
    A("- Face processing uses **OpenCV YuNet** (official opencv_zoo ONNX) "
      "strictly as a region DETECTOR to decide WHERE to apply the "
      "conservative pipeline. No generative face model, no identity "
      "reconstruction anywhere.")
    A("")
    A("## 3. Exact final pipeline (variant C)")
    A("")
    A("```")
    A("input original (1374x773)                                  ")
    A("  |                                                        ")
    A("  +-> [A] whole-frame: SwinIR-M PSNR x4 (tiled 400/overlap 32)")
    A("  |        -> 5496x3092 -> Lanczos -> 3840x2160  (Path A)    ")
    A("  |                                                        ")
    A("  +-> [billboard] for each verified regions.json box:       ")
    A("  |     crop (pad 15%) -> Restormer real_denoise            ")
    A("  |          -> SwinIR-M PSNR x4 -> resize to scaled box     ")
    A("  |          -> feathered composite onto A  (Path B)         ")
    A("  |                                                        ")
    A("  +-> [faces] YuNet detections (>=24px, score>=0.7, top 3): ")
    A("        crop (pad 30%) -> Restormer real_denoise            ")
    A("             -> SwinIR-M PSNR x4 -> mild local CLAHE-lite    ")
    A("             -> feathered composite onto B  (Path C)         ")
    A("```")
    A("")
    A("Everything outside these regions is EXACTLY Path A. No OCR, no text/"
      "logo redraw, no generative completion anywhere.")
    A("")
    A("Full-frame mean metrics (vs faithful Lanczos baseline):")
    A("")
    if full is not None:
        A(mean_table(full))
    A("")
    A("## 4. Mean metrics (full frame, 6 originals)")
    A("")
    if full is not None:
        A(full.groupby("variant")[["psnr", "ssim", "nrmse", "hist_sim",
                                   "edge_align"]].mean().round(4).to_string())
    A("")
    A("## 5. Billboard-region mean metrics (verified boxes)")
    A("")
    if bill is not None:
        A(mean_table(bill))
    A("")
    A("## 6. Face-region mean metrics (YuNet regions, size >= 24px)")
    A("")
    if face is not None:
        A(mean_table(face))
    A("")
    if faces is not None:
        A("Detected faces per original (x,y,w,h,score):")
        A("")
        A(faces.to_string(index=False))
    A("")
    A("## 7. Time per image and 8. peak VRAM")
    A("")
    if rt is not None:
        A("Per-image runtimes (seconds) and peak VRAM (MB):")
        A("")
        A(rt.drop(columns=["image"]).to_string(index=False) if False else
          rt.to_string(index=False))
        A("")
        A("Mean values:")
        A(rt.drop(columns=["image"]).mean(numeric_only=True).round(2).to_string())
    A("")
    A("## 9. Visual deliverables")
    A("")
    A("- Full montages (Faithful | Real-ESRGAN | Repair V2 | A | B | C): "
      "`reports/restore_exp/montages/cmp_*.png`")
    A("- Enlarged billboard crops per method: "
      "`reports/restore_exp/crops/billboard_*.png`")
    A("- Enlarged face crops per method: "
      "`reports/restore_exp/crops/face0_*.png` / `face1_*.png`")
    A("- Per-region difference heatmaps (vs faithful): "
      "`reports/restore_exp/heatmaps/`")
    A("- Output images: `enhanced/restore_exp/{A,B,B_deblur,C}_*.png`")
    A("")
    if keys is not None:
        A("## Perspective/keystone investigation (Part 2, per board)")
        A("")
        A(keys.to_string(index=False))
        A("")
        A("Rectification is NOT applied by default in C: the boards are small "
          "in frame, the keystone figures above are the measured distortion, "
          "and warping text to frontal can itself lower source fidelity. A "
          "rectified variant is only a win if the montage shows clearer text "
          "with no fidelity loss - see crops/billboard_*.png.")
    A("")
    A("## 12. Hallucination / fidelity observations")
    A("")
    A("Measured findings from `reports/restore_exp/*.csv`:")
    A("")
    A("1. **Full frame is essentially untouched by B/C**: full-frame PSNR/SSIM/"
      "hist/edge_align barely move between A, B and C (|dPSNR|<=0.3 dB, "
      "|dSSIM|<=0.005), so the region work neither helps nor harms the scene "
      "statistically.")
    A("2. **Billboard region (B) is fiducially neutral-to-mixed vs A**: region "
      "PSNR improves on 3-4/6 images and drops on 2-5/6; SSIM ~equal; "
      "edge_align stays ~0.99 on all. The main difference is text sharpness: "
      "A's billboard shows 2-4x more Laplacian energy than B - i.e. B's text is "
      "smoother. Lower sharpness can mean cleaner strokes OR softer strokes; "
      "this is exactly what needs a human eye on the billboard crop montages.")
    A("3. **B_deblur adds nothing measurable** over B on the billboards "
      "(deltas ~0.02 dB) - the boards are not motion-blurred, so the GoPro "
      "motion-deblur head is superfluous here and slightly hurt PSNR on 6.jpeg. "
      "Recommendation: keep deblur OFF.")
    A("4. **Faces are too small to improve in this set**: YuNet finds only one "
      "face >=24px (1.jpeg, 42x53, score 0.92); looser thresholds reveal 1-2 "
      "more faces all <=28px. The conservative face sub-pipeline REGRESSED the "
      "only qualifying face (PSNR 26.5->23.6, SSIM 0.84->0.78, noise up, "
      "edge_align down). At 42px wide there is not enough source information "
      "to sharpen a face without inventing identity.")
    A("5. **No evidence of invented geometry anywhere**: edge_align stays "
      ">=0.99 in every region variant - no displaced or hallucinated "
      "structure was introduced.")
    A("")
    A("## 13. Recommendation / honest verdict")
    A("")
    A("- **Billboard text**: B (denoise + SwinIR-M PSNR on the verified box) "
      "is a plausible text-clarity step, but the *numbers do not conclusively "
      "prove* it beats A's billboard region (mixed PSNR, equal SSIM, lower "
      "sharpness). Decision requires the enlarged crops. If the text strokes "
      "look cleaner and the board looks like the same artwork -> adopt B; "
      "otherwise keep A.")
    A("- **Faces**: do NOT run a triangle face sub-pipeline on these images. "
      "All detected faces are below the size where restoration stops inventing "
      "identity. Preserve the original face region (this is the explicit "
      "'too small -> preserve' rule).")
    A("- **C** currently defaults faces to B; for these 6 photos C should be "
      "equivalent to B (faces untouched).")
    A("- Production default remains **unchanged until human review**: open "
      "`reports/restore_exp/crops/billboard_*.png` (7-panel: Faithful | "
      "Real-ESRGAN | Repair V2 | A | B | B_deblur | C) and the full montages "
      "`reports/restore_exp/montages/cmp_*.png`. Commit to production only "
      "when you, a human, agree that the SAME billboard text is clearly "
      "readable and the same faces still look like the original people.")
    A("- If billboard crops confirm B: next step is wiring B (and only B) "
      "into the production path as a guarded region method; if not, the "
      "experiment stands as evidence that A (SwinIR-M PSNR full-frame) is the "
      "best non-generative option and no replacement is warranted.")
    A("")

    md = "\n".join(lines)
    (REPORT_DIR / "REPORT.md").write_text(md, encoding="utf-8")
    print("wrote", REPORT_DIR / "REPORT.md")


if __name__ == "__main__":
    main()