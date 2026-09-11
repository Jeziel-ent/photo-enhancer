# Image Enhancement Engine (Adinn 4K Image Enhancer)

Builds a **real 4K (3840×2160) enhancement pipeline** for real phone-captured
Adinn signage photos.

**Hard rule:** Enhance only. The engine MUST NOT generate, add, remove,
replace, or hallucinate any content. Original billboard artwork/text/logos,
people, vehicles, buildings, roads, surroundings, and geometry must be
preserved exactly.

Allowed operations: denoise, deblur/restoration, super-resolution, sharpening,
exposure/contrast/color correction, artifact reduction.

---

## Environment

- **GPU:** NVIDIA GeForce RTX 3050 (8 GB), CUDA 12.6, driver 560.94
- **Python:** 3.10.0 (venv at `.venv/`)
- **PyTorch:** 2.14.0+cu126 (CUDA confirmed working)
- **OpenCV:** 5.0.0

### Setup (Windows)

```powershell
# 1. Create venv with Python 3.10 (3.14 is too new for the ML stack)
C:\Users\mjezi\AppData\Local\Programs\Python\Python310\python.exe -m venv .venv

# 2. Activate and install
.venv\Scripts\Activate.ps1
pip install -U pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# 3. Download Real-ESRGAN weights (already cached in models/)
#    RealESRGAN_x4plus.pth -> models/
```

Known fix applied: `basicsr/data/degradations.py` import
`torchvision.transforms.functional_tensor` was renamed to
`torchvision.transforms.functional` for newer torchvision.

---

## Directory layout

```
phase1_enhancement/
├── src/
│   ├── enhance.py        # 3 enhancement methods + faithful upscale baseline
│   ├── benchmark.py      # runs all methods -> 4K outputs + per-image metrics
│   ├── compare_report.py # fidelity report + automated preserve-only checks + montages
│   ├── verify_regions.py # MANUAL billboard-box review/editing (overlay montages, interactive)
│   ├── billboard.py      # EXPERIMENTAL auto-detector — NOT used by the benchmark (see below)
│   ├── quality.py        # PSNR/SSIM/NRMSE/hist-sim/edge-align + sharpness/noise/contrast
│   └── make_testscene.py # (dev only) synthetic scenes used only for smoke-testing
├── originals/            # 6 real Adinn phone-captured photos (copies; source untouched in assets/OOH_Images)
├── references/           # reserved (modified ChatGPT examples are NOT ground truth)
├── enhanced/             # 3840×2160 outputs, one PNG per (image, method)
├── models/               # RealESRGAN_x4plus.pth
├── reports/
│   ├── benchmark_metrics.csv      # raw per-image/enhancement metrics
│   ├── fidelity_comparison.csv    # fidelity vs faithful original upscale
│   └── comparisons/               # original|enhanced side-by-side montages
├── requirements.txt
└── README.md
```

Source images are kept untouched at `..\assets\OOH_Images\*.jpeg`.

---

## Methods benchmarked

| # | ID | Description | VRAM |
|---|----|-------------|------|
| A | `realesrgan` | Real-ESRGAN x4plus true 4× model SR (RRDBNet) → exact 4K | ~2 GB |
| B | `tiled` | Same SR but explicitly tiled (1024 px, 120 overlap) + feathered seam blend (VRAM control/stitching) | ~0.5 GB |
| C | `billboard` | preserve the ORIGINAL scene + super-resolve each **manually verified** billboard region(s) at 4×, feathered composite back over the original | ~0.5–2 GB |
| D | `repair` | **restoration-first**: deblur+denoise (bilateral) → Real-ESRGAN 4× → unsharp | ~2 GB |
| V2 | `repair_v2` | **conservative 'enhance-don't-generate'**: texture-preserving mild denoise → Real-ESRGAN 4× → no final sharpen (natural faces first) | ~2 GB |
| – | `classical` | OpenCV-only: fastNlMeans → CLAHE → unsharp → Lanczos (baseline) | minimal |
| – | `hybrid` | classical preprocess → Real-ESRGAN (kept for reference) | ~2 GB |

> **Note:** The SR step uses `outscale=4` — the RRDB network performs the
> learned 4× upscale. (Earlier drafts used `outscale=1` + Lanczos, which was
> not true model SR; all numbers below are on the corrected true-4× basis.)

> **Repair V2 rationale:** Repair D sharpens text well but turns faces
> waxy/painterly/AI-looking — its strong bilateral pre-smoothing (sigmaColor
> 75) and post-unsharp violate the hard rule *enhance, don't generate*.
> `repair_v2` removes both, adds only a mild texture-preserving denoise, and
> applies NO final sharpening. No face detector/restoration is used
> (unreliable on street photos → conservative global processing by rule).
> Priorities: natural faces > source fidelity > billboard/text clarity >
> overall sharpness > noise reduction.

### How to run

```powershell
# 1) produce 4K enhanced outputs + raw metrics
python src\benchmark.py --methods billboard repair repair_v2

# 2) fidelity report + automated preserve-only checks + comparison montages
python src\compare_report.py --methods billboard repair repair_v2
```

### Billboard regions are MANUAL, not auto-detected

The billboard pipeline uses **manually verified bounding boxes**, stored in
`regions.json`. A single photo may contain several billboards, so each image
entry holds a list of boxes — only the relevant Adinn board(s) are selected
by hand:

```json
"1.jpeg": { "boxes": [[x, y, w, h], ...], "image_size": [W, H] }
```

Method C **preserves the original image** and composites the enhanced
regions back into the original scene: everything except the verified
billboard(s) stays exactly as photographed. The automatic detector
(`billboard.py`) is **not used** for benchmarking (see *Billboard
auto-detection status* below).

Verification workflow:

```powershell
# 1. generate review montages with the current boxes drawn on the originals
python src\verify_regions.py

# 2. check each overlay in reports\billboard_review\review_<n>.png
#    correct boxes either by editing regions.json directly, or interactively:
python src\verify_regions.py --mode interactive
#      drag = draw a new box          Enter = add it (or replace Tab-selected)
#      Tab  = cycle selected box      x/r = delete / reset
#      s    = save + next             q/Esc = quit (+save / discard)

# 3. ONLY after every box is verified, run the billboard-crop benchmark
python src\benchmark.py --methods billboard repair
```

`benchmark.py` refuses to run method `billboard` while any image lacks a
verified region — the detector is never substituted silently.

### Billboard auto-detection status (experimental)

`billboard.py` (contours / Hough lines / bright-panel heuristics) is a
research experiment, **not production-ready**. Real OOH billboard photos
contain roads, sky, vehicles, windows and other rectangles that score above
the billboard, so generic rectangle detection is **insufficient for reliable
OOH billboard detection** — there is no learned notion of what a billboard is
(content, logos, lighting, context). The benchmark intentionally relies on
the manual regions; future detector work should move to a learned
object-detection approach and be scored against the manual boxes.

---

## Results (6 real Adinn images, mean across the set)

Fidelity is measured **against the faithful Lanczos upscale of the original**
(no enhancement) — the correct baseline given the preserve-only rule and that
the ChatGPT modified images are not ground truth.

| method | PSNR↑ | SSIM↑ | hist_sim↑ | edge_align↑ | edge_gain× | sharp_gain× | noise↓ | time/img |
|--------|-------|-------|-----------|-------------|------------|-------------|--------|----------|
| **C billboard** | **34.95** | **0.989** | **0.992** | 0.997 | 1.05 | 6.5 | 1.81 | ~1.4 s |
| D repair | 27.19 | 0.870 | **0.911** | **0.999** | 1.02 | 29.0 | **0.83** | ~7.2 s |
| V2 repair_v2 | 26.47 | 0.873 | 0.904 | 0.986 | 1.58 | 48.8 | 1.33 | ~7.2 s |

(Earlier baseline rows, unchanged: A realesrgan ≈ B tiled PSNR 25.79 / SSIM 0.859 /
edge_gain 1.96; classical SSIM 0.912; hybrid worst. Values for
A/B/classical/hybrid come from the 2026-09-04 full run.)

Metric meanings:
- **PSNR / SSIM / hist_sim** vs faithful upscale → how much source content is
  preserved (higher = more faithful).
- **edge_align** → fraction of enhanced edges that coincide with original
  structure. Close to 1 ⇒ no geometry displacement / hallucination.
- **edge_gain× / sharp_gain×** → edge-detail / Laplacian-variance delivered
  relative to the plain original upscale (>1 = sharper). The blurry upscale
  baseline has ~zero variance, so a high sharp_gain largely reflects how much
  high-frequency texture was retained/low-pass-filtered away.
- **noise** → no-reference grain estimate (higher = grainier).

### Reading the table
- **C (billboard)** preserves the ORIGINAL scene and super-resolves only the
  **manually verified** billboard region(s), composited back with a feathered
  edge. It is the strongest preserve-only behaviour — everything but the
  board(s) is pixel-identical to the baseline, which is why its PSNR/SSIM are
  near-perfect. Sharpness is low because the untouched scene stays soft.
- **D (repair)** keeps the best *smoothing-weighted* fidelity (high PSNR,
  hist_sim, edge_align 0.999, lowest noise). Its aggressive bilateral
  denoise + unsharp is exactly why faces turn **waxy/painterly** despite the
  strong numbers: the metric is measured against a blurry baseline, so it
  rewards smoothing, not naturalness.
- **V2 (repair_v2)** drops the pre-SR skin-smoothing and the post-SR unsharp.
  vs D it shows: **SSIM slightly higher (0.873)**, PSNR ~0.7 dB lower, and
  clearly more retained texture (edge_gain 1.58, sharp_gain 48.8) at a modest
  noise cost. Higher sharpness here means **kept texture/grain, not invented
  detail** — it is NOT a claim of being "better".
- Faces: the numeric table cannot judge naturalness. The decisive review is
  visual — see `reports/comparisons/cmp_repair_v2_*.png` vs
  `cmp_repair_*.png`.

### Automated preserve-only checks (all three methods PASS)
- SSIM ≥ 0.80 vs faithful upscale; PSNR ≥ 20 dB; edge_align ≥ 0.90;
- severe histogram drift < 0.65 flagged (tonal shifts from legitimate
  contrast/color correction are expected and not flagged).

---

## Recommendation

- **Scenes WITHOUT people** → select **`repair` (D)**: best raw fidelity
  (PSNR 27.2 dB, SSIM 0.870, hist 0.911, edge_align 0.999), cleanest
  text/logo edges, low grain, ~7.2 s/image.
- **Scenes WITH faces/people** → select **`repair_v2` (V2)**: drops the
  waxy/painterly face artifacts of D by removing the skin-smoothing bilateral
  and the post-SR unsharp. Same SSIM (0.873), ~0.7 dB less PSNR, more retained
  facial texture (edge_gain 1.58) at a mild noise cost — the trade that honors
  *enhance, don't generate*. **Confirm visually** in
  `reports/comparisons/cmp_repair_v2_*.png` that faces look more natural than
  `cmp_repair_*.png`.
- **C `billboard`** adds value when a billboard occupies a large, reliably
  isolated region of the frame (or several boards per photo); it preserves the
  rest of the scene exactly as the original and stays orthogonal to the
  global Repair methods.

Final enforcement: originals and enhanced outputs are kept in separate
folders; source files in `assets/OOH_Images/` are never modified.

*Benchmark date (repair/repair_v2/billboard run): 2026-09-05. GPU: RTX 3050, CUDA 12.6, PyTorch 2.14+cu126.*

