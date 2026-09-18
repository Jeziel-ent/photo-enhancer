# MVP 2 Pipeline — What Changed and Why

Companion to `docs/MVP2_RESEARCH.md` (the evidence) and
`docs/MVP2_BENCHMARK.md` (the regression results). This document is the
concrete "what is different from MVP 1" reference.

## Pipeline diagram

```
Input (any resolution/aspect ratio)
 │
 ├─► aspect_preserving_target(w0, h0)          [NEW — Phase 2]
 │     longest side -> 3840, aspect preserved exactly, even dimensions
 │
 ├─► normalize_daylight_appearance(img)         [NEW — Phase 4]
 │     ├─ adaptive_tonal_correction (MVP 1, unchanged): bounded luminance
 │     │  exposure/contrast fix, adaptive (no-op on a well-exposed photo)
 │     └─ correct_color_cast (NEW): bounded Lab a/b warm/cool cast
 │        reduction, adaptive (no-op on an already-neutral photo)
 │
 ├─► simple_upscale (Lanczos) -> faithful, now at `target` (not fixed 4K)
 │
 ├─► D1-weak: Restormer real-denoise + SwinIR-M x4 SR -> resized to
 │     `target` (was hardcoded 3840x2160 — the one function that ignored
 │     `target` entirely before this session; now fixed) [GPU]
 │     — or classical denoise + IMDN x4 via OpenVINO -> `target` [CPU]
 │
 ├─► F3-natural -> G7-MS multiscale detail -> A+ whole-frame detail
 │     (UNCHANGED — see docs/MVP2_RESEARCH.md Phase 5 for why no SR/detail
 │     model change was made this session)
 │
 ├─► Candidate A billboard-box branch (still dead code for this product —
 │     unchanged, still neutralized by engine_adapter.py)
 │
 └─► resize-to-target guard -> OUTPUT (target BGR uint8, e.g. 3840x2880
       for a 4:3 input — NOT always 3840x2160)
```

## Files changed

### `image_enhancer/src/enhance.py` (GPU path)
- Added `aspect_preserving_target(width, height, max_dim=3840)`.
- `_final_d1_weak(orig, target=(OUT_W, OUT_H))`: now accepts and honors
  `target` (previously hardcoded `(OUT_W, OUT_H)` in both its GPU and CPU-ish
  branches, ignoring whatever `final_enhance`'s own `target` argument was).
- `_final_scale_box(box, w_orig, out_w=OUT_W)`: takes an explicit `out_w`
  instead of the hardcoded module constant (dead code in production today —
  billboard boxes are always neutralized — fixed for correctness
  completeness).
- `final_enhance`: threads `target` into both of the above. **Default
  parameter values are unchanged** (`target=(OUT_W, OUT_H)`), so every
  existing direct caller that never passes `target` — including
  `image_enhancer/tests/test_final_pipeline.py` — is byte-identical to
  before.
- `img_toned, tonal_meta = tonal_correction.adaptive_tonal_correction(img, ...)`
  → `tonal_correction.normalize_daylight_appearance(img, ...)`.

### `image_enhancer/src/enhance_shared.py` (CPU path)
- Added its own copy of `aspect_preserving_target` (duplicated deliberately
  — this module must never import `enhance.py`, which imports torch; see
  this module's own docstring for the documented reason).
- `cpu_final_enhance(img, target=(OUT_W, OUT_H))`: previously took **no**
  `target` parameter at all; now accepts one (default unchanged) and honors
  it throughout (`simple_upscale`, the D1 resize, the final resize guard).
- Same `adaptive_tonal_correction` → `normalize_daylight_appearance` swap.

### `image_enhancer/src/tonal_correction.py`
- `adaptive_tonal_correction` — **completely unchanged** (still
  luminance-only; its own dedicated chroma-stability test still passes
  unmodified).
- Added `correct_color_cast(bgr, return_meta=False)`: bounded, deterministic,
  adaptive Lab a/b cast correction (see MVP2_RESEARCH.md Phase 4 for the
  full design rationale and evidence).
- Added `normalize_daylight_appearance(bgr, return_meta=False)`: composes
  the two, in the same order the pipeline already ran tonal correction
  (before any SR/detail stage). This is the new entry point `enhance.py`/
  `enhance_shared.py` call instead of `adaptive_tonal_correction` directly.

### `backend/engine_adapter.py` (GPU dispatch)
- `enhance_image()`: computes
  `target = engine.aspect_preserving_target(img.shape[1], img.shape[0])`
  right after loading the image, and passes it through
  `_run_with_stage_ticker` into `engine.enhance(METHOD, img, target=target)`.
  This is the ONE call site that turns the aspect-ratio-preserving formula
  from "available" into "actually used in production."

### `backend/cpu_worker.py` (CPU dispatch)
- `_run_one()`: computes
  `target = shared.aspect_preserving_target(img.shape[1], img.shape[0])`
  and passes it to `shared.cpu_final_enhance(img, target=target)`. Mirrors
  the GPU dispatch change exactly.

### Frontend: `frontend/src/components/upload/BillboardCanvas.tsx`
- Replaced the hardcoded `IMAGE_W = 3840` / `IMAGE_H = 2160` module
  constants with `imageWidth`/`imageHeight` props (defaulting to 3840x2160
  for a caller that hasn't loaded real dimensions yet — preserves old
  behavior exactly for that case). Every percent↔pixel conversion
  (`pctToImageRect`, `imageRectToPct`, `pctToBillboardRect`, the minimum-size
  check, the container's CSS `aspect-ratio`) now uses the real per-image
  dimensions.

### Frontend: `frontend/src/lib/imageDimensions.ts` (new)
- `loadImageDimensions(src)`: loads an image off-DOM just to read its real
  `naturalWidth`/`naturalHeight`.

### Frontend: `frontend/src/components/upload/JobResultPanel.tsx`
- New `dimensionsByResult` state, populated via `loadImageDimensions` for
  whichever gallery result (or single-photo comparison) is currently
  active, cached per result id. Falls back to 3840×2160 for the brief
  moment before a freshly-selected image's real size is known.
- `BillboardCanvas` and `BoardThumb` (the board-list thumbnail crop
  preview, which also hardcoded `/3840`/`/2160`) both now receive the
  active image's real dimensions.

## What did NOT change

- GPU/CPU device selection and detection (`backend/device.py`,
  `backend/engine_adapter.py`'s `apply_device_preference`) — untouched.
- Restormer, SwinIR-M, IMDN model weights and inference code — untouched.
- Billboard region neutralization (`_neutralize_billboard_regions`) —
  untouched; billboard boxes remain dead code for this product.
- `backend/billboard_overlay.py` (export-time board outline burn-in) —
  already resolution-agnostic (reads real image dims at compose time), no
  changes needed.
- Single-photo vs. gallery/batch UI behavior, export formats (PNG/JPG/JPEG),
  batch ZIP — untouched; the aspect-ratio and dimension-detection changes
  are purely about what size the image *is*, not how it's edited or
  exported.
- The frontend/backend API contract (`GET/POST /api/jobs/...`) — unchanged;
  no new endpoint or field was needed (the frontend already discovers each
  result's real dimensions client-side from the image bytes themselves).

## Processing time impact

The aspect-ratio change redistributes the same total resize/SR work across
different final dimensions (a 1:1 3840×3840 output is larger than 16:9's
3840×2160, but a 21:9 3840×1606 output is smaller — no systematic
increase). The daylight normalization stage (LAB a/b shift + one CLAHE
pass) is CPU/numpy-only, a few milliseconds, no new GPU call. Both stay
inside the existing measured ~25–45s/image budget
(`docs/PERFORMANCE_OPTIMIZATION.md`) — confirmed by a real end-to-end run
of the full production call path on a real photo: **45.9s**, within the
review's ≤60s target (see `docs/MVP2_RESEARCH.md` "Daylight tuning pass").

## Daylight tuning pass (follow-up to the initial MVP 2 implementation)

The initial `normalize_daylight_appearance` (composed of
`adaptive_tonal_correction` + `correct_color_cast`) measurably
under-corrected a real hazy/overcast street photo — diagnosed as a
structural bug (a large blown-out sky dominating whole-frame percentile
stats and triggering an inappropriate whole-frame DARKENING), not just an
undertuned constant. Fixed with three changes to
`image_enhancer/src/tonal_correction.py`, no new model, no new GPU call:

1. `_histogram_stats`/`detect_issues`/`_build_lut` now use **scene-aware**
   percentiles (blown-highlight cluster excluded) for exposure/contrast
   decisions, so a bright sky can no longer misclassify a dim street scene
   as "overexposed."
2. `correct_color_cast`'s target moved from clinical neutral to a small,
   deliberately-warm daylight point (`DAYLIGHT_B_TARGET=132` vs. neutral
   128), and its strength raised (`CORRECTION_STRENGTH` 0.65 → 0.80) — both
   still hard-bounded (`MAX_CHROMA_SHIFT=18` unchanged) and still adaptive
   (no-op on a photo already at the target).
3. New `enhance_local_clarity`: a bounded CLAHE-based local-contrast boost
   (chroma untouched), addressing the "hazy/flat" look a global LUT alone
   cannot fix — reuses the exact CLAHE configuration already precedented in
   `enhance.py`'s own `classical_preprocess`.

Full before/after evidence (real photo, real production pipeline, plus an
adaptive-restraint check against the 6 existing well-lit reference photos)
is in `docs/MVP2_RESEARCH.md`'s "Daylight tuning pass" section.
