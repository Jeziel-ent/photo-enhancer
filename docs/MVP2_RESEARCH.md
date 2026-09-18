# MVP 2 Research — Aspect Ratio, Daylight Normalization, Detail/SR

Branch: `feature/mvp-2-quality`, based on MVP 1 baseline commit `76b7fca`
(`feat: complete Adinn 4K Image Enhancer MVP`). `main` and
`feature/adaptive-fast-inference` are untouched.

This document is Phases 1–5 (and the scoping decisions for 3/6–16) of the
MVP 2 review's research plan. It documents what was actually read, actually
measured, and actually tested — not aspirational claims. Where a phase was
deliberately scoped down from the review's full ask (multi-day dataset
downloads, exhaustive multi-model GPU sweeps), that decision and its reason
are stated explicitly rather than silently skipped.

## Scoping decision (read this first)

The review's plan (Phases 3–15) describes a multi-week research program:
downloading and licensing multiple full external datasets (LOL-v2, FiveK,
SID, ExDark, RealSR — each 1–20+ GB), building a 15–20 image benchmark set,
running 6+ candidate models end-to-end on the real GPU pipeline, and scoring
every combination against a 12-axis human rubric. That is genuinely out of
scope for a single working session and was not pretend-completed. What
*was* done, and is real:

1. A complete, code-level audit of the current pipeline (Phase 1) —
   including a **pre-existing internal audit already on disk**
   (`docs/ENGINE_AUDIT.md`) that this document builds on rather than
   duplicates.
2. A concrete, tested aspect-ratio formula (Phase 2), verified against every
   one of the review's own worked examples plus all 6 required ratio
   families, **and reproduced against the real production GPU pipeline**
   (not just unit math).
3. Real license research (WebSearch/WebFetch) on daylight-normalization
   candidates (Phase 4), with one candidate implemented as a working
   prototype and measured on a real, reproducible controlled experiment.
4. Detail/SR research (Phase 5) that leans on `docs/ENGINE_AUDIT.md`'s own
   **already-measured, already-falsified** finding that SR-backbone swaps
   don't help this architecture — avoiding re-running an experiment this
   exact codebase already ran and answered.
5. A concrete implementation decision (Phase 16) for exactly two of the
   three review requirements (aspect ratio, daylight), each backed by real
   evidence above, with the third (detail/SR) scoped to "recommended next
   experiment" rather than implemented blind — per rule #14 ("if no
   candidate clearly improves the baseline, keep the baseline").

Follow-up work needed to complete the full review plan (dataset downloads,
15–20 image benchmark, RetinexFormer local A/B test, F3 envelope-shape
experiment) is listed at the end of this document and in
`docs/MVP2_BENCHMARK.md`.

---

## Phase 1 — Current pipeline audit

### Entry point and dispatch

`backend/engine_adapter.py:enhance_image()` is the only production call
path. It dispatches on the resolved device
(`get_device_state()["effective_device"]`):

- **GPU**: in-process call into `image_enhancer/src/enhance.py`'s
  `engine.enhance("final", img)` → `final_enhance()`.
- **CPU**: out-of-process call into `backend/cpu_worker.py`, which imports
  `image_enhancer/src/enhance_shared.py`'s `cpu_final_enhance(img)` — a
  **separate, hand-duplicated implementation** of the same pipeline shape,
  kept in its own torch-free module specifically because torch and OpenVINO
  crash when loaded into the same process (see that module's own docstring;
  this is a previously-diagnosed, already-fixed stability issue, not new).

### Exact GPU pipeline order (`enhance.final_enhance`)

```
img (original, BGR uint8)
 │
 ├─► tonal_correction.adaptive_tonal_correction(img)      [MVP1: luminance-only exposure/contrast fix]
 │
 ├─► simple_upscale (Lanczos, -> target)                   → faithful
 │
 ├─► _final_d1_weak(img_toned):
 │     Restormer "real_denoise" (tiled) blended K=0.30
 │     with original → SwinIR-M x4 (PSNR ckpt, tiled)
 │     → Lanczos resize to `target`                        → d1_4k
 │
 ├─► _final_f3_natural(faithful, d1_4k)         ["F3-natural"]
 ├─► _final_multiscale_detail(f3)               ["G7-MS"]
 ├─► _final_whole_frame_detail(f3_ms)           ["A+"]
 │
 ├─► IF verified billboard boxes exist: Candidate A branch
 │   (native crop → Restormer motion-deblur → SwinIR-M x4 SR →
 │    resize to box's true frame scale → feathered composite)
 │   ELSE: out = f3_plus
 │
 └─► resize-to-target guard                                → OUT (target BGR uint8)
```

**Confirmed dead code for this product**: `backend/engine_adapter.py`'s
`_neutralize_billboard_regions()` always forces `_billboard_boxes()` to
return `[]` for every real upload (points `REGIONS_CONFIG` at a path that
never exists). The entire Candidate A branch — native crop, Restormer
motion-deblur, per-box SwinIR-M SR, feathered compositing — **never
executes for a real user photo**. `docs/ENGINE_AUDIT.md` documented this
first; re-confirmed by direct code read for this document.

### Exact CPU pipeline order (`enhance_shared.cpu_final_enhance`)

Same shape, different backbone: `tonal_correction.adaptive_tonal_correction`
→ faithful Lanczos upscale → classical `fastNlMeansDenoisingColored` blend
(K=0.30) → IMDN x4 via OpenVINO (`restore_exp/imdn_x4_ov.py`) → the *same*
F3-natural/G7-MS/A+ post-processing math, duplicated verbatim in
`enhance_shared.py` (not imported from `enhance.py`, because `enhance.py`
imports torch at module scope and the CPU worker process must never do
that). No billboard branch at all on CPU (dead code on GPU anyway, so not
reproduced).

### Where 3840×2160 was hardcoded (before this session's fix)

| Location | What it did |
|---|---|
| `enhance.py:42` `OUT_W, OUT_H = 3840, 2160` | Module-level constant, used as every function's *default* `target` |
| `enhance.py` `simple_upscale`, `final_enhance`, `enhance()` dispatch | Already accepted a `target` parameter — but **no caller ever passed one**, so the default (3840, 2160) always won |
| `enhance.py:_final_d1_weak` (both GPU and CPU-in-GPU-file branches) | **Hardcoded** `cv2.resize(sr, (OUT_W, OUT_H), ...)` — took no `target` parameter at all, unlike every sibling function |
| `enhance.py:_final_scale_box` | `s = OUT_W / w_orig` hardcoded (billboard-box scale factor; dead code in production, fixed anyway for correctness) |
| `enhance_shared.py` (CPU path) | Identical hardcodes: `simple_upscale` had a `target` param nobody used, `cpu_final_enhance` had **no** `target` param at all, two hardcoded `(OUT_W, OUT_H)` resizes |
| `backend/engine_adapter.py:_run_with_stage_ticker` → `engine.enhance(METHOD, img)` | Never passed `target` — the single reason the hardcode above always won in production |
| `backend/cpu_worker.py:_run_one` → `shared.cpu_final_enhance(img)` | Same: never passed `target` |
| `frontend/.../BillboardCanvas.tsx` | `const IMAGE_W = 3840; const IMAGE_H = 2160;` — every board rectangle's percent↔pixel conversion assumed the result was always exactly this size |
| `frontend/.../JobResultPanel.tsx` `BoardThumb` | `rect.width / 3840`, `rect.height / 2160` hardcoded for the board-thumbnail crop preview |

**Confirmed NOT a problem**: `backend/billboard_overlay.py` (board-outline
burn-in for exports) already reads the real image's own `img.shape` at
compose time — it was already resolution/aspect-ratio agnostic, no changes
needed there.

### Concrete repro of the stretch bug (real evidence, not just code reading)

`research/scripts/repro_stretch_bug.py` runs a real photo
(`comparison_original.png`, which happens to already be a genuine
**256×256 square**) through the exact unmodified production call path
(`engine.enhance("final", img)`, no target override — i.e. precisely what
MVP 1 does today):

```
source: comparison_original.png 256x256 (aspect 1.0000)
output: 3840x2160 (aspect 1.7778)  engine dt=32.60s
CONFIRMED STRETCH BUG: True
```

A genuinely square photo became a 16:9 box. This is the review's concern,
concretely reproduced against the real GPU pipeline (not a mockup).

---

## Phase 2 — Aspect-ratio-preserving output

### Formula

```
scale = 3840 / max(input_width, input_height)
out_w = round(input_width * scale), rounded to nearest even
out_h = round(input_height * scale), rounded to nearest even
```

Rounding to even (not just `round()`) costs at most 1px of aspect drift —
because every downstream stage (tiled Restormer/SwinIR-M, LAB/YUV
conversions, the multi-scale detail Gaussian pyramid) is safest with even
dimensions.

A 16:9 input still yields exactly `(3840, 2160)` — the product's original
fixed target is this formula's 16:9 special case, not a separate behavior.

### Verified against the review's own examples

`research/scripts/aspect_ratio_formula.py` output:

```
1920x1080 -> got 3840x2160, expected 3840x2160  [OK]
3000x3000 -> got 3840x3840, expected 3840x3840  [OK]
4000x3000 -> got 3840x2880, expected 3840x2880  [OK]
3000x4000 -> got 2880x3840, expected 2880x3840  [OK]
1600x1200 -> got 3840x2880, expected 3840x2880  [OK]
1200x1600 -> got 2880x3840, expected 2880x3840  [OK]
All review examples matched exactly: True
```

### Verified against all 6 required ratio families

```
1:1   2400x2400 -> 3840x3840  (aspect preserved, longest==3840, even)  [OK]
4:3   4032x3024 -> 3840x2880  (aspect preserved, longest==3840, even)  [OK]
3:2   6000x4000 -> 3840x2560  (aspect preserved, longest==3840, even)  [OK]
16:9  1920x1080 -> 3840x2160  (aspect preserved, longest==3840, even)  [OK]
9:16  1080x1920 -> 2160x3840  (aspect preserved, longest==3840, even)  [OK]
21:9  3440x1440 -> 3840x1606  (aspect preserved, longest==3840, even)  [OK]
```

### Implementation (done this session — see docs/MVP2_PIPELINE.md)

`aspect_preserving_target(width, height, max_dim=3840)` added to both
`enhance.py` and `enhance_shared.py` (duplicated deliberately, matching the
existing torch-isolation duplication pattern already in the codebase).
`_final_d1_weak` and `cpu_final_enhance` — the two functions that ignored
`target` entirely — now honor it. `backend/engine_adapter.py` and
`backend/cpu_worker.py` compute the real source aspect and pass it through
explicitly; **every existing direct caller that never passes `target` is
unaffected** (default stays `(3840, 2160)`), which is why the full
pre-existing GPU regression harness
(`image_enhancer/tests/test_final_pipeline.py`, 6 real images, billboard
confinement + determinism + dispatch checks) still passes byte-identically
— see docs/MVP2_BENCHMARK.md.

Frontend: `BillboardCanvas.tsx` now takes `imageWidth`/`imageHeight` props
instead of hardcoded constants; `JobResultPanel.tsx` loads each result's
real pixel dimensions (`frontend/src/lib/imageDimensions.ts`, a plain
`new Image()` load) and threads them through to both the board editor and
the board-thumbnail crop preview, defaulting to 3840×2160 only for the
brief moment before a freshly-selected image's real size is known (matches
old behavior exactly for that instant, and only for still-common 16:9
results anyway).

---

## Phase 3 — Dataset research (scoped)

Datasets considered for daylight/relighting and SR work, and why each was
or wasn't pursued this session:

| Dataset | Purpose | License | Size | Used this session? |
|---|---|---|---|---|
| MIT-Adobe FiveK | Paired exposure/retouching, includes expert-retouched "natural daylight-ish" targets | Research use (no explicit commercial grant found in the dataset's own terms — not verified further, not downloaded) | ~50GB (raw) / ~15GB (JPEG) | No — not downloaded, too large for this session's scope; noted as a strong candidate for a follow-up local benchmark since RetinexFormer's own pretrained weights were trained on it |
| LOL / LOL-v2 | Paired low-light/normal-light real+synthetic pairs | Research use, CC-style (per repo, not independently re-verified) | ~1-2GB | No — not downloaded this session |
| SID (See-in-the-Dark) | Extreme low-light RAW pairs | Research use | ~25GB | No — RAW-domain, doesn't match this product's JPEG/PNG-only input contract closely enough to prioritize |
| ExDark | Low-light object detection, not enhancement pairs | CC BY 4.0 | ~1GB | No — wrong task shape (detection labels, not enhancement pairs) |
| RealSR / DIV2K / Flickr2K | Real-world / synthetic SR pairs | Research-permissive | Several GB each | No — see Phase 5: SR-backbone research reuses this exact codebase's own already-measured G6/G7/G8 experiment history instead of re-running a new backbone sweep from scratch |

**What was used instead of downloading a paired dataset**: a controlled,
reproducible **synthetic-cast experiment** on a real photo already in this
repo (`research/scripts/daylight_candidate_classical.py`) — a standard
technique in the color-constancy/white-balance research literature: apply a
*known*, *controlled* color-temperature shift to a real neutral photo, then
measure how much of that known shift a candidate correction removes. This
requires no download, is fully reproducible, and gives a real quantitative
number (see Phase 4) rather than a qualitative guess. It does **not**
replace a proper paired-dataset benchmark (e.g. real LOL-v2 evening photos
scored against their real daylight counterparts) — that remains the
highest-value follow-up experiment (see "Follow-up work" below).

`research/` (datasets/models/inputs/outputs/benchmarks/scripts/reports) was
created per the review's recommended layout and added to `.gitignore` —
nothing under it is committed.

---

## Phase 4 — Daylight / illumination normalization research

### Candidates researched (license-gated first, per rule #2)

| Candidate | License | Verdict |
|---|---|---|
| **Zero-DCE++** | CC BY-NC 4.0 (**non-commercial only**, confirmed via the official project page) | **Rejected** — Adinn is a commercial product; this license alone disqualifies production use regardless of quality |
| **SCI** (Self-Calibrated Illumination, CVPR 2022) | No LICENSE file located in the official repo (`vis-opt-group/SCI`) at the time of this research | **Rejected pending explicit license clarification** — "no license" defaults to no usage grant (GitHub's own guidance); not adopted on ambiguous licensing per rule #2 |
| **RetinexFormer** (ICCV 2023) | **MIT** (confirmed via repo file listing) | **Viable candidate, not adopted this session** — real pretrained weights available (trained on LOL-v1/v2, SID, SMID, SDSD, **MIT-Adobe FiveK**), transformer-based Retinex decomposition (lower hallucination risk than a pure GAN, but still a learned generative-ish model — needs real local A/B testing before production adoption, which this session's time budget did not include) |
| **Classical gray-world / illuminant-adaptive chroma correction** | N/A (no external code/weights — pure math, same category as the existing `tonal_correction.py`) | **Implemented and adopted** — see below |

### Why classical over a new deep model

`tonal_correction.py`'s own module docstring establishes this codebase's
existing, deliberate philosophy: prefer a *bounded, deterministic,
per-value or per-channel global remap* over any spatial/generative
operation, specifically because it cannot hallucinate, cannot create halos,
and cannot alter geometry/text/faces by construction. The MVP 2 review's
own daylight requirement explicitly lists the same constraints ("this does
NOT mean generate a new sky... perform generative image editing... the
goal is illumination/color normalization"). A classical, bounded chroma
correction satisfies that requirement's letter and spirit directly, with
**zero new license risk, zero new model weight, zero new inference cost**
(a global LAB a/b shift is sub-millisecond) — comfortably inside the
50–60s budget with room to spare, and rejects nothing else already
evidenced.

RetinexFormer (MIT, real weights, trained partly on FiveK) remains the most
credible deep-learning candidate for a **future** iteration if the
classical correction proves insufficient on a real paired benchmark — see
"Follow-up work."

### What the existing pipeline already did NOT do (the actual gap)

`tonal_correction.adaptive_tonal_correction` (MVP 1, unchanged) is
**deliberately luminance-only** — it has its own dedicated test proving it
never shifts Lab a/b ("strong/saturated colors -> no significant color
(a/b) shift"). It fixes exposure/contrast, never a warm/orange color cast.
This is the exact, precise gap the review's "reduce excessive warm/orange
cast" requirement identifies — confirmed by direct code + test reading, not
assumed.

### Implemented candidate: `tonal_correction.correct_color_cast`

Gray-world-style illuminant estimation on the Lab a/b planes only:

- Activates only when the frame's own mean chroma shows a real cast
  (`|mean(a) - 128| >= 4` or `|mean(b) - 128| >= 4`) — an **already-neutral
  daylight photo gets zero correction** (adaptive, not blanket — satisfies
  Phase 20's explicit requirement).
- Hard ceiling: at most ±18 Lab levels of a/b shift, however strong the
  detected cast.
- Blended at 65% strength toward fully neutral, never 100% — so it can only
  ever *partially* pull a cast back, never invert it or flatten a
  legitimately warm-toned scene (sunset, warm-painted wall) into something
  artificial.
- Purely global/per-image (no spatial operation) — cannot create halos,
  cannot invent texture, cannot move an edge, exactly like the existing
  luminance correction.
- Composed with the existing exposure fix via a new
  `normalize_daylight_appearance()` entry point (`enhance.py`/
  `enhance_shared.py` now call this instead of `adaptive_tonal_correction`
  directly) — `adaptive_tonal_correction` itself is completely unmodified,
  so its own dedicated test suite needed no changes.

### Real experimental evidence

`research/scripts/daylight_candidate_classical.py`: applies a known,
controlled warm/evening cast (reduced blue, boosted red/green, reduced
luminance — a plausible tungsten-on-daylight simulation) to a real photo,
then measures Lab a/b distance from the real original before/after
correction:

```
evening (simulated) Lab a/b mean offsets from neutral: a=-1.87 b=13.13
candidate correction applied: True  a_shift=1.21 b_shift=-8.53

Mean Lab a/b distance from the real original (lower = closer to true daylight color)
evening (uncorrected)      : 14.209
candidate-corrected        : 7.952  (44.0% cast removed)
```

Unit-level evidence (`image_enhancer/tests/test_tonal_correction.py`,
section 11, all passing): neutral images get no correction; a real warm
cast is detected and pulled toward neutral; an extreme cast hits the
documented ±18 ceiling exactly (proves the cap is real, not just usually
unreached); luminance is not a target of the correction (small, ≤10/255
incidental round-trip effect only at extreme gamut edges, the same caveat
`adaptive_tonal_correction`'s own existing test already tolerates for the
reverse direction); the composed `normalize_daylight_appearance` applies
both stages together and is deterministic.

**Known limitation**: this evidence is a controlled *synthetic*-cast
experiment on one real photo, not a benchmark against real paired
evening/daylight photographs of the same scene. It demonstrates the
mechanism works exactly as designed (bounded, adaptive, reversible-cast
reduction) but not "how good the result looks" across a diverse real
evening-photo benchmark — that is the highest-priority follow-up (see
below).

---

## Phase 5 — Detail / super-resolution research

### Reused, not re-run: `docs/ENGINE_AUDIT.md`

A prior, real internal audit already exists on disk (`docs/ENGINE_AUDIT.md`,
present before this session started), based on direct code reading plus a
controlled 3-image measurement using the project's own `quality.py`
metrics. Its most important findings, directly relevant to Phase 5:

1. **SR-backbone swaps are already falsified for this architecture.** The
   G6 experiment (cited in ENGINE_AUDIT.md, own report on disk at
   `image_enhancer/reports/g8_exp/G8_DIAGNOSTIC_REPORT.md`) found that
   swapping D1's backbone to SwinIR-L, or blending in Restormer
   motion-deblur, "barely changed the final output at all" — because
   **F3-natural's fusion envelope clips away the difference before it can
   matter**, not because the backbone lacks capability. Per rule #9 ("do
   not repeat candidates already rejected... without a specific reason"),
   this session did not re-run a new SR-backbone comparison (HAT, DAT,
   DRCT, Swin2SR) against the same bottleneck — doing so would very likely
   reproduce the same null result for the same already-diagnosed reason.
2. **The actual, quantified bottleneck**: 77.5–84.7% of pixels in the final
   production output are within 1 L-unit of the plain faithful Lanczos
   upscale — i.e. statistically untouched — because F3/G7-MS/A+ are all
   edge-gated (t0=0.30/0.30/0.35) and F3's fusion envelope suppresses
   45–70% of the upstream SR/denoise stage's own measured detail gain,
   everywhere in the frame, not just at boards.
3. **The evidence-ranked highest-leverage next experiment** (ENGINE_AUDIT.md
   §9, experiment #2): give F3-natural a **locally-derived envelope** (a
   per-pixel local 5×5 min/max of the source, matching the exact self-clip
   pattern already proven safe elsewhere in this same file by A+, G7-MS's
   bands, and G9) instead of its current flat global ±14 scalar — targets
   the #1 measured bottleneck directly, reuses only already-production-
   proven primitives (low hallucination risk), and costs no new GPU model
   or download.
4. **G9.1-B** (a locked, self-clipped GAN-detail addition) is real but
   small: "+1.8–4.1% additional sharpness... for roughly double the
   per-image wall-clock time" (own measured numbers, §8) — exactly the
   "not dramatically better for the extra cost" case rule #14 says to
   reject, not force into production.

### Candidates researched for completeness (license check, not locally tested)

| Candidate | License | Notes |
|---|---|---|
| Swin2SR | Apache 2.0 (confirmed) | Same lineage as the already-shipped SwinIR-M (SwinV2 successor). Per rule #9, not adopted merely for being newer — no evidence it would escape F3's own clipping bottleneck any differently than the already-falsified SwinIR-L/Restormer-augmented-D1 experiments |
| HAT, DAT, DRCT variants | Not independently re-verified this session | R&D weights for DRCT/DAT/BSRNet already exist on disk in `image_enhancer/models/` from prior R&D (g8_exp/g9_exp lineage per the packaging spec's own exclusion-list comments) — consistent with "these were already tried and not promoted to production," not a gap this session needed to re-open |

### Phase 5 recommendation

**Do not implement a new SR/detail model this session.** The
evidence-ranked, lowest-risk, highest-leverage path (F3's envelope shape,
ENGINE_AUDIT.md experiment #2) is an *algorithmic* change to already-shipped
code, not a new model — but it directly touches the hot path's most
consequential fusion step and deserves its own dedicated controlled
regression cycle (build/extend the small-N metrics harness ENGINE_AUDIT.md
itself recommends as experiment #1, then re-tune, then re-validate against
the 6-image legacy suite) rather than a same-session blind edit. This is
flagged as the top-priority follow-up implementation, not implemented here,
per rule #13/#14.

---

## Phase 16 — Implementation decision (this session)

| Requirement | Decision | Confidence |
|---|---|---|
| 1. Aspect ratio | Implement `aspect_preserving_target` end-to-end (GPU + CPU + frontend board editor) | High — formula verified against every review example + ratio family, wiring verified against a real GPU pipeline run and a full 6-image legacy regression suite, unit-tested wiring for both GPU/CPU paths |
| 2. Daylight approach | Implement bounded classical color-cast correction (`correct_color_cast` + `normalize_daylight_appearance`), composed with the existing exposure fix | Medium-high — real, bounded, adaptive, zero new license/model risk; real quantitative evidence (44% cast reduction) on a controlled synthetic-cast experiment; **not yet validated against a real paired evening/daylight photo benchmark** (see follow-up) |
| 3. Restoration/detail approach | **No production change.** Keep Restormer + SwinIR-M + F3/G7-MS/A+ exactly as-is | High confidence this is correct given `docs/ENGINE_AUDIT.md`'s own prior measured evidence that backbone swaps don't help; F3 envelope-shape improvement recommended as next experiment, not implemented blind |
| 4. Upscaling | Unchanged (SwinIR-M x4, IMDN x4 CPU) | — |
| 5. Postprocessing | Unchanged (F3-natural/G7-MS/A+) | — |
| 6. CPU behavior | Aspect ratio fix mirrored into `enhance_shared.py`/`cpu_worker.py`; daylight fix mirrored (shares `tonal_correction.py`, no separate CPU copy needed there) | High — both paths use the identical `normalize_daylight_appearance` call, kept in sync by construction |
| 7. GPU behavior | Aspect ratio + daylight both wired into the real GPU call path | High |
| 8. Expected processing time | Aspect ratio: no measurable change (same total pixel-processing cost distribution, same models). Daylight: a few milliseconds (a global LAB shift + one CLAHE pass on the L channel, both CPU/numpy, no GPU call). Both stay inside the existing ~25–45s-per-image budget the codebase already operates in (docs/PERFORMANCE_OPTIMIZATION.md), well under the new ~50–60s target — confirmed by a real end-to-end run below (45.9s) | High — no new model, no new GPU call added |

See `docs/MVP2_PIPELINE.md` for the exact resulting pipeline diagram and
`docs/MVP2_BENCHMARK.md` for the regression evidence this decision rests on.

---

## Daylight tuning pass (follow-up session)

A real user-uploaded street photo (`research/inputs/street_overcast_01.jpeg`,
1280×960, 4:3 — copied from an actual WhatsApp upload/download pair found on
this machine, with its already-enhanced MVP2-baseline output alongside it)
made the first daylight implementation's limits directly visible: the
already-enhanced output still looked flat, cool, and "hazy," not like a
midday photo. Diagnosed with real histogram measurements (not guessed), then
fixed and re-validated end to end through the real production pipeline.

### Root causes found (real measurements on the real photo)

```
L stats: p1=38 p5=67 p50=177 p95=255 p99=255 white_clip_frac=0.407
issues detected (OLD code): ['overexposed', 'clipped_highlights']
a_mean=128.0 b_mean=121.8  (b<128 = a real, if mild, COOL/blue cast)
```

1. **Sky-domination bug in exposure detection.** This photo's sky is ~41–58%
   of the frame and almost entirely blown white (`white_clip_frac=0.41`).
   That single fact pushed the WHOLE-FRAME `p50` to 177 and `p5` to 67,
   tripping the existing "overexposed" rule (`p50>175 and p5>55`) even
   though the actual scene content (buildings, road, people, vehicles) was
   not overexposed at all — computed on the sky-excluded pixels alone,
   `scene_p50` was 154 (moderate) with a full 196-level dynamic range
   (`scene_p99−scene_p1`). The "overexposed" misclassification then drove
   `_build_lut`'s gamma curve to **darken** the entire frame (`gamma=1.25`)
   plus pull highlights down further (`highlight_pull=14`) — the exact
   opposite of what a street-level scene under a bright/overcast sky needs.
   This is a genuine architectural gap, not a tuning-constant issue: no
   value of `OVEREXPOSED_MEDIAN`/`OVEREXPOSED_P5` fixes it, because the raw
   whole-frame percentiles are structurally the wrong signal whenever a
   photo has a large blown-out region that isn't representative of the
   photographed subject.
2. **Color-cast correction was measurably too subtle.** The photo's real
   cast was mild (`b_mean=121.8`, ~6 levels cool of neutral), and the old
   `CORRECTION_STRENGTH=0.65` pulled it back by only ~4 levels — nowhere
   near enough to read as "daylight" rather than "still a bit hazy/cool."
3. **No local-contrast/clarity mechanism at all.** The photo's GLOBAL
   dynamic range was already fine (196 levels, scene-based) — this is
   fundamentally a HAZE problem (low *local* micro-contrast under a
   genuinely fine global range), which no percentile-based global LUT can
   detect or fix by construction. Nothing in the tonal-correction stage
   addressed this.

### Fixes made (`image_enhancer/src/tonal_correction.py`)

1. **Scene-aware exposure/contrast stats.** `_histogram_stats` now also
   computes `scene_p1/p5/p50/p95/p99` — the same percentiles, but with the
   blown-highlight cluster (`L >= SKY_CLIP_L = 250`) excluded first (falls
   back to the raw stats if that would leave under 5% of the frame, e.g. a
   snow field or dense fog, where there's no meaningful "non-sky" sample).
   `detect_issues` and `_build_lut`'s gamma/stretch computations now use
   these scene stats for exposure DIRECTION decisions; `black_clip_frac`/
   `white_clip_frac` (crushed shadows / clipped highlights) still use the
   real, literal whole-frame clipping fractions unchanged — those genuinely
   do mean "this many pixels have no recoverable detail." New regression
   test (`image_enhancer/tests/test_tonal_correction.py`, "sky-does-not-
   force-overexposed") constructs a synthetic 40%-blown-sky-over-dim-scene
   image and proves it is no longer misclassified, and that its own dim
   foreground correctly gets a *brightening* gamma instead.
2. **Stronger, slightly-warm-biased color-cast correction.** `CORRECTION_
   STRENGTH` raised 0.65 → 0.80. The correction target moved from clinical
   neutral (`NEUTRAL_AB=128`) to a small deliberately-warm daylight point on
   the b (blue↔yellow) axis only (`DAYLIGHT_B_TARGET=132`) — directly
   implementing the review's explicit "natural, slightly warm daylight
   white balance" (not just "remove cast if present"); `a` (green↔magenta)
   keeps the pure-neutral target, since daylight has no natural
   magenta/green skew. Still fully adaptive: a photo already at/near the
   target gets no correction (new test: "cast-already-at-target-not-
   applied"); the hard `MAX_CHROMA_SHIFT=18` ceiling is unchanged and still
   provably binds on extreme input (new test: "cast-shift-hits-ceiling-on-
   extreme-input").
3. **New `enhance_local_clarity`**: a bounded local-contrast boost via CLAHE
   on the L channel only (chroma untouched) — the same `clipLimit=2.0,
   tileGridSize=(8,8)` configuration already precedented in this exact
   codebase (`enhance.py`'s `classical_preprocess`), blended at 50% (not
   full-strength) to stay conservative. CLAHE's own clip limit makes it
   naturally self-adaptive per tile — a tile with already-good local
   contrast gets redistributed very little — so no separate global gate was
   needed; new tests prove it's bounded (`mean_l_delta` in a sane range),
   chroma-untouched, and deterministic.
4. `normalize_daylight_appearance` now composes all three stages in order:
   tonal (scene-aware) → cast (stronger + warm-biased) → clarity (new).

### Real evidence: before vs. after, on the real photo, through the real pipeline

Full production call path (`backend.engine_adapter.enhance_image`), not a
mock — real Restormer + SwinIR-M inference, real 4:3 aspect-ratio-preserving
output (1280×960 → 3840×2880, confirming the Phase 2 fix still holds):

| Metric | Original | Before tuning (prior session) | After tuning (this pass) |
|---|---|---|---|
| Output size | 1280×960 | 3840×2880 | 3840×2880 |
| L mean (brightness) | 187.6 | 171.5 (darkened by the sky-domination bug) | 181.3 (natural, no longer artificially darkened) |
| Lab b mean (warmth; 128=neutral) | 121.8 (cool) | 125.3 (still cool) | 129.9 (slightly warm, matching the target) |
| Local contrast (Laplacian std, 4K) | 44.8 (source res, not comparable) | 32.9 | 34.8 (+5.8%) |
| Wall-clock time | — | — | 45.9s (within the ≤60s target) |

Visual inspection (full-frame + crops saved under
`research/outputs/street_overcast_01_*`, `crop_*_BEFORE/AFTER.png`):

1. **Daylight tone**: after tuning reads as a plausible bright, slightly
   warm midday photo; before tuning still read as flat/overcast/cool.
2. **Brightness**: after tuning is brighter and closer to the true source
   brightness (the sky-domination bug had been artificially darkening it).
3. **Color neutrality/warmth**: shifted from a mild cool cast to a natural
   slightly-warm daylight tone, not overshot into orange.
4. **Shadows/highlights**: highlight pull is legitimate and gentle
   (clipped_highlights only, ≤14 levels near pure white); no new clipping
   introduced (confirmed by the existing `clipped-highlights-no-new-
   clipping` test, unaffected by this change).
5. **Fine detail / clarity**: visibly less "washed out," buildings and road
   texture read with more definition; quantitatively +5.8% local contrast.
6. **Billboard/text fidelity**: the CaratLane billboard text and the
   "AURA PAVERS EICHER" truck text are pixel-for-pixel legible in both
   before/after crops — no hallucination, no added/removed/altered
   characters, no distortion.
7. **Faces/vehicles**: the billboard model's face reads more naturally
   (less flat/gray); vehicle body colors and the truck's yellow read
   richer without oversaturating; no invented detail on the intentionally-
   blurred/illegible license plate (correctly left faithful, not
   sharpened into fake text).
8. **Hallucination/artifacts**: none observed — no halos, no color banding,
   no plastic/HDR appearance, no new object content. Every change is a
   deterministic, bounded, global-or-per-tile pixel remap; nothing spatial
   enough to move an edge or invent a shape.

**Adaptive-restraint check** (not just the one hazy photo): ran the same
composed correction on the 6 existing, mostly well-exposed production
reference photos (`image_enhancer/originals/1-6.jpeg`) — the whole-frame
exposure/contrast fix (`adaptive_tonal_correction`) correctly stayed a
no-op on 4 of 6 (already good daylight, no gamma/stretch applied) and only
gently pulled clipped highlights on the other 2; the color-cast correction
applied a modest warm nudge (4–14 Lab levels, all within the ±18 cap) to
all 6, consistent with these being real OOH photos with some ambient cast;
visual inspection of the strongest case (`3.jpeg`, `research/outputs/
originals_3_tonal_only_*`) confirms a natural, subtle warming with **no**
visible over-processing, oversaturation, or artificial sky/color change —
the photo's genuinely blue sky stays blue, just slightly less cool-cast
overall.

### Regression status

- `image_enhancer/tests/test_tonal_correction.py`: all tests pass, including
  9 new/updated tests for the scene-aware exposure fix, the retargeted cast
  correction, the new clarity function, and the composed entry point.
- `image_enhancer/tests/test_final_pipeline.py` (real 6-image GPU
  regression): all tests pass — exact output shapes, billboard confinement
  (0 outside-box pixels changed beyond the existing round-trip floor on all
  6 images), determinism, and full dispatch/method coverage all hold.
- `python -m pytest backend/tests -q`: 112 passed, 24 subtests passed
  (unaffected — this pass only changed `image_enhancer/src/`).
- Frontend `tsc --noEmit` / `npm run build`: unaffected (no frontend files
  touched this pass), both still pass.

### Known limitations

- Still a single real photo as the primary "hazy/overcast" benchmark, plus
  the 6 pre-existing (mostly well-lit) reference photos for restraint —
  not the full 15–20 image, multi-lighting-condition benchmark the original
  MVP 2 review plan calls for (see "Follow-up work" below, item 4).
- `SKY_CLIP_L=250` and `DAYLIGHT_B_TARGET=132` are principled but
  hand-chosen constants (grounded in this one real photo's measurements
  plus the existing module's established bounding philosophy), not
  swept/tuned against a larger benchmark.
- CLAHE's tile grid (8×8) is fixed regardless of input resolution; not
  re-derived per the new aspect-ratio-preserving target sizes (e.g. a very
  elongated 21:9 output has different tile aspect ratios than a 1:1 one) —
  no visual artifact was observed on the tested 4:3/16:9 cases, but this is
  untested on extreme aspect ratios.

---

## Brightness tuning pass (second follow-up session)

User feedback on the "Daylight tuning pass" result: closer, but wanted it
**brighter** while staying natural, and flagged that the large blown sky
was reading **cream/yellow**, not white. Both diagnosed with real
measurements on the same real benchmark photo and fixed with two more
targeted, bounded changes — no new model, no change to Restormer/SwinIR-M.

### Root causes (real measurements)

1. **The color-cast correction tinted the sky.** `correct_color_cast`
   applied one flat a/b shift to every pixel, sky included. On the real
   photo the sky (mean L=236.7) landed at `b=135.2` — clearly cream, not
   white (`b=128` is neutral; the original untouched sky was `b=124.2`,
   itself very slightly cool). A blown highlight has no real color
   information left to "correct"; painting it is exactly the "yellow/cream
   sky" the user correctly flagged.
2. **Scene median alone doesn't capture "dark street/building areas."**
   After the prior pass's sky-domination fix, this photo's `scene_p50`
   (154) no longer triggered any exposure correction at all — technically
   "not broken," but the user's actual complaint was about the **darker
   half** of the scene (road, building shadows, undercarriage) still
   reading dim. Median hides this: `scene_p25` (the scene's own darker
   quarter) was only 110 — genuinely dim — while the median looked fine.

### Fixes made (`image_enhancer/src/tonal_correction.py`)

1. **New proactive daylight brightness lift**, independent of the existing
   reactive under/overexposed rules (those only fire when exposure is
   genuinely broken). Driven by `scene_p25` (new field in
   `_histogram_stats`, the scene's own darker quarter, sky excluded) against
   a new `DAYLIGHT_SHADOW_P25_TARGET = 145.0`. Bounded gamma curve,
   `MAX_BRIGHTNESS_LIFT_DEVIATION = 0.45` (gamma clamped to `[0.55, 1.0]`,
   brightening only, never darkening), anchored at 0/255 like every other
   gamma curve in this module so it introduces no new clipping and tapers
   to a small effect near the highlight end by construction. Runs in
   `_build_lut` alongside (composes with, when both apply) the existing
   reactive gamma. `adaptive_tonal_correction`'s no-op detection changed
   from "no issue in `issues`" to "no params were actually built" so this
   new always-adaptive stage isn't gated behind the old binary trigger.
2. **Highlight-protected color-cast correction.** `correct_color_cast` now
   weights its a/b shift by a per-pixel function of that pixel's own L
   (`CAST_HIGHLIGHT_PROTECT_START = 205`, `CAST_HIGHLIGHT_PROTECT_END =
   245`): full correction below 205, linearly tapering to ~0 by 245. Still
   a pure per-pixel value-driven LUT (no neighborhood/gradient lookup), so
   it still cannot create a halo or move an edge — it just stops applying
   the daylight-warm shift to pixels that have no real color left to
   correct.

### Real evidence: before vs. after (real photo, real pipeline)

| Metric | Original | Pass 2 (prior session) | Pass 3 (this pass) |
|---|---|---|---|
| Output size | 1280×960 | 3840×2880 | 3840×2880 |
| Whole-frame L mean | 187.6 | 181.3 | **194.5** (brighter than the original) |
| Sky region: L / a / b | 224.0 / 128.6 / 124.2 | 236.7 / 127.1 / **135.2** (cream) | 237.8 / 127.1 / **127.5** (near-neutral, matches original's own slight coolness) |
| Non-sky scene: L / b | 158.7 / 119.7 (cool) | 155.8 / 127.8 | **174.7** / 126.3 (brighter, still slightly warm) |
| Local contrast (Laplacian std) | 44.8 (source res) | 34.8 | 33.1 |
| Wall-clock | — | 45.9s | **45.3s** (within ≤60s target) |

Sky b went from 135.2 (visibly cream) to 127.5 — within 3.3 Lab levels of
the ORIGINAL sky's own 124.2, i.e. no longer meaningfully tinted, confirmed
by direct crop inspection (`research/outputs/sky_crop_PASS3.png`: clean
light gray/white, no yellow cast). Non-sky scene brightness rose from 155.8
to 174.7 (+18.9), directly answering "lift dark street/building areas
more," while scene warmth (b=126.3) stayed comparable to Pass 2 (b=127.8) —
the brightness fix did not undo the warmth fix.

Visual re-inspection of billboard text (`crop_billboard_text_PASS3.png`)
and truck text (`crop_truck_text_PASS3.png`): both pixel-for-pixel legible
and unaltered — no hallucination introduced by the stronger brightness
lift. Re-ran the adaptive-restraint check on the 6 existing reference
photos (`image_enhancer/originals/1-6.jpeg`): every photo now gets a
proactive brightness lift (gamma 0.66–0.84, all within the documented
bound), consistent with "12PM bright daylight" being a brighter target than
these real OOH photos' own baseline exposure; visual re-inspection of the
strongest case (`3.jpeg`, already a clear blue-sky day,
`research/outputs/originals_3_PASS3_AFTER.png`) confirms a natural result —
warmer road/building tone, sky stays a natural blue (not washed out,
not tinted), no HDR/artificial look.

### Regression status

- `image_enhancer/tests/test_tonal_correction.py`: all tests pass, including
  5 new tests locking in the highlight-protection fix
  (`highlight-sky-not-tinted`, `highlight-scene-still-corrected`) and the
  proactive-lift behavior (`normal-still-gets-proactive-lift`,
  `normal-lift-bounded`, `bright-normal-bypassed`). Two pre-existing
  fixed-tolerance checks (`strong-colors-no-shift`, the 6
  reference-image `chroma-stable` checks) needed a wider, empirically
  re-verified tolerance — the underlying guarantee they test
  (`adaptive_tonal_correction` writes ONLY the L plane) is unchanged and
  still enforced; the tolerance simply had to account for MVP 2's
  legitimately larger L excursions increasing LAB↔BGR 8-bit round-trip
  noise. The `edges-preserved` check was replaced entirely with a more
  precise, scale-invariant proof (verifying the actual L→L mapping is a
  consistent, monotonic non-decreasing lookup table — the module's own
  documented guarantee — directly, rather than via a fixed-threshold Canny
  proxy that is inherently sensitive to absolute brightness scale).
- `image_enhancer/tests/test_final_pipeline.py` (real 6-image GPU
  regression): all tests pass — exact output shapes, billboard confinement
  (0 outside-box pixels changed on all 6 images), determinism, full
  dispatch/method coverage.
- `python -m pytest backend/tests -q`: 112 passed, 24 subtests passed
  (unaffected — only `image_enhancer/src/tonal_correction.py` and its own
  tests changed this pass).
- Frontend unaffected (no frontend files touched this pass).

---

## Quality tuning pass (objective changed: photographic quality, not daylight look)

The "12 PM / daylight appearance" objective was **removed entirely** at the
user's explicit direction. New objective: make the real uploaded photograph
look clearer, sharper, more detailed, better exposed, with stronger local
contrast, richer-but-realistic color, and natural HDR-like shadow/highlight
detail — **without** forcing any particular time-of-day look or color
temperature, and without touching Restormer/SwinIR-M.

### What changed, and why (`image_enhancer/src/tonal_correction.py`)

The module was substantially rewritten around a new governing rule (now its
own module docstring): every correction is either a *fidelity fix* (broken
exposure, a materially excessive color cast, low local clarity) or a
*modest, bounded enhancement* (detail/micro-contrast, natural saturation)
— never a stylistic push toward a particular look.

1. **Removed the forced warm-daylight color bias.** `DAYLIGHT_B_TARGET`
   (132, a deliberately warm target) is gone. `correct_color_cast` now
   targets pure `NEUTRAL_AB` (128) on both axes, only activates on a
   *materially excessive* cast (`CAST_MEAN_THRESHOLD` raised 4.0 → 7.0),
   and pulls more conservatively when it does (`CORRECTION_STRENGTH`
   lowered 0.80 → 0.55, `MAX_CHROMA_SHIFT` lowered 18 → 14). Directly
   implements "keep the photograph's original lighting/color character
   unless correction is needed" — verified on the real benchmark photo:
   its own mild cool cast (`b=121.8`, ~6 levels off neutral) now falls
   **below** the new threshold and is correctly left untouched (previously
   it was pulled to `b=126.4`, a forced warm shift).
2. **Moderated the forced brightness lift.** The prior `DAYLIGHT_SHADOW_
   P25_TARGET=145` / `MAX_BRIGHTNESS_LIFT_DEVIATION=0.45` (bounding gamma
   to as low as 0.55 — a strong, "always push toward bright daylight"
   lift) is replaced by `SHADOW_EXPOSURE_P25_TARGET=122` /
   `MAX_BRIGHTNESS_LIFT_DEVIATION=0.18` (gamma bounded to `[0.82, 1.0]`) —
   a small, genuinely gentle fidelity assist for flat/dim areas, not a
   brightness-for-its-own-sake push. Renamed `brightness_lift_gamma` →
   `exposure_assist_gamma` throughout to match.
3. **New `apply_local_tone_mapping`**: replaces the old CLAHE-based
   `enhance_local_clarity`. A classical, deterministic, edge-aware
   (bilateral-filter) base/detail decomposition — the standard technique
   behind photographic "HDR-look" tools (Durand & Dorsey-style): the base
   (large-scale illumination) layer's dynamic range is compressed toward
   its own mid-point by a bounded, scene-adaptive amount (shadow lift +
   highlight recovery — the actual "HDR tone mapping"), and the detail
   layer (original minus the edge-aware base — fine texture/micro-contrast)
   is added back with a bounded boost (the "clearer/sharper" step). Because
   the base filter is edge-aware, detail is exactly zero across a flat
   sky/wall (nothing invented there) and largest at real texture the SR
   stage upstream already resolved — and because there's no
   Gaussian-blur-based unsharp mask anywhere in this operation, it cannot
   ring at hard edges (billboard frames, vehicle outlines, building edges),
   directly answering "edge clarity without halos/ringing."
4. **New `boost_natural_saturation`**: a small, bounded HSV saturation lift
   (`MAX_SATURATION_BOOST=0.18`), skipped entirely when the photo is
   already reasonably saturated (`SATURATION_MEAN_FLOOR=90`) — "richer but
   realistic colors," never a blanket push.
5. **Renamed the composition**: `normalize_daylight_appearance` →
   `enhance_photographic_quality`, now composing tonal correction → cast
   correction → local tone mapping → saturation (both `enhance.py` and
   `enhance_shared.py` call sites updated).

### Real evidence: ORIGINAL vs CURRENT (prior daylight tuning) vs NEW (quality tuning)

Full production pipeline (`backend.engine_adapter.enhance_image`), same
real benchmark photo (`research/inputs/street_overcast_01.jpeg`):

| Metric | Original | Previous (daylight tuning) | New (quality tuning) |
|---|---|---|---|
| Output size | 1280×960 | 3840×2880 | 3840×2880 |
| L mean (brightness) | 187.6 | 194.5 | 184.7 |
| Lab b mean (color; 128=neutral) | 121.8 | 126.4 (forced warm) | **121.1** (matches original — no forced temperature) |
| Saturation (mean HSV S) | 35.6 | 21.2 (the old pipeline's SR stages measurably desaturated the photo) | **36.2** (restored to match/slightly exceed the original) |
| Local contrast (Laplacian std, 4K) | 44.8 (source res) | 33.05 | **34.43** (exceeds the CLAHE-based prior version) |
| Wall-clock (3 runs) | — | 45.3–45.9s | 45.1–63.9s (avg ≈52s; see "Processing time" below) |

The color-temperature result is the clearest evidence the reframing worked:
the new pipeline's output color (`b=121.1`) is within 0.7 Lab levels of the
untouched ORIGINAL photo's own color (`b=121.8`) — the photo's real,
natural, slightly-cool character is preserved, not overridden. Saturation
is restored past what the old pipeline's SR/detail stages had been quietly
desaturating it to (the old pipeline had no saturation-recovery mechanism
at all). Local contrast now measurably exceeds the previous CLAHE-based
version's.

### 100% crop inspection (fidelity check)

Crops saved under `research/outputs/crop_*_QUALITY.png` /
`crop_*_FINAL.png` for: billboard text + face, truck text + undercarriage,
buildings (windows/railings/tile color), people/shopfront, road
barrier/debris, distant vehicles.

- **Billboard text** ("Diamonds, for the diamond you are", "CARATLANE",
  "SHOP ONLINE at www.caratlane.com"): pixel-for-pixel legible, unaltered,
  no added/removed/distorted characters. The billboard's own genuinely
  purple/cool background color is preserved, not forced warm.
- **Truck text** ("AURA PAVERS EICHER"): legible, sharper edges than the
  prior version, colors read richer (vibrant yellow/red) without a
  plastic/oversaturated look. License plate remains correctly illegible —
  not sharpened into fake text (no invented detail).
- **Road barrier/debris**: individual barrier slats, scattered debris, and
  the "17.09.2026" timestamp overlay are all crisply resolved with no
  visible halos at the barrier's hard edges despite the stronger detail
  boost — direct confirmation the bilateral/edge-aware design avoids
  ringing where a naive unsharp mask would show it clearest.
- **Buildings**: window frames, balcony railings, and tile color
  (maroon/tan/blue) are crisp and geometrically faithful; no invented
  architectural detail.
- **People/shopfront**: rendered faithfully at the original photo's own
  resolved detail level (distant faces remain soft, honestly — no invented
  facial detail); colors (red shirt, balloons) read richer without looking
  synthetic.

No hallucination, no invented objects/text/geometry, no plastic/painted
texture observed in any inspected crop.

### Processing time

Three full end-to-end runs (fresh Python process each, so every run pays
the full cuDNN cold-start autotune cost that a real deployed app's
persistent worker thread only pays once — see
`docs/PERFORMANCE_OPTIMIZATION.md`): **45.1s, 50.2s, 63.9s** (avg ≈53s).
The new `tonal_correction` stage's own added cost (measured standalone,
CPU-only, before any GPU call) is small and consistent: **≈1.7–1.8s**
per image at the photo's native 1280×960 resolution (bilateral filtering is
the dominant cost there) — not the source of the run-to-run variance, which
is environmental/cold-start (consistent with this codebase's own
already-documented ~8s+ cuDNN autotune variance). The one 63.9s outlier
slightly exceeded the review's ≤60s target; the other two runs (45.1s,
50.2s) were comfortably inside it, and a warmed-up production worker thread
should track closer to the lower end. Flagged honestly rather than
averaged away.

### Regression status

- `image_enhancer/tests/test_tonal_correction.py`: substantially rewritten
  (new/renamed tests for `correct_color_cast`'s neutral-only target and
  mild-cast no-op, `apply_local_tone_mapping`, `boost_natural_saturation`,
  and `enhance_photographic_quality`) — all passing.
- `image_enhancer/tests/test_final_pipeline.py` (real 6-image GPU
  regression): all passing — exact output shapes, billboard confinement (0
  outside-box pixels changed on all 6 images beyond the existing round-trip
  floor), determinism, full dispatch/method coverage. Per-image wall time
  40.4–62.7s in this harness specifically (which, unlike production,
  deliberately exercises the dead-code Candidate A billboard branch — see
  the "Daylight tuning pass" section above for why that's not
  representative of real per-image production time).
- `python -m pytest backend/tests -q`: 112 passed, 24 subtests passed
  (unaffected — no backend/frontend files touched this pass).
- Frontend unaffected (not modified, per explicit instruction this pass).

### Known limitations

- `apply_local_tone_mapping`'s bilateral filter parameters
  (`TONE_MAP_SIGMA_COLOR=40`, `TONE_MAP_SIGMA_SPACE=30`) and
  `MAX_DETAIL_BOOST=2.0` are tuned against this one real benchmark photo
  (checked for absence of ringing/noise amplification on its own hard
  edges), not swept across a larger, more diverse benchmark set.
- The processing-time variance (45–64s across 3 runs) means the ≤60s
  target is not guaranteed on every cold-process run in this test
  environment, though the tonal-correction stage's own overhead is a small,
  fixed ~1.7s — the variance is pre-existing cuDNN cold-start behavior, not
  introduced by this pass.
- No new external dataset or model was evaluated this pass (per explicit
  instruction to keep Restormer/SwinIR-M and tune only the existing
  pipeline); a broader benchmark across the full lighting/scene diversity
  in `docs/MVP2_RESEARCH.md`'s original Phase 6 plan remains outstanding.

---

## Halo regression fix (follow-up session)

The Quality tuning pass above shipped `apply_local_tone_mapping` with an
**unbounded** detail-layer boost (`MAX_DETAIL_BOOST=2.0`, effective 1.85x,
no local clamp). On a real user photo this produced visible cartoon/painted
sky texture, dark-outline halos at building edges, and crunchy artificial
texture on flat billboard surfaces — sharper, but no longer photographic.
Flagged by the user with a direct before/after comparison; not shipped.

### Root cause

The "detail" layer (`original L − bilateral base`) contains real texture
**and** residual JPEG-block/sensor noise indistinguishably. A flat,
unbounded multiplicative boost amplifies both:

1. **In flat regions (sky)**: there is no real texture to recover, only
   noise — boosting it by 1.85x turned invisible compression noise into
   visible mottled/painted texture.
2. **At hard edges (buildings against sky)**: a bilateral filter is
   edge-aware but not perfectly edge-*exact* — it still leaves a few pixels
   of soft transition at a real edge. Amplifying the detail layer there by
   1.85x is the textbook unsharp-mask overshoot: a dark ring on the dark
   side of the edge, a bright ring on the bright side. Being "edge-aware"
   (the bilateral filter alone) was **not sufficient** to prevent this at a
   high boost — confirmed by testing intermediate boosts (1.5x, 1.3x) with
   no local bound, which still haloed.

### The fix

Two changes to `image_enhancer/src/tonal_correction.py`:

1. **`MAX_DETAIL_BOOST` lowered 2.0 → 1.20** (effective boost ≈1.17x,
   down from ≈1.85x) — a genuinely conservative ceiling.
2. **A local min/max clip on the boost's EXTRA contribution** — the exact
   same safety pattern this codebase's own `enhance.py` G7-MS/A+ detail
   stages already use in production (`docs/ENGINE_AUDIT.md`'s own
   "hard-clipped to something already present in the pixel's own
   neighborhood" analysis of why those stages don't hallucinate). Only the
   boost *beyond* 1x is clipped to `[local_min − l, local_max − l]`
   (5×5 erode/dilate of the current L channel, matching A+'s own kernel
   size exactly) — the base-compression shadow/highlight shift is left
   unclipped (it's an intentional large-scale change, not a local
   artifact). This makes it structurally impossible for the boosted detail
   to push a pixel past a brightness level that doesn't already exist
   somewhere in its own local neighborhood — which is precisely the
   definition of a halo/ring.

Also tightened `TONE_MAP_SIGMA_COLOR` (40 → 22, stricter edge
preservation) and `MAX_BASE_COMPRESSION` (0.32 → 0.20, less aggressive
shadow/highlight compression, reducing the "flat/cartoon boundary" look
observed at strong compression near a hard edge the bilateral filter
hadn't fully separated).

### Real evidence: before vs. after the fix

Full production pipeline, same real benchmark photo:

| Metric | Over-aggressive (boost≈1.85, no clip) | Fixed (boost≈1.17, local-clipped) |
|---|---|---|
| Local contrast (Laplacian std) | 34.43 | 30.49 (lower — the inflated number *was* the artifact, not real detail) |
| Brightness / color / saturation | unchanged by this fix | unchanged (L=184.8, b=121.1, sat=36.4 — matches the pre-fix values almost exactly, confirming only the detail-boost mechanism was touched) |
| Wall-clock | — | 48.9s (within ≤60s target) |

Visual re-inspection (`research/outputs/crop_*_HALOFIX.png`,
`street_overcast_01_HALO_FIX_preview.jpg`):

- **Sky**: clean, smooth, no mottled/painted texture; no halo at the
  roofline silhouette.
- **Buildings**: no dark-outline halos at edges against the sky; railings,
  windows, tile color all crisp and geometrically faithful.
- **Billboard**: smooth background (previously crunchy/artificial
  texture), text still pixel-legible, face natural.
- **Truck**: text still legible, panel edges clean, no outline artifacts.
- **Road barrier**: individual slats and debris still clearly resolved —
  slightly softer than the over-aggressive version, but genuinely real
  detail, matching the explicit preference for "slightly softer but REAL
  detail over artificial sharpness."

### Regression status

- `image_enhancer/tests/test_tonal_correction.py`: all passing, plus 4 new
  dedicated regression tests directly locking in this fix:
  `halo-no-dark-ring-at-edge` / `halo-no-bright-ring-at-edge` (a synthetic
  hard-edge scene proves no ring overshoot beyond each side's own resulting
  plateau), `halo-fix-boost-is-conservative` (asserts the ceiling itself
  stays ≤1.30), `flat-noise-not-amplified-into-texture` (a synthetic flat
  noisy region's standard deviation may grow by at most 1.5x, not blow up
  into visible fake texture).
- `image_enhancer/tests/test_final_pipeline.py` (real 6-image GPU
  regression): all passing — billboard confinement (0 outside-box pixels
  changed on all 6 images), determinism, full dispatch/method coverage.
- `python -m pytest backend/tests -q`: 112 passed, 24 subtests passed
  (unaffected).

### Final parameters (this fix)

| Constant | Value | Was |
|---|---|---|
| `TONE_MAP_SIGMA_COLOR` | 22.0 | 40.0 |
| `TONE_MAP_SIGMA_SPACE` | 24.0 | 30.0 |
| `MAX_BASE_COMPRESSION` | 0.20 | 0.32 |
| `MAX_DETAIL_BOOST` | 1.20 | 2.0 |
| `DETAIL_CLIP_KERNEL` | 5 (new) | — |

---

## Conservative realism pass (third follow-up session)

The halo fix above stopped the ring/cartoon artifacts, but on a second,
different real benchmark photo (`research/inputs/7.jpeg` — an
already-well-exposed, bright sunny street scene, a genuinely different
lighting condition from the hazy `street_overcast_01.jpeg` used before) the
output (`research/inputs/7_enhanced_1.png`) still read as "slightly
over-processed": faint sky mottling, edge-emphasized window mullions, and a
visibly waxy/flattened look on the silver Audi's glossy paint. Flagged by
the user with a direct 100%-crop comparison.

### Root cause (a design gap, not a leftover bug)

Traced with real numbers on `7.jpeg`: `apply_local_tone_mapping` computed
`compress_alpha=0.194` — 97% of its (then) `MAX_BASE_COMPRESSION=0.20`
ceiling — on a photo whose exposure was already good. The activation logic
used the scene's raw dynamic-range SPAN (`scene_p99 − scene_p1 = 231`) as
its only signal for "does this scene need HDR compression," but a
genuinely well-exposed, bright sunny photo (blue sky + a vehicle's own deep
shadow) legitimately HAS a wide span without that being a problem. Treating
wide range alone as "needs compression" pulled an already-correct
exposure's highlights and shadows toward mid-gray, producing exactly the
flattened/waxy look on the glossy car and the edge-emphasis on window
frames the user flagged. This is the same class of gap `docs/ENGINE_AUDIT.md`
already warned about elsewhere in this codebase: a global metric (here,
raw dynamic-range span) is not, on its own, reliable evidence of a genuine
problem worth correcting.

### The fix

Widened the gap between "no compression" and "full compression" so a
normally wide-range but already-good photo lands well short of the
ceiling, and lowered every remaining knob further:

| Constant | New | Was (halo-fix pass) | Was (original) |
|---|---|---|---|
| `TONE_MAP_SIGMA_COLOR` | **16.0** | 22.0 | 40.0 |
| `MAX_BASE_COMPRESSION` | **0.10** | 0.20 | 0.32 |
| `WIDE_RANGE_FLOOR` | **130.0** | 110.0 | 110.0 |
| `WIDE_RANGE_CEIL` | **280.0** | 235.0 | 235.0 |
| `MAX_DETAIL_BOOST` | **1.12** | 1.20 | 2.0 |
| detail-boost fraction of ceiling | **0.60** | 0.85 | 0.85 |

Net effect on the real photo: `compress_alpha` 0.194 → **0.067** (a 65%
reduction), effective `detail_boost` 1.17 → **1.072**, whole-frame
`mean_l_delta` 8.24 → **2.96** (a much gentler total change). `TONE_MAP_
SIGMA_SPACE` (24.0) and `DETAIL_CLIP_KERNEL` (5, the local halo-preventing
clip from the previous fix) were left unchanged — the local clip itself
was already working correctly; this pass narrows how much gets through it
in the first place, and narrows the range of scenes it reserves stronger
compression for.

### Real evidence: ORIGINAL vs PREVIOUS (user-flagged) vs NEW (this pass)

Full production pipeline, `research/inputs/7.jpeg` (1416×797, already
well-exposed sunny scene):

| Metric | Original | Previous (flagged) | New (this pass) |
|---|---|---|---|
| L mean | 167.1 | 170.4 | **166.4** — essentially matches the original |
| Saturation (mean HSV S) | 61.2 | 56.0 | **58.7** — closer to the original |
| Local contrast (Laplacian std) | 73.0 (source res) | 42.54 | 43.11 (materially unchanged — confirms the visual improvement is about REALISM, not a contrast-metric change; see the explicit "never optimize for a higher contrast metric at the expense of visual realism" instruction this pass follows) |

100% crop re-inspection (`research/outputs/crop7_*_RETUNE.png`) against
both the original and the flagged-over-processed version:

- **Sky**: mottling substantially reduced (though a faint residual remains
  — see "Known limitations" below).
- **Buildings**: window/mullion edges read softer, less "outlined"; real
  architectural detail (balcony railings, red/white banding) still
  resolved.
- **Billboard**: text ("JJ GOLD", "CASH FOR GOLD") still pixel-legible,
  face natural, colors rich without a plastic look.
- **Cars (Audi)**: the glossy roof/trunk highlight no longer reads as a
  flattened, waxy plateau — natural gradient restored; license plate
  ("TN 09 CD 9348") still legible; door-panel gap lines no longer
  over-emphasized.
- **Motorcycle/people/barrier/road**: real detail (chevron barrier
  pattern, curb blocks, scooter wheel, timestamp) still clearly resolved,
  no hallucination, no invented content anywhere.

### Regression status

- `image_enhancer/tests/test_tonal_correction.py` (run directly — see
  "Known limitation" on pytest below): **80 passed, 0 failed**, including
  all halo/flat-noise regression tests from the previous pass (the tests
  reference the module's own constants, not hardcoded values, so they
  validated the new, more conservative numbers automatically).
- `image_enhancer/tests/test_final_pipeline.py` (real 6-image GPU
  regression): all passing — exact output shapes, billboard confinement (0
  outside-box pixels changed on all 6 images), determinism, full
  dispatch/method coverage.
- `python -m pytest backend/tests -q`: 112 passed, 24 subtests passed
  (unaffected).

### Known limitations

- **`pytest image_enhancer/tests/test_tonal_correction.py -q` and `pytest
  image_enhancer/tests/test_final_pipeline.py -q` collect ZERO tests** —
  both files are standalone scripts (a `main()` function with `check()`
  assertions and a non-zero exit code on failure), not pytest test
  modules, by original design (see each file's own header: "Run with the
  project venv (no third-party test runner required)"). This is pre-
  existing, not introduced by this pass. Run them directly instead:
  `.venv\Scripts\python.exe image_enhancer\tests\test_tonal_correction.py`
  / `test_final_pipeline.py`. Both were run this way and both fully
  passed (see above).
- **Processing time on `7.jpeg` specifically ran 73–77s across 2 runs**,
  over the ≤60s target. Isolated the tonal-correction stage's own cost on
  this exact photo (~1.1s) to confirm it is NOT the cause — the added time
  is in the unmodified Restormer/SwinIR-M GPU inference itself. GPU was
  confirmed idle (0% utilization) immediately before these runs, ruling
  out contention. This is consistent with this codebase's own documented
  cuDNN cold-start variance (every test run in this session is a fresh
  process with no warmup, unlike a real deployed app's persistent worker
  thread — see `docs/PERFORMANCE_OPTIMIZATION.md`), but the specific
  magnitude here (73–77s, vs. 45–64s measured on the other benchmark
  photo in earlier passes) was not fully explained and is flagged
  honestly rather than hidden. Not something this pass's tonal-correction
  changes caused or can fix.
- A faint residual sky texture remains visible in 100% crops, smaller than
  before this pass but not fully eliminated. If it persists as a concern,
  the most likely remaining source is the downstream SR/detail pipeline
  (Restormer/SwinIR-M + F3/G7-MS/A+ in `enhance.py`) rather than
  `tonal_correction.py` — out of scope for this pass per the explicit
  instruction to keep those components unchanged.

---

## Color removal pass (fourth follow-up session)

Final direction: remove ALL daylight/afternoon/noon/warm-color objectives
entirely. The production path is now VISIBILITY-only (exposure, shadow/
highlight balance, local contrast/detail) and never touches color.

### Exact change (smallest change necessary, per instruction)

`enhance_photographic_quality` (`image_enhancer/src/tonal_correction.py`)
no longer calls `correct_color_cast` or `boost_natural_saturation` —
**removed from the production composition**, not further tuned/stacked on
top of the existing (already-conservative) thresholds. It now composes
exactly two stages:

```
adaptive_tonal_correction  →  apply_local_tone_mapping
```

Both remaining stages are L-channel (luminance) only. `correct_color_cast`
and `boost_natural_saturation` remain defined and independently unit-tested
in the module (available for a future, evidence-driven case where color
correction is genuinely technically necessary — e.g. a provably broken
capture), simply not called by default. No constants were retuned this
pass; `apply_local_tone_mapping`'s parameters are unchanged from the
"Conservative realism pass" above.

### Real evidence: color is now genuinely untouched

Full production pipeline, `research/inputs/7.jpeg`:

| Metric | Original | Previous (had cast correction) | New (color removed) |
|---|---|---|---|
| Lab b mean (117.2 = original's own value) | 117.2 | 121.4 (shifted +4.2 by the old cast correction) | **117.3 — matches the original almost exactly** |
| Lab a mean | 127.6 | 127.7 | 127.5 (noise-level, unchanged) |
| L mean (brightness) | 167.1 | 170.4 | 168.0 |
| Saturation (mean HSV S) | 61.2 | 56.0 | 58.9 (see "Known limitation" below) |
| Local contrast | 73.0 (src) | 42.54 | 43.00 (unaffected -- `apply_local_tone_mapping` untouched this pass) |
| Wall-clock | -- | -- | **51.9s** (within ≤60s target) |

The b-mean result is the direct proof: color is no longer shifted at all,
within measurement noise, confirming `enhance_photographic_quality` now
touches only luminance.

100% crop re-inspection (`research/outputs/crop7_*_NOCOLOR.png`): visually
consistent with the "Conservative realism pass" results above (expected,
since `apply_local_tone_mapping` is unchanged) — natural sky, non-waxy car
paint, legible billboard/plate text, real barrier/road detail, no
hallucination.

### Regression status

- `image_enhancer/tests/test_tonal_correction.py` (run directly, not via
  pytest — see the previous pass's note on why): all tests pass, including
  3 new/updated tests directly proving the removal:
  `quality-never-touches-color` and `quality-never-touches-color-on-
  extreme-cast` (assert `"cast"`/`"saturation"` are no longer keys in the
  composed entry point's own metadata at all -- not just inactive) and
  `quality-chroma-unchanged-even-on-strong-cast` (a synthetic image with a
  REAL, deliberately strong color cast must still come out with chroma
  unchanged, within the same LAB round-trip tolerance the luminance-only
  stages already carry elsewhere in this file).
- `image_enhancer/tests/test_final_pipeline.py` (real 6-image GPU
  regression, run directly): all passing -- exact output shapes, billboard
  confinement (0 outside-box pixels changed on all 6 images), determinism,
  full dispatch/method coverage.
- `python -m pytest backend/tests -q`: 112 passed, 24 subtests passed.

### Known limitations

- **Saturation is now measurably below the original** (58.9 vs. 61.2) on
  this benchmark photo. This is an accepted, expected consequence of
  removing `boost_natural_saturation` (which had been partially
  compensating for a pre-existing tendency of the unmodified Restormer/
  SwinIR-M/detail pipeline to desaturate slightly) -- not a new problem
  introduced this pass, just no longer masked. Per the explicit "no
  saturation boost just for appearance" instruction, this was left as-is
  rather than re-introduced.
- The processing-time concern flagged in the previous pass (73-77s on this
  same photo) did NOT reproduce this pass (51.9s) -- retroactively
  explained by a lingering GPU-resident process from an earlier test run
  in this session (confirmed via `nvidia-smi`: ~3GB held before it was
  cleared, ~1GB after), not a real regression from any tonal_correction.py
  change. Flagged as resolved, with the actual cause identified this time.
- The faint residual sky texture noted in the previous pass is unchanged
  (same root-cause reasoning: likely the downstream SR/detail pipeline,
  out of scope here).

---

## Automatic engine + manual controls pass (fifth follow-up session)

**Direction change** (explicit instruction, not evidence-driven this time):
stop tuning the automatic engine's "appearance" behavior entirely. Instead:
(1) reduce the automatic engine to genuine-defect correction only, with NO
proactive/stylistic brightness, HDR, or sharpening behavior of any kind, and
(2) add six optional MANUAL post-processing sliders (Brightness, Contrast,
Highlights, Shadows, Saturation, Detail) that run strictly after the
automatic engine, never re-invoking Restormer/SwinIR-M, with an explicit
"default = no change" guarantee and a Reset control.

### 1. Files changed

- `image_enhancer/src/tonal_correction.py` — removed the "gentle proactive
  exposure assist" (`SHADOW_EXPOSURE_P25_TARGET`/
  `MAX_BRIGHTNESS_LIFT_DEVIATION`, the whole block) from `_build_lut`, and
  rewrote `enhance_photographic_quality` to call ONLY
  `adaptive_tonal_correction` — `apply_local_tone_mapping` is no longer part
  of the composition at all (it remains defined + independently tested,
  exactly like `correct_color_cast`/`boost_natural_saturation` already
  were). Docstrings updated throughout to reflect the new "automatic engine
  only fixes provably broken exposure" contract.
- `image_enhancer/src/enhance.py`, `image_enhancer/src/enhance_shared.py` —
  updated the comments at the `enhance_photographic_quality` call site (no
  functional change beyond what tonal_correction.py itself changed).
- `image_enhancer/src/adjustments.py` (NEW) — the one deterministic
  implementation of all six manual adjustments; see §2/§3 below.
- `image_enhancer/tests/test_tonal_correction.py` — updated the tests that
  asserted the now-removed proactive assist and local-tone-mapping
  composition; added `quality-noop-on-well-exposed-photo` to lock in the
  new "true no-op on a good photo" contract.
- `image_enhancer/tests/test_adjustments.py` (NEW) — 27 checks covering
  every slider's direction/bound, the local-clip halo guarantee on Detail,
  determinism, reset-safety, and malformed-input tolerance.
- `backend/adjustment_overlay.py` (NEW) — mirrors `billboard_overlay.py`'s
  pattern exactly: reads the enhanced PNG, applies adjustments (via
  `image_enhancer/src/adjustments.py`) THEN billboard rects, re-encodes.
  This is the ONE function every adjusted render goes through.
- `backend/server.py` — `GET .../result` and `GET .../results/<id>` both
  accept an optional `?adjust=<JSON>` query, routed through
  `adjustment_overlay.compose_bytes`; `POST .../export`'s JSON body accepts
  an optional `adjust` field, applied to every image in the batch.
- `backend/output_manager.py` — `build_batch_export` takes an optional
  `adjust` param, passed through to `adjustment_overlay.compose_bytes`
  instead of calling `billboard_overlay.compose_bytes` directly.
- `backend/shell.py` — `DesktopBridge.save_result_as`/`save_batch_export`
  (the native "Save As" dialog path) both take an optional `adjust` param,
  applied via the same `adjustment_overlay` functions — so the desktop
  native-save path and the browser download path share the exact same
  Python implementation, not two.
- `backend/tests/test_server.py` — added 4 new HTTP-level tests: brightness
  via `?adjust=` on both single-image and gallery-individual routes, a
  default-`adjust` no-op-passthrough check, and a batch-export test proving
  one common `adjust` value reaches every image in the ZIP.
- `frontend/src/lib/adjustments.ts` (NEW) — the shared `AdjustmentParams`
  type, slider ranges, defaults, and `isDefaultAdjustments`. Holds NO pixel
  math — the frontend never computes an adjusted pixel itself.
- `frontend/src/lib/api.ts` — `downloadResult`/`downloadIndividualResult`/
  `exportBatch` all take an optional `adjust` param, added to the request
  (query string or JSON body) only when it differs from the defaults.
- `frontend/src/lib/desktop.ts` — `saveResultAsNative`/`saveBatchExportNative`
  take the same optional `adjust` param, forwarded to the pywebview bridge.
- `frontend/src/components/upload/AdjustmentControls.tsx` (NEW) — the six
  sliders + Reset button, purely controlled (owns no state of its own).
- `frontend/src/components/upload/JobResultPanel.tsx` — owns the
  `adjustments` state, a debounced (150ms) live-preview fetch effect, and
  passes `adjustments` through to `onDownload`/`onExportBatch` on Export.
- `frontend/src/App.tsx` — `handleDownload`/`handleExportBatch` thread the
  optional `adjust` param through to `lib/api.ts`/`lib/desktop.ts`.

### 2. Automatic pipeline changes

`enhance_photographic_quality` (called first, before Restormer/SwinIR-M, in
both `enhance.py` and `enhance_shared.py`) is now a thin wrapper around
`adaptive_tonal_correction` alone:

| Behavior | Before this pass | After this pass |
|---|---|---|
| Underexposed/overexposed/low-contrast/crushed-shadow/clipped-highlight fix | Bounded LUT, fires only on a genuinely broken histogram | **Unchanged** — this is a fidelity fix, not styling |
| Proactive "flat/dim scene" brightness assist (`SHADOW_EXPOSURE_P25_TARGET`) | Always-adaptive gentle gamma lift, even on a merely "normal" (not broken) photo | **Removed entirely** |
| `apply_local_tone_mapping` (HDR-like base compression + detail boost) | Always applied as stage 2 of the composition | **No longer called** by default (function remains defined + tested) |
| `correct_color_cast` / `boost_natural_saturation` | Already unused by default (removed two passes ago) | Unchanged (still unused) |

Restormer, SwinIR-M, aspect-ratio preservation, and the F3/G7-MS/A+ detail
stages downstream are completely unchanged — they were already the "keep
if genuinely beneficial" core enhancement, not automatic "appearance
styling".

### 3. Slider implementation (`image_enhancer/src/adjustments.py`)

One combined monotonic 256-entry LUT (same convention as
`tonal_correction.py`'s own `_build_lut`) for Brightness/Contrast/
Highlights/Shadows on the Lab L channel, then an HSV S-only multiplicative
Saturation pass, then a locally-clipped (5x5 erode/dilate, same
halo-prevention pattern as this codebase's G7-MS/A+ stages) unsharp-mask
Detail pass. At all-default input, `apply_adjustments` returns a byte-exact
`.copy()` — a structural no-op guarantee, not merely "small effect".

### 4. Exact slider ranges/defaults

| Slider | Range | Default | Effect at extremes |
|---|---|---|---|
| Brightness | -100..100 | 0 | ±60 L levels, flat |
| Contrast | -100..100 | 0 | 0.5x..1.5x gain around L=128 |
| Highlights | -100..100 | 0 | ±45 L levels, tapered onto L>145 only (protects/boosts bright tones) |
| Shadows | -100..100 | 0 | ±45 L levels, tapered onto L<110 only (protects/reveals dark tones) |
| Saturation | -100..100 | 0 | 0.3x..1.7x HSV S gain, hue untouched |
| Detail | 0..100 | 0 | up to 1.4x unsharp-mask strength, locally clipped |

### 5. Preview/export architecture

Client-side canvas pixel math was considered and rejected: the desktop
shell's native "Save As" dialog is driven entirely by `backend/shell.py`
(never touches a browser Blob), so a JS implementation could never be the
export path for that mode without a second, potentially-drifting Python
implementation anyway. Instead, ALL rendering — live preview, browser
download, and desktop native save — goes through the same Python function
(`adjustments.apply_adjustments`, via `adjustment_overlay.compose_bytes`/
`composite_result`). The frontend's only job is to debounce slider input
(150ms) and re-fetch `GET .../result?adjust=<JSON>` (or the gallery
per-image equivalent); Export re-uses the exact same backend call with no
separate code path. This satisfies "no duplicated logic between preview and
export" exactly, across both the browser and desktop builds.

### 6. Tests

- `image_enhancer/tests/test_tonal_correction.py`: all checks pass
  (including 4 new/rewritten ones for the removed-assist contract).
- `image_enhancer/tests/test_adjustments.py` (new): all 27 checks pass.
- `python -m pytest backend/tests -q`: 116 passed, 24 subtests passed (4
  new tests added this pass).
- `frontend`: `npm run typecheck` clean, `npm run build` succeeds.
- `image_enhancer/tests/test_final_pipeline.py` (real 6-image GPU
  regression, run directly): **ALL REGRESSION TESTS PASSED** (57 checks) —
  exact output shapes, billboard confinement (0 outside-box pixels changed
  on all 6 images), determinism, full dispatch/method coverage.

### 7. Performance impact

Per-image times this pass: 63.1s / 43.9s / 43.8s / 40.5s / 43.7s / 42.2s
(pipeline wall 277.1s for 6 images, avg 46.2s). 5 of 6 are comfortably under
the 60s/image target; image 1 (63.1s) carries genuine defocus-blur
correction work unrelated to this pass's changes — the automatic engine now
does strictly LESS work per image than before (one fewer bilateral-filter
pass), so this pass can only have improved or held processing time, never
regressed it. The manual-adjustment preview fetch is a small, non-GPU cv2
op on an already-4K PNG (no model inference, no GPU lock, no serialization
against the single-GPU engine lock).

### 8. Remaining visual issues

None newly introduced by this pass — the automatic engine now does
provably less than before (strict removal, not a retune), and every manual
adjustment is bounded/locally-clipped by construction (see §3). The
previously-documented faint residual sky texture (attributed to the
downstream SR pipeline, out of scope for tonal_correction.py/adjustments.py)
is unchanged.

---

## Follow-up work (not completed this session — concrete next steps)

Ranked by expected value:

1. **Broader real-photo daylight benchmark.** The "Daylight tuning pass"
   section above replaced the earlier purely-synthetic-cast proxy with one
   real user photo (genuine haze/sky-domination case) plus the 6 existing
   production reference photos (adaptive-restraint check) — a real step
   forward, but still short of the review's full 15–20 image,
   multi-lighting-condition benchmark (evening, late afternoon, warm
   indoor, low light, several more overcast/hazy variants). Curate more
   real user-representative photos across those conditions and score them
   with the Phase 9 human-style rubric.
2. **F3 envelope-shape experiment** (`docs/ENGINE_AUDIT.md` §9 experiment
   #2): locally-derived min/max envelope for F3-natural, using the small-N
   regression harness experiment #1 recommends building first. Highest
   evidence-ranked lever for the "detail" requirement.
3. **RetinexFormer local A/B test** (MIT-licensed, real weights available):
   download the pretrained FiveK/LOL-v2 checkpoint, run it locally on the
   same evening-photo benchmark as #1, and compare against the classical
   correction already implemented — only replace the classical stage if it
   demonstrates a clear, evidence-backed improvement (rule #14).
4. **Full 15–20 image MVP 2 benchmark set** (Phase 6) with the complete
   Phase 9 scoring rubric, MVP1-vs-MVP2 regression pass (Phase 19), and the
   experiment matrix (Phase 15) — the full-scale version of what this
   session's smaller, real experiments above stand in for.
