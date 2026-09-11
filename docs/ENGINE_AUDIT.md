# Image Enhancement Engine — Audit (no production code changed)

Scope: `image_enhancer/` only. This is an audit and tuning plan, not an
implementation. No file under `image_enhancer/`, `backend/`, or `frontend/`
was modified to produce it. Evidence comes from (a) direct reading of
`image_enhancer/src/enhance.py` and `g9_exp/recipe_g9.py`, (b) the existing
R&D report history (`image_enhancer/reports/**/*.md`, `*.csv`), and (c) a
small, controlled measurement run (3 real images, reusing existing
`quality.py` metrics) described in §8.

## 1. Current production pipeline — exact execution order

Entry point: `enhance.enhance("final", img)` → `enhance.final_enhance()`.
This is what `backend/engine_adapter.py` actually calls for every real
upload (`METHOD = "final"`, hardcoded).

```
img (original, BGR uint8)
 │
 ├─► simple_upscale (Lanczos)                          → faithful (4K baseline)
 │
 ├─► _final_d1_weak(img):
 │     Restormer "real_denoise" (tiled 512/48) blended
 │     K=0.30 with original  →  SwinIR-M x4 (PSNR ckpt,
 │     tiled 256)  →  Lanczos resize to 4K              → d1_4k
 │
 ├─► _final_f3_natural(faithful, d1_4k)   ["F3-natural"]
 │     band-pass(σ=2.5) the (d1-faithful) L-delta,
 │     edge-gate(t0=0.30), scale ×6, clip ±18,
 │     final envelope clip ±14, chroma from faithful    → f3
 │
 ├─► _final_multiscale_detail(f3)          ["G7-MS"]
 │     3-band Laplacian boost (gains 1.30/1.20/1.15 @
 │     σ=1/2/4), each band clipped to its own local
 │     erode/dilate envelope (ksize 3/5/9), summed,
 │     edge-gate(t0=0.30), clip ±10                     → f3_ms
 │
 ├─► _final_whole_frame_detail(f3_ms)      ["A+"]
 │     single-scale (σ=1.0) local-contrast unsharp,
 │     clipped to local 5×5 min/max, strength 0.6,
 │     edge-gate(t0=0.35), clip ±6                      → f3_plus
 │
 ├─► IF verified regions.json boxes exist for this image:
 │     per box → Candidate A (native crop → Restormer
 │     motion-deblur → SwinIR-M x4 SR → resize to true
 │     frame scale → adaptive-σ Richardson-Lucy +
 │     edge-gated USM touch-up) → feathered composite
 │   ELSE: out = f3_plus                                → final
 │
 └─► resize-to-target guard (no-op if already 3840×2160) → OUT (3840×2160 BGR uint8)
```

**Critical fact for this product specifically:** `backend/engine_adapter.py`
deliberately points `REGIONS_CONFIG` at a path that never exists, for every
call (see its `_neutralize_billboard_regions`, added to stop the shipped
`regions.json` fixture's boxes from leaking onto a generic photo — see the
earlier backend audit). **`_billboard_boxes()` therefore always returns `[]`
for real users of this product, so the entire Candidate A branch — native
crop, Restormer motion-deblur, per-box SwinIR-M SR, adaptive
Richardson-Lucy, feathered compositing — never executes.** Every real user
photo takes the `else: out = f3_plus` branch. This is the single most
consequential fact in this audit: roughly half of `final_enhance`'s
documented sophistication, and the entire G1–G5 R&D lineage that produced
it, is dead code for this product as currently wired up.

## 2. Best existing experimental candidate

**`enhance_g91.py` ("G9.1-B")**, wrapping `g9_exp.recipe_g9.g9_enhance()`
with locked parameters (`strength=0.30, clip=±5, gate_t0=0.40`). It adds a
controlled, self-clipped high-frequency contribution from a *generative*
SwinIR-M **GAN** checkpoint on top of the unmodified `final_enhance`
baseline, using the exact same containment architecture (band-pass → edge
gate → local min/max clip → chroma untouched → global clip) already proven
by F3/G7-MS/A+ — just applied to a generative source instead of a
regression one, with tighter bounds. `disable_on_boards=True` feathers it
to zero inside any verified billboard box, so it's additive-only outside
boards.

It is the best candidate **on paper** (soundest containment design, reuses
already-vetted infrastructure, real regression + real-world test coverage —
`image_enhancer/tests/test_g91_integration_pipeline.py` passes cleanly).
But it is **not wired into production** — `METHODS` in `enhance.py` has no
`"g91"` entry, and `engine_adapter.py` never imports `enhance_g91`. Its
actual measured benefit is small relative to its cost (§8) — "best
candidate" is a statement about architecture and safety, not about
demonstrated cost-adjusted value yet.

No `G10`/`G10.1` experiment exists anywhere in this codebase (checked: no
matching files, directories, or docstring references). The most advanced
work on disk is G9.1-B.

## 3. What each production stage actually contributes

| Stage | Contribution | Evidence |
|---|---|---|
| D1-weak (Restormer denoise blend K=0.30 + SwinIR-M SR) | Real detail recovery vs. faithful Lanczos: PSNR 27.38 vs faithful's own 27.52 baseline-less-denoise comparison, sharper (316.6 vs 296.0), lower noise (1.29 vs 1.37) | `pro_exp/D1_SWEEP_REPORT.md` |
| F3-natural fusion | Transfers *some* of D1's improvement back, but **suppresses 45–70% of D1-weak's Laplacian-variance detail** by design (band-pass + tight envelope) — the single largest detail-loss step in the whole pipeline, everywhere, not just at boards | `reports/g8_exp/G8_DIAGNOSTIC_REPORT.md` §"Direct answers", Q2 |
| G7-MS (multiscale detail) | Recovers some of what F3 suppressed: +25–27% full-frame sharpness at SSIM −0.004…−0.007, PSNR −0.6…−0.7dB vs. the pre-G7-MS baseline | `reports/g7_exp/G7_EXP_REPORT.md` §3 |
| A+ (whole-frame local contrast) | Further sharpening on top of G7-MS's output; original G6 finding: +sharpness at ~1dB PSNR / 0.006 SSIM cost vs. plain D1+F3 | `g6_exp/extend_A_vs_Aplus.csv`, `enhance.py` header |
| Candidate A (billboard boxes) | **Never runs for this product** (see §1) | direct code read |

## 4. Current measurable weaknesses (evidence-based)

1. **Most of every frame receives no meaningful enhancement at all.** In my
   own controlled measurement (§8), **77.5%–84.7% of pixels in the final
   production output are within 1 L-unit of the plain faithful Lanczos
   upscale** — i.e., statistically untouched. This follows directly from
   F3/G7-MS/A+ all being edge-gated (t0 = 0.30 / 0.30 / 0.35): any pixel
   below ~30–35% of the image's own 97th-percentile gradient magnitude gets
   *zero* contribution from any stage. Skies, walls, out-of-focus
   backgrounds, and low-texture skin all fall in this untouched majority.
2. **F3's fusion envelope is the dominant, quantified bottleneck**, not the
   SR backbone. G8 measured a 45–70% Laplacian-variance drop at F3 in every
   region tested, everywhere in the frame (not just billboards). G6 already
   demonstrated the practical consequence: swapping D1's backbone to
   SwinIR-L, or blending in Restormer motion-deblur, "barely changed the
   final output at all" — model-capability upgrades are wasted because F3
   clips away the difference before it can matter.
3. **Billboard compositing throws away detail Candidate A just
   reconstructed** (fixed 4× SR output Lanczos-downsampled to the frame's
   real, lower upscale factor before blending — a ~51% pixel-count cut) —
   but this is moot for this product today since the branch never executes
   (§1). Flagged for completeness, not actionable here.
4. **G9.1-B's real contribution is very subtle relative to its cost**
   (quantified in §8): mean |ΔL| 0.16–0.22 (out of 255), only 4.9–6.9% of
   pixels touched at all, +1.8–4.1% additional sharpness — for roughly
   **double** the per-image wall-clock time.
5. **Traceability gap**: `enhance_g91.py` and `recipe_g9.py` cite tuning
   evidence at `reports/g9_1_exp/{B,real_world}/summary.json` — confirmed
   absent from this checkout. The locked `strength=0.30` parameter cannot
   currently be independently re-derived from committed evidence.
6. **VRAM is measured per-call-peak only, never cumulative-resident.** The
   real deployment (`backend/jobs.py`) is a single long-lived process that
   will end up with Restormer (real-denoise, and motion-deblur if boards
   are ever used again), SwinIR-M PSNR, and — if G9.1-B is adopted —
   SwinIR-M GAN all cached simultaneously in the same process. No existing
   report measures that combined resident footprint on the actual 8GB
   target card; every report measures one call's own peak in isolation.

## 5. Redundant stages / stages suppressing useful detail for this product

- **Candidate A and everything under it** (billboard crop, motion-deblur,
  per-box SR, adaptive RL touch-up, feathered compositing) — 100% dead code
  for this product's real traffic (§1). Not a bug, just currently inert;
  worth knowing before spending audit/tuning effort there.
- **F3-natural's fusion step** is the stage that most actively suppresses
  otherwise-already-computed useful detail (§4.2) — it is the highest-value
  target for improvement, not a redundant stage to remove (removing it
  would lose its real, if partial, fidelity-preserving benefit).

## 6. Parameters currently limiting quality

In order of measured impact:

1. **Edge-gate thresholds** (`_FINAL_F3_T0=0.30`, G7-MS's `_FINAL_MS_GATE_T0=0.30`,
   `_FINAL_APLUS_T0=0.35`) — these, not the clip magnitudes, decide *where*
   any enhancement can happen at all, and directly produce the 77–85%
   untouched-pixel finding.
2. **F3's envelope shape** (`_FINAL_F3_ENV=14.0`, a flat global scalar) —
   already known to be the dominant suppression step (§4.2); already known
   that *widening* the flat scalar (G7-WIDE tested ±20) buys the largest
   sharpness gain but the worst fidelity divergence of any conservative
   candidate tested — i.e. the *shape* of the clip, not just its size, is
   the open question (see Experiment 1 below).
3. **G9.1-B's `strength=0.30`** — real but small at this value (§8); cannot
   currently be tied back to the missing validation summaries (§4.5).

## 7. What can realistically be improved without hallucinating content

Everything the existing R&D chain has already validated is a *self-clipped,
locally-derived* envelope (either a literal local min/max of the source, or
a per-band structuring-element envelope, or an edge-gated band-pass) —
never an unconstrained sharpen and never a generative fill. The realistic
improvement space is:

- Making **more of the frame eligible** for these already-safe mechanisms
  (lower gate thresholds), and
- Making the **envelope shape itself locally adaptive** instead of a flat
  global number (matching the self-clip philosophy F3 itself doesn't yet
  use, but A+/G7-MS/G9 already do) —

both of which stay inside the "hard-clipped to something already present in
the pixel's own neighborhood" guarantee that every accepted stage in this
codebase already relies on. Swapping learned-model backbones is *not* a
realistic path to more quality on its own — G6 already falsified that for
this architecture (§4.2) — so it's ranked low below.

## 8. Controlled measurement (this audit's own evidence)

3 real images (`image_enhancer/originals/2.jpeg`, `4.jpeg`, `6.jpeg`), each
run through the exact same code path `engine_adapter.py` uses (regions
neutralized, zero billboard boxes), using `quality.py`'s existing metrics.
`final_enhance` and `g9_enhance` share a single `final_enhance` call via
`precomputed_baseline` to avoid doubling GPU cost. No production code
touched; script and raw JSON kept outside the repo (scratchpad).

| image | final PSNR/SSIM vs faithful | final sharpness gain | final untouched % (\|ΔL\|<1) | final runtime | final peak VRAM |
|---|---|---|---|---|---|
| 2.jpeg | 31.73 / 0.962 | 14.1× | 77.5% | 44.3s | 2374 MB |
| 4.jpeg | 33.60 / 0.978 | 9.7× | 84.7% | 38.7s | 2504 MB |
| 6.jpeg | 32.33 / 0.964 | 16.5× | 77.6% | 44.3s | 2504 MB |

| image | +G9.1-B PSNR/SSIM vs faithful | +G9.1-B sharpness gain over final | G9 mean\|Δ| / max\|Δ| | G9 % pixels touched | G9 incremental runtime | G9 incremental peak VRAM |
|---|---|---|---|---|---|---|
| 2.jpeg | 31.41 / 0.961 | 1.028× | 0.216 / 5.0 | 6.86% | +33.4s | 1483 MB |
| 4.jpeg | 33.26 / 0.977 | 1.018× | 0.159 / 5.0 | 4.89% | +32.6s | 1486 MB |
| 6.jpeg | 31.92 / 0.962 | 1.041× | 0.198 / 5.0 | 6.11% | +38.1s | 1512 MB |

Determinism: not re-measured here — already established robustly this
session (`image_enhancer/tests/test_final_pipeline.py` and
`test_g91_integration_pipeline.py` both assert and pass byte-identical
reruns). No reason to believe it changed; not worth spending GPU time to
re-confirm for this audit.

Interpretation: G9.1-B is real, safe, and consistent with its own design
intent — but at current parameters it roughly **doubles per-image runtime**
for a **~2–4% additional sharpness gain touching under 7% of pixels at a
mean magnitude of ~0.2/255**. That is not "no benefit" but it is a small
benefit for a large cost, and it is not yet the highest-leverage lever
available (§4.1's untouched-77–85%-of-the-frame finding is bigger and
cheaper to address).

## 9. Recommended controlled experiments, ranked by expected benefit/risk

| # | Experiment | Targets | Risk | Expected benefit | Why this order |
|---|---|---|---|---|---|
| 1 | **Build a standing small-N regression-metrics harness** (≈this audit's script, committed under `image_enhancer/`) using `quality.py` + the untouched-% check, run on a fixed 3–6 image set | tooling, not the pipeline | None (no pipeline change) | High leverage for every experiment below | Nothing else here should be tuned without a repeatable, cheap way to catch a regression; this audit had to build one ad hoc |
| 2 | **Give F3-natural a locally-derived envelope** (per-pixel local 5×5 min/max of the source, matching the exact self-clip pattern already used by A+, G7-MS's bands, and G9) instead of its current flat global `±14` scalar | the #1 measured bottleneck (§4.2) | Low — every primitive involved is already production-proven elsewhere in this file | Medium–High: should recover more of D1-weak's real local contrast in high-dynamic-range regions without repeating G7-WIDE's failure mode (flat widening = worst fidelity divergence tested) | Directly targets the largest, most-consistently-measured weakness, with a *new* combination of already-trusted mechanisms — not a re-run of a rejected idea |
| 3 | **Lower the edge-gate thresholds modestly** (test t0 ∈ {0.20, 0.25} for F3/G7-MS, and a matching drop for A+) | the 77–85% "untouched" finding (§4.1) directly | Low — clip magnitudes (the actual safety bound) are unchanged; only *where* they can apply changes | Medium: shrinks the untouched-pixel majority; must watch the existing `noise` metric for amplification in near-flat regions as the guardrail | Second-highest-leverage, cheapest to test (pure threshold sweep, no new code), same self-clip guarantees hold throughout |
| 4 | **Re-run and commit the G9.1-B validation** that `enhance_g91.py`'s own docstring cites but that no longer exists on disk (`reports/g9_1_exp/{B,real_world}/summary.json`) | traceability gap (§4.5) | None (measurement only) | Makes the existing `strength=0.30` lock either justified or correctable with real evidence, before any adoption decision | Should happen before, not after, considering promoting G9.1-B |
| 5 | **Measure cumulative resident VRAM** with all four models that would be simultaneously cached in the real long-lived `backend/jobs.py` worker (Restormer real-denoise + SwinIR-M PSNR + SwinIR-M GAN, plus Restormer motion-deblur if boards are ever re-enabled) on the actual 8GB target card | deployment risk gap (§4.6) | None (measurement only) | Prevents an in-production OOM surprise if G9.1-B (or any future 4th model) is adopted | Cheap, and a prerequisite for #6 |
| 6 | **Only after #4 and #5**: evaluate wiring G9.1-B into `engine_adapter.py` as an optional/default stage, informed by real validation data and real cumulative-VRAM headroom, weighed explicitly against its measured ~2× runtime cost (§8) | G9.1-B production adoption | Medium (new model in the hot path, real runtime/VRAM cost, currently un-reproducible tuning basis) | Small-to-medium per §8; not zero, but not free either | Architecturally the safest generative option in the codebase, but the cost/benefit case is not yet made — sequenced last on purpose |
| — | **Do not repeat**: SR backbone swaps (SwinIR-L, Restormer-augmented D1) or flatly widening F3's envelope | already falsified | — | — | G6 already showed backbone swaps are invisible downstream of F3's clip (§4.2); G7 already showed flat envelope widening is the highest-fidelity-risk conservative candidate tested. Re-running either without first changing the envelope *shape* (#2) would just reproduce the same negative results. |

## 10. Confidence notes / open gaps

- G9.1-B's own historical validation data is missing (§4.5) — treat its
  locked parameters as *plausible*, not independently re-verified by this
  audit.
- This audit's own measurement (§8) used 3 images, matching the "no huge
  uncontrolled sweep" instruction; the existing R&D reports it corroborates
  used 6–7 images each, so the combined evidence base is reasonably broad
  even though this audit's own contribution is intentionally small.
- Every quality number here is against the *faithful Lanczos upscale*
  baseline, matching this codebase's own established methodology
  (`image_enhancer/README.md`) — not against any external ground truth,
  since none exists for real phone photos.
