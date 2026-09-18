# MVP 2 Benchmark — Regression Evidence

Companion to `docs/MVP2_RESEARCH.md` (why) and `docs/MVP2_PIPELINE.md`
(what changed). This document is the actual test/benchmark evidence that
the MVP 2 changes are safe.

## Hardware

Office/dev machine used for every measurement in this document: NVIDIA
RTX 3050 8GB desktop, Intel Core Ultra 7 265K — the same target hardware
`docs/PERFORMANCE_OPTIMIZATION.md`'s existing MVP 1 timing numbers were
measured on.

## 1. Aspect-ratio formula — pure math verification

`research/scripts/aspect_ratio_formula.py`. All 6 of the review's own
worked examples matched exactly; all 6 required ratio families (1:1, 4:3,
3:2, 16:9, 9:16, 21:9) preserved aspect ratio, had longest side == 3840, and
even dimensions. See `docs/MVP2_RESEARCH.md` Phase 2 for the full output.

## 2. Aspect-ratio stretch bug — real repro against the production pipeline

`research/scripts/repro_stretch_bug.py`, run BEFORE the fix, against a real
photo (`comparison_original.png`, a genuine 256×256 square) through the
exact unmodified `engine.enhance("final", img)` call path:

```
source: comparison_original.png 256x256 (aspect 1.0000)
output: 3840x2160 (aspect 1.7778)  engine dt=32.60s
CONFIRMED STRETCH BUG: True
```

## 2b. Aspect-ratio fix — real repro AFTER the fix (production call path)

`research/scripts/verify_stretch_fix.py` — the same 256×256 square photo,
now run through `backend.engine_adapter.enhance_image()` (the REAL,
currently-shipping-on-this-branch production entry point, not a direct
engine call):

```
source: comparison_original.png 256x256 (aspect 1.0000)
output: 3840x3840 (aspect 1.0000)  wall=21.23s
ASPECT RATIO PRESERVED (no stretch): True
```

The identical input that produced a stretched 3840×2160 output in §2 (before
the fix) now produces a correctly-square 3840×3840 output (after the fix) —
through the real GPU pipeline, real Restormer + SwinIR-M inference, not a
mock.

## 3. Backend unit/wiring tests (fast, no GPU inference)

`python -m pytest backend/tests -q`: **112 passed, 24 subtests passed**
(0.86s excluded — full run ~10.6s). New test files/additions this session:

- `backend/tests/test_aspect_ratio.py` (new, 6 tests / 24 subtests):
  - Formula matches every review example and ratio family.
  - `enhance.py`'s and `enhance_shared.py`'s independently-duplicated
    formula copies agree on every case (never drift apart).
  - **GPU path wiring**: `engine_adapter.enhance_image()` computes the
    correct target for a real 256×256 (1:1) PNG and threads it into
    `engine.enhance()` (the real engine call itself is monkeypatched to a
    cheap stand-in, avoiding a real multi-second GPU call in a unit test —
    same style as the existing `test_engine_adapter.py`'s billboard tests).
    Asserts the saved output file is genuinely 3840×3840, not 3840×2160.
  - **CPU path wiring**: `cpu_worker._run_one()` computes the correct target
    for a real 1200×1600 (portrait) PNG and threads it into
    `cpu_final_enhance()`. Asserts the saved output is genuinely
    2880×3840.
- `backend/tests/test_server.py` (2 new tests):
  - Billboard-overlay compositing maps correctly onto a genuinely
    non-16:9 (400×400 square) result — a rect that would be out-of-bounds
    under a 16:9 assumption at the same pixel offsets is valid and renders
    correctly on the real square image; output dimensions are preserved
    exactly (400×400 in, 400×400 out — never stretched to 16:9).
  - Batch export (ZIP) preserves each image's own real, independent,
    non-16:9 dimensions (320×320 and 480×270 in the same batch) —
    confirms per-image export never forces a common aspect ratio.
- `backend/tests/test_jobs.py`, `test_output_manager.py`,
  `test_desktop_bridge.py`: unchanged, still passing (these predate MVP 2
  and cover the gallery/batch-export feature from the prior session's work,
  unaffected by this session's changes).

## 4. Real GPU pipeline regression — legacy 6-image harness

`image_enhancer/tests/test_final_pipeline.py` (pre-existing, not written
this session) exercises the real `final_enhance` end-to-end on 6 real
production reference photos, asserting exact output shape, billboard
confinement, outside-box pixel stability, determinism, and full
CLI/dispatch behavior. Run **after** this session's aspect-ratio +
daylight-normalization changes, with `target` never explicitly passed (the
exact way every one of this harness's own calls invokes `final_enhance` —
proving default/backward-compatible behavior is untouched):

Run twice this session — once immediately after the aspect-ratio wiring
change, once again after the daylight-normalization change — both **100%
pass**, final run:

```
ALL REGRESSION TESTS PASSED (pipeline wall 282.7s for 6 images)
```

Every check passed both times: exact `(2160, 3840, 3)` output shape for all
6 images (the unchanged default target), billboard-box confinement
(0 outside-box pixels changed beyond the OpenCV round-trip floor, both
runs), outside-box pixel stability, the no-box synthetic-image identity
check, determinism (byte-identical rerun), defocus-only native-recovery
gating, and every `METHODS`/dispatch entry point. This is the strongest
available evidence that threading `target` through `_final_d1_weak` and
composing `normalize_daylight_appearance` did not change MVP 1's own
established behavior for any caller that doesn't opt into the new
aspect-ratio/daylight path.

Note on the 282.7s/6-image (~47s/image average) timing: this harness
deliberately exercises the Candidate A billboard branch (it sets
`BILLBOARD_IMAGE` per image, unlike production, which
`engine_adapter._neutralize_billboard_regions` always disables) — so its
per-image time includes extra Restormer motion-deblur + per-box SwinIR-M SR
that a real user upload never pays. It is not representative of real
per-image production time; see `docs/PERFORMANCE_OPTIMIZATION.md`'s own
~25–27s steady-state GPU number for that (unaffected by this session's
changes — no new model or GPU call was added to the whole-frame path real
users take). Re-run via
`.venv\Scripts\python.exe image_enhancer\tests\test_final_pipeline.py`.

## 5. Daylight color-cast correction — unit evidence

`image_enhancer/tests/test_tonal_correction.py` section 11 (new), run
against real production reference images plus synthetic controlled cases —
**all passing**:

- Neutral image → no correction applied, byte-identical output (adaptive,
  not blanket).
- Real warm cast (R×1.35, B×0.75 on a neutral synthetic photo) → detected,
  corrected, moved measurably toward neutral, bounded within the documented
  ±18 ceiling.
- Extreme cast → correction hits the ±18 ceiling exactly (proves the cap
  binds, not just usually-unreached).
- Luminance is not a target of the chroma correction (small, ≤10/255
  incidental effect only at gamut extremes from the LAB↔BGR round-trip
  itself — the same caveat `adaptive_tonal_correction`'s own pre-existing
  test already carries in the reverse direction; both are inherent to
  8-bit OpenCV colorspace conversion, not a defect in either function).
- `normalize_daylight_appearance` composes both stages correctly on a
  dim+warm synthetic image (both exposure lift and cast pull applied) and
  is deterministic.

Real quantitative evidence (`research/scripts/daylight_candidate_classical.py`,
a controlled synthetic-cast experiment on a real photo — see
`docs/MVP2_RESEARCH.md` Phase 4 for the full methodology and its stated
limitation): **44.0% reduction in Lab a/b distance from the true daylight
original**.

## 6. Frontend

`npx tsc --noEmit`: clean. `npm run build`: succeeds
(`dist/assets/index-DmRMjaVZ.js`, 68 modules transformed, no errors or new
warnings). Board editor (`BillboardCanvas.tsx`) and board-thumbnail preview
(`JobResultPanel.tsx`) now use each result's real detected dimensions
instead of a hardcoded 3840×2160 — verified at the type level and via a
successful production build; not verified via live browser interaction this
session (see "Known limitations" below).

## Known limitations of this benchmark pass

- **No paired real evening/daylight photo benchmark yet** — the 44% figure
  is from one controlled synthetic-cast experiment on one photo, not a
  diverse real-world benchmark. See `docs/MVP2_RESEARCH.md` "Follow-up
  work" #1.
- **No live browser verification** of the frontend dimension-detection
  change (board drawing on a genuinely non-16:9 gallery result, board
  overlay export end-to-end through the UI) — verified at the type-check +
  production-build level only. The backend-level board coordinate mapping
  IS verified end-to-end (see §3's server tests).
- **Detail/SR requirement has no new benchmark** — by design, per
  `docs/MVP2_RESEARCH.md` Phase 5's recommendation to not implement a new
  model this session. `docs/ENGINE_AUDIT.md`'s own pre-existing 3-image
  measurement remains the best available evidence for that requirement.
- **Full 15–20 image MVP 2 benchmark set (review Phase 6) was not built**
  — this document's evidence is real but narrower in scope; see
  `docs/MVP2_RESEARCH.md` "Follow-up work" #4.

## 7. Daylight tuning pass — real photo, real pipeline, real numbers

A follow-up session replaced the §5 synthetic-cast proxy evidence with a
real user-uploaded street photo (haze/overcast, large blown sky, billboard
text, vehicles, buildings) run through the actual production call path
(`backend.engine_adapter.enhance_image`, real Restormer + SwinIR-M
inference — not a mock):

| Metric | Original | Before tuning | After tuning |
|---|---|---|---|
| Output size (aspect preserved) | 1280×960 | 3840×2880 | 3840×2880 |
| L mean (brightness) | 187.6 | 171.5 | 181.3 |
| Lab b mean (warmth, 128=neutral) | 121.8 | 125.3 | 129.9 |
| Local contrast (Laplacian std @ 4K) | — | 32.9 | 34.8 (+5.8%) |
| Wall-clock | — | — | 45.9s (within ≤60s target) |

Billboard text and vehicle text crops confirmed pixel-for-pixel legible and
unaltered in both before/after (no hallucination). Full write-up, root
causes, and the adaptive-restraint check against the 6 existing well-lit
reference photos: `docs/MVP2_RESEARCH.md` "Daylight tuning pass."

Regression re-run after this pass, all passing: `image_enhancer/tests/
test_tonal_correction.py` (9 new/updated tests), `image_enhancer/tests/
test_final_pipeline.py` (real 6-image GPU suite, unaffected billboard
confinement/determinism), `python -m pytest backend/tests -q` (112 passed,
24 subtests — unaffected, no backend files touched this pass), frontend
`tsc`/`build` (unaffected, no frontend files touched this pass).
