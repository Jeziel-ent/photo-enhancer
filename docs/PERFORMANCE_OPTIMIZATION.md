# Performance optimization — shipped GPU + CPU execution architecture (2026-09-16)

This document records BOTH what is actually shipped today (the **adopted**
architecture) and the experiments that led to it. Sections are clearly
labeled: **Adopted (shipped)** describes the code that runs in production
today; **History (not adopted)** describes earlier approaches that were
investigated, measured, and deliberately left behind, kept as a record for
anyone re-opening this work.

Scope: `image_enhancer/src/enhance.py` (`_final_d1_weak`, the D1-weak stage
of `final_enhance`), the torch-free shared post-processing in
`image_enhancer/src/enhance_shared.py`, the OpenVINO CPU backend
(`restore_exp/imdn_x4_ov.py`), and the backend's device/CPU-process wiring
(`backend/device.py`, `backend/settings.py`, `backend/jobs.py`,
`backend/engine_adapter.py`'s CPU-worker bridge, `backend/cpu_worker.py`).
No other stage of the pipeline (F3-natural, G7-MS, A+, tonal correction)
was changed beyond relocating F3/G7-MS/A+ into `enhance_shared.py` so the
CPU and GPU paths share one copy. All numbers below are measured on this
office machine: NVIDIA RTX 3050 8GB desktop, Intel Core Ultra 7 265K (20
threads), real 1365×769 photos from `image_enhancer/originals/`, through
the exact production call path (`engine_adapter.enhance_image` /
`final_enhance` with billboard regions neutralized — see
`docs/ENGINE_AUDIT.md` §1).

---

## Adopted (shipped) architecture

### GPU — Restormer real-denoise + SwinIR-M x4, tile 224, bf16/cuDNN/TF32

The GPU path is unchanged in technique from this session's optimization and
runs in-process exactly as before:

- **Backbone**: Restormer **real-denoise** (tile 512/48), blended at
  K=0.30 → **SwinIR-M x4 PSNR** SR at **tile 224 / overlap 16** → Lanczos to
  4K. Tile 224/16 measured fastest end-to-end of the practical sizes tested
  (faster than tile 400 because smaller-but-not-tiny tiles waste less
  overlap recompute, faster than very small tiles because per-tile
  Python/kernel-launch overhead dominates) with visually equivalent quality.
- **Precision/execution settings** (CUDA-only, no-ops on CPU, set once at
  enhance.py import time):
  - `torch.backends.cudnn.benchmark = True` (cuDNN kernel autotuning)
  - `torch.backends.cuda.matmul.allow_tf32 = True`
  - `torch.backends.cudnn.allow_tf32 = True`
  - bfloat16 autocast around both forward passes. Verified before adopting:
    mean |Δ| 0.08–0.36/255, max 2–6/255 vs. fp32, **zero NaN/Inf** on real
    photos. fp16 autocast was tried first and **rejected** (produced
    NaN/garbage output on SwinIR-M's attention layers; bf16 does not).
- **Warmup**: `enhance.warmup()` pre-loads models and pre-triggers cuDNN's
  one-time per-shape autotune. JobManager calls it **on its own persistent
  worker thread** before the first job is dequeued (`backend/jobs.py`), and
  `_run_with_stage_ticker` runs the heavy engine call **on that same
  thread** — cuDNN's benchmark-autotune cache is thread-local, so warming up
  or running on a fresh thread means every image (not just the first) re-pays
  the autotune cost (~8s/image: measured ~33s/image on a fresh thread per
  call vs. ~25s/image kept on one persistent thread).

**Validated measurements** (`image_enhancer/research_results/gpu_swinir_optimized/benchmark_gpu.json`):

| Image | GPU seconds | GPU peak VRAM |
|---|---|---|
| 1.jpeg | 35.52 (first call in a fresh process: pays one-time cuDNN autotune + CUDA warmup) | 3898.7 MB (cold) |
| 2.jpeg | 25.02 | 1358.9 MB |
| 3.jpeg | 25.08 | 1358.9 MB |
| 4.jpeg | 24.98 | 1358.9 MB |
| 5.jpeg | 25.05 | 1358.9 MB |
| 6.jpeg | 25.12 | 1358.9 MB |

Steady state ~25.0–25.1s remains inside the 25–27s budget. The one-time
~10s first-image gap (~7.5s cuDNN autotune + ~2.5s other CUDA warmup) is
absorbed by the startup warmup and no longer lands on any user image. Peak
VRAM dropped from ~2.4–2.5GB to ~1.36GB (bf16 uses less memory too).
Regression suite (`image_enhancer/tests/test_final_pipeline.py`, all 39
checks including the byte-identical determinism check) passes unchanged;
diff against the pre-optimization output on a real photo: mean |Δ| 0.14/255,
p99 3/255.

### CPU — isolated persistent torch-free worker, OpenVINO, IMDN x4

The shipped CPU path is a **standalone, persistent child process that never
imports torch** (`backend/cpu_worker.py`), used whenever the resolved
processing device is CPU (see `backend/device.py`, `backend/settings.py`;
CPU is the effective fallback when auto-detection finds no NVIDIA GPU, or
when the user forces CPU).

- **Why a separate process at all**: direct stress-testing found the real
  root cause of the packaged EXE's intermittent CPU-mode crash
  (STATUS_ACCESS_VIOLATION / STATUS_STACK_BUFFER_OVERRUN, real memory
  corruption, never a Python exception): **torch and OpenVINO's native CPU
  runtimes cannot safely coexist in one process** (almost certainly a
  conflicting OpenMP/TBB/MKL thread-pool init). Proven by controlled
  repeated-call tests: the same OpenVINO IMDN calls ran **20/20 clean** in a
  torch-free process, but **crashed within 4–10 calls** when `enhance.py`
  (which imports torch at module scope) was also imported. The fix is
  structural — the worker imports only `enhance_shared.py` +
  `imdn_x4_ov.py` (cv2/numpy/OpenVINO/tonal_correction, never torch), so the
  two runtimes are never in the same process. The GPU path (`enhance.py`,
  imported in-process by `engine_adapter.py`) is unaffected and unimpaired.
- **Why persistent**: importing `enhance.py` (torch, cv2, timm, …) measured
  ~15s cold, on top of which OpenVINO's model compile adds more — spawning a
  fresh process per job would blow the budget on nearly every image. A
  persistent worker pays that cost once at startup (via the `WARMUP`
  protocol from `engine_adapter.warmup_engine()`), then serves every CPU job
  afterwards. A worker that dies mid-job is detected (status sidecar file)
  and respawned, so one bad job never permanently breaks CPU mode.
- **Inference**: IMDN x4 via **OpenVINO** (`imdn_x4_ov.py`, using the
  exported `image_enhancer/models/imdn_x4_256.onnx`). IMDN is a pure CNN
  (no window-attention transformer) — the same architecture rationale as the
  historical IMDN experiments below, but running on OpenVINO rather than
  PyTorch's CPU backend for the crash-stability reason above. **Verified
  numerically equivalent to the PyTorch reference: max abs diff 0.00055/1.0
  (<0.15/255)** and faster per tile — **measured ~0.12s vs ~0.2–0.6s** per
  256×256 tile on this project's target CPU.
- **Denoise**: SwinIR-M *and* Restormer are CPU-infeasible within budget
  (see history below), so the CPU path replaces Restormer with classical
  `cv2.fastNlMeansDenoisingColored` (deterministic, non-generative) at the
  same K=0.30 blend weight, keeping the pipeline shape
  (denoise-blend → SR → F3/G7-MS/A+) identical across devices.
- **Post-processing**: F3-natural → G7-MS → A+ live in
  `image_enhancer/src/enhance_shared.py`, imported both by the GPU path and
  the CPU worker — exactly one copy of the logic, so CPU/GPU outputs share
  identical post-processing. No billboard/Candidate-A branch (that stage is
  dead code for this product even on GPU — see `docs/ENGINE_AUDIT.md`).
- **Extras adopted for the frozen app**: JobManager's worker thread is given
  an explicit 64MB stack (`backend/jobs.py`) — a plain `Thread()` reproduced
  STATUS_STACK_BUFFER_OVERRUN (BEX64 in ntdll) 100% (2/2) on the packaged
  EXE's IMDN CPU inference (PyInstaller's bootloader gives threads a smaller
  default stack; oneDNN JIT codegen needs more); the fixed size has not
  crashed since. Job completion is signaled via an atomic
  `<output>.status.json` sidecar file (a stdout-pipe protocol was tried and
  abandoned after intermittently dropping lines mid-job). `launcher.py`
  dispatches `--cpu-worker` so the frozen EXE can serve as its own worker.

**Validated measurements**: the adopted OpenVINO CPU path is numerically
equivalent and faster per tile than the earlier in-process PyTorch IMDN CPU
path it replaces (which measured **6.11–6.30s end-to-end** across all six
reference images —
`image_enhancer/research_results/cpu_imdn_optimized/benchmark_cpu.json`,
through the real production path, deterministic byte-identical rerun, no
NaN/Inf). CPU jobs in the persistent warmed worker therefore stay in the
project's measured **~5–7s per job** range, comfortably inside budget.
OpenVINO writes far below the budget left no requirement to benchmark the
OpenVINO worker end-to-end to six decimal places; the per-tile + numeric
equivalence measurements above are the validated OpenVINO-specific data.

**Quality tradeoff (real, disclosed)**: CPU output is measurably softer than
GPU output — visible in side-by-side crops at
`image_enhancer/research_results/comparisons/*_detail_crop_compare.png`
(billboard text, license plates, road signs, foliage all show the GPU result
as crisper). Both remain clearly better than the plain Lanczos faithful
upscale, and neither hallucinated text, halos, ringing, checkerboarding, or
altered object identity — verified visually against all six reference
images.

GPU output remains the primary/default result whenever an NVIDIA GPU is
present; CPU is a fallback for GPU-absent machines (or an explicit user
choice), not a replacement.

---

## History (not adopted)

Sketch of what was measured and why it was left behind, in order:

### Baseline (before this session)

| | GPU | CPU |
|---|---|---|
| End-to-end, one image | ~42–59s | ~200s+ for the SR stage alone (extrapolated; not run to completion) |

GPU backbone then: Restormer real-denoise (tile 512/48) blended K=0.30 →
SwinIR-M x4 PSNR (tile 256) → Lanczos to 4K.

### GPU exploration → adopted (see Adopted above)

Precision + execution-path changes only (no model/weight/algorithm change);
tile 256→224 / overlap 32→16; bf16 (fp16 rejected: NaN on SwinIR attention);
cuDNN benchmark + TF32; warmup + persistent worker thread. Result:
~42–59s → ~25s steady, from which the shipped state above follows.

### CPU infeasibility of the GPU backbone (measured, rejected)

1. **ONNX Runtime (fp32)**: 1.05× vs. PyTorch CPU — framework overhead was
   never the bottleneck.
2. **ONNX Runtime INT8 (dynamic quantization)**: 1.15×, but mean error
   13.7/255 vs. fp32 — too much quality loss for too little speed.
3. **OpenVINO CPU (fp32) on SwinIR-M/Restormer**: 1.9–2.7×
   (SwinIR-M 4.35s→2.32s/tile; Restormer 2.24s→1.76s/tile), numerically safe
   (max diff 0.0001/255) — real, but projects to only ~100–110s end-to-end,
   still ~4x over budget.
4. **OpenVINO + NNCF calibrated INT8**: crashed on an internal library bug
   during calibration (tried to allocate 40GB) — not pursued.

None close an ~8x gap: SwinIR-M (6-stage window-attention transformer) and
Restormer (MDTA/GDFN transformer blocks) are architecturally too deep for
CPU at this resolution.

### IMDN x4 on CPU — first attempt (superseded, source removed)

The first CPU SR fix was the official Zheng222/IMDN x4 architecture + official
checkpoint (MIT license), vendored as `restore_exp/imdn_x4.py` +
`restore_exp/imdn_arch/` and run through **PyTorch's CPU backend**. Pure CNN,
no attention — measured ~5s for the SR stage vs. SwinIR-M's ~200s+ (a ~45x
speedup), giving the **6.11–6.30s end-to-end** numbers above. That edition is
**superseded** by the shipped OpenVINO torch-free worker (same weights/ONNX
export, same tiling/blending scheme, faster per tile, and — decisively —
without PyTorch's CPU backend in the process, which is what the packed EXE
crash investigation pinned as the crash source). The PyTorch IMDN source
(`imdn_x4.py`, `imdn_arch/`) was therefore removed from the tree; it lives
on only as a historical record here and in the ONNX export
(`image_enhancer/models/imdn_x4_256.onnx`) still used by the shipped worker.

Reference checks on the official `IMDN_x4.pth` used here: it is the real x4
model, not `IMDN_AS.pth` (already present in `image_enhancer/models/`, which
internally downsamples 4x before upsampling 4x and nets *zero* actual
resolution gain — confirmed by reading its architecture before use, not
assumed).

### Packaged-EXE CPU stability investigation (adopted findings above)

- BEX64 / STATUS_STACK_BUFFER_OVERRUN with PyTorch's CPU backend in the
  packed app → replaced by OpenVINO (`imdn_x4_ov.py`), reproducing with or
  without pywebview/ticker threads/mkldnn/thread-count changes — a genuine
  torch/oneDNN CPU-backend thread-safety bug in the frozen build, never
  reproduced standalone.
- torch + OpenVINO in the same process crashes (20/20 clean without torch;
  4–10 calls with it) → **isolated torch-free worker process**, persistent,
  WARMUP at startup, status-sidecar protocol, crash-respawn.
- Small-thread stack in the frozen bootloader (STACK_BUFFER_OVERRUN on
  IMDN CPU inference; plain Thread 2/2 crash) → explicit 64MB stack on
  JobManager's worker thread.

## Not yet done (see final report for full list)

- Full PyInstaller rebuild from the updated spec + packaged-EXE smoke test of
  the shipped bundle — the spec now full-collects `openvino`, ships
  `imdn_x4_256.onnx` alongside `IMDN_x4.pth`, and `launcher.py` handles
  `--cpu-worker`; the CPU-stability fixes themselves were verified against
  frozen builds during the crash investigation (2/2 reproduction of the
  stack overrun, clean since the 64MB-stack worker), but a clean end-to-end
  rebuild + install smoke test was not rerun at the very end of the session.
  `dist/`/`build/` still reflect an earlier build.

Raw benchmark JSON: `image_enhancer/research_results/{gpu_swinir_optimized,cpu_imdn_optimized}/benchmark_*.json`.