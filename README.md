# Adinn 4K Image Enhancer

**Desktop software**, not a website. The user picks one or more images on
their own machine, a local AI pipeline enhances them (GPU when available,
CPU otherwise), and the result is saved back to disk — a single file for
one input, or a ZIP for multiple inputs. No server, no account, no
external AI API; every job runs on the local loopback API and the local
GPU/CPU only.

**Core principle:** the selected photograph is the source of truth.
Enhancement only restores, denoises, deblurs, and upscales — it must never
invent, replace, remove, or hallucinate content (text, logos, faces, people,
vehicles, buildings, roads, signs, objects, geometry, surroundings). See
[`docs/ENGINE_AUDIT.md`](docs/ENGINE_AUDIT.md) and
[`docs/MVP2_RESEARCH.md`](docs/MVP2_RESEARCH.md) for the engine's full
tuning/research history, and [`docs/RELEASE_READINESS.md`](docs/RELEASE_READINESS.md)
for the current release-candidate status.

## What it does

1. Drag-and-drop or pick one or more `.jpg`/`.jpeg`/`.png` photos (up to 50
   per job, 40 MB per file).
2. The automatic engine (Restormer denoise + SwinIR-M ×4 super-resolution +
   an edge-gated fidelity-preserving detail fusion stage, see
   [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)) enhances each one to a
   4K-longest-side output, preserving the source's own aspect ratio exactly
   (no stretch, no crop).
3. Review the result against the original with a compare slider, optionally
   mark billboard/board areas to keep crisp, and fine-tune with manual
   Brightness/Contrast/Highlights/Shadows/Saturation/Detail sliders — each
   image in a batch keeps its own independent slider state.
4. Save: a native Windows "Save As" for one image, or a ZIP export for a
   batch — each image exported with only its own board outlines and its
   own adjustment values, never another image's.
5. Previously saved results are listed on the **Recent** page (persisted
   locally to disk, not browser storage) for quick re-access.

## Major features

- **Automatic 4K enhancement** — denoise, deblur, and AI super-resolution,
  fidelity-first (see the engine's core principle above).
- **Aspect-ratio preservation** — the output's longest side is 4K (3840px);
  the other side follows the source's own ratio, never forced to 16:9.
- **Manual adjustment controls** — Brightness, Contrast, Highlights,
  Shadows, Saturation, Detail; live client-side preview (no network
  round-trip while dragging — see `frontend/src/lib/previewAdjustments.ts`);
  the exact same backend pixel math is what actually gets saved, so the
  preview never drifts from the export.
- **Per-image adjustment state** — in a multi-image batch, each photo
  remembers its own slider values independently; switching the active
  image never leaks one photo's adjustments onto another's.
- **Board/billboard area editor** — draw rectangles over signage/board
  regions to keep them crisp in export; each image's rectangles are its own.
- **Batch export (Save ZIP)** — one shared output format for the whole
  batch, each image's own board outlines and own adjustments applied only
  to that image.
- **Recent results** — a locally persisted history of everything saved via
  the native Save As flow (survives app restarts; not tied to browser
  storage, since the local API's port changes every launch).
- **Settings / device selection** — GPU vs. CPU processing preference, with
  live detection of what's actually available (see GPU/CPU modes below).
- **Native Windows Save As** — real OS file-save dialogs when running as
  the packaged desktop app, not a browser download.
- **Local processing only** — every job runs against the local API
  (loopback only) and the local GPU/CPU; no image or file ever leaves the
  machine.

## GPU / CPU modes

The engine has two independent execution paths, chosen automatically (GPU
preferred when present) or forced via Settings:

- **GPU (NVIDIA/CUDA)** — Restormer real-denoise → SwinIR-M ×4 PSNR →
  fidelity-fusion stages, in-process, models cached across jobs on a
  single-worker queue (one GPU, one job at a time). GPU presence is
  detected via `nvidia-smi` (`backend/device.py`), deliberately *without*
  importing `torch`/initializing CUDA just to check — so device detection
  itself can never crash a GPU-less machine.
- **CPU (OpenVINO)** — a lighter IMDN ×4 super-resolution model via
  OpenVINO, run in a **separate, persistent child process**
  (`backend/cpu_worker.py`) that never imports PyTorch. This exists
  because the packaged/frozen EXE was found (via direct stress-testing) to
  segfault after repeated CPU jobs when OpenVINO and PyTorch's native
  runtimes shared one process — isolating CPU inference into its own
  process, with no `torch` import anywhere in that process's import graph,
  eliminates the conflict entirely. See `backend/cpu_worker.py`'s module
  docstring for the full root-cause writeup.

Both paths produce a real, non-generative 4K enhancement from the same
source photo; the CPU path is measurably softer (a smaller, faster model,
chosen to fit CPU-only runtime budgets) — this is a real, disclosed
quality/speed tradeoff, not a bug. See
[`docs/RELEASE_READINESS.md`](docs/RELEASE_READINESS.md) for measured
timings on both paths.

## Supported input formats

`.jpg` / `.jpeg` / `.png`, up to 40 MB per file, up to 50 files per job
(200 MB total per job) — see `frontend/src/lib/uploadValidation.ts` and the
matching server-side limits in `backend/server.py`.

## Output behavior

- Longest output side is fixed at 4K (3840px); the source's own aspect
  ratio is preserved exactly (`image_enhancer/src/enhance.py`'s
  `aspect_preserving_target`) — never stretched or cropped to 16:9.
- One input → one saved file (PNG or JPEG, user's choice).
- Multiple inputs → one ZIP, one file per image, disambiguated filenames.
- Manual adjustments and board outlines are applied only at
  preview/export time, on top of the engine's own output — the engine's
  saved file itself is never mutated by a slider or a board edit.

## Architecture overview

```
Frontend (React/Vite/Tailwind)
  → local HTTP API (backend/server.py, loopback only)
    → JobManager (backend/jobs.py, single-worker processing queue)
      → engine_adapter.py (the only module that imports the engine)
        → GPU path (in-process, models cached) OR
          CPU path (isolated persistent subprocess, backend/cpu_worker.py)
      → output_manager.py (single-file save / ZIP packaging)
      → recent_history.py (locally persisted save history)
  → backend/shell.py (pywebview desktop shell: native window + native
    Save As, wraps the same local API)
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full breakdown,
including why pywebview was chosen and the CPU isolated-process design.

## Development setup

```powershell
# Backend (from repo root, using the engine's own venv so torch/CUDA are on sys.path)
image_enhancer\.venv\Scripts\python.exe -m backend.server --port 8787

# Frontend (separate terminal)
cd frontend
npm install
npm run dev          # Vite dev server, proxies /api and /health to :8787
```

Run the full desktop shell (native window, same as the packaged app) with:

```powershell
image_enhancer\.venv\Scripts\python.exe -m backend.shell
```

See `backend/README.md` for the shell's setup/verification steps in detail.

## Testing

```powershell
# Backend (pytest; fake process_fn injected so these run in seconds, no GPU needed)
image_enhancer\.venv\Scripts\python.exe -m pytest backend/tests -q

# Frontend (Vitest; pure utility/math functions only — see docs/ARCHITECTURE.md's
# Testing section for why component-level tests are out of scope for now)
cd frontend
npm run typecheck
npm run test
npm run build
```

Engine regression coverage (actual enhancement quality/determinism, not the
HTTP contract) lives in `image_enhancer/tests/`.

## Packaging / installer

A full Windows installer build exists under `packaging/` — PyInstaller
(onedir bundle) → Inno Setup 6 (`.exe` installer), orchestrated end-to-end
by `packaging/build.ps1`. See `packaging/README.md` for build-machine
setup and layout, and [`docs/RELEASE_READINESS.md`](docs/RELEASE_READINESS.md)
for this build's actual verification status.

## Known hardware requirements / limitations

- **Windows only.** The desktop shell uses pywebview + the Edge WebView2
  runtime (preinstalled on Windows 11 and most up-to-date Windows 10).
  macOS/Linux are not packaged or verified.
- **GPU path**: developed and tuned against an RTX 3050-class card (8 GB
  VRAM); NVIDIA/CUDA only (no AMD/Intel GPU acceleration path).
- **CPU path** is real but noticeably slower and softer than GPU — see
  `docs/RELEASE_READINESS.md` for measured numbers; this is a disclosed
  tradeoff, not a defect.
- One GPU, one job at a time — the processing queue is intentionally
  single-worker; it does not parallelize multiple jobs across a multi-GPU
  machine.

## Layout

```
backend/            Application/API layer + desktop shell (see backend/README.md)
├── server.py       local HTTP API — jobs, results, batch export, settings, health
├── jobs.py         single-worker processing queue (one GPU/CPU, one job at a time)
├── engine_adapter.py  the only module that imports image_enhancer/src/enhance.py
├── cpu_worker.py   isolated persistent CPU-mode subprocess (never imports torch)
├── device.py       NVIDIA GPU detection + device-preference resolution
├── output_manager.py  single-file save / ZIP batch export packaging
├── recent_history.py  locally persisted "Recent" save history
├── shell.py        pywebview desktop shell — native window + native Save As
└── tests/, tests_integration/
frontend/           Desktop UI — React + Vite + Tailwind, rendered inside the desktop shell's window
├── src/components/upload/   upload, compare, board editor, adjustment controls, results
├── src/components/recent/   Recent page
├── src/components/settings/ Settings page (device selection)
└── src/lib/                 API client, adjustment math, validation
image_enhancer/     Image Enhancement Engine + local AI models + R&D
├── src/            enhance.py (production entry point) + experiment folders (f2/f3/g1-g9/pro/restore/swinir/research/...)
├── tests/          regression scripts for the production pipeline
└── models/         SwinIR/Restormer/OpenVINO weights (gitignored) — local, no external AI API
packaging/          PyInstaller + Inno Setup Windows installer build
docs/               architecture, engine-audit/research notes, release readiness
```
