# Architecture — Adinn 4K Image Enhancer (desktop)

This is desktop software, not a website. The user selects images on their
own machine, an AI pipeline enhances them using the local GPU (or CPU,
isolated as described below), and the result is saved back to disk (single
file or ZIP) — no server, no account, no external AI API.

This document describes the **current, implemented** architecture. For the
engine's tuning/research history see `ENGINE_AUDIT.md` and
`MVP2_RESEARCH.md`; for current verification status see
`RELEASE_READINESS.md`.

## Layers

```
Frontend (Desktop UI)
  → local API (backend/server.py, loopback only)
    → JobManager (backend/jobs.py — single-worker processing queue)
      → engine_adapter.py (the only module that imports the engine)
        → GPU execution path (in-process, models cached)   OR
          CPU execution path (isolated persistent subprocess)
      → Output Manager (backend/output_manager.py)
      → Recent/export history (backend/recent_history.py)
  → Desktop shell (backend/shell.py — pywebview native window + native
    Save As, DesktopBridge JS<->Python calls)
```

| Layer | Lives in | Responsibility |
|---|---|---|
| Frontend | `frontend/` | React/Vite/Tailwind UI, rendered inside the desktop shell's native window: upload, progress, compare, board editor, adjustment sliders, export, Recent, Settings. Talks to the local API via `fetch("/api/...")`; when running inside the desktop shell, save/export instead go through `desktop.ts`'s `DesktopBridge` calls (native Save As, no browser download). |
| Application/API layer | `backend/` | Local-only HTTP API (loopback, no network exposure). Validates uploads, owns the processing queue, calls the engine, hands finished files to the Output Manager. |
| JobManager | `backend/jobs.py` | Single background worker thread; one job at a time, files within a job processed sequentially. Each file's processing exceptions are caught individually — one bad file fails only that file, never the whole job. |
| Engine adapter | `backend/engine_adapter.py` | The only module that imports `image_enhancer/src/enhance.py`. Dispatches to the GPU or CPU execution path (see below) based on the resolved device preference, and neutralizes the engine's legacy billboard-regions config so it can never fire on a generic photo. |
| Image Enhancement Engine | `image_enhancer/` | `enhance.py` (production entry point) + the full validated R&D history (`f2/f3/g1-g9/pro/restore/swinir/research/...`). Pure image-in, image-out; no knowledge of the UI, HTTP, or files on disk beyond what it's given. |
| Local AI Models | `image_enhancer/models/` | SwinIR-M (production SR backbone), Restormer (denoise), OpenVINO IMDN (CPU-path SR) checkpoints on disk — local-only, no external API calls, gitignored (too large for source control). |
| Output Manager | `backend/output_manager.py` | Single enhanced image → direct file save. Multiple → ZIP packaging, one file per image, disambiguated filenames. Also builds batch exports with per-image board outlines and per-image manual adjustments applied only to that image (`build_batch_export`). |
| Recent history | `backend/recent_history.py` | A JSON file on disk (not browser storage — the local API's port is OS-assigned and differs every launch, so page-origin-scoped storage would not survive an app restart) recording what the user actually saved via the native Save As flow. |
| Desktop shell | `backend/shell.py` | pywebview: starts the local API on a background thread, opens one native window loading the built `frontend/dist`, exposes `DesktopBridge` (native Save As, batch export, Recent, device settings) to the page's JS, shuts the backend down when the window closes. |

The engine layer is intentionally kept ignorant of everything above it —
`enhance.py` takes an image array and returns one. That is what lets it be
called from the desktop shell, a CLI, or a test script without
modification.

## GPU execution path

In-process: `engine_adapter.py` imports `image_enhancer/src/enhance.py`
directly, which runs Restormer real-denoise → a K=0.30 faithful blend →
SwinIR-M ×4 PSNR super-resolution (tiled for 8 GB-class VRAM) → a Lanczos4
resize to the aspect-preserving 4K target → the fidelity-fusion stages
(F3/G7/A+ — edge-gated, self-clipped detail recovery; see
`ENGINE_AUDIT.md` for why each stage exists and what was tried and
rejected). Models are cached at module scope and reused across jobs;
`JobManager`'s single-worker queue ensures only one job uses the GPU at a
time (one RTX-3050-class card, 8 GB VRAM — not designed for concurrent GPU
jobs).

GPU presence is detected via `backend/device.py`, which deliberately shells
out to `nvidia-smi` instead of importing `torch`/initializing a CUDA
context just to answer "is a GPU present" — so device detection itself can
never crash a GPU-less machine or pay CUDA init cost before a device
decision is even made.

## CPU execution path (isolated subprocess) — and why it exists

The CPU path does **not** run in the same process as the rest of the
backend. `backend/cpu_worker.py` runs as a standalone, persistent child
process (spawned via `python -m backend.cpu_worker`) that **never imports
PyTorch** — it uses OpenVINO's IMDN ×4 model only, a lighter super-resolution
model chosen specifically because it meets CPU-only runtime budgets that
SwinIR-M cannot on CPU hardware.

**Why the separate process, specifically:** the packaged/frozen Windows EXE
was found, by direct stress-testing (not guessed), to crash
(`STATUS_ACCESS_VIOLATION` — a genuine segfault, not a Python exception)
after repeated CPU-mode jobs. Two isolation steps narrowed the cause
precisely:

1. A plain script calling OpenVINO's IMDN inference alone, 20 times in a
   loop, with `torch` never imported in that process: 20/20 clean.
2. The **same** script, but importing `enhance` first (which imports
   `torch` at module scope) before doing the identical OpenVINO calls:
   crashed within 4-10 calls, every time.

The conclusion: OpenVINO's and PyTorch's native runtimes conflict when
loaded into the same process (a native-library interaction, not a Python
bug), and it only reproduces in the frozen/packaged EXE's binary layout —
not reliably from source. The fix is process isolation, not a
runtime/threading tweak: `backend.engine_adapter.enhance_image_cpu_subprocess`
talks to the persistent `cpu_worker.py` child process (`CpuWorkerHandle`)
over its own IPC channel, so PyTorch is never imported anywhere in that
process's import graph, for the lifetime of the process. The worker is
kept alive and reused across jobs (not spawned per image) to avoid paying
process-startup and model-load cost on every single photo.

Both paths — GPU and CPU — produce a real, non-generative 4K-longest-side,
aspect-preserving enhancement from the same source photo. The CPU path is
measurably softer (see `RELEASE_READINESS.md` for current measurements);
this is a disclosed, deliberate quality/speed tradeoff for CPU-only
machines, not a bug, and the CPU architecture is not "upgraded" to run the
GPU-path models merely to force output parity — see `README.md`'s GPU/CPU
modes section.

## Desktop shell

**pywebview**, hosting the built `frontend/dist` in a native window, backed
by the local HTTP API run in a background thread of the *same* Python
process. Rationale (still accurate, kept from the original design
decision):

- **One runtime, not two.** pywebview is Python, so the desktop shell and
  the engine share one process/interpreter — no subprocess spawning, no
  port/health-check handshake, no orphan-process cleanup on crash (a real
  problem Electron/Tauri introduce, since their shell is a different
  runtime than the Python engine).
- **No new toolchain.** The repo is Python + TypeScript; pywebview adds
  neither Rust (Tauri) nor a second JS runtime (Electron's Node main
  process).
- **Reuses the existing frontend as-is.** The frontend calls
  `fetch("/api/...")` exactly like it does under the Vite dev server;
  pywebview just starts the same local server and points its window at the
  built frontend instead.
- **Smaller install on Windows.** WebView2 ships with Windows 10 1803+ and
  all of Windows 11, so the installer doesn't bundle a second Chromium the
  way Electron would, on top of the already multi-GB CUDA/model payload.

`backend/shell.py` also exposes a `DesktopBridge` (see `backend/shell.py`
and `frontend/src/lib/desktop.ts`) giving the page's JS access to
OS-native functionality the plain HTTP API can't provide from a browser
sandbox: native Save As dialogs (single image and batch ZIP export),
persisted Recent-history reads, and live device-preference settings.
`isDesktopShell()` in the frontend detects whether this bridge is present
and switches between it and the plain `fetch`/browser-download path
accordingly — the same UI code serves both a plain-browser dev workflow
and the packaged desktop app.

## Processing model

One long-lived Python process holds the GPU-path models in memory
(cached at module scope); `JobManager`'s single-worker queue serializes
all GPU work onto that one process — not one process per image, not
parallel GPU jobs, matching the single-GPU (8 GB VRAM) target hardware.
The CPU path is deliberately the opposite: a separate, persistent-but-
isolated process, for the native-runtime-conflict reason described above,
not for performance.

## Testing

- **Backend**: `pytest` (`backend/tests/`) — the API/job-contract tests
  inject a fake `process_fn` into `JobManager` so they run in seconds
  without a GPU; they prove the HTTP contract, queue behavior, and
  output-packaging logic (including per-image batch-export adjustments),
  not enhancement quality. `backend/tests_integration/` exercises the real
  pipeline when a GPU is available. Engine-quality regression scripts live
  separately in `image_enhancer/tests/`.
- **Frontend**: `vitest` (`frontend/src/**/*.test.ts`), a deliberately
  minimal setup — plain Node, no jsdom, no React Testing Library. Coverage
  is scoped to **pure utility/math functions only**: the live-preview tone
  LUT and weight curves (`lib/previewAdjustments.ts`), the reset/no-op and
  export-payload-inclusion gate (`lib/adjustments.ts`'s
  `isDefaultAdjustments`), and the aspect-ratio/cover-crop geometry
  (`AdjustedPreviewCanvas.tsx`'s `previewSizeForLongestSide`/
  `coverSourceRect`). This is a deliberate scope decision, not an
  oversight: these pure functions are where a regression would silently
  corrupt every preview/export pixel, and they need nothing beyond plain
  Node to test exhaustively. **Not covered**: React component/hook-level
  behavior (e.g. `JobResultPanel.tsx`'s per-image adjustment state, active-
  image switching, slider reset UI) — verifying those properly needs
  jsdom + a DOM testing library, a real additional framework layer that
  was judged disproportionate for this pass; `npm run typecheck` and
  manual verification are the current safety net for that layer. Revisit
  if component-level regressions become a real, recurring pain point.
- `npm run build` (production Vite build) and `npm run typecheck` (`tsc -b
  --noEmit`) are both part of the standard verification pass.

## Folder structure

```
adinn-4k-image-enhancer/
├── backend/                 Application/API layer + Output Manager + desktop shell
│   ├── server.py            local HTTP API — jobs, results, batch export, settings, health
│   ├── jobs.py              single-worker processing queue (one GPU/CPU, one job at a time)
│   ├── engine_adapter.py    the only module that imports image_enhancer/src/enhance.py
│   ├── cpu_worker.py        isolated persistent CPU-mode subprocess (never imports torch)
│   ├── device.py            NVIDIA GPU detection (nvidia-smi) + device-preference resolution
│   ├── output_manager.py    single-file save + ZIP batch export packaging
│   ├── recent_history.py    locally persisted "Recent" save history (JSON on disk)
│   ├── adjustment_overlay.py  the one shared preview/export/batch adjustment-compositing implementation
│   ├── billboard_overlay.py board-rectangle drawing, shared the same way
│   ├── workspace.py         per-job input/output directories (gitignored)
│   ├── shell.py             pywebview entry point: starts the API thread, opens the window, exposes DesktopBridge
│   ├── shell_smoke.py       automated verification of the shell integration
│   └── tests/, tests_integration/
├── frontend/                 Desktop UI (React/Vite/Tailwind), built and loaded by backend/shell.py
│   └── src/                 upload, queue/progress, compare/preview, board editor, adjustments, export, Recent, Settings
├── image_enhancer/           Image Enhancement Engine (unchanged production path) + local AI models
│   ├── src/                  enhance.py (production entry point) + f2/f3/g1-g9/pro/restore/swinir/research/... R&D
│   ├── tests/
│   └── models/                SwinIR/Restormer/OpenVINO weights (gitignored)
├── packaging/                 PyInstaller + Inno Setup Windows installer build (see packaging/README.md)
├── docs/
│   ├── ARCHITECTURE.md        this file
│   ├── ENGINE_AUDIT.md        engine tuning history and rejected candidates
│   ├── MVP2_RESEARCH.md       MVP2 research findings
│   └── RELEASE_READINESS.md   current release-candidate verification status
└── README.md
```

## Known limitations / not yet settled

- **Windows only.** pywebview + WebView2 is the verified target; macOS/Linux
  packaging is unverified (pywebview itself supports both, but the CUDA
  story and the Windows-first dev/packaging workflow in this repo are not
  cross-platform-tested).
- **One GPU, one job at a time** — the queue is intentionally
  single-worker; it does not parallelize across a multi-GPU machine.
- **Job retention** — completed jobs and their `.workspace/` directories
  persist until manually cleared; there is no TTL/eviction yet.
- `evaluate_js`'s direct return value did not reliably carry a resolved
  Promise on the pywebview + WebView2 combination in use (came back as an
  empty object) even though it's documented to; its `callback` parameter
  resolves correctly and is what `shell_smoke.py`/`DesktopBridge` use.
  Worth re-checking on future pywebview versions.
