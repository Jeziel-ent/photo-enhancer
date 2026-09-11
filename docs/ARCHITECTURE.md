# Architecture — Adinn 4K Image Enhancer (desktop)

This is desktop software, not a website. The user selects images on their
own machine, an AI pipeline enhances them using the local GPU, and the
result is saved back to disk (single file or ZIP) — no server, no account,
no external AI API.

## Layers

```
Desktop UI  →  Application/API layer  →  Image Enhancement Engine  →  Local AI Models/GPU
                                                     ↓
                                              Output Manager
```

| Layer | Lives in | Responsibility |
|---|---|---|
| Desktop UI | `frontend/` | React/Vite/Tailwind UI, rendered inside the desktop shell's native window. Select images, show queue/progress/preview, trigger save. No business logic. |
| Application/API layer | `backend/` | Local-only HTTP API (loopback, no network exposure). Validates uploads, owns the processing queue, calls the engine, hands finished files to the Output Manager. Successor to the archived `backend/server.py`, with all PPT/deck logic removed. |
| Image Enhancement Engine | `image_enhancer/` | `enhance.py` + the validated R&D recipes (g1–g9, restore/swinir/etc.). Pure image-in, image-out. No knowledge of the UI, HTTP, or files on disk beyond what it's given. |
| Local AI Models/GPU | `image_enhancer/models/` + `enhance.get_realesrgan()` etc. | Model weights and CUDA/PyTorch execution. Already local-only (RealESRGAN, SwinIR, Restormer checkpoints on disk; no external API calls). |
| Output Manager | `backend/` (new module, not yet written) | Single enhanced image → direct file save. Multiple → ZIP packaging. Owns the user's chosen output folder. |

The engine layer is intentionally kept ignorant of everything above it (already true today — `enhance.py` takes an image array and returns one). That is what lets it be called from a desktop shell, a CLI, or a test script without modification.

## Desktop shell: recommendation and why

The engine is CUDA/PyTorch Python with multi-GB model weights (RealESRGAN, SwinIR, Restormer) — it must run as a real local Python process no matter what UI shell is chosen; it cannot be ported to Node or Rust without reimplementing the models. The frontend, on the other hand, is already a substantial, generic React + Tailwind UI kit (`Button`, `Card`, `Field`, `EmptyState`, layout shell, etc.) worth keeping. So the real decision is only: *what hosts that existing frontend and talks to the Python engine?*

**Recommendation: [pywebview](https://pywebview.flowrl.com/)**, hosting the built `frontend/dist` in a native window, backed by a local HTTP API (same shape as the already-archived `backend/server.py`) run in a background thread of the *same* Python process.

Why, specifically for this codebase:

- **One runtime, not two.** pywebview is Python, so the desktop shell and the engine share one process/interpreter. No subprocess spawning, no port/health-check handshake, no orphan-process cleanup on crash — problems Electron/Tauri both introduce because their shell (Node or Rust) is a different runtime than the Python engine and must talk to it as a sidecar.
- **No new toolchain.** The repo today is Python + TypeScript, nothing else. Tauri would add Rust with zero existing footprint in this repo; Electron would add a second full JS runtime (Node main process) alongside the one already used for the Vite-built frontend. pywebview adds neither.
- **Reuses both halves of existing work as-is.** The frontend keeps calling `fetch("/api/...")` exactly like it does today (see `frontend/vite.config.ts` proxying `/api` to `127.0.0.1:8787`) — pywebview just starts that same local server and points its window at the built frontend instead of the Vite dev server. The already-proven pattern in the archived `backend/server.py` (in-process `import enhance`, a lock serializing calls onto the one GPU) carries over unchanged.
- **Smaller install on the actual target OS.** This project is being developed and will initially ship on Windows, which has shipped Edge WebView2 by default since Windows 10 1803 / all of Windows 11. pywebview uses that OS-provided webview instead of bundling Chromium, so the installer doesn't duplicate a browser engine on top of the multi-GB CUDA/model payload the way Electron would.
- **Packaging pain is the same either way, so it isn't a differentiator.** Freezing PyTorch + CUDA + basicsr/realesrgan into a distributable (PyInstaller or similar) is required whether the shell is Electron, Tauri, or pywebview — none of them make that step easier or harder.

Electron remains the fallback if a real need shows up later that pywebview can't cleanly satisfy (auto-update infrastructure, multiple native windows, deep OS shell integration). Tauri was deprioritized: it would add Rust purely to host a window, and its "sidecar" pattern for bundling the Python engine has to solve the exact same CUDA-freezing problem as the other two options without adding a corresponding benefit here. A pure-Python UI toolkit (PySide6/Qt) was rejected because it throws away the existing React/Tailwind frontend for no functional gain, which conflicts directly with "minimal changes, keep it reusable."

This keeps the boundary swappable: because the API layer talks plain local HTTP, moving to Electron or Tauri later (if ever needed) means replacing only the shell, not the API layer or the engine.

## Processing model

`enhance.get_realesrgan()` caches the loaded model at module scope and the archived backend already serialized calls through a lock because PowerPoint COM and Phase 1's bbox/env-var channel weren't concurrency-safe. The same constraint applies here for a different reason: there is one GPU (RTX 3050, 8 GB). The processing queue in the Application/API layer should run one enhancement job at a time against a single long-lived Python process holding the models in memory — not one process per image, not parallel GPU jobs.

## Proposed folder structure

```
adinn-4k-image-enhancer/
├── backend/                 Application/API layer + Output Manager + desktop shell
│   ├── server.py            local HTTP API — upload/validate/enqueue/status/download, no PPT concepts
│   ├── jobs.py              single-worker processing queue (one GPU, one job at a time)
│   ├── engine_adapter.py    the only module that imports image_enhancer/src/enhance.py
│   ├── output_manager.py    single-file save + ZIP packaging
│   ├── workspace.py         per-job input/output directories (gitignored)
│   ├── shell.py             pywebview entry point: starts the API thread, opens the window (proof of concept)
│   ├── shell_smoke.py       automated verification of the shell integration
│   └── tests/, tests_integration/
├── frontend/                 Desktop UI (React/Vite/Tailwind), built and loaded by backend/shell.py
│   └── src/                 upload, queue/progress, preview, save — not yet implemented (placeholder page only)
├── image_enhancer/           Image Enhancement Engine (unchanged) + local AI models
│   ├── src/                  enhance.py (production entry point) + f2/f3/g1-g9/pro/restore/swinir/f3_gate_exp R&D
│   ├── tests/
│   └── models/                RealESRGAN/SwinIR/Restormer weights (gitignored)
├── docs/
│   └── ARCHITECTURE.md       this file
└── README.md
```

`backend/shell.py` + `backend/shell_smoke.py` exist now as a proof of concept (see `backend/README.md` for how to run/verify them): the real API layer, running on a background thread, serving the real `frontend/dist` build from the same origin, in one native pywebview window. Output-folder selection, packaging, and the actual enhancer UI are still not implemented — that's the next pass, not this one.

## Remaining decisions (not settled here)

- Exact API layer framework: keep the dependency-free `http.server` pattern (current), or move to something like FastAPI for request validation/typing. Either works under pywebview.
- Packaging/distribution tool (PyInstaller vs. Briefcase vs. something else) and how the multi-GB model weights ship (bundled vs. downloaded on first run) — `backend.shell` currently only runs from source, against a `frontend/dist` the developer builds by hand.
- Whether `react-router-dom` is still useful for a single-window desktop app, or whether view switching (upload → queue → results) should just be local component state.
- Multi-GPU / no-GPU (CPU fallback) handling — `enhance.py` already falls back to CPU (`torch.cuda.is_available()`), but the queue/UI should surface that to the user rather than silently running slow.
- macOS/Linux support — pywebview supports both, but the CUDA/model story and PowerShell-based dev workflow in this repo are currently Windows-first; cross-platform packaging is unverified. The shell's own logic (`shell.py`) is not Windows-specific, only the WebView2 runtime requirement is.
- `evaluate_js`'s direct return value did not reliably carry a resolved Promise on the installed pywebview 6.2.1 + WebView2 combination (came back as an empty object) even though it's documented to; its `callback` parameter does resolve correctly and is what `shell_smoke.py` uses. Worth re-checking on future pywebview versions.
