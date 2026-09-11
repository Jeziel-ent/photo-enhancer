# backend/ — Application/API layer + desktop shell

Local-only HTTP API (loopback, 127.0.0.1) between the desktop UI and the
Image Enhancement Engine, plus the pywebview shell that hosts it as a
desktop window. See [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) for
the full layer breakdown and why pywebview was chosen.

## Modules

| File | Responsibility |
|---|---|
| `server.py` | HTTP routing (`POST /api/jobs`, `GET /api/jobs/<id>`, `GET /api/jobs/<id>/result`, `GET /health`), request validation, multipart parsing. Also optionally serves a built static frontend from the same origin (`make_server(static_dir=...)`) — additive, off by default, used only by `shell.py`. |
| `jobs.py` | Job state and the single-worker processing queue — one GPU, one job at a time, files within a job processed sequentially. |
| `engine_adapter.py` | The only module that imports `image_enhancer/src/enhance.py`. Also neutralizes the engine's PPT-era billboard-regions config so it can never fire on a generic photo — read the docstring on `_neutralize_billboard_regions` before touching this file. |
| `output_manager.py` | Packages a job's successful outputs: the single file for one input, a ZIP for multiple. |
| `multipart.py` | Minimal stdlib-only `multipart/form-data` parser. |
| `workspace.py` | Per-job input/output directories under `backend/.workspace/` (gitignored) — uploads are copied here; originals are never touched. |
| `shell.py` | pywebview desktop shell (proof of concept): starts `server.py` on a background thread serving `frontend/dist`, opens one native window pointed at it, shuts the backend down when the window closes. |
| `shell_smoke.py` | Automated verification that the shell actually works end to end — see "Desktop shell" below. |

## Running the API standalone (no window)

```powershell
# from the image_enhancer venv (needs its GPU/torch dependencies on sys.path)
..\image_enhancer\.venv\Scripts\python.exe -m backend.server --port 8787
```

## Desktop shell (pywebview)

### Setup

1. Build the frontend once (the shell serves the static build, not the Vite dev server):
   ```powershell
   cd frontend
   npm install
   npm run build
   cd ..
   ```
2. Install `pywebview` into the same environment the engine uses (it's in `backend/requirements.txt`; not required for the API/tests):
   ```powershell
   image_enhancer\.venv\Scripts\python.exe -m pip install pywebview
   ```
3. **Windows requirement:** the Microsoft Edge **WebView2 Runtime**. It's preinstalled on Windows 11 and on most up-to-date Windows 10 machines. If `backend.shell` fails to open a window, install it (the "Evergreen Bootstrapper" is enough): https://developer.microsoft.com/microsoft-edge/webview2/

### Run it

```powershell
image_enhancer\.venv\Scripts\python.exe -m backend.shell
```

This starts the real API on an OS-assigned loopback port, waits for `GET /health` to succeed, then opens a single native window loading `frontend/dist` from that same origin — so the page's own `fetch("/api/...")` calls work with no CORS setup, exactly as they will once the enhancer UI is built. No browser tab is ever opened; pywebview renders inside its own native window (WebView2 on Windows). Closing the window shuts the backend down and exits cleanly.

### Verify it

```powershell
image_enhancer\.venv\Scripts\python.exe -m backend.shell_smoke
```

Starts the real backend and a real window, then — from **inside the loaded page's own JS engine** (not from Python) — runs `fetch('/health')` and checks it resolves to `200`. This is the same-origin path React itself will use, so a pass here proves the shell integration, not just that the backend can be started. Prints `[smoke] PASSED: ...` and exits 0 on success.

## Tests

## Tests

```powershell
..\image_enhancer\.venv\Scripts\python.exe -m pytest backend/tests -q
```

The API/job-contract tests inject a fake `process_fn` into `JobManager` so
they run in seconds without a GPU. They prove the HTTP contract and queue
behavior, not the enhancement quality — that's `image_enhancer/tests/`.

## Not yet implemented

- The image-enhancer UI itself — `frontend/` still shows only the
  placeholder page from the cleanup pass; the shell loads it as-is.
- Packaging/distribution (PyInstaller or similar) — `backend.shell` only
  runs from source today.
- Output-folder selection / "save enhanced image(s) to..." — today the
  client must `GET .../result` and save the bytes itself.
- Job retention/cleanup — completed jobs and their `.workspace/` directories
  are kept forever; there's no TTL or eviction yet.
