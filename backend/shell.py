"""pywebview desktop shell — proof of concept.

Starts the existing local Application/API layer (backend.server) on an
OS-assigned loopback port, serving the built frontend (frontend/dist) as
static files from that SAME server — so the page's own relative
``fetch("/api/...")`` and ``fetch("/health")`` calls resolve against the
same origin with no CORS involved — then opens a single native window via
pywebview pointed at that local URL. See docs/ARCHITECTURE.md for why
pywebview was chosen over Electron/Tauri for this project.

This is the shell only: no image-enhancer UI work, no packaging, no change
to the API contract (server.py's static-file serving is opt-in and
additive — see its module docstring).

Run it:

    image_enhancer\\.venv\\Scripts\\python.exe -m backend.shell

Requires (Windows): the Microsoft Edge WebView2 Runtime, which ships
pre-installed on Windows 11 and on most up-to-date Windows 10 systems. If
it's missing, pywebview will fail to open a window — install it from
https://developer.microsoft.com/microsoft-edge/webview2/ (the
"Evergreen Bootstrapper" is enough) and re-run.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from . import jobs, recent_history  # noqa: E402
from ._frozen import app_root  # noqa: E402
from .server import make_server  # noqa: E402

REPO_ROOT = app_root()
FRONTEND_DIST = REPO_ROOT / "frontend" / "dist"


class DesktopBridge:
    """Exposed to the loaded page as ``window.pywebview.api`` (see
    ``webview.create_window(..., js_api=...)`` in main()).

    The one thing a browser tab genuinely cannot do that this desktop shell
    can: let the user pick where a finished result is saved via the OS's own
    native Save As dialog. Reads job state directly from the in-process
    JobManager (this shell and the API server share one Python process —
    see docs/ARCHITECTURE.md) rather than round-tripping through HTTP.
    """

    def __init__(self, job_manager: jobs.JobManager):
        self._job_manager = job_manager

    def save_result(self, job_id: str) -> dict:
        return self.save_result_as(job_id, "png")

    # Formats the single-image billboard-editor Save dialog offers. ZIP
    # (batch) jobs never reach this -- they always save via the "png" path
    # below, which is a byte-identical copy of the zip, same as before this
    # method existed (see save_result_as's own docstring).
    _IMAGE_SAVE_FORMATS = {
        "png": ("PNG image (*.png)", ".png"),
        "jpg": ("JPEG image (*.jpg)", ".jpg"),
        "jpeg": ("JPEG image (*.jpeg)", ".jpeg"),
    }

    def save_result_as(
        self,
        job_id: str,
        image_format: str = "png",
        billboard_rects: Optional[list] = None,
    ) -> dict:
        """Opens the native Save As dialog for a completed job's result.

        ``image_format`` is one of "png"/"jpg"/"jpeg" -- ignored for batch
        (ZIP) jobs, which always save as-is. For a single-image job whose
        result is already a PNG (the enhancement engine's only output
        format -- see image_enhancer/src/enhance.py's save_image), "png"
        with NO billboard rects copies the file byte-for-byte (unchanged
        from the original save_result behavior); "jpg"/"jpeg" re-encode it
        via cv2 at a high quality setting.

        ``billboard_rects`` (the frontend's confirmed BillboardRect list, in
        display order) is what makes a save leave the byte-identical path:
        when it's non-empty the rects are composited onto a *copy* of the
        engine's finished PNG by backend/billboard_overlay.py before writing
        the chosen destination. The engine's own output file is never
        touched, no inference runs here, and nothing is rasterized
        frontend-side -- the rects travel as plain coordinates and OpenCV
        draws them here at save time. Malformed/out-of-bounds rects are
        clamped or dropped (see billboard_overlay.clamp_billboard_rects), so
        a bad payload can never produce a broken save.
        """
        job = self._job_manager.get_job(job_id)
        if job is None:
            return {"ok": False, "error": "job not found"}

        status = job.to_status_dict()
        if status["status"] != jobs.STATUS_COMPLETED:
            return {"ok": False, "error": f"job is not completed (status: {status['status']})"}
        if job.result_path is None or not Path(job.result_path).is_file():
            return {"ok": False, "error": "job completed but no result file was found"}

        result_path = Path(job.result_path)
        # pywebview hands over the rect list as actual Python objects; a JSON
        # string only arrives from a non-bridge caller, but tolerate it rather
        # than crash on a malformed payload. Anything that isn't a list is
        # treated as "no billboards" and keeps the byte-identical PNG path.
        if isinstance(billboard_rects, str):
            try:
                billboard_rects = json.loads(billboard_rects)
            except (ValueError, json.JSONDecodeError):
                billboard_rects = None
        if not isinstance(billboard_rects, list):
            billboard_rects = None
        is_zip = result_path.suffix == ".zip"
        fmt = image_format if image_format in self._IMAGE_SAVE_FORMATS else "png"
        suggested_ext = ".zip" if is_zip else self._IMAGE_SAVE_FORMATS[fmt][1]
        suggested_name = (
            job.result_filename or result_path.name
        )
        if not is_zip:
            suggested_name = str(Path(suggested_name).with_suffix(suggested_ext))
        file_types = (
            ("ZIP archive (*.zip)",) if is_zip else (self._IMAGE_SAVE_FORMATS[fmt][0],)
        )

        import webview  # already verified importable by main() before this bridge exists

        window = webview.windows[0]
        destination = window.create_file_dialog(
            webview.SAVE_DIALOG,
            save_filename=suggested_name,
            file_types=file_types,
        )
        if not destination:
            return {"ok": False, "cancelled": True}
        dest_path = destination if isinstance(destination, str) else destination[0]

        try:
            if is_zip or (fmt == "png" and not billboard_rects):
                shutil.copyfile(result_path, dest_path)
            else:
                from .billboard_overlay import composite_result

                ok, error = composite_result(
                    result_path, dest_path, billboard_rects or [], fmt, quality=95)
                if not ok:
                    return {"ok": False, "error": error}
        except OSError as exc:
            return {"ok": False, "error": f"could not save file: {exc}"}
        return {"ok": True, "path": str(dest_path)}

    def save_batch_export(
        self,
        job_id: str,
        image_format: str = "png",
        include_outlines: bool = True,
        images: Optional[list] = None,
    ) -> dict:
        """Opens the native Save As dialog for a fresh batch export ZIP.

        Mirrors save_result_as's rect-tolerance and error handling, but
        builds a brand-new ZIP (backend/output_manager.build_batch_export)
        from each image's OWN rects rather than copying the job's existing
        result.zip — that raw zip never has overlays and only exists in one
        format. ``images`` is a list of {"result_id": str, "rects": [...]}
        dicts, one per image the gallery should export; a stringified JSON
        payload is tolerated the same way billboard_rects is elsewhere.
        """
        job = self._job_manager.get_job(job_id)
        if job is None:
            return {"ok": False, "error": "job not found"}

        status = job.to_status_dict()
        if status["status"] != jobs.STATUS_COMPLETED:
            return {"ok": False, "error": f"job is not completed (status: {status['status']})"}

        if isinstance(images, str):
            try:
                images = json.loads(images)
            except (ValueError, json.JSONDecodeError):
                images = None
        if not isinstance(images, list) or not images:
            return {"ok": False, "error": "no images to export"}

        fmt = image_format if image_format in self._IMAGE_SAVE_FORMATS else "png"
        items = []
        for entry in images:
            if not isinstance(entry, dict):
                continue
            result_id = entry.get("result_id")
            if not isinstance(result_id, str):
                continue
            file_result = job.get_file_result(result_id)
            if file_result is None:
                continue
            rects = entry.get("rects") or []
            items.append((file_result.original_filename, file_result.output_path, rects))
        if not items:
            return {"ok": False, "error": "no matching images to export"}

        from . import workspace
        from .output_manager import build_batch_export

        try:
            zip_path = build_batch_export(
                items, workspace.export_zip_path(job.id), fmt, bool(include_outlines))
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"could not build the export: {exc}"}

        import webview  # already verified importable by main() before this bridge exists

        window = webview.windows[0]
        suggested_name = f"enhanced_images_{job.id[:8]}.zip"
        destination = window.create_file_dialog(
            webview.SAVE_DIALOG,
            save_filename=suggested_name,
            file_types=("ZIP archive (*.zip)",),
        )
        if not destination:
            return {"ok": False, "cancelled": True}
        dest_path = destination if isinstance(destination, str) else destination[0]

        try:
            shutil.copyfile(zip_path, dest_path)
        except OSError as exc:
            return {"ok": False, "error": f"could not save file: {exc}"}
        return {"ok": True, "path": str(dest_path)}

    # ------------------------------------------------------- Recent history
    # The frontend's "Recent" tab needs to survive an app restart, but this
    # shell binds to a fresh ephemeral port every launch (start_backend's
    # port=0), so the page's own origin — and any localStorage tied to it —
    # changes every time. These three methods let the frontend persist and
    # read back a lightweight record of what was ACTUALLY saved (never an
    # assumed Downloads/Desktop/Documents path) via recent_history.py's JSON
    # file instead.

    def record_saved_result(self, entry: dict) -> dict:
        """Called right after a successful save_result() with the real
        destination path the user picked, plus whatever lightweight display
        metadata the frontend already has on hand (a small thumbnail data
        URL for a single image, or a file count for a ZIP) — never the
        full-size image bytes."""
        path = entry.get("path")
        if not path:
            return {"ok": False, "error": "missing path"}
        p = Path(path)
        record = {
            "id": uuid.uuid4().hex,
            "filename": p.name,
            "path": str(p),
            "directory": str(p.parent),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "kind": entry.get("kind") or ("zip" if p.suffix.lower() == ".zip" else "image"),
            "file_count": entry.get("file_count"),
            "thumbnail_data_url": entry.get("thumbnail_data_url"),
        }
        return {"ok": True, "entries": recent_history.add(record)}

    def get_recent_history(self) -> dict:
        return {"ok": True, "entries": recent_history.load()}

    def open_in_explorer(self, path: str) -> dict:
        """Opens Windows File Explorer at ``path``'s directory, selecting
        the file itself when it still exists. The smallest bridge addition
        needed for the Recent tab's directory link — a browser tab has no
        way to do this at all, so there is no non-desktop fallback."""
        p = Path(path)
        directory = p if p.is_dir() else p.parent
        if not directory.is_dir():
            return {"ok": False, "error": "that folder no longer exists"}
        try:
            if p.is_file():
                subprocess.run(["explorer", f"/select,{p}"])
            else:
                os.startfile(str(directory))  # noqa: S606 - local desktop app
        except OSError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True}

    def open_saved_file(self, path: str) -> dict:
        """Opens a previously saved result with its default viewer, for the
        Recent tab's thumbnail/filename click."""
        p = Path(path)
        if not p.is_file():
            return {"ok": False, "error": "that file no longer exists"}
        try:
            os.startfile(str(p))  # noqa: S606 - local desktop app
        except OSError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True}


class ShellError(RuntimeError):
    """A startup problem safe to print directly to the user."""


def _wait_for_health(base_url: str, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    last_exc: Optional[Exception] = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=1) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, OSError) as exc:  # noqa: PERF203
            last_exc = exc
        time.sleep(0.1)
    raise ShellError(f"backend did not become healthy within {timeout}s: {last_exc}")


def start_backend(port: int = 0):
    """Starts the real Application/API layer in a background thread, serving
    frontend/dist as static files from the same origin, and blocks until
    GET /health succeeds against it.

    Returns (httpd, thread, base_url). Caller must call stop_backend() when
    done — see main()'s window-closed handler for the normal path, and its
    finally block for the case where the window never got that far.
    """
    if not (FRONTEND_DIST / "index.html").is_file():
        raise ShellError(
            f"{FRONTEND_DIST} has no index.html — build the frontend first:\n"
            "    cd frontend && npm run build"
        )

    httpd = make_server(port=port, static_dir=FRONTEND_DIST)
    bound_port = httpd.server_address[1]
    base_url = f"http://127.0.0.1:{bound_port}"
    thread = threading.Thread(target=httpd.serve_forever, daemon=True, name="adinn-backend")
    thread.start()
    _wait_for_health(base_url)
    return httpd, thread, base_url


def stop_backend(httpd, thread) -> None:
    """Idempotent: safe to call more than once (e.g. from both the
    window-closed event and a finally block)."""
    try:
        httpd.shutdown()
        httpd.server_close()
    except Exception:  # noqa: BLE001 — best-effort on an already-stopped server
        pass
    thread.join(timeout=5)


def main() -> int:
    try:
        import webview
    except ImportError:
        print(
            "pywebview is not installed in this environment.\n"
            "Install it with: pip install pywebview\n"
            "On Windows this also pulls in pythonnet, and the OS needs the "
            "Edge WebView2 Runtime (see this module's docstring).",
            file=sys.stderr,
        )
        return 1

    try:
        httpd, thread, base_url = start_backend()
    except ShellError as exc:
        print(f"[adinn-shell] could not start: {exc}", file=sys.stderr)
        return 1

    print(f"[adinn-shell] backend ready at {base_url}")

    # Model/cuDNN warmup already started as soon as the backend's JobManager
    # was constructed inside start_backend() — see JobManager._run's own
    # comment for why it must happen on JobManager's own persistent worker
    # thread specifically (not a separate thread here) for the win to carry
    # over to real jobs.

    bridge = DesktopBridge(httpd.job_manager)  # type: ignore[attr-defined]
    window = webview.create_window(
        "Adinn 4K Image Enhancer",
        url=base_url,
        width=1200,
        height=800,
        min_size=(900, 600),
        js_api=bridge,
    )

    stopped = threading.Event()

    def _on_closed() -> None:
        if stopped.is_set():
            return
        stopped.set()
        print("[adinn-shell] window closed, shutting down backend...")
        stop_backend(httpd, thread)

    window.events.closed += _on_closed

    try:
        # Blocks until the window is closed. gui=None lets pywebview pick the
        # best native backend for the OS (edgechromium/WebView2 on Windows);
        # it never falls back to opening a system browser tab.
        webview.start()
    finally:
        if not stopped.is_set():
            stopped.set()
            stop_backend(httpd, thread)

    return 0


if __name__ == "__main__":
    sys.exit(main())
