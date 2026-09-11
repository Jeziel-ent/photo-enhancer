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

import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIST = REPO_ROOT / "frontend" / "dist"

from .server import make_server  # noqa: E402


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

    window = webview.create_window(
        "Adinn 4K Image Enhancer",
        url=base_url,
        width=1200,
        height=800,
        min_size=(900, 600),
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
