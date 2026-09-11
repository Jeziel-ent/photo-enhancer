"""Automated smoke test for the pywebview desktop shell.

Not a pytest test — opening a real native window doesn't fit the unit-test
harness (backend/tests stays GPU- and GUI-free). Run directly, on a machine
with a display and WebView2 available:

    image_enhancer\\.venv\\Scripts\\python.exe -m backend.shell_smoke

What it proves: starts the real backend (frontend/dist served as static
files + the real API), opens the real pywebview window pointed at it, and
evaluates ``fetch('/health')`` INSIDE that window's JS engine once the page
has loaded — the same environment React runs in — to prove the loaded page
can reach the backend on the same origin with a plain relative fetch. Then
closes the window and shuts the backend down cleanly. Exits 0 on success,
1 on any failure (import missing, backend didn't start, fetch failed, or
the check never ran).
"""

from __future__ import annotations

import sys
import threading
import time

from . import shell

_result: dict = {}
_done = threading.Event()


def _on_resolved(value) -> None:
    _result["health_status"] = value
    _done.set()


def _run_check(window) -> None:
    try:
        # evaluate_js's *return value* does not reliably carry a resolved
        # Promise on every backend (observed empty {} with pywebview 6.2.1's
        # edgechromium backend despite the promise-support docs) -- its
        # `callback` parameter does, so that's what actually proves the
        # fetch happened and resolved inside the loaded page.
        window.evaluate_js(
            "fetch('/health').then(r => r.status).catch(e => -1)",
            _on_resolved,
        )
        if not _done.wait(timeout=10):
            _result["error"] = "evaluate_js callback never fired within 10s"
    except Exception as exc:  # noqa: BLE001
        _result["error"] = str(exc)
        _done.set()
    finally:
        window.destroy()


def main() -> int:
    try:
        import webview
    except ImportError:
        print("pywebview is not installed — pip install pywebview", file=sys.stderr)
        return 1

    try:
        httpd, thread, base_url = shell.start_backend()
    except shell.ShellError as exc:
        print(f"[smoke] could not start backend: {exc}", file=sys.stderr)
        return 1

    print(f"[smoke] backend ready at {base_url}")

    window = webview.create_window(
        "Adinn shell smoke test", url=base_url, width=800, height=600)

    t0 = time.time()
    webview.start(_run_check, window)
    elapsed = time.time() - t0

    shell.stop_backend(httpd, thread)

    if not _done.is_set():
        print("[smoke] FAILED: window closed before the health check ran", file=sys.stderr)
        return 1
    if "error" in _result:
        print(f"[smoke] FAILED: {_result['error']}", file=sys.stderr)
        return 1

    status = _result.get("health_status")
    print(f"[smoke] fetch('/health').status from inside the loaded page "
          f"(in {elapsed:.1f}s) => {status!r}")
    if status != 200:
        print("[smoke] FAILED: /health did not return 200 from within the webview",
              file=sys.stderr)
        return 1

    print("[smoke] PASSED: pywebview window loaded frontend/dist and successfully "
          "called GET /health on the local backend from the same origin.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
