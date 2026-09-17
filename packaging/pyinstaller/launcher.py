"""PyInstaller entry point for the installed Adinn 4K Image Enhancer.

Thin on purpose: it only sets ADINN_INSTALLED so backend/workspace.py
redirects job/recent-history storage to a per-user writable folder (an
installed app typically lives under Program Files, which a standard user
cannot write to), then defers to the exact same backend.shell.main() the
dev/source-tree entry point (`python -m backend.shell`) already uses --
nothing about the startup flow itself is reimplemented for packaging.

Dual entry point: a frozen PyInstaller EXE has no separate bundled
python.exe to spawn a helper process from, so this same EXE also serves as
the CPU-mode enhancement worker (backend/cpu_worker.py) -- see
engine_adapter.enhance_image_cpu_subprocess for why that worker needs to
run as a standalone process at all. Invoked as
``Adinn4KImageEnhancer.exe --cpu-worker <input> <output>``, checked FIRST,
before anything else in this file runs (mutex creation, ADINN_INSTALLED,
importing backend.shell) -- that machinery is for the real desktop app
only and would be pure overhead (or, for the mutex, actively confusing) in
a short-lived one-image worker process.
"""

from __future__ import annotations

import os
import sys

if len(sys.argv) >= 2 and sys.argv[1] == "--cpu-worker":
    from backend.cpu_worker import main as _cpu_worker_main
    sys.exit(_cpu_worker_main(sys.argv[2:]))

os.environ.setdefault("ADINN_INSTALLED", "1")

# Named mutex the Inno Setup installer checks (AppMutex) to detect the app is
# already running and refuse to install/uninstall over it with a clear
# native message, instead of silently corrupting a live install. Held for
# the lifetime of this process; released automatically on exit/crash.
if sys.platform == "win32":
    import ctypes

    _MUTEX_NAME = "Adinn4KImageEnhancerRunning"
    ctypes.windll.kernel32.CreateMutexW(None, False, _MUTEX_NAME)

from backend.shell import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
