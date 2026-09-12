"""PyInstaller entry point for the installed Adinn 4K Image Enhancer.

Thin on purpose: it only sets ADINN_INSTALLED so backend/workspace.py
redirects job/recent-history storage to a per-user writable folder (an
installed app typically lives under Program Files, which a standard user
cannot write to), then defers to the exact same backend.shell.main() the
dev/source-tree entry point (`python -m backend.shell`) already uses --
nothing about the startup flow itself is reimplemented for packaging.
"""

from __future__ import annotations

import os
import sys

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
