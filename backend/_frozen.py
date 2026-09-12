"""Where this app's own bundled resources (frontend/dist, image_enhancer/)
actually live on disk, whether running from the source tree or from a
PyInstaller-frozen build.

Every other module that needs this (backend/shell.py, backend/engine_adapter.py)
computed it themselves via ``Path(__file__).resolve().parents[1]`` -- correct
only when running from source, since a frozen build's real, on-disk layout
under ``sys._MEIPASS`` doesn't have to mirror the source tree at all. This is
the one place that distinction is made.
"""

from __future__ import annotations

import sys
from pathlib import Path


def app_root() -> Path:
    """The directory bundled resources are addressed relative to: the repo
    root when running from source, or PyInstaller's bundle root
    (``sys._MEIPASS``) when frozen -- see packaging/pyinstaller/adinn.spec's
    ``datas``, which places frontend/dist and image_enhancer/ at the same
    relative paths under that root either way."""
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent.parent
