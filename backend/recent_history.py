"""Locally persisted history of images the user actually saved via the
native Save As flow (see backend/shell.py's DesktopBridge).

This is deliberately a JSON file on disk, not the frontend's own
localStorage: backend/shell.py starts the local API server on an
OS-assigned ephemeral port (``start_backend(port=0)``), so the page's own
origin (``http://127.0.0.1:<port>``) is different every launch and browser
storage tied to that origin would not survive an app restart. A file next
to the rest of this app's local state does.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from . import workspace

MAX_ENTRIES = 50

_LOCK = threading.Lock()


def _history_path() -> Path:
    # Computed per-call (like workspace.py's own helpers) rather than once
    # at import time, so tests can patch workspace.WORKSPACE_ROOT.
    return workspace.WORKSPACE_ROOT / "recent_history.json"


def _read_all() -> list[dict]:
    try:
        raw = _history_path().read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return []  # corrupt/partial file — treat as empty rather than crash
    return data if isinstance(data, list) else []


def load() -> list[dict]:
    """Every saved-result record, newest first."""
    with _LOCK:
        return _read_all()


def add(entry: dict) -> list[dict]:
    """Prepends ``entry`` (newest first) and returns the updated, persisted
    list, capped at MAX_ENTRIES so this file can't grow without bound."""
    with _LOCK:
        entries = _read_all()
        entries.insert(0, entry)
        entries = entries[:MAX_ENTRIES]
        path = _history_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        return entries
