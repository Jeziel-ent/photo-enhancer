"""Locally persisted user settings (currently just processing-device
preference). Same JSON-file-next-to-the-workspace pattern as
recent_history.py, for the same reason: this shell binds to a fresh
ephemeral port every launch, so page-origin-scoped localStorage would not
survive an app restart.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from . import device, workspace

_LOCK = threading.Lock()


def _settings_path() -> Path:
    return workspace.WORKSPACE_ROOT / "settings.json"


def _read_all() -> dict:
    try:
        raw = _settings_path().read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {}  # corrupt/partial file — treat as defaults rather than crash
    return data if isinstance(data, dict) else {}


def get_processing_device_preference() -> str:
    pref = _read_all().get("processing_device")
    return pref if pref in device.VALID_PREFERENCES else device.DEFAULT_PREFERENCE


def set_processing_device_preference(preference: str) -> str:
    """Validates and persists ``preference``. Returns the value actually
    stored (falls back to the default if given something invalid)."""
    if preference not in device.VALID_PREFERENCES:
        preference = device.DEFAULT_PREFERENCE
    with _LOCK:
        data = _read_all()
        data["processing_device"] = preference
        path = _settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return preference
