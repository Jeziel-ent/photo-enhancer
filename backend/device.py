"""NVIDIA GPU detection and processing-device resolution.

Deliberately does NOT import torch/CUDA to detect hardware: this module is
called very early (before engine_adapter._import_engine() has ever run, so
before torch itself is imported into the process — see jobs.py's
JobManager._run) and must never risk initializing a CUDA context just to
answer "is a GPU present". Detection instead shells out to ``nvidia-smi``
(installed by every NVIDIA driver package, including on machines this app
will actually ship to) with a short timeout, exactly the same
lightweight, driver-only technique tools like Docker Desktop use for the
same purpose. Any failure (not installed, times out, non-zero exit, no
output) is treated as "no supported GPU" -- never raised, never crashes
startup.

The result of THAT detection is what decides, once, whether
CUDA_VISIBLE_DEVICES is set before enhance.py (and therefore torch) is
first imported in this process -- see resolve_and_apply() and
jobs.py's JobManager._run.
"""

from __future__ import annotations

import subprocess
from typing import Optional

VALID_PREFERENCES = ("auto", "gpu", "cpu")
DEFAULT_PREFERENCE = "auto"

_NVIDIA_SMI_TIMEOUT_S = 3.0


def detect_nvidia_gpu() -> Optional[str]:
    """Returns the first detected NVIDIA GPU's name, or None if nvidia-smi
    is missing, times out, errors, or reports no devices. Never raises."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=_NVIDIA_SMI_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    name = result.stdout.strip().splitlines()[0].strip() if result.stdout.strip() else ""
    return name or None


def resolve_effective_device(preference: str) -> dict:
    """Resolves a user preference ("auto"/"gpu"/"cpu") against actually
    detected hardware. Returns a dict with:

      preference        -- what the user asked for (unchanged)
      detected_gpu      -- GPU name string, or None
      effective_device  -- "gpu" or "cpu": what will ACTUALLY run. Never
                            reports "gpu" unless a GPU was actually
                            detected -- see module docstring's "never
                            silently pretend GPU mode is active" rule.
      warning           -- user-facing string, or None

    "auto": gpu if detected, else cpu, no warning.
    "gpu": gpu if detected; else effective falls back to cpu (so the app
        keeps working) but with a warning explaining the fallback --
        the caller must surface this, never hide it.
    "cpu": always cpu, no CUDA-visibility check needed, no warning. GPU is
        still probed (nvidia-smi is not a CUDA/torch call, so this does not
        violate "do not initialize CUDA") purely so the UI can show
        "Detected: <name>" even while the user has forced CPU.
    """
    if preference not in VALID_PREFERENCES:
        preference = DEFAULT_PREFERENCE
    detected = detect_nvidia_gpu()

    if preference == "cpu":
        return dict(preference=preference, detected_gpu=detected,
                     effective_device="cpu", warning=None)

    if preference == "gpu":
        if detected:
            return dict(preference=preference, detected_gpu=detected,
                         effective_device="gpu", warning=None)
        return dict(
            preference=preference, detected_gpu=None, effective_device="cpu",
            warning=("GPU mode was selected, but no supported NVIDIA GPU was "
                     "detected. Using CPU instead -- switch to Auto or CPU "
                     "to clear this message."))

    # auto
    effective = "gpu" if detected else "cpu"
    return dict(preference=preference, detected_gpu=detected,
                 effective_device=effective, warning=None)
