"""Local, controlled workspace for job input/output files.

Every uploaded file is copied into a per-job directory under here before
anything touches it — the engine and output manager never see, and never
write to, wherever the user's original file actually lives on disk. Nothing
under here is committed to git (see .gitignore); it is disposable, generated
state, the same way ``backend/.deckstore/`` was for the retired PPT product.
"""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path


def _default_workspace_root() -> Path:
    """Dev/source-tree default: a sibling of this file, exactly as before.

    An installed build sets ADINN_INSTALLED=1 (see the Inno Setup script)
    because its own directory (typically under Program Files) is not
    writable by a standard user — job input/output files and recent_history
    have to live somewhere the current user can actually write to instead.
    """
    if os.environ.get("ADINN_INSTALLED") == "1" and "LOCALAPPDATA" in os.environ:
        return Path(os.environ["LOCALAPPDATA"]) / "Adinn4KImageEnhancer" / "workspace"
    return Path(__file__).resolve().parent / ".workspace"


WORKSPACE_ROOT = _default_workspace_root()


def new_job_id() -> str:
    return uuid.uuid4().hex


def job_dir(job_id: str) -> Path:
    return WORKSPACE_ROOT / job_id


def input_dir(job_id: str) -> Path:
    return job_dir(job_id) / "input"


def output_dir(job_id: str) -> Path:
    return job_dir(job_id) / "output"


def result_zip_path(job_id: str) -> Path:
    return job_dir(job_id) / "result.zip"


def export_zip_path(job_id: str) -> Path:
    """A separate ZIP from result_zip_path — holds a batch *export* (chosen
    format + optional per-image board overlays), rebuilt fresh on every
    export request rather than reusing/overwriting the raw batch result."""
    return job_dir(job_id) / "export.zip"


def prepare_job_dirs(job_id: str) -> tuple[Path, Path]:
    """Creates (and returns) this job's input/ and output/ directories."""
    in_dir = input_dir(job_id)
    out_dir = output_dir(job_id)
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    return in_dir, out_dir


def purge_job(job_id: str) -> None:
    """Best-effort removal of a job's entire workspace directory."""
    shutil.rmtree(job_dir(job_id), ignore_errors=True)
