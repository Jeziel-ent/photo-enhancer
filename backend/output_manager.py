"""Turns a job's per-file enhancement results into the single artifact the
API serves back from ``GET /api/jobs/<id>/result``: the enhanced image
itself for a one-image job, or a ZIP of every successfully enhanced image
for a multi-image job — regardless of how many of those images failed.
"""

from __future__ import annotations

import zipfile
from pathlib import Path


def build_result(
    succeeded: list[tuple[str, Path]],
    zip_path: Path,
    is_batch: bool,
) -> Path:
    """Returns the path to serve for a completed job.

    ``succeeded`` is a list of ``(original_filename, enhanced_path)`` for
    every file that enhanced successfully, in upload order. Raises
    ValueError if there is nothing to package — callers should only invoke
    this once at least one file has succeeded.
    """
    if not succeeded:
        raise ValueError("no successfully enhanced files to package")
    if not is_batch:
        return succeeded[0][1]

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        used_names: set[str] = set()
        for original_name, enhanced_path in succeeded:
            arcname = _unique_arcname(original_name, enhanced_path, used_names)
            zf.write(enhanced_path, arcname=arcname)
    return zip_path


def _unique_arcname(original_filename: str, enhanced_path: Path, used_names: set[str]) -> str:
    stem = Path(original_filename).stem or enhanced_path.stem
    candidate = f"{stem}_enhanced{enhanced_path.suffix}"
    if candidate not in used_names:
        used_names.add(candidate)
        return candidate
    # Two uploads shared a filename stem (e.g. two "photo.jpg" from different
    # folders) — disambiguate rather than overwrite one inside the zip.
    n = 2
    while f"{stem}_enhanced_{n}{enhanced_path.suffix}" in used_names:
        n += 1
    candidate = f"{stem}_enhanced_{n}{enhanced_path.suffix}"
    used_names.add(candidate)
    return candidate
