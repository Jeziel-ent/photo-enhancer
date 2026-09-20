"""Turns a job's per-file enhancement results into the single artifact the
API serves back from ``GET /api/jobs/<id>/result``: the enhanced image
itself for a one-image job, or a ZIP of every successfully enhanced image
for a multi-image job — regardless of how many of those images failed.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

_EXPORT_EXTENSIONS = {"png": ".png", "jpg": ".jpg", "jpeg": ".jpeg"}


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
            stem = Path(original_name).stem or enhanced_path.stem
            arcname = _unique_arcname(stem, enhanced_path.suffix, used_names)
            zf.write(enhanced_path, arcname=arcname)
    return zip_path


def build_batch_export(
    items: list[tuple],
    zip_path: Path,
    image_format: str,
    include_outlines: bool,
    adjust: dict | None = None,
) -> Path:
    """Builds the "Save ZIP" batch export: every image is re-encoded to one
    COMMON ``image_format`` ("png"/"jpg"/"jpeg"), gets ONLY its own board
    rectangles burned in (never another image's) — and only when
    ``include_outlines`` is on — and applies manual adjustments
    (Brightness/Contrast/Highlights/Shadows/Saturation/Detail) via
    backend/adjustment_overlay.py — the one shared implementation preview
    and export both use. Each image's values are its OWN: an ``items``
    entry may carry a per-image ``adjust`` dict as its 4th element
    (``(name, path, rects, per_image_adjust)``); when it is None/absent the
    common ``adjust`` param is used for that image. Setting equal void
    values for every image behaves exactly like the previous common-adjust
    single-payload export. The original per-file engine outputs are never
    modified; this always writes a fresh zip.

    ``items`` is ``(original_filename, enhanced_path, rects)`` per image, in
    the order they should appear in the zip (or the 4-tuple form above).
    Raises ValueError if empty.
    """
    if not items:
        raise ValueError("no images to export")
    from . import adjustment_overlay  # local import: keeps cv2 off jobs.py's import path

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    ext = _EXPORT_EXTENSIONS.get(image_format, ".png")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        used_names: set[str] = set()
        for item in items:
            if len(item) >= 4:
                original_name, enhanced_path, rects, per_image_adjust = item
            else:
                original_name, enhanced_path, rects = item
                per_image_adjust = None
            item_adjust = per_image_adjust if per_image_adjust is not None else adjust
            data = adjustment_overlay.compose_bytes(
                enhanced_path, rects if include_outlines else [], item_adjust, image_format)
            stem = Path(original_name).stem or enhanced_path.stem
            arcname = _unique_arcname(stem, ext, used_names)
            zf.writestr(arcname, data)
    return zip_path


def _unique_arcname(stem: str, suffix: str, used_names: set[str]) -> str:
    candidate = f"{stem}_enhanced{suffix}"
    if candidate not in used_names:
        used_names.add(candidate)
        return candidate
    # Two uploads shared a filename stem (e.g. two "photo.jpg" from different
    # folders) — disambiguate rather than overwrite one inside the zip.
    n = 2
    while f"{stem}_enhanced_{n}{suffix}" in used_names:
        n += 1
    candidate = f"{stem}_enhanced_{n}{suffix}"
    used_names.add(candidate)
    return candidate
