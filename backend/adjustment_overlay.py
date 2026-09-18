"""Final-render compositing for MVP 2's manual post-processing adjustments
(Brightness/Contrast/Highlights/Shadows/Saturation/Detail).

This is the ONE place adjustments are ever applied to a saved/served image --
the live preview fetch, the single-image download, and the batch export all
call `compose_bytes`/`compose_result` below, which always applies adjustments
(image_enhancer/src/adjustments.py -- pure, deterministic, bounded) BEFORE
billboard rects (backend/billboard_overlay.py's draw_billboard_rects) so a
confirmed board outline is always drawn crisp on top of the adjusted photo,
never itself tinted/blurred by a slider. There is no second implementation
anywhere else in this codebase -- preview and export can never drift apart.

The engine's own output file is never modified and this module never
re-runs any inference; adjustments run strictly after the automatic engine
(Restormer + SwinIR-M + tonal_correction.py), exactly like billboard_overlay.
"""

from __future__ import annotations

from typing import Optional

import cv2

from . import billboard_overlay

_JPEG_QUALITY = 95


def _resolve_image_enhancer_src() -> None:
    """Lazily puts image_enhancer/src on sys.path so `adjustments` can be
    imported -- mirrors engine_adapter.py's own IMAGE_ENHANCER_SRC wiring,
    kept local to this module so importing backend/adjustment_overlay.py
    never requires torch (adjustments.py is torch-free, cv2/numpy only)."""
    import sys
    from pathlib import Path

    from ._frozen import app_root

    src = app_root() / "image_enhancer" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def render_bgr(img, rects, adjust: Optional[dict]):
    """Returns a new BGR image with adjustments applied, then rects drawn on
    top. `img` is left untouched. `adjust` may be None/empty/default (a
    no-op); `rects` may be empty (no outlines drawn)."""
    _resolve_image_enhancer_src()
    import adjustments as adj  # image_enhancer/src/adjustments.py

    out = adj.apply_adjustments(img, adjust) if adjust else img.copy()
    if rects:
        out = billboard_overlay.draw_billboard_rects(out, rects)
    return out


def compose_bytes(src_path, rects, adjust: Optional[dict] = None,
                   image_format: str = "png", quality: Optional[int] = None) -> bytes:
    """Reads `src_path`, applies adjustments then billboard rects, and
    returns encoded bytes in `image_format` ("png"/"jpg"/"jpeg"). An empty
    `rects` + falsy `adjust` re-encodes the clean source untouched. Raises
    ValueError when the source can't be decoded or the result can't be
    encoded (mirrors billboard_overlay.compose_bytes exactly)."""
    img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not read the enhanced result")
    out = render_bgr(img, rects, adjust)
    if image_format in ("jpg", "jpeg"):
        ok, buffer = cv2.imencode(
            ".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, quality if quality is not None else _JPEG_QUALITY])
    else:
        ok, buffer = cv2.imencode(".png", out)
    if not ok:
        raise ValueError("could not encode the composited image")
    return buffer.tobytes()


def composite_result(src_path, dest_path, rects, adjust: Optional[dict] = None,
                      image_format: str = "png", quality: Optional[int] = None) -> tuple:
    """Re-encodes `src_path` (adjustments + rects applied) to `dest_path`.
    Returns (True, None) on success or (False, error-message). Never touches
    `src_path`. Mirrors billboard_overlay.composite_result exactly, for the
    desktop shell's native Save As path."""
    img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
    if img is None:
        return False, "could not read the enhanced result"
    out = render_bgr(img, rects, adjust)
    if image_format in ("jpg", "jpeg"):
        ok = cv2.imwrite(str(dest_path), out,
                         [cv2.IMWRITE_JPEG_QUALITY, quality if quality is not None else _JPEG_QUALITY])
    else:
        ok = cv2.imwrite(str(dest_path), out)
    if not ok:
        return False, "could not write the output image"
    return True, None
