"""Final-export billboard overlay compositing for backend/shell.py and
backend/server.py.

The enhancement engine never touches billboard data (see engine_adapter.py).
Instead, when a single-image result is saved *with* confirmed billboard
rectangles, this module burns a clean red outline plus a small "Board" label
onto a copy of the engine's finished PNG (matching the frontend editor's
confirmed style: no translucent fill, no dark label bar). The engine's output
file is never modified and this module never re-runs any inference.

Everything here is defensive about its input: rectangles are clamped to the
image bounds, degenerate/fully-out-of-bounds rects are skipped, and a malformed
list simply renders nothing (which is what an empty list does).
"""

from __future__ import annotations

from typing import List, Optional

import cv2

_RED = (0, 0, 230)  # BGR (shared with the frontend editor's board red)
_LABEL_TEXT = "Board"
_LABEL_HALO = (255, 255, 255)  # thin white halo so the red label stays legible
_MIN_THICKNESS = 3
_THICKNESS_DIVISOR = 480  # thickness = min(w,h)/480 -> 8px on a 3840x2160 frame
_LABEL_GAP = 6  # vertical gap between the rect border and the label
_LABEL_MIN_MARGIN = 6  # frame-edge padding for the label
_LABEL_FONT_SCALE_DIVISOR = 2400.0  # smaller than the old bar label -> ~1.6 at 4K
_JPEG_QUALITY = 95


def clamp_billboard_rects(rects, width: int, height: int) -> List[dict]:
    """Normalise a list of rect dicts to in-bounds, non-degenerate rects.

    Each rect must expose x/y/width/height (the frontend's BillboardRect).
    Values are coerced to ints and clamped so the rect never extends past the
    image; rects whose clamped area is empty are dropped instead of crashing
    downstream drawing.
    """
    cleaned: List[dict] = []
    if not isinstance(rects, list):
        return cleaned
    for raw in rects:
        if not isinstance(raw, dict):
            continue
        try:
            x = int(raw.get("x"))
            y = int(raw.get("y"))
            w = int(raw.get("width"))
            h = int(raw.get("height"))
        except (TypeError, ValueError):
            continue
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = max(0, min(w, width - x))
        h = max(0, min(h, height - y))
        if w <= 0 or h <= 0:
            continue
        cleaned.append({"x": x, "y": y, "width": w, "height": h})
    return cleaned


def draw_billboard_rects(img, rects) -> "cv2.typing.MatLike":
    """Return a new image with the given billboard rects drawn on it.

    ``img`` is left untouched. Each rect gets a clean red outline and a small
    "Board" label just above it — no translucent interior fill, no dark label
    bar (the same visual style as the frontend editor's confirmed mode).
    """
    out = img.copy()
    height, width = out.shape[:2]
    rects = clamp_billboard_rects(rects, width, height)
    if not rects:
        return out

    thickness = max(_MIN_THICKNESS, min(width, height) // _THICKNESS_DIVISOR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.9, width / _LABEL_FONT_SCALE_DIVISOR)
    font_thickness = max(2, thickness // 2)

    for rect in rects:
        cv2.rectangle(
            out,
            (rect["x"], rect["y"]),
            (rect["x"] + rect["width"], rect["y"] + rect["height"]),
            _RED,
            thickness,
        )

    (text_w, text_h), baseline = cv2.getTextSize(
        _LABEL_TEXT, font, font_scale, font_thickness)
    margin = max(_LABEL_MIN_MARGIN, thickness)
    for rect in rects:
        # Center the label over the rect, keeping it inside the frame.
        x0 = rect["x"] + rect["width"] // 2 - text_w // 2
        x0 = max(margin, min(x0, width - text_w - margin))

        # Preferred spot: just above the rect's top border. If that would run
        # off the top, drop it just inside the rect's top edge instead.
        label_bottom = rect["y"] - thickness - _LABEL_GAP
        if label_bottom - text_h < 0:
            label_bottom = rect["y"] + thickness + _LABEL_GAP + text_h + baseline
        label_bottom = min(label_bottom, height - margin)

        # White halo underneath the red text keeps it legible over any photo
        # without adding a dark bar behind it.
        cv2.putText(out, _LABEL_TEXT, (x0, label_bottom), font, font_scale,
                    _LABEL_HALO, font_thickness + 2, cv2.LINE_AA)
        cv2.putText(out, _LABEL_TEXT, (x0, label_bottom), font, font_scale,
                    _RED, font_thickness, cv2.LINE_AA)

    return out


def composite_result(src_path, dest_path, rects, image_format: str = "png",
                     quality: Optional[int] = None) -> tuple:
    """Re-encode ``src_path`` with rects drawn on, writing ``dest_path``.

    Returns (True, None) on success or (False, error-message). Never touches
    ``src_path``. ``image_format`` is "png"/"jpg"/"jpeg"; JPEG is written at
    ``quality`` (default 95), matching the app's pre-existing re-encode path.
    """
    img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
    if img is None:
        return False, "could not read the enhanced result"
    composed = draw_billboard_rects(img, rects)
    if image_format in ("jpg", "jpeg"):
        ok = cv2.imwrite(str(dest_path), composed,
                         [cv2.IMWRITE_JPEG_QUALITY, quality if quality is not None else _JPEG_QUALITY])
    else:
        ok = cv2.imwrite(str(dest_path), composed)
    if not ok:
        return False, "could not write the output image"
    return True, None


def compose_png_bytes(src_path, rects) -> bytes:
    """Render ``src_path`` with rects drawn on and return PNG bytes.

    Used by the HTTP result route (browser download), so the compositing
    stays in Python while only the tiny rect metadata crosses the wire.
    Raises ValueError when the source can't be decoded or the PNG can't be
    encoded.
    """
    return compose_bytes(src_path, rects, "png")


def compose_bytes(src_path, rects, image_format: str = "png",
                   quality: Optional[int] = None) -> bytes:
    """Like compose_png_bytes but for any of the app's export formats
    ("png"/"jpg"/"jpeg"). An empty/falsy ``rects`` list re-encodes the clean
    source with no overlay drawn at all (not just an overlay of nothing) —
    used by the batch export route's "include board outlines: off" path.
    Raises ValueError when the source can't be decoded or re-encoded.
    """
    img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not read the enhanced result")
    composed = draw_billboard_rects(img, rects) if rects else img
    if image_format in ("jpg", "jpeg"):
        ok, buffer = cv2.imencode(
            ".jpg", composed,
            [cv2.IMWRITE_JPEG_QUALITY, quality if quality is not None else _JPEG_QUALITY])
    else:
        ok, buffer = cv2.imencode(".png", composed)
    if not ok:
        raise ValueError("could not encode the composited image")
    return buffer.tobytes()