"""Auto-detect the billboard/signage panel region in a street photo.

>>> EXPERIMENTAL -- NOT PRODUCTION-READY <<<

This whole module is a research experiment. Generic rectangle detection
(contours, straight-line segments, bright-panel thresholding) is UNSUFFICIENT
for reliable OOH billboard detection on real phone photos: it selects roads,
sky, vehicles and unrelated rectangles, and it has no notion of what a
billboard IS (content, logos, brand presence, lighting, context).

The benchmarked billboard pipeline (method C) must therefore use MANUALLY
VERIFIED bounding boxes from `regions.json` (see `verify_regions.py`). This
detector is kept only as a separate experimental component for future work
(e.g. a learned object-detection model) and must NEVER silently substitute
for the verified regions in the benchmark.

Target (for future experimental iterations): a bright, distinctly-framed
rectangular panel (Adinn billboard) located interior to the frame (not the
whole scene).

Strategy:
1. Find candidate rectangles from (a) nested contours and (b) long straight
   line segments.
2. Score each candidate on how 'billboard-like' it is:
   - interior brightness (a lit panel is bright)
   - border edge-support ratio (a frame produces strong edges on its ring)
   - 'interior-ness' penalty: boxes touching the image border are likely the
     full scene, not the board.
   - aspect ratio between ~1.2 and ~4 (wide board).
   - area between ~3% and ~45% of the frame.
Returns (x, y, w, h) top-left + size, or None.
"""

import cv2
import numpy as np

EXPERIMENTAL = True


def detect_billboard(img, debug=False):
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 60, 150)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                             iterations=1)

    cands = []
    cands += _from_contours(gray, edges, H, W)
    cands += _from_lines(edges, H, W)
    cands += _from_bright_panels(gray, H, W)

    if not cands:
        return None

    min_area = 0.03 * H * W
    max_area = 0.45 * H * W
    best, best_score = None, -1.0
    for (x, y, w, h) in cands:
        x = int(x); y = int(y); w = int(w); h = int(h)
        x = max(0, min(x, W - 1)); y = max(0, min(y, H - 1))
        w = min(w, W - x); h = min(h, H - y)
        if w < 50 or h < 40:
            continue
        area = w * h
        if not (min_area <= area <= max_area):
            continue
        ar = w / h
        if ar < 1.2 or ar > 4.5:
            continue
        score = _score(gray, x, y, w, h, H, W)
        if score > best_score:
            best_score = score
            best = (x, y, w, h)

    return best


# ---------------------------------------------------------------- candidates

def _from_contours(gray, edges, H, W):
    out = []
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if len(c) < 4:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w < 50 or h < 40:
            continue
        area = cv2.contourArea(c)
        rect_area = w * h
        if rect_area <= 0:
            continue
        # keep boxy shapes: contour fills most of its bounding rect
        if area / rect_area > 0.55:
            out.append((x, y, w, h))
    return out


def _from_lines(edges, H, W):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            threshold=max(max(H, W) // 40, 60),
                            minLineLength=min(H, W) // 8, maxLineGap=15)
    if lines is None:
        return []
    out = []
    for l in lines:
        l = np.asarray(l).reshape(-1)
        if l.size < 4:
            continue
        x1, y1, x2, y2 = l[:4].astype(int)
        if x2 - x1 > y2 - y1:  # dominant horizontal
            out.append((min(x1, x2), min(y1, y2), abs(x2 - x1),
                        max(0, int(abs(x2 - x1) * 0.4))))
        else:
            out.append((min(x1, x2), min(y1, y2),
                        max(0, int(abs(y2 - y1) * 0.6)), abs(y2 - y1)))
    return out


def _from_bright_panels(gray, H, W):
    """Find a bright, large, interior rectangle via thresholding."""
    out = []
    for thresh in (200, 180, 160):
        mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w < 50 or h < 40:
                continue
            out.append((x, y, w, h))
    return out


# ------------------------------------------------------------------ scoring

def _score(gray, x, y, w, h, H, W):
    # interior brightness
    pad_x = int(w * 0.05); pad_y = int(h * 0.05)
    interior = gray[y + pad_y: y + h - pad_y, x + pad_x: x + w - pad_x]
    bright = float(np.mean(interior)) / 255.0 if interior.size else 0.0

    # border edge-support: fraction of the box border ring covered by edges
    ring = np.zeros_like(gray)
    cv2.rectangle(ring, (x, y), (x + w, y + h), 255, 2)
    ring_binary = ring > 0
    edges = cv2.Canny(gray, 60, 150) > 0
    support = float(np.logical_and(edges, ring_binary).sum()) / max(1.0, ring_binary.sum())

    # interior-ness: penalize boxes touching image border (likely full scene)
    touches = 0
    if x == 0: touches += 1
    if y == 0: touches += 1
    if x + w >= W: touches += 1
    if y + h >= H: touches += 1
    interior_penalty = 1.0 if touches == 0 else (0.45 ** touches)

    score = 0.40 * bright + 0.45 * support + 0.15 * interior_penalty
    return score
