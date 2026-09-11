"""Billboard perspective/planarity investigation (Part 2).

Determines, per verified board, whether the surface is approximately frontal
(no rectification needed) or keystoned enough to warrant 4-point rectification
before restoration. Reports per-board diagnostics; does NOT change the default
pipeline - the decision is left for the A/B/C comparison to judge.

Method: upscale the padded crop 4x, threshold the brightest large panel,
approximate a quadrilateral, and measure corner angles. A board whose quad
is within KEYS_TOL of 90 degrees is considered frontal.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

KEYS_TOL = 3.0
MIN_AREA_FRAC = 0.15


def _order_corners(pts):
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], np.float32)


def _angles(quad):
    out = []
    for i in range(4):
        a = quad[i]
        b = quad[(i + 1) % 4]
        c = quad[(i + 2) % 4]
        v1 = b - a
        v2 = c - b
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        out.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    return out


def analyze(bgr_orig, box, pad_ratio=0.15, upscale=4):
    H, W = bgr_orig.shape[:2]
    x, y, w, h = box
    px = int(w * pad_ratio)
    py = int(h * pad_ratio)
    x0, y0 = max(0, x - px), max(0, y - py)
    x1, y1 = min(W, x + w + px), min(H, y + h + py)
    crop = bgr_orig[y0:y1, x0:x1]
    up = cv2.resize(crop, (crop.shape[1] * upscale, crop.shape[0] * upscale),
                    interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    crop_area = up.shape[0] * up.shape[1]
    best = None
    if cnts:
        best = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(best) / crop_area < MIN_AREA_FRAC:
            best = None
    if best is None:
        quad = np.array([[0, 0], [up.shape[1], 0],
                         [up.shape[1], up.shape[0]], [0, up.shape[0]]],
                        np.float32)
        source = "full-crop fallback"
    else:
        rect = cv2.minAreaRect(best)
        quad = _order_corners(np.array(cv2.boxPoints(rect), np.float32))
        source = "min-area-rect of largest panel"
    angs = _angles(quad)
    keystone = float(max(abs(a - 90) for a in angs))
    frontal = keystone <= KEYS_TOL
    return {"found": True, "keystone_deg": round(keystone, 2), "angles": angs,
            "rectify_recommended": not frontal, "quad_source": source,
            "note": "frontal" if frontal else f"keystone {keystone:.1f} deg"}