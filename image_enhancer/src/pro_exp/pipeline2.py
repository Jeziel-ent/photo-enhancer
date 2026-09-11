"""Professional photo enhancement pipelines (isolated experiment).

All full-frame variants are 3840x2160 BGR uint8:

  A    SwinIR-M PSNR x4, full frame (reused from restored_exp Path-A).
  D1   Restormer real-denoise BEFORE SwinIR-M PSNR x4.
  D2   SwinIR-M PSNR x4 first, then Restormer real-denoise on the 4K result.
  P2   A + ProLook lite (professional deterministic polish).
  P2b  A + ProLook standard (stronger, still non-generative).
  P3   P2 + feathered billboard composite of RECb regions.

Region recipes (crops built from the ORIGINAL, not from full-frame results):
  RECa  billboard crop: SwinIR-M PSNR x4 directly on the crop.
  RECb  billboard crop: Restormer real-denoise -> SwinIR-M PSNR x4.
  RECc  billboard crop: RECb + ProLook lite.
  FC    face crop: Restormer real-denoise -> SwinIR-M PSNR x4 -> ProLook lite.

Everything is non-generative: only existing pixels are re-combined.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pro_exp import prolook  # noqa: E402
from restore_exp import restormer, swinir_m  # noqa: E402

PAD_RATIO = 0.15
OUT_W, OUT_H = 3840, 2160


def _ramp(n, m):
    r = np.arange(n, dtype=np.float32)
    d = np.minimum(r, n - 1 - r)
    return np.maximum(np.minimum(d / max(m, 1), 1.0), 0.25)


def tiled_restore(model, img, tile=1024, overlap=64):
    """Same-size Restormer real-denoise with feathered tile blending,
    safe for 8 GB VRAM on 4K inputs."""
    h, w = img.shape[:2]
    tile = min(tile, h, w)
    stride = tile - overlap
    rows = list(range(0, h - tile, stride)) + [h - tile]
    cols = list(range(0, w - tile, stride)) + [w - tile]
    single = len(rows) == 1 and len(cols) == 1
    out = np.zeros((h, w, 3), np.float32)
    wsum = np.zeros((h, w), np.float32)
    for y in rows:
        for x in cols:
            r = restormer.enhance_bgr(model, img[y:y + tile, x:x + tile])
            if single:
                wgt = np.ones((tile, tile), np.float32)
            else:
                ry = _ramp(tile, overlap)
                wgt = ry[:, None] * ry[None, :]
            out[y:y + tile, x:x + tile] += r.astype(np.float32) * wgt[:, :, None]
            wsum[y:y + tile, x:x + tile] += wgt
    return np.clip(out / (wsum[:, :, None] + 1e-9), 0, 255).astype(np.uint8)


def recipe_crop(orig, pr, mode, device, tile=400):
    """Build a region recipe crop (4x native upscale of the ORIGINAL crop).

    pr = (x0, y0, x1, y1) padded region in ORIGINAL coordinates."""
    x0, y0, x1, y1 = pr
    crop = orig[y0:y1, x0:x1]
    c = crop
    if mode in ("RECb", "RECc", "FC"):
        c = restormer.enhance_bgr(restormer.load("real_denoise", device), c)
    c = swinir_m.sr_bgr(c, device, tile=tile)
    if mode in ("RECc", "FC"):
        c = prolook.lite(c)
    return c


def to_box_size(crop, rect):
    w, h = rect[2], rect[3]
    if crop.shape[:2] != (h, w):
        crop = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return crop


def detail_window(orig, exclude_rects, win=160):
    """Pick the sharpest informative sub-window outside billboard/face boxes,
    in ORIGINAL coordinates (used as the fine-detail strip for zoom montages)."""
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    H, W = gray.shape
    banned = np.zeros((H, W), bool)
    for r in exclude_rects:
        x, y, w, h = r
        x = max(0, x - 24)
        y = max(0, y - 24)
        xs = min(W, x + w + 48)
        ys = min(H, y + h + 48)
        banned[y:ys, x:xs] = True
    win = min(win, H, W)
    step = max(4, win // 4)
    best, best_rect = -1.0, (0, 0, win, win)
    for y in range(0, H - win + 1, step):
        for x in range(0, W - win + 1, step):
            if banned[y:y + win, x:x + win].any():
                continue
            s = float(mag[y:y + win, x:x + win].mean())
            if s > best:
                best, best_rect = s, (x, y, win, win)
    return best_rect


def text_proxies(bgr):
    """No-reference text-legibility proxies on a billboard crop."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    gm = cv2.magnitude(gx, gy)
    strong = gm[gm >= np.percentile(gm, 95)]
    te = float(np.mean(strong)) if strong.size else 0.0
    hp = g - cv2.GaussianBlur(g, (0, 0), 2.0)
    return {
        "te_contrast": round(te, 2),
        "hp_energy": round(float(np.std(hp)), 2),
    }