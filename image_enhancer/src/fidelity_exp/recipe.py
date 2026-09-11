"""Fidelity-first post-processing recipes on top of the D1-weak baseline.

All operations are pure NumPy/OpenCV on the 4K output (CPU only). They only
re-combine EXISTING pixels of the photograph - they never invent content:

  E1 TexReinject : pull a small, strong-edge-masked amount of the ORIGINAL
                   photograph's own luminance high-frequency (grain/micro
                   texture) back onto the D1-weak 4K output. Restores natural
                   photographic texture and source fidelity, countering the
                   PSNR-SR "painterly" smoothing, without any generation.
  E2 SmartUSM    : luminance-only, noise-thresholded, strong-edge-protected
                   unsharp that crisps glyph edges / fine detail without
                   halos or face roughening.
  E3 Combined    : D1-weak -> SmartUSM -> TexReinject (natural grain last).
"""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _lab_l(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)[:, :, 0]


def _edge_norm(lum, pnorm=95.0):
    gx = cv2.Sobel(lum, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(lum, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    if pnorm:
        thr = float(np.percentile(mag, pnorm))
        return mag / (thr + 1e-9)
    m = float(mag.max())
    return mag / m if m > 0 else mag


def _lab_recompose(bgr, l_new):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab_float = lab.astype(np.float32)
    lab_float[:, :, 0] = np.clip(l_new, 0, 255)
    out = cv2.cvtColor(lab_float.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return out


def tex_reinject(bgr, faithful4k, alpha=0.50, sigma=0.9, clip=4.0, edge_boost=2.0):
    """bgr: D1-weak 4K output; faithful4k: honest Lanczos upscale of the
    original (the source of the photograph's real grain/micro-texture).

    The layer is clamped to +/- `clip` gray levels so ONLY grain-scale
    variation is re-injected (never mid-scale structure or JPEG blocking),
    and the (percentile-normalized) edge mask keeps grain OFF real structure
    so fidelity stays at the baseline level."""
    L = _lab_l(bgr)
    Lf = _lab_l(faithful4k)
    tex = np.clip(Lf - cv2.GaussianBlur(Lf, (0, 0), sigma), -clip, clip)
    en = _edge_norm(L)
    mask = np.clip(1.0 - edge_boost * en, 0.0, 1.0)
    l_new = L + alpha * tex * mask
    return _lab_recompose(bgr, l_new)


def smart_usm(bgr, amount=0.25, sigma=1.2, thresh=10.0, edge_supp=0.6):
    L = _lab_l(bgr)
    g = cv2.GaussianBlur(L, (0, 0), sigma)
    delta = L - g
    delta = np.where(np.abs(delta) < thresh, 0.0, delta)
    en = _edge_norm(L)
    gain = amount * (1.0 - edge_supp * np.clip(en, 0.0, 1.0))
    l_new = L + gain * delta
    return _lab_recompose(bgr, l_new)


def combined(bgr, faithful4k, alpha=0.30):
    usm = smart_usm(bgr)
    return tex_reinject(usm, faithful4k, alpha=alpha)


def micro_texture(bgr):
    """Median of 3x3 local standard deviation - a no-reference proxy for
    micro-texture retained (higher = less smooth/plastic/painterly)."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.boxFilter(g, -1, (3, 3), normalize=True)
    mean2 = cv2.boxFilter(g * g, -1, (3, 3), normalize=True)
    var = np.maximum(mean2 - mean * mean, 0.0)
    return float(np.median(np.sqrt(var)))