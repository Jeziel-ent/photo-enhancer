"""G4: stacked billboard recovery — G1-text + G3-PSF, non-generative.

Pipeline:
  Original -> F3-natural -> G1-text -> G4(=G1-text + G3-PSF)

Only inside verified billboard boxes (regions.json). Everything outside the
boxes is byte-identical to F3-natural throughout. Chroma (a*/b*) never
touched — all deltas are L-channel only, feathered onto F3-natural's own
Lab plane.

G1-text already applies RL-deconvolution at 4K scale (fixed sigma 1.2,
clip ±16). G4 additionally applies G3-PSF: Richardson-Lucy at NATIVE box
scale with a kernel capped to the recoverable fraction of the per-box
measured Gaussian PSF (sigma_k = clamp(0.40 * measured_sigma, 0.9, 1.2),
12 iters). Native-scale is the right scale for the genuinely large blur
(1.jpeg sigma 3.19 px native vs 4K-equivalent 0.43 px), which is why
G3-PSF alone lifted 1.jpeg's readability from +3.2% (G1) to +6.8%
(G3). Stacking combines the two complementary stages.

CORE RULE: ENHANCE — DON'T GENERATE.
No OCR redraw, no generated text, no GAN/diffusion, no invented strokes.
If information is absent, remain blurry.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.restoration import richardson_lucy

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fidelity_exp.recipe import _edge_norm  # noqa: E402
from f2_exp.recipe2 import composite_alpha  # noqa: E402

FEATHER = 16
BP_SIGMA = 2.0
GATE_T0 = 0.35
G1_CLIP = 16.0
G3_CLIP = 8.0
RL_ITERS = 12
RL_KAPPA = 0.40
RL_K_MIN = 0.9
RL_K_MAX = 1.2


# ---------------------------------------------------------------------------
# native-scale G3-PSF delta (mirrors recipe_g3.psf_box_delta)
# ---------------------------------------------------------------------------
def _edge_gate(l, t0=GATE_T0, pnorm=97.0):
    en = _edge_norm(l, pnorm=pnorm)
    return np.clip((en - t0) / (1.0 - t0), 0.0, 1.0)


def _bandpass(delta, sig=BP_SIGMA):
    return delta - cv2.GaussianBlur(delta, (0, 0), sig)


def _gauss_psf(sigma):
    ksz = max(3, int(round(sigma * 6)) | 1)
    k1 = cv2.getGaussianKernel(ksz, sigma)
    return (k1 @ k1.T) / (k1 @ k1.T).sum()


def _bilateral(l):
    return cv2.bilateralFilter(l.astype(np.float32), d=5,
                               sigmaColor=15, sigmaSpace=5)


def psf_delta_native(nat_l, sigma_nat):
    """G3-PSF delta at native scale: RL with PSF-capped kernel."""
    k = float(np.clip(RL_KAPPA * sigma_nat, RL_K_MIN, RL_K_MAX)) if sigma_nat > 0 else 1.0
    den = _bilateral(nat_l)
    psf = _gauss_psf(k)
    rl = richardson_lucy(np.clip(den, 0, 255).astype(np.float64) / 255.0,
                         psf, num_iter=RL_ITERS, clip=False)
    rl = np.clip(rl, 0, 1).astype(np.float32) * 255.0
    d = _bandpass(rl - nat_l)
    return np.clip(d, -G3_CLIP, G3_CLIP) * _edge_gate(nat_l)


# ---------------------------------------------------------------------------
# stack onto a base image
# ---------------------------------------------------------------------------
def composite_box_delta(f3_lab_l, delta_native, box4, feather=FEATHER):
    bx, by, bw, bh = box4
    xs = min(bx + bw, f3_lab_l.shape[1])
    ys = min(by + bh, f3_lab_l.shape[0])
    hh, ww = ys - by, xs - bx
    d4 = cv2.resize(delta_native, (ww, hh), interpolation=cv2.INTER_LANCZOS4)
    full = np.zeros_like(f3_lab_l)
    full[by:ys, bx:xs] = d4
    alpha = composite_alpha(hh, ww, max(2, int(feather)))
    out = f3_lab_l.copy()
    out[by:ys, bx:xs] += full[by:ys, bx:xs] * alpha
    return out


def g4(base_bgr, prepped):
    """Apply G4 = G1-text + G3-PSF stacking onto base (typically G1-text output).

    base_bgr: BGR image already containing F3-natural + G1-text box deltas.
    prepped: list of (native_l_box, sigma_nat, box4).
    Returns G4 BGR.
    """
    lab = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    new_l = lab[:, :, 0].copy()
    for nat_l, _sig, b4 in prepped:
        if b4[2] < 8 or b4[3] < 8 or nat_l is None:
            continue
        d = psf_delta_native(nat_l, _sig)
        new_l = composite_box_delta(new_l, d, b4)
    lab[:, :, 0] = new_l
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)