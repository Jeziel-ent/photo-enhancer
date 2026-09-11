"""G5: FINAL billboard text recovery test -- classical / non-generative only.

G4 (stacked G1+G3-PSF) is visually REJECTED: only small incremental gains, text
still not sharp enough. G5 is one final attempt at recovering EXISTING text
from ORIGINAL pixels, F3-natural as full-scene base, verified boxes only.

Three strongest classical reconstruction methods tested (all native-scale,
L-channel only, source-edge gated, clipped, feathered onto F3-natural):

  G5-Wiener : accurately estimated per-box Gaussian PSF + Wiener deconvolution
              with NSR auto-estimated from the box's own flattest patch,
              reflect-padded FFT (edge taper), source-spectrum frequency mask.
  G5-RLC    : constrained Richardson-Lucy with the measured PSF and a
              discrepancy-style iteration rule (blurrier box -> fewer iters),
              stronger pre-denoise, non-negativity inherent to RL.
  G5        : edge/frequency-constrained local reconstruction -- extrema-clipped,
              edge-masked local-contrast restoration. The high-boost residual is
              hard-clipped to the pixel's own local 5x5 [min,max] from the
              ORIGINAL crop, so NO new brightness level can ever be created
              (no overshoot, no ringing, no invented strokes by construction);
              applied only where the source itself has strong edges.

Prototype result (1/3/5.jpeg): Wiener and RLC with the TRUE PSF both SOFTEN
text (te -3..-11%) -- the full-inverse route is exhausted. Only the
edge-constrained local method recovers contrast (te +4..+10%, edge_align 1.0).
All three ship in this file so the test is complete and reproducible; G5
(= local) is the headline candidate judged ACCEPT/REJECT.

CORE RULE: ENHANCE -- DON'T GENERATE. No OCR/redraw/GAN/diffusion/SR/faces.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.restoration import richardson_lucy

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from g3_exp.recipe_g3 import (  # noqa: E402
    _bilateral,
    _edge_gate,
    _bandpass,
    _gauss_psf,
    composite_box_delta,
    estimate_psf_sigma,
)

FEATHER = 16
GATE_T0 = 0.35
LOCAL_STRENGTH = 1.0
LOCAL_CLIP = 10.0
WRLC_CLIP = 10.0


def _nsr_of(l):
    """Noise-to-signal ratio from the box's own flattest 24x24 patch."""
    f = l.astype(np.float32)
    lap = cv2.Laplacian(f, cv2.CV_32F, ksize=3)
    H, W = f.shape
    ws = 24
    best, bv = 1e18, 1.0
    for yy in range(0, H - ws + 1, 12):
        for xx in range(0, W - ws + 1, 12):
            v = float(lap[yy:yy + ws, xx:xx + ws].var())
            if v < best:
                best = v
                p = f[yy:yy + ws, xx:xx + ws]
                bv = float((p - cv2.GaussianBlur(p, (0, 0), 1.0)).var())
    sv = float(f.var()) + 1e-9
    return float(np.clip(bv / sv, 1e-4, 0.05))


def wiener_box_delta(a_l_box, sigma_nat):
    """G5-Wiener: true measured PSF + auto NSR Wiener, reflect-padded."""
    sig = float(np.clip(sigma_nat, 0.7, 3.5)) if sigma_nat > 0 else 1.0
    den = _bilateral(a_l_box)
    H, W = den.shape
    psf = _gauss_psf(sig)
    psf = psf / psf.sum()
    py, px = psf.shape
    pad = max(py, px)
    dp = np.pad(den, pad, mode="reflect")
    Hp, Wp = dp.shape
    pb = np.zeros((Hp, Wp), np.float32)
    cy, cx = Hp // 2, Wp // 2
    pb[cy - py // 2:cy - py // 2 + py, cx - px // 2:cx - px // 2 + px] = psf
    P = np.fft.fft2(pb)
    Gf = np.fft.fft2(dp)
    k = _nsr_of(a_l_box)
    F = P.conj() * Gf / (np.abs(P) ** 2 + k)
    S = np.fft.fft2(dp)
    mask = np.abs(S) >= 0.01 * np.abs(S).max()
    F = F * mask.astype(np.float32)
    out = np.fft.ifft2(F).real.astype(np.float32)[pad:-pad, pad:-pad]
    d = _bandpass(out - a_l_box, 2.0) * _edge_gate(a_l_box, GATE_T0)
    return np.clip(d, -WRLC_CLIP, WRLC_CLIP)


def rlc_box_delta(a_l_box, sigma_nat):
    """G5-RLC: measured PSF, discrepancy-style iteration count, strong denoise."""
    s = float(np.clip(sigma_nat, 1.0, 2.5)) if sigma_nat > 0 else 1.2
    it = int(max(4, min(10, round(18.0 / (1.0 + s)))))
    den = cv2.bilateralFilter(a_l_box.astype(np.float32), d=7,
                              sigmaColor=12, sigmaSpace=7)
    psf = _gauss_psf(s)
    psf = psf / psf.sum()
    r = richardson_lucy(np.clip(den, 0, 255).astype(np.float64) / 255.0,
                        psf, num_iter=it, clip=False)
    r = np.clip(r, 0, 1).astype(np.float32) * 255.0
    d = _bandpass(r - a_l_box, 2.0) * _edge_gate(a_l_box, GATE_T0)
    return np.clip(d, -WRLC_CLIP, WRLC_CLIP)


def local_box_delta(a_l_box, strength=LOCAL_STRENGTH):
    """G5 (headline): extrema-clipped edge-masked local contrast restoration."""
    f = a_l_box.astype(np.float32)
    blur = cv2.GaussianBlur(f, (0, 0), 1.0)
    raw = f - blur
    lo = cv2.erode(f, np.ones((5, 5), np.uint8))
    hi = cv2.dilate(f, np.ones((5, 5), np.uint8))
    clipped = np.clip(raw, lo - f, hi - f)
    d = strength * clipped * _edge_gate(f, GATE_T0)
    return np.clip(d, -LOCAL_CLIP, LOCAL_CLIP)


def apply_variant(f3_bgr, prepped, name):
    """Composite ONE G5 variant onto F3-natural (own Lab plane, boxes only)."""
    if name == "G5-Wiener":
        delta_fn = wiener_box_delta
    elif name == "G5-RLC":
        delta_fn = rlc_box_delta
    elif name == "G5":
        delta_fn = local_box_delta
    else:
        raise ValueError(name)

    lab = cv2.cvtColor(f3_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    new_l = lab[:, :, 0].copy()
    for nat_l, sig, b4 in prepped:
        if b4[2] < 8 or b4[3] < 8 or nat_l is None:
            continue
        if name == "G5":
            d = delta_fn(nat_l)
        else:
            d = delta_fn(nat_l, sig)
        new_l = composite_box_delta(new_l, d, b4, feather=FEATHER)
    lab[:, :, 0] = new_l
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8),
                        cv2.COLOR_LAB2BGR)


def apply_all(f3_bgr, prepped):
    return {n: apply_variant(f3_bgr, prepped, n)
            for n in ("G5-Wiener", "G5-RLC", "G5")}