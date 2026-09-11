"""G1: billboard TEXT-readability restoration via non-generative deconvolution.

CORE RULE: ENHANCE -- DON'T GENERATE.

F3-natural (see f3_exp/recipe3.py) solved the painterly-look problem but its
structure term only ever re-admits BAND-PASSED SR structure gated to the
source's own strong edges -- it never touches the ORIGINAL crop's own blur.
The F3 report documents the resulting ceiling: billboard `te_contrast`
saturates at ~220-226 for any gate strength because F3-by-design forbids the
one thing that would raise it further (SR fine-band == invented detail).

G1 tests a different, still non-generative idea: recover contrast that the
photo's OWN optics/motion blur removed, from the ORIGINAL pixels only, via
classic deconvolution -- Richardson-Lucy with a small FIXED Gaussian PSF (no
blind kernel estimation, no learned prior). RL redistributes energy that
already exists in the crop; it cannot draw a stroke that was never resolved
in the source, so a character that is genuinely lost stays blurry (as the
mission requires).

Everything happens ONLY inside the manually verified boxes from
`regions.json`. The full scene is F3-natural, byte-for-byte, everywhere else.

Per-box pipeline:
  1. mild edge-preserving denoise (bilateral) on the ORIGINAL crop's L
     channel -- deconvolution amplifies noise, so denoise first.
  2. Richardson-Lucy deconvolution, small fixed Gaussian PSF, few iterations.
  3. structure delta = RL_result - original, band-passed (drop any coarse/
     low-frequency drift, exactly like F3's SR delta construction) so only
     edge/grain-scale change is ever used.
  4. gated STRICTLY to the box's own strong source edges (percentile-
     normalized gradient magnitude, tighter t0 than F3-bb's 0.20) -- flat
     background and weak/soft edges are left untouched.
  5. optional very mild post unsharp, same edge gate, capped at F3-bb's
     bb_usm=0.25 strength.
  6. feathered composite back onto the F3-natural output (chroma always
     stays the ORIGINAL's a*/b*, never SR/derived -- same invariant as F3).

Variants:
  G1-natural            F3-natural, byte-identical (control / no box touch)
  G1-text-conservative  RL(sigma=1.0, 6 iters), tight gate t0=0.45, clip 10
  G1-text               RL(sigma=1.2, 12 iters), gate t0=0.30, clip 16,
                         post-USM 0.25 (matches F3-bb's box USM cap)
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
USM_SIGMA = 0.8
USM_THRESH = 10.0


def _lab_l(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)[:, :, 0]


def _edge_gate(l, t0):
    """1 only on STRONG gradients of the SOURCE crop, else 0 (see F3)."""
    en = _edge_norm(l, pnorm=97.0)
    return np.clip((en - t0) / (1.0 - t0), 0.0, 1.0)


def _richardson_lucy_l(l_box, sigma, iters):
    """Fixed small Gaussian PSF, few RL iterations. No blind estimation."""
    lf = np.clip(l_box, 0, 255).astype(np.float64) / 255.0
    ksz = max(3, int(round(sigma * 6)) | 1)
    k1 = cv2.getGaussianKernel(ksz, sigma)
    psf = (k1 @ k1.T)
    psf = psf / psf.sum()
    rl = richardson_lucy(lf, psf, num_iter=iters, clip=False)
    return np.clip(rl, 0.0, 1.0).astype(np.float32) * 255.0


def g1_box_delta(a_l_box, sigma, iters, gate_t0, delta_clip, usm_amt):
    """Non-generative luminance delta for one billboard box, derived only
    from the ORIGINAL crop's own pixels (no SR, no synthesis)."""
    den = cv2.bilateralFilter(a_l_box.astype(np.float32), d=5,
                              sigmaColor=15, sigmaSpace=5)
    rl = _richardson_lucy_l(den, sigma, iters)
    sc = rl - a_l_box
    sc_bp = sc - cv2.GaussianBlur(sc, (0, 0), BP_SIGMA)
    gate = _edge_gate(a_l_box, gate_t0)
    db = np.clip(sc_bp, -delta_clip, delta_clip) * gate

    if usm_amt > 0:
        blur = cv2.GaussianBlur(a_l_box, (0, 0), USM_SIGMA)
        udelta = a_l_box - blur
        udelta = np.where(np.abs(udelta) >= USM_THRESH, usm_amt * udelta, 0.0)
        db = db + udelta * gate

    return db


def g1(a_bgr, f3_natural_bgr, boxes4, sigma, iters, gate_t0, delta_clip,
       usm_amt, feather=FEATHER):
    """a_bgr = ORIGINAL faithful 4K BGR; f3_natural_bgr = accepted F3-natural
    output (full-scene base, left untouched outside boxes4).

    Reuses f3_natural_bgr's OWN Lab plane (single BGR<->Lab round trip,
    same as F3 itself) instead of re-deriving chroma from a_bgr a second
    time -- avoids compounding uint8 Lab round-trip quantization noise
    (OpenCV's BGR<->Lab conversion is not perfectly idempotent, ~1-2 levels)
    outside the boxes, so pixels there stay effectively byte-identical to
    F3-natural."""
    a_l = _lab_l(a_bgr)
    lab_out = cv2.cvtColor(f3_natural_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    for (bx, by, bw, bh) in boxes4:
        if bw < 8 or bh < 8:
            continue
        xs, ys = min(bx + bw, a_l.shape[1]), min(by + bh, a_l.shape[0])
        hh, ww = ys - by, xs - bx
        a_l_box = a_l[by:ys, bx:xs]
        db = g1_box_delta(a_l_box, sigma, iters, gate_t0, delta_clip, usm_amt)
        base_box = lab_out[by:ys, bx:xs, 0]
        new_box = np.clip(base_box + db, 0, 255)
        alpha = composite_alpha(hh, ww, max(2, int(feather)))
        lab_out[by:ys, bx:xs, 0] = base_box * (1.0 - alpha) + new_box * alpha

    return cv2.cvtColor(np.clip(lab_out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def g1_natural(a_bgr, f3_natural_bgr, boxes4=None):
    """Control: F3-natural, byte-identical. No box restoration applied."""
    return f3_natural_bgr.copy()


def g1_text_conservative(a_bgr, f3_natural_bgr, boxes4):
    return g1(a_bgr, f3_natural_bgr, boxes4,
              sigma=1.0, iters=6, gate_t0=0.45, delta_clip=10.0, usm_amt=0.15)


def g1_text(a_bgr, f3_natural_bgr, boxes4):
    return g1(a_bgr, f3_natural_bgr, boxes4,
              sigma=1.2, iters=12, gate_t0=0.30, delta_clip=16.0, usm_amt=0.25)
