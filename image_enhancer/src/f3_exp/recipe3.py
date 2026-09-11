"""F3: final photographic experiment - original photo as the base.

Why F2 still looked AI: F2 built on the SR image and its safety mask kept MORE
SR fine-band exactly at strong edges (where SR overshoot lives), and chroma
came from the SR. F3 removes both:

  BASE     = the ORIGINAL photograph (faithful Lanczos 4K). Source of truth.
  SR       = D1-weak output. Used ONLY as a band-passed structure delta.
  CHROMA   = the original photograph's a*/b* (never the SR's).
  EDGES    = positions and strength only from the ORIGINAL's own gradients.

Pipeline:
  Original -> (unchanged D1-weak SR stage) -> SwinIR-M x4  (scale only)
             -> F3 fusion:
                  L_out = L_original
                        + w * clip( bandpass(SR - original) ) * edge_gate(original)
                  edge_gate = 0 except on STRONG source edges (no SR in flats,
                              none outside source edges => nothing invented)
             -> (F3-bb only) inside verified boxes: higher w/clip + unsharp gated
                  by strong source edges only (blur stays blur below threshold)
             -> envelope L +-14 around the original
             -> compose with ORIGINAL chroma -> 3840x2160

Parameters were calibrated on the metric evidence of the first F3 run
(F3-natural at w=0.5 looked like a faithful copy) and a 2-pass sweep on
image 1: text energy (te_contrast) saturates at ~220-226 for ANY gate strength
(SR fine-band is required for real text recovery, which F3 intentionally
forbids), while global clarity rises with edge weight. The two calibrated
working points:

  F3-natural   w=6,  clip +-18, gate t_0=0.30  (crisp but conservative;
               PSNR 35.2, SSIM 0.967, edge_align 1.00 on image 1)
  F3-balanced  w=10, clip +-26, gate t_0=0.20  (max usable strength;
               PSNR 34.1, SSIM 0.951, edge_align 0.993 on image 1)
  F3-bb        F3-balanced + box-only w=2.0 / +-14 + strict-edge USM 0.25
"""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fidelity_exp.recipe import _edge_norm  # noqa: E402
from f2_exp.recipe2 import composite_alpha  # noqa: E402

ENV = 14.0
BP_SIGMA = 2.5
USM_SIGMA = 0.8
USM_THRESH = 12.0
FEATHER = 16


def _lab_l(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)[:, :, 0]


def _compose_from_original(a_bgr, l_new):
    lab = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] = np.clip(l_new, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def _edge_gate(l, t0):
    """1 only on STRONG gradients of the SOURCE (original), else 0.

    t0 = fraction of the p97 edge-energy below which SR structure is
    suppressed. t0=0.60 (F2-era mask) was found to gate almost everything
    away; calibrated values are 0.20-0.30.
    """
    en = _edge_norm(l, pnorm=97.0)
    return np.clip((en - t0) / (1.0 - t0), 0.0, 1.0)


def _structure_term(s_l, a_l):
    sc = s_l - a_l
    return sc - cv2.GaussianBlur(sc, (0, 0), BP_SIGMA)


def f3(a_bgr, s_bgr, w_s, clip, t0, boxes4=None, bi_w=None, bi_clip=None,
       bb_usm=0.0):
    """a_bgr = original (faithful) 4K BGR; s_bgr = D1-weak SR 4K BGR."""
    a_l = _lab_l(a_bgr)
    s_l = _lab_l(s_bgr)
    gate = _edge_gate(a_l, t0)
    sc = _structure_term(s_l, a_l)
    dL = w_s * np.clip(sc, -clip, clip) * gate

    if boxes4:
        for (bx, by, bw, bh) in boxes4:
            if bw < 8 or bh < 8:
                continue
            xs, ys = min(bx + bw, a_l.shape[1]), min(by + bh, a_l.shape[0])
            hh, ww = ys - by, xs - bx
            db = bi_w * np.clip(sc[by:ys, bx:xs], -bi_clip, bi_clip) * \
                gate[by:ys, bx:xs]
            if bb_usm > 0:
                ub = a_l[by:ys, bx:xs]
                dlt = ub - cv2.GaussianBlur(ub, (0, 0), USM_SIGMA)
                gb = gate[by:ys, bx:xs]
                boost = np.where(np.abs(dlt) >= USM_THRESH, bb_usm * dlt, 0.0)
                db = db + boost * gb
            a = composite_alpha(hh, ww, max(2, int(FEATHER)))
            dL[by:ys, bx:xs] = dL[by:ys, bx:xs] * (1.0 - a) + db * a

    l_new = a_l + np.clip(dL, -ENV, ENV)
    return _compose_from_original(a_bgr, l_new)


def f3_natural(a_bgr, s_bgr):
    return f3(a_bgr, s_bgr, w_s=6.0, clip=18.0, t0=0.30)


def f3_balanced(a_bgr, s_bgr):
    return f3(a_bgr, s_bgr, w_s=10.0, clip=26.0, t0=0.20)


def f3_bb(a_bgr, s_bgr, boxes4):
    return f3(a_bgr, s_bgr, w_s=10.0, clip=26.0, t0=0.20, boxes4=boxes4,
              bi_w=2.0, bi_clip=14.0, bb_usm=0.25)