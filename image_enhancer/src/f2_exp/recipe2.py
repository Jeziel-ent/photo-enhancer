"""F2: multi-scale detail-fusion for a photographic result.

Design (technically safest investigated approach):
  Original photo = SOURCE OF TRUTH. AI SR = structure assistance only.

  D1-weak SR output provides coarse (low-frequency) structure;
  the faithful Lanczos upscale of the ORIGINAL provides the fine
  (high-frequency) real photographic texture. The two are fused in a
  Laplacian pyramid where, per octave band, the ORIGINAL's band weight is
  high for fine scales (real grain) and drops to ~0 for coarse scales (SR
  structure). Every fusion point is attenuated by the SR edge mask (no
  double-edges / no halo) and the whole result is clamped to a small
  +/-DELTA "safety envelope" around the SR base so we can only refine, never
  invent. No generation, no OCR, no redraw: only existing pixels.

Variants (all built on the same D1-weak SR base, CPU-only):
  F2-fuse  pyramid fusion only
  F2       F2-fuse + very mild local contrast (CLAHE blend 0.2)
  F2-bb    F2 + slightly stronger fine-band weight INSIDE the verified
           billboard boxes only (feathered, same grid => alignment kept) +
           restrained tiny box-only unsharp (amount 0.12)
"""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fidelity_exp.recipe import _edge_norm  # noqa: E402

GLOBAL_W = [0.85, 0.45, 0.15, 0.06, 0.03, 0.0]   # orig weight per band (fine->coarse)
BB_W = [1.00, 0.75, 0.35, 0.12, 0.06, 0.0]
DELTA = 12.0
EDGE_SCALE = 2.0


def _lab_l(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)[:, :, 0]


def _lab_recompose(bgr, l_new):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] = np.clip(l_new, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def _gauss_pyr(l, levels):
    pyr = [l]
    cur = l
    for _ in range(levels):
        cur = cv2.pyrDown(cur)
        pyr.append(cur)
    return pyr


def _lap_pyr(pyr):
    lp = []
    for i in range(len(pyr) - 1):
        up = cv2.pyrUp(pyr[i + 1], dstsize=(pyr[i].shape[1], pyr[i].shape[0]))
        lp.append(pyr[i] - up)
    lp.append(np.copy(pyr[-1]))
    return lp


def _fuse(sr_l, f4k_l, weights, emask_l):
    """Fuse two L* planes: fine bands from original, coarse from SR."""
    n = len(weights)
    sp = _gauss_pyr(sr_l, n - 1)
    fp = _gauss_pyr(f4k_l, n - 1)
    sl = _lap_pyr(sp)
    fl = _lap_pyr(fp)
    out = []
    for k in range(n):
        m = cv2.resize(emask_l, (sl[k].shape[1], sl[k].shape[0]),
                       interpolation=cv2.INTER_AREA)
        ow = weights[k] * (0.5 + 0.5 * np.clip(m, 0.0, 1.0))
        out.append(ow * fl[k] + (1.0 - ow) * sl[k])
    res = np.copy(out[-1])
    for k in range(n - 2, -1, -1):
        up = cv2.pyrUp(res, dstsize=(out[k].shape[1], out[k].shape[0]))
        res = up + out[k]
    return res


def _edge_mask(sr_l):
    en = _edge_norm(sr_l)
    return np.clip(1.0 - EDGE_SCALE * en, 0.0, 1.0)


def _envelope(sr_l, l_new):
    delta = np.clip(l_new - sr_l, -DELTA, DELTA)
    return sr_l + delta


def fuse_l(l, sr_l, f4k_l, weights=GLOBAL_W, box_weights=None, boxes4=None,
          edge=24):
    """l: base BGR 4K. Returns fused BGR. box_weights: optional stronger
    weights applied inside boxes4 (4K rects, same grid, feathered)."""
    sr_l0 = _lab_l(l)
    f4k = _lab_l(f4k_l)
    emask = _edge_mask(sr_l0)
    fused = _fuse(sr_l0, f4k, weights, emask)

    if box_weights is not None and boxes4:
        for (bx, by, bw, bh) in boxes4:
            if bw < 8 or bh < 8:
                continue
            xs, ys = min(bx + bw, fused.shape[1]), min(by + bh, fused.shape[0])
            sub_sr = sr_l0[by:ys, bx:xs]
            sub_f4 = f4k[by:ys, bx:xs]
            sub_em = emask[by:ys, bx:xs]
            sub_fused = _fuse(sub_sr, sub_f4, box_weights, sub_em)
            hh, ww = sub_fused.shape
            a = composite_alpha(hh, ww, edge)
            fused[by:ys, bx:xs] = (sub_fused * a +
                                   fused[by:ys, bx:xs] * (1.0 - a))

    out = _envelope(sr_l0, fused)
    return _lab_recompose(l, out)


def mild_lc(bgr, clip=0.9, amount=0.2):
    """Very mild local luminance contrast (CLAHE blend)."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l = lab[:, :, 0]
    cl = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(
        l.astype(np.uint8)).astype(np.float32)
    l_new = l * (1.0 - amount) + cl * amount
    return _lab_recompose(bgr, l_new)


def box_usm(bgr, boxes4, amount=0.12, sigma=0.9):
    """Restrained unsharp INSIDE billboard boxes only (feathered)."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l = lab[:, :, 0]
    g = cv2.GaussianBlur(l, (0, 0), sigma)
    delta = l - g
    for (bx, by, bw, bh) in boxes4:
        if bw < 8 or bh < 8:
            continue
        xs, ys = min(bx + bw, l.shape[1]), min(by + bh, l.shape[0])
        hh, ww = ys - by, xs - bx
        a = composite_alpha(hh, ww, max(2, int(0.05 * min(hh, ww))))
        sub = amount * delta[by:ys, bx:xs]
        l[by:ys, bx:xs] += a * sub
    lab[:, :, 0] = np.clip(l, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def composite_alpha(h, w, edge):
    a = np.ones((h, w), np.float32)
    edge = max(1, int(edge))
    for i in range(min(edge, h)):
        t = (i + 1) / (edge + 1)
        a[i, :] = t
        a[h - 1 - i, :] = t
    for i in range(min(edge, w)):
        t = (i + 1) / (edge + 1)
        a[:, i] = np.minimum(a[:, i], t)
        a[:, w - 1 - i] = np.minimum(a[:, w - 1 - i], t)
    return a


def f2_fuse(bgr, faithful4k):
    return fuse_l(bgr, bgr, faithful4k, weights=GLOBAL_W)


def f2(bgr, faithful4k):
    return mild_lc(fuse_l(bgr, bgr, faithful4k, weights=GLOBAL_W))


def f2_bb(bgr, faithful4k, boxes4):
    fused = fuse_l(bgr, bgr, faithful4k, weights=GLOBAL_W, box_weights=BB_W,
                   boxes4=boxes4)
    fused = mild_lc(fused)
    return box_usm(fused, boxes4)