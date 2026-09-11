"""G7: conservative, non-generative candidates built ON TOP of the G6 A+
whole-frame baseline, aimed at closing the perceptual gap against a
generative 4K reference while preserving fidelity.

Isolated experiment. Does NOT modify enhance.py, g6_exp, or any Phase 2/3/
backend/frontend code. Candidate A and A+ are reused UNCHANGED from
g6_exp.recipe_g6 (candidate_a, candidate_a_plus, apply_boards,
whole_frame_local_contrast) so every G7 candidate is judged against the same
already-vetted baseline instead of a re-implementation of it.

Every candidate below only ever widens or reshuffles WHOLE-FRAME contrast
that is already self-clipped to a locally-derived envelope taken from the
image itself (a local min/max box, or a band-limited residual around the
existing F3 output) -- never an unbounded sharpen, and never anything that
can plausibly invent a new brightness level, letterform, or structure.
Billboard regions are still fully replaced by Candidate A's own
already-validated board reconstruction (apply_boards), unchanged, for every
non-diagnostic candidate.

Candidates:

  G7-RL    Full-frame post-F3 controlled deblur: the SAME per-pixel adaptive-
           sigma Richardson-Lucy + edge-gated residual technique used by
           Candidate A's own board touch-up (enhance._final_board_touchup),
           generalized to the whole 4K frame instead of one box, at a
           reduced iteration count (8) since a whole frame is far less
           uniformly blurred than a single billboard crop. Applied AFTER F3
           natural, and the standard A+ local-contrast pass (which hard-clips
           to each pixel's own local 5x5 [min,max]) is re-applied on top so
           the final output still carries that self-clip guarantee.

  G7-MS    Multi-scale (3-level DoG/Laplacian-style) detail recovery: F3's
           L-plane is split into 3 frequency bands + a low-pass base. Each
           band is boosted by a modest gain (1.15-1.3x) and then hard-clipped
           to a *local min/max envelope derived from the source image itself*
           at a structuring-element size matched to that band's scale, so a
           band's boosted contribution can never push a pixel past a
           brightness level that does not already exist somewhere in its own
           neighborhood. Standard A+ pass reapplied on top.

  G7-GF    Guided/edge-preserving detail split: this environment's OpenCV
           build has no cv2.ximgproc (checked: `hasattr(cv2, "ximgproc")` is
           False), so a joint bilateral base/detail split is used instead
           (edge-preserving base via cv2.bilateralFilter, detail = L - base).
           The detail layer is boosted conservatively and edge-gated, then
           self-clipped to the same local 5x5 [min,max] envelope used by A+.
           Standard A+ pass reapplied on top.

  G7-WIDE  Same as A+, but F3's own edge-gated fusion envelope is widened
           from +-14 to +-20 before the A+ pass, letting slightly more of
           the SR-driven D1 structure through the F3 fusion gate. Everything
           else (weight, clip, gate threshold) is identical to production
           F3-natural. Measures whether the extra headroom buys a real gain
           or mostly buys halos/hallucination risk.

  G7-GAN   DIAGNOSTIC ONLY, explicitly rejected by design: Real-ESRGAN
           x4plus (enhance.realesrgan_enhance, already vetted/rejected
           elsewhere in this repo for exactly this reason) run on the FULL
           FRAME, to visually anchor "how sharp is generative" as an upper
           bound. No board pass is applied on top (we want to see, and then
           reject, exactly what the GAN itself does inside the JJ GOLD/GRT
           boxes) -- box coordinates are still computed so the same box crops
           can be pulled for metrics/montages.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import enhance  # noqa: E402
from g6_exp import recipe_g6  # noqa: E402

# Re-exported verbatim from G6 so G7's own module is the single import site
# a caller needs for "baseline + all new candidates".
candidate_a = recipe_g6.candidate_a
candidate_a_plus = recipe_g6.candidate_a_plus
apply_boards = recipe_g6.apply_boards
whole_frame_local_contrast = recipe_g6.whole_frame_local_contrast


# --------------------------------------------------------------- G7-RL
_RL_ITERS = 8
_RL_CLIP = 16.0
_RL_GATE_T0 = 0.30
_RL_BP_SIGMA = 2.0
_RL_SIGMA_MIN = 0.8
_RL_SIGMA_MAX = 3.5
_RL_SIGMA_DEFAULT = 1.2


def whole_frame_rl_deblur(bgr, iters=_RL_ITERS, clip=_RL_CLIP,
                          gate_t0=_RL_GATE_T0, bp_sigma=_RL_BP_SIGMA):
    """Candidate A's own board touch-up technique (adaptive-sigma RL +
    band-limited, edge-gated residual), generalized to the WHOLE frame.
    Same math as enhance._final_board_touchup, fewer iterations (a full
    scene is not uniformly blurred the way one billboard crop is)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    sigma = enhance._final_psf_sigma(gray)
    if sigma <= 0:
        sigma = _RL_SIGMA_DEFAULT
    sigma = float(np.clip(sigma, _RL_SIGMA_MIN, _RL_SIGMA_MAX))

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l = lab[:, :, 0]
    den = cv2.bilateralFilter(l, d=5, sigmaColor=15, sigmaSpace=5)
    ksz = max(3, int(round(sigma * 6)) | 1)
    k1 = cv2.getGaussianKernel(ksz, sigma)
    psf = k1 @ k1.T
    psf = psf / psf.sum()
    rl = enhance._final_richardson_lucy(
        np.clip(den, 0, 255).astype(np.float64) / 255.0, psf, iters)
    rl = np.clip(rl, 0, 1).astype(np.float32) * 255.0

    sc = rl - l
    sc_bp = sc - cv2.GaussianBlur(sc, (0, 0), bp_sigma)
    gate = enhance._final_edge_gate(l, gate_t0)
    db = np.clip(sc_bp, -clip, clip) * gate
    lab[:, :, 0] = np.clip(l + db, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def candidate_g7_rl(orig, target=(3840, 2160)):
    faithful = enhance.simple_upscale(orig, target)
    d1, _peak, device = enhance._final_d1_weak(orig)
    f3 = enhance._final_f3_natural(faithful, d1)
    f3_rl = whole_frame_rl_deblur(f3)
    f3_final = whole_frame_local_contrast(f3_rl)
    device_t = torch.device(device)
    out, stages = apply_boards(f3_final, orig, device_t)
    return out, stages


# --------------------------------------------------------------- G7-MS
_MS_GAINS = (1.30, 1.20, 1.15)
_MS_SIGMAS = (1.0, 2.0, 4.0)
_MS_KSIZES = (3, 5, 9)
_MS_GATE_T0 = 0.30
_MS_CLIP = 10.0


def multiscale_detail_boost(bgr, gains=_MS_GAINS, sigmas=_MS_SIGMAS,
                            ksizes=_MS_KSIZES, gate_t0=_MS_GATE_T0,
                            overall_clip=_MS_CLIP):
    """3-level DoG/Laplacian-style multi-scale detail recovery. Each band's
    boosted contribution is hard-clipped to a LOCAL min/max envelope of the
    SOURCE L-plane at a structuring-element size matched to that band's
    scale -- so a band can only push a pixel toward a brightness level that
    already exists somewhere in its own local neighborhood, never invent
    one. The final combined delta is additionally edge-gated and globally
    clipped before being applied."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l = lab[:, :, 0]
    g1 = cv2.GaussianBlur(l, (0, 0), sigmas[0])
    g2 = cv2.GaussianBlur(l, (0, 0), sigmas[1])
    g3 = cv2.GaussianBlur(l, (0, 0), sigmas[2])
    bands = (l - g1, g1 - g2, g2 - g3)
    base = g3

    recon = base.copy()
    for band, gain, ksz in zip(bands, gains, ksizes):
        k = np.ones((ksz, ksz), np.uint8)
        lo = cv2.erode(l, k)
        hi = cv2.dilate(l, k)
        boosted = np.clip(band * gain, lo - l, hi - l)
        recon = recon + boosted

    delta = recon - l
    gate = enhance._final_edge_gate(l, gate_t0)
    delta = np.clip(delta * gate, -overall_clip, overall_clip)
    lab[:, :, 0] = np.clip(l + delta, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def candidate_g7_ms(orig, target=(3840, 2160)):
    faithful = enhance.simple_upscale(orig, target)
    d1, _peak, device = enhance._final_d1_weak(orig)
    f3 = enhance._final_f3_natural(faithful, d1)
    f3_ms = multiscale_detail_boost(f3)
    f3_final = whole_frame_local_contrast(f3_ms)
    device_t = torch.device(device)
    out, stages = apply_boards(f3_final, orig, device_t)
    return out, stages


# --------------------------------------------------------------- G7-GF
_GF_DETAIL_GAIN = 1.30
_GF_GATE_T0 = 0.30
_GF_CLIP = 8.0
_GF_SIGMA_COLOR = 25
_GF_SIGMA_SPACE = 9

HAS_XIMGPROC = hasattr(cv2, "ximgproc")


def guided_detail_boost(bgr, detail_gain=_GF_DETAIL_GAIN, gate_t0=_GF_GATE_T0,
                        clip=_GF_CLIP):
    """Edge-preserving base/detail split + conservative detail boost.

    This environment's OpenCV build has no cv2.ximgproc (guidedFilter is
    unavailable: `hasattr(cv2, 'ximgproc')` is False here), so per the task's
    documented fallback this uses a joint bilateral base/detail split
    instead (cv2.bilateralFilter as the edge-preserving base estimator).
    Boosted detail is hard-clipped to the same local 5x5 [min,max] envelope
    used by the A+ pass, so it cannot invent a new brightness level, and is
    edge-gated + globally clipped."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l = lab[:, :, 0]
    if HAS_XIMGPROC:  # pragma: no cover -- not available in this env
        base = cv2.ximgproc.guidedFilter(guide=l, src=l, radius=4, eps=200)
    else:
        base = cv2.bilateralFilter(l, d=9, sigmaColor=_GF_SIGMA_COLOR,
                                   sigmaSpace=_GF_SIGMA_SPACE)
    detail = l - base
    lo = cv2.erode(l, np.ones((5, 5), np.uint8))
    hi = cv2.dilate(l, np.ones((5, 5), np.uint8))
    boosted = np.clip(detail * detail_gain, lo - l, hi - l)
    gate = enhance._final_edge_gate(l, gate_t0)
    d = np.clip(boosted * gate, -clip, clip)
    lab[:, :, 0] = np.clip(l + d, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def candidate_g7_gf(orig, target=(3840, 2160)):
    faithful = enhance.simple_upscale(orig, target)
    d1, _peak, device = enhance._final_d1_weak(orig)
    f3 = enhance._final_f3_natural(faithful, d1)
    f3_gf = guided_detail_boost(f3)
    f3_final = whole_frame_local_contrast(f3_gf)
    device_t = torch.device(device)
    out, stages = apply_boards(f3_final, orig, device_t)
    return out, stages


# --------------------------------------------------------------- G7-WIDE
_WIDE_ENV = 20.0


def f3_natural_wide(faithful4k, d1_4k, env=_WIDE_ENV):
    """Identical to enhance._final_f3_natural except the final edge-gated
    fusion envelope is widened from the production +-14 to +-`env` (default
    +-20), letting slightly more D1 structure through the F3 gate before the
    A+ pass. Weight, per-pixel clip, and gate threshold are unchanged from
    production so only the envelope parameter differs."""
    a_l = enhance._final_lab_l(faithful4k)
    s_l = enhance._final_lab_l(d1_4k)
    gate = enhance._final_edge_gate(a_l, enhance._FINAL_F3_T0)
    sc = (s_l - a_l) - cv2.GaussianBlur(s_l - a_l, (0, 0), enhance._FINAL_F3_BP_SIGMA)
    dL = enhance._FINAL_F3_W * np.clip(
        sc, -enhance._FINAL_F3_CLIP, enhance._FINAL_F3_CLIP) * gate
    l_new = a_l + np.clip(dL, -env, env)
    lab = cv2.cvtColor(faithful4k, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] = np.clip(l_new, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def candidate_g7_wide(orig, target=(3840, 2160)):
    faithful = enhance.simple_upscale(orig, target)
    d1, _peak, device = enhance._final_d1_weak(orig)
    f3w = f3_natural_wide(faithful, d1)
    f3_final = whole_frame_local_contrast(f3w)
    device_t = torch.device(device)
    out, stages = apply_boards(f3_final, orig, device_t)
    return out, stages


# --------------------------------------------------------------- G7-GAN (diagnostic)
def get_boxes4(orig):
    """Compute the same box4/psf-sigma bookkeeping apply_boards() produces,
    WITHOUT running Candidate A's board reconstruction -- used by the GAN
    diagnostic, which must show the GAN's own (unedited) behaviour inside
    the billboard boxes rather than Candidate A's board composited over it."""
    H0, W0 = orig.shape[:2]
    boxes = enhance._billboard_boxes(orig)
    boxes4, sigmas, defocus_boxes = [], [], []
    for (x, y, w, h) in boxes:
        vb = enhance._final_valid_box((x, y, w, h), W0, H0)
        if vb is None:
            continue
        vx, vy, vw, vh = vb
        bx, by, bw, bh = enhance._final_scale_box(vb, W0)
        xs = min(bx + bw, enhance.OUT_W)
        ys = min(by + bh, enhance.OUT_H)
        if xs - bx < 8 or ys - by < 8:
            continue
        box4 = (bx, by, xs - bx, ys - by)
        boxes4.append(box4)
        nc = orig[vy:vy + vh, vx:vx + vw]
        sig = enhance._final_psf_sigma(cv2.cvtColor(nc, cv2.COLOR_BGR2GRAY))
        sigmas.append(sig)
        if sig >= enhance._FINAL_DEFOCUS_SIGMA:
            defocus_boxes.append(box4)
    return boxes4, sigmas, defocus_boxes


def candidate_g7_gan(orig, target=(3840, 2160)):
    """DIAGNOSTIC UPPER BOUND ONLY -- not a production candidate. Real-ESRGAN
    x4plus on the full frame, deliberately with NO billboard-box pass on top,
    so the montage/metrics show exactly what the GAN itself produces inside
    the JJ GOLD / GRT boxes for the fidelity check the report documents."""
    out = enhance.realesrgan_enhance(orig, target=target)
    boxes4, sigmas, defocus_boxes = get_boxes4(orig)
    stages = {"boxes4": boxes4, "psf_sigmas": sigmas,
             "defocus_boxes": defocus_boxes}
    return out, stages
