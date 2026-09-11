"""G3: SPECIALIZED billboard TEXT recovery -- three non-generative approaches.

CORE RULE: ENHANCE -- DON'T GENERATE. (same invariant as G1/G2)
Original photograph = ONLY source of truth. No OCR redraw, no generated text,
no GAN/diffusion, no invented strokes. If information is absent -> stay blurry.

G1 (src/g1_exp) already proved RL deconvolution with a FIXED tiny sigma helps
modestly (+11.3% mean te_contrast) but its kernel badly underestimates the true
native blur (~2.5px Gaussian, measured per box). G2 proved that using a LARGER
neural/LSUP model (or an aggressive full inverse) invents structure / rings.

G3 tests three CLASSICAL approaches that derive everything from the ORIGINAL
native box pixels (native scale is where the real blur/loss happened):

  1. G3-PSF   PSF-based deconvolution  : Richardson-Lucy with a kernel capped to
       the *recoverable* fraction of the per-box measured Gaussian PSF
       (sigma_k = clamp(0.40 * measured_sigma, 0.9, 1.2) px, native) -- only the
       part of the blur that is safely invertible is inverted; the rest is left.
  2. G3-Edge  edge-aware text recovery : anisotropic shock filter (Osher-Rudin)
       on the native L channel -- a classical PDE that drives existing gradients
       to step edges WITHOUT moving their location or changing their endpoints.
       Band-passed + strong-edge gated + clipped, so flat background and
       genuinely soft strokes get ZERO change (nothing invented).
  3. G3-Freq  frequency-domain constrained reconstruction : Laplacian-band
       selective gain. Per-octave gains come only from the measured source PSF
       attenuation (capped inverse), applied to the source's own bands and gated
       to strong source edges -- never admits a frequency the source lacks.

Every candidate is computed at NATIVE box scale, Band-Passed (drop coarse drift),
gated to the box's OWN strong source edges (percentile-normalized gradient,
t0 = 0.35, same construction as F3/G1), clipped to +/-8 L levels, Lanczos-
upscaled to the 4K box, then feathered (16px) onto F3-natural's OWN Lab L plane.
Chroma (a*/b*) is NEVER touched and remains F3-natural's (which is the original's
chroma). Everything outside the verified regions.json boxes is byte-identical to
F3-natural.

Verification helpers used by run.py:
  stroke_mask_iou(boxL_ref, boxL_pred): adaptive-threshold stroke-core IoU -- a
     topology check that letterforms did not change (dark-run cores preserved).
  overshoot_frac(src, pred): fraction of box where pred exceeds the source's own
     local (3x3) extrema -- catches broadening/thickening/halo strokes.
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
D_CLIP = 8.0
RL_ITERS = 12
SHOCK_ITERS = 14
SHOCK_DT = 0.20


# ---------------------------------------------------------------------------
# source-information: per-box PSF estimate (median edge-rise Gaussian sigma)
# ---------------------------------------------------------------------------
def estimate_psf_sigma(gray, edge_pct=99.0):
    """Median 10%-90% rise width / 2.35 on the strongest vertical edges =
    a Gaussian PSF sigma estimate for this box at NATIVE resolution."""
    g = cv2.GaussianBlur(gray, (0, 0), 0.4)
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0)
    mag = np.abs(gx)
    thr = np.percentile(mag, edge_pct)
    ys, xs = np.nonzero(mag >= thr)
    H, W = g.shape
    sigs = []
    for y, x in zip(ys[::19], xs[::19]):
        lo, hi = max(0, x - 15), min(W, x + 16)
        row = g[y, lo:hi].astype(np.float64)
        if row.size < 6:
            continue
        mn, mx = row.min(), row.max()
        if mx - mn < 20:
            continue
        t10, t90 = mn + 0.1 * (mx - mn), mn + 0.9 * (mx - mn)
        i10 = int(np.argmin(np.abs(row - t10))) + lo
        i90 = int(np.argmin(np.abs(row - t90))) + lo
        w = abs(i90 - i10)
        if 0.3 <= w <= 12:
            sigs.append(w / 2.35)
    if len(sigs) < 5:
        return -1.0
    return float(np.median(sigs))


def _lab_l(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)[:, :, 0]


def _edge_gate(l, t0=GATE_T0, pnorm=97.0):
    en = _edge_norm(l, pnorm=pnorm)
    return np.clip((en - t0) / (1.0 - t0), 0.0, 1.0)


def _bandpass(delta, sig=BP_SIGMA):
    return delta - cv2.GaussianBlur(delta, (0, 0), sig)


def _gauss_psf(sigma):
    ksz = max(3, int(round(sigma * 6)) | 1)
    k1 = cv2.getGaussianKernel(ksz, sigma)
    return k1 @ k1.T


def _bilateral(l):
    return cv2.bilateralFilter(l.astype(np.float32), d=5,
                               sigmaColor=15, sigmaSpace=5)


# ---------------------------------------------------------------------------
# candidates (all native-scale, all return a gated band-passed clipped delta)
# ---------------------------------------------------------------------------
def psf_box_delta(a_l_box, sigma_nat):
    """G3-PSF: RL with kernel capped to the recoverable fraction of the
    measured native blur. Only ~0.4x of the true kernel is inverted; the rest
    is intentionally left (full inverse == ringing / invented strokes, G2's
    lesson)."""
    k = float(np.clip(0.40 * sigma_nat, 0.9, 1.2)) if sigma_nat > 0 else 1.0
    den = _bilateral(a_l_box)
    psf = _gauss_psf(k)
    psf = psf / psf.sum()
    rl = richardson_lucy(np.clip(den, 0, 255).astype(np.float64) / 255.0,
                         psf, num_iter=RL_ITERS, clip=False)
    rl = np.clip(rl, 0, 1).astype(np.float32) * 255.0
    d = _bandpass(rl - a_l_box)
    return np.clip(d, -D_CLIP, D_CLIP) * _edge_gate(a_l_box)


def _shock(l, iters, dt):
    u = _bilateral(l).copy()
    for _ in range(iters):
        gx = cv2.Sobel(u, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(u, cv2.CV_32F, 0, 1)
        gxx = cv2.Sobel(u, cv2.CV_32F, 2, 0)
        gyy = cv2.Sobel(u, cv2.CV_32F, 0, 2)
        gxy = cv2.Sobel(u, cv2.CV_32F, 1, 1)
        mag = cv2.magnitude(gx, gy)
        eta = (gx * gx * gxx + 2 * gx * gy * gxy + gy * gy * gyy) \
            / (mag * mag + 1.0)
        eta = np.nan_to_num(eta, nan=0.0, posinf=0.0, neginf=0.0)
        u = u - dt * np.sign(eta) * mag
    return u


def edge_box_delta(a_l_box):
    """G3-Edge: anisotropic shock-filter edge recovery. Drives existing
    gradients to rectangular step edges WITHOUT moving the edge location or
    changing the plateau values on either side (no new brightness invented)."""
    s = _shock(a_l_box, SHOCK_ITERS, SHOCK_DT)
    d = _bandpass(s - a_l_box)
    return np.clip(d, -D_CLIP, D_CLIP) * _edge_gate(a_l_box)


def freq_box_delta(a_l_box, sigma_nat):
    """G3-Freq: Laplacian-band selective gain. Each octave's gain is the
    capped inverse of the measured Gaussian PSF attenuation at that octave --
    i.e. it only re-amplifies energy that the source's own blur removed, and
    never goes above `cap` (no invented frequency content)."""
    sig = max(0.5, sigma_nat) if sigma_nat > 0 else 1.0
    den = _bilateral(a_l_box)
    h, w = den.shape
    nlev = min(3, int(np.floor(np.log2(min(h, w)))) - 1)
    down = den
    pyr = [den.copy()]
    for _ in range(nlev):
        down = cv2.pyrDown(down)
        pyr.append(down)
    out = den.copy()
    prev = den
    cap = 2.5
    for i in range(nlev):
        up = cv2.pyrUp(pyr[i + 1])
        up = cv2.resize(up, (prev.shape[1], prev.shape[0]),
                        interpolation=cv2.INTER_LINEAR)
        band = prev - up
        cyi = (i + 1) / nlev
        freq = 2.0 ** (1 + 2 * i) * cyi if cyi > 0 else 2.0 ** (i + 1)
        atten = np.exp(-2 * np.pi ** 2 * sig ** 2 *
                       (freq / (2 * min(prev.shape))) ** 2)
        gain = float(np.clip(1.0 / (atten + 1e-6), 1.0, cap))
        out = out + (gain - 1.0) * band
        prev = up
    d = _bandpass(out - den)
    return np.clip(d, -10.0, 10.0) * _edge_gate(a_l_box)


# ---------------------------------------------------------------------------
# composite (native delta -> 4K box -> feather onto F3-natural Lab plane)
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
    new = f3_lab_l.copy()
    new[by:ys, bx:xs] += full[by:ys, bx:xs] * alpha
    return new


def apply_variant(f3_bgr, prepped, name):
    """Composite ONE G3 variant onto F3-natural.

    prepped: list of (native_l_box, sigma_nat, box4) -- native L crops from the
    ORIGINAL photo, the sigma measured by estimate_psf_sigma, and the 4K-scaled
    verified box. Deltas are computed natively, band-passed, edge-gated,
    clipped, Lanczos-upscaled and feathered onto F3-natural's own Lab plane.
    Chroma (a*/b*) is never touched. Outside boxes: byte-identical to F3."""
    if name == "G3-PSF":
        delta_fn = psf_box_delta
    elif name == "G3-Edge":
        delta_fn = edge_box_delta
    elif name == "G3-Freq":
        delta_fn = freq_box_delta
    else:
        raise ValueError(name)

    lab = cv2.cvtColor(f3_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    base_l = lab[:, :, 0].copy()
    new_l = base_l.copy()

    for nat_l, sig, b4 in prepped:
        if b4[2] < 8 or b4[3] < 8 or nat_l is None:
            continue
        if name in ("G3-PSF", "G3-Freq"):
            d = delta_fn(nat_l, sig)
        else:
            d = delta_fn(nat_l)
        new_l = composite_box_delta(new_l, d, b4)

    lab[:, :, 0] = new_l
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8),
                        cv2.COLOR_LAB2BGR)


def apply_all(f3_bgr, prepped):
    """Run all three G3 candidates. Returns dict name -> BGR (F3 outside boxes)."""
    return {n: apply_variant(f3_bgr, prepped, n)
            for n in ("G3-PSF", "G3-Edge", "G3-Freq")}


# ---------------------------------------------------------------------------
# fidelity verification (no-ref hallucination checks used by run.py / report)
# ---------------------------------------------------------------------------
def stroke_mask_iou(ref_l, pred_l, block=41, c=-12):
    """IoU of adaptive-thresholded dark stroke cores between reference box and
    candidate box. High IoU => same letter boundaries / shapes preserved."""
    def _mask(l):
        l8 = np.clip(l, 0, 255).astype(np.uint8)
        th = cv2.adaptiveThreshold(l8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, block, c)
        core = th == 0
        core = cv2.erode(core.astype(np.uint8), np.ones((2, 2), np.uint8))
        return core.astype(bool)
    a = _mask(ref_l)
    b = _mask(pred_l)
    inter = (a & b).sum()
    union = (a | b).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def overshoot_frac(ref_l, pred_l, pad=1, tol=2.0):
    """Fraction of box pixels where pred exceeds the reference's own local
    (3x3) extrema by more than `tol` L levels. Counts stroke-broadening /
    halo / overshoot that would change letterforms or photo geometry."""
    if ref_l.shape != pred_l.shape:
        ref_l = cv2.resize(ref_l, (pred_l.shape[1], pred_l.shape[0]),
                           interpolation=cv2.INTER_AREA)
    dil = cv2.dilate(ref_l, np.ones((3, 3), np.uint8))
    ero = cv2.erode(ref_l, np.ones((3, 3), np.uint8))
    bad = ((pred_l > dil + tol) | (pred_l < ero - tol)).astype(np.float32)
    if pad > 0:
        bad = bad[pad:-pad, pad:-pad]
        if bad.size == 0:
            return 0.0
    return float(bad.mean())