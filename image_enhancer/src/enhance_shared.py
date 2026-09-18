"""Torch-free pipeline stages shared between the GPU path (enhance.py) and
the CPU path (backend/cpu_worker.py, via imdn_x4_ov.py).

Why this module exists, and why it must NEVER import torch: this session's
investigation into the packaged EXE's CPU-mode crash (STATUS_ACCESS_VIOLATION,
segfault) found the actual root cause was torch and OpenVINO's native CPU
runtimes conflicting when loaded into the SAME process (almost certainly a
duplicate/incompatible OpenMP or TBB thread-pool initialization) --
verified by direct repeated-call stress tests: `enhance.py`'s
`_final_d1_weak` (which imports torch) crashed within 4 calls when paired
with OpenVINO's IMDN SR; the exact same F3/G7-MS/A+ post-processing logic,
called from a script that imports `imdn_x4_ov` but never imports `torch`
at all, ran 20/20 clean at similar speed. So `backend/cpu_worker.py`
imports THIS module (and imdn_x4_ov, cv2, numpy, tonal_correction) --
never `enhance.py` itself, which pulls in torch at module scope.

Everything here is copied verbatim from enhance.py's own F3-natural/
G7-MS/A+ implementation (same constants, same math, same clipping) so
CPU and GPU output share identical post-processing -- only the denoise+SR
backbone differs by device (see docs/PERFORMANCE_OPTIMIZATION.md).
enhance.py imports these same names from here rather than redefining them,
so there is exactly one copy of this logic, not two that could drift.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

OUT_W, OUT_H = 3840, 2160


def aspect_preserving_target(width: int, height: int, max_dim: int = OUT_W) -> tuple:
    """Identical formula to enhance.py's own aspect_preserving_target (kept
    as a separate copy here for the same reason every other function in
    this module is duplicated rather than imported -- see this module's own
    docstring: enhance.py imports torch at module scope, and the CPU worker
    process must never import torch at all)."""
    if width <= 0 or height <= 0:
        raise ValueError("aspect_preserving_target expects positive width/height")
    scale = max_dim / max(width, height)
    out_w = round(width * scale)
    out_h = round(height * scale)
    out_w = max(2, out_w - (out_w % 2))
    out_h = max(2, out_h - (out_h % 2))
    return (out_w, out_h)


def load_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def simple_upscale(img, target=(OUT_W, OUT_H)):
    """Faithful upscale of the original with NO restoration/enhancement.
    This is the preserve-only baseline used for fidelity comparisons."""
    return cv2.resize(img, target, interpolation=cv2.INTER_LANCZOS4)


def save_image(img, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise IOError(f"Failed to write image: {path}")
    return path


_FINAL_F3_W = 6.0
_FINAL_F3_CLIP = 18.0
_FINAL_F3_T0 = 0.25
_FINAL_F3_ENV = 14.0
_FINAL_F3_BP_SIGMA = 2.5
_FINAL_APLUS_STRENGTH = 0.6
_FINAL_APLUS_CLIP = 6.0
_FINAL_APLUS_T0 = 0.35


def _final_lab_l(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)[:, :, 0]


def _final_edge_norm_97(lum):
    gx = cv2.Sobel(lum, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(lum, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    thr = float(np.percentile(mag, 97.0))
    return mag / (thr + 1e-9)


def _final_edge_gate(l, t0):
    en = _final_edge_norm_97(l)
    return np.clip((en - t0) / (1.0 - t0), 0.0, 1.0)


def final_f3_natural(faithful4k, d1_4k):
    """F3-natural (proven): original photo + band-passed D1 structure on
    strong source edges only, original chroma, +-14 envelope."""
    a_l = _final_lab_l(faithful4k)
    s_l = _final_lab_l(d1_4k)
    gate = _final_edge_gate(a_l, _FINAL_F3_T0)
    sc = (s_l - a_l) - cv2.GaussianBlur(s_l - a_l, (0, 0), _FINAL_F3_BP_SIGMA)
    dL = _FINAL_F3_W * np.clip(sc, -_FINAL_F3_CLIP, _FINAL_F3_CLIP) * gate
    l_new = a_l + np.clip(dL, -_FINAL_F3_ENV, _FINAL_F3_ENV)
    lab = cv2.cvtColor(faithful4k, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] = np.clip(l_new, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


_FINAL_MS_GAINS = (1.30, 1.20, 1.15)
_FINAL_MS_SIGMAS = (1.0, 2.0, 4.0)
_FINAL_MS_KSIZES = (3, 5, 9)
_FINAL_MS_GATE_T0 = 0.30
_FINAL_MS_CLIP = 10.0


def final_multiscale_detail(bgr):
    """G7-MS (validated in phase1_enhancement/src/g7_exp/recipe_g7.py as
    `multiscale_detail_boost`): 3-level DoG/Laplacian-style multi-scale
    detail recovery on TOP of F3, BEFORE the A+ whole-frame detail pass.
    Parameters ported verbatim from the g7_exp sweep -- do not retune here."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l = lab[:, :, 0]
    g1 = cv2.GaussianBlur(l, (0, 0), _FINAL_MS_SIGMAS[0])
    g2 = cv2.GaussianBlur(l, (0, 0), _FINAL_MS_SIGMAS[1])
    g3 = cv2.GaussianBlur(l, (0, 0), _FINAL_MS_SIGMAS[2])
    bands = (l - g1, g1 - g2, g2 - g3)
    base = g3

    recon = base.copy()
    for band, gain, ksz in zip(bands, _FINAL_MS_GAINS, _FINAL_MS_KSIZES):
        k = np.ones((ksz, ksz), np.uint8)
        lo = cv2.erode(l, k)
        hi = cv2.dilate(l, k)
        boosted = np.clip(band * gain, lo - l, hi - l)
        recon = recon + boosted

    delta = recon - l
    gate = _final_edge_gate(l, _FINAL_MS_GATE_T0)
    delta = np.clip(delta * gate, -_FINAL_MS_CLIP, _FINAL_MS_CLIP)
    lab[:, :, 0] = np.clip(l + delta, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def final_whole_frame_detail(bgr):
    """A+ (proven in g6_exp): edge-masked local contrast on TOP of F3, hard-
    clipped to each pixel's own local 5x5 [min, max] from this SAME image.
    Exact implementation and parameters (strength 0.6, clip +-6, gate
    t0=0.35) reused unchanged from g6_exp/recipe_g6.py's winning candidate."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l = lab[:, :, 0]
    blur = cv2.GaussianBlur(l, (0, 0), 1.0)
    raw = l - blur
    lo = cv2.erode(l, np.ones((5, 5), np.uint8))
    hi = cv2.dilate(l, np.ones((5, 5), np.uint8))
    clipped = np.clip(raw, lo - l, hi - l)
    gate = _final_edge_gate(l, _FINAL_APLUS_T0)
    d = np.clip(_FINAL_APLUS_STRENGTH * clipped * gate,
               -_FINAL_APLUS_CLIP, _FINAL_APLUS_CLIP)
    lab[:, :, 0] = np.clip(l + d, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


_FINAL_CPU_D1_K = 0.30
_FINAL_CPU_DN_H = 5
_FINAL_CPU_DN_HCOLOR = 5
_FINAL_CPU_SR_OVERLAP = 16


def cpu_final_enhance(img, target=(OUT_W, OUT_H)):
    """The CPU-mode equivalent of enhance.final_enhance -- same pipeline
    shape (tonal correction -> faithful upscale -> denoise+SR blend ->
    F3-natural -> G7-MS -> A+), same shared post-processing functions
    above, but with the GPU path's Restormer+SwinIR-M backbone replaced by
    classical fastNlMeansDenoisingColored + IMDN x4 via OpenVINO (see
    imdn_x4_ov.py and this module's own docstring for why: torch cannot be
    imported anywhere in this call graph without risking the
    torch+OpenVINO native crash this module exists to avoid).

    ``target`` defaults to the classic (OUT_W, OUT_H) so any existing
    direct caller that never passes it keeps byte-identical behavior; pass
    aspect_preserving_target(w0, h0) to preserve the source's own aspect
    ratio (see that function's docstring) -- backend/cpu_worker.py does
    this for every real job.

    No billboard/Candidate-A branch: that stage is dead code for this
    product even on the GPU path (engine_adapter always neutralizes
    billboard regions -- see docs/ENGINE_AUDIT.md), so it is not
    reproduced here.
    """
    import tonal_correction
    from restore_exp import imdn_x4_ov

    if img is None or not isinstance(img, np.ndarray) or img.ndim != 3 \
            or img.shape[2] != 3:
        raise ValueError("cpu_final_enhance expects a BGR uint8 image")
    h0, w0 = img.shape[:2]
    if h0 < 16 or w0 < 16:
        raise ValueError("cpu_final_enhance expects an image of at least 16x16")

    # MVP 2: see enhance.py's identical call -- enhance_photographic_quality
    # only fixes a genuinely broken exposure/contrast defect and is a no-op
    # otherwise (no proactive brightness lift, no HDR tone mapping/detail
    # boost, no color correction -- all of that is now a manual adjustment,
    # applied later via adjustments.py). CPU and GPU paths stay in sync.
    img_toned, _meta = tonal_correction.enhance_photographic_quality(img, return_meta=True)
    faithful = simple_upscale(img_toned, target)

    dn = cv2.fastNlMeansDenoisingColored(
        img_toned, None, _FINAL_CPU_DN_H, _FINAL_CPU_DN_HCOLOR, 7, 21)
    k = _FINAL_CPU_D1_K
    blend = np.clip(img_toned.astype(np.float32) * (1.0 - k)
                     + dn.astype(np.float32) * k, 0, 255).astype(np.uint8)
    sr = imdn_x4_ov.sr_bgr(blend, overlap=_FINAL_CPU_SR_OVERLAP)
    d1 = cv2.resize(sr, target, interpolation=cv2.INTER_LANCZOS4)

    f3 = final_f3_natural(faithful, d1)
    f3_ms = final_multiscale_detail(f3)
    f3_plus = final_whole_frame_detail(f3_ms)

    if f3_plus.shape[1] != target[0] or f3_plus.shape[0] != target[1]:
        f3_plus = cv2.resize(f3_plus, target, interpolation=cv2.INTER_LANCZOS4)
    return f3_plus


def cpu_warmup():
    """Pre-loads IMDN's OpenVINO model/compiled graph before any real
    upload is processed -- see enhance.warmup()'s own docstring for the
    GPU-side equivalent and why this matters for perceived first-job
    latency."""
    from restore_exp import imdn_x4_ov
    dummy = np.zeros((256, 256, 3), dtype=np.uint8)
    imdn_x4_ov.sr_bgr(dummy, overlap=0)
