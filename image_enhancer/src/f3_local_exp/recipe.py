"""F3 locally-derived self-envelope experiment.

Replaces F3's flat ±14 luminance envelope with a per-pixel locally-derived
self-envelope using morphological erode/dilate on the original L-plane.
This is the same containment philosophy proven in G7-MS, A+, and G9:
the enhancement delta at each pixel is clipped to the range of brightness
levels that already exist in that pixel's local neighborhood.

Changes ONLY the outer envelope clip. All other F3 components preserved:
  - Band-pass structure term (Gaussian subtraction at sigma=2.5)
  - Per-pixel inner clip (±18)
  - Edge gate (Sobel p97-normalized, t0=0.30)
  - Weight (6.0x)
  - Downstream stages (G7-MS, A+, Candidate A) untouched

Variant A (ksize=3): tightest self-envelope, local [min,max] at 3×3.
  At edges with local contrast C, allows only C L-levels of movement.
  Much tighter than flat ±14 at high-contrast edges.

Variant B (ksize=5, global_cap=10): self-envelope with safety cap.
  Local envelope provides the adaptivity; the global cap catches
  extreme cases where the local range is still too wide.

Variant C (ksize=5, blend=0.5): hybrid — 50% self-envelope + 50% flat.
  Leverages local adaptivity while retaining the flat cap's
  edge-overshoot protection.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fidelity_exp.recipe import _edge_norm  # noqa: E402

BP_SIGMA = 2.5     # band-pass sigma (same as F3 baseline)
F3_ENV = 14.0       # baseline flat envelope for hybrid variant


def _lab_l(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)[:, :, 0]


def _compose_from_original(a_bgr, l_new):
    lab = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] = np.clip(l_new, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def _edge_gate(l, t0):
    en = _edge_norm(l, pnorm=97.0)
    return np.clip((en - t0) / (1.0 - t0), 0.0, 1.0)


def _structure_term(s_l, a_l):
    sc = s_l - a_l
    return sc - cv2.GaussianBlur(sc, (0, 0), BP_SIGMA)


def _compute_dL(a_l, s_l, w_s, clip, t0):
    """Compute the raw weighted+gated delta (same for all variants)."""
    gate = _edge_gate(a_l, t0)
    sc = _structure_term(s_l, a_l)
    return w_s * np.clip(sc, -clip, clip) * gate


# ----------------------------------------------------- Variant A: tight self-env
def f3_self_env_tight(a_bgr, s_bgr, w_s=6.0, clip=18.0, t0=0.30, ksize=3):
    """Self-envelope with tight 3×3 kernel. Local range is ~5-10 L-levels
    at typical edges, much tighter than flat ±14."""
    a_l = _lab_l(a_bgr)
    s_l = _lab_l(s_bgr)
    dL = _compute_dL(a_l, s_l, w_s, clip, t0)

    k = np.ones((ksize, ksize), np.uint8)
    lo = cv2.erode(a_l, k)
    hi = cv2.dilate(a_l, k)
    l_new = a_l + np.clip(dL, lo - a_l, hi - a_l)
    return _compose_from_original(a_bgr, l_new)


# --------------------------------------------- Variant B: self-env + global cap
def f3_self_env_capped(a_bgr, s_bgr, w_s=6.0, clip=18.0, t0=0.30,
                       ksize=5, global_cap=10.0):
    """Self-envelope with global safety cap. The self-envelope provides
    local adaptivity; the global cap catches extreme cases."""
    a_l = _lab_l(a_bgr)
    s_l = _lab_l(s_bgr)
    dL = _compute_dL(a_l, s_l, w_s, clip, t0)

    k = np.ones((ksize, ksize), np.uint8)
    lo = cv2.erode(a_l, k)
    hi = cv2.dilate(a_l, k)
    l_new = a_l + np.clip(np.clip(dL, lo - a_l, hi - a_l),
                          -global_cap, global_cap)
    return _compose_from_original(a_bgr, l_new)


# ------------------------------------------------------- Variant C: hybrid blend
def f3_hybrid(a_bgr, s_bgr, w_s=6.0, clip=18.0, t0=0.30,
              ksize=5, blend=0.5):
    """Hybrid: 50% self-envelope + 50% flat cap. The envelope provides
    local adaptivity; the flat cap provides edge-overshoot protection."""
    a_l = _lab_l(a_bgr)
    s_l = _lab_l(s_bgr)
    dL = _compute_dL(a_l, s_l, w_s, clip, t0)

    k = np.ones((ksize, ksize), np.uint8)
    lo = cv2.erode(a_l, k)
    hi = cv2.dilate(a_l, k)
    env_delta = np.clip(dL, lo - a_l, hi - a_l)
    flat_delta = np.clip(dL, -F3_ENV, F3_ENV)
    l_new = a_l + blend * env_delta + (1.0 - blend) * flat_delta
    return _compose_from_original(a_bgr, l_new)


# ------------------------------------ Variant D: local-contrast envelope capped at 14
def f3_local_contrast_env(a_bgr, s_bgr, w_s=6.0, clip=18.0, t0=0.30,
                          ksize=15, k_scale=1.0, hard_cap=14.0):
    """Self-envelope from LOCAL luminance variance, always <= hard_cap.

    The envelope at each pixel is proportional to the local standard
    deviation of the ORIGINAL L-plane (Gaussian-smoothed variance in a
    ksize window). Where contrast is high (strong edges), the envelope
    saturates at the baseline ±14 cap — identical to the flat envelope.
    Where contrast is low (smooth gradients, mid-partial edges), it
    tightens proportionally so the delta can never out-run the local
    texture that already exists — the halo-prevention property.

    This is the ONLY interpretation of "locally-derived" that is provably
    never wider than the baseline flat ±14 envelope.
    """
    a_l = _lab_l(a_bgr)
    s_l = _lab_l(s_bgr)
    dL = _compute_dL(a_l, s_l, w_s, clip, t0)

    # local variance via E[L^2] - E[L]^2
    mean = cv2.GaussianBlur(a_l, (0, 0), ksize / 6.0)
    mean_sq = cv2.GaussianBlur(a_l * a_l, (0, 0), ksize / 6.0)
    local_var = np.maximum(mean_sq - mean * mean, 0.0)
    local_std = np.sqrt(local_var)

    env = np.clip(k_scale * local_std, 0.0, hard_cap)
    env = np.maximum(env, 0.5)  # floor so truly flat pixels keep a tiny ceiling
    l_new = a_l + np.clip(dL, -env, env)
    return _compose_from_original(a_bgr, l_new)


# Map variant names to (function, params)
VARIANTS = {
    "self_env_k3": (f3_self_env_tight, {"ksize": 3}),
    "self_env_k5_cap10": (f3_self_env_capped, {"ksize": 5, "global_cap": 10.0}),
    "hybrid_50": (f3_hybrid, {"ksize": 5, "blend": 0.5}),
    "local_contrast_env": (f3_local_contrast_env, {"ksize": 15, "k_scale": 1.0}),
}
