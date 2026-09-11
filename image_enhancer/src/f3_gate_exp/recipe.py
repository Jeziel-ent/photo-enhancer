"""F3 edge-gate threshold sweep (isolated R&D experiment).

Tests the effect of loosening F3's edge-gate threshold t0 while keeping
everything else IDENTICAL to production:

  baseline  t0 = 0.30   (production _FINAL_F3_T0)
  candidate t0 = 0.25
  candidate t0 = 0.20

Kept unchanged (all read directly from production enhance.py, never copied):
  ±14 F3 envelope          (_FINAL_F3_ENV)
  weight W = 6.0           (_FINAL_F3_W)
  inner clip ±18           (_FINAL_F3_CLIP)
  band-pass sigma 2.5      (_FINAL_F3_BP_SIGMA)
  edge norm p97            (_final_edge_norm_97)
  downstream stages        (_final_multiscale_detail / _final_whole_frame_detail)
  models                   (D1-weak pipeline, untouched)

f3_gate_t0() is a byte-for-byte re-implementation of enhance._final_f3_natural
with the sole change being the t0 argument passed to _final_edge_gate().

The gate is  gate = clip((en - t0)/(1 - t0), 0, 1) where en = Sobel
magnitude normalized by the p97 percentile. So t0 is the fraction of the
p97-normalized edge energy below which SR structure is fully suppressed.
Lowering t0 admits weaker/mid-strength source edges (and noise-like
gradients) into F3's enhancement.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import enhance  # noqa: E402  (production, READ-ONLY, unmodified)


def f3_gate_t0(faithful4k, d1_4k, t0):
    """Exact production _final_f3_natural, parameterized ONLY by t0."""
    a_l = enhance._final_lab_l(faithful4k)
    s_l = enhance._final_lab_l(d1_4k)
    gate = enhance._final_edge_gate(a_l, t0)
    sc = (s_l - a_l) - cv2.GaussianBlur(s_l - a_l, (0, 0),
                                        enhance._FINAL_F3_BP_SIGMA)
    dL = (enhance._FINAL_F3_W
          * np.clip(sc, -enhance._FINAL_F3_CLIP, enhance._FINAL_F3_CLIP)
          * gate)
    l_new = a_l + np.clip(dL, -enhance._FINAL_F3_ENV, enhance._FINAL_F3_ENV)
    lab = cv2.cvtColor(faithful4k, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] = np.clip(l_new, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def gate_coverage(lum, t0s=(0.30, 0.25, 0.20)):
    """Fraction of pixels where gate > threshold at each t0, for diagnostics."""
    en = enhance._final_edge_norm_97(lum)
    out = {}
    for t0 in t0s:
        gate = enhance._final_edge_gate(lum, t0)
        out[t0] = {
            "pct_gate_gt_0.05": round(float((gate > 0.05).mean() * 100), 2),
            "pct_gate_full_1.0": round(float((gate >= 0.999).mean() * 100), 2),
            "mean_gate": round(float(gate.mean()), 4),
        }
    return out