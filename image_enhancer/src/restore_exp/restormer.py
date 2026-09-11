"""Restormer runner (official arch + official release weights).

Restormer: Efficient Transformer for High-Resolution Image Restoration
(CVPR 2022, swz30/Restormer, MIT). Regression model - no adversarial or
generative head. Supports the real_denoising (SIDD) and motion_deblurring
(GoPro) checkpoints used by this experiment.

The arch file restormer_arch.py is vendored unmodified from the official
repo (basicsr/models/archs/restormer_arch.py). Weights are the
official GitHub release v1.0 files with byte-verified sizes.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from restore_exp.restormer_arch import Restormer

WEIGHTS = {
    "real_denoise": (
        "real_denoising.pth",
        104611957,
        dict(
            inp_channels=3, out_channels=3, dim=48,
            num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
            heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
            bias=False, LayerNorm_type="BiasFree", dual_pixel_task=False,
        ),
        "RealDenoise (SIDD)",
    ),
    "motion_deblur": (
        "motion_deblurring.pth",
        104700429,
        dict(
            inp_channels=3, out_channels=3, dim=48,
            num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
            heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
            bias=False, LayerNorm_type="WithBias", dual_pixel_task=False,
        ),
        "MotionDeblur (GoPro)",
    ),
}

_cache = {}


def load(kind, device):
    if kind in _cache:
        return _cache[kind]
    fname, size, cfg, label = WEIGHTS[kind]
    path = ROOT / "models" / "restore" / fname
    if not path.exists() or path.stat().st_size != size:
        raise RuntimeError(
            f"Restormer weight mismatch for {fname}: expected {size} bytes, "
            f"found {path.stat().st_size if path.exists() else 0}. Refusing a "
            "possibly-tampered/unofficial checkpoint.")
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = sd.get("params", sd)
    model = Restormer(**cfg)
    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, (
        f"Restormer {kind} state dict mismatch: {missing} / {unexpected}")
    model.eval().to(device)
    _cache[kind] = model
    return model


@torch.no_grad()
def enhance_bgr(model, bgr):
    """Same-size restoration (denoise/deblur) of a BGR uint8 image."""
    ph = (8 - bgr.shape[0] % 8) % 8
    pw = (8 - bgr.shape[1] % 8) % 8
    bgr_p = np.pad(bgr, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    lq = bgr_p.astype(np.float32) / 255.0
    t = lq[:, :, [2, 1, 0]].transpose(2, 0, 1)[None]
    t = torch.from_numpy(np.ascontiguousarray(t))
    t = t.to(next(model.parameters()).device)
    out = model(t)
    rgb = out.squeeze(0).float().cpu().clamp_(0, 1).numpy()
    bgr_out = rgb[[2, 1, 0]].transpose(1, 2, 0)
    bgr_out = bgr_out[:bgr.shape[0], :bgr.shape[1]]
    return (bgr_out * 255.0).round().astype(np.uint8)


@torch.no_grad()
def enhance_bgr_timed(model, bgr):
    t0 = time.perf_counter()
    out = enhance_bgr(model, bgr)
    return out, time.perf_counter() - t0