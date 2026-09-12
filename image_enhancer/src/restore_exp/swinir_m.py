"""SwinIR-M PSNR (x4 real-world SR) runner, mirroring src/swinir/experiment.

Uses the official vendored network_swinir.py and the official, byte-verified
realSR PSNR checkpoint already used in the completed SwinIR-X experiment.
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "swinir"))

from network_swinir import SwinIR as SwinIRNet

SCALE = 4
WINDOW_SIZE = 8
PTH = "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_PSNR.pth"
BYTES = 67129849

_cache = {}


def build(device):
    if device in _cache:
        return _cache[device]
    model = SwinIRNet(
        upscale=SCALE, in_chans=3, img_size=64, window_size=WINDOW_SIZE,
        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
        num_heads=[6] * 6, mlp_ratio=2, upsampler="nearest+conv",
        resi_connection="1conv")
    path = ROOT / "models" / "swinir" / PTH
    if not path.exists() or path.stat().st_size != BYTES:
        raise RuntimeError(
            f"SwinIR-M PSNR weight mismatch: expected {BYTES} bytes, found "
            f"{path.stat().st_size if path.exists() else 0}. Refusing a "
            "possibly-tampered/unofficial checkpoint.")
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = sd.get("params_ema", sd)
    model.load_state_dict(sd, strict=True)
    model.eval().to(device)
    _cache[device] = model
    return model


def prep_input(lq_bgr):
    lq = lq_bgr.astype(np.float32) / 255.0
    lq = lq[:, :, [2, 1, 0]]
    t = torch.from_numpy(lq).float().permute(2, 0, 1).unsqueeze(0)
    _, _, h0, w0 = t.size()
    hp = (h0 // WINDOW_SIZE + 1) * WINDOW_SIZE - h0
    wp = (w0 // WINDOW_SIZE + 1) * WINDOW_SIZE - w0
    t = torch.cat([t, torch.flip(t, [2])], 2)[:, :, :h0 + hp, :]
    t = torch.cat([t, torch.flip(t, [3])], 3)[:, :, :, :w0 + wp]
    return t, (h0, w0)


def tiled_forward(model, img_lq, tile=400, overlap=32):
    b, c, h, w = img_lq.size()
    tile = min(tile, h, w)
    stride = tile - overlap
    h_idx = list(range(0, h - tile, stride)) + [h - tile]
    w_idx = list(range(0, w - tile, stride)) + [w - tile]
    E = torch.zeros(b, c, h * SCALE, w * SCALE, device=img_lq.device)
    Wt = torch.zeros_like(E)
    with torch.inference_mode():
        for hi in h_idx:
            for wi in w_idx:
                patch = img_lq[..., hi:hi + tile, wi:wi + tile]
                out = model(patch)
                E[..., hi * SCALE:(hi + tile) * SCALE,
                  wi * SCALE:(wi + tile) * SCALE].add_(out)
                Wt[..., hi * SCALE:(hi + tile) * SCALE,
                   wi * SCALE:(wi + tile) * SCALE].add_(torch.ones_like(out))
    return E.div_(Wt)


@torch.inference_mode()
def sr_bgr(bgr, device, tile=400, overlap=32):
    """4x super-resolve a BGR uint8 image, returns uint8 BGR."""
    t, (h0, w0) = prep_input(bgr)
    t = t.to(device)
    out = tiled_forward(build(device), t, tile, overlap)
    out = out[..., :h0 * SCALE, :w0 * SCALE]
    rgb = out.squeeze(0).float().cpu().clamp_(0, 1).numpy()
    bgr_out = (rgb[[2, 1, 0]].transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
    return bgr_out