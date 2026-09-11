"""G2: Next-generation non-generative image restoration engine.

CORE RULE: ENHANCE -- DON'T GENERATE.
No GANs, no diffusion, no OCR redraws, no face hallucination.
All models use pure regression / L1 loss or classical optical/fusion principles.

Evaluates 4 distinct candidate restoration approaches:
  1. G2-Defocus: Restormer single-image defocus deblurring on native crop -> Lanczos to 4K.
  2. G2-SwinIR-M: Native crop 4x SR via SwinIR-M PSNR (11.7M params, L1 loss).
  3. G2-SwinIR-L: Native crop 4x SR via SwinIR-Large PSNR (28.0M params, 9 stages, 240 dims, L1 loss).
  4. G2-Fusion: SwinIR-L PSNR luminance structure + 100% original photo chroma (a*, b*)
                + bounded envelope + edge transition boost, feathered onto F3-natural.

Everything outside verified billboard boxes from regions.json remains byte-for-byte
identical to F3-natural.
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from restore_exp.restormer_arch import Restormer
from restore_exp import restormer
from swinir.experiment import build_model as build_swinir, MODELS as SWINIR_MODELS
from f2_exp.recipe2 import composite_alpha

FEATHER = 16

_models_cache = {}


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_model(name):
    if name in _models_cache:
        return _models_cache[name]

    device = get_device()
    if name == "defocus":
        cfg = dict(
            inp_channels=3, out_channels=3, dim=48,
            num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
            heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
            bias=False, LayerNorm_type="WithBias", dual_pixel_task=False,
        )
        model = Restormer(**cfg)
        pth = ROOT / "models" / "restore" / "single_image_defocus_deblurring.pth"
        sd = torch.load(pth, map_location="cpu", weights_only=True)
        model.load_state_dict(sd.get("params", sd), strict=True)
        model.eval().to(device)
    elif name == "swinir_m":
        model = build_swinir(SWINIR_MODELS["m_psnr"], device)
    elif name == "swinir_l":
        model = build_swinir(SWINIR_MODELS["l_psnr"], device)
    else:
        raise ValueError(f"Unknown model name: {name}")

    _models_cache[name] = model
    return model


def run_swinir_sr(model, bgr_crop, target_w, target_h):
    """Run SwinIR on a BGR crop (native scale), return BGR at target_w, target_h."""
    device = next(model.parameters()).device
    lq = bgr_crop.astype(np.float32) / 255.0
    lq = lq[:, :, [2, 1, 0]]
    t = torch.from_numpy(lq).float().permute(2, 0, 1).unsqueeze(0).to(device)
    _, _, ch, cw = t.size()
    hp = (8 - ch % 8) % 8
    wp = (8 - cw % 8) % 8
    t_pad = torch.nn.functional.pad(t, (0, wp, 0, hp), mode="reflect")
    with torch.no_grad():
        out_sr = model(t_pad)[..., :ch * 4, :cw * 4]
    sr_rgb = out_sr.squeeze(0).float().cpu().clamp(0, 1).numpy()
    sr_bgr = (sr_rgb[[2, 1, 0]].transpose(1, 2, 0) * 255.0).round().astype(np.uint8)

    interp = cv2.INTER_AREA if sr_bgr.shape[0] > target_h else cv2.INTER_LANCZOS4
    return cv2.resize(sr_bgr, (target_w, target_h), interpolation=interp)


def run_defocus_deblur(model, bgr_crop, target_w, target_h):
    """Run Restormer defocus deblurring on native crop, upscale to target."""
    deb_native = restormer.enhance_bgr(model, bgr_crop)
    return cv2.resize(deb_native, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)


def fuse_structure_and_chroma(orig_4k_crop, sr_4k_crop, env_limit=28.0, edge_boost=0.20):
    """Fuses SR luminance structure with 100% original photo chroma (a*, b*).
    
    Guarantees:
      - Color fidelity: 100% faithful to source photo chroma.
      - Contrast: recovers sharp stroke transitions from non-generative SR.
      - Envelope: bounded to env_limit so no unnatural drift can occur.
    """
    lab_orig = cv2.cvtColor(orig_4k_crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_sr = cv2.cvtColor(sr_4k_crop, cv2.COLOR_BGR2LAB).astype(np.float32)

    l_orig = lab_orig[:, :, 0]
    l_sr = lab_sr[:, :, 0]

    # Luminance delta bounded to envelope
    dl = np.clip(l_sr - l_orig, -env_limit, env_limit)

    # High frequency edge transition boost
    blur = cv2.GaussianBlur(l_sr, (0, 0), 1.0)
    hf = l_sr - blur

    l_final = np.clip(l_orig + dl + edge_boost * hf, 0, 255)
    lab_out = np.stack([l_final, lab_orig[:, :, 1], lab_orig[:, :, 2]], axis=-1)
    return cv2.cvtColor(np.clip(lab_out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def apply_candidate_box(orig_bgr, f3_natural_bgr, boxes_native, boxes_4k, candidate_fn, feather=FEATHER):
    """Apply candidate processing only inside verified billboard boxes.
    Feather the box boundaries onto F3-natural base. Outside boxes is 100% F3-natural.
    """
    out = f3_natural_bgr.copy()
    for (bn, b4) in zip(boxes_native, boxes_4k):
        nx, ny, nw, nh = bn
        tx, ty, tw, th = b4
        if nw < 4 or nh < 4 or tw < 8 or th < 8:
            continue

        native_crop = orig_bgr[ny:ny + nh, nx:nx + nw]
        faithful_4k_crop = cv2.resize(native_crop, (tw, th), interpolation=cv2.INTER_LANCZOS4)

        restored_box = candidate_fn(native_crop, faithful_4k_crop, tw, th)

        alpha = composite_alpha(th, tw, max(2, int(feather)))
        if alpha.ndim == 2:
            alpha = alpha[:, :, None]

        base_box = out[ty:ty + th, tx:tx + tw].astype(np.float32)
        if restored_box.shape[:2] != (th, tw):
            restored_box = cv2.resize(restored_box, (tw, th), interpolation=cv2.INTER_LANCZOS4)

        blended = base_box * (1.0 - alpha) + restored_box.astype(np.float32) * alpha
        out[ty:ty + th, tx:tx + tw] = np.clip(blended, 0, 255).astype(np.uint8)

    return out


def g2_defocus(orig_bgr, f3_natural_bgr, boxes_native, boxes_4k):
    model = get_model("defocus")
    return apply_candidate_box(
        orig_bgr, f3_natural_bgr, boxes_native, boxes_4k,
        lambda nc, fc, tw, th: run_defocus_deblur(model, nc, tw, th)
    )


def g2_swinir_m(orig_bgr, f3_natural_bgr, boxes_native, boxes_4k):
    model = get_model("swinir_m")
    return apply_candidate_box(
        orig_bgr, f3_natural_bgr, boxes_native, boxes_4k,
        lambda nc, fc, tw, th: run_swinir_sr(model, nc, tw, th)
    )


def g2_swinir_l(orig_bgr, f3_natural_bgr, boxes_native, boxes_4k):
    model = get_model("swinir_l")
    return apply_candidate_box(
        orig_bgr, f3_natural_bgr, boxes_native, boxes_4k,
        lambda nc, fc, tw, th: run_swinir_sr(model, nc, tw, th)
    )


def g2_fusion(orig_bgr, f3_natural_bgr, boxes_native, boxes_4k):
    model = get_model("swinir_l")

    def _fuse_proc(nc, fc, tw, th):
        sr_box = run_swinir_sr(model, nc, tw, th)
        return fuse_structure_and_chroma(fc, sr_box, env_limit=28.0, edge_boost=0.20)

    return apply_candidate_box(
        orig_bgr, f3_natural_bgr, boxes_native, boxes_4k,
        _fuse_proc
    )
