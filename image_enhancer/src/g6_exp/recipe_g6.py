"""G6: whole-image quality R&D on top of the Candidate A production baseline.

Isolated experiment. Does NOT modify enhance.py. Billboard-region processing
reuses Candidate A's own functions completely unchanged
(enhance._final_candidate_a_board / enhance._final_composite_box_bgr) so any
whole-image change here is judged in isolation from the already-validated
board pipeline -- we are not re-litigating board quality, only asking whether
the SURROUNDING SCENE can be safely improved further.

Candidates:

  A   baseline: enhance.final_enhance() completely unmodified.
  A+  A's F3 base + one extra, already-proven-safe pass reused verbatim from
      this repo's own G5 experiment family: edge-constrained local-contrast
      recovery, extrema-clipped to the pixel's own local 5x5 [min,max] (no
      new brightness level can ever be created, so no invented texture is
      possible), applied to the WHOLE frame at a mild strength -- not just
      inside billboard boxes.
  B   D1 gains an extra Restormer motion-deblur pass (conservatively blended
      with the existing denoise output) before SwinIR-M SR; F3 fusion
      unchanged. Tests whether real deblurring (not just denoising) at the
      global stage recovers usable environmental/road/vehicle detail.
  C   D1's SR backbone is swapped from SwinIR-M to the larger, still
      PSNR-trained (non-GAN, faithful) SwinIR-L checkpoint already present in
      this repo (phase1_enhancement/models/swinir/..._SwinIR-L_x4_PSNR.pth).
      Tests whether more model capacity alone helps without changing the
      algorithm or introducing a GAN-trained backbone.
  D   Whole image = candidate B's global result; billboard regions =
      Candidate A's own board result, composited onto B's stronger base
      instead of onto plain F3.

All candidates keep: original chroma/geometry untouched outside the
enhancement transforms already used elsewhere in this repo, no OCR/redraw,
no GAN/diffusion, no face work, no invented strokes.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "swinir"))

import enhance  # noqa: E402
from restore_exp import restormer  # noqa: E402
from network_swinir import SwinIR as SwinIRNet  # noqa: E402

# ------------------------------------------------------------- A+ : whole-frame local contrast
# Reused verbatim (same math, same clip) from g5_exp.recipe_g5.local_box_delta,
# generalized to operate on the whole L-plane instead of one box.
_APLUS_STRENGTH = 0.6   # deliberately milder than G5's own 1.0 board strength
_APLUS_CLIP = 6.0        # deliberately tighter than G5's board clip of 10.0
_APLUS_GATE_T0 = 0.35


def whole_frame_local_contrast(bgr, strength=_APLUS_STRENGTH,
                               clip=_APLUS_CLIP, gate_t0=_APLUS_GATE_T0):
    """A+'s extra pass: edge-masked local contrast, hard-clipped to each
    pixel's own local 5x5 [min, max] from THIS SAME image, so it can only
    sharpen contrast that is already there -- it cannot create a new
    brightness level, and therefore cannot invent a stroke or texture."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l = lab[:, :, 0]
    blur = cv2.GaussianBlur(l, (0, 0), 1.0)
    raw = l - blur
    lo = cv2.erode(l, np.ones((5, 5), np.uint8))
    hi = cv2.dilate(l, np.ones((5, 5), np.uint8))
    clipped = np.clip(raw, lo - l, hi - l)
    gate = enhance._final_edge_gate(l, gate_t0)
    d = np.clip(strength * clipped * gate, -clip, clip)
    lab[:, :, 0] = np.clip(l + d, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


# --------------------------------------------------------------- B : D1 + global deblur
_B_DEBLUR_K = 0.35  # blend weight of the deblurred result into D1's input


def d1_denoise_deblur(orig, device):
    """Candidate B's D1: like enhance._final_d1_weak, but the denoised image
    ALSO gets a conservative Restormer motion-deblur pass blended in before
    SwinIR-M SR, instead of denoise alone."""
    from pro_exp import pipeline2
    from restore_exp import swinir_m
    dn_model = restormer.load("real_denoise", device)
    dn_img = pipeline2.tiled_restore(dn_model, orig,
                                     tile=enhance._FINAL_DN_TILE,
                                     overlap=enhance._FINAL_DN_OVERLAP)
    deblur_model = restormer.load("motion_deblur", device)
    db_img = pipeline2.tiled_restore(deblur_model, orig,
                                     tile=enhance._FINAL_DN_TILE,
                                     overlap=enhance._FINAL_DN_OVERLAP)
    k = enhance._FINAL_D1_K
    kb = _B_DEBLUR_K
    blend = (orig.astype(np.float32) * (1.0 - k - kb)
             + dn_img.astype(np.float32) * k
             + db_img.astype(np.float32) * kb)
    blend = np.clip(blend, 0, 255).astype(np.uint8)
    sr = swinir_m.sr_bgr(blend, device, tile=enhance._FINAL_SR_TILE)
    out = cv2.resize(sr, (enhance.OUT_W, enhance.OUT_H),
                     interpolation=cv2.INTER_LANCZOS4)
    return out


# --------------------------------------------------------------- C : SwinIR-L backbone
_SWINIR_L_PTH = "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_PSNR.pth"
_swinir_l_cache = {}


def _swinir_l_build(device):
    if device in _swinir_l_cache:
        return _swinir_l_cache[device]
    path = ROOT / "models" / "swinir" / _SWINIR_L_PTH
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = sd.get("params_ema", sd.get("params", sd))
    model = SwinIRNet(upscale=4, in_chans=3, img_size=64, window_size=8,
                      img_range=1., depths=[6] * 9, embed_dim=240,
                      num_heads=[8] * 9, mlp_ratio=2,
                      upsampler="nearest+conv", resi_connection="3conv")
    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    model.eval().to(device)
    _swinir_l_cache[device] = model
    return model


def _swinir_l_sr(bgr, device, tile=192, overlap=16):
    """4x super-resolve via SwinIR-L. Same tiling scheme as swinir_m.sr_bgr,
    reusing its input prep / tiled-forward helpers (only the model differs)."""
    from restore_exp import swinir_m
    t, (h0, w0) = swinir_m.prep_input(bgr)
    t = t.to(device)
    b, c, h, w = t.size()
    tile = min(tile, h, w)
    stride = tile - overlap
    h_idx = list(range(0, h - tile, stride)) + [h - tile]
    w_idx = list(range(0, w - tile, stride)) + [w - tile]
    SCALE = 4
    model = _swinir_l_build(device)
    E = torch.zeros(b, c, h * SCALE, w * SCALE, device=device)
    Wt = torch.zeros_like(E)
    with torch.no_grad():
        for hi in h_idx:
            for wi in w_idx:
                patch = t[..., hi:hi + tile, wi:wi + tile]
                out = model(patch)
                E[..., hi * SCALE:(hi + tile) * SCALE,
                  wi * SCALE:(wi + tile) * SCALE].add_(out)
                Wt[..., hi * SCALE:(hi + tile) * SCALE,
                   wi * SCALE:(wi + tile) * SCALE].add_(torch.ones_like(out))
    out = E.div_(Wt)[..., :h0 * SCALE, :w0 * SCALE]
    rgb = out.squeeze(0).float().cpu().clamp_(0, 1).numpy()
    return (rgb[[2, 1, 0]].transpose(1, 2, 0) * 255.0).round().astype(np.uint8)


def d1_swinir_l(orig, device):
    """Candidate C's D1: identical to production except the SR backbone is
    SwinIR-L (PSNR, non-GAN) instead of SwinIR-M."""
    from pro_exp import pipeline2
    dn_model = restormer.load("real_denoise", device)
    dn_img = pipeline2.tiled_restore(dn_model, orig,
                                     tile=enhance._FINAL_DN_TILE,
                                     overlap=enhance._FINAL_DN_OVERLAP)
    k = enhance._FINAL_D1_K
    blend = (orig.astype(np.float32) * (1.0 - k)
             + dn_img.astype(np.float32) * k)
    blend = np.clip(blend, 0, 255).astype(np.uint8)
    sr = _swinir_l_sr(blend, device, tile=192)
    out = cv2.resize(sr, (enhance.OUT_W, enhance.OUT_H),
                     interpolation=cv2.INTER_LANCZOS4)
    return out


# ------------------------------------------------------------------- board pass (shared)
def apply_boards(frame_bgr, orig, device):
    """Composite Candidate A's OWN, already-validated board pipeline onto
    `frame_bgr` (whichever whole-image candidate produced it). Identical to
    enhance.final_enhance's box loop, extracted so every candidate here can
    reuse it unmodified."""
    H0, W0 = orig.shape[:2]
    boxes = enhance._billboard_boxes(orig)
    boxes4, sigmas, defocus_boxes = [], [], []
    out = frame_bgr.copy()
    for (x, y, w, h) in boxes:
        vb = enhance._final_valid_box((x, y, w, h), W0, H0)
        if vb is None:
            continue
        vx, vy, vw, vh = vb
        bx, by, bw, bh = enhance._final_scale_box(vb, W0)
        xs = min(bx + bw, out.shape[1])
        ys = min(by + bh, out.shape[0])
        if xs - bx < 8 or ys - by < 8:
            continue
        box4 = (bx, by, xs - bx, ys - by)
        boxes4.append(box4)
        nc = orig[vy:vy + vh, vx:vx + vw]
        sig = enhance._final_psf_sigma(cv2.cvtColor(nc, cv2.COLOR_BGR2GRAY))
        sigmas.append(sig)
        if sig >= enhance._FINAL_DEFOCUS_SIGMA:
            defocus_boxes.append(box4)
        board_bgr = enhance._final_candidate_a_board(nc, device)
        out = enhance._final_composite_box_bgr(out, board_bgr, box4)
    return out, {"boxes4": boxes4, "psf_sigmas": sigmas,
                "defocus_boxes": defocus_boxes}


# --------------------------------------------------------------------- candidates
def candidate_a(orig, target=(3840, 2160)):
    """Baseline: production enhance.final_enhance(), unmodified."""
    out, stages = enhance.final_enhance(orig, target=target, return_stages=True)
    return out, stages


def candidate_a_plus(orig, target=(3840, 2160)):
    faithful = enhance.simple_upscale(orig, target)
    d1, _peak, device = enhance._final_d1_weak(orig)
    f3 = enhance._final_f3_natural(faithful, d1)
    f3_plus = whole_frame_local_contrast(f3)
    device_t = torch.device(device)
    out, stages = apply_boards(f3_plus, orig, device_t)
    return out, stages


def candidate_b(orig, target=(3840, 2160)):
    faithful = enhance.simple_upscale(orig, target)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d1 = d1_denoise_deblur(orig, device)
    f3 = enhance._final_f3_natural(faithful, d1)
    out, stages = apply_boards(f3, orig, device)
    return out, stages


def candidate_c(orig, target=(3840, 2160)):
    faithful = enhance.simple_upscale(orig, target)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d1 = d1_swinir_l(orig, device)
    f3 = enhance._final_f3_natural(faithful, d1)
    out, stages = apply_boards(f3, orig, device)
    return out, stages


def candidate_d(orig, target=(3840, 2160)):
    """Whole image from candidate B's global pipeline; boards from Candidate
    A's own board pipeline, composited onto B's base instead of plain F3."""
    faithful = enhance.simple_upscale(orig, target)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d1 = d1_denoise_deblur(orig, device)
    f3_b = enhance._final_f3_natural(faithful, d1)
    out, stages = apply_boards(f3_b, orig, device)
    return out, stages
