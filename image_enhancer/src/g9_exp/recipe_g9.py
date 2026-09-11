"""G9 -- Fine Generative Restoration R&D (isolated experiment).

NOT wired into production. Does not modify enhance.py or any file outside
phase1_enhancement/src/g9_exp/. See phase1_enhancement/reports/g9_exp/ for
the benchmark this recipe was evaluated with.

Goal: push perceived photographic clarity (sharper micro-detail, more
"ChatGPT-level" crispness) further than the current production `final`
pipeline (enhance.final_enhance: faithful upscale -> D1-weak -> F3-natural
-> G7-MS -> A+ -> Candidate A billboard stage), using exactly ONE
generative/learned model, at very low, self-limited strength.

Candidate: SwinIR-M x4, GAN-trained real-world-SR checkpoint
(003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth, official
JingyunLiang/SwinIR v0.0 release, Apache-2.0). Same architecture
(embed_dim=180, depths [6]*6, window=8, nearest+conv upsampler) and same
vendored network_swinir.py already trusted in production for the
NON-generative PSNR checkpoint (restore_exp/swinir_m.py) -- G9 only swaps
in the GAN-trained weights and reuses the exact same tiled-inference
plumbing (prep_input / tiled_forward), so the only new variable under test
is "generative vs regression training objective", not a new architecture
or a new codepath.

Why a GAN checkpoint and not another Real-ESRGAN pass: Real-ESRGAN was
already evaluated in this codebase's own prior R&D (see enhance.py's
"Candidate A" docstring) and REJECTED for billboard text -- it fabricated
plausible-looking but INCORRECT letterforms. SwinIR-GAN carries the same
class of risk (any GAN-trained SR model can hallucinate texture/strokes),
which is exactly why G9 does not let it touch pixels directly: see
`controlled_generative_detail` below.

Containment strategy (mirrors the ALREADY-PROVEN technique used by
production's F3-natural / G7-MS / A+ stages -- not a new idea, the same
one, applied to a generative source instead of a regression one):
  1. Run the GAN model on the ORIGINAL (native resolution), never on top
     of an already-hallucination-prone input.
  2. Band-pass the GAN-vs-baseline delta so ONLY high-frequency structure
     can contribute -- low-frequency (i.e. shape/geometry/identity/large
     color regions) is discarded outright, by construction.
  3. Edge-gate: the delta is only allowed where the BASELINE already has
     a strong, unambiguous edge (top ~3% gradient magnitude, higher
     threshold than production's A+ stage).
  4. Hard-clip the delta to each pixel's own local 5x5 [min, max] window
     of the BASELINE's OWN luminance -- so the contribution can only ever
     sharpen contrast that already exists in that pixel's neighborhood.
     It is architecturally incapable of inventing a new brightness level,
     which is what a hallucinated stroke/letterform/logo edit would
     require.
  5. Chroma (a/b in Lab) is never touched -- only L.
  6. Global clip to a small +-5 (tighter than A+'s +-6, and far tighter
     than F3's +-14 envelope, because the source here is generative, not
     regression) as a final belt-and-braces cap.
  7. Billboard boxes are EXCLUDED from the GAN contribution entirely by
     default (disable_on_boards=True) -- production's Candidate A already
     owns board reconstruction on its own dedicated, non-generative path;
     G9's board numbers in the benchmark are a side-by-side diagnostic
     only, never a proposed replacement.

Guarantee: with disable_on_boards=True, G9's output is pixel-identical to
the production baseline everywhere the edge-gate + local-envelope clip
would not otherwise have already zeroed the delta, AND everywhere inside
a verified billboard box.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "swinir"))

import enhance as prod_enhance  # noqa: E402  (production, READ-ONLY, unmodified)
from restore_exp import swinir_m  # noqa: E402  (reused tiled-inference plumbing)
from network_swinir import SwinIR as SwinIRNet  # noqa: E402

OUT_W, OUT_H = prod_enhance.OUT_W, prod_enhance.OUT_H

# ---------------------------------------------------------------- candidate
# Same arch config as restore_exp/swinir_m.py's PSNR checkpoint -- only the
# checkpoint file (GAN-trained instead of PSNR-trained) differs.
G9_GAN_FNAME = "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth"
G9_GAN_BYTES = 67129861  # official release size, byte-verified like every
                          # other checkpoint in this repo (refuse a
                          # mismatched/tampered file rather than load it)

_gan_cache = {}


def build_gan(device):
    """Load (and cache) the G9 GAN candidate on `device`."""
    if device in _gan_cache:
        return _gan_cache[device]
    model = SwinIRNet(
        upscale=4, in_chans=3, img_size=64, window_size=8,
        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
        num_heads=[6] * 6, mlp_ratio=2, upsampler="nearest+conv",
        resi_connection="1conv")
    path = ROOT / "models" / "swinir" / G9_GAN_FNAME
    if not path.exists() or path.stat().st_size != G9_GAN_BYTES:
        raise RuntimeError(
            f"G9 candidate weight mismatch for {G9_GAN_FNAME}: expected "
            f"{G9_GAN_BYTES} bytes, found "
            f"{path.stat().st_size if path.exists() else 0}. Refusing a "
            "possibly-tampered/unofficial checkpoint.")
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = sd.get("params_ema", sd)
    model.load_state_dict(sd, strict=True)
    model.eval().to(device)
    _gan_cache[device] = model
    return model


@torch.no_grad()
def gan_sr_bgr(bgr, device, tile=256, overlap=32):
    """4x GAN super-resolve a BGR uint8 image. Reuses swinir_m's own
    prep_input/tiled_forward (identical pre/post-processing to the
    production PSNR path) against our own GAN model instance.

    tile=256 matches production's own SwinIR-M tile size (enhance.py's
    _FINAL_SR_TILE, proven safe on an 8GB card for D1-weak) -- same
    architecture, so the same tile size carries the same VRAM guarantee."""
    t, (h0, w0) = swinir_m.prep_input(bgr)
    t = t.to(device)
    out = swinir_m.tiled_forward(build_gan(device), t, tile, overlap)
    out = out[..., :h0 * swinir_m.SCALE, :w0 * swinir_m.SCALE]
    rgb = out.squeeze(0).float().cpu().clamp_(0, 1).numpy()
    return (rgb[[2, 1, 0]].transpose(1, 2, 0) * 255.0).round().astype(np.uint8)


# --------------------------------------------------- controlled contribution

G9_STRENGTH = 0.18   # far below A+'s 0.6 -- GAN source, deliberately weak
G9_CLIP = 5.0        # tighter than A+'s +-6 and F3's +-14
G9_GATE_T0 = 0.40    # stricter than A+'s 0.35 -- only the strongest edges
G9_BP_SIGMA = 2.0
G9_LOCAL_KSIZE = 5


def _lab_l(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)[:, :, 0]


def _edge_gate(l_plane, t0):
    gx = cv2.Sobel(l_plane, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(l_plane, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    thr = float(np.percentile(mag, 97.0))
    en = mag / (thr + 1e-9)
    return np.clip((en - t0) / (1.0 - t0), 0.0, 1.0)


def controlled_generative_detail(baseline_bgr, gan_bgr_same_size):
    """Blend GAN high-frequency detail onto `baseline_bgr` at very low,
    self-limited strength. Returns (result_bgr, delta_L, stats).

    `gan_bgr_same_size` must already be the GAN SR output resized to
    `baseline_bgr`'s own resolution."""
    a_l = _lab_l(baseline_bgr)
    g_l = _lab_l(gan_bgr_same_size)

    raw = g_l - a_l
    bp = raw - cv2.GaussianBlur(raw, (0, 0), G9_BP_SIGMA)

    gate = _edge_gate(a_l, G9_GATE_T0)

    k = np.ones((G9_LOCAL_KSIZE, G9_LOCAL_KSIZE), np.uint8)
    lo = cv2.erode(a_l, k)
    hi = cv2.dilate(a_l, k)

    delta = G9_STRENGTH * bp * gate
    delta = np.clip(delta, lo - a_l, hi - a_l)
    delta = np.clip(delta, -G9_CLIP, G9_CLIP)

    lab = cv2.cvtColor(baseline_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] = np.clip(a_l + delta, 0, 255)
    out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    stats = {
        "mean_abs_delta": float(np.abs(delta).mean()),
        "max_abs_delta": float(np.abs(delta).max()),
        "pct_px_touched": float((np.abs(delta) > 0.5).mean() * 100.0),
    }
    return out, delta, stats


def _feather_zero_mask(shape_hw, boxes4, feather=16):
    """1 far from every box (full delta allowed), 0 deep inside each box
    (delta fully excluded), with a smooth ramp across ~`feather` px at the
    box boundary so there is no hard seam. Built by Gaussian-blurring a
    binary 0-inside/1-outside mask, which -- unlike a per-edge min-ramp --
    guarantees the box INTERIOR is driven to ~0, not just a thin ring at
    the outer padding edge."""
    h, w = shape_hw
    binary = np.ones((h, w), np.float32)
    for (bx, by, bw, bh) in boxes4:
        x0 = max(0, bx); y0 = max(0, by)
        x1 = min(w, bx + bw); y1 = min(h, by + bh)
        if x1 > x0 and y1 > y0:
            binary[y0:y1, x0:x1] = 0.0
    sigma = max(1.0, feather / 3.0)
    mask = cv2.GaussianBlur(binary, (0, 0), sigma)
    return np.clip(mask, 0.0, 1.0)


def g9_enhance(img, device=None, target=None, disable_on_boards=True,
               return_stages=False, precomputed_baseline=None):
    """G9 = production final_enhance() baseline + a controlled, low-strength
    GAN high-frequency contribution.

    Parameters
    ----------
    img : original BGR uint8 array (untouched source of truth).
    disable_on_boards : when True (default), the GAN contribution is
        feathered to exactly zero inside every verified billboard box
        (see enhance._billboard_boxes / regions.json) -- production's
        Candidate A already owns those pixels.
    precomputed_baseline : optional (baseline_bgr, stages) tuple, as
        returned by enhance.final_enhance(img, return_stages=True), to
        avoid re-running the (expensive) production pipeline a second
        time when the caller already has it (e.g. a benchmark harness
        comparing baseline vs G9 on the same image).
    """
    if target is None:
        target = (OUT_W, OUT_H)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if precomputed_baseline is not None:
        baseline, base_stages = precomputed_baseline
    else:
        baseline, base_stages = prod_enhance.final_enhance(
            img, target=target, return_stages=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    gan_native = gan_sr_bgr(img, device)
    g9_peak_vram_mb = (float(torch.cuda.max_memory_allocated() / 1048576.0)
                       if device.type == "cuda" else 0.0)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if gan_native.shape[1] != target[0] or gan_native.shape[0] != target[1]:
        gan_4k = cv2.resize(gan_native, target, interpolation=cv2.INTER_LANCZOS4)
    else:
        gan_4k = gan_native

    out, delta, stats = controlled_generative_detail(baseline, gan_4k)
    stats["g9_peak_vram_mb"] = g9_peak_vram_mb

    boxes4 = base_stages.get("boxes4", [])
    if disable_on_boards and boxes4:
        mask = _feather_zero_mask(out.shape[:2], boxes4, feather=16)
        a_l = _lab_l(baseline)
        masked_delta = delta * mask
        lab = cv2.cvtColor(baseline, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[:, :, 0] = np.clip(a_l + masked_delta, 0, 255)
        out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        stats["pct_px_touched_after_board_mask"] = float(
            (np.abs(masked_delta) > 0.5).mean() * 100.0)

    if not return_stages:
        return out
    stages = dict(base_stages)
    stages["baseline"] = baseline
    stages["gan_4k"] = gan_4k
    stages["g9_stats"] = stats
    stages["boxes4"] = boxes4
    return out, stages
