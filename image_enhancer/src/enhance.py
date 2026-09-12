"""Core image enhancement pipeline for phase 1.

Only restoration/enhancement. No content generation, addition, removal,
or replacement. Preserves all original objects, text, layout, and geometry.

Methods implemented:
  1. classical     - OpenCV-only: denoise + CLAHE + sharpen + Lanczos upscale
  2. realesrgan    - Real-ESRGAN x4plus (CNN super-resolution)
  3. hybrid        - classical preprocess -> realesrgan -> final touch
  4. tiled         - tiled Real-ESRGAN with feathered seam blending
  5. billboard     - verified billboard regions SR-composited onto the ORIGINAL scene
  6. repair        - stronger restoration-first stack (aggressive deblur/denoise)
  7. repair_v2     - conservative 'enhance, don't generate' stack (natural faces
                     first; no waxy/painterly artifacts, no face restoration)
8. final         - approved photographic image engine (research-complete):
                     F3-natural full scene + G1-text verified-box billboard
                     text + conservative native recovery for STRONGLY
                     DEFOCUSED boxes only. Non-generative throughout.
"""

import os
import time
from pathlib import Path

import numpy as np
import cv2
import torch

import tonal_correction

OUT_W, OUT_H = 3840, 2160

_REALESRGAN_CACHE = {}


def get_realesrgan(model_name="RealESRGAN_x4plus"):
    """Lazily load (and cache) a Real-ESRGAN model on the best device."""
    if model_name in _REALESRGAN_CACHE:
        return _REALESRGAN_CACHE[model_name]

    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet

    model_root = Path(os.environ.get(
        "REALESRGAN_MODELS", str(Path(__file__).parent.parent / "models")
    ))

    if model_name == "RealESRGAN_x4plus":
        arch = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                       num_block=23, num_grow_ch=32, scale=4)
        model_path = model_root / "RealESRGAN_x4plus.pth"
        if not model_path.exists():
            model_path = str(model_path).replace(".pth", ".pth")
    else:
        raise ValueError(f"Unknown Real-ESRGAN model: {model_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    upsampler = RealESRGANer(
        scale=4,
        model_path=str(model_path),
        model=arch,
        tile=256,
        tile_pad=10,
        pre_pad=0,
        half=False if device.type == "cpu" else True,
        device=device,
    )
    _REALESRGAN_CACHE[model_name] = upsampler
    return upsampler


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


# ---------------------------------------------------------------- classical

def classical_preprocess(img):
    """Denoise + CLAAHE + sharpen AT ORIGINAL RESOLUTION (no upscale).
    Used by both the classical method (which then upscales) and hybrid."""
    denoised = cv2.fastNlMeansDenoisingColored(
        img, None, h=7, hColor=7, templateWindowSize=7, searchWindowSize=21
    )

    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    out = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    out = cv2.convertScaleAbs(out, alpha=1.02, beta=0)

    blur = cv2.GaussianBlur(out, (0, 0), 1.0)
    out = cv2.addWeighted(out, 1.4, blur, -0.4, 0)
    return out


def classical_enhance(img, target=(OUT_W, OUT_H)):
    """OpenCV-only pipeline:
       fastNlMeans denoise -> CLAHE color -> sharpen -> Lanczos upscale."""
    out = classical_preprocess(img)
    out = cv2.resize(out, target, interpolation=cv2.INTER_LANCZOS4)
    return out


# ------------------------------------------------------------------ realesrgan

def realesrgan_enhance(img, target=(OUT_W, OUT_H)):
    """True 4x model-driven super-resolution via Real-ESRGAN, then resize to
    exactly 3840x2160. Uses outscale=4 so the RRDB network performs the
    learned 4x upscale (not a plain Lanczos resize)."""
    upsampler = get_realesrgan("RealESRGAN_x4plus")
    with torch.no_grad():
        output, _ = upsampler.enhance(img, outscale=4)
    if output.shape[:2][::-1] != target:
        output = cv2.resize(output, target, interpolation=cv2.INTER_LANCZOS4)
    return output


# --------------------------------------------------------------------- hybrid

def hybrid_enhance(img, target=(OUT_W, OUT_H)):
    """Classical denoise/contrast first, then Real-ESRGAN, then subtle
    final sharpen + tone. Best of both worlds when content is noisy."""
    pre = classical_preprocess(img)
    up = realesrgan_enhance(pre, target=target)
    blur = cv2.GaussianBlur(up, (0, 0), 0.6)
    out = cv2.addWeighted(up, 1.15, blur, -0.15, 0)
    return out


# ------------------------------------------------- B: explicit tiled super-resolution

def _tile_ranges(size, tile, overlap):
    """Yield (start, end, has_prev_boundary) windows covering `size`.
    has_prev_boundary: this tile starts before the previous tile ends
    (i.e. it has a left/top overlap neighbor)."""
    windows = []
    start = 0
    first = True
    while start < size:
        end = min(start + tile, size)
        windows.append((start, end, not first))
        if end >= size:
            break
        start = end - overlap
        first = False
    return windows


def _tile_weight(h, w, boundary_l, boundary_r, edge):
    """1 everywhere except a linear ramp down to 0 over `edge` px on sides
    that have an overlapping neighbor (boundary_l: left/top, boundary_r:
    right/bottom). Outer image edges are kept at weight 1 so the image border
    is never divided by ~0."""
    wt = np.ones((h, w), np.float32)
    edge = max(1, int(edge))
    if boundary_l:
        for i in range(min(edge, h)):
            t = (i + 1) / (edge + 1)
            wt[i, :] = t
    if boundary_r:
        for i in range(min(edge, h)):
            t = (i + 1) / (edge + 1)
            wt[h - 1 - i, :] = t
    return wt


def tiled_enhance(img, target=(OUT_W, OUT_H), tile=1024, overlap=120):
    """Explicit tiled Real-ESRGAN with feathered seam blending.

    Each tile is super-resolved independently (so each fits in VRAM), then
    placed on a common 4x output grid. In overlap regions, tiles are blended
    with a linear ramp so no visible seams appear. The imagery outer border is
    always covered at full weight from the edge-most tile (no border artifacts).
    Content is unchanged — this only controls memory/stitching."""
    upsampler = get_realesrgan("RealESRGAN_x4plus")
    h, w = img.shape[:2]
    rows = _tile_ranges(h, tile, overlap)
    cols = _tile_ranges(w, tile, overlap)

    out_h, out_w = h * 4, w * 4
    acc = np.zeros((out_h, out_w, 3), np.float64)
    wsum = np.zeros((out_h, out_w, 1), np.float64)

    edge_src = overlap
    for (r0, r1, b_top) in rows:
        for (c0, c1, b_left) in cols:
            tile_img = img[r0:r1, c0:c1]
            with torch.no_grad():
                sr, _ = upsampler.enhance(tile_img, outscale=4)
            sr_h, sr_w = sr.shape[:2]
            O0, O1 = r0 * 4, r0 * 4 + sr_h
            P0, P1 = c0 * 4, c0 * 4 + sr_w
            O1 = min(O1, out_h); P1 = min(P1, out_w)
            # vertical/horizontal overlap feather flags
            b_bottom = (r1 < h)   # has neighbor below
            b_right = (c1 < w)    # has neighbor right
            wt = _tile_weight(O1 - O0, P1 - P0, b_top or b_left, b_bottom or b_right,
                              edge=edge_src * 4)
            piece = sr[0:O1 - O0, 0:P1 - P0].astype(np.float64)
            acc[O0:O1, P0:P1] += piece * wt[:, :, None]
            wsum[O0:O1, P0:P1] += wt[:, :, None]

    out = (acc / np.maximum(wsum, 1e-6)).astype(np.uint8)
    if out.shape[:2][::-1] != target:
        out = cv2.resize(out, target, interpolation=cv2.INTER_LANCZOS4)
    return out


# ------------------------------------------------- C: billboard-crop + composite

_REGION_CONFIG = None
_WARNED_NO_BOX = False


def _load_region_config():
    global _REGION_CONFIG
    if _REGION_CONFIG is not None:
        return _REGION_CONFIG
    cfg = Path(os.environ.get(
        "REGIONS_CONFIG", str(Path(__file__).parent.parent / "regions.json")))
    import json
    if cfg.exists():
        text = cfg.read_text(encoding="utf-8-sig")
        _REGION_CONFIG = json.loads(text) if text.strip() else {}
    else:
        _REGION_CONFIG = {}
    return _REGION_CONFIG


def _entry_boxes(entry):
    """Normalize a regions.json entry to a list of (x, y, w, h) boxes.

    A single photo may contain SEVERAL billboards, so the schema uses a
    'boxes' list. The legacy single 'box' key is still accepted."""
    if not entry:
        return []
    raw = entry.get("boxes")
    if not raw:
        raw = [entry.get("box")] if entry.get("box") else []
    boxes = []
    for b in raw:
        if not b or any(v is None for v in b):
            continue
        try:
            boxes.append(tuple(int(v) for v in b))
        except (TypeError, ValueError):
            continue
    return boxes


def _billboard_boxes(img):
    """Return a list of (x, y, w, h) billboard regions for `img` from the
    MANUALLY VERIFIED regions config ONLY.

    One photo may contain several target billboards; every box is enhanced and
    composited back. The automatic detector is deliberately NOT used as a
    fallback: generic rectangle detection is unreliable for OOH billboards (it
    selects roads, sky, vehicles and unrelated rectangles), and not every
    detected rectangle is a relevant Adinn board. Returns [] when no verified
    boxes are configured."""
    H, W = img.shape[:2]
    cfg = _load_region_config()
    # caller may pass the original filename for precise matching
    name = os.environ.get("BILLBOARD_IMAGE", "")
    boxes = _entry_boxes(cfg.get(name, {})) if name else []
    if not boxes:
        # fallback: unique entry matching this image size (single-image-set only)
        match = [e for e in cfg.values() if e.get("image_size") == [W, H]]
        if len(match) == 1:
            boxes = _entry_boxes(match[0])

    clamped = []
    for box in boxes:
        x, y, w, h = box
        x = max(0, min(x, W - 1)); y = max(0, min(y, H - 1))
        w = min(w, W - x); h = min(h, H - y)
        if w >= 2 and h >= 2:
            clamped.append((x, y, w, h))
    return clamped


def _composite_alpha(h, w, edge):
    """Alpha mask for blending an enhanced billboard crop back into the
    background: 1 in the interior, ramping to 0 over `edge` px at the border
    so the placement is seamless."""
    a = np.ones((h, w), np.float32)
    edge = max(1, int(edge))
    for i in range(min(edge, h)):
        t = (i + 1) / (edge + 1)
        a[i, :] = t
        a[h - 1 - i, :] = t
    for i in range(min(edge, w)):
        t = (i + 1) / (edge + 1)
        a[:, i] = np.minimum(a[:, i], t)
        a[:, w - 1 - i] = np.minimum(a[:, w - 1 - i], t)
    return a[:, :, None]


def billboard_composite_enhance(img, target=(OUT_W, OUT_H), pad_ratio=0.15,
                                bg_method="realesrgan"):
    """Enhance ONLY the manually verified billboard regions and composite them
    back onto the faithfully-upscaled ORIGINAL scene.

    Preserve-only behaviour: every part of the photo except the target
    billboard(s) stays exactly as the original (content must not be generated,
    added, removed, or replaced). Each verified billboard crop is super-resolved
    and blended back with a feathered alpha mask."""
    # background = faithful upscale of the ORIGINAL scene (no whole-scene change)
    bg = simple_upscale(img, target)
    boxes = _billboard_boxes(img)
    if not boxes:
        global _WARNED_NO_BOX
        if not _WARNED_NO_BOX:
            _WARNED_NO_BOX = True
            print("!! billboard (C): no manually verified regions configured "
                  "for the current image; returning the faithful original "
                  "upscale unchanged. Verify regions first: "
                  "python src\\verify_regions.py")
        return bg

    H, W = img.shape[:2]
    s = OUT_W / W
    out = bg.copy()
    for (x, y, w, h) in boxes:
        # pad the crop a little so the frame border is included
        px = int(w * pad_ratio); py = int(h * pad_ratio)
        cx0 = max(0, x - px); cy0 = max(0, y - py)
        cx1 = min(W, x + w + px); cy1 = min(H, y + h + py)
        if cx1 - cx0 < 2 or cy1 - cy0 < 2:
            continue
        crop = img[cy0:cy1, cx0:cx1]

        # enhanced crop (super-resolved, then sized to match target coords)
        up_crop, _ = enhance("realesrgan", crop)  # SR whole crop to 4x

        # target mapping: scale = OUT_W / W
        t_cx0 = int(round(cx0 * s)); t_cy0 = int(round(cy0 * s))
        t_cw = int(round((cx1 - cx0) * s)); t_ch = int(round((cy1 - cy0) * s))
        if t_cw < 2 or t_ch < 2:
            continue
        target_crop = cv2.resize(up_crop, (t_cw, t_ch),
                                 interpolation=cv2.INTER_LANCZOS4)

        alpha = _composite_alpha(t_ch, t_cw,
                                 edge=max(2, int(0.1 * min(t_ch, t_cw))))
        region = out[t_cy0:t_cy0 + t_ch, t_cx0:t_cx0 + t_cw].astype(np.float32)
        blended = target_crop.astype(np.float32) * alpha + region * (1 - alpha)
        out[t_cy0:t_cy0 + t_ch, t_cx0:t_cx0 + t_cw] = \
            np.clip(blended, 0, 255).astype(np.uint8)
    return out


# ------------------------------------------------- D: stronger restoration + SR

def deblur_denoise(img, strength=1.0):
    """Edge-preserving deblur + denoise (bilateral) plus an unsharp boost.
    Reduces motion/compression blur and noise before super-resolution so the
    SR network sees cleaner high-frequency content."""
    b = cv2.bilateralFilter(img, d=9, sigmaColor=75 * strength, sigmaSpace=9 * strength)
    # residual sharpening (unsharp) to recover edges after smoothing
    blur = cv2.GaussianBlur(b, (0, 0), 1.2)
    sharp = cv2.addWeighted(b, 1 + 0.5 * strength, blur, -0.5 * strength, 0)
    return sharp


def repair_enhance(img, target=(OUT_W, OUT_H)):
    """Restoration-first stack: deblur+denoise -> Real-ESRGAN -> final unsharp.
    Aims for stronger recovery of fine text/logos without hallucinating."""
    cleaned = deblur_denoise(img, strength=1.0)
    up = realesrgan_enhance(cleaned, target=target)
    blur = cv2.GaussianBlur(up, (0, 0), 0.7)
    out = cv2.addWeighted(up, 1.2, blur, -0.2, 0)
    return out


# ---------------------------------------------- V2: conservative Repair (ENHANCE, DON'T GENERATE)

def _conservative_preprocess(img):
    """Texture-preserving denoise.

    A MILD small-window bilateral filter — range/size far below the aggressive
    settings used by Repair D (sigmaColor=75/sigmaSpace=9/d=9) — removes grain
    while keeping low-contrast facial detail (pores, skin texture). Repair D's
    strong bilateral smoothing is the main cause of the waxy/painterly face
    look, so V2 must never use it. No unsharp before SR either (unsharp also
    amplifies that fake-plastic texture prior to super-resolution)."""
    return cv2.bilateralFilter(img, d=5, sigmaColor=30, sigmaSpace=5)


def _conservative_finish(up):
    """Conservative final step: the Real-ESRGAN result is returned UNCHANGED.

    Deliberately NOT unsharpened. A fine-scale post-SR sharpen would re-boost
    high-frequency structure on faces into a synthetic/AI appearance — exactly
    what V2 exists to avoid. SR already delivers the billboard/text clarity."""
    return up


def repair_v2_enhance(img, target=(OUT_W, OUT_H)):
    """REPAIR V2 — 'enhance, don't generate'.

    Original
      -> conservative preprocessing (texture-preserving mild denoise)
      -> Real-ESRGAN 4x
      -> conservative final enhancement (none beyond the SR output)
      -> 3840x2160

    Design priorities (in order):
      1. natural-looking faces and people
      2. source fidelity
      3. billboard / text / artwork clarity
      4. overall sharpness
      5. noise reduction

    Policy decisions:
      - NO face detector is added: for real street photos reliable face
        detection is not guaranteed, so per project rules we use conservative
        GLOBAL processing instead of a detection mask.
      - NO face restoration / reconstruction (no GFPGAN, no CodeFormer, no
        generative face model).
      - The original photograph is the source of truth; the pipeline never
        invents facial, text, artwork, vehicle, building or scene detail.
      - Evaluated separately from the Billboard Composite method."""
    cleaned = _conservative_preprocess(img)
    up = realesrgan_enhance(cleaned, target=target)
    return _conservative_finish(up)


# --------------------------------------- E: final photographic image engine
#
# Approved research direction:
#   F3-natural   = full-scene photographic base (original photo + band-passed
#                  D1-weak SR structure on strong source edges, original chroma)
#   Candidate A  = billboard-specific pass ONLY inside verified regions.json
#                  boxes, run on the NATIVE-resolution crop (not the shared 4K
#                  frame): Restormer motion-deblur -> SwinIR-M x4 SR -> a mild
#                  adaptive-sigma Richardson-Lucy + edge-gated USM touch-up.
#                  Selected over the prior fixed-sigma-RL-only G1 pass and a
#                  conservative-native-recovery add-on after a controlled R&D
#                  comparison (isolated harness, non-generative candidates
#                  only): a dedicated, already-vetted deblurring model plus a
#                  crop-first SR pass (full model capacity on the board, not
#                  shared with surrounding scene content) materially improved
#                  real billboard text/logo edge clarity with no hallucinated
#                  content, confirmed by direct pixel comparison against the
#                  prior production output. Real-ESRGAN was evaluated and
#                  REJECTED in that R&D pass: it fabricated plausible-looking
#                  but incorrect letterforms on real board text.
#   A+ (whole-    = a G6 R&D finding: swapping/augmenting D1's backbone
#   image detail)   (denoise+deblur blend, or a larger SwinIR-L SR model)
#                  barely changed the final output at all, because F3's own
#                  +-14 envelope clips away almost all of the difference
#                  before it can matter. The one whole-image change that DID
#                  make a real, measurable, and safe difference runs AFTER
#                  F3 instead of feeding into it: an edge-gated local-contrast
#                  pass reused verbatim from this repo's own g5_exp family,
#                  hard-clipped to each pixel's OWN local 5x5 [min,max] from
#                  F3's own output -- so it can only sharpen contrast that is
#                  already there and can never create a new brightness level.
#                  Confirmed across all six reference images plus the real
#                  JJ GOLD/GRT photo: +30-45% sharpness with flat/sky regions
#                  changing by at most the 8-bit Lab round-trip floor, no
#                  halos/ringing, negligible runtime cost. Billboard regions
#                  are unaffected: Candidate A's board reconstruction fully
#                  replaces this pass's output inside its feathered box.
#
# Proven parameters, refactored here (no experimental folders in production):
#   D1-weak: K=0.30 Restormer-denoise blend + SwinIR-M x4, tiled for 8GB VRAM
#   F3-natural: w=6 / clip +-18 / gate t0=0.30 / envelope +-14
#   A+ whole-image detail: strength 0.6, clip +-6, gate t0=0.35
#   Candidate A touch-up: adaptive RL sigma (measured per-box, clamped
#     0.8-3.5), 12 iters, gate t0=0.30, clip +-16, USM 0.2
#
# Guarantees: 3840x2160 BGR uint8 output; no OCR/redraw, no GAN/diffusion, no
# face work, no invented strokes; every pixel outside the verified boxes is
# untouched by the billboard stage; inside a box, only the feathered board
# region is replaced by its own dedicated, non-generative reconstruction of
# that same board (never a different or fabricated one).

_FINAL_D1_K = 0.30
_FINAL_DN_TILE = 512
_FINAL_DN_OVERLAP = 48
_FINAL_SR_TILE = 256
_FINAL_F3_W = 6.0
_FINAL_F3_CLIP = 18.0
_FINAL_F3_T0 = 0.25
_FINAL_F3_ENV = 14.0
_FINAL_F3_BP_SIGMA = 2.5
_FINAL_APLUS_STRENGTH = 0.6
_FINAL_APLUS_CLIP = 6.0
_FINAL_APLUS_T0 = 0.35
_FINAL_TOUCHUP_SIGMA_MIN = 0.8
_FINAL_TOUCHUP_SIGMA_MAX = 3.5
_FINAL_TOUCHUP_SIGMA_DEFAULT = 1.2
_FINAL_TOUCHUP_ITERS = 12
_FINAL_TOUCHUP_T0 = 0.30
_FINAL_TOUCHUP_CLIP = 16.0
_FINAL_TOUCHUP_USM = 0.2
_FINAL_TOUCHUP_USM_SIGMA = 0.8
_FINAL_TOUCHUP_USM_THRESH = 10.0
_FINAL_TOUCHUP_BP_SIGMA = 2.0
_FINAL_BOARD_TILE = 512
_FINAL_FEATHER = 16
_FINAL_DEFOCUS_SIGMA = 2.8


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


def _final_d1_weak(orig):
    """D1-weak (proven): K=0.30 Restormer-denoise blend + SwinIR-M x4 -> 4K.

    Tiled so peak VRAM stays near ~2.4 GB (RTX 3050 8GB-safe). Returns
    (d1_4k, peak_vram_mb, device_str)."""
    from pro_exp import pipeline2
    from restore_exp import restormer, swinir_m
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    dn_model = restormer.load("real_denoise", device)
    dn_img = pipeline2.tiled_restore(dn_model, orig,
                                     tile=_FINAL_DN_TILE,
                                     overlap=_FINAL_DN_OVERLAP)
    k = _FINAL_D1_K
    blend = (orig.astype(np.float32) * (1.0 - k)
             + dn_img.astype(np.float32) * k)
    blend = np.clip(blend, 0, 255).astype(np.uint8)
    sr = swinir_m.sr_bgr(blend, device, tile=_FINAL_SR_TILE)
    out = cv2.resize(sr, (OUT_W, OUT_H), interpolation=cv2.INTER_LANCZOS4)
    peak = 0.0
    if torch.cuda.is_available():
        peak = float(torch.cuda.max_memory_allocated() / 1048576.0)
        # No torch.cuda.empty_cache() here: this app processes one job at a
        # time on a single GPU with nothing else contending for VRAM (see
        # docs/ARCHITECTURE.md's processing model), so releasing the
        # allocator's cached blocks back to the driver after every single
        # image bought nothing but ~0.1-0.2s of pure overhead per image,
        # immediately re-requested for the very next tile/image anyway.
        # peak-memory reporting above is unaffected either way.
    return out, peak, str(device)


def _final_f3_natural(faithful4k, d1_4k):
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


def _final_multiscale_detail(bgr):
    """G7-MS (validated in phase1_enhancement/src/g7_exp/recipe_g7.py as
    `multiscale_detail_boost`): 3-level DoG/Laplacian-style multi-scale
    detail recovery on TOP of F3, BEFORE the A+ whole-frame
    detail pass. F3's L-plane is split into 3 frequency bands + a low-pass
    base; each band is boosted by a modest gain (1.30/1.20/1.15 at
    sigma=1/2/4) and hard-clipped to a LOCAL min/max envelope of the SOURCE
    L-plane at a structuring-element size matched to that band's scale
    (3/5/9 px), so a band's boosted contribution can never push a pixel past
    a brightness level that does not already exist somewhere in its own
    neighborhood. The combined delta is additionally edge-gated and globally
    clipped (+-10) before being applied. Parameters ported verbatim from the
    g7_exp sweep -- do not retune here."""
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


def _final_whole_frame_detail(bgr):
    """A+ (proven in g6_exp): edge-masked local contrast on TOP of F3, hard-
    clipped to each pixel's own local 5x5 [min, max] from this SAME image, so
    it can only sharpen contrast that is already there -- it cannot create a
    new brightness level, and therefore cannot invent a stroke or texture.
    Exact implementation and parameters (strength 0.6, clip +-6, gate t0=0.35)
    reused unchanged from g6_exp/recipe_g6.py's winning candidate."""
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


def _final_valid_box(box, W, H):
    """Clamp a verified (x, y, w, h) box to the image; None if degenerate."""
    try:
        x, y, w, h = (int(v) for v in box)
    except (TypeError, ValueError):
        return None
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = min(w, W - x)
    h = min(h, H - y)
    if w < 8 or h < 8:
        return None
    return (x, y, w, h)


def _final_scale_box(box, w_orig):
    s = OUT_W / w_orig
    x, y, w, h = box
    return (int(round(x * s)), int(round(y * s)),
            int(round(w * s)), int(round(h * s)))


def _final_psf_sigma(native_gray):
    """Per-box Gaussian PSF sigma estimate (median 10-90% edge rise / 2.35)
    at NATIVE resolution. Returns -1.0 when unmeasurable."""
    g = cv2.GaussianBlur(native_gray, (0, 0), 0.4)
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0)
    mag = np.abs(gx)
    thr = np.percentile(mag, 99.0)
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


def _final_richardson_lucy(img01, psf, iters):
    from skimage.restoration import richardson_lucy
    return richardson_lucy(img01, psf, num_iter=iters, clip=False)


def _final_board_touchup(bgr):
    """Candidate A's final touch-up (proven in R&D): adaptive-sigma
    Richardson-Lucy + edge-gated unsharp mask on the L channel only (chroma
    from the deblur/SR stages is preserved). Unlike the prior fixed-sigma
    G1 pass, the RL kernel sigma is MEASURED from this crop (clamped to a
    sane range) so deconvolution is matched to the crop's own actual blur
    instead of assuming a fixed amount for every board."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    sigma = _final_psf_sigma(gray)
    if sigma <= 0:
        sigma = _FINAL_TOUCHUP_SIGMA_DEFAULT
    sigma = float(np.clip(sigma, _FINAL_TOUCHUP_SIGMA_MIN, _FINAL_TOUCHUP_SIGMA_MAX))

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l = lab[:, :, 0]
    den = cv2.bilateralFilter(l, d=5, sigmaColor=15, sigmaSpace=5)
    ksz = max(3, int(round(sigma * 6)) | 1)
    k1 = cv2.getGaussianKernel(ksz, sigma)
    psf = k1 @ k1.T
    psf = psf / psf.sum()
    rl = _final_richardson_lucy(
        np.clip(den, 0, 255).astype(np.float64) / 255.0, psf,
        _FINAL_TOUCHUP_ITERS)
    rl = np.clip(rl, 0, 1).astype(np.float32) * 255.0
    sc = rl - l
    sc_bp = sc - cv2.GaussianBlur(sc, (0, 0), _FINAL_TOUCHUP_BP_SIGMA)
    gate = _final_edge_gate(l, _FINAL_TOUCHUP_T0)
    db = np.clip(sc_bp, -_FINAL_TOUCHUP_CLIP, _FINAL_TOUCHUP_CLIP) * gate
    blur = cv2.GaussianBlur(l, (0, 0), _FINAL_TOUCHUP_USM_SIGMA)
    udelta = l - blur
    udelta = np.where(np.abs(udelta) >= _FINAL_TOUCHUP_USM_THRESH,
                      _FINAL_TOUCHUP_USM * udelta, 0.0)
    db = db + udelta * gate
    lab[:, :, 0] = np.clip(l + db, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def _final_candidate_a_board(native_bgr, device, target_wh=None):
    """Candidate A (won the R&D comparison, see module docstring above):
    native billboard crop -> Restormer motion-deblur -> SwinIR-M x4 SR ->
    single resize to the box's true frame scale -> mild adaptive touch-up.
    Runs entirely on this one board's own native crop (never the shared
    whole-frame tiles), so one board's processing cannot affect another's.
    Uses only already-vetted, already-downloaded, non-generative models
    already present in this repository.

    `target_wh`, when given, is the exact (width, height) this board must
    occupy in the final frame (i.e. its `box4` footprint, at the frame's
    own upscale factor `OUT_W / W0` -- not a fixed 4x). SwinIR-M is
    architecturally fixed at 4x, so its native output is resized ONCE
    directly to `target_wh` before the touch-up runs, instead of being
    left at 4x and downsampled again later by the compositor. Running the
    deconvolution/unsharp touch-up at the true target resolution (rather
    than at 4x, only to have that detail thrown away by a later resize)
    is the point of the fix -- it avoids the redundant upscale-then-
    downscale round trip that discarded recovered detail. When
    `target_wh` is omitted, the crop is left at the native 4x SR size
    (prior behavior), for callers that still expect that."""
    from pro_exp import pipeline2
    from restore_exp import restormer, swinir_m
    h, w = native_bgr.shape[:2]
    deblur_model = restormer.load("motion_deblur", device)
    deblurred = pipeline2.tiled_restore(deblur_model, native_bgr,
                                        tile=min(_FINAL_BOARD_TILE, h, w),
                                        overlap=32)
    sr = swinir_m.sr_bgr(deblurred, device,
                         tile=min(_FINAL_BOARD_TILE, h, w), overlap=16)
    if sr.shape[1] != w * 4 or sr.shape[0] != h * 4:
        sr = cv2.resize(sr, (w * 4, h * 4), interpolation=cv2.INTER_LANCZOS4)
    if target_wh is not None and (sr.shape[1], sr.shape[0]) != target_wh:
        sr = cv2.resize(sr, target_wh, interpolation=cv2.INTER_LANCZOS4)
    return _final_board_touchup(sr)


def _final_composite_box_bgr(frame_bgr, board_bgr, box4):
    """Feather one enhanced board crop (full BGR, own chroma) into the 4K
    frame at box4. Only this feathered region changes; everything outside
    the box, including the rest of the frame produced by F3, is untouched."""
    bx, by, bw, bh = box4
    xs = min(bx + bw, frame_bgr.shape[1])
    ys = min(by + bh, frame_bgr.shape[0])
    hh, ww = ys - by, xs - bx
    if hh < 8 or ww < 8:
        return frame_bgr
    if board_bgr.shape[:2] != (hh, ww):
        board_bgr = cv2.resize(board_bgr, (ww, hh), interpolation=cv2.INTER_LANCZOS4)
    alpha = _composite_alpha(hh, ww, max(2, int(_FINAL_FEATHER)))
    region = frame_bgr[by:ys, bx:xs].astype(np.float32)
    blended = region * (1.0 - alpha) + board_bgr.astype(np.float32) * alpha
    out = frame_bgr.copy()
    out[by:ys, bx:xs] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


def final_enhance(img, target=(OUT_W, OUT_H), return_stages=False):
    """Approved final photographic image engine.

    Original -> faithful 4K -> D1-weak SR -> F3-natural (full scene)
      -> G7-MS multi-scale detail boost (self-clipped per-band Laplacian/
         DoG recovery, see _final_multiscale_detail; promoted from
         phase1_enhancement/src/g7_exp after validation)
      -> A+ whole-frame detail (edge-gated local contrast, clipped to each
         pixel's own local 5x5 [min,max] -- the whole frame, not just boards)
      -> Candidate A (verified billboard boxes only: native crop ->
         Restormer motion-deblur -> SwinIR-M x4 SR -> single resize to the
         box's true frame scale (OUT_W / W0, matching box4) -> adaptive
         touch-up -> feathered back in, each box independent of the
         others)
      -> 3840x2160 BGR uint8.

    Adaptive tonal correction (tonal_correction.py) runs first, on the
    whole-frame path only: a bounded, deterministic, luminance-only global
    tone remap that activates ONLY when the source's own histogram shows a
    genuine exposure/contrast problem (underexposure, overexposure, low
    contrast, crushed shadows, or clipped highlights), and is a total no-op
    otherwise. Verified billboard-box crops are deliberately reconstructed
    from the UNCORRECTED original below (native_crop uses `img`, not the
    tonally-corrected copy) -- Candidate A's job is the most faithful
    possible reproduction of that specific board's real content, so it must
    not be affected by a whole-frame exposure fix.

    No OCR/redraw, no GAN/diffusion, no face work, no invented content.
    With return_stages=True also returns a dict with intermediates
    (tonal_correction/faithful/d1_weak/f3/f3_ms/f3_plus/boxes4/psf_sigmas/
    defocus_boxes/peak_vram_mb).
    """
    if img is None or not isinstance(img, np.ndarray) or img.ndim != 3 \
            or img.shape[2] != 3:
        raise ValueError("final_enhance expects a BGR uint8 image")
    H0, W0 = img.shape[:2]
    if H0 < 16 or W0 < 16:
        raise ValueError("final_enhance expects an image of at least 16x16")

    stages = {}
    img_toned, tonal_meta = tonal_correction.adaptive_tonal_correction(img, return_meta=True)
    stages["tonal_correction"] = tonal_meta

    faithful = simple_upscale(img_toned, target)
    stages["faithful"] = faithful

    d1, peak_vram_mb, device = _final_d1_weak(img_toned)
    stages["d1_weak"] = d1
    stages["peak_vram_mb"] = peak_vram_mb
    stages["device"] = device

    f3 = _final_f3_natural(faithful, d1)
    stages["f3"] = f3

    f3_ms = _final_multiscale_detail(f3)
    stages["f3_ms"] = f3_ms

    f3_plus = _final_whole_frame_detail(f3_ms)
    stages["f3_plus"] = f3_plus

    boxes4, sigmas, defocus_boxes = [], [], []
    boxes = _billboard_boxes(img)
    if boxes:
        device_t = torch.device(device)
        out = f3_plus.copy()
        for (x, y, w, h) in boxes:
            vb = _final_valid_box((x, y, w, h), W0, H0)
            if vb is None:
                continue
            vx, vy, vw, vh = vb
            bx, by, bw, bh = _final_scale_box(vb, W0)
            xs = min(bx + bw, out.shape[1])
            ys = min(by + bh, out.shape[0])
            if xs - bx < 8 or ys - by < 8:
                continue
            box4 = (bx, by, xs - bx, ys - by)
            boxes4.append(box4)
            # Each box is processed from its OWN native crop of the original
            # image, independently of every other box and of F3's whole-frame
            # result — one board's processing cannot affect another's, and a
            # board's own dedicated reconstruction fully replaces (rather than
            # merely nudges) F3's content inside its feathered region.
            nc = img[vy:vy + vh, vx:vx + vw]
            sig = _final_psf_sigma(cv2.cvtColor(nc, cv2.COLOR_BGR2GRAY))
            sigmas.append(sig)
            if sig >= _FINAL_DEFOCUS_SIGMA:
                defocus_boxes.append(box4)
            board_bgr = _final_candidate_a_board(
                nc, device_t, target_wh=(box4[2], box4[3]))
            out = _final_composite_box_bgr(out, board_bgr, box4)
    else:
        out = f3_plus

    if out.shape[1] != target[0] or out.shape[0] != target[1]:
        out = cv2.resize(out, target, interpolation=cv2.INTER_LANCZOS4)
    stages["boxes4"] = boxes4
    stages["psf_sigmas"] = sigmas
    stages["defocus_boxes"] = defocus_boxes
    if return_stages:
        return out, stages
    return out


# ------------------------------------------------------------------- dispatch

METHODS = {
    "classical": classical_enhance,
    "realesrgan": realesrgan_enhance,
    "hybrid": hybrid_enhance,
    "tiled": tiled_enhance,
    "billboard": billboard_composite_enhance,
    "repair": repair_enhance,
    "repair_v2": repair_v2_enhance,
    "final": final_enhance,
}


def enhance(method, img, target=(OUT_W, OUT_H)):
    if method not in METHODS:
        raise KeyError(f"Unknown method: {method}. Available: {list(METHODS)}")
    fn = METHODS[method]
    t0 = time.perf_counter()
    out = fn(img, target)
    dt = time.perf_counter() - t0
    return out, dt
