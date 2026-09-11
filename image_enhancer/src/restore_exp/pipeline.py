"""Content-aware, fidelity-first restoration pipeline (Parts 2 and 3).

Isolated experiment. Builds the multi-scale variants used by the A/B/C
benchmark on top of the faithful full-frame upscale and the whole-scene
SwinIR-M PSNR output (Path A):

  A            whole-frame SwinIR-M PSNR x4 (full scene)
  B            A + conservative billboard-region restore (verified boxes)
  B_deblur     A + billboard restore with an extra motion-deblur stage
  C            B + conservative face-region restore (YuNet detection only)

Only verified `regions.json` boxes are touched for billboards, and only
detected, confidently-large faces are touched for faces. Everything else stays
exactly as Path A. No text/logo is ever recreated from OCR or generation.
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import enhance  # noqa: E402  (region config + simple_upscale reuse)
from restore_exp import faces as faces_mod  # noqa: E402
from restore_exp import restormer, swinir_m  # noqa: E402

OUT_W, OUT_H = 3840, 2160
PAD_RATIO = 0.15
FACE_PAD = 0.30
FACE_MIN_SIZE = 24
FACE_SCORE = 0.70
CLAHE_CLIP = 1.2


def _region_config():
    return enhance._load_region_config()


def boxes_for(stem):
    cfg = _region_config()
    entry = cfg.get(f"{stem}.jpeg") or cfg.get(stem) or {}
    return enhance._entry_boxes(entry)


def scale_rect(box, w_orig, out_w=OUT_W):
    s = out_w / w_orig
    x, y, w, h = box
    r = tuple(int(round(v * s)) for v in (x, y, w, h))
    return r


def padded(box, pad_ratio, W, H):
    x, y, w, h = box
    px = int(round(w * pad_ratio))
    py = int(round(h * pad_ratio))
    cx0 = max(0, x - px)
    cy0 = max(0, y - py)
    cx1 = min(W, x + w + px)
    cy1 = min(H, y + h + py)
    return cx0, cy0, cx1, cy1


def composite_alpha(h, w, edge):
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


def _mild_local_enhance(bgr, amount=0.3):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(8, 8)).apply(l)
    lf = l.astype(np.float32)
    l_adj = (lf * (1 - amount) + cl.astype(np.float32) * amount).astype(np.uint8)
    lab2 = cv2.merge((l_adj, a, b))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def restore_sr_crop(orig_bgr, denoise, deblur, device, local_enhance=0.3, sr_tile=400):
    """Conservative region restore: (optional deblur) -> denoise -> SwinIR-M
    PSNR x4 -> mild local luminance contrast. Never invents detail."""
    img = orig_bgr
    if deblur:
        img = restormer.enhance_bgr(restormer.load("motion_deblur", device), img)
    if denoise:
        img = restormer.enhance_bgr(restormer.load("real_denoise", device), img)
    img = swinir_m.sr_bgr(img, device, tile=sr_tile)
    if local_enhance > 0:
        img = _mild_local_enhance(img, local_enhance)
    return img


def place_region(out, restored, src_rect, tar_rect, edge_ratio=0.08):
    sx0, sy0, sx1, sy1 = src_rect
    tx, ty, tw, th = tar_rect
    tx, ty = max(0, tx), max(0, ty)
    th = min(th, out.shape[0] - ty)
    tw = min(tw, out.shape[1] - tx)
    if tw < 2 or th < 2:
        return
    target_crop = restored
    if target_crop.shape[:2] != (th, tw):
        target_crop = cv2.resize(target_crop, (tw, th), cv2.INTER_LANCZOS4)
    alpha = composite_alpha(th, tw, edge=max(2, int(edge_ratio * min(th, tw))))
    region = out[ty:ty + th, tx:tx + tw].astype(np.float32)
    blended = target_crop.astype(np.float32) * alpha + region * (1 - alpha)
    out[ty:ty + th, tx:tx + tw] = np.clip(blended, 0, 255).astype(np.uint8)


def _crop_orig(orig, pad_rect):
    x0, y0, x1, y1 = pad_rect
    return orig[y0:y1, x0:x1]


def apply_billboards(out, orig, stem, boxes, device, deblur=False, stats=None, tag=""):
    H, W = orig.shape[:2]
    t0 = time.perf_counter()
    n = 0
    for box in boxes:
        pr = padded(box, PAD_RATIO, W, H)
        crop = _crop_orig(orig, pr)
        restored = restore_sr_crop(crop, denoise=True, deblur=deblur,
                                   device=device)
        tar = scale_rect((pr[0], pr[1], pr[2] - pr[0], pr[3] - pr[1]), W)
        tar_rect = (tar[0], tar[1], tar[2], tar[3])
        place_region(out, restored, (pr[0], pr[1], pr[2], pr[3]), tar_rect)
        n += 1
    if stats is not None:
        stats[f"billboard_{tag}_s"] = round(time.perf_counter() - t0, 2)
        stats["billboard_count"] = n
    return out


def apply_faces(out, orig, dets, device, stats=None, tag="", max_faces=3):
    H, W = orig.shape[:2]
    t0 = time.perf_counter()
    n = 0
    for d in dets[:max_faces]:
        if d["w"] < FACE_MIN_SIZE or d["h"] < FACE_MIN_SIZE:
            continue
        box = (d["x"], d["y"], d["w"], d["h"])
        pr = padded(box, FACE_PAD, W, H)
        crop = _crop_orig(orig, pr)
        restored = restore_sr_crop(crop, denoise=True, deblur=False,
                                   device=device)
        tar = scale_rect((pr[0], pr[1], pr[2] - pr[0], pr[3] - pr[1]), W)
        place_region(out, restored, (pr[0], pr[1], pr[2], pr[3]),
                     (tar[0], tar[1], tar[2], tar[3]))
        n += 1
    if stats is not None:
        stats[f"faces_{tag}_s"] = round(time.perf_counter() - t0, 2)
        stats["face_count"] = n
    return out


def build_variants(stem, orig, device, reuse_dir, stats=None):
    """Returns dict of variant -> BGR uint8 3840x2160 image."""
    out_dir = Path(reuse_dir) if reuse_dir else ROOT / "enhanced" / "swinir"
    a_path = out_dir / f"swinir_m_psnr_{stem}.png"
    if not a_path.exists():
        raise FileNotFoundError(f"Path A output missing: {a_path}")
    a = cv2.imread(str(a_path), cv2.IMREAD_COLOR)

    t0 = time.perf_counter()
    dets = faces_mod.detect_faces(orig)
    if stats is not None:
        stats["detect_s"] = round(time.perf_counter() - t0, 2)

    boxes = boxes_for(stem)
    b = a.copy()
    if boxes:
        b = apply_billboards(b, orig, stem, boxes, device, deblur=False,
                             stats=stats, tag="b")
    b_deblur = a.copy()
    if boxes:
        b_deblur = apply_billboards(b_deblur, orig, stem, boxes, device,
                                    deblur=True, stats=stats, tag="b_deblur")

    c = b.copy()
    if dets:
        c = apply_faces(c, orig, dets, device, stats=stats, tag="c")

    return {
        "A": a,
        "B": b,
        "B_deblur": b_deblur,
        "C": c,
        "faces": dets,
        "stats": stats,
    }