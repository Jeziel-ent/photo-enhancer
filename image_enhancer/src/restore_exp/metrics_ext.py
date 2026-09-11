"""Region-level fidelity metrics + difference heatmaps (Part 5).

Compares each variant region against the SAME region of the faithful Lanczos
upscale of the original (the preserve-source reference), using the project's
quality.py metrics plus region-local sharpness/noise so the report can reason
about text-edge and face sharpness without inventing a metric.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from quality import (  # noqa: E402
    compute_no_reference, compute_psnr, edge_alignment,
    histogram_similarity, compute_ssim,
)


def crop_rect(img, rect):
    x, y, w, h = rect
    x = max(0, min(x, img.shape[1] - 1))
    y = max(0, min(y, img.shape[0] - 1))
    w = min(w, img.shape[1] - x)
    h = min(h, img.shape[0] - y)
    if w < 1 or h < 1:
        return None
    return img[y:y + h, x:x + w]


def region_metrics(ref_crop, pred_crop):
    if ref_crop is None or pred_crop is None:
        return None
    if ref_crop.shape[:2] != pred_crop.shape[:2]:
        h = min(ref_crop.shape[0], pred_crop.shape[0])
        w = min(ref_crop.shape[1], pred_crop.shape[1])
        ref_crop = ref_crop[:h, :w]
        pred_crop = pred_crop[:h, :w]
    if min(ref_crop.shape[:2]) < 4:
        return None
    nr = compute_no_reference(pred_crop)
    nr_base = compute_no_reference(ref_crop)
    return {
        "psnr": round(float(compute_psnr(ref_crop, pred_crop)), 2),
        "ssim": round(float(compute_ssim(ref_crop, pred_crop)), 4),
        "hist_sim": round(float(histogram_similarity(ref_crop, pred_crop)), 4),
        "edge_align": round(float(edge_alignment(ref_crop, pred_crop)), 4),
        "sharp": round(nr["sharpness"], 2),
        "sharp_ref": round(nr_base["sharpness"], 2),
        "sharp_gain_x": round(nr["sharpness"] / (nr_base["sharpness"] + 1e-9), 3),
        "noise": round(nr["noise"], 3),
        "noise_ref": round(nr_base["noise"], 3),
    }


def diff_heatmap(ref_crop, pred_crop, out_path, scale_vis=1.0):
    if ref_crop is None or pred_crop is None:
        return
    if ref_crop.shape[:2] != pred_crop.shape[:2]:
        h = min(ref_crop.shape[0], pred_crop.shape[0])
        w = min(ref_crop.shape[1], pred_crop.shape[1])
        ref_crop = ref_crop[:h, :w]
        pred_crop = pred_crop[:h, :w]
    diff = cv2.absdiff(ref_crop.astype(np.float32), pred_crop.astype(np.float32))
    mag = diff.mean(axis=2)
    mag = mag / (mag.max() + 1e-9)
    viz = np.clip(mag * 255, 0, 255).astype(np.uint8)
    viz = cv2.applyColorMap(viz, cv2.COLORMAP_JET)
    if scale_vis != 1.0:
        viz = cv2.resize(viz, None, fx=scale_vis, fy=scale_vis,
                         interpolation=cv2.INTER_NEAREST)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), viz)