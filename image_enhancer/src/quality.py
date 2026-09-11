"""Quality/fidelity metrics for image enhancement benchmarking.

Measures signal-level similarity (PSNR, SSIM) against a reference, plus
no-reference perceptual metrics (NIQE) and content-consistency checks
(histogram similarity) to verify we did NOT alter the underlying content.
"""

import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import normalized_root_mse as nrmse_metric


def to_float_rgb(img):
    """Convert BGR uint8 ndarray to RGB float32 in [0,1]."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 and img.shape[2] == 3 else img
    return rgb.astype(np.float32) / 255.0


def compute_psnr(ref, pred):
    """PSNR in dB between reference and predicted (uint8 BGR arrays)."""
    return psnr_metric(ref, pred)


def compute_ssim(ref, pred):
    """SSIM in [0,1] between reference and predicted (uint8 BGR arrays)."""
    return ssim_metric(ref, pred, channel_axis=2, data_range=255)


def compute_nrmse(ref, pred):
    """Normalized root mean square error (lower is better)."""
    return nrmse_metric(ref, pred)


def compute_sharpness(img):
    """Laplacian variance (higher = sharper). Robust no-reference sharpness."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def compute_noise_estimate(img, bs=8, max_blocks=800):
    """Imateston-style noise estimate using flat regions of the image.
    Uses a bounded random sample of small patches (fast on 4K images)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = gray.shape
    nrows = H // bs
    ncols = W // bs
    if nrows == 0 or ncols == 0:
        return 0.0
    idx = np.arange(nrows * ncols)
    idx = idx[np.random.default_rng(0).permutation(len(idx))][:max_blocks]
    rs = (idx // ncols) * bs
    cs = (idx % ncols) * bs
    blocks = np.stack([gray[r:r + bs, c:c + bs] for r, c in zip(rs, cs)])
    stds = blocks.reshape(len(blocks), -1).std(axis=1)
    return float(np.median(stds))


def compute_contrast(img):
    """RMS contrast of the luminance plane (higher = more contrast)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return float(gray.std())


def _edge_density(gray):
    edges = cv2.Canny(gray, 50, 150)
    return float(edges.sum() / edges.size)


def compute_no_reference(img):
    """No-reference enhancement-quality metrics (all higher=better except
    noise which should be low but NOT zero)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {
        "sharpness": compute_sharpness(img),
        "noise": compute_noise_estimate(img),
        "contrast": compute_contrast(img),
        "edge_density": _edge_density(gray),
    }


def histogram_similarity(ref, pred):
    """Histogram intersection in [0,1] of the full image.

    A high value (>0.9) indicates the enhanced image's global color/intensity
    distribution closely matches the source => content preserved.
    """
    scores = []
    for c in range(3):
        r_hist = cv2.calcHist([ref], [c], None, [256], [0, 256])
        p_hist = cv2.calcHist([pred], [c], None, [256], [0, 256])
        r_hist = r_hist / (np.sum(r_hist) + 1e-8)
        p_hist = p_hist / (np.sum(p_hist) + 1e-8)
        scores.append(float(np.minimum(r_hist, p_hist).sum()))
    return float(np.mean(scores))


def edge_density_change(ref, pred):
    """Ratio (pred_edges / ref_edges); ~1.0 means sharpening improved edges
    without removing them. Values >1 indicate better-defined edges."""
    def edge_density(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(g, 50, 150)
        return float(edges.sum() / edges.size)

    ref_d = edge_density(ref)
    pred_d = edge_density(pred)
    if ref_d == 0:
        return 1.0
    return pred_d / ref_d


def evaluate_similarity(ref, pred):
    """Full similarity block (requires both images same spatial size)."""
    if ref.shape != pred.shape:
        raise ValueError(
            f"Shapes differ: ref {ref.shape} vs pred {pred.shape}. "
            "Resize before measuring."
        )
    return {
        "psnr": float(compute_psnr(ref, pred)),
        "ssim": float(compute_ssim(ref, pred)),
        "nrmse": float(compute_nrmse(ref, pred)),
        "hist_sim": float(histogram_similarity(ref, pred)),
        "edge_align": float(edge_alignment(ref, pred)),
    }


def edge_alignment(ref, pred):
    """Fraction of edges in the enhanced image that exist in the same place
    as in the reference (original). High value (~1.0) means geometry/content
    is preserved — no hallucinated or displaced structures.

    Uses a common gradient-magnitude mask with a small tolerance band so
    legitimate sharpening does not count as misalignment.
    """
    def edge_mask(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
        mag = cv2.magnitude(gx, gy)
        _, m = cv2.threshold(mag, 0.12 * mag.max(), 1, cv2.THRESH_BINARY)
        return m.astype(bool)

    ref_e = edge_mask(ref)
    pred_e = edge_mask(pred)
    if pred_e.sum() == 0:
        return 0.0
    # dilated ref edges = tolerance band around original structure
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    band = cv2.dilate(ref_e.astype(np.uint8), k).astype(bool)
    matched = pred_e & band
    return float(matched.sum() / pred_e.sum())


def evaluate_no_reference(pred):
    """Quality-only metrics that do not require a reference."""
    return compute_no_reference(pred)
