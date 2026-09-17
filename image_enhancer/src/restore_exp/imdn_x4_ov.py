"""IMDN x4 CPU inference via OpenVINO -- the CPU (no-GPU) execution backend.

Why this exists (see docs/PERFORMANCE_OPTIMIZATION.md's CPU stability
section for the full investigation): the packaged/frozen Windows EXE
crashed intermittently (STATUS_STACK_BUFFER_OVERRUN / Windows Event Log
BEX64, a stack-cookie check failure -- genuine memory corruption, not a
Python exception) during repeated CPU inference through PyTorch's CPU
backend (the earlier PyTorch IMDN runner, imdn_x4.py -- superseded by this
module and since removed; see docs/PERFORMANCE_OPTIMIZATION.md's history
section), reproducing with or without pywebview, with or
without the progress-ticker thread, with mkldnn disabled, and with
OMP_NUM_THREADS=1 -- i.e. not tied to any single thread-churn pattern
that could be isolated and designed around. That points to a genuine
thread-safety bug in PyTorch/oneDNN's CPU backend specific to this
project's frozen build, not a bug in this project's own code. It never
reproduced in an isolated PyInstaller+torch+cv2+IMDN repro run
standalone (no JobManager/HTTP server/pywebview) -- only inside the real
multi-threaded desktop app.

OpenVINO is a fully separate inference runtime (its own thread pool, no
shared state with PyTorch's ATen/oneDNN CPU backend) built specifically to
be embedded inside arbitrary host applications, which is exactly the
scenario that broke PyTorch's CPU backend here. Verified numerically
equivalent to the PyTorch reference (max abs diff 0.00055/1.0, i.e.
<0.15/255) and faster (measured ~0.12s vs ~0.2-0.6s per 256x256 tile on
this project's target CPU).

The GPU path (Restormer + SwinIR-M via PyTorch/CUDA in enhance.py) is
completely unaffected -- this module is only ever imported/used when
torch.cuda.is_available() is False.
"""

from pathlib import Path

import numpy as np
import openvino as ov

SCALE = 4
ONNX_NAME = "imdn_x4_256.onnx"

_core = None
_compiled = None


def _get_compiled():
    global _core, _compiled
    if _compiled is not None:
        return _compiled
    root = Path(__file__).resolve().parent.parent.parent
    onnx_path = root / "models" / ONNX_NAME
    if not onnx_path.exists():
        raise RuntimeError(
            f"IMDN OpenVINO ONNX model missing: {onnx_path}. This is a "
            "packaged-build asset (exported once from the official IMDN_x4 "
            "PyTorch checkpoint, byte-for-byte equivalent -- see "
            "docs/PERFORMANCE_OPTIMIZATION.md), not something a user "
            "provides.")
    _core = ov.Core()
    ov_model = _core.read_model(str(onnx_path))
    # IMDN's ONNX graph only has static shape info for the exported tile
    # size (256x256); reshape to accept exactly that -- tiled_forward below
    # always feeds fixed 256x256 tiles, matching the export, so this is not
    # a functional restriction in practice.
    _compiled = _core.compile_model(ov_model, "CPU")
    return _compiled


def prep_input(lq_bgr):
    lq = lq_bgr.astype(np.float32) / 255.0
    lq = lq[:, :, [2, 1, 0]]
    return np.ascontiguousarray(lq.transpose(2, 0, 1)[None, ...])


_TILE = 256  # must match the exported ONNX graph's static input shape


def tiled_forward(compiled, img_lq, overlap=16):
    """Same tiling/blending scheme as the superseded PyTorch IMDN runner
    (imdn_x4.py, removed -- see module docstring), but pure numpy
    (no torch tensors) so the CPU path never touches PyTorch's CPU backend
    at all -- see module docstring for why that matters here.

    Always feeds exactly 256x256 patches (the exported ONNX graph's static
    input shape): images smaller than that in either dimension are
    reflect-padded up first, matching the reflect-padding already used
    elsewhere in this codebase for window-size alignment (e.g.
    restormer.py's own multiple-of-8 padding)."""
    tile = _TILE
    b, c, h0, w0 = img_lq.shape
    ph, pw = max(0, tile - h0), max(0, tile - w0)
    if ph or pw:
        img_lq = np.pad(img_lq, ((0, 0), (0, 0), (0, ph), (0, pw)), mode="reflect")
    b, c, h, w = img_lq.shape
    stride = tile - overlap
    h_idx = list(range(0, h - tile, stride)) + [h - tile]
    w_idx = list(range(0, w - tile, stride)) + [w - tile]
    E = np.zeros((b, c, h * SCALE, w * SCALE), dtype=np.float32)
    Wt = np.zeros_like(E)
    out_port = compiled.output(0)
    for hi in h_idx:
        for wi in w_idx:
            patch = img_lq[..., hi:hi + tile, wi:wi + tile]
            out = compiled([patch])[out_port]
            E[..., hi * SCALE:(hi + tile) * SCALE, wi * SCALE:(wi + tile) * SCALE] += out
            Wt[..., hi * SCALE:(hi + tile) * SCALE, wi * SCALE:(wi + tile) * SCALE] += 1.0
    out_full = E / Wt
    return out_full[..., :h0 * SCALE, :w0 * SCALE]


def sr_bgr(bgr, overlap=16):
    """4x super-resolve a BGR uint8 image via OpenVINO CPU, returns uint8
    BGR. Signature intentionally drops the `device`/torch.device and
    `tile` arguments the superseded PyTorch IMDN runner's sr_bgr took
    (imdn_x4.py, removed) -- this backend is CPU-only by
    construction and always tiles at the exported ONNX graph's fixed
    256x256 shape (see _TILE in tiled_forward)."""
    compiled = _get_compiled()
    lq = prep_input(bgr)
    out = tiled_forward(compiled, lq, overlap)
    rgb = np.clip(out[0], 0.0, 1.0)
    bgr_out = (rgb[[2, 1, 0]].transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
    return bgr_out
