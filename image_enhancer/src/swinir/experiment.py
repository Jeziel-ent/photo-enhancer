"""Isolated SwinIR experiment (official real-world x4 models).

Reproduces the official JingyunLiang/SwinIR inference path (task=real_sr,
scale=4, window_size=8, reflection padding to a multiple of window_size, then
tiled forward with overlap). Saves full-size model output resized to the
project's 3840x2160 working resolution so it can be compared apples-to-apples
with the existing enhanced/ methods and their fidelity montages.

This experiment is intentionally self-contained:
  - network_swinir.py  (official, unmodified, vendored next to this file)
  - main_test_swinir_ref.py  (official reference, vendored, not imported)
It does not touch src/enhance.py's METHODS or the existing benchmark.

Usage:
  .venv/Scripts/python src/swinir/experiment.py [--tile 400] [--overlap 32]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from network_swinir import SwinIR as SwinIRNet
from enhance import simple_upscale
from quality import evaluate_similarity, compute_no_reference

OUT_W, OUT_H = 3840, 2160
WINDOW_SIZE = 8
SCALE = 4

# Official real-world SR x4 checkpoints (release v0.0), verified byte sizes.
MODELS = {
    "m_gan": {
        "pth": "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth",
        "bytes": 67129861,
        "large": False,
        "label": "SwinIR-M GAN",
    },
    "m_psnr": {
        "pth": "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_PSNR.pth",
        "bytes": 67129849,
        "large": False,
        "label": "SwinIR-M PSNR",
    },
    "l_gan": {
        "pth": "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth",
        "bytes": 142473939,
        "large": True,
        "label": "SwinIR-L GAN",
    },
    "l_psnr": {
        "pth": "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_PSNR.pth",
        "bytes": 142473947,
        "large": True,
        "label": "SwinIR-L PSNR",
    },
}

SSIM_FLOOR = 0.80   # same guardrails as compare_report.py
HIST_SEVERE = 0.65
PSNR_FLOOR = 20.0


def build_model(cfg, device):
    if cfg["large"]:
        model = SwinIRNet(
            upscale=SCALE, in_chans=3, img_size=64, window_size=WINDOW_SIZE,
            img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
            num_heads=[8] * 9, mlp_ratio=2, upsampler="nearest+conv",
            resi_connection="3conv")
    else:
        model = SwinIRNet(
            upscale=SCALE, in_chans=3, img_size=64, window_size=WINDOW_SIZE,
            img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
            num_heads=[6] * 6, mlp_ratio=2, upsampler="nearest+conv",
            resi_connection="1conv")

    path = ROOT / "models" / "swinir" / cfg["pth"]
    actual = path.stat().st_size if path.exists() else 0
    if actual != cfg["bytes"]:
        raise RuntimeError(
            f"Model file size mismatch for {cfg['pth']}: expected "
            f"{cfg['bytes']} bytes, found {actual}. Refusing to run "
            "a possibly-tampered/unofficial checkpoint.")
    sd = torch.load(path, map_location="cpu")
    param_key = "params_ema"
    sd = sd[param_key] if param_key in sd.keys() else sd
    model.load_state_dict(sd, strict=True)
    return model.eval().to(device)


def prep_input(lq_bgr):
    """Official preprocessing: BGR uint8 -> NCHW RGB float [0,1], reflect-pad
    so H,W are multiples of window_size."""
    lq = lq_bgr.astype(np.float32) / 255.0
    lq = lq[:, :, [2, 1, 0]]                       # BGR -> RGB
    t = torch.from_numpy(lq).float().permute(2, 0, 1).unsqueeze(0)  # NCHW
    _, _, h0, w0 = t.size()
    hp = (h0 // WINDOW_SIZE + 1) * WINDOW_SIZE - h0
    wp = (w0 // WINDOW_SIZE + 1) * WINDOW_SIZE - w0
    t = torch.cat([t, torch.flip(t, [2])], 2)[:, :, :h0 + hp, :]
    t = torch.cat([t, torch.flip(t, [3])], 3)[:, :, :, :w0 + wp]
    return t, (h0, w0)


def tiled_forward(model, img_lq, device, tile, overlap):
    """Official tiled test() with equal-weight overlap blending."""
    b, c, h, w = img_lq.size()
    tile = min(tile, h, w)
    assert tile % WINDOW_SIZE == 0, "tile must be a multiple of window_size"
    stride = tile - overlap
    h_idx = list(range(0, h - tile, stride)) + [h - tile]
    w_idx = list(range(0, w - tile, stride)) + [w - tile]
    E = torch.zeros(b, c, h * SCALE, w * SCALE, device=device)
    Wt = torch.zeros_like(E)
    with torch.no_grad():
        for hi in h_idx:
            for wi in w_idx:
                patch = img_lq[..., hi:hi + tile, wi:wi + tile]
                out = model(patch)
                E[..., hi * SCALE:(hi + tile) * SCALE,
                  wi * SCALE:(wi + tile) * SCALE].add_(out)
                Wt[..., hi * SCALE:(hi + tile) * SCALE,
                   wi * SCALE:(wi + tile) * SCALE].add_(torch.ones_like(out))
    return E.div_(Wt)


def run_model(model_cfg, device, images, tile, overlap):
    cfg = MODELS[model_cfg]
    model = build_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    rows, peaks = [], []
    for stem in sorted(images):
        p = ROOT / "originals" / f"{stem}.jpeg"
        src = cv2.imread(str(p), cv2.IMREAD_COLOR)
        t, (h0, w0) = prep_input(src)
        t = t.to(device)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = tiled_forward(model, t, device, tile, overlap)
        out = out[..., :h0 * SCALE, :w0 * SCALE]
        peak = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        dt = time.perf_counter() - t0

        rgb = out.squeeze(0).float().cpu().clamp_(0, 1).numpy()
        bgr = np.transpose(rgb[[2, 1, 0]], (1, 2, 0))
        img4x = (bgr * 255.0).round().astype(np.uint8)
        if img4x.shape[:2][::-1] != (OUT_W, OUT_H):
            img4x = cv2.resize(img4x, (OUT_W, OUT_H), interpolation=cv2.INTER_LANCZOS4)

        out_dir = ROOT / "enhanced" / "swinir"
        out_dir.mkdir(parents=True, exist_ok=True)
        opath = out_dir / f"swinir_{model_cfg}_{stem}.png"
        cv2.imwrite(str(opath), img4x)

        rows.append({"model": cfg["label"], "image": f"{stem}.jpeg",
                     "time_s": round(dt, 2), "peak_vram_mb": round(peak, 1),
                     "native_w": out.shape[3], "native_h": out.shape[2]})
        print(f"  {cfg['label']} {stem}: {dt:.1f}s  peak VRAM {peak:.0f} MB -> {opath.name}")
    return rows, n_params


def metrics_and_montage(model_cfg, images, overlap):
    cfg = MODELS[model_cfg]
    out_dir = ROOT / "enhanced" / "swinir"
    mont_dir = ROOT / "reports" / "swinir" / "montages"
    mont_dir.mkdir(parents=True, exist_ok=True)
    rows, warns = [], []
    for stem in sorted(images):
        p = ROOT / "originals" / f"{stem}.jpeg"
        orig = cv2.imread(str(p), cv2.IMREAD_COLOR)
        faithful = simple_upscale(orig, (OUT_W, OUT_H))
        enh = cv2.imread(str(out_dir / f"swinir_{model_cfg}_{stem}.png"), cv2.IMREAD_COLOR)
        sim = evaluate_similarity(faithful, enh)
        nr = compute_no_reference(enh)
        nr_base = compute_no_reference(faithful)
        rows.append({
            "model": cfg["label"], "image": f"{stem}.jpeg",
            "psnr": round(sim["psnr"], 2), "ssim": round(sim["ssim"], 4),
            "nrmse": round(sim["nrmse"], 4), "hist_sim": round(sim["hist_sim"], 4),
            "edge_align": round(sim["edge_align"], 4),
            "sharp_enh": round(nr["sharpness"], 2), "sharp_base": round(nr_base["sharpness"], 2),
            "sharp_gain_x": round(nr["sharpness"] / (nr_base["sharpness"] + 1e-9), 3),
            "edge_gain_x": round(nr["edge_density"] / (nr_base["edge_density"] + 1e-9), 3),
            "noise": round(nr["noise"], 3),
        })
        checks = []
        if sim["ssim"] < SSIM_FLOOR:
            checks.append(f"SSIM {sim['ssim']:.3f}<{SSIM_FLOOR}")
        if sim["hist_sim"] < HIST_SEVERE:
            checks.append(f"hist {sim['hist_sim']:.3f}<{HIST_SEVERE}")
        if sim["psnr"] < PSNR_FLOOR:
            checks.append(f"PSNR {sim['psnr']:.2f}<{PSNR_FLOOR}")
        if sim["edge_align"] < 0.9:
            checks.append(f"edge_align {sim['edge_align']:.3f}<0.9")
        if checks:
            warns.append(f"{cfg['label']}/{stem}.jpeg FIDELITY DRIFT: {', '.join(checks)}")

        # 5-way visual montage: baseline | realesrgan | repair_v2 | swinir
        panels = [
            (faithful, f"Original - faithful Lanczos baseline ({OUT_W}x{OUT_H})", (255, 200, 60)),
            (cv2.imread(str(ROOT / "enhanced" / f"realesrgan_{stem}.png"), cv2.IMREAD_COLOR), "Real-ESRGAN x4plus", (80, 255, 120)),
            (cv2.imread(str(ROOT / "enhanced" / f"repair_v2_{stem}.png"), cv2.IMREAD_COLOR), "Repair V2 (conservative)", (80, 255, 120)),
            (enh, f"{cfg['label']} x4 (official)", (80, 200, 255)),
        ]
        mont = build_montage(panels, mont_dir / f"cmp_swinir_{model_cfg}_{stem}.png")
        rows[-1]["montage"] = str(mont.relative_to(ROOT))
    return rows, warns


def build_montage(panels, out_path, scale=0.4):
    label_h = 30
    rows_img = []
    tw, th = 0, 0
    for img, label, color in panels:
        h, w = img.shape[:2]
        rw, rh = int(w * scale), int(h * scale)
        p = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_AREA)
        p = cv2.copyMakeBorder(p, label_h, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
        cv2.putText(p, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        rows_img.append(p)
        tw, th = rw, rh
    sep = np.full((th, tw, 3), 255, np.uint8)
    frames = []
    for i, p in enumerate(rows_img):
        frames.append(p)
        if i < len(rows_img) - 1:
            frames.append(sep)
    mont = np.vstack(frames)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), mont)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", type=int, default=400)
    ap.add_argument("--overlap", type=int, default=32)
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--metrics-only", action="store_true",
                    help="skip inference, recompute metrics/montages from existing outputs")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    images = sorted(p.stem for p in (ROOT / "originals").glob("*.jpeg"))
    assert images, "no originals found"

    run_rows, metric_rows = [], []
    if args.metrics_only:
        warns_all = []
        for mc in args.models:
            r, w = metrics_and_montage(mc, images, args.overlap)
            metric_rows += r
            warns_all += w
    else:
        for mc in args.models:
            print(f"== {mc} ({MODELS[mc]['label']}) ==")
            rr, n_params = run_model(mc, device, images, args.tile, args.overlap)
            r, w = metrics_and_montage(mc, images, args.overlap)
            run_rows += rr
            metric_rows += r
            print(f"   params: {n_params:.2f}M")
            for q in w:
                print("   WARN:", q)

    if run_rows:
        pd.DataFrame(run_rows).to_csv(ROOT / "reports" / "swinir" / "runtime.csv", index=False)
    df = pd.DataFrame(metric_rows)
    df.to_csv(ROOT / "reports" / "swinir" / "swinir_metrics.csv", index=False)

    print("\nRuntime summary:")
    if run_rows:
        rt = pd.DataFrame(run_rows).groupby("model").agg(
            time_s=("time_s", "mean"), vram_mb=("peak_vram_mb", "max")).round(2)
        print(rt.to_string())

    print("\nMean fidelity vs faithful original upscale:")
    agg = df.groupby("model")[["psnr", "ssim", "hist_sim", "edge_align",
                               "sharp_gain_x", "edge_gain_x", "noise"]].mean().round(3)
    print(agg.to_string())

    shots = {
        "runtime": ROOT / "reports" / "swinir" / "runtime.csv",
        "metrics": ROOT / "reports" / "swinir" / "swinir_metrics.csv",
        "montages": ROOT / "reports" / "swinir" / "montages",
        "outputs": ROOT / "enhanced" / "swinir",
    }
    with open(ROOT / "reports" / "swinir" / "summary.json", "w") as f:
        json.dump({k: str(v.relative_to(ROOT)) for k, v in shots.items()}, f, indent=2)
    print("\nExperiment artifacts ->")
    for k, v in shots.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()