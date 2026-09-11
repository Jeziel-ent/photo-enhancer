"""F3 local-envelope experiment runner.

Runs the full production pipeline with F3's flat ±14 envelope replaced by
the locally-derived self-envelope. Produces a JSON report in the same format
as regression_harness.py for direct comparison against the locked baseline.

Usage:
  .venv\Scripts\python.exe src\f3_local_exp\run.py
  .venv\Scripts\python.exe src\f3_local_exp\run.py --compare
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
REPO_ROOT = ROOT.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import enhance  # noqa: E402  (production, READ-ONLY)
import quality  # noqa: E402
from regression_harness import (  # noqa: E402
    discover_benchmark_images,
    neutralize_billboard_regions,
    pixel_diff_metrics,
    run_one_image,
    run_determinism_check,
    save_report,
    load_report,
    compare_reports,
    print_comparison_table,
    OUTPUTS_DIR,
    PIXEL_CHANGED_THRESHOLD,
    DEFAULT_TOLERANCES,
)

from f3_local_exp.recipe import f3_self_envelope  # noqa: E402

EXPERIMENT_NAME = "f3_local_env"
REPORTS_DIR = ROOT / "reports" / "regression_baseline"


def _vram_reset(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def _vram_read(device):
    if device.type != "cuda":
        return {"peak_vram_mb": None, "current_vram_mb": None}
    peak = torch.cuda.max_memory_allocated() / 1048576.0
    current = torch.cuda.memory_allocated() / 1048576.0
    torch.cuda.empty_cache()
    return {"peak_vram_mb": round(float(peak), 1), "current_vram_mb": round(float(current), 1)}


def run_candidate_pipeline(img, device):
    """Run the full final_enhance pipeline with self-envelope F3.

    Reproduces the exact same stages as enhance.final_enhance(), only
    replacing _final_f3_natural with f3_self_envelope."""
    target = (enhance.OUT_W, enhance.OUT_H)

    faithful = enhance.simple_upscale(img, target)

    d1, peak_vram_mb, dev_str = enhance._final_d1_weak(img)

    # F3 variant: self-envelope instead of flat ±14
    with neutralize_billboard_regions():
        f3 = f3_self_envelope(faithful, d1, w_s=6.0, clip=18.0, t0=0.30, ksize=5)

    # Downstream stages — IDENTICAL to production
    f3_ms = enhance._final_multiscale_detail(f3)
    f3_plus = enhance._final_whole_frame_detail(f3_ms)

    boxes4, sigmas, defocus_boxes = [], [], []
    with neutralize_billboard_regions():
        boxes = enhance._billboard_boxes(img)
    if boxes:
        device_t = torch.device(device)
        out = f3_plus.copy()
        for (x, y, w, h) in boxes:
            vb = enhance._final_valid_box((x, y, w, h), img.shape[1], img.shape[0])
            if vb is None:
                continue
            vx, vy, vw, vh = vb
            bx, by, bw, bh = enhance._final_scale_box(vb, img.shape[1])
            xs = min(bx + bw, out.shape[1])
            ys = min(by + bh, out.shape[0])
            if xs - bx < 8 or ys - by < 8:
                continue
            box4 = (bx, by, xs - bx, ys - by)
            boxes4.append(box4)
            nc = img[vy:vy + vh, vx:vx + vw]
            sig = enhance._final_psf_sigma(cv2.cvtColor(nc, cv2.COLOR_BGR2GRAY))
            sigmas.append(sig)
            if sig >= enhance._FINAL_DEFOCUS_SIGMA:
                defocus_boxes.append(box4)
            board_bgr = enhance._final_candidate_a_board(
                nc, device_t, target_wh=(box4[2], box4[3]))
            out = enhance._final_composite_box_bgr(out, board_bgr, box4)
    else:
        out = f3_plus

    if out.shape[1] != target[0] or out.shape[0] != target[1]:
        out = cv2.resize(out, target, interpolation=cv2.INTER_LANCZOS4)
    return out, {"peak_vram_mb": peak_vram_mb, "boxes4": boxes4,
                 "psf_sigmas": sigmas, "defocus_boxes": defocus_boxes}


def run_one_candidate(path, device):
    """Run candidate pipeline on one image and compute all metrics."""
    img = enhance.load_image(str(path))
    faithful = enhance.simple_upscale(img)

    _vram_reset(device)
    t0 = time.perf_counter()
    out, stages = run_candidate_pipeline(img, device)
    runtime_s = time.perf_counter() - t0
    vram = _vram_read(device)

    sim = quality.evaluate_similarity(faithful, out)
    nr_out = quality.evaluate_no_reference(out)
    nr_faithful = quality.evaluate_no_reference(faithful)
    pdiff = pixel_diff_metrics(faithful, out)

    record = {
        "image": path.name,
        "source": str(path.relative_to(REPO_ROOT)) if REPO_ROOT in path.parents else str(path),
        "orig_wh": [int(img.shape[1]), int(img.shape[0])],
        "out_wh": [int(out.shape[1]), int(out.shape[0])],
        "method": f"final_{EXPERIMENT_NAME}",
        "device": str(device),
        "runtime_s": round(runtime_s, 2),
        **vram,
        "similarity_vs_faithful": {k: round(v, 4) for k, v in sim.items()},
        "no_reference": {k: round(v, 4) for k, v in nr_out.items()},
        "no_reference_faithful": {k: round(v, 4) for k, v in nr_faithful.items()},
        "sharpness_gain_over_faithful": round(
            nr_out["sharpness"] / max(nr_faithful["sharpness"], 1e-6), 3),
        **{k: round(v, 4) for k, v in pdiff.items()},
        "output_sha256": "",  # filled below
    }
    return record, out


def build_candidate_report(image_paths=None, save_outputs=True):
    """Build a full candidate report for the self-envelope F3 variant."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = image_paths or discover_benchmark_images()

    if save_outputs:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    total_runtime = 0.0
    for path in images:
        print(f"  running candidate on {path.name} ...", flush=True)
        t_img0 = time.perf_counter()
        record, out = run_one_candidate(path, device)
        total_runtime += time.perf_counter() - t_img0

        # Compute output sha256
        import hashlib
        record["output_sha256"] = hashlib.sha256(out.tobytes()).hexdigest()

        if save_outputs:
            out_path = OUTPUTS_DIR / f"{EXPERIMENT_NAME}_{Path(path.name).stem}.png"
            enhance.save_image(out, str(out_path))
            record["output_path"] = str(out_path.relative_to(REPO_ROOT))
        records.append(record)
        print(f"    runtime={record['runtime_s']}s peak_vram_mb={record['peak_vram_mb']} "
              f"psnr={record['similarity_vs_faithful']['psnr']} "
              f"ssim={record['similarity_vs_faithful']['ssim']} "
              f"pct_pixels_changed={record['pct_pixels_changed']}%", flush=True)

    # Determinism check on first image
    print(f"  determinism check on {images[0].name} ...", flush=True)
    img0 = enhance.load_image(str(images[0]))
    with neutralize_billboard_regions():
        out1, _ = run_candidate_pipeline(img0, device)
    with neutralize_billboard_regions():
        out2, _ = run_candidate_pipeline(img0, device)
    identical = bool(np.array_equal(out1, out2))
    max_abs_diff = 0.0 if identical else float(
        np.abs(out1.astype(np.int16) - out2.astype(np.int16)).max())
    determinism = {"image": images[0].name, "rerun_identical": identical,
                   "max_abs_diff": max_abs_diff}
    print(f"    rerun_identical={identical}", flush=True)

    return {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "method": f"final_{EXPERIMENT_NAME}",
        "device": str(device),
        "pixel_changed_threshold": PIXEL_CHANGED_THRESHOLD,
        "images": records,
        "determinism_check": determinism,
        "totals": {
            "image_count": len(records),
            "total_runtime_s": round(total_runtime, 1),
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="F3 local-envelope experiment runner")
    parser.add_argument("--compare", action="store_true",
                        help="run candidate and compare against locked baseline")
    parser.add_argument("--no-save-outputs", action="store_true")
    args = parser.parse_args(argv)

    baseline_path = REPORTS_DIR / "baseline_final.json"
    candidate_path = REPORTS_DIR / f"candidate_{EXPERIMENT_NAME}.json"

    report = build_candidate_report(save_outputs=not args.no_save_outputs)
    save_report(report, candidate_path)
    print(f"\nCandidate report saved to {candidate_path}")

    if args.compare:
        if not baseline_path.exists():
            print(f"ERROR: baseline not found at {baseline_path}")
            print("Run: .venv\\Scripts\\python.exe src\\regression_harness.py --lock-baseline")
            return 1
        baseline = load_report(baseline_path)
        comparison = compare_reports(baseline, report)
        print_comparison_table(comparison)
        return 0 if comparison["overall_ok"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
