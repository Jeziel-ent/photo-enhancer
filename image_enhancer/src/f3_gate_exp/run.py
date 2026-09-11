"""Full-pipeline runner producing regression-harness-compatible reports for
the F3 t0 candidates, so they can be compared against baseline_final.json
with the standing harness.

Usage:
  .venv\\Scripts\\python.exe src\\f3_gate_exp\\run.py                  -> both t0=0.25 and t0=0.20
  .venv\\Scripts\\python.exe src\\f3_gate_exp\\run.py --t0 0.25
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import enhance  # noqa: E402
import quality  # noqa: E402
from regression_harness import (  # noqa: E402
    discover_benchmark_images,
    neutralize_billboard_regions,
    pixel_diff_metrics,
    load_report,
    compare_reports,
    print_comparison_table,
    PIXEL_CHANGED_THRESHOLD,
)
from f3_gate_exp.recipe import f3_gate_t0  # noqa: E402

REPORTS_DIR = ROOT / "reports" / "regression_baseline"
OUTPUTS_DIR = REPORTS_DIR / "outputs"


def run_candidate_pipeline(img, device, t0):
    """final_enhance with only F3's t0 changed."""
    target = (enhance.OUT_W, enhance.OUT_H)
    faithful = enhance.simple_upscale(img, target)
    d1, peak_vram, _ = enhance._final_d1_weak(img)

    with neutralize_billboard_regions():
        f3 = f3_gate_t0(faithful, d1, t0)
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
    return out


def run_one(path, device, t0):
    img = enhance.load_image(str(path))
    faithful = enhance.simple_upscale(img)

    t0s = time.perf_counter()
    out = run_candidate_pipeline(img, device, t0)
    runtime_s = time.perf_counter() - t0s

    peak = float(torch.cuda.max_memory_allocated() / 1048576.0) if device.type == "cuda" else None
    cur = float(torch.cuda.memory_allocated() / 1048576.0) if device.type == "cuda" else None
    if device.type == "cuda":
        torch.cuda.empty_cache()

    sim = quality.evaluate_similarity(faithful, out)
    nr = quality.evaluate_no_reference(out)
    nr_f = quality.evaluate_no_reference(faithful)
    pd = pixel_diff_metrics(faithful, out)

    record = {
        "image": path.name,
        "source": str(path.relative_to(ROOT.parent)) if ROOT.parent in path.parents else str(path),
        "orig_wh": [int(img.shape[1]), int(img.shape[0])],
        "out_wh": [int(out.shape[1]), int(out.shape[0])],
        "method": f"final_t0_{t0:.2f}",
        "device": str(device),
        "runtime_s": round(runtime_s, 2),
        "peak_vram_mb": round(peak, 1) if peak else None,
        "current_vram_mb": round(cur, 1) if cur else None,
        "similarity_vs_faithful": {k: round(v, 4) for k, v in sim.items()},
        "no_reference": {k: round(v, 4) for k, v in nr.items()},
        "no_reference_faithful": {k: round(v, 4) for k, v in nr_f.items()},
        "sharpness_gain_over_faithful": round(
            nr["sharpness"] / max(nr_f["sharpness"], 1e-6), 3),
        **{k: round(v, 4) for k, v in pd.items()},
        "output_sha256": hashlib.sha256(out.tobytes()).hexdigest(),
    }
    return record, out


def build_report(t0, save_outputs=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = discover_benchmark_images()
    records = []
    total_rt = 0.0
    for path in images:
        t_start = time.perf_counter()
        rec, out = run_one(path, device, t0)
        total_rt += time.perf_counter() - t_start
        if save_outputs:
            op = OUTPUTS_DIR / f"t0_{str(t0).replace('.','')}_{Path(path.name).stem}.png"
            enhance.save_image(out, str(op))
            rec["output_path"] = str(op.relative_to(ROOT.parent))
        records.append(rec)
        print(f"  {path.name}: psnr={rec['similarity_vs_faithful']['psnr']} "
              f"ssim={rec['similarity_vs_faithful']['ssim']} "
              f"rt={rec['runtime_s']}s", flush=True)

    img0 = enhance.load_image(str(images[0]))
    with neutralize_billboard_regions():
        o1 = run_candidate_pipeline(img0, device, t0)
    with neutralize_billboard_regions():
        o2 = run_candidate_pipeline(img0, device, t0)
    det = {"image": images[0].name,
           "rerun_identical": bool(np.array_equal(o1, o2)),
           "max_abs_diff": 0.0 if np.array_equal(o1, o2) else float(
               np.abs(o1.astype(np.int16) - o2.astype(np.int16)).max())}

    return {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "method": f"final_t0_{t0:.2f}",
        "device": str(device),
        "pixel_changed_threshold": PIXEL_CHANGED_THRESHOLD,
        "images": records,
        "determinism_check": det,
        "totals": {"image_count": len(records), "total_runtime_s": round(total_rt, 1)},
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--t0", type=float, default=None, help="single threshold; default=both")
    ap.add_argument("--no-save-outputs", action="store_true")
    args = ap.parse_args(argv)

    baseline = load_report(REPORTS_DIR / "baseline_final.json")
    t0s = [args.t0] if args.t0 else [0.25, 0.20]
    for t0 in t0s:
        report = build_report(t0, save_outputs=not args.no_save_outputs)
        cand_path = REPORTS_DIR / f"candidate_t0_{str(t0).replace('.', '')}.json"
        cand_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved candidate report -> {cand_path}\n")
        comparison = compare_reports(baseline, report)
        print_comparison_table(comparison)
        print(f"\n{'=' * 70}\n" if t0 not in t0s[-1:] else "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())