"""Experiment 3 follow-up: expanded-dataset validation of F3 t0=0.25.

Runs the production final pipeline with ONLY F3's t0 changed (0.25) and
with the locked production t0 (0.30) on an expanded real-world dataset:
  - the 6 canonical originals (originals/)
  - mumbai-01.jpeg (genuinely distinct real upload under backend/.deckstore)
  - 21 distinct site photos (archive/.../phase2_presentation/test_data/assets/
    site_*.jpeg/png), all content-distinct from the originals

Byte-identical duplicates are excluded by SHA-256 content hash.

For every image it records (against the faithful 4K Lanczos baseline):
  PSNR, SSIM, NRMSE, hist_sim, edge_align, sharpness, noise, contrast,
  edge_density, mean_pixel_diff, pct_pixels_changed, runtime_s, peak VRAM,
  and output SHA-256 — for BOTH t0 values. Also runs a repeat-run
  determinism check (byte-for-byte) on a representative subset.

Outputs:
  reports/regression_baseline/validate_t0_025.json   (candidate report)
  reports/regression_baseline/validate_t0_030.json   (reference report)
  src/f3_gate_exp/validate_outputs/{t0}/{name}.png   (saved 4K outputs)
"""

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
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
    neutralize_billboard_regions,
    pixel_diff_metrics,
    load_report,
    compare_reports,
    print_comparison_table,
    DEFAULT_TOLERANCES,
    PIXEL_CHANGED_THRESHOLD,
    _file_sha256,
)
from f3_gate_exp.recipe import f3_gate_t0  # noqa: E402

REPORTS_DIR = ROOT / "reports" / "regression_baseline"
VALIDATE_OUT = ROOT / "src" / "f3_gate_exp" / "validate_outputs"

REF_T0 = 0.30
CAND_T0 = 0.25

# Additional distinct real-world site photos (phase-2 presentation test data).
SITE_GLOB = (
    ROOT.parent / "archive" / "ppt_automation" / "phase2_presentation"
    / "test_data" / "assets" / "site_*.*"
)
SITE_EXTS = {".jpeg", ".jpg", ".png"}


def discover_validation_images() -> list[Path]:
    """originals + mumbai deckstore upload + distinct site photos, deduped."""
    seen = set()
    images = []

    def add(path: Path) -> None:
        h = _file_sha256(path)
        if h in seen:
            print(f"  skipping {path.name} (duplicate content)")
            return
        seen.add(h)
        images.append(path)

    for p in sorted((ROOT / "originals").glob("*.jpeg")):
        add(p)
    mum = ROOT.parent / "backend" / ".deckstore" / \
        "testclient-autumn-launch" / "uploads" / "mumbai-01.jpeg"
    if mum.is_file():
        add(mum)
    if SITE_GLOB.parent.is_dir():
        for p in sorted(SITE_GLOB.parent.glob("site_*.*")):
            if p.suffix.lower() in SITE_EXTS and p.is_file():
                add(p)
    return images


def run_candidate(img, device, t0):
    """Production final pipeline with only F3's t0 changed (no board boxes —
    matched to what real uploads get through backend/engine_adapter.py)."""
    target = (enhance.OUT_W, enhance.OUT_H)
    faithful = enhance.simple_upscale(img, target)
    with neutralize_billboard_regions():
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        d1, _, _ = enhance._final_d1_weak(img)
        f3 = f3_gate_t0(faithful, d1, t0)
    f3_ms = enhance._final_multiscale_detail(f3)
    out = enhance._final_whole_frame_detail(f3_ms)
    peak = (float(torch.cuda.max_memory_allocated() / 1048576.0)
            if device.type == "cuda" else None)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out, peak


def measure(img, out, runtime_s, peak_vram, device, method, path):
    faithful = enhance.simple_upscale(img)
    sim = quality.evaluate_similarity(faithful, out)
    nr = quality.evaluate_no_reference(out)
    nr_f = quality.evaluate_no_reference(faithful)
    pd = pixel_diff_metrics(faithful, out)
    return {
        "image": path.name,
        "source": str(path.relative_to(ROOT.parent))
        if ROOT.parent in path.parents else str(path),
        "orig_wh": [int(img.shape[1]), int(img.shape[0])],
        "out_wh": [int(out.shape[1]), int(out.shape[0])],
        "method": method,
        "device": str(device),
        "runtime_s": round(runtime_s, 2),
        "peak_vram_mb": round(peak_vram, 1) if peak_vram else None,
        "current_vram_mb": None,
        "similarity_vs_faithful": {k: round(v, 4) for k, v in sim.items()},
        "no_reference": {k: round(v, 4) for k, v in nr.items()},
        "no_reference_faithful": {k: round(v, 4) for k, v in nr_f.items()},
        "sharpness_gain_over_faithful": round(
            nr["sharpness"] / max(nr_f["sharpness"], 1e-6), 3),
        **{k: round(v, 4) for k, v in pd.items()},
        "output_sha256": hashlib.sha256(out.tobytes()).hexdigest(),
    }


def determinism(path, device, t0, repeats=2):
    img = enhance.load_image(str(path))
    outs = []
    for _ in range(repeats):
        with neutralize_billboard_regions():
            o, _ = run_candidate(img, device, t0)
        outs.append(o)
    identical = all(np.array_equal(outs[0], o) for o in outs[1:])
    max_diff = 0.0
    for o in outs[1:]:
        d = float(np.abs(outs[0].astype(np.int16) - o.astype(np.int16)).max())
        max_diff = max(max_diff, d)
    return {"image": path.name, "t0": t0,
            "repeats": repeats, "rerun_identical": identical,
            "max_abs_diff": max_diff}


def build(t0, images, device, save_outputs=True):
    records, total_rt = [], 0.0
    outdir = VALIDATE_OUT / f"t0_{t0:.2f}"
    outdir.mkdir(parents=True, exist_ok=True)
    for path in images:
        img = enhance.load_image(str(path))
        t_img = time.perf_counter()
        out, peak = run_candidate(img, device, t0)
        rt = time.perf_counter() - t_img
        total_rt += rt
        rec = measure(img, out, rt, peak, device, f"final_t0_{t0:.2f}", path)
        if save_outputs:
            op = outdir / f"{Path(path.name).stem}.png"
            enhance.save_image(out, str(op))
            rec["output_path"] = str(op.relative_to(ROOT.parent))
        records.append(rec)
        print(f"  {path.name:<28} psnr={rec['similarity_vs_faithful']['psnr']:.4f} "
              f"ssim={rec['similarity_vs_faithful']['ssim']:.4f} "
              f"rt={rt:5.1f}s vram={rec['peak_vram_mb']}", flush=True)
    return {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "method": f"final_t0_{t0:.2f}",
        "device": str(device),
        "pixel_changed_threshold": PIXEL_CHANGED_THRESHOLD,
        "t0": t0,
        "images": records,
        "determinism_check": None,
        "totals": {"image_count": len(records),
                   "total_runtime_s": round(total_rt, 1)},
    }


def aggregate(report):
    """Aggregate + worst-case summary across all images."""
    m = defaultdict(list)
    for r in report["images"]:
        for k, v in r["similarity_vs_faithful"].items():
            m[f"sim_{k}"].append(v)
        for k, v in r["no_reference"].items():
            m[f"nr_{k}"].append(v)
        for k in ("mean_pixel_diff", "pct_pixels_changed", "runtime_s"):
            m[k].append(r[k])
        if r["peak_vram_mb"] is not None:
            m["peak_vram_mb"].append(r["peak_vram_mb"])
    agg = {}
    worst = {}
    for k, vals in m.items():
        if not vals:
            continue
        arr = np.array(vals)
        agg[k] = {"mean": float(arr.mean()), "min": float(arr.min()),
                  "max": float(arr.max())}
        worst[k] = {"worst_value": float(arr.min()), "image": None}
        i = int(arr.argmin())
        worst[k]["image"] = report["images"][i]["image"]
    return {"aggregate": agg, "worst_case": worst}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-save-outputs", action="store_true")
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = discover_validation_images()
    print(f"Validation set: {len(images)} distinct images")
    for p in images:
        print(f"  {p}")

    ref = build(REF_T0, images, device, save_outputs=not args.no_save_outputs)
    cand = build(CAND_T0, images, device, save_outputs=not args.no_save_outputs)

    # determinism on a representative subset (1 original, mumbai, 2 sites)
    det_paths = [images[0]]
    for p in images:
        if p.name == "mumbai-01.jpeg":
            det_paths.append(p)
    site_imgs = [p for p in images if p.name.startswith("site_")]
    if len(site_imgs) >= 2:
        det_paths.extend([site_imgs[0], site_imgs[-1]])
    det_results = []
    for p in dict.fromkeys(det_paths):
        print(f"  determinism: {p.name} (t0={CAND_T0}) ...", flush=True)
        det_results.append(determinism(p, device, CAND_T0))
    ref["determinism_check"] = None
    cand["determinism_check"] = {"t0": CAND_T0, "runs": det_results}

    out_ref = REPORTS_DIR / "validate_t0_030.json"
    out_cand = REPORTS_DIR / "validate_t0_025.json"
    out_ref.write_text(json.dumps(ref, indent=2), encoding="utf-8")
    out_cand.write_text(json.dumps(cand, indent=2), encoding="utf-8")
    print(f"\nSaved reference -> {out_ref}\nSaved candidate -> {out_cand}")

    R_tot = ref["totals"]["total_runtime_s"]
    C_tot = cand["totals"]["total_runtime_s"]
    print(f"\n{'=' * 70}\n AGGREGATE (mean over {len(images)} images)\n{'=' * 70}")
    n_cols = ("metric", "base(0.30)", "cand(0.25)", "delta")
    rows = []
    a_ref = aggregate(ref)["aggregate"]
    a_cand = aggregate(cand)["aggregate"]
    for k in sorted(a_ref):
        if k not in a_cand:
            continue
        rows.append((k, a_ref[k]["mean"], a_cand[k]["mean"],
                     a_cand[k]["mean"] - a_ref[k]["mean"]))
    print(f" {'metric':<26} {'base':>10} {'cand':>10} {'delta':>10}")
    for k, b, c, d in rows:
        flag = ""
        print(f" {k:<26} {b:>10.4f} {c:>10.4f} {d:>10.4f}{flag}")
    print(f"\n total_runtime_s  base={R_tot:.1f}  cand={C_tot:.1f}")

    print(f"\n{'=' * 70}\n DIRECT HARNESS COMPARISON (base=candidate? no: "
          f"candidate vs locked production reference)\n{'=' * 70}")
    comp = compare_reports(ref, cand, tolerances=DEFAULT_TOLERANCES)
    per_image = comp["rows"]

    print(f"\n{'=' * 70}\n WORST-CASE PER METRIC (candidate vs reference)\n{'=' * 70}")
    print(f" sim_psnr      : {cand['images'][np.argmin([r['similarity_vs_faithful']['psnr'] for r in cand['images']])]['image']}")
    w = (np.argmin([r["similarity_vs_faithful"]["psnr"] - b["similarity_vs_faithful"]["psnr"]
                    for r, b in zip(cand["images"], ref["images"])]))
    for mi in ("psnr", "ssim", "edge_align", "hist_sim"):
        name = mi
        deltas = [r["similarity_vs_faithful"][mi] - b["similarity_vs_faithful"][mi]
                  for r, b in zip(cand["images"], ref["images"])]
        i = int(np.argmin(deltas))
        print(f" {name:<14} min delta={deltas[i]:.4f}  @ {cand['images'][i]['image']}")
    print(f"\n{'=' * 70}\n DETERMINISM\n{'=' * 70}")
    for d in det_results:
        print(f" {d['image']:<28} rerun_identical={d['rerun_identical']} "
              f"max_abs_diff={d['max_abs_diff']}")

    aggpath = REPORTS_DIR / "validate_aggregate.json"
    aggpath.write_text(json.dumps({
        "reference": aggregate(ref), "candidate": aggregate(cand)}, indent=2),
        encoding="utf-8")
    print(f"\nSaved aggregate -> {aggpath}")
    return 0 if comp["overall_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())