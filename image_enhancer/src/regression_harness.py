"""Standing regression/metrics harness for the production enhancement engine.

Read-only tool: imports and calls `enhance.py` exactly as it already exists
(same as every other script in this directory) and never modifies it, the
engine's other modules, or any source image. Lives alongside `benchmark.py`
/ `compare_report.py` (the existing category of R&D tooling in this repo),
not inside the fast unit-test suite.

What it does:
  - Runs the current production `final` pipeline (or any other method in
    `enhance.METHODS`) on a fixed set of real images.
  - Neutralizes the engine's billboard-regions config first (see
    `neutralize_billboard_regions` below) so measurements reflect the exact
    behavior real uploads get through `backend/engine_adapter.py` — zero
    billboard boxes, full-frame path only. This module does NOT import
    anything from `backend/`; the few lines that do this are a intentional,
    self-contained re-implementation so the engine stays independent of the
    API layer (see docs/ARCHITECTURE.md's layering).
  - Measures, per image, everything quality.py already provides (PSNR,
    SSIM, NRMSE, hist_sim, edge_align, sharpness, noise, contrast,
    edge_density) plus the two metrics quality.py does NOT provide and this
    task asked for explicitly: mean pixel difference and percent of pixels
    changed (both vs. the faithful Lanczos baseline, matching this
    project's own established "vs. faithful" methodology).
  - Records runtime and peak/current CUDA VRAM per image (0/None on CPU).
  - Runs one determinism check (rerun one image, compare byte-for-byte).
  - Saves a machine-readable JSON report and every enhanced output PNG
    under image_enhancer/reports/regression_baseline/ (gitignored, exactly
    like every other experiment's reports/ directory in this repo).
  - Can compare a fresh run's JSON against a previously locked baseline
    JSON and print a per-image, per-metric delta table with pass/fail
    tolerances, for evaluating a future experiment without re-deriving the
    baseline.

Usage (GPU required; running any action below is the explicit opt-in this
task asked for — invoking the script with no action flag does nothing):

    # from image_enhancer/, using its own venv:
    .venv\\Scripts\\python.exe src\\regression_harness.py --lock-baseline
    .venv\\Scripts\\python.exe src\\regression_harness.py --run-only --method final
    .venv\\Scripts\\python.exe src\\regression_harness.py --compare reports\\regression_baseline\\baseline_final.json reports\\regression_baseline\\candidate.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent  # image_enhancer/
REPO_ROOT = ROOT.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import enhance  # noqa: E402
import quality  # noqa: E402

ORIGINALS_DIR = ROOT / "originals"
REPORTS_DIR = ROOT / "reports" / "regression_baseline"
OUTPUTS_DIR = REPORTS_DIR / "outputs"

# Real Adinn client photos from prior real-world testing of this engine
# (gitignored, disposable fixtures under backend/.deckstore/*/uploads/) —
# included opportunistically for broader real-world coverage beyond the six
# canonical originals, but never required: the harness must still work if
# this directory has been cleaned up. NOTE: most files under .deckstore are
# byte-identical copies of originals/1.jpeg (backend's own test suite seeds
# every fake "client upload" from that one sample file) — verified by
# content hash. mumbai-01.jpeg is the only genuinely distinct real upload
# found on disk when this was checked; discover_benchmark_images() also
# de-duplicates by content hash as a defense against this recurring.
_DECKSTORE_EXTRAS = [
    "testclient-autumn-launch/uploads/mumbai-01.jpeg",
]

# Percent-changed threshold: a per-pixel (any channel) absolute BGR
# difference above this counts as "changed". 2/255 is small enough to
# catch real enhancement while ignoring 8-bit LAB round-trip noise (every
# stage in enhance.py round-trips through uint8 LAB at least once).
PIXEL_CHANGED_THRESHOLD = 2.0

DEFAULT_TOLERANCES = {
    # metric: (direction, max_allowed_regression)
    # direction "higher_better" fails if candidate < baseline - tol;
    # "lower_better" fails if candidate > baseline + tol.
    "similarity_vs_faithful.psnr": ("higher_better", 1.0),
    "similarity_vs_faithful.ssim": ("higher_better", 0.01),
    "similarity_vs_faithful.edge_align": ("higher_better", 0.02),
    "similarity_vs_faithful.hist_sim": ("higher_better", 0.02),
    "no_reference.sharpness": ("higher_better", None),  # informational only
    "no_reference.noise": ("lower_better", 1.0),
    "pct_pixels_changed": ("informational", None),
    "runtime_s": ("lower_better", None),  # informational (hardware-dependent)
}


# --------------------------------------------------------------- neutralize

@contextmanager
def neutralize_billboard_regions():
    """Forces enhance._billboard_boxes() to return [] for the duration of
    this block — the exact configuration every real upload gets via
    backend/engine_adapter.py's own neutralization (re-implemented here,
    not imported, so image_enhancer/ has no dependency on backend/). See
    docs/ENGINE_AUDIT.md §1 for why this matters: the shipped
    image_enhancer/regions.json has real boxes for these exact six
    originals, and its same-size fallback could otherwise leak a
    billboard-box reconstruction onto what should be a generic full-frame
    measurement."""
    sentinel = REPORTS_DIR / "__no_billboard_regions__.json"  # never created
    prev_regions = os.environ.get("REGIONS_CONFIG")
    prev_billboard = os.environ.get("BILLBOARD_IMAGE")
    os.environ["REGIONS_CONFIG"] = str(sentinel)
    os.environ.pop("BILLBOARD_IMAGE", None)
    enhance._REGION_CONFIG = None
    try:
        yield
    finally:
        if prev_regions is None:
            os.environ.pop("REGIONS_CONFIG", None)
        else:
            os.environ["REGIONS_CONFIG"] = prev_regions
        if prev_billboard is None:
            os.environ.pop("BILLBOARD_IMAGE", None)
        else:
            os.environ["BILLBOARD_IMAGE"] = prev_billboard
        enhance._REGION_CONFIG = None


# ---------------------------------------------------------------- discovery

def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def discover_benchmark_images() -> list[Path]:
    """The six canonical reference originals (always required, committed to
    the repo) plus any real Adinn client photos still present under
    backend/.deckstore (best-effort, skipped if absent). De-duplicates by
    file content hash: backend/.deckstore accumulates whatever its own test
    suite last wrote under each deck id, which has repeatedly turned out to
    be copies of originals/1.jpeg rather than distinct photos — a
    duplicate would silently inflate "image_count" without adding any real
    coverage, so it's dropped rather than counted twice."""
    images = sorted(ORIGINALS_DIR.glob("*.jpeg"))
    if not images:
        raise FileNotFoundError(f"no reference images found under {ORIGINALS_DIR}")
    seen_hashes = {_file_sha256(p) for p in images}
    deckstore_root = REPO_ROOT / "backend" / ".deckstore"
    for rel in _DECKSTORE_EXTRAS:
        candidate = deckstore_root / rel
        if not candidate.is_file():
            continue
        h = _file_sha256(candidate)
        if h in seen_hashes:
            print(f"  skipping {candidate} — duplicate content of an already-included image")
            continue
        seen_hashes.add(h)
        images.append(candidate)
    return images


# ----------------------------------------------------------------- metrics

def sha256_of_array(img: np.ndarray) -> str:
    return hashlib.sha256(img.tobytes()).hexdigest()


def pixel_diff_metrics(ref: np.ndarray, pred: np.ndarray) -> dict:
    """Metrics quality.py does not already provide: mean absolute per-pixel
    BGR difference, and the percent of pixels with any channel changed by
    more than PIXEL_CHANGED_THRESHOLD. Both vs. the faithful baseline,
    matching every other metric in this harness."""
    diff = np.abs(ref.astype(np.float32) - pred.astype(np.float32))
    mean_pixel_diff = float(diff.mean())
    changed = (diff.max(axis=2) > PIXEL_CHANGED_THRESHOLD)
    pct_pixels_changed = float(changed.mean() * 100.0)
    return {"mean_pixel_diff": mean_pixel_diff, "pct_pixels_changed": pct_pixels_changed}


def _vram_reset(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def _vram_read(device: torch.device) -> dict:
    if device.type != "cuda":
        return {"peak_vram_mb": None, "current_vram_mb": None}
    peak = torch.cuda.max_memory_allocated() / 1048576.0
    current = torch.cuda.memory_allocated() / 1048576.0
    torch.cuda.empty_cache()
    return {"peak_vram_mb": round(float(peak), 1), "current_vram_mb": round(float(current), 1)}


# ------------------------------------------------------------------- runner

def run_one_image(path: Path, method: str, device: torch.device) -> tuple[dict, np.ndarray]:
    img = enhance.load_image(str(path))
    faithful = enhance.simple_upscale(img)

    with neutralize_billboard_regions():
        _vram_reset(device)
        t0 = time.perf_counter()
        out, runtime_reported_s = enhance.enhance(method, img)
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
        "method": method,
        "device": str(device),
        "runtime_s": round(runtime_s, 2),
        **vram,
        "similarity_vs_faithful": {k: round(v, 4) for k, v in sim.items()},
        "no_reference": {k: round(v, 4) for k, v in nr_out.items()},
        "no_reference_faithful": {k: round(v, 4) for k, v in nr_faithful.items()},
        "sharpness_gain_over_faithful": round(
            nr_out["sharpness"] / max(nr_faithful["sharpness"], 1e-6), 3),
        **{k: round(v, 4) for k, v in pdiff.items()},
        "output_sha256": sha256_of_array(out),
    }
    return record, out


def run_determinism_check(path: Path, method: str, device: torch.device) -> dict:
    """Reruns one image and checks the output is byte-for-byte identical.
    Cheap, single-image check — determinism for this exact pipeline is
    already exhaustively covered by image_enhancer/tests/test_final_pipeline.py
    and test_g91_integration_pipeline.py (all 6 originals, both assert and
    pass 'deterministic'); this is a lightweight confirmation tied to
    THIS harness's own baseline run, not a replacement for those suites."""
    img = enhance.load_image(str(path))
    with neutralize_billboard_regions():
        out1, _ = enhance.enhance(method, img)
    with neutralize_billboard_regions():
        out2, _ = enhance.enhance(method, img)
    identical = bool(np.array_equal(out1, out2))
    max_abs_diff = 0.0 if identical else float(
        np.abs(out1.astype(np.int16) - out2.astype(np.int16)).max())
    return {"image": path.name, "rerun_identical": identical, "max_abs_diff": max_abs_diff}


def build_report(
    method: str,
    image_paths: list[Path] | None = None,
    save_outputs: bool = True,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = image_paths or discover_benchmark_images()

    if save_outputs:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    total_runtime = 0.0
    for path in images:
        print(f"  running {method!r} on {path.name} ...", flush=True)
        t_img0 = time.perf_counter()
        record, out = run_one_image(path, method, device)
        total_runtime += time.perf_counter() - t_img0
        if save_outputs:
            out_path = OUTPUTS_DIR / f"{method}_{Path(path.name).stem}.png"
            enhance.save_image(out, str(out_path))
            record["output_path"] = str(out_path.relative_to(REPO_ROOT))
        records.append(record)
        print(f"    runtime={record['runtime_s']}s peak_vram_mb={record['peak_vram_mb']} "
              f"psnr={record['similarity_vs_faithful']['psnr']} "
              f"ssim={record['similarity_vs_faithful']['ssim']} "
              f"pct_pixels_changed={record['pct_pixels_changed']}%", flush=True)

    print(f"  determinism check on {images[0].name} ...", flush=True)
    determinism = run_determinism_check(images[0], method, device)
    print(f"    rerun_identical={determinism['rerun_identical']}", flush=True)

    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "device": str(device),
        "pixel_changed_threshold": PIXEL_CHANGED_THRESHOLD,
        "images": records,
        "determinism_check": determinism,
        "totals": {
            "image_count": len(records),
            "total_runtime_s": round(total_runtime, 1),
        },
    }


def save_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- comparison

def _get_path(d: dict, dotted: str):
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def compare_reports(baseline: dict, candidate: dict, tolerances: dict = DEFAULT_TOLERANCES) -> dict:
    """Per-image, per-metric comparison of a candidate run against a locked
    baseline. Returns a dict with per-image rows and an overall pass/fail —
    used to evaluate a future experiment (e.g. a tuned parameter) without
    re-deriving what "acceptable" means each time."""
    base_by_name = {r["image"]: r for r in baseline["images"]}
    rows = []
    overall_ok = True
    for cand_rec in candidate["images"]:
        name = cand_rec["image"]
        base_rec = base_by_name.get(name)
        if base_rec is None:
            rows.append({"image": name, "status": "NO_BASELINE"})
            continue
        row = {"image": name, "metrics": {}}
        for metric, (direction, tol) in tolerances.items():
            b = _get_path(base_rec, metric)
            c = _get_path(cand_rec, metric)
            if b is None or c is None:
                continue
            delta = c - b
            ok = True
            if direction == "higher_better" and tol is not None:
                ok = delta >= -tol
            elif direction == "lower_better" and tol is not None:
                ok = delta <= tol
            row["metrics"][metric] = {
                "baseline": b, "candidate": c, "delta": round(delta, 4), "ok": ok,
            }
            if not ok:
                overall_ok = False
        rows.append(row)
    return {"overall_ok": overall_ok, "rows": rows}


def print_comparison_table(comparison: dict) -> None:
    for row in comparison["rows"]:
        print(f"\n{row['image']}:")
        if "metrics" not in row:
            print(f"  {row.get('status', 'unknown')}")
            continue
        for metric, v in row["metrics"].items():
            flag = "OK" if v["ok"] else "REGRESSION"
            print(f"  {metric:38s} baseline={v['baseline']:<10} candidate={v['candidate']:<10} "
                  f"delta={v['delta']:<10} [{flag}]")
    print(f"\nOVERALL: {'PASS' if comparison['overall_ok'] else 'FAIL (regression detected)'}")


# ---------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Standing regression/metrics harness for the production "
                    "enhancement engine. GPU required. Running any action "
                    "below is the explicit opt-in — no action runs by default.")
    parser.add_argument("--method", default="final", choices=list(enhance.METHODS.keys()))
    parser.add_argument("--lock-baseline", action="store_true",
                        help="run the harness and save the result as the locked baseline "
                             "(reports/regression_baseline/baseline_<method>.json)")
    parser.add_argument("--run-only", action="store_true",
                        help="run the harness and print results without saving a baseline")
    parser.add_argument("--out", type=Path, default=None,
                        help="output path for --lock-baseline or --run-only (default: "
                             "reports/regression_baseline/{baseline,run}_<method>.json)")
    parser.add_argument("--compare", nargs=2, metavar=("BASELINE_JSON", "CANDIDATE_JSON"),
                        help="compare an existing candidate report against a locked baseline")
    parser.add_argument("--no-save-outputs", action="store_true",
                        help="skip writing enhanced PNGs (still writes the JSON report)")
    args = parser.parse_args(argv)

    if args.compare:
        baseline = load_report(Path(args.compare[0]))
        candidate = load_report(Path(args.compare[1]))
        comparison = compare_reports(baseline, candidate)
        print_comparison_table(comparison)
        return 0 if comparison["overall_ok"] else 1

    if not args.lock_baseline and not args.run_only:
        parser.print_help()
        print("\nNo action flag given (--lock-baseline / --run-only / --compare) — "
              "nothing was run. This is intentional: GPU execution requires explicit opt-in.")
        return 0

    report = build_report(args.method, save_outputs=not args.no_save_outputs)

    if args.lock_baseline:
        out_path = args.out or (REPORTS_DIR / f"baseline_{args.method}.json")
    else:
        out_path = args.out or (REPORTS_DIR / f"run_{args.method}.json")
    save_report(report, out_path)
    print(f"\nSaved {'baseline' if args.lock_baseline else 'run'} report to {out_path}")
    print(f"Images: {report['totals']['image_count']}, "
          f"total runtime: {report['totals']['total_runtime_s']}s, "
          f"determinism: {report['determinism_check']['rerun_identical']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
