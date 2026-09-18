"""DIAGNOSTIC ONLY -- captures every real intermediate stage of the actual
production pipeline (enhance.final_enhance, called via the real
backend.engine_adapter code path) for one real photo, with per-stage wall
time, so the artificial/"painted-watery" texture can be traced to an exact
stage.

Modifies NOTHING in image_enhancer/src/ or backend/: Restormer-only and
SwinIR-M-only outputs (not otherwise exposed by final_enhance's own
`return_stages` API) are captured by monkeypatching the two real functions
`_final_d1_weak` itself calls (`pro_exp.pipeline2.tiled_restore` and
`restore_exp.swinir_m.sr_bgr`) with a thin recording wrapper around the
UNMODIFIED original function -- same arguments in, same return value out,
nothing about the real computation changes. No parameter is read, changed,
or tuned anywhere in this script.

Run (needs the GPU-enabled root .venv):
    .venv\\Scripts\\python.exe research\\scripts\\diagnostic_stage_capture.py <input_image>
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "image_enhancer" / "src"))

OUT_DIR = REPO_ROOT / "research" / "outputs" / "diagnostic_stages"


def main() -> None:
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "research/inputs/7.jpeg"
    tag = input_path.stem
    out_dir = OUT_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    import enhance  # image_enhancer/src/enhance.py -- the real production module
    from pro_exp import pipeline2
    from restore_exp import swinir_m

    src = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if src is None:
        raise SystemExit(f"could not read {input_path}")
    h0, w0 = src.shape[:2]
    print(f"source: {input_path.name} {w0}x{h0}")
    cv2.imwrite(str(out_dir / "00_original.png"), src)

    target = enhance.aspect_preserving_target(w0, h0)
    print(f"aspect-preserving target: {target}")

    # --- Neutralize billboard regions exactly as backend/engine_adapter.py
    # does, so this diagnostic matches the real production call path.
    import os
    sentinel = REPO_ROOT / "backend" / ".workspace" / "__no_billboard_regions__.json"
    os.environ["REGIONS_CONFIG"] = str(sentinel)
    os.environ.pop("BILLBOARD_IMAGE", None)
    enhance._REGION_CONFIG = None

    captured: dict = {}
    timings: dict = {}

    real_tiled_restore = pipeline2.tiled_restore
    real_sr_bgr = swinir_m.sr_bgr

    def capturing_tiled_restore(*args, **kwargs):
        t0 = time.perf_counter()
        result = real_tiled_restore(*args, **kwargs)
        timings["restormer"] = time.perf_counter() - t0
        captured["restormer_only"] = result.copy()
        return result

    def capturing_sr_bgr(*args, **kwargs):
        t0 = time.perf_counter()
        result = real_sr_bgr(*args, **kwargs)
        timings["swinir_m"] = time.perf_counter() - t0
        captured["swinir_m_only"] = result.copy()
        return result

    pipeline2.tiled_restore = capturing_tiled_restore
    swinir_m.sr_bgr = capturing_sr_bgr
    try:
        t_total0 = time.perf_counter()
        out, stages = enhance.final_enhance(src, target=target, return_stages=True)
        total_time = time.perf_counter() - t_total0
    finally:
        pipeline2.tiled_restore = real_tiled_restore
        swinir_m.sr_bgr = real_sr_bgr

    # --- Save every real intermediate, in real pipeline order.
    # stages["tonal_correction"] is the META dict, not the image array (see
    # final_enhance's own `stages["tonal_correction"] = tonal_meta`) -- the
    # toned image itself isn't separately stashed, so it's re-derived below
    # (same function, same input, deterministic -- exactly what the real
    # run above already computed internally; does not affect that run).

    # tonal-corrected image: final_enhance doesn't stash the array itself in
    # `stages` (only its meta dict), so recompute it identically for
    # inspection -- same function, same input, deterministic, so this is
    # exactly what the real run above already used internally.
    import tonal_correction
    img_toned = tonal_correction.enhance_photographic_quality(src)
    cv2.imwrite(str(out_dir / "01_tonal_quality_preprocessing.png"), img_toned)

    if captured.get("restormer_only") is not None:
        cv2.imwrite(str(out_dir / "02_restormer_only.png"), captured["restormer_only"])
    if captured.get("swinir_m_only") is not None:
        cv2.imwrite(str(out_dir / "03_swinir_m_only.png"), captured["swinir_m_only"])
    cv2.imwrite(str(out_dir / "04_d1_weak_restormer_plus_swinir_resized.png"), stages["d1_weak"])
    cv2.imwrite(str(out_dir / "05_f3_natural_fusion.png"), stages["f3"])
    cv2.imwrite(str(out_dir / "06_g7_ms_multiscale_detail.png"), stages["f3_ms"])
    cv2.imwrite(str(out_dir / "07_a_plus_whole_frame_detail_FINAL.png"), stages["f3_plus"])
    cv2.imwrite(str(out_dir / "08_final_output.png"), out)

    print("\n== Stage shapes and timings ==")
    print(f"00_original                         : {src.shape[1]}x{src.shape[0]}")
    print(f"01_tonal_quality_preprocessing       : {img_toned.shape[1]}x{img_toned.shape[0]}")
    if captured.get("restormer_only") is not None:
        r = captured["restormer_only"]
        print(f"02_restormer_only                    : {r.shape[1]}x{r.shape[0]}  "
              f"time={timings.get('restormer', 0):.2f}s")
    if captured.get("swinir_m_only") is not None:
        s = captured["swinir_m_only"]
        print(f"03_swinir_m_only                     : {s.shape[1]}x{s.shape[0]}  "
              f"time={timings.get('swinir_m', 0):.2f}s")
    d1 = stages["d1_weak"]
    print(f"04_d1_weak (restormer+swinir, resized): {d1.shape[1]}x{d1.shape[0]}")
    f3 = stages["f3"]
    print(f"05_f3_natural_fusion                 : {f3.shape[1]}x{f3.shape[0]}")
    f3ms = stages["f3_ms"]
    print(f"06_g7_ms_multiscale_detail            : {f3ms.shape[1]}x{f3ms.shape[0]}")
    f3plus = stages["f3_plus"]
    print(f"07_a_plus_whole_frame_detail (FINAL)  : {f3plus.shape[1]}x{f3plus.shape[0]}")
    print(f"\nTOTAL pipeline wall time: {total_time:.2f}s")
    print(f"peak_vram_mb: {stages.get('peak_vram_mb')}")
    print(f"\nAll stages saved to: {out_dir}")


if __name__ == "__main__":
    main()
