"""MVP 2 Phase 1/7 -- concrete repro of the current stretch bug.

Takes a real photo, makes a non-16:9 (square) crop of it, and runs it
through the EXACT unmodified production call path
(engine.enhance("final", img) via backend.engine_adapter's own import
mechanism) with no target override -- i.e. precisely what the shipping
MVP 1 app does today. Confirms the output is hard-stretched to 3840x2160
regardless of the input's own aspect ratio.

Does not modify any production file. Writes into research/outputs/ only.

Run (needs the GPU-enabled root .venv):
    .venv\\Scripts\\python.exe research\\scripts\\repro_stretch_bug.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "image_enhancer" / "src"))
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = REPO_ROOT / "research" / "outputs" / "phase1_stretch_repro"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE = REPO_ROOT / "comparison_original.png"


def main() -> None:
    import enhance  # image_enhancer/src/enhance.py, same flat-module import engine_adapter uses

    src = cv2.imread(str(SOURCE), cv2.IMREAD_COLOR)
    if src is None:
        raise SystemExit(f"could not read {SOURCE}")
    h0, w0 = src.shape[:2]
    print(f"source: {SOURCE.name} {w0}x{h0} (aspect {w0/h0:.4f})")

    # Center-crop to a SQUARE (1:1) -- a case explicitly called out by the
    # review as one that must NOT become 3840x2160.
    side = min(w0, h0)
    x0 = (w0 - side) // 2
    y0 = (h0 - side) // 2
    square = src[y0:y0 + side, x0:x0 + side].copy()
    sq_path = OUT_DIR / "input_square_crop.png"
    cv2.imwrite(str(sq_path), square)
    print(f"square crop: {square.shape[1]}x{square.shape[0]} (aspect "
          f"{square.shape[1]/square.shape[0]:.4f}) -> {sq_path}")

    # Neutralize billboard regions exactly as backend/engine_adapter.py does,
    # so this repro matches production behavior precisely (no accidental
    # billboard-box reconstruction firing on an unrelated region).
    import os
    sentinel = REPO_ROOT / "backend" / ".workspace" / "__no_billboard_regions__.json"
    os.environ["REGIONS_CONFIG"] = str(sentinel)
    os.environ.pop("BILLBOARD_IMAGE", None)
    enhance._REGION_CONFIG = None

    print("running engine.enhance('final', img) -- exactly the production call path, "
          "no target override (this IS the current MVP 1 behavior) ...")
    t0 = time.perf_counter()
    out, dt = enhance.enhance("final", square)
    elapsed = time.perf_counter() - t0
    oh, ow = out.shape[:2]
    print(f"output: {ow}x{oh} (aspect {ow/oh:.4f})  engine dt={dt:.2f}s  wall={elapsed:.2f}s")

    out_path = OUT_DIR / "output_current_mvp1_stretched.png"
    cv2.imwrite(str(out_path), out)
    print(f"saved -> {out_path}")

    stretched = (ow, oh) == (3840, 2160) and abs(square.shape[1] / square.shape[0] - ow / oh) > 0.01
    print(f"\nCONFIRMED STRETCH BUG: {stretched}  "
          f"(a 1:1 input became {ow}x{oh}, a 16:9 box, instead of preserving its own aspect ratio)")


if __name__ == "__main__":
    main()
