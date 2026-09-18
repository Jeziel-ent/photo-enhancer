"""MVP 2 Phase 2/18 -- proves the aspect-ratio fix works through the REAL,
FIXED production call path (backend.engine_adapter.enhance_image), not just
the underlying enhance.py formula. Companion to repro_stretch_bug.py, which
demonstrates the ORIGINAL bug via the raw engine module call (still
byte-for-byte how a caller that never passes `target` behaves, by design --
see docs/MVP2_PIPELINE.md's "default parameter values are unchanged" note).

Run:
    .venv\\Scripts\\python.exe research\\scripts\\verify_stretch_fix.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "image_enhancer" / "src"))

OUT_DIR = REPO_ROOT / "research" / "outputs" / "phase2_stretch_fix_verified"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE = REPO_ROOT / "comparison_original.png"


def main() -> None:
    from backend import engine_adapter

    src = cv2.imread(str(SOURCE), cv2.IMREAD_COLOR)
    h0, w0 = src.shape[:2]
    print(f"source: {SOURCE.name} {w0}x{h0} (aspect {w0/h0:.4f})")

    input_path = OUT_DIR / "input_square.png"
    output_path = OUT_DIR / "output_fixed_aspect_preserved.png"
    cv2.imwrite(str(input_path), src)

    print("running backend.engine_adapter.enhance_image() -- the REAL, FIXED "
          "production call path (what the shipping app now actually calls) ...")
    t0 = time.perf_counter()
    engine_adapter.enhance_image(input_path, output_path)
    elapsed = time.perf_counter() - t0

    out = cv2.imread(str(output_path))
    oh, ow = out.shape[:2]
    print(f"output: {ow}x{oh} (aspect {ow/oh:.4f})  wall={elapsed:.2f}s")

    preserved = abs(w0 / h0 - ow / oh) < 0.01 and max(ow, oh) == 3840
    print(f"\nASPECT RATIO PRESERVED (no stretch): {preserved}  "
          f"(1:1 input stayed 1:1 at {ow}x{oh}, NOT forced to 3840x2160)")


if __name__ == "__main__":
    main()
