"""G8-fix verification: compare the OLD (git HEAD, fixed-4x-then-downsample)
Candidate A billboard path against the NEW (this-branch, single-resize-to-
true-scale) path implemented in phase1_enhancement/src/enhance.py.

Does not modify enhance.py (already fixed on disk) or any other production
file. Loads the git-HEAD copy of enhance.py (saved as
reports/g8_fix_exp/enhance_old_HEAD.py by run_compare's caller / this file's
own git-show step) under a separate module name ("enhance_old") purely for
this comparison; the on-disk enhance.py used by the rest of the app is
untouched.

Usage:
  .venv/Scripts/python src/g8_fix_exp/run_compare.py
"""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import enhance as enhance_new  # noqa: E402  (this-branch, fixed)
from quality import compute_no_reference  # noqa: E402

OLD_SRC = ROOT / "reports" / "g8_fix_exp" / "enhance_old_HEAD.py"
OUT_DIR = ROOT / "reports" / "g8_fix_exp"
CROPS_DIR = OUT_DIR / "crops"
FULL_DIR = OUT_DIR / "full"
CROPS_DIR.mkdir(parents=True, exist_ok=True)
FULL_DIR.mkdir(parents=True, exist_ok=True)

BENCH_IMG = ROOT.parent / "backend" / ".deckstore" / \
    "dreamland-ventures-chennai-sivagangai-ooh-proposal" / "uploads" / \
    "dreamland-ventures-chennai-01.jpeg"
BENCH_BOXES = [("jj_gold", (543, 131, 247, 171)), ("grt", (502, 316, 192, 110))]

ORIGINALS_DIR = ROOT / "originals"
REGIONS_JSON = ROOT / "regions.json"


def load_old_module():
    spec = importlib.util.spec_from_file_location("enhance_old", str(OLD_SRC))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["enhance_old"] = mod
    spec.loader.exec_module(mod)
    return mod


def native_to_4k(box, w0, out_w):
    s = out_w / w0
    x, y, w, h = box
    return (int(round(x * s)), int(round(y * s)),
            int(round(w * s)), int(round(h * s)))


def crop(img, box):
    x, y, w, h = box
    x = max(0, min(x, img.shape[1] - 1))
    y = max(0, min(y, img.shape[0] - 1))
    w = min(w, img.shape[1] - x)
    h = min(h, img.shape[0] - y)
    return img[y:y + h, x:x + w]


def _boxes_close(b1, b2, tol=8):
    return all(abs(a - c) <= tol for a, c in zip(b1, b2))


def run_instrumented(mod, img):
    """Run mod.final_enhance once, capturing each pre-composite billboard
    crop via a local monkey-patch of mod._final_composite_box_bgr (restored
    immediately after). Works identically against the old or new module."""
    captured = []
    orig_composite = mod._final_composite_box_bgr

    def capture_and_composite(frame_bgr, board_bgr, box4):
        captured.append((board_bgr.copy(), box4))
        return orig_composite(frame_bgr, board_bgr, box4)

    mod._final_composite_box_bgr = capture_and_composite
    try:
        t0 = time.time()
        out, stages = mod.final_enhance(img, return_stages=True)
        elapsed = time.time() - t0
    finally:
        mod._final_composite_box_bgr = orig_composite
    return out, stages, captured, elapsed


def metrics_for_crop(img):
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    sobel_mean = float(cv2.magnitude(gx, gy).mean())
    nr = compute_no_reference(img)
    return {
        "lap_var": round(lap_var, 2),
        "sobel_mean": round(sobel_mean, 2),
        "nr_sharpness": round(nr["sharpness"], 2),
        "noise": round(nr["noise"], 3),
        "w": int(img.shape[1]), "h": int(img.shape[0]),
    }


def process_image(mod, tag, img, boxes, env_setup, w0, out_w):
    """boxes: list of (name, native_box). env_setup(): sets BILLBOARD_IMAGE /
    REGIONS_CONFIG and clears the module's cached region config."""
    env_setup(mod)
    out, stages, captured, elapsed = run_instrumented(mod, img)
    boxes4 = stages.get("boxes4", [])
    result = {"elapsed_s": round(elapsed, 2),
              "peak_vram_mb": stages.get("peak_vram_mb"),
              "boxes": {}}
    for name, native_box in boxes:
        box4 = native_to_4k(native_box, w0, out_w)
        billboard_crop_img = None
        for board_bgr, cbox4 in captured:
            if _boxes_close(cbox4, box4):
                billboard_crop_img = board_bgr
                break
        final_crop = crop(out, box4)
        cv2.imwrite(str(CROPS_DIR / f"{tag}_{name}_{mod.__name__}_precomposite.png"),
                    billboard_crop_img if billboard_crop_img is not None
                    else np.zeros((8, 8, 3), np.uint8))
        cv2.imwrite(str(CROPS_DIR / f"{tag}_{name}_{mod.__name__}_final.png"),
                    final_crop)
        result["boxes"][name] = {
            "box4": list(box4),
            "precomposite_size": None if billboard_crop_img is None else
                [int(billboard_crop_img.shape[1]), int(billboard_crop_img.shape[0])],
            "precomposite_metrics": metrics_for_crop(billboard_crop_img),
            "final_metrics": metrics_for_crop(final_crop),
        }
    cv2.imwrite(str(FULL_DIR / f"{tag}_{mod.__name__}_full.png"), out)
    return result, out


def bench_env_setup(mod):
    img = cv2.imread(str(BENCH_IMG), cv2.IMREAD_COLOR)
    H, W = img.shape[:2]
    regions_path = OUT_DIR / "bench_two_board_regions.json"
    name = "two_board.jpeg"
    regions_path.write_text(json.dumps(
        {name: {"boxes": [list(b) for _, b in BENCH_BOXES],
                "image_size": [W, H]}}))
    os.environ["REGIONS_CONFIG"] = str(regions_path)
    os.environ["BILLBOARD_IMAGE"] = name
    mod._REGION_CONFIG = None


def make_original_env_setup(fname):
    def _setup(mod):
        os.environ["REGIONS_CONFIG"] = str(REGIONS_JSON)
        os.environ["BILLBOARD_IMAGE"] = fname
        mod._REGION_CONFIG = None
    return _setup


def determinism_check(mod, img, env_setup):
    env_setup(mod)
    out1, _, _, _ = run_instrumented(mod, img)
    env_setup(mod)
    out2, _, _, _ = run_instrumented(mod, img)
    if out1.shape != out2.shape:
        return {"identical": False, "reason": "shape mismatch"}
    diff = cv2.absdiff(out1, out2)
    max_diff = int(diff.max())
    nonzero = int(np.count_nonzero(diff))
    return {"identical": max_diff == 0, "max_abs_diff": max_diff,
            "nonzero_px_channels": nonzero}


def main():
    old = load_old_module()
    new = enhance_new

    all_results = {}

    # --- Benchmark (JJ GOLD + GRT) ---
    bench_img = cv2.imread(str(BENCH_IMG), cv2.IMREAD_COLOR)
    w0 = bench_img.shape[1]
    print("== Benchmark: dreamland-ventures-chennai-01.jpeg ==")
    old_res, _ = process_image(old, "bench", bench_img, BENCH_BOXES,
                                bench_env_setup, w0, old.OUT_W)
    new_res, _ = process_image(new, "bench", bench_img, BENCH_BOXES,
                                bench_env_setup, w0, new.OUT_W)
    print(f"  old: {old_res['elapsed_s']}s peak_vram={old_res['peak_vram_mb']}MB")
    print(f"  new: {new_res['elapsed_s']}s peak_vram={new_res['peak_vram_mb']}MB")
    all_results["bench"] = {"old": old_res, "new": new_res}

    det = determinism_check(new, bench_img, bench_env_setup)
    print(f"  determinism (new, 2 runs): {det}")
    all_results["bench"]["determinism_new"] = det

    # --- 6 reference originals ---
    regions = json.loads(REGIONS_JSON.read_text())
    for fname, spec in regions.items():
        tag = Path(fname).stem
        img_path = ORIGINALS_DIR / fname
        if not img_path.exists():
            print(f"skip {fname}: not found at {img_path}")
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        w0 = img.shape[1]
        box = tuple(spec["boxes"][0])
        boxes = [("board", box)]
        env_setup = make_original_env_setup(fname)
        print(f"== {fname} ==")
        old_res, _ = process_image(old, tag, img, boxes, env_setup, w0, old.OUT_W)
        new_res, _ = process_image(new, tag, img, boxes, env_setup, w0, new.OUT_W)
        print(f"  old: {old_res['elapsed_s']}s peak_vram={old_res['peak_vram_mb']}MB "
              f"precomposite_size={old_res['boxes']['board']['precomposite_size']}")
        print(f"  new: {new_res['elapsed_s']}s peak_vram={new_res['peak_vram_mb']}MB "
              f"precomposite_size={new_res['boxes']['board']['precomposite_size']}")
        all_results[tag] = {"old": old_res, "new": new_res}

    out_path = OUT_DIR / "compare_results.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
