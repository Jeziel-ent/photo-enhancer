"""Category-level inspection: baseline vs candidate per content-type ROI.

Computes per-ROI PSNR/SSIM/mean-pixel-diff comparing faithful vs baseline
and faithful vs candidate at three content categories automatically:
  1. signs/text + buildings  -> regions.json billboard boxes (scaled to 4K)
  2. faces                   -> YuNet detections (scaled to 4K)
  3. sky/flat (top half)     -> lowest-variance 96px tiles
  4. road/surface (bottom)   -> lowest-variance 96px tiles
  5. fine structures         -> pipeline2.detail_window crop
  6. high-texture            -> highest-Laplacian-variance 96px tiles
"""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import enhance  # noqa: E402
import quality  # noqa: E402
from f3_local_exp.recipe import f3_self_env_tight, f3_self_env_capped  # noqa: E402
from f3_local_exp.recipe import f3_local_contrast_env  # noqa: E402


def _variance_tiles(gray, tile=96, top_half=False, max_keep=3):
    H, W = gray.shape
    if top_half:
        ys = range(0, H // 2, tile)
    else:
        ys = range(H // 2, H, tile)
    scored = []
    for y in ys:
        for x in range(0, W, tile):
            patch = gray[y:min(y + tile, H), x:min(x + tile, W)]
            if patch.size < 400:
                continue
            scored.append((patch.std(), x, y, min(tile, W - x), min(tile, H - y)))
    scored.sort()
    return scored[:max_keep]


def roi_psnr_ssim(ref, pred, x, y, w, h):
    r = ref[y:y + h, x:x + w]
    p = pred[y:y + h, x:x + w]
    if r.size < 64 or r.shape != p.shape:
        return None
    return {"psnr": round(float(quality.compute_psnr(r, p)), 2),
            "ssim": round(float(quality.compute_ssim(r, p)), 4),
            "pd": round(float(np.abs(r.astype(np.float32) - p.astype(np.float32)).mean()), 3)}


def main():
    from regression_harness import discover_benchmark_images
    import json
    import restore_exp
    from restore_exp import faces as faces_mod
    from pro_exp import pipeline2

    regions = json.loads((ROOT / "regions.json").read_text(encoding="utf-8"))

    all_images = discover_benchmark_images()
    for path in all_images:
        stem = path.stem
        img = enhance.load_image(str(path))
        faithful = enhance.simple_upscale(img)
        d1, _, _ = enhance._final_d1_weak(img)

        # --- produce three outputs (cheap once D1 is cached) ---
        # baseline
        base_f3 = enhance._final_f3_natural(faithful, d1)
        base_ms  = enhance._final_multiscale_detail(base_f3)
        base_out = enhance._final_whole_frame_detail(base_ms)

        # self_env_k3
        k3_f3 = f3_self_env_tight(faithful, d1, ksize=3)
        k3_ms  = enhance._final_multiscale_detail(k3_f3)
        k3_out = enhance._final_whole_frame_detail(k3_ms)

        # self_env_k5 + cap 10
        cap_f3 = f3_self_env_capped(faithful, d1, ksize=5, global_cap=10.0)
        cap_ms = enhance._final_multiscale_detail(cap_f3)
        cap_out = enhance._final_whole_frame_detail(cap_ms)

        variants = {"baseline": base_out, "self_env_k3": k3_out,
                    "self_env_k5_cap10": cap_out}

        orig = cv2.imread(str(path), cv2.IMREAD_COLOR)
        H, W = orig.shape[:2]
        s = enhance.OUT_W / W

        # --- detect ROIs ---
        cats = {}
        # signs/text/buildings
        for bi, box in enumerate(regions.get(path.name, {}).get("boxes", [])):
            cats[f"signs_bldg_{bi}"] = tuple(int(round(v * s)) for v in box)

        # faces
        dets = faces_mod.detect_faces(orig)
        for fi, d in enumerate(dets[:3]):
            cats[f"face_{fi}"] = tuple(
                int(round(v * s)) for v in (d["x"], d["y"], d["w"], d["h"]))

        # sky/flat and road/surface
        g = cv2.cvtColor(faithful, cv2.COLOR_BGR2GRAY)
        for si, (_, x, y, w, h) in enumerate(
                _variance_tiles(g, tile=96, top_half=True, max_keep=3)):
            cats[f"sky_flat_{si}"] = (x, y, w, h)
        for si, (_, x, y, w, h) in enumerate(
                _variance_tiles(g, tile=96, top_half=False, max_keep=3)):
            cats[f"road_{si}"] = (x, y, w, h)

        # fine structures
        ex_boxes = regions.get(path.name, {}).get("boxes", [])
        ex_faces = [(d["x"], d["y"], d["w"], d["h"]) for d in dets[:2]]
        try:
            dw = pipeline2.detail_window(
                orig, ex_boxes + ex_faces, win=160)
            cats["fine_struct"] = tuple(int(round(v * s)) for v in dw)
        except Exception:
            pass

        # high-texture (buildings/vehicles/vegetation)
        for ti, (_, x, y, w, h) in enumerate(
                _variance_tiles(g, tile=96, top_half=False, max_keep=2)):
            cats[f"hi_tex_{ti}"] = (x, y, w, h)

        # --- measure and print ---
        print(f"\n{'=' * 55}\n  {path.name}\n{'=' * 55}")
        hdr = f"  {'roi':<18}{'baseline':>24}{'k3':>24}{'cap10':>24}"
        print(hdr)
        for cat_name, rect in sorted(cats.items()):
            x, y, w, h = rect
            r = roi_psnr_ssim(faithful, base_out, x, y, w, h)
            k = roi_psnr_ssim(faithful, k3_out, x, y, w, h)
            c = roi_psnr_ssim(faithful, cap_out, x, y, w, h)
            if r is None:
                continue
            def fmt(d):
                if d is None:
                    return " "*24
                return f"P={d['psnr']:>5.1f} S={d['ssim']:.4f} pd={d['pd']:.2f}"
            r_k3 = roi_psnr_ssim(faithful, k3_out, x, y, w, h)
            r_c  = roi_psnr_ssim(faithful, cap_out, x, y, w, h)
            psnr_base = r["psnr"]
            psnr_k3   = k["psnr"] if k else None
            psnr_cap  = c["psnr"] if c else None
            print(f"  {cat_name:<18}{fmt(r)}"
                  f"{'  d='+str(round(psnr_k3-psnr_base,2)) if psnr_k3 else '':>10}"
                  f"{'':>10}"
                  f"{'  d='+str(round(psnr_cap-psnr_base,2)) if psnr_cap else ''}")


if __name__ == "__main__":
    main()