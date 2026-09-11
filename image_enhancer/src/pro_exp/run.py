"""Professional Photo Enhancement - isolated experiment orchestrator.

Runs the proposed variants on all 6 real originals, computes full-frame and
region metrics, produces montages + zoom crops and writes CSVs.

Usage:
  .venv/Scripts/python src/pro_exp/run.py
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import enhance  # noqa: E402
from quality import compute_no_reference, evaluate_similarity  # noqa: E402
from restore_exp import faces as faces_mod  # noqa: E402
from restore_exp import pipeline, restormer, swinir_m  # noqa: E402
from restore_exp.metrics_ext import crop_rect, region_metrics  # noqa: E402
from restore_exp.plan_abc import build_montage  # noqa: E402
from pro_exp import pipeline2, prolook  # noqa: E402

FVAR = ["A", "D1", "D2", "P2", "P2b", "P3"]
REC = ["RECa", "RECb", "RECc"]
LABELS = {
    "A": "A: SwinIR-M PSNR (baseline)",
    "D1": "D1: denoise BEFORE SR",
    "D2": "D2: SR then denoise",
    "P2": "P2: A + ProLook lite",
    "P2b": "P2b: A + ProLook standard",
    "P3": "P3: ProLook + billboard (RECb)",
    "RECa": "RECa: billboard crop SwinIR",
    "RECb": "RECb: crop denoise->SwinIR",
    "RECc": "RECc: RECb + ProLook",
    "FC": "FC: face crop denoise->SwinIR->ProLook",
}
COLORS = {
    "Faithful": (60, 200, 255),
    "A": (80, 200, 255),
    "D1": (200, 160, 255),
    "D2": (120, 120, 120),
    "P2": (110, 255, 180),
    "P2b": (140, 140, 255),
    "P3": (70, 220, 220),
    "RECa": (0, 200, 200),
    "RECb": (0, 180, 255),
    "RECc": (0, 255, 0),
    "FC": (0, 180, 255),
}

OUT_DIR = ROOT / "enhanced" / "pro_exp"
REPORT_DIR = ROOT / "reports" / "pro_exp"
A_DIR = ROOT / "enhanced" / "restore_exp"
FULL_SCALE, BILL_SCALE, FACE_SCALE, DETAIL_SCALE = 0.32, 2.0, 3.0, 2.2


def save(img, rel):
    p = OUT_DIR / rel if img.shape[0] > 1000 else REPORT_DIR / "crops" / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), img)


def order_and_filter(cols):
    panels = [(img, lab, col) for lab, img, col in cols if img is not None]
    panels = [p for p in panels if np.ndim(p[0]) == 3 and p[0].shape[0] > 4
              and p[0].shape[1] > 4]
    if not panels:
        return panels
    ref = panels[0][0]
    th, tw = ref.shape[:2]
    out = []
    for img, lab, col in panels:
        if img.shape[:2] != (th, tw):
            img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        out.append((img, lab, col))
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "montages").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "crops").mkdir(parents=True, exist_ok=True)

    images = sorted(p.stem for p in (ROOT / "originals").glob("*.jpeg"))
    assert images, "no originals found"

    full_rows, bill_rows, face_rows, face_csv, detail_rows = [], [], [], [], []
    text_rows, runtime_rows = [], []
    dets_map = {}
    rec_crops = {}
    dw_map = {}

    for stem in images:
        print(f"== {stem}.jpeg ==")
        orig = cv2.imread(str(ROOT / "originals" / f"{stem}.jpeg"), cv2.IMREAD_COLOR)
        faithful = enhance.simple_upscale(orig, (3840, 2160))
        H, W = orig.shape[:2]

        t_det = time.perf_counter()
        dets = faces_mod.detect_faces(orig)
        det_s = time.perf_counter() - t_det
        dets_map[stem] = dets

        a_path = A_DIR / f"A_{stem}.png"
        if not a_path.exists():
            raise FileNotFoundError(f"Path A output missing: {a_path}")
        A = cv2.imread(str(a_path), cv2.IMREAD_COLOR)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        stats = {"image": f"{stem}.jpeg", "detect_s": round(det_s, 2)}
        dn = restormer.load("real_denoise", device)
        peak_mb = 0.0

        def stage_peak():
            if not torch.cuda.is_available():
                return 0.0
            p = torch.cuda.max_memory_allocated() / 1048576.0
            torch.cuda.reset_peak_memory_stats()
            return p

        t0 = time.perf_counter()
        d1a = restormer.enhance_bgr(dn, orig)
        t_dn = time.perf_counter() - t0
        t0 = time.perf_counter()
        d1 = swinir_m.sr_bgr(d1a, device, tile=256)
        t_sr = time.perf_counter() - t0
        d1 = cv2.resize(d1, (3840, 2160), interpolation=cv2.INTER_LANCZOS4)
        stats["d1_dn_s"] = round(t_dn, 2)
        stats["d1_sr_s"] = round(t_sr, 2)
        peak_mb = max(peak_mb, stage_peak())
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        t0 = time.perf_counter()
        d2 = pipeline2.tiled_restore(dn, A, tile=640, overlap=64)
        stats["d2_s"] = round(time.perf_counter() - t0, 2)
        peak_mb = max(peak_mb, stage_peak())
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        t0 = time.perf_counter()
        p2 = prolook.lite(A)
        stats["p2_s"] = round(time.perf_counter() - t0, 2)
        t0 = time.perf_counter()
        p2b = prolook.standard(A)
        stats["p2b_s"] = round(time.perf_counter() - t0, 2)

        p3 = p2.copy()
        boxes = pipeline.boxes_for(stem)
        stats["n_billboards"] = len(boxes)
        t0 = time.perf_counter()
        for bi, box in enumerate(boxes):
            pr = pipeline.padded(box, pipeline2.PAD_RATIO, W, H)
            rec = {mode: pipeline2.recipe_crop(orig, pr, mode, device)
                   for mode in REC}
            rec_crops[(stem, bi)] = rec
            for mode in REC:
                cv2.imwrite(str(REPORT_DIR / "crops"
                                / f"rec_b{bi}_{stem}_{mode}.png"), rec[mode])
            tar = pipeline.scale_rect((pr[0], pr[1], pr[2] - pr[0], pr[3] - pr[1]), W)
            pipeline.place_region(p3, rec["RECb"], (pr[0], pr[1], pr[2], pr[3]), tar)
        stats["bill_s"] = round(time.perf_counter() - t0, 2)
        peak_mb = max(peak_mb, stage_peak())

        variants = {"A": A, "D1": d1, "D2": d2, "P2": p2, "P2b": p2b, "P3": p3}
        for var in FVAR:
            cv2.imwrite(str(OUT_DIR / f"{var}_{stem}.png"), variants[var])
        stats["peak_vram_mb"] = round(peak_mb, 1)
        print(f"   A/D1/D2/P2/P2b/P3 built: peak {stats['peak_vram_mb']:.0f} MB, "
              f"faces={len(dets)}, boxes={len(boxes)}")
        runtime_rows.append(stats)

        for var in FVAR:
            m = evaluate_similarity(faithful, variants[var])
            nr = compute_no_reference(variants[var])
            full_rows.append({"image": f"{stem}.jpeg", "variant": var, **m,
                              "sharpness": round(nr["sharpness"], 2),
                              "noise": round(nr["noise"], 3),
                              "contrast": round(nr["contrast"], 4),
                              "edge_density": round(nr["edge_density"], 5)})

        for bi, box in enumerate(boxes):
            rect4 = pipeline.scale_rect(box, W)
            ref = crop_rect(faithful, rect4)
            for var in FVAR:
                m = region_metrics(ref, crop_rect(variants[var], rect4))
                if m:
                    bill_rows.append({"image": f"{stem}.jpeg", "box": bi,
                                      "variant": var, **m})
            for mode in REC:
                rc = pipeline2.to_box_size(rec_crops[(stem, bi)][mode], rect4)
                m = region_metrics(ref, rc)
                if m:
                    bill_rows.append({"image": f"{stem}.jpeg", "box": bi,
                                      "variant": mode, **m})
                    tp = pipeline2.text_proxies(rc)
                    text_rows.append({"image": f"{stem}.jpeg", "box": bi,
                                      "variant": mode, **tp})
            for var in ("A", "D1", "D2", "P2", "P2b", "P3"):
                tp = pipeline2.text_proxies(crop_rect(variants[var], rect4))
                text_rows.append({"image": f"{stem}.jpeg", "box": bi,
                                  "variant": var, **tp})

        for fi, d in enumerate(dets[:2]):
            face_csv.append({"image": f"{stem}.jpeg", "face": fi,
                             "x": d["x"], "y": d["y"], "w": d["w"], "h": d["h"],
                             "score": round(d["score"], 3), "area": int(d["area"])})
            fx, fy, fw, fh = pipeline.scale_rect((d["x"], d["y"], d["w"], d["h"]), W)
            ref = crop_rect(faithful, (fx, fy, fw, fh))
            if ref is None:
                continue
            for var in ("A", "D1", "P2"):
                m = region_metrics(ref, crop_rect(variants[var], (fx, fy, fw, fh)))
                if m:
                    face_rows.append({"image": f"{stem}.jpeg", "face": fi,
                                      "variant": var, **m})
            pr = pipeline.padded((d["x"], d["y"], d["w"], d["h"]), 0.30, W, H)
            fc = pipeline2.recipe_crop(orig, pr, "FC", device)
            fc = pipeline2.to_box_size(fc, (fx, fy, fw, fh))
            m = region_metrics(ref, fc)
            if m:
                face_rows.append({"image": f"{stem}.jpeg", "face": fi,
                                  "variant": "FC", **m})
            cv2.imwrite(str(REPORT_DIR / "crops" / f"rec_face{fi}_{stem}.png"), fc)

        ex = boxes + [(d["x"], d["y"], d["w"], d["h"]) for d in dets[:3]]
        dw = pipeline2.detail_window(orig, ex, win=160)
        dw_map[stem] = dw
        s = pipeline2.OUT_W / W
        rect4 = tuple(int(round(v * s)) for v in dw)
        ref = crop_rect(faithful, rect4)
        if ref is not None:
            for var in ("A", "D1", "P2"):
                m = region_metrics(ref, crop_rect(variants[var], rect4))
                if m:
                    detail_rows.append({"image": f"{stem}.jpeg",
                                        "variant": var, **m})

        sec = sum(v for k, v in stats.items() if k.endswith("_s"))
        print(f"   metrics done: runtime {sec:.1f}s")

    pd.DataFrame(full_rows).to_csv(REPORT_DIR / "full_metrics.csv", index=False)
    pd.DataFrame(bill_rows).to_csv(REPORT_DIR / "billboard_metrics.csv", index=False)
    pd.DataFrame(face_rows).to_csv(REPORT_DIR / "face_metrics.csv", index=False)
    pd.DataFrame(face_csv).to_csv(REPORT_DIR / "faces_detected.csv", index=False)
    pd.DataFrame(text_rows).to_csv(REPORT_DIR / "text_proxies.csv", index=False)
    pd.DataFrame(detail_rows).to_csv(REPORT_DIR / "detail_metrics.csv", index=False)
    pd.DataFrame(runtime_rows).to_csv(REPORT_DIR / "runtime.csv", index=False)

    for stem in images:
        orig = cv2.imread(str(ROOT / "originals" / f"{stem}.jpeg"), cv2.IMREAD_COLOR)
        faithful = enhance.simple_upscale(orig, (3840, 2160))
        H, W = orig.shape[:2]
        v = {var: cv2.imread(str(OUT_DIR / f"{var}_{stem}.png"), cv2.IMREAD_COLOR)
             for var in FVAR}
        panels = [
            (faithful, "Faithful Lanczos", COLORS["Faithful"]),
            (v["A"], LABELS["A"], COLORS["A"]),
            (v["D1"], LABELS["D1"], COLORS["D1"]),
            (v["D2"], LABELS["D2"], COLORS["D2"]),
            (v["P2"], LABELS["P2"], COLORS["P2"]),
            (v["P2b"], LABELS["P2b"], COLORS["P2b"]),
        ]
        build_montage(panels, REPORT_DIR / "montages" / f"cmp_full_{stem}.png",
                      scale=FULL_SCALE)

        boxes = pipeline.boxes_for(stem)
        if boxes:
            bx, by, bw, bh = pipeline.scale_rect(boxes[0], W)
            cols = [
                (f"Faithful {stem}", faithful, COLORS["Faithful"]),
                (f"A {stem}", v["A"], COLORS["A"]),
                (f"D1 {stem}", v["D1"], COLORS["D1"]),
                (f"D2 {stem}", v["D2"], COLORS["D2"]),
                (f"P2 {stem}", v["P2"], COLORS["P2"]),
                (f"P2b {stem}", v["P2b"], COLORS["P2b"]),
                (f"P3 {stem}", v["P3"], COLORS["P3"]),
            ]
            cols = [(lab, crop_rect(img, (bx, by, bw, bh)), col)
                    for lab, img, col in cols]
            for mode in REC:
                cols.append((f"{mode} {stem}",
                             pipeline2.to_box_size(rec_crops[(stem, 0)][mode],
                                                   (bx, by, bw, bh)),
                             COLORS[mode]))
            build_montage(order_and_filter(cols),
                          REPORT_DIR / "crops" / f"billboard_{stem}.png",
                          scale=BILL_SCALE)

        dets = dets_map[stem]
        for fi, d in enumerate(dets[:2]):
            fx, fy, fw, fh = pipeline.scale_rect((d["x"], d["y"], d["w"], d["h"]), W)
            fc = cv2.imread(str(REPORT_DIR / "crops" / f"rec_face{fi}_{stem}.png"),
                            cv2.IMREAD_COLOR)
            cols = [
                (f"Faithful {stem}", faithful, COLORS["Faithful"]),
                (f"A {stem}", v["A"], COLORS["A"]),
                (f"D1 {stem}", v["D1"], COLORS["D1"]),
                (f"P2 {stem}", v["P2"], COLORS["P2"]),
                (f"FC {stem}", fc, COLORS["FC"]),
            ]
            cols = [(lab, crop_rect(img, (fx, fy, fw, fh)), col)
                    for lab, img, col in cols]
            build_montage(order_and_filter(cols),
                          REPORT_DIR / "crops" / f"face{fi}_{stem}.png",
                          scale=FACE_SCALE)

        dw = dw_map[stem]
        s = pipeline2.OUT_W / W
        rect4 = tuple(int(round(v * s)) for v in dw)
        cols = [
            (f"Faithful {stem}", faithful, COLORS["Faithful"]),
            (f"A {stem}", v["A"], COLORS["A"]),
            (f"D1 {stem}", v["D1"], COLORS["D1"]),
            (f"P2 {stem}", v["P2"], COLORS["P2"]),
        ]
        cols = [(lab, crop_rect(img, rect4), col) for lab, img, col in cols]
        build_montage(order_and_filter(cols),
                          REPORT_DIR / "crops" / f"detail_{stem}.png",
                          scale=DETAIL_SCALE)

    print("\nruntime mean (s):")
    print(pd.DataFrame(runtime_rows).drop(columns=["image"]).mean(
        numeric_only=True).round(3).to_string())
    print("\nfull-frame mean (A/D1/D2/P2/P2b/P3):")
    print(pd.DataFrame(full_rows).drop(columns=["image"]).groupby(
        "variant").mean(numeric_only=True).round(3).to_string())
    print("\npro_exp artifacts ->", REPORT_DIR)


if __name__ == "__main__":
    main()
