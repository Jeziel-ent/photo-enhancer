"""Recompute and save the billboard recipe crops (RECa/RECb/RECc) for the
pro_exp montages after a run produced the full-frame variants but the montage
pass was interrupted. Short GPU pass (crops only).

Usage:
  .venv/Scripts/python src/pro_exp/rec_extract.py
"""

import sys
from pathlib import Path

import cv2
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from restore_exp import faces as faces_mod  # noqa: E402
from restore_exp import pipeline, restormer  # noqa: E402
from pro_exp import pipeline2  # noqa: E402

REC = ["RECa", "RECb", "RECc"]
FACE_PAD = 0.30
REPORT_DIR = ROOT / "reports" / "pro_exp"
CROPS = REPORT_DIR / "crops"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    CROPS.mkdir(parents=True, exist_ok=True)
    dn = restormer.load("real_denoise", device)
    for p in sorted((ROOT / "originals").glob("*.jpeg")):
        stem = p.stem
        orig = cv2.imread(str(p), cv2.IMREAD_COLOR)
        H, W = orig.shape[:2]
        for bi, box in enumerate(pipeline.boxes_for(stem)):
            pr = pipeline.padded(box, pipeline2.PAD_RATIO, W, H)
            for mode in REC:
                crop = pipeline2.recipe_crop(orig, pr, mode, device)
                cv2.imwrite(str(CROPS / f"rec_b{bi}_{stem}_{mode}.png"), crop)
                print(f"{stem} box{bi} {mode} -> {crop.shape[1]}x{crop.shape[0]}")
        for fi, d in enumerate(faces_mod.detect_faces(orig)[:2]):
            fx, fy, fw, fh = pipeline.scale_rect(
                (d["x"], d["y"], d["w"], d["h"]), W)
            pr = pipeline.padded((d["x"], d["y"], d["w"], d["h"]), FACE_PAD, W, H)
            fc = pipeline2.recipe_crop(orig, pr, "FC", device)
            fc = pipeline2.to_box_size(fc, (fx, fy, fw, fh))
            cv2.imwrite(str(CROPS / f"rec_face{fi}_{stem}.png"), fc)
            print(f"{stem} face{fi} FC -> {fc.shape[1]}x{fc.shape[0]}")
    print("done")


if __name__ == "__main__":
    main()