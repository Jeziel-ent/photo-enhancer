"""Face-region detection for the conservative face pipeline.

OpenCV YuNet (opencv_zoo official ONNX, detection only). Used exclusively to
locate face regions so they can be processed conservatively; never to
synthesise or replace identity content. Faces below a size floor are left
untouched (preserve-original behaviour).
"""

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = ROOT / "models" / "face_detect" / "face_detection_yunet_2023mar.onnx"

MIN_SIZE = 24


def _load():
    if not MODEL_PATH.exists():
        raise RuntimeError(f"YuNet ONNX not present: {MODEL_PATH}")
    return cv2.FaceDetectorYN.create(
        str(MODEL_PATH), "", (320, 320),
        score_threshold=0.6, nms_threshold=0.3, top_k=5000)


def detect_faces(bgr, min_size=MIN_SIZE, score=0.7):
    """Return list of dicts with x,y,w,h top-left in original pixel coords."""
    h, w = bgr.shape[:2]
    fd = _load()
    fd.setInputSize((w, h))
    ok, faces = fd.detect(bgr)
    out = []
    if ok and faces is not None:
        for f in faces:
            x, y, fw, fh = (float(v) for v in f[:4])
            s = float(f[-1])
            if s >= score and fw >= min_size and fh >= min_size:
                out.append({"x": int(x), "y": int(y),
                            "w": int(fw), "h": int(fh), "score": s, "area": fw * fh})
    out.sort(key=lambda d: -d["area"])
    return out