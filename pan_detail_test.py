import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, "image_enhancer/src")

from pan_arch import PAN


MODEL_PATH = Path("image_enhancer/models/pan/pan_4x.pt")
INPUT_DIR = Path("image_enhancer/originals")
OUTPUT_DIR = Path("pan_detail_test")

TARGET_SIZE = (3840, 2160)

TILE = 256
OVERLAP = 16
STRIDE = TILE - OVERLAP

DETAIL_STRENGTH = 0.30
EDGE_THRESHOLD = 0.20
DETAIL_CLIP = 8.0


def load_model():
    device = torch.device("cuda")

    model = PAN(
        in_nc=3,
        out_nc=3,
        nf=40,
        unf=24,
        nb=16,
        scale=4,
    ).to(device).eval()

    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state, strict=True)

    return model, device


def make_weight(size, overlap):
    weight = np.ones((size, size), dtype=np.float32)

    ramp = np.linspace(0.0, 1.0, overlap, dtype=np.float32)

    weight[:, :overlap] *= ramp[None, :]
    weight[:, -overlap:] *= ramp[::-1][None, :]
    weight[:overlap, :] *= ramp[:, None]
    weight[-overlap:, :] *= ramp[::-1, None]

    return np.maximum(weight, 1e-6)


def pan_upscale(model, device, image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w = rgb.shape[:2]

    pad_h = (STRIDE - (h - TILE) % STRIDE) % STRIDE
    pad_w = (STRIDE - (w - TILE) % STRIDE) % STRIDE

    padded = cv2.copyMakeBorder(
        rgb,
        0,
        pad_h,
        0,
        pad_w,
        cv2.BORDER_REFLECT_101,
    )

    ph, pw = padded.shape[:2]

    output = np.zeros(
        (ph * 4, pw * 4, 3),
        dtype=np.float32,
    )

    weights = np.zeros(
        (ph * 4, pw * 4, 1),
        dtype=np.float32,
    )

    weight = make_weight(TILE * 4, OVERLAP * 4)

    with torch.inference_mode():
        for y in range(0, ph - TILE + 1, STRIDE):
            for x in range(0, pw - TILE + 1, STRIDE):

                tile = padded[
                    y:y + TILE,
                    x:x + TILE,
                ]

                tensor = (
                    torch.from_numpy(tile)
                    .permute(2, 0, 1)
                    .float()
                    .div(255.0)
                    .unsqueeze(0)
                    .to(device)
                )

                result = model(tensor).clamp(0, 1)

                result = (
                    result[0]
                    .permute(1, 2, 0)
                    .cpu()
                    .numpy()
                )

                oy = y * 4
                ox = x * 4

                output[
                    oy:oy + TILE * 4,
                    ox:ox + TILE * 4,
                ] += result * weight[..., None]

                weights[
                    oy:oy + TILE * 4,
                    ox:ox + TILE * 4,
                ] += weight[..., None]

    output /= np.maximum(weights, 1e-6)

    output = output[:h * 4, :w * 4]

    output = (
        np.clip(output, 0, 1) * 255
    ).round().astype(np.uint8)

    return cv2.cvtColor(output, cv2.COLOR_RGB2BGR)


def controlled_detail(image):
    """
    Conservative luminance-only detail enhancement.

    No generative processing.
    No color synthesis.
    No geometry modification.
    """

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    l_float = l.astype(np.float32)

    # Local high-frequency detail.
    blur = cv2.GaussianBlur(
        l_float,
        (0, 0),
        sigmaX=1.2,
    )

    detail = l_float - blur

    # Edge strength.
    gx = cv2.Sobel(
        l_float,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        l_float,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    edge = cv2.magnitude(gx, gy)

    edge /= max(float(edge.max()), 1.0)

    # Suppress enhancement in very smooth regions.
    gate = np.clip(
        (edge - EDGE_THRESHOLD)
        / max(1.0 - EDGE_THRESHOLD, 1e-6),
        0.0,
        1.0,
    )

    enhanced_detail = (
        detail
        * gate
        * DETAIL_STRENGTH
    )

    # Hard safety bound.
    enhanced_detail = np.clip(
        enhanced_detail,
        -DETAIL_CLIP,
        DETAIL_CLIP,
    )

    enhanced_l = np.clip(
        l_float + enhanced_detail,
        0,
        255,
    ).astype(np.uint8)

    result = cv2.merge(
        [enhanced_l, a, b]
    )

    return cv2.cvtColor(
        result,
        cv2.COLOR_LAB2BGR,
    )


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    model, device = load_model()

    print("=" * 60)
    print("PAN + CONTROLLED DETAIL TEST")
    print("=" * 60)
    print("GPU:", torch.cuda.get_device_name(0))
    print("DETAIL STRENGTH:", DETAIL_STRENGTH)
    print("EDGE THRESHOLD:", EDGE_THRESHOLD)
    print("DETAIL CLIP:", DETAIL_CLIP)
    print()

    for image_path in sorted(INPUT_DIR.glob("*.jpeg")):

        print("Processing:", image_path.name)

        image = cv2.imread(str(image_path))

        if image is None:
            raise RuntimeError(
                f"Could not read {image_path}"
            )

        start = time.perf_counter()

        pan = pan_upscale(
            model,
            device,
            image,
        )

        detailed = controlled_detail(pan)

        detailed = cv2.resize(
            detailed,
            TARGET_SIZE,
            interpolation=cv2.INTER_LANCZOS4,
        )

        elapsed = time.perf_counter() - start

        output_path = (
            OUTPUT_DIR
            / f"{image_path.stem}_pan_detail_4k.png"
        )

        cv2.imwrite(
            str(output_path),
            detailed,
        )

        print(
            f"  Time: {elapsed:.3f}s"
        )
        print(
            f"  Output: {detailed.shape}"
        )
        print(
            f"  Saved: {output_path}"
        )
        print()

    print("=" * 60)
    print("DONE")
    print("=" * 60)

if __name__ == "__main__":
    main()