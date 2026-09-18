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
OUTPUT_DIR = Path("pan_4k_benchmark")

TARGET_W = 3840
TARGET_H = 2160

TILE_SIZE = 256
OVERLAP = 16
STRIDE = TILE_SIZE - OVERLAP


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    """Smooth feathering weight for tile stitching."""
    weight = np.ones((size, size), dtype=np.float32)

    if overlap > 0:
        ramp = np.linspace(0.0, 1.0, overlap, dtype=np.float32)

        weight[:, :overlap] *= ramp[None, :]
        weight[:, -overlap:] *= ramp[::-1][None, :]
        weight[:overlap, :] *= ramp[:, None]
        weight[-overlap:, :] *= ramp[::-1, None]

    return np.maximum(weight, 1e-6)


def process_image(model, device, image_path, output_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(f"Could not read: {image_path}")

    original_h, original_w = image.shape[:2]

    # Convert BGR -> RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Pad so every input tile is exactly 256x256.
    pad_h = (STRIDE - (original_h - TILE_SIZE) % STRIDE) % STRIDE
    pad_w = (STRIDE - (original_w - TILE_SIZE) % STRIDE) % STRIDE

    padded = cv2.copyMakeBorder(
        rgb,
        0,
        pad_h,
        0,
        pad_w,
        cv2.BORDER_REFLECT_101,
    )

    padded_h, padded_w = padded.shape[:2]

    output_h = padded_h * 4
    output_w = padded_w * 4

    output = np.zeros(
        (output_h, output_w, 3),
        dtype=np.float32,
    )

    weights = np.zeros(
        (output_h, output_w, 1),
        dtype=np.float32,
    )

    weight = make_weight(TILE_SIZE * 4, OVERLAP * 4)

    tile_count = 0

    start = time.perf_counter()

    with torch.inference_mode():
        for y in range(0, padded_h - TILE_SIZE + 1, STRIDE):
            for x in range(0, padded_w - TILE_SIZE + 1, STRIDE):

                tile = padded[
                    y:y + TILE_SIZE,
                    x:x + TILE_SIZE,
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
                    oy:oy + TILE_SIZE * 4,
                    ox:ox + TILE_SIZE * 4,
                ] += result * weight[..., None]

                weights[
                    oy:oy + TILE_SIZE * 4,
                    ox:ox + TILE_SIZE * 4,
                ] += weight[..., None]

                tile_count += 1

    output /= np.maximum(weights, 1e-6)

    # Remove padding.
    output = output[
        :original_h * 4,
        :original_w * 4,
    ]

    # Convert to uint8.
    output = (
        np.clip(output, 0.0, 1.0) * 255.0
    ).round().astype(np.uint8)

    # Final 4K target.
    output = cv2.resize(
        output,
        (TARGET_W, TARGET_H),
        interpolation=cv2.INTER_LANCZOS4,
    )

    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(output_path), output)

    elapsed = time.perf_counter() - start

    return {
        "input": (original_w, original_h),
        "tiles": tile_count,
        "time": elapsed,
        "output": output.shape,
        "finite": np.isfinite(output).all(),
    }


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    model, device = load_model()

    print("=" * 60)
    print("ADINN PAN 4K BENCHMARK")
    print("=" * 60)
    print("DEVICE:", torch.cuda.get_device_name(0))
    print("MODEL:", MODEL_PATH)
    print("TILE:", TILE_SIZE)
    print("OVERLAP:", OVERLAP)
    print("TARGET:", f"{TARGET_W}x{TARGET_H}")
    print()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    results = []

    for image_path in sorted(INPUT_DIR.glob("*.jpeg")):
        output_path = OUTPUT_DIR / f"{image_path.stem}_pan_4k.png"

        print(f"Processing {image_path.name}...")

        result = process_image(
            model,
            device,
            image_path,
            output_path,
        )

        results.append(result)

        print(
            f"  Input:  {result['input'][0]}x{result['input'][1]}"
        )
        print(f"  Tiles:  {result['tiles']}")
        print(f"  Time:   {result['time']:.3f}s")
        print(f"  Output: {result['output']}")
        print(f"  Finite: {result['finite']}")
        print()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_vram = (
            torch.cuda.max_memory_allocated()
            / 1024
            / 1024
        )
    else:
        peak_vram = 0

    times = [r["time"] for r in results]

    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    for path, result in zip(
        sorted(INPUT_DIR.glob("*.jpeg")),
        results,
    ):
        print(
            f"{path.name}: "
            f"{result['time']:.3f}s, "
            f"{result['tiles']} tiles, "
            f"{result['output'][1]}x{result['output'][0]}"
        )

    print()
    print(f"MEAN TIME: {np.mean(times):.3f}s")
    print(f"MIN TIME:  {np.min(times):.3f}s")
    print(f"MAX TIME:  {np.max(times):.3f}s")
    print(f"PEAK VRAM: {peak_vram:.1f} MB")
    print()
    print("OUTPUT DIR:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()