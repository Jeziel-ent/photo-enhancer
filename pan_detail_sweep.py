from pathlib import Path

import cv2
import numpy as np


INPUT_DIR = Path("pan_4k_benchmark")
OUTPUT_DIR = Path("pan_detail_sweep")

STRENGTHS = [0.30, 0.55, 0.80]
EDGE_THRESHOLD = 0.20
DETAIL_CLIP = 8.0


def controlled_detail(image, strength):
    """
    Conservative luminance-only detail enhancement.

    The operation modifies only existing luminance detail.
    No generative processing or color synthesis.
    """

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    l_float = l.astype(np.float32)

    # Existing high-frequency luminance detail.
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

    # Only enhance meaningful edges.
    gate = np.clip(
        (edge - EDGE_THRESHOLD)
        / max(1.0 - EDGE_THRESHOLD, 1e-6),
        0.0,
        1.0,
    )

    enhanced_detail = (
        detail
        * gate
        * strength
    )

    # Keep the experiment bounded.
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

    inputs = sorted(
        INPUT_DIR.glob("*_pan_4k.png")
    )

    if not inputs:
        raise RuntimeError(
            "No PAN 4K benchmark outputs found."
        )

    print("=" * 60)
    print("PAN DETAIL STRENGTH SWEEP")
    print("=" * 60)
    print("INPUT:", INPUT_DIR.resolve())
    print("STRENGTHS:", STRENGTHS)
    print("EDGE THRESHOLD:", EDGE_THRESHOLD)
    print("DETAIL CLIP:", DETAIL_CLIP)
    print()

    for input_path in inputs:

        image = cv2.imread(str(input_path))

        if image is None:
            raise RuntimeError(
                f"Could not read: {input_path}"
            )

        print("Processing:", input_path.name)

        for strength in STRENGTHS:

            result = controlled_detail(
                image,
                strength,
            )

            output_name = (
                f"{input_path.stem}"
                f"_detail_{strength:.2f}.png"
            )

            output_path = OUTPUT_DIR / output_name

            cv2.imwrite(
                str(output_path),
                result,
            )

            print(
                f"  strength={strength:.2f}"
                f" -> {output_path.name}"
            )

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print("OUTPUT:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()