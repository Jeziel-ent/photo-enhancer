"""MVP 2 Phase 2 -- aspect-ratio-preserving target-size formula.

Pure math, no GPU/model dependency. Verifies the "longest side = 3840,
preserve aspect ratio" formula against every example the MVP 2 review gave,
plus the full required ratio test set (1:1, 4:3, 3:2, 16:9, 9:16, 21:9).

Run:
    .venv\\Scripts\\python.exe research\\scripts\\aspect_ratio_formula.py
"""

from __future__ import annotations

MAX_DIM = 3840


def aspect_preserving_target(width: int, height: int, max_dim: int = MAX_DIM) -> tuple[int, int]:
    """The MVP 2 target-size formula: longest side becomes `max_dim`,
    aspect ratio preserved exactly, no stretch, no crop.

    Rounds to the nearest EVEN integer on both axes -- not just round():
    every downstream stage (Restormer/SwinIR-M tiling, cv2 LAB/YUV
    conversions, the multi-scale detail Gaussian pyramid) is safest with
    even dimensions, and it costs at most 1px of aspect-ratio drift, far
    below anything visible. Odd source aspect ratios (e.g. 3000x3000,
    already even) are unaffected.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width/height must be positive")
    scale = max_dim / max(width, height)
    out_w = round(width * scale)
    out_h = round(height * scale)
    # Round to nearest even, minimum 2.
    out_w = max(2, out_w - (out_w % 2))
    out_h = max(2, out_h - (out_h % 2))
    return out_w, out_h


# (input_w, input_h, expected_w, expected_h) -- exactly the review's own examples.
REVIEW_EXAMPLES = [
    (1920, 1080, 3840, 2160),
    (3000, 3000, 3840, 3840),
    (4000, 3000, 3840, 2880),
    (3000, 4000, 2880, 3840),
    (1600, 1200, 3840, 2880),
    (1200, 1600, 2880, 3840),
]

# The 6 required ratio families (Phase 2 / Phase 18), with one representative
# real-ish resolution each (arbitrary phone/camera-plausible sizes, not tied
# to any specific source image).
RATIO_FAMILIES = [
    ("1:1", 2400, 2400),
    ("4:3", 4032, 3024),
    ("3:2", 6000, 4000),
    ("16:9", 1920, 1080),
    ("9:16", 1080, 1920),
    ("21:9", 3440, 1440),
]


def _check_no_stretch(w0: int, h0: int, w1: int, h1: int, tol: float = 0.01) -> bool:
    """True if the output aspect ratio matches the input's within tol."""
    return abs((w0 / h0) - (w1 / h1)) < tol


def main() -> None:
    print(f"MAX_DIM = {MAX_DIM}\n")

    print("== Review's own worked examples ==")
    all_ok = True
    for w0, h0, exp_w, exp_h in REVIEW_EXAMPLES:
        got_w, got_h = aspect_preserving_target(w0, h0)
        ok = (got_w, got_h) == (exp_w, exp_h)
        all_ok &= ok
        print(f"{w0}x{h0} -> got {got_w}x{got_h}, expected {exp_w}x{exp_h}  "
              f"[{'OK' if ok else 'MISMATCH'}]")

    print("\n== Required ratio families ==")
    for name, w0, h0 in RATIO_FAMILIES:
        w1, h1 = aspect_preserving_target(w0, h0)
        no_stretch = _check_no_stretch(w0, h0, w1, h1)
        longest_is_3840 = max(w1, h1) == MAX_DIM
        even = (w1 % 2 == 0) and (h1 % 2 == 0)
        status = "OK" if (no_stretch and longest_is_3840 and even) else "FAIL"
        print(f"{name:5s} {w0}x{h0} -> {w1}x{h1}  "
              f"(aspect preserved: {no_stretch}, longest==3840: {longest_is_3840}, even: {even})  [{status}]")

    print(f"\nAll review examples matched exactly: {all_ok}")


if __name__ == "__main__":
    main()
