"""MVP 2 Phase 18 regression tests: aspect-ratio-preserving output.

Covers the review's own worked examples plus the full required ratio family
set (1:1, 4:3, 3:2, 16:9, 9:16, 21:9), and proves the wiring from
backend/engine_adapter.py (GPU) and backend/cpu_worker.py (CPU) actually
computes and threads a source-aspect-matching target through to the engine,
without paying for a real multi-second GPU/OpenVINO inference call per test
(the engine call itself is monkeypatched to a cheap stand-in that records
what target it received and returns a correctly-shaped dummy array — the
same "prove the wiring, not re-run the model" approach
test_engine_adapter.py's own billboard-neutralization tests already use).

Real end-to-end proof that the actual SR/restoration pipeline honors a
non-default target (not just that the wiring passes the right number) is
research/scripts/repro_stretch_bug.py + docs/MVP2_RESEARCH.md's own
recorded run — a real GPU image through the real engine, gated behind torch
being importable, which is why it lives under research/ instead of here
(this suite must stay fast and torch-import-tolerant like the rest of
backend/tests).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_ENHANCER_SRC = REPO_ROOT / "image_enhancer" / "src"
if str(IMAGE_ENHANCER_SRC) not in sys.path:
    sys.path.insert(0, str(IMAGE_ENHANCER_SRC))

from backend import cpu_worker, engine_adapter  # noqa: E402

try:
    import enhance  # noqa: E402  (image_enhancer/src/enhance.py)
    ENHANCE_IMPORT_ERROR = None
except Exception as exc:  # noqa: BLE001 - e.g. torch not installed in this env
    enhance = None
    ENHANCE_IMPORT_ERROR = exc

import enhance_shared  # noqa: E402  (torch-free, always importable)

# The review's own worked examples, verbatim.
REVIEW_EXAMPLES = [
    (1920, 1080, 3840, 2160),
    (3000, 3000, 3840, 3840),
    (4000, 3000, 3840, 2880),
    (3000, 4000, 2880, 3840),
    (1600, 1200, 3840, 2880),
    (1200, 1600, 2880, 3840),
]

# One representative resolution per required ratio family.
RATIO_FAMILIES = {
    "1:1": (2400, 2400),
    "4:3": (4032, 3024),
    "3:2": (6000, 4000),
    "16:9": (1920, 1080),
    "9:16": (1080, 1920),
    "21:9": (3440, 1440),
}


class AspectPreservingTargetFormulaTestCase(unittest.TestCase):
    """enhance_shared's copy is always importable (torch-free) -- this is
    the primary test target. A second class below additionally proves
    enhance.py's copy (used by the real GPU path) matches it exactly."""

    def test_review_examples_exact(self):
        for w0, h0, exp_w, exp_h in REVIEW_EXAMPLES:
            with self.subTest(input=(w0, h0)):
                got = enhance_shared.aspect_preserving_target(w0, h0)
                self.assertEqual(got, (exp_w, exp_h))

    def test_ratio_families_no_stretch_and_longest_side_3840(self):
        for name, (w0, h0) in RATIO_FAMILIES.items():
            with self.subTest(ratio=name):
                w1, h1 = enhance_shared.aspect_preserving_target(w0, h0)
                self.assertEqual(max(w1, h1), 3840)
                # Aspect ratio preserved within the <1px-per-side rounding
                # this formula documents.
                self.assertAlmostEqual(w0 / h0, w1 / h1, delta=0.01)
                self.assertEqual(w1 % 2, 0)
                self.assertEqual(h1 % 2, 0)

    def test_rejects_non_positive_dimensions(self):
        with self.assertRaises(ValueError):
            enhance_shared.aspect_preserving_target(0, 100)
        with self.assertRaises(ValueError):
            enhance_shared.aspect_preserving_target(100, -1)


@unittest.skipIf(enhance is None, f"enhance module unavailable: {ENHANCE_IMPORT_ERROR}")
class GpuEngineFormulaMatchesCpuFormulaTestCase(unittest.TestCase):
    """The two independently-maintained copies (enhance.py can't import
    enhance_shared.py's copy either, for the same torch-isolation reason
    cpu_final_enhance's own docstring gives) must never drift apart."""

    def test_formulas_agree_on_every_case(self):
        cases = [(w, h) for (w, h, _, _) in REVIEW_EXAMPLES] + list(RATIO_FAMILIES.values())
        for w0, h0 in cases:
            with self.subTest(input=(w0, h0)):
                self.assertEqual(
                    enhance.aspect_preserving_target(w0, h0),
                    enhance_shared.aspect_preserving_target(w0, h0),
                )


@unittest.skipIf(enhance is None, f"enhance module unavailable: {ENHANCE_IMPORT_ERROR}")
class GpuPathWiringTestCase(unittest.TestCase):
    """Proves backend/engine_adapter.py's GPU path computes the source's
    own aspect-preserving target and threads it into engine.enhance() --
    without paying for a real Restormer/SwinIR-M inference call. The real
    engine.enhance is monkeypatched to a cheap stand-in that records what
    target it received and returns a correctly-shaped dummy array, mirroring
    test_engine_adapter.py's own "prove the wiring" style."""

    def setUp(self):
        self._prev_device_state = dict(engine_adapter._DEVICE_STATE)
        engine_adapter._DEVICE_STATE["effective_device"] = "cuda"
        self.addCleanup(lambda: engine_adapter._DEVICE_STATE.update(self._prev_device_state))

    def test_non_16_9_input_gets_its_own_aspect_preserving_target(self):
        calls = {}

        def fake_enhance(method, img, target=(3840, 2160)):
            calls["method"] = method
            calls["target"] = target
            out = np.zeros((target[1], target[0], 3), dtype=np.uint8)
            return out, 0.01

        input_dir = REPO_ROOT / "backend" / ".workspace" / "_test_aspect_ratio"
        input_dir.mkdir(parents=True, exist_ok=True)
        input_path = input_dir / "square.png"
        output_path = input_dir / "square_out.png"
        try:
            import cv2
            square = np.full((256, 256, 3), 128, dtype=np.uint8)  # 1:1 input
            cv2.imwrite(str(input_path), square)

            with mock.patch.object(enhance, "enhance", side_effect=fake_enhance):
                engine_adapter.enhance_image(input_path, output_path)

            self.assertEqual(calls["target"], (3840, 3840))
            out = cv2.imread(str(output_path))
            self.assertEqual((out.shape[1], out.shape[0]), (3840, 3840))
        finally:
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)


class CpuPathWiringTestCase(unittest.TestCase):
    """Same proof as GpuPathWiringTestCase but for backend/cpu_worker.py's
    _run_one, which takes its `shared` module as an explicit parameter --
    so the REAL (torch-free) enhance_shared module is used directly, with
    only cpu_final_enhance itself monkeypatched to avoid a real OpenVINO
    inference call."""

    def test_non_16_9_input_gets_its_own_aspect_preserving_target(self):
        calls = {}

        def fake_cpu_final_enhance(img, target=(3840, 2160)):
            calls["target"] = target
            return np.zeros((target[1], target[0], 3), dtype=np.uint8)

        work_dir = REPO_ROOT / "backend" / ".workspace" / "_test_aspect_ratio_cpu"
        work_dir.mkdir(parents=True, exist_ok=True)
        input_path = work_dir / "portrait.png"
        output_path = work_dir / "portrait_out.png"
        try:
            import cv2
            portrait = np.full((1600, 1200, 3), 128, dtype=np.uint8)  # 4:3 portrait (H, W)
            cv2.imwrite(str(input_path), portrait)

            with mock.patch.object(
                enhance_shared, "cpu_final_enhance", side_effect=fake_cpu_final_enhance,
            ):
                cpu_worker._run_one(enhance_shared, str(input_path), str(output_path))

            # portrait.png is 1200 wide x 1600 tall -> aspect_preserving_target(1200, 1600)
            self.assertEqual(calls["target"], (2880, 3840))
            out = cv2.imread(str(output_path))
            self.assertEqual((out.shape[1], out.shape[0]), (2880, 3840))

            status_path = Path(str(output_path) + ".status.json")
            self.assertTrue(status_path.is_file())
        finally:
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)
            Path(str(output_path) + ".status.json").unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
