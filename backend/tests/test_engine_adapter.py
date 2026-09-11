"""Regression test for the billboard-regions neutralization in
engine_adapter.py.

image_enhancer/regions.json is the R&D engine's own fixture: manually
verified billboard boxes for the six sample OOH photos, keyed by filename
and also matched by (width, height) as a fallback when the filename doesn't
match. If a user's generic upload to this product happens to share one of
those six exact resolutions, that fallback could silently make
enhance.final_enhance() run billboard-box reconstruction over some unrelated
region of an unrelated photo -- exactly the kind of unintended alteration
the product must never do. This test proves engine_adapter closes that off,
using the real `enhance` module and the real regions.json (no GPU compute
involved -- only `_billboard_boxes()`, not the SR pipeline, is exercised).
"""

import os
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_ENHANCER_SRC = REPO_ROOT / "image_enhancer" / "src"
if str(IMAGE_ENHANCER_SRC) not in sys.path:
    sys.path.insert(0, str(IMAGE_ENHANCER_SRC))

from backend import engine_adapter  # noqa: E402

try:
    import enhance  # noqa: E402  (image_enhancer/src/enhance.py)
    ENHANCE_IMPORT_ERROR = None
except Exception as exc:  # noqa: BLE001 - e.g. torch not installed in this env
    enhance = None
    ENHANCE_IMPORT_ERROR = exc


@unittest.skipIf(enhance is None, f"enhance module unavailable: {ENHANCE_IMPORT_ERROR}")
class BillboardRegionsNeutralizedTestCase(unittest.TestCase):
    REGIONS_JSON = REPO_ROOT / "image_enhancer" / "regions.json"

    def setUp(self):
        # Snapshot and restore process-global state this test touches, so it
        # can't leak into other tests regardless of pass/fail.
        self._prev_regions = os.environ.get("REGIONS_CONFIG")
        self._prev_billboard = os.environ.get("BILLBOARD_IMAGE")
        self._prev_region_config = enhance._REGION_CONFIG
        self.addCleanup(self._restore)

    def _restore(self):
        if self._prev_regions is None:
            os.environ.pop("REGIONS_CONFIG", None)
        else:
            os.environ["REGIONS_CONFIG"] = self._prev_regions
        if self._prev_billboard is None:
            os.environ.pop("BILLBOARD_IMAGE", None)
        else:
            os.environ["BILLBOARD_IMAGE"] = self._prev_billboard
        enhance._REGION_CONFIG = self._prev_region_config

    def test_the_landmine_is_real_without_the_fix(self):
        """Establishes the premise: pointing the engine straight at its own
        regions.json DOES leak a billboard box onto an image that merely
        shares a recorded image_size, with no filename match at all."""
        self.assertTrue(self.REGIONS_JSON.is_file(), "fixture moved?")
        os.environ["REGIONS_CONFIG"] = str(self.REGIONS_JSON)
        os.environ.pop("BILLBOARD_IMAGE", None)  # no filename given
        enhance._REGION_CONFIG = None

        # 1.jpeg's recorded image_size in regions.json is [1374, 773].
        dummy = np.zeros((773, 1374, 3), dtype=np.uint8)
        boxes = enhance._billboard_boxes(dummy)
        self.assertTrue(boxes, "expected the same-size fallback to leak a box")

    def test_neutralize_closes_the_landmine(self):
        os.environ["REGIONS_CONFIG"] = str(self.REGIONS_JSON)
        os.environ.pop("BILLBOARD_IMAGE", None)
        enhance._REGION_CONFIG = None

        prev_regions, prev_billboard = engine_adapter._neutralize_billboard_regions(enhance)
        try:
            dummy = np.zeros((773, 1374, 3), dtype=np.uint8)  # same size as 1.jpeg
            boxes = enhance._billboard_boxes(dummy)
            self.assertEqual(boxes, [], "neutralization must force zero billboard boxes")
        finally:
            engine_adapter._restore_billboard_regions(enhance, prev_regions, prev_billboard)

        self.assertEqual(os.environ.get("REGIONS_CONFIG"), str(self.REGIONS_JSON))

    def test_neutralize_also_defeats_exact_filename_match(self):
        """Even an upload that happens to be named "1.jpeg" (matching by
        name, not just by size) must not pick up the fixture's box."""
        os.environ["REGIONS_CONFIG"] = str(self.REGIONS_JSON)
        os.environ["BILLBOARD_IMAGE"] = "1.jpeg"
        enhance._REGION_CONFIG = None

        prev_regions, prev_billboard = engine_adapter._neutralize_billboard_regions(enhance)
        try:
            dummy = np.zeros((773, 1374, 3), dtype=np.uint8)
            self.assertEqual(enhance._billboard_boxes(dummy), [])
        finally:
            engine_adapter._restore_billboard_regions(enhance, prev_regions, prev_billboard)

    def test_restore_leaves_no_trace_when_no_prior_env_vars(self):
        os.environ.pop("REGIONS_CONFIG", None)
        os.environ.pop("BILLBOARD_IMAGE", None)
        prev_regions, prev_billboard = engine_adapter._neutralize_billboard_regions(enhance)
        engine_adapter._restore_billboard_regions(enhance, prev_regions, prev_billboard)
        self.assertNotIn("REGIONS_CONFIG", os.environ)
        self.assertNotIn("BILLBOARD_IMAGE", os.environ)


if __name__ == "__main__":
    unittest.main()
