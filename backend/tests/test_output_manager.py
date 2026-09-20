import tempfile
import unittest
import zipfile
from pathlib import Path

import cv2
import numpy as np

from backend.output_manager import build_batch_export, build_result


class TestBuildResult(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp_dir = Path(self._tmp.name)

    def _make_output(self, name: str, content: bytes) -> Path:
        path = self.tmp_dir / name
        path.write_bytes(content)
        return path

    def test_single_file_passthrough(self):
        out = self._make_output("000_photo.png", b"enhanced-bytes")
        result = build_result([("photo.jpg", out)], self.tmp_dir / "result.zip", is_batch=False)
        self.assertEqual(result, out)
        self.assertFalse((self.tmp_dir / "result.zip").exists())

    def test_multiple_files_zipped(self):
        out1 = self._make_output("000_a.png", b"AAA")
        out2 = self._make_output("001_b.png", b"BBB")
        zip_path = self.tmp_dir / "result.zip"
        result = build_result(
            [("a.jpg", out1), ("b.jpg", out2)], zip_path, is_batch=True)
        self.assertEqual(result, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            names = sorted(zf.namelist())
            self.assertEqual(names, ["a_enhanced.png", "b_enhanced.png"])
            self.assertEqual(zf.read("a_enhanced.png"), b"AAA")
            self.assertEqual(zf.read("b_enhanced.png"), b"BBB")

    def test_zip_of_partial_success_only_includes_succeeded(self):
        out1 = self._make_output("000_a.png", b"AAA")
        zip_path = self.tmp_dir / "result.zip"
        # only one of three uploads succeeded
        result = build_result([("a.jpg", out1)], zip_path, is_batch=True)
        with zipfile.ZipFile(result) as zf:
            self.assertEqual(zf.namelist(), ["a_enhanced.png"])

    def test_duplicate_stems_are_disambiguated(self):
        out1 = self._make_output("000_photo.png", b"AAA")
        out2 = self._make_output("001_photo.png", b"BBB")
        zip_path = self.tmp_dir / "result.zip"
        result = build_result(
            [("photo.jpg", out1), ("photo.jpg", out2)], zip_path, is_batch=True)
        with zipfile.ZipFile(result) as zf:
            names = sorted(zf.namelist())
            self.assertEqual(names, ["photo_enhanced.png", "photo_enhanced_2.png"])

    def test_no_succeeded_files_raises(self):
        with self.assertRaises(ValueError):
            build_result([], self.tmp_dir / "result.zip", is_batch=True)


def _count_red_pixels(img, x0, y0, x1, y1):
    region = img[y0:y1, x0:x1]
    red = (region[:, :, 2] > 180) & (region[:, :, 1] < 90) & (region[:, :, 0] < 90)
    return int(red.sum())


class TestBuildBatchExport(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp_dir = Path(self._tmp.name)

    def _make_png(self, name: str) -> Path:
        path = self.tmp_dir / name
        cv2.imwrite(str(path), np.full((2160, 3840, 3), 100, np.uint8))
        return path

    def test_common_format_applies_to_every_image(self):
        out1 = self._make_png("000_a.png")
        out2 = self._make_png("001_b.png")
        zip_path = self.tmp_dir / "export.zip"
        result = build_batch_export(
            [("a.jpg", out1, []), ("b.jpg", out2, [])], zip_path, "jpg", include_outlines=True)
        with zipfile.ZipFile(result) as zf:
            names = sorted(zf.namelist())
            self.assertEqual(names, ["a_enhanced.jpg", "b_enhanced.jpg"])

    def test_png_export_extension(self):
        out1 = self._make_png("000_a.png")
        zip_path = self.tmp_dir / "export.zip"
        result = build_batch_export(
            [("a.jpg", out1, [])], zip_path, "png", include_outlines=False)
        with zipfile.ZipFile(result) as zf:
            self.assertEqual(zf.namelist(), ["a_enhanced.png"])

    def test_jpeg_export_extension(self):
        out1 = self._make_png("000_a.png")
        zip_path = self.tmp_dir / "export.zip"
        result = build_batch_export(
            [("a.jpg", out1, [])], zip_path, "jpeg", include_outlines=False)
        with zipfile.ZipFile(result) as zf:
            self.assertEqual(zf.namelist(), ["a_enhanced.jpeg"])

    def test_each_image_only_gets_its_own_rects(self):
        out1 = self._make_png("000_a.png")
        out2 = self._make_png("001_b.png")
        zip_path = self.tmp_dir / "export.zip"
        rects_a = [{"x": 100, "y": 100, "width": 300, "height": 200}]
        result = build_batch_export(
            [("a.jpg", out1, rects_a), ("b.jpg", out2, [])],
            zip_path, "png", include_outlines=True)
        with zipfile.ZipFile(result) as zf:
            a_bytes = zf.read("a_enhanced.png")
            b_bytes = zf.read("b_enhanced.png")
        a_img = cv2.imdecode(np.frombuffer(a_bytes, np.uint8), cv2.IMREAD_COLOR)
        b_img = cv2.imdecode(np.frombuffer(b_bytes, np.uint8), cv2.IMREAD_COLOR)
        self.assertGreater(_count_red_pixels(a_img, 100, 100, 400, 108), 0)
        self.assertEqual(_count_red_pixels(b_img, 100, 100, 400, 108), 0)

    def test_include_outlines_off_produces_clean_images(self):
        out1 = self._make_png("000_a.png")
        zip_path = self.tmp_dir / "export.zip"
        rects_a = [{"x": 100, "y": 100, "width": 300, "height": 200}]
        result = build_batch_export(
            [("a.jpg", out1, rects_a)], zip_path, "png", include_outlines=False)
        with zipfile.ZipFile(result) as zf:
            a_bytes = zf.read("a_enhanced.png")
        a_img = cv2.imdecode(np.frombuffer(a_bytes, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(_count_red_pixels(a_img, 100, 100, 400, 108), 0)

    def test_duplicate_stems_are_disambiguated(self):
        out1 = self._make_png("000_photo.png")
        out2 = self._make_png("001_photo.png")
        zip_path = self.tmp_dir / "export.zip"
        result = build_batch_export(
            [("photo.jpg", out1, []), ("photo.jpg", out2, [])],
            zip_path, "png", include_outlines=False)
        with zipfile.ZipFile(result) as zf:
            names = sorted(zf.namelist())
            self.assertEqual(names, ["photo_enhanced.png", "photo_enhanced_2.png"])

    def test_no_images_raises(self):
        with self.assertRaises(ValueError):
            build_batch_export([], self.tmp_dir / "export.zip", "png", include_outlines=True)

    # ---------------------------------------------- per-image `adjust`
    # (4-tuple items: (name, path, rects, per_image_adjust)) -- regression
    # coverage for the per-image manual-adjustment batch export feature.
    # Uses lossless PNG throughout so exported bytes can be compared for
    # EXACT pixel equality against backend.adjustment_overlay.render_bgr
    # (the one shared implementation preview/export/batch all call), not
    # just "looks different".

    def test_per_image_adjust_applied_independently(self):
        from backend import adjustment_overlay

        out_a = self._make_png("000_a.png")
        out_b = self._make_png("001_b.png")
        base = cv2.imread(str(out_a), cv2.IMREAD_COLOR)
        adjust_a = {"brightness": 20, "contrast": 10, "highlights": 0,
                    "shadows": 0, "saturation": 0, "detail": 15}
        adjust_b = {"brightness": -20, "contrast": -10, "highlights": 0,
                    "shadows": 0, "saturation": 0, "detail": 0}
        zip_path = self.tmp_dir / "export.zip"

        result = build_batch_export(
            [("a.jpg", out_a, [], adjust_a), ("b.jpg", out_b, [], adjust_b)],
            zip_path, "png", include_outlines=True)

        with zipfile.ZipFile(result) as zf:
            a_bytes = zf.read("a_enhanced.png")
            b_bytes = zf.read("b_enhanced.png")
        a_img = cv2.imdecode(np.frombuffer(a_bytes, np.uint8), cv2.IMREAD_COLOR)
        b_img = cv2.imdecode(np.frombuffer(b_bytes, np.uint8), cv2.IMREAD_COLOR)

        # both exported successfully, at the source's full resolution
        self.assertEqual(a_img.shape, base.shape)
        self.assertEqual(b_img.shape, base.shape)

        # each image received ONLY its own adjustment values: exact
        # pixel-for-pixel match against the shared render_bgr() applied
        # with that image's OWN adjust dict on the SAME identical source
        expected_a = adjustment_overlay.render_bgr(base, [], adjust_a)
        expected_b = adjustment_overlay.render_bgr(base, [], adjust_b)
        np.testing.assert_array_equal(a_img, expected_a)
        np.testing.assert_array_equal(b_img, expected_b)

        # adjustments are not shared/cross-applied: A (brightened) must be
        # strictly brighter than B (darkened) despite an identical source
        self.assertGreater(float(a_img.mean()), float(base.mean()))
        self.assertLess(float(b_img.mean()), float(base.mean()))
        self.assertGreater(float(a_img.mean()), float(b_img.mean()))
        # and neither leaked the other's values onto it
        self.assertFalse(np.array_equal(a_img, b_img))

    def test_entries_without_adjust_fall_back_to_common_adjust(self):
        from backend import adjustment_overlay

        out_a = self._make_png("000_a.png")
        base = cv2.imread(str(out_a), cv2.IMREAD_COLOR)
        common_adjust = {"brightness": 30, "contrast": 0, "highlights": 0,
                          "shadows": 0, "saturation": 0, "detail": 0}
        zip_path = self.tmp_dir / "export.zip"

        # 4-tuple item with per_image_adjust=None must fall back to the
        # shared `adjust` kwarg -- same as before this feature existed
        result = build_batch_export(
            [("a.jpg", out_a, [], None)], zip_path, "png",
            include_outlines=False, adjust=common_adjust)

        with zipfile.ZipFile(result) as zf:
            a_bytes = zf.read("a_enhanced.png")
        a_img = cv2.imdecode(np.frombuffer(a_bytes, np.uint8), cv2.IMREAD_COLOR)
        expected = adjustment_overlay.render_bgr(base, [], common_adjust)
        np.testing.assert_array_equal(a_img, expected)

    def test_legacy_three_tuple_items_still_work_with_per_image_adjust(self):
        """Backward compatibility: a batch mixing OLD 3-tuple items (no
        per-image adjust at all) and NEW 4-tuple items (own adjust) in the
        SAME call must not break either -- the 3-tuple item behaves exactly
        as it did before this feature existed (untouched, since no common
        `adjust` is given either), and the 4-tuple item still gets its own
        adjustment."""
        from backend import adjustment_overlay

        out_a = self._make_png("000_a.png")
        out_b = self._make_png("001_b.png")
        base = cv2.imread(str(out_a), cv2.IMREAD_COLOR)
        adjust_b = {"brightness": -25, "contrast": 0, "highlights": 0,
                    "shadows": 0, "saturation": 0, "detail": 0}
        zip_path = self.tmp_dir / "export.zip"

        result = build_batch_export(
            [("a.jpg", out_a, []), ("b.jpg", out_b, [], adjust_b)],
            zip_path, "png", include_outlines=False)

        with zipfile.ZipFile(result) as zf:
            a_bytes = zf.read("a_enhanced.png")
            b_bytes = zf.read("b_enhanced.png")
        a_img = cv2.imdecode(np.frombuffer(a_bytes, np.uint8), cv2.IMREAD_COLOR)
        b_img = cv2.imdecode(np.frombuffer(b_bytes, np.uint8), cv2.IMREAD_COLOR)

        # legacy 3-tuple item: no adjust applied at all (no common `adjust`
        # was passed either) -- byte-identical to the untouched source
        np.testing.assert_array_equal(a_img, base)
        # the 4-tuple sibling in the SAME batch still got its own adjust
        expected_b = adjustment_overlay.render_bgr(base, [], adjust_b)
        np.testing.assert_array_equal(b_img, expected_b)
        self.assertLess(float(b_img.mean()), float(a_img.mean()))


if __name__ == "__main__":
    unittest.main()
