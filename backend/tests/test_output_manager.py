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


if __name__ == "__main__":
    unittest.main()
