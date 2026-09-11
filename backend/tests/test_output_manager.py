import tempfile
import unittest
import zipfile
from pathlib import Path

from backend.output_manager import build_result


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


if __name__ == "__main__":
    unittest.main()
