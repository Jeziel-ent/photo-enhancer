"""Unit tests for backend/shell.py's DesktopBridge — the parts that don't
need a real window (save_result and open_in_explorer's happy paths do, and
are exercised manually / by shell_smoke.py instead)."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend import recent_history
from backend.shell import DesktopBridge


class RecordSavedResultTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._patch = mock.patch.object(
            recent_history.workspace, "WORKSPACE_ROOT", Path(self._tmp.name))
        self._patch.start()
        self.addCleanup(self._patch.stop)
        self.bridge = DesktopBridge(job_manager=None)

    def test_record_saved_result_requires_a_path(self):
        result = self.bridge.record_saved_result({})
        self.assertEqual(result, {"ok": False, "error": "missing path"})

    def test_record_saved_result_infers_image_kind_from_extension(self):
        dest = Path(self._tmp.name) / "photo_enhanced.png"
        result = self.bridge.record_saved_result({"path": str(dest)})
        self.assertTrue(result["ok"])
        entry = result["entries"][0]
        self.assertEqual(entry["filename"], "photo_enhanced.png")
        self.assertEqual(entry["directory"], str(dest.parent))
        self.assertEqual(entry["kind"], "image")
        self.assertIsNone(entry["file_count"])

    def test_record_saved_result_infers_zip_kind_from_extension(self):
        dest = Path(self._tmp.name) / "enhanced_images.zip"
        result = self.bridge.record_saved_result(
            {"path": str(dest), "file_count": 3})
        entry = result["entries"][0]
        self.assertEqual(entry["kind"], "zip")
        self.assertEqual(entry["file_count"], 3)

    def test_get_recent_history_round_trips_through_record(self):
        dest = Path(self._tmp.name) / "photo_enhanced.png"
        self.bridge.record_saved_result({"path": str(dest)})
        result = self.bridge.get_recent_history()
        self.assertTrue(result["ok"])
        self.assertEqual(len(result["entries"]), 1)
        self.assertEqual(result["entries"][0]["filename"], "photo_enhanced.png")


class OpenInExplorerTestCase(unittest.TestCase):
    def setUp(self):
        self.bridge = DesktopBridge(job_manager=None)

    def test_missing_directory_is_reported_gracefully(self):
        result = self.bridge.open_in_explorer(
            r"C:\this\path\does\not\exist\anywhere\photo.png")
        self.assertEqual(result, {"ok": False, "error": "that folder no longer exists"})


class OpenSavedFileTestCase(unittest.TestCase):
    def setUp(self):
        self.bridge = DesktopBridge(job_manager=None)

    def test_missing_file_is_reported_gracefully(self):
        result = self.bridge.open_saved_file(
            r"C:\this\path\does\not\exist\anywhere\photo.png")
        self.assertEqual(result, {"ok": False, "error": "that file no longer exists"})


if __name__ == "__main__":
    unittest.main()
