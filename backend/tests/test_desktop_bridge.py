"""Unit tests for backend/shell.py's DesktopBridge — the parts that don't
need a real window (save_result and open_in_explorer's happy paths do, and
are exercised manually / by shell_smoke.py instead)."""

import sys
import tempfile
import types
import unittest
import json
import zipfile
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from backend import jobs, recent_history
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


class _FakeJobManager:
    def __init__(self, job):
        self._job = job

    def get_job(self, job_id):
        return self._job if job_id == self._job.id else None


def _count_red_pixels(img, x0, y0, x1, y1):
    """Counts pixels inside a box that look like the baked board-red border
    (BGR (0,0,230)) — the plain photo content (neutral gray here) never
    matches, so this is a reliable detector of the drawn outline/label."""
    region = img[y0:y1, x0:x1]
    red = (region[:, :, 2] > 180) & (region[:, :, 1] < 90) & (region[:, :, 0] < 90)
    return int(red.sum())


def _fake_webview_module(dest_path):
    """A minimal stand-in for the `webview` module — this test never opens
    a real window, so it only needs create_file_dialog to hand back a
    caller-chosen destination path, like the user picking one in the
    native Save As dialog."""
    fake = types.ModuleType("webview")
    fake.SAVE_DIALOG = "save"

    class _FakeWindow:
        def create_file_dialog(self, dialog_type, save_filename=None, file_types=None):
            del dialog_type, save_filename, file_types
            return dest_path

    fake.windows = [_FakeWindow()]
    return fake


class SaveResultAsTestCase(unittest.TestCase):
    """Covers save_result_as — the backend method behind the billboard
    editor's Save Image format dropdown (frontend/src/components/upload/
    JobResultPanel.tsx) and its confirmed-rectangle burn-in. It is pure
    image I/O (cv2 imread/imwrite plus billboard_overlay compositing) and
    never touches the enhancement engine; the engine's own output file is
    never written to, and with no rectangles a PNG save is still a
    byte-identical copy of it (overlays are the ONLY reason a PNG save ever
    re-encodes)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

        result_path = Path(self._tmp.name) / "photo_enhanced.png"
        # A neutral gray frame (not red) so the baked overlay's red border
        # and tint are clearly detectable in the saved pixels.
        image = np.full((2160, 3840, 3), 100, dtype=np.uint8)
        cv2.imwrite(str(result_path), image)

        file_result = jobs.FileResult(
            original_filename="photo.jpg",
            input_path=Path(self._tmp.name) / "photo.jpg",
        )
        self.job = jobs.Job(id="job-1", files=[file_result])
        self.job.status = jobs.STATUS_COMPLETED
        self.job.result_path = result_path
        self.job.result_filename = "photo_enhanced.png"

        self.bridge = DesktopBridge(job_manager=_FakeJobManager(self.job))

    def _save_as(self, image_format, dest_name, billboard_rects=None):
        dest_path = Path(self._tmp.name) / dest_name
        fake_webview = _fake_webview_module(str(dest_path))
        with mock.patch.dict(sys.modules, {"webview": fake_webview}):
            if billboard_rects is None:
                result = self.bridge.save_result_as(self.job.id, image_format)
            else:
                result = self.bridge.save_result_as(
                    self.job.id, image_format, billboard_rects)
        return result, dest_path

    def test_png_save_is_a_byte_identical_copy(self):
        result, dest_path = self._save_as("png", "out.png")
        self.assertTrue(result["ok"])
        self.assertEqual(dest_path.read_bytes(), self.job.result_path.read_bytes())
        saved = cv2.imread(str(dest_path))
        self.assertEqual(saved.shape[:2], (2160, 3840))

    def test_png_with_a_rect_bakes_red_border_and_preserves_dims(self):
        rects = [{"x": 500, "y": 300, "width": 800, "height": 400}]
        result, dest_path = self._save_as("png", "out.png", rects)
        self.assertTrue(result["ok"])
        saved = cv2.imread(str(dest_path))
        self.assertEqual(saved.shape[:2], (2160, 3840))
        # The rect's top border row must contain pure-red pixels now.
        self.assertGreater(_count_red_pixels(saved, 500, 300, 1300, 308), 0)
        # Far away from any rectangle nothing gets painted.
        self.assertEqual(_count_red_pixels(saved, 2200, 1600, 3400, 2100), 0)

    def test_png_rect_is_outline_only_no_interior_tint(self):
        # The confirmed style is a clean red outline with no translucent
        # interior fill and no dark label bar: well inside the rect (clear of
        # the 8px border and the label above it) the photo stays untouched.
        rects = [{"x": 500, "y": 300, "width": 800, "height": 400}]
        result, dest_path = self._save_as("png", "out.png", rects)
        self.assertTrue(result["ok"])
        saved = cv2.imread(str(dest_path))
        region = saved[300 + 14:700 - 14, 500 + 14:1300 - 14]
        red = (region[:, :, 2] > 180) & (region[:, :, 1] < 90) & (region[:, :, 0] < 90)
        self.assertEqual(int(red.sum()), 0)

    def test_png_with_multiple_rects_bakes_all_of_them(self):
        rects = [
            {"x": 100, "y": 120, "width": 300, "height": 200},
            {"x": 2400, "y": 900, "width": 700, "height": 500},
            {"x": 200, "y": 1500, "width": 400, "height": 300},
        ]
        result, dest_path = self._save_as("png", "out.png", rects)
        self.assertTrue(result["ok"])
        saved = cv2.imread(str(dest_path))
        self.assertEqual(saved.shape[:2], (2160, 3840))
        for rect in rects:
            self.assertGreater(
                _count_red_pixels(
                    saved, rect["x"], rect["y"],
                    rect["x"] + rect["width"], rect["y"] + 8), 0)

    def test_png_out_of_bounds_and_degenerate_rects_clamp_safely(self):
        rects = [
            {"x": -400, "y": -200, "width": 600, "height": 400},  # clipped at origin
            {"x": 10000, "y": 10000, "width": 100, "height": 100},  # fully outside
            {"x": 100, "y": 100, "width": 0, "height": 0},  # zero area -> skipped
            "not-a-rect",  # ignored, not a crash
        ]
        result, dest_path = self._save_as("png", "out.png", rects)
        self.assertTrue(result["ok"])
        saved = cv2.imread(str(dest_path))
        self.assertEqual(saved.shape[:2], (2160, 3840))
        # The rect clamped to the origin still paints there.
        self.assertGreater(_count_red_pixels(saved, 0, 0, 200, 8), 0)

    def test_jpg_with_a_rect_bakes_red_border(self):
        rects = [{"x": 600, "y": 400, "width": 500, "height": 300}]
        result, dest_path = self._save_as("jpg", "out.jpg", rects)
        self.assertTrue(result["ok"])
        saved = cv2.imread(str(dest_path))
        self.assertEqual(saved.shape[:2], (2160, 3840))
        self.assertGreater(_count_red_pixels(saved, 600, 400, 1100, 406), 0)

    def test_jpeg_alias_bakes_the_same_as_jpg(self):
        rects = [{"x": 600, "y": 400, "width": 500, "height": 300}]
        result, dest_path = self._save_as("jpeg", "out.jpeg", rects)
        self.assertTrue(result["ok"])
        saved = cv2.imread(str(dest_path))
        self.assertEqual(saved.shape[:2], (2160, 3840))
        self.assertGreater(_count_red_pixels(saved, 600, 400, 1100, 406), 0)

    def test_jpg_encoding_goes_through_quality_95(self):
        rects = [{"x": 600, "y": 400, "width": 500, "height": 300}]
        with mock.patch("backend.billboard_overlay.cv2.imwrite", return_value=True) as mocked:
            result, _ = self._save_as("jpg", "out.jpg", rects)
        self.assertTrue(result["ok"])
        self.assertEqual(mocked.call_args[0][2], [cv2.IMWRITE_JPEG_QUALITY, 95])

    def test_original_engine_output_is_untouched_after_save_with_rects(self):
        original = self.job.result_path.read_bytes()
        rects = [{"x": 500, "y": 300, "width": 800, "height": 400}]
        result, _ = self._save_as("png", "out.png", rects)
        self.assertTrue(result["ok"])
        self.assertEqual(self.job.result_path.read_bytes(), original)

    def test_empty_rect_list_keeps_byte_identical_png_copy(self):
        result, dest_path = self._save_as("png", "out.png", [])
        self.assertTrue(result["ok"])
        self.assertEqual(dest_path.read_bytes(), self.job.result_path.read_bytes())

    def test_json_string_rects_are_accepted(self):
        # A non-bridge caller stringifying the payload (e.g. JSON.stringify
        # in the browser) must still work.
        payload = json.dumps([{"x": 500, "y": 300, "width": 800, "height": 400}])
        result, dest_path = self._save_as("png", "out.png", payload)
        self.assertTrue(result["ok"])
        saved = cv2.imread(str(dest_path))
        self.assertEqual(saved.shape[:2], (2160, 3840))
        self.assertGreater(_count_red_pixels(saved, 500, 300, 1300, 308), 0)

    def test_non_list_rect_payload_is_treated_as_no_rects(self):
        result, dest_path = self._save_as("png", "out.png", {"not": "a list"})
        self.assertTrue(result["ok"])
        self.assertEqual(dest_path.read_bytes(), self.job.result_path.read_bytes())

    def test_jpg_save_reencodes_and_preserves_dimensions(self):
        result, dest_path = self._save_as("jpg", "out.jpg")
        self.assertTrue(result["ok"])
        self.assertTrue(dest_path.is_file())
        saved = cv2.imread(str(dest_path))
        self.assertEqual(saved.shape[:2], (2160, 3840))
        # JPEG is lossy, so bytes differ from the PNG source, but it must
        # still be a valid, readably-decoded image of the same content.
        self.assertNotEqual(dest_path.read_bytes(), self.job.result_path.read_bytes())

    def test_jpeg_alias_behaves_the_same_as_jpg(self):
        result, dest_path = self._save_as("jpeg", "out.jpeg")
        self.assertTrue(result["ok"])
        saved = cv2.imread(str(dest_path))
        self.assertEqual(saved.shape[:2], (2160, 3840))

    def test_cancelled_dialog_reports_cancelled(self):
        fake_webview = _fake_webview_module(None)
        with mock.patch.dict(sys.modules, {"webview": fake_webview}):
            result = self.bridge.save_result_as(self.job.id, "png")
        self.assertEqual(result, {"ok": False, "cancelled": True})

    def test_unknown_job_id_is_reported(self):
        result = self.bridge.save_result_as("does-not-exist", "png")
        self.assertEqual(result, {"ok": False, "error": "job not found"})


class SaveBatchExportTestCase(unittest.TestCase):
    """Covers save_batch_export — the gallery's native "Save ZIP", which
    builds a fresh export zip (one common format, each image only its own
    rects) rather than copying the job's existing raw result.zip."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

        def _make_file(name, result_id):
            out_path = Path(self._tmp.name) / name
            image = np.full((2160, 3840, 3), 100, dtype=np.uint8)
            cv2.imwrite(str(out_path), image)
            return jobs.FileResult(
                original_filename=name.replace("_out", ""),
                input_path=Path(self._tmp.name) / name.replace("_out", ""),
                result_id=result_id,
                output_path=out_path,
            )

        files = [_make_file("a.png", "000"), _make_file("b.png", "001")]
        self.job = jobs.Job(id="job-batch", files=files)
        self.job.status = jobs.STATUS_COMPLETED
        self.bridge = DesktopBridge(job_manager=_FakeJobManager(self.job))

    def _save(self, images, image_format="png", include_outlines=True, dest_name="out.zip"):
        dest_path = Path(self._tmp.name) / dest_name
        fake_webview = _fake_webview_module(str(dest_path))
        with mock.patch.dict(sys.modules, {"webview": fake_webview}):
            result = self.bridge.save_batch_export(
                self.job.id, image_format, include_outlines, images)
        return result, dest_path

    def test_builds_zip_with_common_format(self):
        images = [
            {"result_id": "000", "rects": []},
            {"result_id": "001", "rects": []},
        ]
        result, dest_path = self._save(images, "jpg")
        self.assertTrue(result["ok"])
        with zipfile.ZipFile(dest_path) as zf:
            self.assertEqual(sorted(zf.namelist()), ["a_enhanced.jpg", "b_enhanced.jpg"])

    def test_each_image_gets_only_its_own_rects(self):
        images = [
            {"result_id": "000", "rects": [{"x": 100, "y": 100, "width": 300, "height": 200}]},
            {"result_id": "001", "rects": []},
        ]
        result, dest_path = self._save(images, "png", include_outlines=True)
        self.assertTrue(result["ok"])
        with zipfile.ZipFile(dest_path) as zf:
            a_img = cv2.imdecode(
                np.frombuffer(zf.read("a_enhanced.png"), np.uint8), cv2.IMREAD_COLOR)
            b_img = cv2.imdecode(
                np.frombuffer(zf.read("b_enhanced.png"), np.uint8), cv2.IMREAD_COLOR)
            self.assertGreater(_count_red_pixels(a_img, 100, 100, 400, 108), 0)
            self.assertEqual(_count_red_pixels(b_img, 100, 100, 400, 108), 0)

    def test_include_outlines_off_produces_clean_images(self):
        images = [
            {"result_id": "000", "rects": [{"x": 100, "y": 100, "width": 300, "height": 200}]},
        ]
        result, dest_path = self._save(
            [images[0]], "png", include_outlines=False, dest_name="clean.zip")
        self.assertTrue(result["ok"])
        with zipfile.ZipFile(dest_path) as zf:
            a_img = cv2.imdecode(
                np.frombuffer(zf.read("a_enhanced.png"), np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(_count_red_pixels(a_img, 100, 100, 400, 108), 0)

    def test_cancelled_dialog_reports_cancelled(self):
        fake_webview = _fake_webview_module(None)
        with mock.patch.dict(sys.modules, {"webview": fake_webview}):
            result = self.bridge.save_batch_export(
                self.job.id, "png", True, [{"result_id": "000", "rects": []}])
        self.assertEqual(result, {"ok": False, "cancelled": True})

    def test_unknown_job_id_is_reported(self):
        result = self.bridge.save_batch_export(
            "does-not-exist", "png", True, [{"result_id": "000", "rects": []}])
        self.assertEqual(result, {"ok": False, "error": "job not found"})

    def test_empty_images_list_is_reported(self):
        result = self.bridge.save_batch_export(self.job.id, "png", True, [])
        self.assertEqual(result, {"ok": False, "error": "no images to export"})

    def test_json_string_images_are_accepted(self):
        payload = json.dumps([{"result_id": "000", "rects": []}])
        result, dest_path = self._save(payload, "png", dest_name="from_json.zip")
        self.assertTrue(result["ok"])
        with zipfile.ZipFile(dest_path) as zf:
            self.assertEqual(zf.namelist(), ["a_enhanced.png"])


if __name__ == "__main__":
    unittest.main()
