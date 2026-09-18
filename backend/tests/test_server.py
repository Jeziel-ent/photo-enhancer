"""HTTP-level tests for the Application/API layer.

Spins up a real backend.server.ApiHandler server on an ephemeral port with a
fake JobManager.process_fn (no GPU/engine involved) and drives it with real
HTTP requests — this proves the wire contract (status codes, JSON shape,
multipart handling, result download) independently of enhancement quality,
which is covered by image_enhancer/tests instead.
"""

import json
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from backend import jobs as jobs_module
from backend.jobs import JobManager
from backend.server import make_server
from backend.tests.multipart_helpers import build_multipart_body


def _fake_process(input_path, output_path):
    if "bad" in input_path.name:
        raise RuntimeError("simulated corrupt image")
    output_path.write_bytes(b"PNGDATA:" + input_path.read_bytes())


class ApiServerTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls._workspace_patch = mock.patch.object(
            jobs_module.workspace, "WORKSPACE_ROOT", Path(cls._tmp.name))
        cls._workspace_patch.start()

        cls.manager = JobManager(process_fn=_fake_process)
        cls.httpd = make_server(port=0, job_manager=cls.manager)
        cls.base_url = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=5)
        cls._workspace_patch.stop()
        cls._tmp.cleanup()

    def _get(self, path):
        req = urllib.request.Request(self.base_url + path, method="GET")
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, resp.read(), dict(resp.headers)
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read(), dict(exc.headers)

    def _post_multipart(self, path, files, extra_fields=None):
        content_type, body = build_multipart_body(files, extra_fields)
        req = urllib.request.Request(
            self.base_url + path, data=body, method="POST",
            headers={"Content-Type": content_type})
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, resp.read(), dict(resp.headers)
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read(), dict(exc.headers)

    def _wait_for_status(self, job_id, target_statuses, timeout=5.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            status, body, _ = self._get(f"/api/jobs/{job_id}")
            data = json.loads(body)
            if data.get("status") in target_statuses:
                return data
            time.sleep(0.02)
        self.fail(f"job {job_id} never reached {target_statuses}")

    # ---------------------------------------------------------------- health
    def test_health(self):
        status, body, _ = self._get("/health")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {"ok": True})

    # ------------------------------------------------------------- job create
    def test_create_job_single_file_and_download_result(self):
        status, body, _ = self._post_multipart(
            "/api/jobs", [("files", "photo.jpg", "image/jpeg", b"\xff\xd8raw")])
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertTrue(data["ok"])
        self.assertEqual(data["total_count"], 1)
        job_id = data["job_id"]

        final = self._wait_for_status(job_id, {"completed", "failed"})
        self.assertEqual(final["status"], "completed")
        self.assertEqual(final["completed_count"], 1)
        self.assertEqual(final["errors"], [])

        status, body, headers = self._get(f"/api/jobs/{job_id}/result")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "image/png")
        self.assertIn("photo_enhanced.png", headers["Content-Disposition"])
        self.assertEqual(body, b"PNGDATA:\xff\xd8raw")

    def test_create_job_multiple_files_and_download_zip(self):
        status, body, _ = self._post_multipart("/api/jobs", [
            ("files", "one.jpg", "image/jpeg", b"111"),
            ("files", "two.png", "image/png", b"222"),
        ])
        self.assertEqual(status, 200)
        job_id = json.loads(body)["job_id"]

        final = self._wait_for_status(job_id, {"completed", "failed"})
        self.assertEqual(final["status"], "completed")
        self.assertEqual(final["completed_count"], 2)

        status, body, headers = self._get(f"/api/jobs/{job_id}/result")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "application/zip")
        with zipfile.ZipFile(BytesIO(body)) as zf:
            self.assertEqual(
                sorted(zf.namelist()), ["one_enhanced.png", "two_enhanced.png"])

    def test_partial_failure_reported_in_status(self):
        status, body, _ = self._post_multipart("/api/jobs", [
            ("files", "good.jpg", "image/jpeg", b"111"),
            ("files", "bad.jpg", "image/jpeg", b"222"),
        ])
        job_id = json.loads(body)["job_id"]
        final = self._wait_for_status(job_id, {"completed", "failed"})
        self.assertEqual(final["status"], "completed")
        self.assertEqual(len(final["errors"]), 1)
        self.assertEqual(final["errors"][0]["filename"], "bad.jpg")

    # ----------------------------------------------------------- validation
    def test_create_job_with_no_files_is_rejected(self):
        status, body, _ = self._post_multipart(
            "/api/jobs", [], extra_fields={"note": "no files here"})
        self.assertEqual(status, 400)
        self.assertFalse(json.loads(body)["ok"])

    def test_create_job_with_unsupported_extension_is_rejected(self):
        status, body, _ = self._post_multipart(
            "/api/jobs", [("files", "doc.txt", "text/plain", b"not an image")])
        self.assertEqual(status, 400)
        self.assertIn("unsupported file type", json.loads(body)["error"])

    def test_create_job_with_empty_file_is_rejected(self):
        status, body, _ = self._post_multipart(
            "/api/jobs", [("files", "empty.jpg", "image/jpeg", b"")])
        self.assertEqual(status, 400)

    def test_non_multipart_post_is_rejected(self):
        req = urllib.request.Request(
            self.base_url + "/api/jobs", data=b"{}", method="POST",
            headers={"Content-Type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req)
        self.assertEqual(ctx.exception.code, 400)

    # ------------------------------------------------------------- lookups
    def test_get_status_for_unknown_job_is_404(self):
        status, body, _ = self._get("/api/jobs/does-not-exist")
        self.assertEqual(status, 404)

    def test_get_result_for_unknown_job_is_404(self):
        status, body, _ = self._get("/api/jobs/does-not-exist/result")
        self.assertEqual(status, 404)

    def test_get_result_before_completion_is_409(self):
        # A dedicated server + manager whose process_fn blocks, so the job is
        # still queued/processing when we hit GET .../result for real.
        release = threading.Event()

        def slow_process(input_path, output_path):
            release.wait(timeout=5)
            output_path.write_bytes(b"ok")

        manager = JobManager(process_fn=slow_process)
        httpd = make_server(port=0, job_manager=manager)
        base_url = f"http://127.0.0.1:{httpd.server_address[1]}"
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            content_type, body = build_multipart_body(
                [("files", "x.jpg", "image/jpeg", b"1")])
            req = urllib.request.Request(
                base_url + "/api/jobs", data=body, method="POST",
                headers={"Content-Type": content_type})
            with urllib.request.urlopen(req) as resp:
                job_id = json.loads(resp.read())["job_id"]

            deadline = time.time() + 5
            while time.time() < deadline:
                status_req = urllib.request.Request(f"{base_url}/api/jobs/{job_id}")
                with urllib.request.urlopen(status_req) as resp:
                    if json.loads(resp.read())["status"] in ("queued", "processing"):
                        break
                time.sleep(0.01)

            result_req = urllib.request.Request(f"{base_url}/api/jobs/{job_id}/result")
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(result_req)
            self.assertEqual(ctx.exception.code, 409)
        finally:
            release.set()
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=5)

    def test_get_result_with_billboard_composites_rectangles_onto_png(self):
        status, body, _ = self._post_multipart(
            "/api/jobs", [("files", "photo.jpg", "image/jpeg", b"\xff\xd8raw")])
        self.assertEqual(status, 200)
        job_id = json.loads(body)["job_id"]
        self._wait_for_status(job_id, {"completed"})

        # Override the fake-process result with a real decodeable PNG so the
        # compositing route has actual pixels to work with.
        real_png = Path(self._tmp.name) / "real.png"
        cv2.imwrite(str(real_png), np.full((216, 384, 3), 100, np.uint8))
        job = self.manager.get_job(job_id)
        job.result_path = real_png
        job.result_filename = "photo_enhanced.png"

        rects = [{"x": 40, "y": 30, "width": 120, "height": 80}]
        query = urllib.parse.quote(json.dumps(rects))
        status, body, headers = self._get(f"/api/jobs/{job_id}/result?billboard={query}")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "image/png")
        decoded = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded.shape[:2], (216, 384))
        region = decoded[30:38, 40:160]
        red = (region[:, :, 2] > 180) & (region[:, :, 1] < 90) & (region[:, :, 0] < 90)
        self.assertGreater(int(red.sum()), 0)

    def test_get_result_with_adjust_applies_brightness(self):
        status, body, _ = self._post_multipart(
            "/api/jobs", [("files", "photo.jpg", "image/jpeg", b"\xff\xd8raw")])
        self.assertEqual(status, 200)
        job_id = json.loads(body)["job_id"]
        self._wait_for_status(job_id, {"completed"})

        real_png = Path(self._tmp.name) / "real_adjust.png"
        cv2.imwrite(str(real_png), np.full((100, 100, 3), 120, np.uint8))
        job = self.manager.get_job(job_id)
        job.result_path = real_png
        job.result_filename = "photo_enhanced.png"

        query = urllib.parse.quote(json.dumps({"brightness": 60}))
        status, body, headers = self._get(f"/api/jobs/{job_id}/result?adjust={query}")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "image/png")
        decoded = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
        self.assertIsNotNone(decoded)
        self.assertGreater(int(decoded.mean()), 120)

    def test_get_result_with_default_adjust_is_noop(self):
        status, body, _ = self._post_multipart(
            "/api/jobs", [("files", "photo.jpg", "image/jpeg", b"\xff\xd8raw")])
        self.assertEqual(status, 200)
        job_id = json.loads(body)["job_id"]
        self._wait_for_status(job_id, {"completed"})

        query = urllib.parse.quote(json.dumps({"brightness": 0, "detail": 0}))
        status, body, headers = self._get(f"/api/jobs/{job_id}/result?adjust={query}")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "image/png")
        self.assertEqual(body, b"PNGDATA:\xff\xd8raw")

    def test_get_individual_result_with_adjust_applies_brightness(self):
        job_id, final = self._create_batch_job([
            ("files", "one.jpg", "image/jpeg", b"111"),
        ])
        result_id = final["results"][0]["id"]
        file_result = self.manager.get_job(job_id).get_file_result(result_id)
        real_png = Path(self._tmp.name) / "real_gallery_adjust.png"
        cv2.imwrite(str(real_png), np.full((80, 80, 3), 120, np.uint8))
        file_result.output_path = real_png

        query = urllib.parse.quote(json.dumps({"brightness": 60}))
        status, body, headers = self._get(
            f"/api/jobs/{job_id}/results/{result_id}?adjust={query}")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "image/png")
        decoded = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
        self.assertIsNotNone(decoded)
        self.assertGreater(int(decoded.mean()), 120)

    def test_get_result_with_invalid_billboard_query_falls_back_to_raw_png(self):
        status, body, _ = self._post_multipart(
            "/api/jobs", [("files", "photo.jpg", "image/jpeg", b"\xff\xd8raw")])
        self.assertEqual(status, 200)
        job_id = json.loads(body)["job_id"]
        self._wait_for_status(job_id, {"completed"})

        query = urllib.parse.quote("not-json")
        status, body, headers = self._get(
            f"/api/jobs/{job_id}/result?billboard={query}")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "image/png")
        self.assertEqual(body, b"PNGDATA:\xff\xd8raw")

    def test_unknown_route_is_404(self):
        status, _, _ = self._get("/api/nope")
        self.assertEqual(status, 404)

    # -------------------------------------------------------- gallery/batch
    def _post_json(self, path, payload):
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + path, data=body, method="POST",
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, resp.read(), dict(resp.headers)
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read(), dict(exc.headers)

    def _create_batch_job(self, files):
        status, body, _ = self._post_multipart("/api/jobs", files)
        self.assertEqual(status, 200)
        job_id = json.loads(body)["job_id"]
        final = self._wait_for_status(job_id, {"completed", "failed"})
        self.assertEqual(final["status"], "completed")
        return job_id, final

    def test_status_includes_per_file_results_in_upload_order(self):
        job_id, final = self._create_batch_job([
            ("files", "one.jpg", "image/jpeg", b"111"),
            ("files", "two.png", "image/png", b"222"),
        ])
        self.assertEqual(
            final["results"],
            [{"id": "000", "filename": "one.jpg"}, {"id": "001", "filename": "two.png"}],
        )

    def test_status_results_excludes_failed_files(self):
        job_id, final = self._create_batch_job([
            ("files", "good.jpg", "image/jpeg", b"111"),
            ("files", "bad.jpg", "image/jpeg", b"222"),
        ])
        self.assertEqual(final["results"], [{"id": "000", "filename": "good.jpg"}])

    def test_get_individual_result_returns_that_files_own_png(self):
        job_id, _ = self._create_batch_job([
            ("files", "one.jpg", "image/jpeg", b"111"),
            ("files", "two.png", "image/png", b"222"),
        ])
        status, body, headers = self._get(f"/api/jobs/{job_id}/results/001")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "image/png")
        self.assertEqual(body, b"PNGDATA:222")

    def test_get_individual_result_unknown_id_is_404(self):
        job_id, _ = self._create_batch_job([
            ("files", "one.jpg", "image/jpeg", b"111"),
        ])
        status, _, _ = self._get(f"/api/jobs/{job_id}/results/999")
        self.assertEqual(status, 404)

    def test_get_individual_result_unknown_job_is_404(self):
        status, _, _ = self._get("/api/jobs/does-not-exist/results/000")
        self.assertEqual(status, 404)

    def test_export_batch_returns_zip_with_common_format(self):
        job_id, _ = self._create_batch_job([
            ("files", "one.jpg", "image/jpeg", b"111"),
            ("files", "two.png", "image/png", b"222"),
        ])
        # Swap in real decodeable PNGs so the export route has real pixels.
        job = self.manager.get_job(job_id)
        for f in job.files:
            real_png = Path(self._tmp.name) / f"{f.result_id}.png"
            cv2.imwrite(str(real_png), np.full((216, 384, 3), 100, np.uint8))
            f.output_path = real_png

        status, body, headers = self._post_json(f"/api/jobs/{job_id}/export", {
            "format": "jpg",
            "include_outlines": True,
            "images": [
                {"result_id": "000", "rects": [{"x": 10, "y": 10, "width": 50, "height": 40}]},
                {"result_id": "001", "rects": []},
            ],
        })
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "application/zip")
        with zipfile.ZipFile(BytesIO(body)) as zf:
            self.assertEqual(
                sorted(zf.namelist()), ["one_enhanced.jpg", "two_enhanced.jpg"])
            one = cv2.imdecode(
                np.frombuffer(zf.read("one_enhanced.jpg"), np.uint8), cv2.IMREAD_COLOR)
            two = cv2.imdecode(
                np.frombuffer(zf.read("two_enhanced.jpg"), np.uint8), cv2.IMREAD_COLOR)
            one_red = (one[:, :, 2] > 180) & (one[:, :, 1] < 90) & (one[:, :, 0] < 90)
            two_red = (two[:, :, 2] > 180) & (two[:, :, 1] < 90) & (two[:, :, 0] < 90)
            self.assertGreater(int(one_red.sum()), 0)
            self.assertEqual(int(two_red.sum()), 0)

    def test_export_batch_applies_common_adjust_to_every_image(self):
        job_id, _ = self._create_batch_job([
            ("files", "one.jpg", "image/jpeg", b"111"),
            ("files", "two.png", "image/png", b"222"),
        ])
        job = self.manager.get_job(job_id)
        for f in job.files:
            real_png = Path(self._tmp.name) / f"{f.result_id}_adj.png"
            cv2.imwrite(str(real_png), np.full((80, 80, 3), 120, np.uint8))
            f.output_path = real_png

        status, body, headers = self._post_json(f"/api/jobs/{job_id}/export", {
            "format": "png",
            "include_outlines": False,
            "adjust": {"brightness": 60},
            "images": [
                {"result_id": "000", "rects": []},
                {"result_id": "001", "rects": []},
            ],
        })
        self.assertEqual(status, 200)
        with zipfile.ZipFile(BytesIO(body)) as zf:
            for name in ("one_enhanced.png", "two_enhanced.png"):
                decoded = cv2.imdecode(
                    np.frombuffer(zf.read(name), np.uint8), cv2.IMREAD_COLOR)
                self.assertGreater(int(decoded.mean()), 120)

    def test_export_batch_invalid_format_is_400(self):
        job_id, _ = self._create_batch_job([
            ("files", "one.jpg", "image/jpeg", b"111"),
        ])
        status, body, _ = self._post_json(f"/api/jobs/{job_id}/export", {
            "format": "bmp",
            "include_outlines": False,
            "images": [{"result_id": "000", "rects": []}],
        })
        self.assertEqual(status, 400)
        self.assertFalse(json.loads(body)["ok"])

    def test_export_batch_unknown_result_id_is_404(self):
        job_id, _ = self._create_batch_job([
            ("files", "one.jpg", "image/jpeg", b"111"),
        ])
        status, _, _ = self._post_json(f"/api/jobs/{job_id}/export", {
            "format": "png",
            "include_outlines": False,
            "images": [{"result_id": "999", "rects": []}],
        })
        self.assertEqual(status, 404)

    # ------------------------------------------ MVP 2: non-16:9 aspect ratio
    # Board coordinates/overlay export are handled entirely by
    # backend/billboard_overlay.py, which reads the real image's own
    # cv2.imread shape at compose time -- it was already resolution/aspect
    # agnostic before MVP 2 (only the GPU/CPU enhancement pipeline's own
    # hardcoded 3840x2160 target needed fixing, see docs/MVP2_RESEARCH.md
    # Phase 1). These tests prove that HTTP-layer contract holds for a
    # genuinely non-16:9 "enhanced" result (a 3840x3840 square, as a 1:1
    # input would now produce), not just the classic 16:9 box every other
    # test in this file already uses.
    def test_billboard_composite_maps_correctly_on_square_result(self):
        status, body, _ = self._post_multipart(
            "/api/jobs", [("files", "square.jpg", "image/jpeg", b"\xff\xd8raw")])
        job_id = json.loads(body)["job_id"]
        self._wait_for_status(job_id, {"completed"})

        # Swap in a real, decodeable SQUARE PNG (not 16:9) -- what a 1:1
        # input now produces end to end.
        square_png = Path(self._tmp.name) / "square_real.png"
        cv2.imwrite(str(square_png), np.full((400, 400, 3), 100, np.uint8))
        job = self.manager.get_job(job_id)
        job.result_path = square_png
        job.result_filename = "square_enhanced.png"

        # A rect anchored near the bottom-right, which would be OUT OF
        # BOUNDS on a 16:9 assumption at this same pixel size but is valid
        # on the real 400x400 square.
        rects = [{"x": 300, "y": 300, "width": 80, "height": 80}]
        query = urllib.parse.quote(json.dumps(rects))
        status, body, headers = self._get(f"/api/jobs/{job_id}/result?billboard={query}")
        self.assertEqual(status, 200)
        decoded = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape[:2], (400, 400))  # dimensions preserved, not stretched
        region = decoded[300:308, 300:380]
        red = (region[:, :, 2] > 180) & (region[:, :, 1] < 90) & (region[:, :, 0] < 90)
        self.assertGreater(int(red.sum()), 0)

    def test_batch_export_preserves_non_16_9_dimensions_per_image(self):
        job_id, _ = self._create_batch_job([
            ("files", "square.jpg", "image/jpeg", b"111"),
            ("files", "portrait.png", "image/png", b"222"),
        ])
        job = self.manager.get_job(job_id)
        square_path = Path(self._tmp.name) / "sq.png"
        portrait_path = Path(self._tmp.name) / "pt.png"
        cv2.imwrite(str(square_path), np.full((320, 320, 3), 100, np.uint8))   # 1:1
        cv2.imwrite(str(portrait_path), np.full((480, 270, 3), 100, np.uint8))  # 9:16-ish (H x W)
        job.files[0].output_path = square_path
        job.files[1].output_path = portrait_path

        status, body, headers = self._post_json(f"/api/jobs/{job_id}/export", {
            "format": "png",
            "include_outlines": False,
            "images": [
                {"result_id": "000", "rects": []},
                {"result_id": "001", "rects": []},
            ],
        })
        self.assertEqual(status, 200)
        with zipfile.ZipFile(BytesIO(body)) as zf:
            names = sorted(zf.namelist())
            self.assertEqual(names, ["portrait_enhanced.png", "square_enhanced.png"])
            sq = cv2.imdecode(np.frombuffer(zf.read("square_enhanced.png"), np.uint8), cv2.IMREAD_COLOR)
            pt = cv2.imdecode(np.frombuffer(zf.read("portrait_enhanced.png"), np.uint8), cv2.IMREAD_COLOR)
            # Neither image was stretched/cropped/resized by the export path --
            # each keeps its own real (non-16:9) dimensions from the engine.
            self.assertEqual(sq.shape[:2], (320, 320))
            self.assertEqual(pt.shape[:2], (480, 270))

    def test_export_batch_unknown_job_is_404(self):
        status, _, _ = self._post_json("/api/jobs/does-not-exist/export", {
            "format": "png",
            "include_outlines": False,
            "images": [{"result_id": "000", "rects": []}],
        })
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()
