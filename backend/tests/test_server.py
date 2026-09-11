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
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path
from unittest import mock

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

    def test_unknown_route_is_404(self):
        status, _, _ = self._get("/api/nope")
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()
