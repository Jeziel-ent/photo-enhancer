"""REAL end-to-end integration tests for the Application/API layer.

Unlike backend/tests/ (fast, mocked, no GPU), these drive the actual
production path: POST /api/jobs -> JobManager's real worker thread ->
engine_adapter.enhance_image -> image_enhancer/src/enhance.py's "final"
method on the real GPU -> GET the result back over real HTTP -- using the
real sample photos in image_enhancer/originals/.

See README.md in this directory for how to run this (opt-in, GPU required,
several minutes).
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
ORIGINALS_DIR = REPO_ROOT / "image_enhancer" / "originals"
IMAGE_ENHANCER_SRC = REPO_ROOT / "image_enhancer" / "src"

RUN_GPU_TESTS = os.environ.get("ADINN_RUN_GPU_TESTS") == "1"

from backend import engine_adapter  # noqa: E402
from backend.jobs import JobManager  # noqa: E402
from backend.server import make_server  # noqa: E402
from backend.tests.multipart_helpers import build_multipart_body  # noqa: E402

EXPECTED_W, EXPECTED_H = 3840, 2160


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _decode_and_check(data: bytes, label: str) -> None:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise AssertionError(f"{label}: could not decode as an image")
    h, w = img.shape[:2]
    if (w, h) != (EXPECTED_W, EXPECTED_H):
        raise AssertionError(f"{label}: expected {EXPECTED_W}x{EXPECTED_H}, got {w}x{h}")
    std = float(img.std())
    if std < 5.0:
        raise AssertionError(f"{label}: output looks blank/degenerate (std={std:.2f})")


class _LoopbackOnlyNetworkGuard:
    """Raises if anything in this process opens a socket to a non-loopback
    address -- a dynamic proof that a real enhancement run makes no
    external network / AI API calls. Our own test traffic (HTTP client <->
    the local server) is loopback and stays allowed."""

    _ALLOWED_HOSTS = {"127.0.0.1", "::1", "localhost"}

    def __enter__(self):
        self._orig_connect = socket.socket.connect
        allowed = self._ALLOWED_HOSTS
        orig = self._orig_connect

        def guarded_connect(sock_self, address):
            host = address[0] if isinstance(address, tuple) else address
            if host not in allowed:
                raise RuntimeError(
                    f"blocked a non-loopback network connection to {address!r} "
                    "-- the enhancement pipeline must never call out")
            return orig(sock_self, address)

        socket.socket.connect = guarded_connect
        return self

    def __exit__(self, *exc):
        socket.socket.connect = self._orig_connect


def _http_get(base_url, path):
    req = urllib.request.Request(base_url + path, method="GET")
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, resp.read(), dict(resp.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), dict(exc.headers)


def _http_post_job(base_url, files):
    content_type, body = build_multipart_body(files)
    req = urllib.request.Request(
        base_url + "/api/jobs", data=body, method="POST",
        headers={"Content-Type": content_type})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def _poll_until_done(base_url, job_id, timeout, progress_log):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        _status, body, _headers = _http_get(base_url, f"/api/jobs/{job_id}")
        data = json.loads(body)
        if data != last:
            progress_log.append((time.time(), dict(data)))
            last = data
        if data["status"] in ("completed", "failed"):
            return data
        time.sleep(0.25)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s (last={last})")


@unittest.skipUnless(RUN_GPU_TESTS, "set ADINN_RUN_GPU_TESTS=1 to run the real GPU pipeline")
class RealPipelineIntegrationTestCase(unittest.TestCase):
    """Uses the real, unmocked production wiring: JobManager() with its
    default (real) process_fn, served over a real HTTP server."""

    @classmethod
    def setUpClass(cls):
        cls._network_guard = _LoopbackOnlyNetworkGuard()
        cls._network_guard.__enter__()
        cls.manager = JobManager()  # default process_fn == engine_adapter.enhance_image
        cls.httpd = make_server(port=0, job_manager=cls.manager)
        cls.base_url = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=5)
        cls._network_guard.__exit__(None, None, None)

    # ---------------------------------------------------------------- tests

    def test_single_image_real_pipeline_and_no_billboard_leak(self):
        """Single real image through the real default wiring: PNG result,
        correct dimensions, original untouched, progress correct -- and,
        using 3.jpeg (which has a REAL recorded box in
        image_enhancer/regions.json), proves no billboard-region
        reconstruction leaks into this generic upload. Wraps the real
        enhance.METHODS["final"] to observe return_stages=True output
        without altering enhance.py or changing what actually runs.
        """
        sys.path.insert(0, str(IMAGE_ENHANCER_SRC))
        import enhance

        src = ORIGINALS_DIR / "3.jpeg"
        original_hash_before = _sha256(src)

        captured_boxes = []
        original_final = enhance.final_enhance

        def spying_final_enhance(img, target=(enhance.OUT_W, enhance.OUT_H), return_stages=False):
            out, stages = original_final(img, target, return_stages=True)
            captured_boxes.append(stages.get("boxes4"))
            return (out, stages) if return_stages else out

        enhance.METHODS["final"] = spying_final_enhance
        try:
            t0 = time.time()
            created = _http_post_job(self.base_url, [
                ("files", src.name, "image/jpeg", src.read_bytes()),
            ])
            job_id = created["job_id"]
            self.assertEqual(created["total_count"], 1)

            progress_log = []
            final = _poll_until_done(self.base_url, job_id, timeout=240, progress_log=progress_log)
            elapsed = time.time() - t0
        finally:
            enhance.METHODS["final"] = original_final

        self.assertEqual(final["status"], "completed")
        self.assertEqual(final["completed_count"], 1)
        self.assertEqual(final["total_count"], 1)
        self.assertEqual(final["errors"], [])
        self.assertEqual(final["progress"], 1.0)
        counts = [entry["completed_count"] for _, entry in progress_log]
        self.assertEqual(counts, sorted(counts), "completed_count must never go backwards")

        self.assertEqual(len(captured_boxes), 1)
        self.assertEqual(
            captured_boxes[0], [],
            "billboard boxes leaked into a generic upload of an image with a "
            "real regions.json entry -- engine_adapter neutralization regressed")

        status, body, headers = _http_get(self.base_url, f"/api/jobs/{job_id}/result")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "image/png")
        self.assertTrue(body.startswith(b"\x89PNG\r\n\x1a\n"), "result is not a valid PNG")
        _decode_and_check(body, "single-image result")

        self.assertEqual(_sha256(src), original_hash_before, "original was modified on disk")

        print(f"\n[single:{src.name}] elapsed={elapsed:.1f}s output={len(body)} bytes, "
              f"billboard_boxes={captured_boxes[0]}")

    def test_partial_failure_with_real_engine(self):
        """One real good image + one corrupt upload, through the real
        (unmocked) engine_adapter -- proves partial-failure handling holds
        against the actual engine's real exceptions, not a fake's."""
        good = ORIGINALS_DIR / "5.jpeg"
        good_hash_before = _sha256(good)

        t0 = time.time()
        created = _http_post_job(self.base_url, [
            ("files", good.name, "image/jpeg", good.read_bytes()),
            ("files", "corrupt.jpg", "image/jpeg", b"\xff\xd8not-a-real-jpeg-body"),
        ])
        job_id = created["job_id"]
        self.assertEqual(created["total_count"], 2)

        final = _poll_until_done(self.base_url, job_id, timeout=240, progress_log=[])
        elapsed = time.time() - t0

        self.assertEqual(final["status"], "completed")
        self.assertEqual(final["completed_count"], 2)
        self.assertEqual(len(final["errors"]), 1)
        self.assertEqual(final["errors"][0]["filename"], "corrupt.jpg")

        status, body, headers = _http_get(self.base_url, f"/api/jobs/{job_id}/result")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "application/zip")
        self.assertTrue(body.startswith(b"PK"), "result is not a valid ZIP")
        with zipfile.ZipFile(BytesIO(body)) as zf:
            entries = zf.namelist()
            self.assertEqual(len(entries), 1, "zip must contain only the surviving image")
            _decode_and_check(zf.read(entries[0]), f"partial-failure surviving entry {entries[0]}")

        self.assertEqual(_sha256(good), good_hash_before)
        print(f"\n[partial-failure] elapsed={elapsed:.1f}s errors={final['errors']}")


@unittest.skipUnless(RUN_GPU_TESTS, "set ADINN_RUN_GPU_TESTS=1 to run the real GPU pipeline")
class RealMultiImageSequentialTestCase(unittest.TestCase):
    """A dedicated server/manager instrumented to time each real
    engine_adapter.enhance_image call, proving multi-file jobs really are
    processed sequentially against the real GPU (not just against a fast
    fake, which backend/tests/test_jobs.py already covers)."""

    def test_multiple_images_real_pipeline_sequential(self):
        names = ["1.jpeg", "2.jpeg", "4.jpeg"]
        sources = [ORIGINALS_DIR / n for n in names]
        hashes_before = {p: _sha256(p) for p in sources}

        intervals: list[tuple[float, float, str]] = []
        intervals_lock = threading.Lock()

        def timed_enhance_image(input_path, output_path):
            start = time.time()
            engine_adapter.enhance_image(input_path, output_path)  # the real thing
            end = time.time()
            with intervals_lock:
                intervals.append((start, end, input_path.name))

        with _LoopbackOnlyNetworkGuard():
            manager = JobManager(process_fn=timed_enhance_image)
            httpd = make_server(port=0, job_manager=manager)
            base_url = f"http://127.0.0.1:{httpd.server_address[1]}"
            thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            try:
                t0 = time.time()
                created = _http_post_job(base_url, [
                    ("files", p.name, "image/jpeg", p.read_bytes()) for p in sources
                ])
                job_id = created["job_id"]
                self.assertEqual(created["total_count"], 3)

                progress_log = []
                final = _poll_until_done(base_url, job_id, timeout=600, progress_log=progress_log)
                elapsed = time.time() - t0

                status, body, headers = _http_get(base_url, f"/api/jobs/{job_id}/result")
            finally:
                httpd.shutdown()
                httpd.server_close()
                thread.join(timeout=5)

        self.assertEqual(final["status"], "completed")
        self.assertEqual(final["completed_count"], 3)
        self.assertEqual(final["errors"], [])
        seen_files = {entry["current_file"] for _, entry in progress_log if entry["current_file"]}
        self.assertEqual(seen_files, set(names), "every uploaded file should appear as current_file")
        counts = [entry["completed_count"] for _, entry in progress_log]
        self.assertEqual(counts, sorted(counts))

        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "application/zip")
        self.assertTrue(body.startswith(b"PK"))
        with zipfile.ZipFile(BytesIO(body)) as zf:
            entries = sorted(zf.namelist())
            self.assertEqual(len(entries), 3)
            for name in entries:
                _decode_and_check(zf.read(name), f"zip entry {name}")

        self.assertEqual(len(intervals), 3)
        intervals.sort()
        for (s1, e1, n1), (s2, e2, _n2) in zip(intervals, intervals[1:]):
            self.assertLessEqual(e1, s2, f"{n1} overlapped with the next file: {intervals}")

        for p in sources:
            self.assertEqual(_sha256(p), hashes_before[p], f"{p.name} was modified on disk")

        per_file = ", ".join(f"{n}={e - s:.1f}s" for s, e, n in intervals)
        print(f"\n[batch-3] total_elapsed={elapsed:.1f}s per_file=({per_file}) "
              f"entries={entries}")


if __name__ == "__main__":
    unittest.main()
