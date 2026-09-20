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

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _expected_wh_for(src_path: Path) -> tuple:
    """The real aspect-preserving 4K target for this exact source image --
    computed from the SAME production function the engine itself uses
    (enhance.aspect_preserving_target), not a fixed 3840x2160 assumption.

    Found while writing the CPU-path regression suite
    (test_cpu_pipeline.py): several of the originals used by this file
    (2/3/5.jpeg) do NOT reduce to an exact 16:9 ratio at a 3840 width --
    e.g. 3.jpeg's real target is 3840x2162, 5.jpeg's is 3840x2158 -- so a
    fixed EXPECTED_W/EXPECTED_H constant (this file's prior approach,
    apparently never actually exercised against the current
    aspect-preserving engine) would silently fail these exact tests. This
    computes the real per-image target instead."""
    sys.path.insert(0, str(IMAGE_ENHANCER_SRC))
    import enhance
    img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
    return enhance.aspect_preserving_target(img.shape[1], img.shape[0])


def _decode_and_check(data: bytes, expected_wh: tuple, label: str) -> None:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise AssertionError(f"{label}: could not decode as an image")
    h, w = img.shape[:2]
    if (w, h) != tuple(expected_wh):
        raise AssertionError(
            f"{label}: expected {expected_wh[0]}x{expected_wh[1]} "
            f"(aspect-preserving 4K target), got {w}x{h}")
    if max(w, h) != 3840:
        raise AssertionError(f"{label}: 4K-longest-side target not met (max dim {max(w, h)})")
    arrf = img.astype(np.float64)
    if not np.all(np.isfinite(arrf)):
        raise AssertionError(f"{label}: output contains NaN/Inf pixels")
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
        _decode_and_check(body, _expected_wh_for(src), "single-image result")

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
            _decode_and_check(zf.read(entries[0]), _expected_wh_for(good),
                              f"partial-failure surviving entry {entries[0]}")

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
        expected_by_stem = {p.stem: _expected_wh_for(p) for p in sources}
        with zipfile.ZipFile(BytesIO(body)) as zf:
            entries = sorted(zf.namelist())
            self.assertEqual(len(entries), 3)
            for name in entries:
                stem = next(s for s in expected_by_stem if name.startswith(s))
                _decode_and_check(zf.read(name), expected_by_stem[stem], f"zip entry {name}")

        self.assertEqual(len(intervals), 3)
        intervals.sort()
        for (s1, e1, n1), (s2, e2, _n2) in zip(intervals, intervals[1:]):
            self.assertLessEqual(e1, s2, f"{n1} overlapped with the next file: {intervals}")

        for p in sources:
            self.assertEqual(_sha256(p), hashes_before[p], f"{p.name} was modified on disk")

        per_file = ", ".join(f"{n}={e - s:.1f}s" for s, e, n in intervals)
        print(f"\n[batch-3] total_elapsed={elapsed:.1f}s per_file=({per_file}) "
              f"entries={entries}")


@unittest.skipUnless(RUN_GPU_TESTS, "set ADINN_RUN_GPU_TESTS=1 to run the real GPU pipeline")
class RealGpuDeviceAndRepeatedJobsTestCase(unittest.TestCase):
    """Forced-GPU device selection + a dedicated repeated-jobs stability
    run (cold-start vs. warm timing, peak VRAM, CUDA stability across many
    consecutive real jobs) -- the existing tests above exercise auto/GPU
    incidentally but don't isolate these specifically."""

    @classmethod
    def setUpClass(cls):
        from backend import settings
        cls._prior_device_preference = settings.get_processing_device_preference()
        settings.set_processing_device_preference("gpu")

        cls._network_guard = _LoopbackOnlyNetworkGuard()
        cls._network_guard.__enter__()
        cls.manager = JobManager()
        cls.httpd = make_server(port=0, job_manager=cls.manager)
        cls.base_url = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

        # Force one job through so JobManager._run()'s apply_device_preference
        # + warmup_engine() actually run (lazy, on first real job) before any
        # assertion on get_device_state().
        created = _http_post_job(cls.base_url, [
            ("files", "2.jpeg", "image/jpeg", (ORIGINALS_DIR / "2.jpeg").read_bytes()),
        ])
        _poll_until_done(cls.base_url, created["job_id"], timeout=240, progress_log=[])
        state = engine_adapter.get_device_state()
        assert state["effective_device"] == "gpu", (
            f"setUpClass warmup job did not run on GPU (state={state})")

    @classmethod
    def tearDownClass(cls):
        from backend import settings
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=5)
        cls._network_guard.__exit__(None, None, None)
        settings.set_processing_device_preference(cls._prior_device_preference)

    def test_forced_gpu_preference_resolves_to_gpu(self):
        state = engine_adapter.get_device_state()
        self.assertEqual(state["preference"], "gpu")
        self.assertEqual(state["effective_device"], "gpu")
        self.assertIsNotNone(state["detected_gpu"])

    def test_5_sequential_and_10_repeated_gpu_jobs(self):
        """5 sequential single-image GPU jobs across different real
        photos, then 10 repeated jobs on the same image to isolate warm
        per-job timing and CUDA/VRAM stability from per-image variance."""
        import torch

        def run_one(src: Path, timeout=240) -> float:
            expected_wh = _expected_wh_for(src)
            hash_before = _sha256(src)
            t0 = time.time()
            created = _http_post_job(self.base_url, [
                ("files", src.name, "image/jpeg", src.read_bytes()),
            ])
            final = _poll_until_done(self.base_url, created["job_id"], timeout=timeout, progress_log=[])
            elapsed = time.time() - t0
            self.assertEqual(final["status"], "completed")
            self.assertEqual(final["errors"], [])
            status, body, headers = _http_get(self.base_url, f"/api/jobs/{created['job_id']}/result")
            self.assertEqual(status, 200)
            _decode_and_check(body, expected_wh, f"gpu:{src.name}")
            self.assertEqual(_sha256(src), hash_before)
            return elapsed

        names_5 = ["2.jpeg", "5.jpeg", "1.jpeg", "4.jpeg", "6.jpeg"]
        timings_5 = [(n, run_one(ORIGINALS_DIR / n)) for n in names_5]

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        repeat_src = ORIGINALS_DIR / "2.jpeg"
        timings_10 = [run_one(repeat_src) for _ in range(10)]
        peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)
                         if torch.cuda.is_available() else None)

        mean_t = sum(timings_10) / len(timings_10)
        five_report = ", ".join(f"{n}={t:.1f}s" for n, t in timings_5)
        print(f"\n[gpu:5-sequential] {five_report}")
        print(f"[gpu:10-repeat] mean={mean_t:.1f}s min={min(timings_10):.1f}s "
              f"max={max(timings_10):.1f}s peak_vram_mb={peak_vram_mb} "
              f"all={['%.1f' % t for t in timings_10]}")

        # warm per-job timing should be stable, not trending upward
        # (a growing trend would indicate a leak-driven slowdown)
        first_half = sum(timings_10[:5]) / 5
        second_half = sum(timings_10[5:]) / 5
        self.assertLess(second_half, first_half * 1.5,
                         f"job time grew suspiciously across repeats: "
                         f"first5_avg={first_half:.1f}s second5_avg={second_half:.1f}s")


if __name__ == "__main__":
    unittest.main()
