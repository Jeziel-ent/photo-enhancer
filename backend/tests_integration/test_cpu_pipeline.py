"""REAL end-to-end integration tests for the CPU execution path.

Companion to test_real_pipeline.py (GPU path) -- same real, unmocked
wiring (JobManager -> real HTTP server -> engine_adapter.enhance_image),
but forces device preference to "cpu" so every job actually goes through
backend.engine_adapter.enhance_image_cpu_subprocess -> the persistent,
isolated backend/cpu_worker.py child process (OpenVINO IMDN x4, no torch
import anywhere in that process -- see cpu_worker.py's own module
docstring for why that isolation exists).

Runs on ANY machine (no GPU required) -- CPU mode is always available.
Skipped by default; set ADINN_RUN_CPU_TESTS=1 to opt in:

    $env:ADINN_RUN_CPU_TESTS = "1"
    ..\\image_enhancer\\.venv\\Scripts\\python.exe -m pytest backend/tests_integration/test_cpu_pipeline.py -v -s

Run this file in its own process/invocation, separate from
ADINN_RUN_GPU_TESTS runs -- apply_device_preference() is documented as
"call exactly once before torch is ever imported in this process" and is
NOT meant to hot-swap devices mid-process (see engine_adapter.py).

IMPORTANT (found while writing this test, real product behavior, not a
test bug): calling engine_adapter.apply_device_preference("cpu") directly
in THIS thread is not enough once a real JobManager is involved --
JobManager._run() re-resolves the device preference itself, on its OWN
worker thread, by reading backend.settings.get_processing_device_preference()
(the persisted settings.json, default "auto") right before its first
warmup_engine() call -- see jobs.py's own docstring on that call, which
says this ordering is load-bearing and intentional (CUDA-visibility must
be decided before torch's first import, once, per process). On a machine
with a real GPU (this dev machine has one), "auto" resolves to GPU,
silently overriding an in-thread-only apply_device_preference("cpu") call
made before the JobManager existed. The correct way to force CPU mode for
a real JobManager-driven run -- exactly as the real Settings UI does via
PUT /api/settings -- is backend.settings.set_processing_device_preference
("cpu") BEFORE constructing the JobManager, so JobManager._run() reads
"cpu" too. RealCpuPipelineIntegrationTestCase below does this (and
restores the prior persisted preference in tearDownClass).
"""

from __future__ import annotations

import os
import sys
import threading
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
ORIGINALS_DIR = REPO_ROOT / "image_enhancer" / "originals"
IMAGE_ENHANCER_SRC = REPO_ROOT / "image_enhancer" / "src"

RUN_CPU_TESTS = os.environ.get("ADINN_RUN_CPU_TESTS") == "1"

from backend import engine_adapter, settings  # noqa: E402
from backend.jobs import JobManager  # noqa: E402
from backend.server import make_server  # noqa: E402
from backend.tests_integration.test_real_pipeline import (  # noqa: E402
    _http_get, _http_post_job, _poll_until_done, _sha256, _LoopbackOnlyNetworkGuard,
)

# CPU mode preference MUST be applied before enhance.py/torch is ever
# imported in this process (see apply_device_preference's docstring) --
# guarded so merely collecting this file (e.g. a broad `pytest backend`
# sweep with the env var unset) never forces CPU mode process-wide.
if RUN_CPU_TESTS:
    engine_adapter.apply_device_preference("cpu")


def _expected_target_wh(src_path: Path) -> tuple:
    """The real aspect-preserving 4K target for this exact source image --
    computed from the SAME production function the engine itself uses
    (enhance.aspect_preserving_target), not a hardcoded assumption, so this
    check stays correct for any source aspect ratio."""
    sys.path.insert(0, str(IMAGE_ENHANCER_SRC))
    import enhance
    img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
    return enhance.aspect_preserving_target(img.shape[1], img.shape[0])


def _decode_and_check_cpu_output(data: bytes, expected_wh: tuple, label: str) -> None:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise AssertionError(f"{label}: could not decode as an image")
    h, w = img.shape[:2]
    if (w, h) != expected_wh:
        raise AssertionError(f"{label}: expected {expected_wh[0]}x{expected_wh[1]} "
                              f"(aspect-preserving 4K target), got {w}x{h}")
    if max(w, h) != 3840:
        raise AssertionError(f"{label}: 4K-longest-side target not met (max dim {max(w, h)})")
    arrf = img.astype(np.float64)
    if not np.all(np.isfinite(arrf)):
        raise AssertionError(f"{label}: output contains NaN/Inf pixels")
    std = float(img.std())
    if std < 5.0:
        raise AssertionError(f"{label}: output looks blank/degenerate (std={std:.2f})")


@unittest.skipUnless(RUN_CPU_TESTS, "set ADINN_RUN_CPU_TESTS=1 to run the real CPU pipeline")
class CpuDeviceSelectionTestCase(unittest.TestCase):
    """Verifies the forced "cpu" preference actually resolves to CPU as the
    effective device, and stays CPU even though this test-runner machine
    may well have a real NVIDIA GPU available."""

    def test_cpu_preference_forces_cpu_effective_device(self):
        state = engine_adapter.get_device_state()
        self.assertEqual(state["preference"], "cpu")
        self.assertEqual(state["effective_device"], "cpu",
                          "forced CPU preference did not resolve to CPU -- "
                          "device selection is broken")

    def test_cpu_worker_module_never_imports_torch(self):
        """Static source check: cpu_worker.py's entire import graph must
        never reference torch -- this IS the isolation the whole CPU
        architecture depends on (see docs/ARCHITECTURE.md's CPU execution
        path section)."""
        source = (REPO_ROOT / "backend" / "cpu_worker.py").read_text(encoding="utf-8")
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("import torch") or stripped.startswith("from torch"):
                raise AssertionError(f"cpu_worker.py imports torch directly: {stripped!r}")


@unittest.skipUnless(RUN_CPU_TESTS, "set ADINN_RUN_CPU_TESTS=1 to run the real CPU pipeline")
class RealCpuPipelineIntegrationTestCase(unittest.TestCase):
    """Real HTTP -> real JobManager -> real engine_adapter.enhance_image,
    forced to the CPU subprocess path."""

    @classmethod
    def setUpClass(cls):
        # See the module docstring's "IMPORTANT" note: JobManager._run()
        # re-resolves the device preference from persisted settings on its
        # own worker thread, so it must be forced there too, not just in
        # this thread -- exactly how the real Settings UI forces it.
        cls._prior_device_preference = settings.get_processing_device_preference()
        settings.set_processing_device_preference("cpu")

        cls._network_guard = _LoopbackOnlyNetworkGuard()
        cls._network_guard.__enter__()
        cls.manager = JobManager()  # default process_fn == engine_adapter.enhance_image
        cls.httpd = make_server(port=0, job_manager=cls.manager)
        cls.base_url = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

        # Force one job through immediately so JobManager._run()'s
        # apply_device_preference("cpu") + warmup_engine() actually happen
        # (they run lazily, on first real job) before any test asserts on
        # get_device_state() or _CPU_WORKER.
        created = _http_post_job(cls.base_url, [
            ("files", "2.jpeg", "image/jpeg", (ORIGINALS_DIR / "2.jpeg").read_bytes()),
        ])
        _poll_until_done(cls.base_url, created["job_id"], timeout=300, progress_log=[])
        state = engine_adapter.get_device_state()
        assert state["effective_device"] == "cpu", (
            f"setUpClass warmup job did not run on CPU (state={state}) -- "
            "device-preference forcing failed, every test in this class "
            "would silently run on GPU instead")

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=5)
        cls._network_guard.__exit__(None, None, None)
        settings.set_processing_device_preference(cls._prior_device_preference)

    def _run_single(self, src: Path, timeout=300) -> tuple:
        original_hash_before = _sha256(src)
        expected_wh = _expected_target_wh(src)
        t0 = time.time()
        created = _http_post_job(self.base_url, [
            ("files", src.name, "image/jpeg", src.read_bytes()),
        ])
        job_id = created["job_id"]
        final = _poll_until_done(self.base_url, job_id, timeout=timeout, progress_log=[])
        elapsed = time.time() - t0
        self.assertEqual(final["status"], "completed")
        self.assertEqual(final["errors"], [])
        status, body, headers = _http_get(self.base_url, f"/api/jobs/{job_id}/result")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "image/png")
        _decode_and_check_cpu_output(body, expected_wh, f"cpu:{src.name}")
        self.assertEqual(_sha256(src), original_hash_before, "original was modified on disk")
        return elapsed, expected_wh

    def test_5_sequential_single_image_cpu_jobs_and_persistent_worker(self):
        """5 sequential single-image CPU jobs (item: 'at minimum 5
        sequential single-image jobs'). Also proves the CPU worker process
        is PERSISTENT (same PID reused across jobs, not respawned per
        image) -- the whole point of CpuWorkerHandle existing."""
        names = ["2.jpeg", "5.jpeg", "1.jpeg", "4.jpeg", "6.jpeg"]
        pids = []
        timings = []
        for name in names:
            elapsed, expected_wh = self._run_single(ORIGINALS_DIR / name)
            timings.append((name, elapsed, expected_wh))
            proc = engine_adapter._CPU_WORKER._proc
            self.assertIsNotNone(proc, "CPU worker process not running after a completed job")
            self.assertIsNone(proc.poll(), "CPU worker process exited unexpectedly")
            pids.append(proc.pid)

        self.assertEqual(len(set(pids)), 1,
                          f"CPU worker was NOT persistent across jobs -- PIDs seen: {pids}")

        report = "\n".join(f"  {n}: {e:.1f}s -> {w[0]}x{w[1]}" for n, e, w in timings)
        print(f"\n[cpu:5-sequential] persistent worker pid={pids[0]}\n{report}")

    def test_multi_image_cpu_batch(self):
        """One multi-image CPU batch job (item: 'at least 1 multi-image
        batch')."""
        import zipfile
        from io import BytesIO

        names = ["2.jpeg", "5.jpeg", "3.jpeg"]
        sources = [ORIGINALS_DIR / n for n in names]
        expected = {p.name: _expected_target_wh(p) for p in sources}

        t0 = time.time()
        created = _http_post_job(self.base_url, [
            ("files", p.name, "image/jpeg", p.read_bytes()) for p in sources
        ])
        job_id = created["job_id"]
        self.assertEqual(created["total_count"], 3)
        final = _poll_until_done(self.base_url, job_id, timeout=900, progress_log=[])
        elapsed = time.time() - t0

        self.assertEqual(final["status"], "completed")
        self.assertEqual(final["completed_count"], 3)
        self.assertEqual(final["errors"], [])

        status, body, headers = _http_get(self.base_url, f"/api/jobs/{job_id}/result")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "application/zip")
        with zipfile.ZipFile(BytesIO(body)) as zf:
            entries = sorted(zf.namelist())
            self.assertEqual(len(entries), 3)
            for entry_name, src_name in zip(entries, sorted(names)):
                # filenames are "<stem>_enhanced.png"; match by best-effort stem prefix
                matched = next((n for n in names if entry_name.startswith(Path(n).stem)), None)
                self.assertIsNotNone(matched, f"could not match zip entry {entry_name} to a source")
                _decode_and_check_cpu_output(zf.read(entry_name), expected[matched],
                                             f"cpu-batch:{entry_name}")

        print(f"\n[cpu:batch-3] total_elapsed={elapsed:.1f}s entries={entries}")

    def test_10_repeated_jobs_stability_if_time_permits(self):
        """Repeated-job stability check (item: '10 repeated jobs if
        runtime permits'). Uses the SAME small image 10x to isolate
        repeated-inference stability from per-image content variance."""
        src = ORIGINALS_DIR / "2.jpeg"
        expected_wh = _expected_target_wh(src)
        timings = []
        proc_before = engine_adapter._CPU_WORKER._proc
        pid_before = proc_before.pid if proc_before else None
        for i in range(10):
            elapsed, _ = self._run_single(src, timeout=180)
            timings.append(elapsed)
        proc_after = engine_adapter._CPU_WORKER._proc
        self.assertIsNotNone(proc_after)
        self.assertIsNone(proc_after.poll(), "CPU worker died during the 10-job repeat run")
        pid_after = proc_after.pid
        mean_t = sum(timings) / len(timings)
        print(f"\n[cpu:10-repeat] pid_before={pid_before} pid_after={pid_after} "
              f"(persistent={pid_before == pid_after}) mean={mean_t:.1f}s "
              f"min={min(timings):.1f}s max={max(timings):.1f}s all={['%.1f' % t for t in timings]}")
        self.assertEqual(pid_before, pid_after,
                          "CPU worker was restarted at some point during 10 repeated jobs")


if __name__ == "__main__":
    unittest.main()
