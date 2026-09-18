import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from backend import jobs as jobs_module
from backend.jobs import STATUS_COMPLETED, STATUS_FAILED, JobManager


def _wait_until(predicate, timeout=5.0, interval=0.01):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


class JobManagerTestCase(unittest.TestCase):
    """Uses an isolated workspace directory and a fake process_fn so these
    exercise the job/queue contract without needing a GPU or the real
    image_enhancer engine."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._workspace_patch = mock.patch.object(
            jobs_module.workspace, "WORKSPACE_ROOT", Path(self._tmp.name))
        self._workspace_patch.start()
        self.addCleanup(self._workspace_patch.stop)

    def _wait_done(self, manager, job_id, timeout=5.0):
        _wait_until(
            lambda: manager.get_job(job_id).to_status_dict()["status"]
            in (STATUS_COMPLETED, STATUS_FAILED),
            timeout=timeout,
        )
        return manager.get_job(job_id)

    def test_single_file_success(self):
        def fake_process(input_path, output_path):
            output_path.write_bytes(b"enhanced")

        manager = JobManager(process_fn=fake_process)
        job = manager.create_job([("photo.jpg", b"raw-bytes")])
        self.assertEqual(job.status, jobs_module.STATUS_QUEUED)

        job = self._wait_done(manager, job.id)
        status = job.to_status_dict()
        self.assertEqual(status["status"], STATUS_COMPLETED)
        self.assertEqual(status["completed_count"], 1)
        self.assertEqual(status["total_count"], 1)
        self.assertEqual(status["errors"], [])
        self.assertTrue(job.result_path.is_file())
        self.assertEqual(job.result_path.read_bytes(), b"enhanced")
        self.assertTrue(job.result_filename.endswith("_enhanced.png"))

    def test_original_file_is_never_modified(self):
        original = b"\xff\xd8ORIGINAL-BYTES"

        def fake_process(input_path, output_path):
            # A misbehaving processor that tried to mutate the input would
            # corrupt this check.
            assert input_path.read_bytes() == original
            output_path.write_bytes(b"enhanced")

        manager = JobManager(process_fn=fake_process)
        job = manager.create_job([("photo.jpg", original)])
        job = self._wait_done(manager, job.id)
        self.assertEqual(job.status, STATUS_COMPLETED)
        self.assertEqual(job.files[0].input_path.read_bytes(), original)

    def test_multiple_files_all_succeed_produce_zip(self):
        def fake_process(input_path, output_path):
            output_path.write_bytes(input_path.read_bytes().upper())

        manager = JobManager(process_fn=fake_process)
        job = manager.create_job([
            ("a.jpg", b"aaa"), ("b.jpg", b"bbb"), ("c.jpg", b"ccc"),
        ])
        job = self._wait_done(manager, job.id)
        self.assertEqual(job.status, STATUS_COMPLETED)
        self.assertEqual(job.result_path.suffix, ".zip")
        self.assertEqual(job.to_status_dict()["completed_count"], 3)

    def test_partial_failure_keeps_job_completed_with_errors(self):
        def fake_process(input_path, output_path):
            if "bad" in input_path.name:
                raise RuntimeError("simulated corrupt image")
            output_path.write_bytes(b"ok")

        manager = JobManager(process_fn=fake_process)
        job = manager.create_job([
            ("good1.jpg", b"1"), ("bad.jpg", b"2"), ("good2.jpg", b"3"),
        ])
        job = self._wait_done(manager, job.id)
        status = job.to_status_dict()
        self.assertEqual(status["status"], STATUS_COMPLETED)
        self.assertEqual(status["completed_count"], 3)
        self.assertEqual(len(status["errors"]), 1)
        self.assertEqual(status["errors"][0]["filename"], "bad.jpg")
        self.assertIn("simulated corrupt image", status["errors"][0]["message"])

    def test_all_files_failing_marks_job_failed(self):
        def fake_process(input_path, output_path):
            raise RuntimeError("engine unavailable")

        manager = JobManager(process_fn=fake_process)
        job = manager.create_job([("a.jpg", b"1"), ("b.jpg", b"2")])
        job = self._wait_done(manager, job.id)
        status = job.to_status_dict()
        self.assertEqual(status["status"], STATUS_FAILED)
        self.assertEqual(len(status["errors"]), 2)
        self.assertIsNone(job.result_path)

    def test_status_results_list_ids_and_filenames_in_upload_order(self):
        def fake_process(input_path, output_path):
            output_path.write_bytes(b"ok")

        manager = JobManager(process_fn=fake_process)
        job = manager.create_job([("a.jpg", b"1"), ("b.jpg", b"2")])
        job = self._wait_done(manager, job.id)
        self.assertEqual(
            job.to_status_dict()["results"],
            [{"id": "000", "filename": "a.jpg"}, {"id": "001", "filename": "b.jpg"}],
        )

    def test_status_results_excludes_failed_files(self):
        def fake_process(input_path, output_path):
            if "bad" in input_path.name:
                raise RuntimeError("nope")
            output_path.write_bytes(b"ok")

        manager = JobManager(process_fn=fake_process)
        job = manager.create_job([("good.jpg", b"1"), ("bad.jpg", b"2")])
        job = self._wait_done(manager, job.id)
        self.assertEqual(
            job.to_status_dict()["results"], [{"id": "000", "filename": "good.jpg"}])

    def test_get_file_result_looks_up_by_result_id(self):
        def fake_process(input_path, output_path):
            output_path.write_bytes(b"ok")

        manager = JobManager(process_fn=fake_process)
        job = manager.create_job([("a.jpg", b"1"), ("b.jpg", b"2")])
        job = self._wait_done(manager, job.id)
        found = job.get_file_result("001")
        self.assertIsNotNone(found)
        self.assertEqual(found.original_filename, "b.jpg")
        self.assertIsNone(job.get_file_result("999"))

    def test_unknown_job_id_returns_none(self):
        manager = JobManager(process_fn=lambda i, o: None)
        self.assertIsNone(manager.get_job("does-not-exist"))

    def test_jobs_process_sequentially_never_concurrently(self):
        """Proves the single-GPU processing model: two jobs submitted back
        to back must never have their files processed at overlapping
        times, even though each job's files are handed to the same
        process_fn from one background worker thread."""
        concurrent_count = 0
        max_concurrent = 0
        state_lock = threading.Lock()

        def fake_process(input_path, output_path):
            nonlocal concurrent_count, max_concurrent
            with state_lock:
                concurrent_count += 1
                max_concurrent = max(max_concurrent, concurrent_count)
            time.sleep(0.05)
            output_path.write_bytes(b"ok")
            with state_lock:
                concurrent_count -= 1

        manager = JobManager(process_fn=fake_process)
        job1 = manager.create_job([("a.jpg", b"1"), ("b.jpg", b"2")])
        job2 = manager.create_job([("c.jpg", b"3"), ("d.jpg", b"4")])

        self._wait_done(manager, job1.id)
        self._wait_done(manager, job2.id)
        self.assertEqual(max_concurrent, 1)


if __name__ == "__main__":
    unittest.main()
