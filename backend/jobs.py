"""Job lifecycle and the single-worker processing queue.

One GPU, one job at a time, and within a job, files are processed
sequentially — see "Processing model" in docs/ARCHITECTURE.md. This module
owns job state only; it knows nothing about HTTP (server.py) and nothing
about how a file actually gets enhanced beyond the callable it's given
(engine_adapter.enhance_image by default, or a fake in tests).
"""

from __future__ import annotations

import inspect
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from . import engine_adapter, output_manager, workspace

STATUS_QUEUED = "queued"
STATUS_PROCESSING = "processing"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

ProcessFn = Callable[[Path, Path], None]

# Ordered stage -> "how far through one file" fraction, used to blend the
# current file's in-progress stage into the job's overall progress number
# instead of jumping straight from 0% to 100% per file. Mirrors
# engine_adapter.STAGES; kept as plain strings here so jobs.py has no import
# dependency on engine internals beyond the stage name constants it already
# imports for wiring up the callback.
STAGE_LABELS = {
    engine_adapter.STAGE_PREPARING: "Preparing",
    engine_adapter.STAGE_ANALYZING: "Analyzing",
    engine_adapter.STAGE_RESTORING: "Restoring",
    engine_adapter.STAGE_ENHANCING_DETAILS: "Enhancing details",
    engine_adapter.STAGE_UPSCALING: "Upscaling to 4K",
    engine_adapter.STAGE_FINALIZING: "Finalizing",
}

_STAGE_FRACTION = {
    engine_adapter.STAGE_PREPARING: 0.02,
    engine_adapter.STAGE_ANALYZING: 0.10,
    engine_adapter.STAGE_RESTORING: 0.35,
    engine_adapter.STAGE_ENHANCING_DETAILS: 0.65,
    engine_adapter.STAGE_UPSCALING: 0.85,
    engine_adapter.STAGE_FINALIZING: 0.97,
}


@dataclass
class FileResult:
    original_filename: str
    input_path: Path
    output_path: Optional[Path] = None
    error: Optional[str] = None


@dataclass
class Job:
    id: str
    files: list[FileResult]
    status: str = STATUS_QUEUED
    current_file: Optional[str] = None
    current_stage: Optional[str] = None
    completed_count: int = 0
    created_at: float = field(default_factory=time.time)
    fatal_error: Optional[str] = None
    result_path: Optional[Path] = None
    result_filename: Optional[str] = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    @property
    def total_count(self) -> int:
        return len(self.files)

    def _current_file_fraction(self) -> float:
        if self.status != STATUS_PROCESSING or self.current_stage is None:
            return 0.0
        return _STAGE_FRACTION.get(self.current_stage, 0.0)

    def _progress(self) -> float:
        if self.total_count == 0:
            return 1.0
        return (self.completed_count + self._current_file_fraction()) / self.total_count

    def _errors(self) -> list[dict]:
        return [
            {"filename": f.original_filename, "message": f.error}
            for f in self.files
            if f.error
        ]

    def to_status_dict(self) -> dict:
        """A consistent snapshot of this job's state, safe to read while the
        worker thread is concurrently mutating it."""
        with self.lock:
            return {
                "job_id": self.id,
                "status": self.status,
                "progress": round(self._progress(), 4),
                "current_file": self.current_file,
                "current_stage": self.current_stage,
                "current_stage_label": STAGE_LABELS.get(self.current_stage),
                "completed_count": self.completed_count,
                "total_count": self.total_count,
                "errors": self._errors(),
            }


class JobManager:
    """Owns every job's state and a single background worker thread that
    processes jobs — and the files within each job — strictly one at a
    time, matching the one-GPU processing model.

    ``process_fn`` defaults to the real engine adapter and is overridable so
    the API/job contract can be exercised in tests without a GPU.
    """

    def __init__(self, process_fn: ProcessFn = engine_adapter.enhance_image):
        self._process_fn = process_fn
        try:
            params = inspect.signature(process_fn).parameters
            self._supports_on_stage = "on_stage" in params or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
        except (TypeError, ValueError):
            self._supports_on_stage = False
        self._jobs: dict[str, Job] = {}
        self._jobs_lock = threading.Lock()
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def create_job(self, uploads: list[tuple[str, bytes]]) -> Job:
        """``uploads`` is a list of (original_filename, raw_bytes), already
        validated by the caller (server.py). Writes each upload into this
        job's own workspace input directory, then enqueues it for
        processing; never touches wherever the original bytes came from."""
        job_id = workspace.new_job_id()
        in_dir, _out_dir = workspace.prepare_job_dirs(job_id)
        files: list[FileResult] = []
        for index, (original_filename, data) in enumerate(uploads):
            safe_name = Path(original_filename).name or f"file_{index}"
            input_path = in_dir / f"{index:03d}_{safe_name}"
            input_path.write_bytes(data)
            files.append(FileResult(original_filename=original_filename, input_path=input_path))

        job = Job(id=job_id, files=files)
        with self._jobs_lock:
            self._jobs[job_id] = job
        self._queue.put(job_id)
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        with self._jobs_lock:
            return self._jobs.get(job_id)

    def _run(self) -> None:
        while True:
            job_id = self._queue.get()
            job = self.get_job(job_id)
            if job is None:
                continue
            self._process_job(job)

    def _process_job(self, job: Job) -> None:
        with job.lock:
            job.status = STATUS_PROCESSING

        out_dir = workspace.output_dir(job.id)
        for file_result in job.files:
            with job.lock:
                job.current_file = file_result.original_filename
                job.current_stage = engine_adapter.STAGE_PREPARING

            def _on_stage(stage: str, _job=job) -> None:
                with _job.lock:
                    _job.current_stage = stage

            output_path = out_dir / f"{file_result.input_path.stem}.png"
            try:
                if self._supports_on_stage:
                    self._process_fn(file_result.input_path, output_path, on_stage=_on_stage)
                else:
                    self._process_fn(file_result.input_path, output_path)
            except Exception as exc:  # noqa: BLE001 — one bad file must not sink the job
                with job.lock:
                    file_result.error = str(exc)
            else:
                file_result.output_path = output_path
            with job.lock:
                job.completed_count += 1
                job.current_stage = None

        with job.lock:
            job.current_file = None

        succeeded = [(f.original_filename, f.output_path) for f in job.files if f.output_path]
        if not succeeded:
            with job.lock:
                job.status = STATUS_FAILED
                job.fatal_error = "all files failed to enhance"
            return

        is_batch = job.total_count > 1
        try:
            result_path = output_manager.build_result(
                succeeded, workspace.result_zip_path(job.id), is_batch=is_batch)
        except Exception as exc:  # noqa: BLE001
            with job.lock:
                job.status = STATUS_FAILED
                job.fatal_error = f"could not package result: {exc}"
            return

        if is_batch:
            result_filename = f"enhanced_images_{job.id[:8]}.zip"
        else:
            result_filename = f"{Path(job.files[0].original_filename).stem}_enhanced.png"

        with job.lock:
            job.result_path = result_path
            job.result_filename = result_filename
            job.status = STATUS_COMPLETED
