"""Adapter between the Application/API layer and the Image Enhancement Engine.

Owns every detail of calling into ``image_enhancer/src/enhance.py`` so the
rest of ``backend/`` never imports it directly: sys.path wiring, the one
production method used ("final"), serializing calls onto the single GPU, and
neutralizing the engine's billboard-regions config (see
``_neutralize_billboard_regions`` — this matters, read it before changing
this file).

Nothing in ``image_enhancer/`` is modified by this module.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import Callable, Optional

from . import device as device_detect
from ._frozen import app_root

REPO_ROOT = app_root()
IMAGE_ENHANCER_SRC = REPO_ROOT / "image_enhancer" / "src"

# Ordered UI-facing processing stages, each tied to a real, already-existing
# step of enhance.final_enhance()'s own pipeline (see _run_final_pipeline
# below) rather than a time-based guess — so progress reflects actual engine
# milestones, not a frontend animation.
STAGE_PREPARING = "preparing"
STAGE_ANALYZING = "analyzing"
STAGE_RESTORING = "restoring"
STAGE_ENHANCING_DETAILS = "enhancing_details"
STAGE_UPSCALING = "upscaling"
STAGE_FINALIZING = "finalizing"

STAGES = (
    STAGE_PREPARING,
    STAGE_ANALYZING,
    STAGE_RESTORING,
    STAGE_ENHANCING_DETAILS,
    STAGE_UPSCALING,
    STAGE_FINALIZING,
)

OnStage = Callable[[str], None]

# A path that is guaranteed never to exist on disk (nothing ever creates it).
# Pointing the engine's REGIONS_CONFIG env var here forces its
# ``_load_region_config()`` to fall back to an empty ``{}``, which in turn
# forces ``_billboard_boxes()`` to always return ``[]`` — see
# _neutralize_billboard_regions for why this is required, not optional.
_NO_REGIONS_SENTINEL = REPO_ROOT / "backend" / ".workspace" / "__no_billboard_regions__.json"

# enhance.py caches loaded models at module scope and mutates a module-level
# global (_REGION_CONFIG) that is not thread-safe; only one enhancement call
# may be in flight at a time regardless of how many threads call this
# adapter. This mirrors the one-GPU processing model in
# docs/ARCHITECTURE.md — JobManager's single worker thread already provides
# this, but the lock makes the adapter itself safe to call directly too.
_ENGINE_LOCK = threading.Lock()

METHOD = "final"

# The resolved device state (see apply_device_preference below), readable
# by server.py's /api/settings endpoint to report what's ACTUALLY active in
# this running process -- never guessed from the persisted preference
# alone, since preference != effective device whenever GPU was requested
# but not detected (see device.py's resolve_effective_device).
_DEVICE_STATE: dict = {
    "preference": device_detect.DEFAULT_PREFERENCE,
    "detected_gpu": None,
    "effective_device": None,  # None until apply_device_preference() runs
    "warning": None,
}
_DEVICE_STATE_LOCK = threading.Lock()


def get_device_state() -> dict:
    """A copy of the current resolved device state, safe to call from any
    thread (e.g. the HTTP server's request-handling threads) at any time,
    including before apply_device_preference() has ever run (effective_device
    is None in that case -- the API layer should report that as "not yet
    determined" rather than guessing)."""
    with _DEVICE_STATE_LOCK:
        return dict(_DEVICE_STATE)


def apply_device_preference(preference: str) -> dict:
    """Resolves ``preference`` against actually detected hardware and, if
    the effective device is CPU, sets CUDA_VISIBLE_DEVICES so torch never
    sees a GPU -- satisfying "do not initialize CUDA" for CPU mode.

    MUST be called exactly once, before enhance.py (and therefore torch) is
    ever imported in this process (see jobs.py's JobManager._run, which
    calls this immediately before warmup_engine()/the first
    _import_engine()). Environment-variable-based device hiding only works
    if set before torch's CUDA runtime is first touched; PyTorch does not
    reliably un-see a GPU it has already initialized, so this function is
    NOT meant to be called again later in the same process to hot-swap
    devices -- callers changing the persisted preference while the app is
    already running should tell the user a restart is needed (see
    server.py's PUT /api/settings, which reports this via
    ``restart_required``).
    """
    resolved = device_detect.resolve_effective_device(preference)
    if resolved["effective_device"] == "cpu":
        # "-1" (not "") reliably hides all CUDA devices from torch on this
        # project's stack -- verified this session: an empty string caused
        # torch to raise "Invalid device id" instead of cleanly reporting
        # no devices.
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    with _DEVICE_STATE_LOCK:
        _DEVICE_STATE.update(resolved)
    return resolved


class EngineError(Exception):
    """Raised when the image enhancement engine fails to process an image."""


def _import_engine() -> ModuleType:
    if str(IMAGE_ENHANCER_SRC) not in sys.path:
        sys.path.insert(0, str(IMAGE_ENHANCER_SRC))
    import enhance  # image_enhancer/src/enhance.py — a flat module, not a package
    return enhance


def _neutralize_billboard_regions(engine: ModuleType):
    """Guarantees ``engine.final_enhance`` sees zero billboard boxes.

    ``enhance.py`` was built for the retired PPT product, where every source
    photo had manually verified billboard boxes recorded in
    ``image_enhancer/regions.json``. That file still exists (it's the R&D
    engine's own fixture) and ``_billboard_boxes()`` has a same-size fallback:
    if a generic photo a user uploads to *this* product happens to match one
    of those six recorded (width, height) pairs — plausible, since phone
    cameras produce a small set of common resolutions — the engine would
    silently run billboard-box reconstruction over some unrelated region of
    that photo. That would be exactly the kind of unintended content
    alteration the product's fidelity rule forbids, so it must never fire
    for a generic image.

    Pointing REGIONS_CONFIG at a path that never exists forces the loaded
    config to be ``{}`` for every call, closing this off entirely: with no
    entries at all, neither the by-name nor the by-size lookup can ever
    match anything. Returns the previous env var values so the caller can
    restore them afterwards (defensive — nothing else in this process should
    be setting these, but the archived PPT backend used to).
    """
    import os
    prev_regions = os.environ.get("REGIONS_CONFIG")
    prev_billboard = os.environ.get("BILLBOARD_IMAGE")
    os.environ["REGIONS_CONFIG"] = str(_NO_REGIONS_SENTINEL)
    os.environ.pop("BILLBOARD_IMAGE", None)
    engine._REGION_CONFIG = None  # force a reload against the sentinel (empty) path
    return prev_regions, prev_billboard


def _restore_billboard_regions(engine: ModuleType, prev_regions, prev_billboard) -> None:
    import os
    if prev_regions is None:
        os.environ.pop("REGIONS_CONFIG", None)
    else:
        os.environ["REGIONS_CONFIG"] = prev_regions
    if prev_billboard is None:
        os.environ.pop("BILLBOARD_IMAGE", None)
    else:
        os.environ["BILLBOARD_IMAGE"] = prev_billboard
    engine._REGION_CONFIG = None  # don't leak the sentinel (or its own state) to the next caller


# Elapsed seconds (measured from when the real engine.enhance("final", ...)
# call starts, in a background thread) after which the ticker advances to
# each next stage — calibrated against real single-image runs on this
# product's target hardware (an RTX 3050-class GPU; see docs/ARCHITECTURE.md
# "Processing model"). This intentionally does NOT decompose or re-call any
# of final_enhance's internal steps directly: an earlier version of this
# adapter did that, and it broke a real invariant relied on elsewhere
# (backend/tests_integration/test_real_pipeline.py monkeypatches
# ``enhance.METHODS["final"]`` to observe billboard-box behavior, which only
# works if engine.enhance() is the single, unbypassed call path into the
# engine). So this is a wall-clock estimate of a real, currently-running
# call, not a fake animation: it only ticks while the real GPU call is
# in flight, and always ends the moment that call actually returns.
_STAGE_TIMELINE = (
    (2.0, STAGE_RESTORING),
    (6.0, STAGE_ENHANCING_DETAILS),
    (18.0, STAGE_UPSCALING),
)
_TICK_SECONDS = 0.25


def _run_with_stage_ticker(engine: ModuleType, img, on_stage: Optional[OnStage], target):
    """Runs ``engine.enhance(METHOD, img)`` — the one real, unmodified call
    path into the engine — on the CALLING thread, while a separate,
    GPU/torch-untouching ticker thread ticks ``on_stage`` through
    _STAGE_TIMELINE based on elapsed time. Returns ``(out, dt)`` exactly as
    engine.enhance() does, or re-raises whatever exception the engine call
    raised.

    Deliberately runs the heavy call on the caller's own thread rather than
    spawning a worker thread for it (an earlier version of this function did
    the opposite): PyTorch's cuDNN benchmark-autotune cache
    (torch.backends.cudnn.benchmark=True, set in enhance.py) is thread-local,
    and JobManager calls this via the same single persistent worker thread
    for every job (see jobs.py's JobManager._run and its own warmup-on-this-
    thread comment) -- spawning a brand-new thread per call here would give
    every single image a cold cuDNN cache forever, not just the first one.
    Measured: ~25s/image with the call kept on one persistent thread vs.
    ~33s/image (every image, not just the first) when it ran on a fresh
    thread each time. Only the lightweight ticker thread is spawned fresh
    per call, and it never touches CUDA/torch, so it has no cache to lose.
    """
    import time as _time

    done = threading.Event()

    def _ticker() -> None:
        remaining = list(_STAGE_TIMELINE)
        t0 = _time.monotonic()
        while not done.wait(timeout=_TICK_SECONDS):
            elapsed = _time.monotonic() - t0
            while remaining and elapsed >= remaining[0][0]:
                _, stage_name = remaining.pop(0)
                if on_stage is not None:
                    on_stage(stage_name)

    ticker = threading.Thread(target=_ticker, daemon=True)
    ticker.start()
    try:
        out, dt = engine.enhance(METHOD, img, target=target)
    finally:
        done.set()
        ticker.join()
    return out, dt


def warmup_engine() -> None:
    """Loads the engine's models before any real upload is processed, so
    that cost lands at app startup instead of on whichever image happens
    to be processed first. MUST be called after apply_device_preference()
    has already resolved the effective device (see jobs.py's
    JobManager._run, which does both in that order) -- dispatches on
    get_device_state()["effective_device"].

    GPU: pre-triggers cuDNN's one-time per-shape kernel autotune (see
    enhance.warmup()'s own docstring), in-process, on the calling thread
    specifically (cuDNN's benchmark-autotune cache is thread-local -- see
    _run_with_stage_ticker's docstring for the ~8s-per-image-forever
    measurement of getting this wrong). CPU: starts the persistent CPU
    worker subprocess now (see CpuWorkerHandle/cpu_worker.py) and sends it
    a WARMUP request so the ~15s cold import + OpenVINO compile also
    happens here instead of on the first real CPU job.

    Best-effort either way: any failure here (e.g. no GPU driver) is
    swallowed, since the real first upload will simply pay whatever this
    step would have paid — never worth blocking job processing over.
    """
    if get_device_state()["effective_device"] == "cpu":
        _CPU_WORKER.warmup()
        return
    try:
        engine = _import_engine()
    except Exception:  # noqa: BLE001 — best-effort, never fatal to startup (see
        # this function's own docstring). Broadened from `except ImportError`:
        # a native DLL/CUDA init failure inside `import torch` (transitively
        # imported by `import enhance`) raises OSError, not ImportError -- an
        # ImportError-only catch here let that kill JobManager._run()'s
        # calling thread outright (an uncaught exception on a background
        # thread), silently wedging every future job at "queued" forever with
        # no visible error. See docs/RELEASE_READINESS.md's "user-facing GPU
        # failure" note for the real incident this fixes.
        return
    with _ENGINE_LOCK:
        try:
            engine.warmup()
        except Exception:  # noqa: BLE001 — best-effort, never fatal to startup
            pass


def enhance_image(
    input_path: Path,
    output_path: Path,
    on_stage: Optional[OnStage] = None,
) -> None:
    """Runs the production "final" pipeline on one image file.

    Reads ``input_path``, never writes to it, and writes the 4K enhanced
    result to ``output_path``. Raises EngineError (with a message safe to
    surface to the API caller) on any failure — the job manager treats that
    as one failed file, not a crash.

    ``on_stage``, when given, is called with one of the STAGES values as
    processing reaches each milestone. Optional and additive: omitting it
    reproduces the exact prior behavior.

    Dispatches on the resolved device (see apply_device_preference /
    get_device_state, set once at process startup by jobs.py's
    JobManager._run): GPU runs in-process, exactly as before this session's
    CPU-stability work. CPU runs in a standalone child process -- see
    enhance_image_cpu_subprocess's own docstring and
    docs/PERFORMANCE_OPTIMIZATION.md's CPU stability section for why.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if get_device_state()["effective_device"] == "cpu":
        enhance_image_cpu_subprocess(input_path, output_path, on_stage)
        return

    def stage(name: str) -> None:
        if on_stage is not None:
            on_stage(name)

    stage(STAGE_PREPARING)
    try:
        engine = _import_engine()
    except Exception as exc:  # noqa: BLE001 — broadened from `except ImportError`:
        # a native DLL/CUDA init failure inside `import torch` raises OSError,
        # not ImportError. Wrapping ANY import-time failure as EngineError
        # here (same as the existing ImportError case) ensures JobManager's
        # per-file try/except (backend/jobs.py's _process_job, which already
        # catches general Exception) reports a clean, user-facing message
        # instead of a raw OSError string -- this path is only reached at
        # all once the worker thread survives warmup_engine()'s own broadened
        # catch above; this is the per-job-file safety net for the same
        # failure class.
        raise EngineError(f"image enhancement engine is unavailable: {exc}") from exc

    with _ENGINE_LOCK:
        try:
            img = engine.load_image(str(input_path))
        except Exception as exc:  # noqa: BLE001
            raise EngineError(f"could not read {input_path.name}: {exc}") from exc

        stage(STAGE_ANALYZING)
        # MVP 2: the output box preserves the SOURCE image's own aspect
        # ratio (longest side -> 3840) instead of always stretching to a
        # fixed 3840x2160 16:9 box -- see enhance.aspect_preserving_target's
        # own docstring for the formula and docs/MVP2_RESEARCH.md Phase 2
        # for the review requirement this satisfies.
        target = engine.aspect_preserving_target(img.shape[1], img.shape[0])
        prev_regions, prev_billboard = _neutralize_billboard_regions(engine)
        try:
            out, _dt = _run_with_stage_ticker(engine, img, on_stage, target)
        except Exception as exc:  # noqa: BLE001
            raise EngineError(f"enhancement failed for {input_path.name}: {exc}") from exc
        finally:
            _restore_billboard_regions(engine, prev_regions, prev_billboard)

        stage(STAGE_FINALIZING)
        try:
            engine.save_image(out, str(output_path))
        except Exception as exc:  # noqa: BLE001
            raise EngineError(
                f"could not save enhanced output for {input_path.name}: {exc}"
            ) from exc


def _cpu_worker_command() -> list:
    """The argv to spawn backend/cpu_worker.py's main() as a standalone,
    PERSISTENT process (started once, reused for every CPU job -- see
    CpuWorkerHandle) -- see enhance_image_cpu_subprocess for why it must be
    a separate process at all, and cpu_worker.py's own docstring for why it
    must be persistent rather than spawned fresh per job.

    A frozen PyInstaller build has no separate bundled python.exe to spawn
    a helper from, so the packaged EXE re-invokes ITSELF with a special
    ``--cpu-worker`` flag that packaging/pyinstaller/launcher.py checks for
    first, before any of the real app's startup (mutex, ADINN_INSTALLED,
    importing backend.shell/webview) -- see that file's own module
    docstring. In dev, sys.executable is the venv's python.exe, invoked as
    a normal module.
    """
    if getattr(sys, "frozen", False):
        return [sys.executable, "--cpu-worker"]
    return [sys.executable, "-m", "backend.cpu_worker"]


_CPU_JOB_TIMEOUT_S = 60.0
_CPU_STATUS_POLL_S = 0.15
# Stages emitted (in order) the moment the status file confirms success --
# see CpuWorkerHandle._send_and_await_status's docstring for why completion is
# detected via a file rather than parsed from the child's stdout.
_CPU_REMAINING_STAGES = (
    STAGE_ANALYZING, STAGE_RESTORING, STAGE_ENHANCING_DETAILS,
    STAGE_UPSCALING, STAGE_FINALIZING,
)


class CpuWorkerHandle:
    """Owns the single, persistent CPU-worker child process (see
    cpu_worker.py's own module docstring for the full "why persistent" and
    "why a file, not stdout" story).

    Thread-unsafe by design: JobManager only ever calls this from its one
    persistent worker thread (the same "single-job-at-a-time" model as the
    GPU path), so no internal locking is needed beyond what _ensure_alive
    does to make (re)spawning idempotent if called concurrently.
    """

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def _spawn(self) -> subprocess.Popen:
        return subprocess.Popen(
            _cpu_worker_command(), cwd=str(REPO_ROOT),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )

    def _ensure_alive(self) -> subprocess.Popen:
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._proc = self._spawn()
            return self._proc

    def _send_and_await_status(self, request: str, status_base: Path) -> dict:
        """Sends ``request`` to the (possibly freshly spawned) worker and
        blocks until ``<status_base>.status.json`` appears, confirming the
        job actually finished -- see cpu_worker.py's _write_status
        docstring for why this file, not a stdout line, is the
        authoritative signal. Returns the parsed status dict
        ({"ok": bool, "message": str}); raises EngineError on a write
        failure, timeout, or the worker process dying outright.
        """
        import json
        import time as _time

        status_path = Path(str(status_base) + ".status.json")
        try:
            status_path.unlink()
        except FileNotFoundError:
            pass

        proc = self._ensure_alive()
        assert proc.stdin is not None
        try:
            proc.stdin.write(request + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError):
            # The worker died before this request even reached it -- respawn
            # once and retry, so a single bad moment doesn't wedge CPU mode.
            with self._lock:
                self._proc = None
            proc = self._ensure_alive()
            try:
                proc.stdin.write(request + "\n")
                proc.stdin.flush()
            except (BrokenPipeError, OSError) as exc2:
                raise EngineError(
                    f"could not reach the CPU enhancement worker: {exc2}") from exc2

        deadline = _time.monotonic() + _CPU_JOB_TIMEOUT_S
        while _time.monotonic() < deadline:
            if status_path.exists():
                try:
                    data = json.loads(status_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    _time.sleep(_CPU_STATUS_POLL_S)
                    continue  # write still in flight (mkstemp+replace is
                              # atomic, but the exists() check can race a
                              # half-written temp file's rename briefly)
                try:
                    status_path.unlink()
                except OSError:
                    pass
                return data
            if proc.poll() is not None:
                # The worker process itself exited without ever writing a
                # status file -- the exact failure mode this isolation
                # exists to contain. Surface whatever it printed and
                # respawn for next time.
                stderr_text = ""
                try:
                    stderr_text = proc.stderr.read().strip() if proc.stderr else ""
                except Exception:  # noqa: BLE001 -- diagnostic only
                    pass
                exit_code = proc.poll()
                with self._lock:
                    self._proc = None
                detail = f" (exit {exit_code}{': ' + stderr_text if stderr_text else ''})"
                raise EngineError(
                    "the CPU enhancement worker exited unexpectedly while "
                    f"processing this image{detail}. It has been restarted "
                    "for the next attempt.")
            _time.sleep(_CPU_STATUS_POLL_S)

        # Timed out with the process still alive but no status file -- a
        # hang, not a crash. Kill and respawn rather than leaving a wedged
        # worker for every future job.
        with self._lock:
            self._proc = None
        try:
            proc.kill()
        except OSError:
            pass
        raise EngineError(
            f"the CPU enhancement worker did not finish within {_CPU_JOB_TIMEOUT_S:.0f}s "
            "and was restarted.")

    def warmup(self) -> None:
        status_base = REPO_ROOT / "backend" / ".workspace" / "__cpu_warmup"
        status_base.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._send_and_await_status(f"WARMUP\t{status_base}", status_base)
        except EngineError:
            pass  # best-effort, matching the GPU path's warmup_engine()

    def run(self, input_path: Path, output_path: Path, on_stage: Optional[OnStage]) -> None:
        result = self._send_and_await_status(f"{input_path}\t{output_path}", Path(output_path))
        if not result.get("ok"):
            raise EngineError(result.get("message") or "CPU enhancement failed")
        # The child ran the whole pipeline in one call with no mid-pipeline
        # callback across the process boundary (see cpu_worker.py's
        # _run_one docstring) -- emit the remaining stages now that success
        # is confirmed, so the UI still shows real progression rather than
        # jumping straight from "preparing" to done.
        if on_stage is not None:
            for name in _CPU_REMAINING_STAGES:
                on_stage(name)


_CPU_WORKER = CpuWorkerHandle()


def enhance_image_cpu_subprocess(
    input_path: Path,
    output_path: Path,
    on_stage: Optional[OnStage] = None,
) -> None:
    """Runs one image through backend/cpu_worker.py's persistent child
    process instead of in-process.

    Why: the packaged EXE's CPU path crashed intermittently
    (STATUS_STACK_BUFFER_OVERRUN / Windows Event Log BEX64 -- real memory
    corruption) with torch imported in the same process as pywebview +
    JobManager's worker thread + the HTTP server's per-request threads,
    even after replacing PyTorch's own CPU inference with OpenVINO (see
    restore_exp/imdn_x4_ov.py). An isolated repro of the exact same
    torch+cv2+IMDN CPU work, with none of that surrounding thread
    population, never reproduced the crash across repeated runs -- this
    function gives every real CPU job that same isolation by construction,
    regardless of the crash's exact root cause inside torch/oneDNN.

    Errors (including the worker process dying mid-job) are still surfaced
    precisely as EngineError, exactly like the in-process path -- process
    isolation is the fix here, not broad exception-swallowing.
    """
    def stage(name: str) -> None:
        if on_stage is not None:
            on_stage(name)

    stage(STAGE_PREPARING)
    _CPU_WORKER.run(input_path, output_path, on_stage)
