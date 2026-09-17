"""Runs CPU-mode enhancement jobs in a standalone child process that NEVER
imports torch.

Why this exists -- the real root cause, found by direct stress-testing,
not guessed: the packaged/frozen Windows EXE crashed (STATUS_ACCESS_VIOLATION,
a genuine segfault, not a Python exception) during repeated CPU-mode jobs.
Two isolation steps narrowed it precisely:

1. A plain script calling OpenVINO's IMDN inference (imdn_x4_ov.py) alone,
   20 times in a loop, with torch never imported in that process: 20/20
   clean.
2. The SAME script, but importing ``enhance`` first (which imports torch at
   module scope) before doing the identical OpenVINO calls: crashed within
   4-10 calls, every time this was tried.

Conclusion: torch and OpenVINO's native CPU runtimes cannot safely coexist
in the same process on this project's stack (almost certainly a conflicting
OpenMP/TBB/MKL thread-pool initialization between the two libraries) --
this is not about pywebview, JobManager, or thread counts, all of which
were tried and ruled out first. The fix is structural: this worker imports
``enhance_shared`` (image_enhancer/src/enhance_shared.py) and
``restore_exp/imdn_x4_ov.py`` directly -- cv2/numpy/OpenVINO only, never
``enhance.py``, never torch -- so the two runtimes are never in the same
process at all. The GPU path (enhance.py, imported only by
backend/engine_adapter.py's in-process call) is completely unaffected.

Protocol: a PERSISTENT process, started once (lazily, on the first CPU job;
see engine_adapter._CPU_WORKER) and reused for every subsequent CPU
job in the app's lifetime -- NOT spawned fresh per job. That matters:
importing enhance.py (torch, cv2, timm, ...) alone measured ~15s cold on
this project's target hardware, on top of which OpenVINO's own model
compile adds more -- spawning a fresh process per job would blow the
25-27s budget on nearly every image, not just the first. A persistent
worker pays that cost once, then each job after the first runs in the
same ~5-7s this project already measured for the in-process CPU path,
while still keeping the actual per-job inference in a process that never
shares pywebview/JobManager/HTTP-server threads with the crash-prone host
process (see module docstring below for the full "why").

argv is empty; job requests arrive one per line on stdin as
``input_path\toutput_path``, and the worker loops forever processing them
until stdin closes (parent process exit) or a line is exactly "QUIT".
Progress is reported by printing one of engine_adapter.STAGES per line to
stdout as each stage is reached (flushed immediately), followed by either
"DONE" (success) or "ERROR: <message>" (failure) once that job finishes --
the parent (engine_adapter.py's CpuWorkerHandle) reads this line-by-line
and maps it back to the specific job waiting on it. A worker that dies
mid-job (the very failure mode this isolation exists to contain) is
detected by the parent via EOF/process exit and respawned fresh for the
next job -- so one bad job never permanently breaks CPU mode for the rest
of the session.

Only ever used when the resolved processing device is CPU (see
engine_adapter.apply_device_preference / get_device_state) -- the GPU path
is completely unaffected and continues to run in-process, unchanged.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent.parent


def _write_status(output_path: str, ok: bool, message: str = "") -> None:
    """Writes the AUTHORITATIVE done/error signal for one job, as a small
    JSON sidecar file next to the output image -- not via stdout.

    Why not stdout: a stdout-pipe protocol (print() lines the parent reads
    with ``for line in proc.stdout``) was tried first and measured to
    intermittently stop delivering lines mid-job even though the worker
    process was still alive and had NOT crashed (confirmed: the parent saw
    the pipe end without a DONE/ERROR line while proc.poll() still
    returned None) -- almost certainly some native library used during
    model load (OpenVINO/oneDNN's own logging setup is a common culprit
    for this class of bug) doing low-level stdout file-descriptor
    manipulation that Python's io.TextIOWrapper doesn't see, desyncing the
    parent's read from the child's actual writes. A file the OS filesystem
    layer -- not a library that might repoint an fd -- mediates has no
    equivalent failure mode. stdout prints are kept alongside this for
    best-effort human-readable logging only; nothing depends on them
    arriving.
    """
    import json
    import tempfile
    status_path = output_path + ".status.json"
    payload = json.dumps({"ok": ok, "message": message})
    fd, tmp_path = tempfile.mkstemp(dir=str(Path(output_path).parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp_path, status_path)
    except Exception:  # noqa: BLE001 -- best-effort cleanup if the write itself failed
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _run_one(shared, input_path: str, output_path: str) -> None:
    """Runs one image through the CPU pipeline (enhance_shared.py --
    cv2/numpy/OpenVINO only, never torch; see module docstring). The
    parent's authoritative signal is the ``<output_path>.status.json``
    sidecar file written by _write_status -- see its docstring for why.
    stdout stage/DONE/ERROR lines are still printed for human-readable
    logs, best-effort only."""
    def stage(name: str) -> None:
        # Stage markers correspond 1:1 to engine_adapter.STAGES; duplicated
        # as plain strings here rather than imported from engine_adapter to
        # keep this worker importable/runnable with zero backend/ package
        # dependencies (it must work as a bare argv-invocation from a
        # frozen EXE that never otherwise imports the backend package in
        # this process).
        print(name, flush=True)

    stage("preparing")
    try:
        img = shared.load_image(input_path)
    except Exception as exc:  # noqa: BLE001
        msg = f"could not read {Path(input_path).name}: {exc}"
        print(f"ERROR: {msg}", flush=True)
        _write_status(output_path, ok=False, message=msg)
        return

    stage("analyzing")
    try:
        out = shared.cpu_final_enhance(img)
    except Exception as exc:  # noqa: BLE001
        msg = f"enhancement failed for {Path(input_path).name}: {exc}"
        print(f"ERROR: {msg}", flush=True)
        _write_status(output_path, ok=False, message=msg)
        return
    # cpu_final_enhance runs to completion in one call (unlike the GPU
    # path's ticker, which estimates progress through an opaque call by
    # elapsed time) -- CPU mode is fast enough (see
    # docs/PERFORMANCE_OPTIMIZATION.md) that emitting the remaining stage
    # markers right after completion, rather than trying to instrument
    # mid-pipeline callbacks across a process boundary, is a reasonable,
    # simple tradeoff.
    stage("restoring")
    stage("enhancing_details")
    stage("upscaling")
    stage("finalizing")

    try:
        shared.save_image(out, output_path)
    except Exception as exc:  # noqa: BLE001
        msg = f"could not save enhanced output for {Path(input_path).name}: {exc}"
        print(f"ERROR: {msg}", flush=True)
        _write_status(output_path, ok=False, message=msg)
        return

    print("DONE", flush=True)
    _write_status(output_path, ok=True)


def main(argv: list[str]) -> int:  # noqa: ARG001 -- argv unused, kept for symmetry with __main__
    # subprocess.Popen's bufsize=1 (line buffering) is documented as
    # unreliable for text-mode PIPES on Windows -- explicitly reconfiguring
    # this process's own stdout for line buffering is the robust fix (and
    # is what actually made the parent's line-by-line reads reliable; see
    # docs/PERFORMANCE_OPTIMIZATION.md's CPU stability section).
    sys.stdout.reconfigure(line_buffering=True)

    # Never touch CUDA in this process, regardless of what the parent's
    # environment looks like -- this worker only ever exists to run the
    # CPU pipeline. (The parent already sets this before spawning, this is
    # defense in depth so the worker is correct in isolation too.)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    repo_root = _repo_root()
    sys.path.insert(0, str(repo_root / "image_enhancer" / "src"))

    # No billboard/regions handling here at all (unlike enhance.py's GPU
    # path, which engine_adapter._neutralize_billboard_regions must
    # actively suppress) -- enhance_shared.cpu_final_enhance never has a
    # billboard/Candidate-A branch to begin with (see its own docstring),
    # so there is nothing to neutralize.
    try:
        import enhance_shared as shared  # noqa: E402 -- cv2/numpy/OpenVINO only, no torch
    except Exception as exc:  # noqa: BLE001
        # No job line to answer yet -- report on the first one requested.
        print(f"ERROR: image enhancement engine is unavailable: {exc}", flush=True)
        shared = None  # type: ignore[assignment]

    for line in sys.stdin:
        line = line.strip()
        if not line or line == "QUIT":
            break
        if shared is None:
            print("ERROR: image enhancement engine failed to load at worker startup",
                  flush=True)
            continue
        if line.startswith("WARMUP\t"):
            # Pre-loads the model and pre-compiles OpenVINO's graph (see
            # enhance_shared.cpu_warmup's own docstring) now, while the app
            # is still starting up, instead of on whichever image is
            # processed first -- mirrors the GPU path's own
            # warmup_engine() timing. ``status_base`` is caller-supplied
            # (like a real job's output_path) rather than a hardcoded
            # location, for the same file-based-signal reason
            # _write_status exists at all -- _write_status appends
            # ".status.json" to it, exactly like a real job.
            _, status_base = line.split("\t", 1)
            try:
                shared.cpu_warmup()
            except Exception as exc:  # noqa: BLE001 -- best-effort, like the GPU path's warmup
                print(f"ERROR: {exc}", flush=True)
                _write_status(status_base, ok=False, message=str(exc))
                continue
            print("DONE", flush=True)
            _write_status(status_base, ok=True)
            continue
        try:
            input_path, output_path = line.split("\t", 1)
        except ValueError:
            print(f"ERROR: malformed job request: {line!r}", flush=True)
            continue
        _run_one(shared, input_path, output_path)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
