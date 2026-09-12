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

import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import Callable, Optional

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


def _run_with_stage_ticker(engine: ModuleType, img, on_stage: Optional[OnStage]):
    """Runs ``engine.enhance(METHOD, img)`` — the one real, unmodified call
    path into the engine — on a background thread, while this thread ticks
    ``on_stage`` through _STAGE_TIMELINE based on elapsed time. Returns
    ``(out, dt)`` exactly as engine.enhance() does, or re-raises whatever
    exception the engine call raised.
    """
    import time as _time

    done = threading.Event()
    result: dict = {}

    def _worker() -> None:
        try:
            result["out"], result["dt"] = engine.enhance(METHOD, img)
        except Exception as exc:  # noqa: BLE001 — re-raised on the caller's thread below
            result["exc"] = exc
        finally:
            done.set()

    worker = threading.Thread(target=_worker, daemon=True)
    t0 = _time.monotonic()
    worker.start()

    remaining = list(_STAGE_TIMELINE)
    while not done.wait(timeout=_TICK_SECONDS):
        elapsed = _time.monotonic() - t0
        while remaining and elapsed >= remaining[0][0]:
            _, stage_name = remaining.pop(0)
            if on_stage is not None:
                on_stage(stage_name)
    worker.join()

    if "exc" in result:
        raise result["exc"]
    return result["out"], result["dt"]


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
    processing reaches each milestone — see _run_with_stage_ticker for what
    "reaches" means for the stages inside the single opaque engine call.
    Optional and additive: omitting it reproduces the exact prior behavior.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    def stage(name: str) -> None:
        if on_stage is not None:
            on_stage(name)

    stage(STAGE_PREPARING)
    try:
        engine = _import_engine()
    except ImportError as exc:
        raise EngineError(f"image enhancement engine is unavailable: {exc}") from exc

    with _ENGINE_LOCK:
        try:
            img = engine.load_image(str(input_path))
        except Exception as exc:  # noqa: BLE001
            raise EngineError(f"could not read {input_path.name}: {exc}") from exc

        stage(STAGE_ANALYZING)
        prev_regions, prev_billboard = _neutralize_billboard_regions(engine)
        try:
            out, _dt = _run_with_stage_ticker(engine, img, on_stage)
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
