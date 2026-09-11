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

REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_ENHANCER_SRC = REPO_ROOT / "image_enhancer" / "src"

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


def enhance_image(input_path: Path, output_path: Path) -> None:
    """Runs the production "final" pipeline on one image file.

    Reads ``input_path``, never writes to it, and writes the 4K enhanced
    result to ``output_path``. Raises EngineError (with a message safe to
    surface to the API caller) on any failure — the job manager treats that
    as one failed file, not a crash.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    try:
        engine = _import_engine()
    except ImportError as exc:
        raise EngineError(f"image enhancement engine is unavailable: {exc}") from exc

    with _ENGINE_LOCK:
        try:
            img = engine.load_image(str(input_path))
        except Exception as exc:  # noqa: BLE001
            raise EngineError(f"could not read {input_path.name}: {exc}") from exc

        prev_regions, prev_billboard = _neutralize_billboard_regions(engine)
        try:
            out, _dt = engine.enhance(METHOD, img)
        except Exception as exc:  # noqa: BLE001
            raise EngineError(f"enhancement failed for {input_path.name}: {exc}") from exc
        finally:
            _restore_billboard_regions(engine, prev_regions, prev_billboard)

        try:
            engine.save_image(out, str(output_path))
        except Exception as exc:  # noqa: BLE001
            raise EngineError(
                f"could not save enhanced output for {input_path.name}: {exc}"
            ) from exc
