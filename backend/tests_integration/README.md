# backend/tests_integration/ — real end-to-end pipeline tests

These are **not** part of the normal test suite (`backend/tests/`). They
drive the actual production path over real HTTP with no mocking:

```
POST /api/jobs -> JobManager's real worker thread -> engine_adapter.enhance_image
    -> image_enhancer/src/enhance.py "final" method on the real GPU
    -> GET /api/jobs/<id>/result over real HTTP
```

using the real sample photos in `image_enhancer/originals/`. Each test takes
40–90+ seconds (real Real-ESRGAN/SwinIR/Restormer inference), so the whole
module takes several minutes.

Skipped by default — set `ADINN_RUN_GPU_TESTS=1` to opt in, even if this
directory is accidentally swept up by a broader `pytest backend` run:

```powershell
$env:ADINN_RUN_GPU_TESTS = "1"
..\image_enhancer\.venv\Scripts\python.exe -m pytest backend/tests_integration -v -s
```

`-s` is recommended so the per-test timing lines print live.

`backend/tests/` (the fast suite) must never depend on a GPU — that
constraint is what this separate directory exists to protect.
