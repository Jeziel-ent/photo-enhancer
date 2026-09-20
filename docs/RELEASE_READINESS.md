# Release Readiness — Adinn 4K Image Enhancer

Snapshot of verified state as of this session. Only lists what was
actually run and observed on this machine — see each section's own
"verified" vs "not verified" framing.

## FINAL RELEASE CANDIDATE — full closeout (latest session)

Final rebuild and end-to-end verification of the release-candidate bundle
(`dist/Adinn4KImageEnhancer/`), built with the release-closeout spec
(see "PACKAGING" below). Everything below was run against the FINAL
bundle, not an intermediate build:

- **Full regression, re-run this session:** backend tests **121 passed +
  24 subtests passed**; correlation-fusion tests **17/17 pass**; frontend
  vitest **30/30 pass**; frontend `tsc -b --noEmit` and `vite` production
  build both exit 0.
- **Packaged GPU smoke (final bundle, real photo, real HTTP path):**
  launch -> `/health` 200 -> `/api/settings` resolved `effective=gpu` ->
  `POST /api/jobs` -> job **completed, 0 errors** (41s incl. cold start)
  -> `/api/jobs/<id>/result` downloaded == valid 3840x2162 RGB PNG
  (9.48 MB) -> `POST /api/jobs/<id>/export` batch export == valid ZIP
  containing `2_enhanced.png` (9.31 MB). Pass.
- **Packaged CPU smoke (final bundle, same photo, device pre-set to `cpu`
  on disk before launch):** boot resolved `effective=cpu`, a real
  `--cpu-worker` child subprocess was spawned, job **completed, 0 errors**
  (11.8s), result == valid 3840x2162 RGB PNG (8.49 MB — different from the
  GPU output, as expected for the lighter CPU model). Pass. (The device
  preference is resolved once at boot — `PUT /api/settings` mid-session
  cannot hot-swap it, by design; `restart_required` is returned.)
- **Bundle audit (final bundle):** `torch_cuda.dll` (774 MB) +
  c10/c10_cuda/torch_cpu present, 18 CUDA runtime libs, all 4 shipping
  models present, all R&D weights excluded (see PACKAGING), stale MSVC
  DLLs absent, `frontend/dist/index.html` + `regions.json` present.
- **Final bundle size: 4.50 GB** (down from 4.88 GB after R&D-weight
  exclusions).

## ENGINE — correlation-gated fusion integration (latest session)

The GPU production chain's F3 stage (`D1 -> F3-natural -> G7 -> A+`) was
replaced with a validated correlation-gated fusion mechanism
(`D1 -> correlation_fusion -> G7 -> A+`) in `image_enhancer/src/enhance.py`
— a multi-week research arc's conclusion (see
`image_enhancer/reports/research/correlation_fusion_final_validation/`
for the full robustness sweep and production-integration report). Gates
on **local correlation** between D1's and faithful's own band-passed
structure (genuine texture correlates with faint real source structure;
noise/SR-invention does not) rather than F3's edge-gradient-magnitude
gate. Parameters (sigma=3.0, t0=0.30, w=2.0) confirmed inside a stable,
artifact-free neighborhood by a staged 3x3x3 robustness sweep across all
6 real benchmark photos, not a fragile single-point optimum. G7 and A+ are
byte-for-byte unmodified; the CPU pipeline
(`enhance_shared.py`/`cpu_worker.py`) is a genuinely separate code path
and is also unmodified — confirmed by reading the module, not assumed.

**Regression tests**: new `image_enhancer/tests/test_correlation_fusion.py`
(17 checks: dimensions, finite/range, determinism, two true identity
cases, the documented +-18.0 envelope bound, chroma-untouched, constant
sanity, GPU-free production-wiring check) — all pass. Full backend suite
**121 passed, 24 subtests passed** — baseline preserved exactly (expected:
`backend/tests` never imports `enhance.py` internals).

**Real GPU validation**: both a direct 6-image engine regression
(`test_final_pipeline.py`, 44/44 checks) and the actual production job
queue (`JobManager` -> `engine_adapter.enhance_image`: 5 sequential jobs +
1 six-image batch, all completed, 0 errors, ~22-48s/image, 1.42 GB peak
VRAM) both pass against the modified engine.

**Packaged rebuild**: required a spec update first — this session's new
R&D directories (`research/`, `d1_res_exp/`, etc.) were not yet in
`adinn.spec`'s exclusion list and would otherwise have shipped in the
bundle; added them, verified absent from the rebuilt `_internal/`.
Packaged GPU (1 single + 1 three-image batch) and packaged CPU (1 job,
restart-applied device preference) validation both pass against the
freshly rebuilt bundle, as do the manual-adjustment and batch-export
endpoints the UI's sliders/export button call. Full detail:
`image_enhancer/reports/research/correlation_fusion_final_validation/PRODUCTION_INTEGRATION.md`.

**Not covered**: interactive GUI click-through (dragging sliders/drawing
board-editor rectangles in the actual rendered window) — no GUI-automation
tool is available in this environment. Every HTTP endpoint those UI
actions call was validated directly instead; a short manual pass over the
rendered window is recommended before shipping.

## PACKAGING

**PyInstaller result:** succeeds cleanly from a clean build state
(`packaging/build.ps1` steps 1-2), producing
`dist/Adinn4KImageEnhancer/Adinn4KImageEnhancer.exe` + `_internal/`.

**Stale DLL root cause (found and fixed this session):** the build
bundled a stale Microsoft Visual C++ redistributable
(`msvcp140.dll`/`vcruntime140.dll`/`vcruntime140_1.dll`, version
**14.36.32532.0**) at the flat `_internal` root, collected transitively by
one of the `collect_all()` calls (pythonnet/cv2/timm). Because PyInstaller's
bootloader adds `_internal` to the process DLL search path via
`AddDllDirectory`, and those directories are searched **before**
`System32`, the stale bundled copy shadowed the correct, current
system-installed copy (**14.50.35719.0** on this machine) — causing
torch's `c10.dll` to fail initialization (`WinError 1114`) the moment
`enhance.py` is imported inside the frozen process. This silently wedged
every GPU job at `"queued"` forever with no visible error. Full diagnostic:
`image_enhancer/reports/packaged_gpu_dll_diag/REPORT.md`.

**Permanent spec fix:** `packaging/pyinstaller/adinn.spec`'s existing
`_DROP_BUNDLED_NAMES` exclusion set (the same mechanism already used to
drop OpenCV's unused FFmpeg backend and Pillow's AVIF codec) now also
drops these 3 filenames by exact basename, letting the app fall through to
the system-installed copy instead. Validated with a **clean rebuild from
the modified spec** (not a post-hoc file removal) — the 3 files are
confirmed absent from the fresh `_internal/` root, and no plain-named
duplicate replaced them (only numpy's own pre-existing, differently-named
hash-suffixed private copies remain, e.g. `numpy.libs\msvcp140-<hash>.dll`,
which don't collide by exact filename and were never part of this issue).

**Final bundled model list** (`_internal/image_enhancer/models/`):
| File | Size | Used by |
|---|---|---|
| `swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_PSNR.pth` | 65 MB | GPU SR backbone |
| `restore/real_denoising.pth` | 100 MB | GPU Restormer denoise |
| `restore/motion_deblurring.pth` | 100 MB | Billboard/Candidate-A path (currently dead code — boxes always neutralized — kept defensively; see `docs/ENGINE_AUDIT.md`) |
| `imdn_x4_256.onnx` | 2.8 MB | CPU-path SR (OpenVINO) |

**Final bundle size:** **4.50 GB** (onedir, uncompressed on disk; down
from 4.88 GB after this closeout's model exclusions — see below).

**Spec update (latest session):** `_SRC_RESEARCH_DIRS` extended to
exclude this session's new R&D directories (`d1_res_exp`,
`d1_residual_exp`, `f3_clip_exp`, `f3_diag_exp`, `f3_env_w_exp`,
`f3_t0_exp`, `post_swinir_fusion_diag_exp`, `research/`) — see ENGINE
section above. A clean rebuild from this updated spec first failed
partway through analysis (exit 1, no traceback) on a machine with ~3.6 GB
free RAM at the time; an immediate retry completed successfully (exit 0).
Worth knowing if a CI/build machine hits the same failure under memory
pressure — it was not a code defect.

**Spec update (release closeout, latest session):** `_MODELS_EXCLUDED`
extended to drop the remaining R&D experiment weights from the bundle —
`models/EDSR_x4.pb` (36.8 MB), `models/pan/pan_4x_tile256_int8.bin` +
`.xml`, `models/real_drct/Real_DRCT_GAN_SRx4_mse_net_g_latest` (234.2
MB); `_SRC_RESEARCH_DIRS` extended with `restore_exp/carn_pan_exp`
(which carried a duplicate `real_denoising.pth`-style checkpoint). CAUTION:
`real_denoising.pth` by that bare name must NOT be added to
`_MODELS_EXCLUDED` — the required `models/restore/real_denoising.pth`
(100 MB) ships under that exact name. Final bundle audited: all four
shipping models present (`swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_PSNR.pth`,
`restore/real_denoising.pth`, `restore/motion_deblurring.pth`,
`imdn_x4_256.onnx`) and every excluded artifact absent.

**GPU packaged verification (this session, clean-rebuilt bundle):**
1 single job + 5 sequential jobs + 1 three-image batch — **all completed
successfully**, correct aspect-preserving dimensions, all pixels finite,
clean stderr (no exceptions).

**CPU packaged verification (this session, same clean-rebuilt bundle):**
direct `--cpu-worker` invocation + one HTTP-path 2-image batch (forced via
`PUT /api/settings`) — **both completed successfully**, output unaffected
by the DLL exclusion (CPU path never imports torch — see
`docs/ARCHITECTURE.md`'s CPU execution path section).

**Packaged smoke re-run (release closeout, FINAL bundle):** GPU job +
batch export and a CPU job (device pre-set on disk) all pass — see the
"FINAL RELEASE CANDIDATE" section at the top for full detail.

**Not yet built/verified:** the Inno Setup `.exe` installer itself —
`ISCC.exe` (Inno Setup 6) is **not installed** on this dev machine
(checked `%LOCALAPPDATA%\Programs\Inno Setup 6`, `%ProgramFiles`,
`%ProgramFiles(x86)`, and PATH). PyInstaller output is packaging-verified;
the script, prereq staging, and config under `packaging/inno/` are written
but the compiled installer has never been produced or run. Per
instructions, Inno Setup was NOT installed automatically to build it.

## VC++ RUNTIME

**Required version:** a current Microsoft Visual C++ 2015-2022
Redistributable (x64) — specifically, whatever version torch's `cu128`
build was linked against. This machine's working system copy is
**14.50.35719.0**; the (now-excluded) stale bundled copy was 14.36.32532.0.
Torch's own `_load_dll_libraries()` code (`torch/__init__.py`) already
prints official guidance when this redistributable is entirely missing,
pointing to Microsoft's official download:
`https://aka.ms/vs/17/release/vc_redist.x64.exe`.

**Does the installer bundle/verify it?** **Yes, implemented this
session** (release closeout): the official `vc_redist.x64.exe`
(**14.44.35211.0**, downloaded from `https://aka.ms/vs/17/release/vc_redist.x64.exe`,
Authenticode-signed by Microsoft, SHA-256
`CC0FF0EB1DC3F5188AE6300FAEF32BF5BEEBA4BDD6E8E445A9184072096B713B`) is
staged at `packaging/vc_redist/vc_redist.x64.exe` (gitignored). `adinn_setup.iss`
now: stages it via a `Flags: dontcopy` file entry, checks the system
copy's version in
`HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64` against the
required minimum (14.44.35211), and runs it silently
(`/install /quiet /norestart`) from `PrepareToInstall` only when the
system copy is older/missing; the "System Requirements" page also names
the VC++ prerequisite. `packaging/build.ps1` fails fast up front if the
staged redist is missing. **Limitation:** the installer itself cannot be
compiled on this machine (no ISCC.exe), so the bundled-VC++ install path
has NOT been exercised end-to-end — the script is written and statically
validated only.

**Clean-machine status:** **not verified this session** (no clean Windows
machine/VM was available — see Phase 7/Known Limitations below). This
machine already has a current redistributable installed (14.50.35719.0,
>= the staged minimum), which is why runtime validation succeeds here but
does not by itself prove a genuinely clean machine (one with no VC++
redistributable ever installed by anything) would work.

## GPU (clean-rebuilt packaged bundle, RTX 4050 Laptop GPU)

| Test | Result | Timing |
|---|---|---|
| Single job (cold start) | completed | 67s |
| 5 sequential jobs (warm) | all completed | 23-27s each |
| 3-image batch | completed | 73s total (~24.3s/image) |
| Peak VRAM (from the prior integration-test session, same hardware class) | — | 1358.9 MB |

Output validity: every job produced a correctly aspect-preserving,
4K-longest-side, all-finite, non-degenerate PNG (spot-checked by decode +
`np.isfinite` + std-deviation check).

## CPU (clean-rebuilt packaged bundle)

| Test | Result | Timing |
|---|---|---|
| Direct `--cpu-worker` single image | completed | (stage-by-stage; not separately timed this run) |
| 2-image batch (full HTTP/JobManager path) | completed | 20s total (~10s/image) |

Note: CPU-path timing observed this session (~10-30s/image across
different runs) is close to, sometimes faster than, GPU warm timing on
this specific machine (a fast CPU + mobile GPU) — this is a
machine-specific timing observation, not a claim that CPU output quality
matches GPU (it does not; CPU uses a lighter model by design — see
`docs/ARCHITECTURE.md`).

## Job-hang safety fix (this session)

Separately from the DLL fix, found and fixed a related **real** bug while
verifying jobs can't hang indefinitely on a GPU init failure:
`backend/engine_adapter.py`'s `warmup_engine()` and `enhance_image()` both
only caught `ImportError` around `_import_engine()` — but a native
DLL/CUDA init failure inside `import torch` raises `OSError`, not
`ImportError`. Before this fix, that specific exception type would have
killed `JobManager._run()`'s background thread outright (an uncaught
exception on a daemon thread), silently wedging **every** future job at
`"queued"` forever with zero user-visible error — exactly the failure mode
the DLL bug itself produced. Broadened both catches to `except Exception`,
matching what each function's own docstring already promised
("best-effort... any failure here is swallowed" / wraps failures as a
clean `EngineError`). Two new regression tests
(`backend/tests/test_engine_adapter.py::EngineImportFailureDoesNotWedgeTheQueueTestCase`)
mock `_import_engine` to raise `OSError` and assert neither function lets
it propagate/leak unwrapped. Full suite: **121/121 passing**. No UI
change; no change to any successful-path behavior.

## KNOWN LIMITATIONS

Only limitations actually verified this session:

1. **Inno Setup installer not built/run** — `ISCC.exe` (Inno Setup 6) is
   not installed on this dev machine (the tool was deliberately NOT
   auto-installed per instructions); PyInstaller output only. The
   installer script, VC++ prerequisite staging, and setup docs are written
   but the compiled `.exe` has never been produced or executed, so the
   VC++ install path is statically validated only.
2. **VC++ Redistributable install path not exercised end-to-end** — the
   mechanism is now implemented (staged official `vc_redist.x64.exe` +
   version-gated silent install in `adinn_setup.iss`), but without a
   compiled installer the actual clean-machine silent-install cannot be
   run here; script correctness is by inspection only.
3. **Clean-machine install not verified** — no clean Windows machine/VM
   was available this session. This machine has a current redistributable
   (14.50.35719.0) and a working GPU; if/when a clean machine becomes
   available, it should be the first thing tested (installer -> clean
   launch -> GPU job -> VC++ detection path).
4. **`backend/README.md` documentation staleness** (noted in the prior
   product-state audit, not this session's packaging work): still
   describes an early pre-implementation state in places, inconsistent
   with the current `README.md`/`docs/ARCHITECTURE.md`. Not in this
   session's explicit scope; flagged for a future doc pass.
5. **`motion_deblurring.pth` (100 MB) ships but is unreachable** in this
   product's current configuration (billboard/Candidate-A path always
   neutralized) — kept defensively per `docs/ENGINE_AUDIT.md`'s own
   reasoning, not a packaging bug.
6. **R&D weights exclusion (KNOWN LIMITATION #4 from prior session) is
   RESOLVED** — see PACKAGING spec-update note: `EDSR_x4.pb`,
   `pan_4x_tile256_int8.{bin,xml}`, the Real-DRCT checkpoint, and the
   `restore_exp/carn_pan_exp` source checkpoint dir are all excluded from
   the final bundle (4.88 GB -> 4.50 GB).
