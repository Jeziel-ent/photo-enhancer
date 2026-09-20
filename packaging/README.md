# Packaging — Windows installer

Turns the existing pywebview + React/Vite + Python app into a self-contained
Windows installer: end users need **no** Python, Node.js, or terminal
commands. See `docs/ARCHITECTURE.md` for the app architecture this packages
unchanged; this directory only adds a build/distribution layer on top of it.

## Layout

| Path | Purpose |
|---|---|
| `pyinstaller/adinn.spec` | PyInstaller build spec — turns the app into a onedir bundle (`dist/Adinn4KImageEnhancer/`). |
| `pyinstaller/launcher.py` | Tiny entry point PyInstaller actually runs: sets `ADINN_INSTALLED=1`, creates the "already running" mutex, then calls the *existing* `backend.shell.main()` — no startup logic is reimplemented. |
| `inno/adinn_setup.iss` | Inno Setup 6 script — wizard flow, system-requirements page, components/shortcuts, uninstaller, and the VC++ runtime prerequisite step. Compiles `dist/Adinn4KImageEnhancer/` into one installer `.exe`. |
| `vc_redist/vc_redist.x64.exe` | **Official** Microsoft Visual C++ 2015-2022 Redistributable (x64), staged for the installer's prerequisite step (gitignored — see below). |
| `assets/adinn.ico` | App/installer icon, generated from `frontend/src/assets/adinn-icon-512.png` (Pillow — already a project dependency, no new one added). |
| `assets/wizard_image.bmp`, `wizard_small.bmp` | Wizard branding bitmaps, generated the same way. |
| `build.ps1` | Orchestrates all three steps (frontend → PyInstaller → Inno Setup) in one command. |
| `output/` | Compiled installer lands here (gitignored). |

## One-time build-machine setup

(Only needed on the machine *building* the installer — never on an end
user's machine.)

```powershell
# PyInstaller into the project's existing venv
.venv\Scripts\python.exe -m pip install pyinstaller

# Inno Setup 6
winget install --id JRSoftware.InnoSetup -e

# Stage the official Microsoft VC++ Redistributable with the installer
# (packaging\vc_redist\vc_redist.x64.exe). Download it ONLY from Microsoft's
# own link, then verify before first use (see below):
Invoke-WebRequest -Uri "https://aka.ms/vs/17/release/vc_redist.x64.exe" -OutFile "packaging\vc_redist\vc_redist.x64.exe"
```

## Build

```powershell
.\packaging\build.ps1
```

or step by step:

```powershell
cd frontend; npm install; npm run build; cd ..
.venv\Scripts\python.exe -m PyInstaller packaging\pyinstaller\adinn.spec --noconfirm --distpath dist --workpath build
"$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe" packaging\inno\adinn_setup.iss
```

Output: `packaging\output\Adinn4KImageEnhancer-Setup-0.1.0.exe`.

## VC++ runtime prerequisite (required on the target machine)

The packaged app's frozen torch/CUDA stack links against the **Microsoft
Visual C++ 2015-2022 Redistributable (x64)** (`msvcp140.dll` /
`vcruntime140.dll` / `vcruntime140_1.dll`). The stale private copies that
previously shipped *inside* the bundle are deliberately excluded (see
`docs/RELEASE_READINESS.md` -> VC++ RUNTIME for why), so a machine that never
had a current runtime installed by anything would otherwise fail with a
missing-DLL error the first time a GPU job imports the engine.

The installer therefore:

1. **Stages the official redistributable** — `packaging\vc_redist\vc_redist.x64.exe`,
   bundled into the installer payload (`Flags: dontcopy`, staged to `{tmp}`
   at install time, never installed as part of the app, never left on disk).
2. **Checks the registry first** — `HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64`
   (Major/Minor/Bld). If an installed runtime `>= 14.44.35211` (the build
   staged with this release) already exists, the redistributable is **not**
   run at all.
3. **Installs silently only when required** — `vc_redist.x64.exe /install /quiet /norestart`
   from the staged official binary, before any app file is written; a
   non-zero exit aborts the install with the manual download link shown.

The staged binary is **gitignored** (a 24 MB third-party signed artifact —
see `.gitignore`: `packaging/vc_redist/`). Before first use on a build
machine, download it from Microsoft's own link and verify it:

```powershell
# Official source, Microsoft-signed:
Invoke-WebRequest -Uri "https://aka.ms/vs/17/release/vc_redist.x64.exe" -OutFile "packaging\vc_redist\vc_redist.x64.exe"
Get-AuthenticodeSignature "packaging\vc_redist\vc_redist.x64.exe" | Format-List SignerCertificate  # CN=Microsoft Corporation
Get-FileHash "packaging\vc_redist\vc_redist.x64.exe" -Algorithm SHA256   # 14.44.35211.0 = CC0FF0EB1DC3F5188AE6300FAEF32BF5BEEBA4BDD6E8E445A9184072096B713B
```

Do **not** source this file from a third-party website; do not silently run
anything else system-wide during install.

## Why PyInstaller + Inno Setup

- **PyInstaller** is the only realistic option for freezing a CPython app
  with a CUDA-enabled PyTorch stack (`basicsr`/`realesrgan`/`gfpgan`) into
  something that runs without a Python install — it's the standard, most
  maintained tool for this, and needed no new project dependency beyond
  itself (dev-machine only).
- **Inno Setup** is a mature, scriptable, free-for-this-use Windows
  installer builder that natively covers most of the required wizard (
  directory selection with disk-space checking, component selection,
  real per-file install progress, Start Menu/Desktop shortcuts, an
  uninstaller, upgrade detection) without extra frameworks — only the
  System Requirements page needed custom Pascal Script.
- Both are the two tools the task explicitly named as the preferred
  approach.

## What had to change in the app itself (installed-path compatibility only)

Nothing about the enhancement algorithms, the frontend UI, or the API
contract changed. Three small, additive changes were needed so the *same*
code works both from the source tree (dev) and once frozen/installed:

1. **`backend/_frozen.py`** (new) — `app_root()` returns the repo root when
   running from source, or PyInstaller's `sys._MEIPASS` when frozen.
   `backend/shell.py` and `backend/engine_adapter.py` now call this instead
   of hardcoding `Path(__file__).resolve().parents[1]`, which only happened
   to be correct in dev mode.
2. **`backend/workspace.py`** — `WORKSPACE_ROOT` (job input/output files +
   `recent_history.json`) now resolves to `%LOCALAPPDATA%\Adinn4KImageEnhancer\workspace`
   when the launcher sets `ADINN_INSTALLED=1`, instead of a folder next to
   the app's own files — Program Files isn't writable by a standard user.
   Dev/test behavior (no env var set) is completely unchanged.
3. **`packaging/pyinstaller/launcher.py`** (new, packaging-only) — sets that
   env var, creates the `Adinn4KImageEnhancerRunning` mutex Inno's
   `AppMutex` checks before install/uninstall, then calls
   `backend.shell.main()` unchanged.

`image_enhancer/src/enhance.py`'s own model-path resolution
(`Path(__file__).parent.parent / "models"`) needed **no change at all**:
`image_enhancer/src` is bundled as loose, uncompiled `.py` files (not frozen
into PyInstaller's archive) specifically so `engine_adapter.py`'s existing
`sys.path.insert(...); import enhance` mechanism — and every relative path
inside `enhance.py` — keeps working exactly as it does today.

## PyInstaller spec notes / gotchas hit and fixed

- `image_enhancer/src` is bundled as **loose data files**, not analyzed by
  PyInstaller — `engine_adapter.py` intentionally imports `enhance.py` via
  `sys.path` manipulation, which PyInstaller's static analyzer cannot trace
  into. Every third-party package `enhance.py` (and its `pro_exp`/
  `restore_exp`/`swinir` helpers) imports directly therefore has to be
  listed as a `hiddenimport` — found by statically scanning every `.py`
  file under `image_enhancer/src` for top-level imports (a one-off Python
  `ast` scan, not part of the shipped build) and cross-checking against
  what actually failed at runtime: `cv2`, `numpy`, `PIL`, `skimage`,
  `pandas`, `matplotlib`, `einops`, `h5py`, `natsort`, `requests`, `scipy`,
  `tqdm`, `yaml`.
- `basicsr`, `realesrgan`, `gfpgan`, and `timm` use a plugin/registry
  pattern (dynamic `importlib` scanning of their own package directory) —
  plain `hiddenimports` doesn't reach their submodules, so the spec uses
  `collect_all()` for each.
- `pythonnet` (pywebview's Windows WebView2/edgechromium backend dependency)
  ships its own embedded .NET runtime as plain data files
  (`pythonnet/runtime/*.dll`) with no PyInstaller hook of its own —
  `collect_all("pythonnet")` bundles them; without this, the packaged app's
  window failed to open (`WebViewException: You must have pythonnet
  installed`) even though `import pythonnet` worked fine.
- Encountered and worked around a **CPython 3.10.0** bug: PyInstaller's
  bytecode scanner (`dis.get_instructions`) raised
  `IndexError: tuple index out of range` while statically analyzing certain
  stdlib/third-party modules. Fixed at the interpreter level — upgraded the
  shared `Python 3.10.0` install to `3.10.11` via
  `winget upgrade --id Python.Python.3.10` (same minor version; the
  existing `.venv` and all installed packages kept working unmodified,
  verified by rerunning `pytest backend/tests` before and after — 55/55
  passed both times). This was a system-level, ABI-compatible patch
  upgrade, not a project dependency change.
- `pywebview`'s non-Windows platform backends (`cocoa`/`gtk`/`qt`/`android`/
  `cef`) are excluded — they're never used by this Windows-only shell and
  needlessly bloat/complicate analysis.

## Real end-to-end verification performed

With no GUI/mouse-click automation available in the environment this was
built in, verification used the packaged app's real local HTTP API instead
of clicking through the window — this exercises the identical code path
(`backend.server`/`backend.jobs`/`backend.engine_adapter`) a real click
would, just driven by a script instead of a mouse:

1. Built the frontend, ran PyInstaller, ran ISCC — all three succeeded.
2. Launched `dist\Adinn4KImageEnhancer\Adinn4KImageEnhancer.exe` directly —
   its pywebview window opened for real and loaded the real built frontend
   (confirmed a display/session was actually available).
3. Copied the onedir bundle to `C:\AdinnCleanInstallTest\` (a path outside
   the repo, simulating an installed location) and launched it from there —
   confirms path resolution is not an accident of running near the source
   tree.
4. POSTed a real photo (`image_enhancer/originals/2.jpeg`, then again with
   `5.jpeg` from the clean-directory copy) to the running instance's
   `/api/jobs`, polled real status/stage progress, and downloaded the
   result: both produced a genuine, valid **3840×2160 PNG** — visually
   inspected, correct content, no corruption.
5. Confirmed `%LOCALAPPDATA%\Adinn4KImageEnhancer\workspace\` held the job's
   input/output files (proving the Program-Files-safe redirect works), then
   deleted both the clean-directory copy and that AppData folder and
   confirmed both were fully gone — the same file footprint an uninstall
   removes.
6. Re-ran `pytest backend/tests` after every change (60/60 passing) to
   confirm none of the three source changes above regressed anything.

**Not verified** (needs a human with mouse/keyboard, or a Windows VM this
environment doesn't have): actually clicking through the installer wizard's
pages, the real elevated (`PrivilegesRequired=admin`) install into
`Program Files`, the Recent tab's native Save-As dialog end to end, and the
uninstaller's own UI. The underlying mechanisms for all of these (Inno's
native directory/disk-space page, the `DesktopBridge.save_result`/
`record_saved_result` methods, Inno's generated uninstaller) are either
exercised by existing automated tests (`backend/tests/test_desktop_bridge.py`,
`test_recent_history.py`) or are Inno Setup's own well-established built-in
behavior, not custom code — but a manual pass on a real machine before
shipping is still worth doing.

## Size optimization

Installed size dropped **5.20 GB → 4.68 GB** (−10.0%), installer **2.34 GB →
1.95 GB** (−16.6%), by tracing the real "final" pipeline's import graph line
by line (every `import`/`from` statement in `enhance.py`,
`tonal_correction.py`, `pro_exp/pipeline2.py`, `pro_exp/prolook.py`,
`restore_exp/restormer.py`, `restore_exp/restormer_arch.py`,
`restore_exp/swinir_m.py`, `swinir/network_swinir.py`) and removing exactly
what that graph never reaches. Nothing about image quality, the enhancement
algorithm, model checkpoints the "final" method actually loads, CUDA/GPU
behavior, or frontend functionality changed — confirmed by rebuilding and
rerunning `pytest backend/tests` (60/60) and the real 6-image GPU regression
suite (all checks passed) against this exact spec, plus a real GPU job
through the rebuilt packaged `.exe` producing a correct 3840×2160 PNG.

**Packages removed** (`adinn.spec`'s old `collect_all` list and
hiddenimports): `basicsr`, `realesrgan`, `gfpgan` — engine_adapter.py
hardcodes `METHOD="final"`, and the only things that import these three are
`get_realesrgan()`/`realesrgan_enhance()`/`tiled_enhance()` in `enhance.py`
(different, unused methods) — `gfpgan` isn't referenced anywhere in
`image_enhancer/src` at all. Their own declared dependencies (`scikit-image`,
`lmdb`, `tb-nightly`, `yapf`, `addict`, `future` — confirmed via their
`dist-info/METADATA` `Requires-Dist` lines) left with them. `matplotlib` and
`pandas` were also excluded: PyInstaller's own official `torch` hook
unconditionally does `collect_submodules("torch")`, which pulls in
`torch.utils.tensorboard.writer` (imports `matplotlib`) and a couple of
distributed/functorch visualization utilities we never call; nothing in the
real call graph touches `pandas` either (it's used only by R&D
benchmark/report scripts under `image_enhancer/src` that are bundled as
inert loose data, never executed). `lmdb` is used only by
`torchvision/datasets/lsun.py` (a dataset loader we never call).

**Packages kept despite not being directly imported by our own code**:
`scipy` — re-appeared automatically after removing it from our own
hiddenimports, because it turned out to be a genuine transitive dependency
of `timm` itself (`timm/layers/interpolate.py`,
`timm/layers/pos_embed_rel.py`), and `timm` is required
(`network_swinir.py: from timm.models.layers import ...`). `torchvision`,
`requests`, `tqdm`, `yaml` — kept defensively since `timm`/`huggingface_hub`
may reach for them internally and each costs under 1 MB.

**Models excluded** (`_MODELS_EXCLUDED` in `adinn.spec`, ~501 MB of the
764 MB packaged before): `RealESRGAN_x4plus.pth` (64 MB, only
`get_realesrgan()`), `single_image_defocus_deblurring.pth` (100 MB, only
`g2_exp/recipe_g2.py`), both SwinIR-**L** checkpoints (136 MB × 2, only
`swinir/experiment.py` / `g6_exp/recipe_g6.py` — `enhance.py`'s production SR
step uses `swinir_m.py`, whose own `PTH` constant names the "-M" checkpoint,
never "-L"), and the SwinIR-M-**GAN** checkpoint (65 MB, only
`enhance_g91.py` — a separate experimental module `engine_adapter` never
imports — `g9_exp/recipe_g9.py`, `swinir/experiment.py`), plus
`face_detection_yunet_2023mar.onnx` (228 KB, only `restore_exp/faces.py`;
"no face work" is a stated invariant of `final_enhance`'s own docstring).
Every exclusion was traced with `grep -rl <filename> image_enhancer/src/` —
none are referenced by anything `backend/` ever imports.

**Model kept despite being unreachable in today's shipped configuration**:
`motion_deblurring.pth` (100 MB) is used by `enhance.py`'s own
`_final_candidate_a_board` — the billboard/Candidate-A branch of the *same*
"final" method, just gated behind billboard boxes that
`engine_adapter._neutralize_billboard_regions` always forces to `[]` for
this product today. That's a product-configuration fact, not proof the
model is unreachable from `final_enhance`'s own code, so per the task's own
caution ("do not remove a model merely because it's unused in one
configuration") it was kept rather than risk a future billboard re-enable
silently breaking.

**Investigated but NOT applied** — the single largest theoretical
opportunity: `torch/lib/*.dll` is 4.0 GB (75% of the installed size), and
individual CUDA redistributable DLLs like `cusparse64_12.dll` (362 MB),
`cufft64_11.dll` (263 MB), `cusolver64_11.dll` (215 MB),
`cusolverMg64_11.dll` (150 MB), `nvperf_host.dll` (21 MB, profiling-only),
and `cupti64_2025.1.1.dll` (4.3 MB, profiling-only) are not referenced
anywhere in our own source (no sparse tensors, FFT, or `torch.linalg` calls
in the pipeline). Unlike every removal above, this can't be verified with
grep-level rigor: these are internal C++ dependencies inside a compiled,
official PyTorch wheel that PyTorch itself doesn't document or support
cherry-picking, and a missing one could fail silently on some rarely-hit
internal code path our 6-image regression doesn't happen to exercise —
exactly the kind of silent GPU-behavior change the task explicitly forbids
trading for size. Left alone; a dedicated, longer validation effort (ideally
with PyTorch's own guidance) would be needed before touching `torch/lib`.

Full size backup of the pre-optimization build was not kept on disk (it was
compared in place, then discarded) — the "before" numbers above come from
direct measurement at the time (see this session's tool output), not an
estimate.
