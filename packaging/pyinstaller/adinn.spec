# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build spec for the Adinn 4K Image Enhancer desktop app.

Build (from the repo root, using the same venv the app already runs from):

    .venv\\Scripts\\python.exe -m PyInstaller packaging\\pyinstaller\\adinn.spec --noconfirm

Produces a one-dir bundle at dist/Adinn4KImageEnhancer/ containing:
  - Adinn4KImageEnhancer.exe (the launcher -> backend.shell.main())
  - the full Python runtime + all third-party deps (torch/cv2/basicsr/...)
  - image_enhancer/src (loose .py -- engine_adapter.py's sys.path+import
    "enhance" as a flat module needs real files on disk here, not a frozen
    archive) and image_enhancer/models (the AI model weights)
  - frontend/dist (the built React UI backend.shell serves)

Design notes:
  - backend/*.py IS analyzed/compiled normally (it's a real package with no
    unusual import tricks of its own) -- only image_enhancer/src is shipped
    as loose data files, because engine_adapter.py intentionally imports
    enhance.py via sys.path manipulation + `import enhance`, not a normal
    package import PyInstaller's static analyzer could trace.
  - Nothing in image_enhancer/ or backend/'s existing logic is changed by
    this file; the only source changes made for packaging were
    backend/_frozen.py (resolves bundled-resource paths under
    sys._MEIPASS when frozen) and backend/workspace.py (redirects the
    job workspace + recent-history file to %LOCALAPPDATA% when installed,
    since Program Files isn't writable by a standard user).
"""

from pathlib import Path
from PyInstaller.utils.hooks import collect_all

REPO_ROOT = Path(SPECPATH).resolve().parents[1]

# --------------------------------------------------------------- model files
# Every packaged .pth/.onnx mapped to the exact code that loads it (see
# packaging/README.md "Size optimization" section for the full trace).
# engine_adapter.py hardcodes METHOD="final" and always neutralizes
# billboard regions, so enhance.final_enhance()'s whole-frame path -- 100%
# of what this product ever executes -- only ever touches checkpoints
# selected by torch.cuda.is_available() in _final_d1_weak: real_denoising.pth
# + the SwinIR-M "PSNR" checkpoint on GPU, or IMDN_x4.pth (classical
# fastNlMeansDenoisingColored needs no weight file) on CPU -- see
# _final_d1_weak's own docstring/comments in enhance.py for why CPU uses a
# different, lighter SR backbone (SwinIR-M measures ~200s+ for one image on
# CPU, an ~8x overshoot of this product's 25-27s budget that no safe
# precision/tiling/ONNX tuning closes; IMDN x4 is a plain CNN with no
# window-attention transformer and measures ~5s for the same stage). Both
# checkpoint sets must ship together since the packaged app must run on
# whatever hardware the end user's machine actually has.
_MODELS_DIR = REPO_ROOT / "image_enhancer" / "models"
_MODELS_EXCLUDED = {
    # Only referenced by get_realesrgan()/realesrgan_enhance()/tiled_enhance()
    # in enhance.py -- methods engine_adapter never calls (METHOD is always
    # "final").
    "RealESRGAN_x4plus.pth",
    # Only referenced by g2_exp/recipe_g2.py, a standalone R&D experiment
    # script never imported by enhance.py or anything backend/ touches.
    "single_image_defocus_deblurring.pth",
    # Only referenced by swinir/experiment.py and g6_exp/recipe_g6.py (R&D
    # experiments); enhance.py's production SR step uses swinir_m.py, whose
    # own PTH constant names the "-M" (Medium) checkpoint below, never "-L".
    "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth",
    "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_PSNR.pth",
    # Only referenced by enhance_g91.py (a separate experimental module
    # engine_adapter never imports), g9_exp/recipe_g9.py, and
    # swinir/experiment.py -- not enhance.py's production "M x4 PSNR" path.
    "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth",
    # Only referenced by restore_exp/faces.py, which nothing in the
    # production call graph imports -- "no face work" is a stated invariant
    # of enhance.final_enhance's own module docstring.
    "face_detection_yunet_2023mar.onnx",
    # --- Phase A (package-size audit) additions, all grep-verified to have
    # zero references anywhere in the production call graph (enhance.py's
    # "final" METHOD path + the OpenVINO CPU worker) ---
    # R&D-only SR checkpoints (g1_exp..g9_exp / f*_exp recipe trees) --
    # nothing backend/ or enhance.py's production path ever loads them.
    "DRCT-L_SRx4_ImageNet-pretrain.pth",
    "Real_DRCT_GAN_SRx4_mse_net_g_latest.zip",
    "EDSR_x4.pt",
    "DAT_S_x4.pth",
    "BSRNet.pth",
    # IMDN_AS.pth: R&D-only variant. IMDN_x4.pth: the LEGACY PyTorch IMDN
    # checkpoint -- the CPU/OpenVINO path loads the ONNX export
    # (imdn_x4_256.onnx), never this .pth (the PyTorch IMDN source module
    # was removed from the repo; see docs/PERFORMANCE_OPTIMIZATION.md).
    "IMDN_AS.pth",
    "IMDN_x4.pth",
    # Only referenced by the pan_* benchmark scripts (which ship under
    # image_enhancer/src only as inert research data).
    "pan_4x.pt",
    # A 1.2+ GB root-level .zip sitting next to the real
    # 003_..._SwinIR-M_x4_PSNR.pth (same stem, different extension, appeared
    # a day after every other checkpoint's timestamp) -- grepped
    # image_enhancer/src for the literal filename and every importer: zero
    # references. Nothing in the production call graph (or anywhere else)
    # ever loads a .zip model file; the app only reads .pth/.onnx weights
    # directly. Excluded on the same "zero production references" basis as
    # every other entry in this set, not a change to which model WEIGHTS
    # ship -- the real .pth this appears to be a stray archive/backup of is
    # still bundled via image_enhancer/models/swinir/ below.
    "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_PSNR.zip",
}
# Root-level duplicate of models/swinir/003_...SwinIR-M_x4_PSNR.pth --
# byte-identical (same MD5) to the REQUIRED copy restore_exp/swinir_m.py
# actually loads (path = ROOT/"models"/"swinir"/PTH). Kept out by RELATIVE
# PATH (not filename, or the required swinir/ copy would be dropped too).
_MODELS_EXCLUDED_RELPATHS = {
    Path("003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_PSNR.pth"),
}
# Kept despite being unreachable in THIS product's shipped configuration:
# motion_deblurring.pth is used by enhance.py's own _final_candidate_a_board
# (the billboard/Candidate-A branch of the SAME "final" method), just gated
# behind billboard boxes that engine_adapter always neutralizes to [] today.
# That is a product-configuration fact, not proof the model is unreachable
# from final_enhance's own code -- kept rather than risk a future billboard
# re-enable silently breaking. (real_denoising.pth and the SwinIR-M "PSNR"
# checkpoint are the two the whole-frame path actually loads every run.)

model_datas = []
for f in _MODELS_DIR.rglob("*"):
    rel = f.relative_to(_MODELS_DIR)
    if f.is_file() and f.name not in _MODELS_EXCLUDED and rel not in _MODELS_EXCLUDED_RELPATHS:
        model_datas.append((str(f), str(Path("image_enhancer/models") / rel.parent)))

# image_enhancer/src is bundled as loose data (engine_adapter.py imports
# enhance.py via sys.path + `import enhance`, not a traceable package
# import). It is filtered to the PRODUCTION call graph only -- the R&D
# recipe trees (g*_exp, f*_exp) and standalone benchmark/experiment scripts
# are inert dead weight in the bundle (they reference modules like pandas/
# skimage that aren't even shipped) and are excluded here. The module list
# below is the conservative inverse of the audit: everything the production
# graph can reach (enhance, tonal_correction, enhance_shared, billboard,
# pro_exp/pipeline2 + prolook, restore_exp/{restormer, restormer_arch,
# swinir_m, imdn_x4_ov}, swinir/network_swinir) is KEPT; everything else
# under src/ is R&D-only and dropped.
_SRC_DIR = REPO_ROOT / "image_enhancer" / "src"
_SRC_RESEARCH_DIRS = {
    "f2_exp", "f3_exp", "f3_gate_exp", "f3_local_exp", "fidelity_exp",
    "g1_exp", "g2_exp", "g3_exp", "g4_exp", "g5_exp", "g6_exp", "g7_exp",
    "g8_exp", "g8_fix_exp", "g9_exp", "restore_exp/ref",
}
_SRC_RESEARCH_FILES = {
    "benchmark.py", "benchmark_final.py", "compare_report.py",
    "enhance_g91.py", "make_testscene.py", "pan_arch.py", "quality.py",
    "regression_harness.py", "verify_regions.py",
    "restore_exp/faces.py", "restore_exp/metrics_ext.py",
    "restore_exp/perspective.py", "restore_exp/pipeline.py",
    "restore_exp/plan_abc.py", "restore_exp/report.py",
    "pro_exp/d1_sweep.py", "pro_exp/d1_sweep_report.py", "pro_exp/montage.py",
    "pro_exp/rec_extract.py", "pro_exp/report.py", "pro_exp/run.py",
    "swinir/experiment.py", "swinir/main_test_swinir_ref.py",
}
src_datas = []
for f in _SRC_DIR.rglob("*"):
    if not f.is_file():
        continue
    rel = f.relative_to(_SRC_DIR)
    parts = rel.parts
    if "__pycache__" in parts:
        continue
    if any(str(Path(*parts[:i])).replace("\\", "/") in _SRC_RESEARCH_DIRS
           for i in range(1, len(parts) + 1)):
        continue
    if str(rel).replace("\\", "/") in _SRC_RESEARCH_FILES:
        continue
    src_datas.append((str(f), str(Path("image_enhancer/src") / rel.parent)))

datas = [
    (str(REPO_ROOT / "image_enhancer" / "regions.json"), "image_enhancer"),
    (str(REPO_ROOT / "frontend" / "dist"), "frontend/dist"),
] + src_datas + model_datas
binaries = []
# image_enhancer/src (enhance.py + its pro_exp/restore_exp/swinir helpers) is
# bundled as loose data below, never traced by PyInstaller's static analyzer
# -- so every top-level package IT imports directly (not just what
# backend/*.py imports) has to be listed here explicitly, or it's simply
# absent from the frozen build. This list was narrowed from an earlier,
# broader AST scan of every .py file under image_enhancer/src (which
# over-included R&D/benchmark-only scripts final_enhance's own call graph
# never reaches) down to exactly what enhance.py + tonal_correction.py +
# pro_exp/pipeline2.py + pro_exp/prolook.py + restore_exp/restormer.py +
# restore_exp/restormer_arch.py + restore_exp/swinir_m.py +
# swinir/network_swinir.py actually import, verified line by line.
hiddenimports = [
    "webview", "webview.platforms.edgechromium",
    "torch", "torchvision", "cv2", "numpy", "PIL",
    # restormer_arch.py; network_swinir.py.
    "einops", "timm",
    # Small, kept defensively: huggingface_hub/timm's own internal code may
    # reach for these even though nothing in our own call graph imports them
    # directly, and each costs well under 1 MB.
    "requests", "tqdm", "yaml",
]

# basicsr/realesrgan/gfpgan (and their own scipy/scikit-image/pandas/
# matplotlib-pulling dependency trees) were removed from this build entirely:
# confirmed by tracing the real "final" pipeline's import graph line by line
# that engine_adapter.py's one hardcoded call path (METHOD="final") never
# reaches get_realesrgan()/realesrgan_enhance()/tiled_enhance() (the only
# things that import realesrgan/basicsr) or gfpgan (unreferenced anywhere in
# image_enhancer/src). Validated by rebuilding and rerunning the full test
# suite + real 6-image GPU regression against this exact spec -- see
# packaging/README.md.
#
# pythonnet ships its own embedded .NET runtime (pythonnet/runtime/*.dll) as
# plain data files with no dedicated PyInstaller hook, needed for
# pywebview's Windows edgechromium/WebView2 backend (clr_loader's own hook
# only grabs clr_loader's own native loader, not pythonnet's runtime
# payload). cv2 (opencv-python) ships many native DLLs of its own that
# plain hiddenimports don't reliably pull in, so it gets the same
# full-collection treatment. timm uses a registry-ish layout too.
#
# openvino (the CPU/no-GPU execution backend for IMDN x4 -- see
# restore_exp/imdn_x4_ov.py's module docstring for why PyTorch's own CPU
# backend was replaced with it) ships its inference-engine core as native
# DLLs plus a device-plugin registry (plugins.xml + one DLL per backend,
# e.g. openvino_intel_cpu_plugin.dll) that a plain hiddenimports entry
# would miss entirely -- there is no pyinstaller-hooks-contrib hook for it
# (checked), so it needs the same full-collection treatment as the others
# here, verified by actually running the frozen CPU path, not assumed.
for pkg in ("pythonnet", "cv2", "timm"):
    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(pkg)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hiddenimports

# openvino (the CPU/no-GPU execution backend for IMDN x4 -- see
# restore_exp/imdn_x4_ov.py's module docstring) is collected differently:
# collect_all(), then the native payload is PRUNED to exactly what the
# OpenVINO -> IMDN ONNX -> CPU path loads. The stock wheel ships every
# device backend and every model frontend; only the CPU plugin and the ONNX
# frontend are ever exercised here (the module does `import openvino as ov`
# + `ov.Core().read_model(imdn_x4_256.onnx)` + compile FOR CPU). OpenVINO
# discovers device plugins and frontends dynamically from openvino/libs,
# and openvino/__init__.py only eagerly imports openvino.frontend.frontend
# (not the per-framework python frontends), so dropping the unused native
# DLLs does not break `import openvino`. Kept: openvino.dll (core),
# openvino_intel_cpu_plugin.dll, openvino_onnx_frontend.dll, the oneTBB
# runtime the CPU plugin needs, and the python binding .pyd + the onnx
# python frontend. Dropped: NPU compiler/plugin/runtime (the bulk), the
# GPU plugin, the TensorFlow/Paddle/PyTorch/JAX/TFLite frontend DLLs, the
# AUTO/BATCH/HETERO meta-plugins, import .lib files, C++ headers,
# type stubs and the 8.8 MB libs/cache.json metadata. Verified by rebuilding
# and running the frozen CPU worker (OpenVINO + IMDN) AND the GPU app
# (OpenVINO and Torch coexist in the same frozen process).
_OV_KEEP_NATIVE_DLLS = {
    "openvino.dll",
    "openvino_intel_cpu_plugin.dll",
    "openvino_onnx_frontend.dll",
    "tbb12.dll",
    "tbbbind_2_5.dll",
    "tbbmalloc.dll",
    "tbbmalloc_proxy.dll",
}
_OV_DROP_SUFFIXES = {".lib", ".h", ".hpp", ".pyi"}


def _ov_keep(src: str) -> bool:
    p = Path(src)
    name = p.name
    slashed = str(p).replace("\\", "/").lower()
    if "/openvino/" not in slashed:
        return True
    if p.suffix.lower() in _OV_DROP_SUFFIXES or name == "cache.json":
        return False
    if "/openvino/libs/" in slashed and name.lower().endswith(".dll"):
        return name in _OV_KEEP_NATIVE_DLLS
    if "/openvino/frontend/" in slashed and name.lower().endswith(".pyd"):
        # Keep only the ONNX python frontend binding; the other frameworks'
        # bindings are never imported (see above).
        return "onnx" in slashed
    return True


_ov_datas, _ov_binaries, _ov_hiddenimports = collect_all("openvino")
datas += [d for d in _ov_datas if _ov_keep(d[0])]
binaries += [b for b in _ov_binaries if _ov_keep(b[0])]
hiddenimports += _ov_hiddenimports

block_cipher = None

a = Analysis(
    [str(Path(SPECPATH) / "launcher.py")],
    pathex=[str(REPO_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        # This is a Windows-only desktop shell (WebView2/edgechromium) --
        # pywebview's other platform backends are never used here.
        "webview.platforms.cocoa",
        "webview.platforms.gtk",
        "webview.platforms.qt",
        "webview.platforms.android",
        "webview.platforms.cef",
        # PyInstaller's own official torch hook does `collect_submodules
        # ("torch")` unconditionally (every torch submodule, to avoid
        # missing dynamic imports) -- that pulls in matplotlib (only used by
        # torch.utils.tensorboard's writer and a couple of distributed/
        # functorch visualization utilities we never call) and pandas
        # (present only as a transitive dependency PyInstaller inferred from
        # some torch submodule; nothing in the real "final" pipeline's
        # import graph -- verified line by line, see packaging/README.md --
        # touches either). Excluding them here overrides that; if this ever
        # turns out to be wrong, the fix is to delete these two lines, not
        # to guess further.
        "matplotlib",
        "pandas",
        # Only referenced by torchvision's LSUN dataset loader
        # (torchvision/datasets/lsun.py), which nothing here calls -- we
        # never load training datasets, only run inference.
        "lmdb",
        # --- Phase B Group 1: numba / llvmlite (~116 MB) ---
        # Nothing in the production call graph imports either (grepped
        # backend/ + image_enhancer/src, and traced every importer in the
        # generated xref): the only modules that reach numba are
        # torch.testing._internal.common_cuda (test-only, pulled in by the
        # official torch hook's collect_submodules("torch")),
        # numpy.random._examples / numpy tests, onnxruntime.transformers.
        # benchmark, pandas (already excluded above) and facexlib (not
        # bundled at all). llvmlite exists solely as numba's JIT backend.
        # The frozen app never imports any of those paths.
        "numba",
        "llvmlite",
        # --- Phase B Group 2: scipy / scipy.libs (~82 MB) ---
        # No production module imports scipy, and every graph importer is
        # lazy/optional: timm.layers.pos_embed_rel only reaches scipy behind
        # `_USE_SCIPY = int(os.environ.get("TIMM_USE_SCIPY_INTERP", 0)) > 0`
        # (default OFF; the pure-torch RegularGridInterpolator fallback runs
        # instead, and SwinIR-M uses none of that code path anyway);
        # networkx/numpy.f2py/torch.testing._internal import it inside
        # functions that the inference path never calls. Verified by
        # rebuilding + running the frozen GPU (Restormer + SwinIR-M) and CPU
        # (OpenVINO IMDN) paths -- a scipy-less `import torch` and CUDA init
        # both succeeded.
        "scipy",
        # --- Phase B Group 3: onnx / onnxruntime / onnxscript (~37 MB) ---
        # Production CPU inference goes through OpenVINO (restore_exp/
        # imdn_x4_ov.py loads imdn_x4_256.onnx via openvino.runtime), never
        # ONNX Runtime. The only graph importers are timm.utils.onnx (timm's
        # export helper, collected by collect_all("timm")), torch.onnx.* /
        # torch.utils.tensorboard._onnx_graph (the ONNX exporter and
        # tensorboard, never called at inference), and the onnxscript
        # exporter stack -- all off the production path.
        "onnx",
        "onnxruntime",
        "onnxscript",
        # --- Phase B Group 7: huggingface_hub download ecosystem (~11 MB) ---
        # collect_all("timm") pulls in timm/models/_hub.py (imported eagerly
        # by timm/models/__init__.py at "from ._hub import (...)"), whose
        # entire purpose is downloading pretrained weights from the
        # HuggingFace Hub -- never used here, since every checkpoint this
        # product loads is a local .pth/.onnx file read via plain
        # torch.load()/openvino, never timm's create_model(pretrained=True)
        # or hf_hub_download(). Confirmed safe by reading _hub.py itself:
        # both its `import safetensors.torch` and its
        # `from huggingface_hub import ...` are wrapped in
        # try/except ImportError, setting _has_safetensors/_has_hf_hub =
        # False on failure -- these are optional dependencies by timm's own
        # design, not a hard requirement. network_swinir.py (the only
        # production timm import: `from timm.models.layers import
        # DropPath, to_2tuple, trunc_normal_`) never touches _hub.py's
        # public API either way. hf_xet (huggingface_hub's own optional Rust
        # accelerator, itself behind another try/except in
        # huggingface_hub.utils._xet) has no purpose whatsoever without
        # huggingface_hub. yaml/tqdm/requests/certifi/charset_normalizer/
        # idna/urllib3 ride along only as huggingface_hub/requests'
        # transitive dependencies -- individually verified each has no
        # OTHER eager importer in the frozen graph (torch's own only yaml
        # user, torch._library.fake_profile, does `import yaml` inside a
        # function body, and is itself never imported by anything -- pure
        # dead code for an inference-only build; openvino/webview/pythonnet
        # import none of these at all).
        "huggingface_hub",
        "hf_xet",
        "safetensors",
        "yaml",
        "tqdm",
        "requests",
        "certifi",
        "charset_normalizer",
        "idna",
        "urllib3",
        # --- Phase B Group 7 (cont.): protobuf (~1.4 MB) ---
        # google.protobuf's only real consumer in the torch ecosystem is
        # torch.utils.tensorboard (training-time event-log logging, never
        # called at inference -- and torch/__init__.py does not import it
        # eagerly either). Neither openvino's nor torch's own Python code
        # imports google.protobuf directly (grepped both site-packages
        # trees); OpenVINO's ONNX frontend (kept -- see Group 4 above) does
        # its protobuf parsing inside its C++ core
        # (openvino_onnx_frontend.dll), not via this Python package.
        "google",
        "protobuf",
    ],
    noarchive=False,
    optimize=0,
)

# Enforce the openvino native-DLL allowlist one final time on the resolved
# TOC: PyInstaller's post-analysis binary pass can re-add sibling plugin DLLs
# that live in the same openvino/libs folder even though nothing imports them
# (verified with pefile against every kept DLL -- the CPU plugin, ONNX
# frontend, core and python binding import only openvino.dll + system DLLs;
# none import the NPU/GPU/frontend plugins). This drops only openvino paths;
# every other binary is untouched.
#
# --- Phase B Group 5: OpenCV's bundled FFmpeg videoio backend (~29 MB) ---
# Production image I/O is cv2.imread/cv2.imwrite only (grepped backend/ +
# image_enhancer/src: no VideoCapture/VideoWriter anywhere), so OpenCV's
# videoio FFmpeg codec payload is never used -- cv2 loads it lazily from its
# VideoCapture/VideoWriter factory, not at import or for image I/O. Dropped
# by exact basename so no other package is affected.
_DROP_BUNDLED_NAMES = {
    "opencv_videoio_ffmpeg500_64.dll",
    # --- Phase B Group 6: Pillow's AVIF codec (~7.5 MB) ---
    # Production inputs are JPG/JPEG/PNG only (no avif reference anywhere in
    # backend/ or image_enhancer/src). Pillow discovers image codecs through
    # its plugin registry, which tolerates a missing optional _avif
    # extension (ImportError is caught -> AVIF simply unavailable); no other
    # format is affected.
    "_avif.cp310-win_amd64.pyd",
}


def _keep_final(src: str) -> bool:
    if Path(src).name in _DROP_BUNDLED_NAMES:
        return False
    return _ov_keep(src)


a.binaries = [e for e in a.binaries if _keep_final(str(e[0])) and _keep_final(str(e[1]))]
a.datas = [e for e in a.datas if _keep_final(str(e[0])) and _keep_final(str(e[1]))]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Adinn4KImageEnhancer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # verified working (see packaging/README.md's test log)
                     # with console=True during development; windowed for
                     # the real distributable, matching a normal desktop app.
    icon=str(REPO_ROOT / "packaging" / "assets" / "adinn.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="Adinn4KImageEnhancer",
)
