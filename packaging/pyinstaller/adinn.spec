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
# of what this product ever executes -- only ever touches two checkpoints.
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
    if f.is_file() and f.name not in _MODELS_EXCLUDED:
        model_datas.append((str(f), str(Path("image_enhancer/models") / f.relative_to(_MODELS_DIR).parent)))

datas = [
    (str(REPO_ROOT / "image_enhancer" / "src"), "image_enhancer/src"),
    (str(REPO_ROOT / "image_enhancer" / "regions.json"), "image_enhancer"),
    (str(REPO_ROOT / "frontend" / "dist"), "frontend/dist"),
] + model_datas
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
for pkg in ("pythonnet", "cv2", "timm"):
    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(pkg)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hiddenimports

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
    ],
    noarchive=False,
    optimize=0,
)
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
