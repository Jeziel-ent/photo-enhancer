<#
.SYNOPSIS
  Builds the full Adinn 4K Image Enhancer Windows installer end to end:
  frontend -> PyInstaller onedir bundle -> Inno Setup installer .exe.

.DESCRIPTION
  Run from the repo root's PowerShell:
      .\packaging\build.ps1

  Requires (once, on the machine doing the build -- NOT on end-user
  machines, which need nothing installed):
    - Node.js/npm (to build the frontend)
    - The project's existing .venv, with PyInstaller installed into it
      (".venv\Scripts\python.exe -m pip install pyinstaller")
    - Inno Setup 6's ISCC.exe (installed via
      "winget install JRSoftware.InnoSetup" or from https://jrsoftware.org/isinfo.php)

  Each step fails fast (non-zero exit) rather than continuing on a broken
  intermediate artifact.
#>

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

# Official Microsoft VC++ 2015-2022 Redistributable (x64) staged next to the
# Inno script (packaging/vc_redist/vc_redist.x64.exe). Required by the frozen
# torch/CUDA stack on a machine with no runtime already present; the Inno
# script installs it silently, but ONLY when the registry shows no current
# enough build, and always from the official-verified staged binary. Obtain
# it from https://aka.ms/vs/17/release/vc_redist.x64.exe and verify it (see
# packaging/README.md) before building the installer. Fail fast here so a
# missing prerequisite can never silently produce an installer that drops it.
$VcRedistPath = Join-Path $PSScriptRoot "vc_redist\vc_redist.x64.exe"
if (-not (Test-Path $VcRedistPath)) {
    throw "Staged prerequisite missing: $VcRedistPath. Download the official vc_redist.x64.exe from https://aka.ms/vs/17/release/vc_redist.x64.exe and place it there before building the installer."
}
Write-Host "  [prereq] VC++ redistributable staged: $VcRedistPath" -ForegroundColor DarkGray

function Find-Iscc {
    $candidates = @(
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
    )
    foreach ($c in $candidates) { if (Test-Path $c) { return $c } }
    $onPath = Get-Command ISCC.exe -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }
    throw "ISCC.exe (Inno Setup 6) not found. Install it: winget install JRSoftware.InnoSetup"
}

Write-Host "== 1/3 Building frontend ==" -ForegroundColor Cyan
Push-Location "$RepoRoot\frontend"
npm install
npm run build
Pop-Location
if (-not (Test-Path "$RepoRoot\frontend\dist\index.html")) {
    throw "frontend build did not produce frontend/dist/index.html"
}

Write-Host "== 2/3 Packaging the Python app with PyInstaller ==" -ForegroundColor Cyan
Remove-Item -Recurse -Force "$RepoRoot\build","$RepoRoot\dist" -ErrorAction SilentlyContinue
& "$RepoRoot\.venv\Scripts\python.exe" -m PyInstaller `
    "$RepoRoot\packaging\pyinstaller\adinn.spec" `
    --noconfirm --distpath "$RepoRoot\dist" --workpath "$RepoRoot\build"
if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed" }
if (-not (Test-Path "$RepoRoot\dist\Adinn4KImageEnhancer\Adinn4KImageEnhancer.exe")) {
    throw "PyInstaller did not produce dist\Adinn4KImageEnhancer\Adinn4KImageEnhancer.exe"
}

Write-Host "== 3/3 Compiling the Inno Setup installer ==" -ForegroundColor Cyan
$iscc = Find-Iscc
& $iscc "$RepoRoot\packaging\inno\adinn_setup.iss"
if ($LASTEXITCODE -ne 0) { throw "Inno Setup compile failed" }

Write-Host "`nDone. Installer output:" -ForegroundColor Green
Get-ChildItem "$RepoRoot\dist\Adinn4KImageEnhancer-Setup-*.exe" | ForEach-Object {
    "{0}  ({1:N1} MB)" -f $_.FullName, ($_.Length / 1MB)
}
