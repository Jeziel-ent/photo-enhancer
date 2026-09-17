; Adinn 4K Image Enhancer -- Windows installer (Inno Setup 6)
;
; Compile (after the PyInstaller onedir build exists at dist\Adinn4KImageEnhancer):
;   "C:\Users\<you>\AppData\Local\Programs\Inno Setup 6\ISCC.exe" packaging\inno\adinn_setup.iss
;
; Produces packaging\output\Adinn4KImageEnhancer-Setup-<version>.exe
;
; This script only packages the already-built PyInstaller output plus a few
; branding assets -- it does not build the frontend or the Python app itself
; (see packaging/build.ps1 for the full pipeline).

#define AppName "Adinn 4K Image Enhancer"
#define AppVersion "0.1.0"
#define AppPublisher "Adinn"
#define AppExeName "Adinn4KImageEnhancer.exe"
#define AppMutexName "Adinn4KImageEnhancerRunning"
#define DistDir "..\..\dist\Adinn4KImageEnhancer"
#define AssetsDir "..\assets"

[Setup]
; Fixed GUID -- do not change between releases, or Windows/Inno will treat
; every future version as an unrelated app instead of an in-place upgrade.
AppId={{6C2C6E1E-6B2E-4B7B-9B0B-ADD1774B4B01}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
VersionInfoVersion={#AppVersion}
VersionInfoCompany={#AppPublisher}
VersionInfoDescription={#AppName} Setup
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
OutputDir=..\output
OutputBaseFilename=Adinn4KImageEnhancer-Setup-{#AppVersion}
SetupIconFile={#AssetsDir}\adinn.ico
UninstallDisplayIcon={app}\{#AppExeName}
UninstallDisplayName={#AppName}
Compression=lzma2/ultra64
SolidCompression=yes
; Multi-GB payload (models + torch/CUDA runtime) -- large-address/64-bit-aware.
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; Program Files needs an admin token; Inno shows its own native UAC/"access
; denied" messaging if elevation is refused -- nothing custom needed here.
PrivilegesRequired=admin
MinVersion=10.0
; Refuses to install/uninstall while the app holds this mutex (see
; packaging/pyinstaller/launcher.py) -- shows Inno's own native
; "still running, please close it" message instead of corrupting a live copy.
AppMutex={#AppMutexName}
WizardStyle=modern
WizardImageFile={#AssetsDir}\wizard_image.bmp
WizardSmallImageFile={#AssetsDir}\wizard_small.bmp
WizardImageStretch=no
DisableWelcomePage=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[CustomMessages]
; "%2022" here was a mistaken attempt to encode the bullet character (U+2022,
; used correctly in the app's own splash screen -- see
; frontend/src/components/splash/SplashScreen.tsx) into this message. This
; .iss file is plain ASCII with no encoding/BOM declared, and Inno Setup's
; CustomMessages substitution only understands %1/%2/... numbered
; parameters -- "%2022" is neither valid Unicode escape syntax nor a
; parameter reference, so it was rendered completely literally. Rather than
; convert this file's encoding (fragile across locales/codepages -- exactly
; the class of bug this was) or add another placeholder mechanism, this uses
; a plain ASCII separator, matching this same file's existing "[OK]"/"[!]"
; markers on the System Requirements page for the same reliability reason.
english.TaglineText=Enhance - Denoise - Upscale
english.ComponentAppDesc=The application itself (UI, local processing engine, runtime files).
english.ComponentModelsDesc=Local AI models used for 4K enhancement (denoise, super-resolution, detail recovery). Required -- the app cannot enhance images without these.
english.TaskDesktopDesc=Create a &desktop shortcut
english.TaskStartMenuDesc=Create a &Start Menu shortcut
english.FinishedReady=Adinn 4K Image Enhancer is ready.
english.SysReqTitle=System Requirements
english.SysReqSubtitle=Here's what this computer offers for local AI processing.

[Types]
Name: "full"; Description: "Full installation"

[Components]
Name: "app"; Description: "Adinn 4K Image Enhancer (application + runtime files)"; Types: full; Flags: fixed
Name: "models"; Description: "AI enhancement models"; Types: full; Flags: fixed

[Tasks]
Name: "desktopicon"; Description: "{cm:TaskDesktopDesc}"; GroupDescription: "Additional shortcuts:"; Flags: checkedonce
Name: "startmenuicon"; Description: "{cm:TaskStartMenuDesc}"; GroupDescription: "Additional shortcuts:"; Flags: checkedonce

[Files]
; The app itself: the PyInstaller onedir bundle's launcher exe + all its
; runtime dependencies (_internal\*), EXCEPT the AI model weights, which are
; their own component below so their real copy progress reads as "AI models"
; to the user rather than being lost inside a generic "app" step.
Source: "{#DistDir}\{#AppExeName}"; DestDir: "{app}"; Components: app; Flags: ignoreversion
Source: "{#DistDir}\_internal\*"; DestDir: "{app}\_internal"; Components: app; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "image_enhancer\models\*"

Source: "{#DistDir}\_internal\image_enhancer\models\*"; DestDir: "{app}\_internal\image_enhancer\models"; Components: models; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: startmenuicon
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"; Tasks: startmenuicon
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Best-effort: remove anything the app itself may have written under its own
; install directory (logs, __pycache__ from a from-source debug run, etc.)
; that Inno's own file-tracking wouldn't otherwise know to remove.
Type: filesandordirs; Name: "{app}\_internal\__pycache__"

[Code]
var
  SysReqPage: TWizardPage;
  SysReqMemo: TNewMemo;

function GetWindowsVersionLine(): String;
var
  Version: TWindowsVersion;
begin
  GetWindowsVersionEx(Version);
  Result := Format('Windows %d.%d (build %d)', [Version.Major, Version.Minor, Version.Build]);
end;

function GetEnvOrDefault(const Name, Default: String): String;
begin
  Result := GetEnv(Name);
  if Result = '' then Result := Default;
end;

{ Inno Setup's Pascal Script Format() does NOT support the %f (float)
  specifier -- unlike Delphi's Format, it only implements %d/%u/%x/%e/%g/%s
  (confirmed against jrsoftware.org's own Format()/support-function
  reference, which lists no %f example and no FormatFloat function at
  all). Passing %.1f raised a runtime "Format '...' invalid or
  incompatible with argument" error on the System Requirements page for
  every value (RAM, free disk space) that used it. This formats a value to
  exactly one decimal place using only integer arithmetic + IntToStr, then
  callers build their message with Format's %s, which IS supported. }
function FormatOneDecimal(const Value: Extended): String;
var
  TenthsTotal, WholePart, TenthsPart: Int64;
begin
  TenthsTotal := Round(Value * 10);
  WholePart := TenthsTotal div 10;
  TenthsPart := TenthsTotal mod 10;
  if TenthsPart < 0 then TenthsPart := -TenthsPart; { Value is never negative here, but be exact }
  Result := IntToStr(WholePart) + '.' + IntToStr(TenthsPart);
end;

{ Runs a PowerShell one-liner and returns its trimmed stdout, or '' on any
  failure -- used only for optional, best-effort GPU detection; never blocks
  or fails the install if it can't determine an answer. }
function RunPowerShellCapture(const Script: String): String;
var
  ResultCode: Integer;
  TmpFile: String;
  Lines: TArrayOfString;
  Output: AnsiString;
begin
  Result := '';
  TmpFile := ExpandConstant('{tmp}\adinn_gpu_check.txt');
  if Exec(ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe'),
     '-NoProfile -NonInteractive -Command "' + Script + ' | Out-File -Encoding utf8 ''' + TmpFile + '''"',
     '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
  begin
    if (ResultCode = 0) and LoadStringsFromFile(TmpFile, Lines) and (GetArrayLength(Lines) > 0) then
      Result := Trim(Lines[0]);
  end;
end;

function DetectGpuLine(): String;
var
  GpuName: String;
begin
  GpuName := RunPowerShellCapture(
    '(Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match ''NVIDIA'' } | Select-Object -First 1 -ExpandProperty Name)');
  if GpuName <> '' then
    Result := '[OK] NVIDIA GPU detected: ' + GpuName + ' (recommended -- fast local processing)'
  else
    Result := '[!] No NVIDIA GPU detected -- the app will fall back to CPU processing (slower, but fully supported)';
end;

function DetectRamLine(): String;
var
  TotalMB: Integer;
begin
  { physical RAM via Inno's built-in function, in MB }
  TotalMB := 0;
  try
    TotalMB := StrToIntDef(RunPowerShellCapture(
      '[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1MB)'), 0);
  except
  end;
  if TotalMB >= 8192 then
    Result := Format('[OK] RAM: %s GB (recommended)', [FormatOneDecimal(TotalMB / 1024)])
  else if TotalMB > 0 then
    Result := Format('[!] RAM: %s GB (8 GB+ recommended; the app may run slowly)', [FormatOneDecimal(TotalMB / 1024)])
  else
    Result := '[!] RAM: could not be determined';
end;

function DetectCpuLine(): String;
var
  CpuName: String;
begin
  CpuName := RunPowerShellCapture('(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)');
  if CpuName = '' then CpuName := 'Unknown CPU';
  Result := '[OK] CPU: ' + CpuName;
end;

function DetectDiskLine(): String;
var
  FreeBytes, TotalBytes: Int64;
  Drive: String;
begin
  { The destination-folder wizard page separately, natively blocks Next if
    the selected drive doesn't have enough free space -- this line is just
    informational, shown before the user has even picked a drive. }
  Drive := ExtractFileDrive(ExpandConstant('{autopf}')) + '\';
  if GetSpaceOnDisk64(Drive, FreeBytes, TotalBytes) then
    Result := Format('[OK] Free space on %s %s GB available (~6 GB required)', [Drive, FormatOneDecimal(FreeBytes / 1024 / 1024 / 1024)])
  else
    Result := 'Disk space required: ~6 GB free on the installation drive';
end;

procedure InitializeWizard();
begin
  SysReqPage := CreateCustomPage(wpWelcome, CustomMessage('SysReqTitle'), CustomMessage('SysReqSubtitle'));

  SysReqMemo := TNewMemo.Create(SysReqPage);
  SysReqMemo.Parent := SysReqPage.Surface;
  SysReqMemo.Left := 0;
  SysReqMemo.Top := 0;
  SysReqMemo.Width := SysReqPage.SurfaceWidth;
  SysReqMemo.Height := SysReqPage.SurfaceHeight;
  SysReqMemo.ReadOnly := True;
  SysReqMemo.ScrollBars := ssVertical;
  SysReqMemo.Font.Name := 'Segoe UI';
  SysReqMemo.Font.Size := 9;

  { Tagline on the welcome page, under the default title/subtitle. }
  if WizardForm.WelcomeLabel2 <> nil then
    WizardForm.WelcomeLabel2.Caption :=
      CustomMessage('TaglineText') + #13#10#13#10 + WizardForm.WelcomeLabel2.Caption;
end;

procedure CurPageChanged(CurPageID: Integer);
var
  Lines: TStringList;
begin
  if CurPageID = SysReqPage.ID then
  begin
    Lines := TStringList.Create;
    try
      Lines.Add('Checking this computer against what Adinn 4K Image Enhancer needs...');
      Lines.Add('');
      Lines.Add('[OK] ' + GetWindowsVersionLine());
      Lines.Add(DetectCpuLine());
      Lines.Add(DetectRamLine());
      Lines.Add(DetectDiskLine());
      Lines.Add(DetectGpuLine());
      Lines.Add('');
      Lines.Add('An NVIDIA GPU is not required to install or run this app -- it always');
      Lines.Add('works fully offline on the CPU too, just slower per image. Nothing above');
      Lines.Add('blocks installation.');
      SysReqMemo.Lines := Lines;
    finally
      Lines.Free;
    end;
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  { Inno's own installation page already shows real, per-file progress (the
    actual file being copied -- including each AI model .pth as it's
    written) and a real overall percentage; this only adds friendlier
    section headers at the natural phase boundaries. }
  if CurStep = ssInstall then
    WizardForm.StatusLabel.Caption := 'Preparing...';
end;

function InitializeUninstall(): Boolean;
begin
  Result := True;
end;
