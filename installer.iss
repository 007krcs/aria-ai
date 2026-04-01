; ============================================================
; ARIA Windows Setup Wizard — Inno Setup 6 Script
; ============================================================
; Requirements: Inno Setup 6.3+ from https://jrsoftware.org/isinfo.php
;
; To build:
;   1. Open this file in Inno Setup Compiler (ISCC.exe)
;   2. Press Ctrl+F9 or click Build > Compile
;   3. Output: dist\installer\ARIA_Setup_1.0.0.exe
;
; Optional branding assets (create these for a fully branded installer):
;   assets\wizard_banner.bmp   — 164 x 314 px, 24-bit BMP (left panel on inner pages)
;   assets\wizard_header.bmp   —  55 x  58 px, 24-bit BMP (top-right logo)
;   assets\aria.ico            — Multi-resolution .ico (16/32/48/256 px)
; ============================================================

#define AppName        "ARIA"
#define AppVersion     "1.0.0"
#define AppPublisher   "ARIA AI"
#define AppDescription "Adaptive Reasoning Intelligence Architecture"
#define AppExeName     "ARIA.exe"
#define AppURL         "http://localhost:8000"
#define AppID          "{{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}"
#define ARIA_HAS_ASSETS

; ── [Setup] ─────────────────────────────────────────────────────────────────
[Setup]
AppId={#AppID}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}
AppUpdatesURL={#AppURL}
AppCopyright=Copyright (C) 2024-2026 ARIA AI

; Installation directory
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
AllowNoIcons=yes

; Require admin (needed for Program Files + registry)
PrivilegesRequired=admin
PrivilegesRequiredOverridesAllowed=commandline

; Output
OutputDir=dist\installer
OutputBaseFilename=ARIA_Setup_{#AppVersion}
SetupIconFile=assets\aria.ico

; Wizard appearance
WizardStyle=modern
WizardSizePercent=120,110
WizardResizable=no

; Branding images (comment out if files don't exist yet)
#ifdef ARIA_HAS_ASSETS
WizardImageFile=assets\wizard_banner.bmp
WizardSmallImageFile=assets\wizard_header.bmp
WizardImageStretch=no
WizardImageBackColor=$1A1A2E
#endif

; Compression (best ratio, good speed)
Compression=lzma2/ultra
SolidCompression=yes
InternalCompressLevel=ultra

; Minimum Windows 10 1903
MinVersion=10.0.18362

; Architecture
ArchitecturesInstallIn64BitMode=x64compatible

; Uninstaller
UninstallDisplayIcon={app}\{#AppExeName}
UninstallDisplayName={#AppName} {#AppVersion}
UninstallFilesDir={app}

; Logging (helps debugging)
SetupLogging=yes

; No restart needed
RestartIfNeededByRun=no
CloseApplications=yes
CloseApplicationsFilter=ARIA.exe,python.exe

; Version info embedded in setup exe
VersionInfoVersion={#AppVersion}.0
VersionInfoCompany={#AppPublisher}
VersionInfoDescription={#AppName} Setup
VersionInfoProductName={#AppName}
VersionInfoProductVersion={#AppVersion}

; ── [Messages] — custom branding text ───────────────────────────────────────
[Messages]
WelcomeLabel1=Welcome to the [name] Setup Wizard
WelcomeLabel2=This wizard will guide you through the installation of [name/ver].%n%nARIA is your local, privacy-first AI assistant — powered entirely on your device. No data leaves your computer.%n%nClick Next to continue, or Cancel to exit Setup.
FinishedLabel=Setup has successfully installed [name] on your computer.%n%nARIA will run in the system tray and open in your browser at http://localhost:8000%n%nClick Finish to exit this wizard.
FinishedHeadingLabel=Completing the [name] Setup Wizard
SelectDirLabel3=Setup will install [name] into the following folder.
SelectDirBrowseLabel=To continue, click Next. If you would like to select a different folder, click Browse.
SelectComponentsLabel2=Select the components you want to install; clear the components you do not want to install. Click Next when you are ready to continue.
ReadyLabel1=Setup is now ready to begin installing [name] on your computer.
ReadyLabel2a=Click Install to proceed with the installation.
InstallingLabel=Please wait while Setup installs [name] on your computer...
StatusExtractFiles=Extracting files...
StatusCreateIcons=Creating application shortcuts...
StatusCreateDirs=Creating directories...
UninstalledAll=[name] was successfully removed from your computer.

; ── [Types] — install presets ────────────────────────────────────────────────
[Types]
Name: "full";    Description: "Full Installation (recommended)"
Name: "compact"; Description: "Compact — Core + Web UI only"
Name: "custom";  Description: "Custom Installation";   Flags: iscustom

; ── [Components] — feature checkboxes ───────────────────────────────────────
[Components]
Name: "core";    Description: "ARIA Core Engine (required)";         Types: full compact custom;  Flags: fixed
Name: "webui";   Description: "Web Dashboard (browser interface)";   Types: full compact custom
Name: "voice";   Description: "Voice Assistant (wake word + TTS)";   Types: full custom
Name: "models";  Description: "Local Model Storage (models\ folder)"; Types: full custom

; ── [Tasks] ─────────────────────────────────────────────────────────────────
[Tasks]
Name: "desktopicon";   Description: "Create a &desktop shortcut";               GroupDescription: "Shortcuts:";        Flags: checkedonce
Name: "startmenuicon"; Description: "Create a &Start Menu entry";               GroupDescription: "Shortcuts:";        Flags: checkedonce
Name: "startauto";     Description: "Launch ARIA automatically at &Windows startup"; GroupDescription: "Startup options:";  Flags: unchecked

; ── [Files] ──────────────────────────────────────────────────────────────────
[Files]
; ── Launcher executable ──────────────────────────────────────────────────────
Source: "dist\ARIA.exe";              DestDir: "{app}";            Flags: ignoreversion;                            Components: core

; ── Core Python source ───────────────────────────────────────────────────────
Source: "server.py";                  DestDir: "{app}";            Flags: ignoreversion;                            Components: core
Source: "main.py";                    DestDir: "{app}";            Flags: ignoreversion;                            Components: core
Source: "run.py";                     DestDir: "{app}";            Flags: ignoreversion;                            Components: core
Source: "requirements.txt";           DestDir: "{app}";            Flags: ignoreversion;                            Components: core
Source: ".env";                       DestDir: "{app}";            Flags: ignoreversion skipifsourcedoesntexist;    Components: core

; ── Project modules ──────────────────────────────────────────────────────────
Source: "core\*";       DestDir: "{app}\core";       Flags: ignoreversion recursesubdirs createallsubdirs;          Components: core
Source: "agents\*";     DestDir: "{app}\agents";     Flags: ignoreversion recursesubdirs createallsubdirs;          Components: core
Source: "pipelines\*";  DestDir: "{app}\pipelines";  Flags: ignoreversion recursesubdirs createallsubdirs;          Components: core
Source: "tools\*";      DestDir: "{app}\tools";      Flags: ignoreversion recursesubdirs createallsubdirs;          Components: core
Source: "system\*";     DestDir: "{app}\system";     Flags: ignoreversion recursesubdirs createallsubdirs;          Components: core
Source: "research\*";   DestDir: "{app}\research";   Flags: ignoreversion recursesubdirs createallsubdirs;          Components: core

; ── Frontend React build ─────────────────────────────────────────────────────
Source: "app\dist\*";   DestDir: "{app}\app\dist";   Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist; Components: webui

; ── Model storage folder ─────────────────────────────────────────────────────
Source: "models\*";     DestDir: "{app}\models";     Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist; Components: models

; ── License ──────────────────────────────────────────────────────────────────
Source: "LICENSE.txt";                DestDir: "{app}";            Flags: ignoreversion;                            Components: core

; ── Pip bootstrap helper ─────────────────────────────────────────────────────
Source: "install_deps.bat";           DestDir: "{app}";            Flags: ignoreversion skipifsourcedoesntexist;    Components: core

; ── [Dirs] ───────────────────────────────────────────────────────────────────
[Dirs]
Name: "{app}\data"
Name: "{app}\logs"
Name: "{app}\models";  Components: models

; ── [Icons] ──────────────────────────────────────────────────────────────────
[Icons]
; Desktop
Name: "{autodesktop}\{#AppName}";           Filename: "{app}\{#AppExeName}";  WorkingDir: "{app}";  Comment: "{#AppDescription}";  IconFilename: "{app}\{#AppExeName}";  Tasks: desktopicon
; Start Menu
Name: "{group}\{#AppName}";                 Filename: "{app}\{#AppExeName}";  WorkingDir: "{app}";  Comment: "{#AppDescription}";  IconFilename: "{app}\{#AppExeName}";  Tasks: startmenuicon
Name: "{group}\ARIA Dashboard (Browser)";   Filename: "{app}\{#AppExeName}";  WorkingDir: "{app}";  Parameters: "--browser";        Tasks: startmenuicon
Name: "{group}\Uninstall {#AppName}";       Filename: "{uninstallexe}";       WorkingDir: "{app}";

; ── [Registry] ───────────────────────────────────────────────────────────────
[Registry]
; Auto-start (optional, unchecked)
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "{#AppName}"; ValueData: """{app}\{#AppExeName}"""; Flags: uninsdeletevalue; Tasks: startauto
; App registration (for Add/Remove Programs details)
Root: HKLM; Subkey: "Software\{#AppPublisher}\{#AppName}"; ValueType: string; ValueName: "InstallPath"; ValueData: "{app}"; Flags: uninsdeletekey
Root: HKLM; Subkey: "Software\{#AppPublisher}\{#AppName}"; ValueType: string; ValueName: "Version";     ValueData: "{#AppVersion}"

; ── [Run] — post-install steps ───────────────────────────────────────────────
[Run]
; Install Python dependencies
Filename: "{cmd}"; Parameters: "/c ""{app}\install_deps.bat"" > ""{app}\logs\pip_install.log"" 2>&1"; WorkingDir: "{app}"; Flags: runhidden waituntilterminated; StatusMsg: "Installing Python dependencies (this may take a minute)..."; Components: core; Check: PythonInstalled
; Launch ARIA (optional, shown as checkbox on Finish page)
Filename: "{app}\{#AppExeName}"; Description: "Launch {#AppName} now"; Flags: nowait postinstall skipifsilent; WorkingDir: "{app}"

; ── [UninstallRun] ───────────────────────────────────────────────────────────
[UninstallRun]
Filename: "taskkill"; Parameters: "/F /IM ARIA.exe"; Flags: runhidden; RunOnceId: "KillARIA"
Filename: "taskkill"; Parameters: "/F /IM python.exe /FI ""WINDOWTITLE eq ARIA*"""; Flags: runhidden; RunOnceId: "KillARIAPython"

; ── [Code] ───────────────────────────────────────────────────────────────────
[Code]

// ─── Python detection ────────────────────────────────────────────────────────
function PythonInstalled(): Boolean;
var
  ResultCode: Integer;
  Output: AnsiString;
begin
  // Try python first, then python3
  Result := Exec(ExpandConstant('{cmd}'), '/c python --version', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) and (ResultCode = 0);
  if not Result then
    Result := Exec(ExpandConstant('{cmd}'), '/c python3 --version', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) and (ResultCode = 0);
end;

// ─── install_deps.bat generator ──────────────────────────────────────────────
procedure CreateDepsBatchFile();
var
  BatchPath: String;
  Lines: TArrayOfString;
begin
  BatchPath := ExpandConstant('{app}\install_deps.bat');
  SetArrayLength(Lines, 12);
  Lines[0]  := '@echo off';
  Lines[1]  := 'echo ARIA Dependency Installer';
  Lines[2]  := 'echo ===========================';
  Lines[3]  := 'echo.';
  Lines[4]  := 'cd /d "%~dp0"';
  Lines[5]  := 'echo Upgrading pip...';
  Lines[6]  := 'python -m pip install --upgrade pip --quiet';
  Lines[7]  := 'echo Installing ARIA dependencies...';
  Lines[8]  := 'python -m pip install -r requirements.txt --quiet';
  Lines[9]  := 'echo.';
  Lines[10] := 'echo Done! All dependencies installed.';
  Lines[11] := 'exit /b 0';
  SaveStringsToFile(BatchPath, Lines, False);
end;

// ─── Pre-setup check ─────────────────────────────────────────────────────────
function InitializeSetup(): Boolean;
var
  ErrCode: Integer;
begin
  Result := True;

  if not PythonInstalled() then begin
    if MsgBox(
      'Python 3.11 or later is required but was not found on this system.' + #13#10 + #13#10 +
      'Please install Python before continuing:' + #13#10 +
      '  1. Click Yes to open the Python download page' + #13#10 +
      '  2. Download and run Python 3.11 installer' + #13#10 +
      '  3. IMPORTANT: Check "Add Python to PATH"' + #13#10 +
      '  4. Re-run this ARIA installer' + #13#10 + #13#10 +
      'Would you like to open the Python download page now?',
      mbConfirmation, MB_YESNO
    ) = IDYES then begin
      ShellExec('open', 'https://www.python.org/downloads/release/python-31110/', '', '', SW_SHOWNORMAL, ewNoWait, ErrCode);
    end;
    Result := False;
  end;
end;

// ─── Post-extract: generate the batch file ───────────────────────────────────
procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then begin
    CreateDepsBatchFile();
  end;
end;

// ─── Custom wizard page labels ───────────────────────────────────────────────
procedure InitializeWizard();
begin
  // Welcome page
  WizardForm.WelcomeLabel1.Font.Size  := 14;
  WizardForm.WelcomeLabel1.Font.Style := [fsBold];

  // Make the window caption show ARIA branding
  WizardForm.Caption := 'ARIA Setup - Adaptive Reasoning Intelligence Architecture';
end;

// ─── Validate destination doesn't contain spaces (Python path safety) ─────────
function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  if CurPageID = wpSelectDir then begin
    if Pos(' ', ExpandConstant('{app}')) > 0 then begin
      if MsgBox(
        'The installation path contains spaces:' + #13#10 +
        ExpandConstant('{app}') + #13#10 + #13#10 +
        'This can sometimes cause issues with Python. ' +
        'We recommend a path without spaces (e.g. C:\ARIA).' + #13#10 + #13#10 +
        'Continue anyway?',
        mbConfirmation, MB_YESNO
      ) = IDNO then
        Result := False;
    end;
  end;
end;

// ─── Uninstall: confirm before proceeding ────────────────────────────────────
function InitializeUninstall(): Boolean;
begin
  Result := MsgBox(
    'This will remove ARIA from your computer.' + #13#10 + #13#10 +
    'Your personal data in the "data\" folder will be kept.' + #13#10 +
    'To remove everything, delete the installation folder manually after uninstalling.' + #13#10 + #13#10 +
    'Continue with uninstall?',
    mbConfirmation, MB_YESNO
  ) = IDYES;
end;
