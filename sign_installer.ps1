# ARIA Installer Code Signing Script
# Run once to create a self-signed cert, then signs the installer.
# Run as Administrator: Right-click PowerShell -> "Run as Administrator"
# Usage: powershell -ExecutionPolicy Bypass -File sign_installer.ps1

$ErrorActionPreference = "Stop"

$CertSubject  = "CN=ARIA AI, O=ARIA AI, L=Local"
$CertFile     = "$PSScriptRoot\aria_codesign.pfx"
$CertPassword = ConvertTo-SecureString -String "ARIA_sign_2026" -Force -AsPlainText
$InstallerPath = "$PSScriptRoot\dist\installer\ARIA_Setup_1.0.0.exe"

Write-Host ""
Write-Host "=== ARIA Installer Signing ===" -ForegroundColor Cyan
Write-Host ""

# ── Step 1: Create or reuse certificate ──────────────────────────────────────
$existing = Get-ChildItem Cert:\CurrentUser\My -CodeSigningCert |
            Where-Object { $_.Subject -like "*ARIA AI*" } |
            Select-Object -First 1

if ($existing) {
    Write-Host "[OK] Found existing ARIA code-signing cert: $($existing.Thumbprint)" -ForegroundColor Green
    $cert = $existing
} else {
    Write-Host "[..] Creating self-signed code-signing certificate..." -ForegroundColor Yellow
    $cert = New-SelfSignedCertificate `
        -Type CodeSigningCert `
        -Subject $CertSubject `
        -FriendlyName "ARIA AI Code Signing" `
        -CertStoreLocation "Cert:\CurrentUser\My" `
        -KeyExportPolicy Exportable `
        -KeySpec Signature `
        -HashAlgorithm SHA256 `
        -NotAfter (Get-Date).AddYears(5)
    Write-Host "[OK] Certificate created: $($cert.Thumbprint)" -ForegroundColor Green
}

# ── Step 2: Export PFX (for backup / CI use) ─────────────────────────────────
if (-not (Test-Path $CertFile)) {
    Export-PfxCertificate -Cert $cert -FilePath $CertFile -Password $CertPassword | Out-Null
    Write-Host "[OK] Certificate exported to aria_codesign.pfx" -ForegroundColor Green
}

# ── Step 3: Trust the certificate locally (suppresses SmartScreen) ───────────
Write-Host "[..] Trusting certificate in local machine stores..." -ForegroundColor Yellow

$stores = @("Root", "TrustedPublisher")
foreach ($store in $stores) {
    $storeObj = New-Object System.Security.Cryptography.X509Certificates.X509Store($store, "LocalMachine")
    $storeObj.Open("ReadWrite")
    $existing_in_store = $storeObj.Certificates | Where-Object { $_.Thumbprint -eq $cert.Thumbprint }
    if (-not $existing_in_store) {
        $storeObj.Add($cert)
        Write-Host "[OK] Added to LocalMachine\$store" -ForegroundColor Green
    } else {
        Write-Host "[OK] Already trusted in LocalMachine\$store" -ForegroundColor Green
    }
    $storeObj.Close()
}

# ── Step 4: Sign the installer ───────────────────────────────────────────────
if (-not (Test-Path $InstallerPath)) {
    Write-Host ""
    Write-Host "[FAIL] Installer not found: $InstallerPath" -ForegroundColor Red
    Write-Host "       Run installer.iss in Inno Setup first." -ForegroundColor Red
    exit 1
}

Write-Host "[..] Signing installer..." -ForegroundColor Yellow
$result = Set-AuthenticodeSignature `
    -FilePath $InstallerPath `
    -Certificate $cert `
    -TimestampServer "http://timestamp.sectigo.com" `
    -HashAlgorithm SHA256

if ($result.Status -eq "Valid") {
    Write-Host "[OK] Installer signed successfully!" -ForegroundColor Green
} else {
    # Timestamp server may be slow — retry without timestamp
    Write-Host "[..] Retrying without timestamp server..." -ForegroundColor Yellow
    $result = Set-AuthenticodeSignature `
        -FilePath $InstallerPath `
        -Certificate $cert `
        -HashAlgorithm SHA256
    if ($result.Status -eq "Valid") {
        Write-Host "[OK] Installer signed (no timestamp)." -ForegroundColor Green
    } else {
        Write-Host "[FAIL] Signing failed: $($result.StatusMessage)" -ForegroundColor Red
        exit 1
    }
}

# ── Step 5: Verify ───────────────────────────────────────────────────────────
$sig = Get-AuthenticodeSignature -FilePath $InstallerPath
Write-Host ""
Write-Host "=== Signature Info ===" -ForegroundColor Cyan
Write-Host "  File    : $InstallerPath"
Write-Host "  Status  : $($sig.Status)"
Write-Host "  Signer  : $($sig.SignerCertificate.Subject)"
Write-Host "  Expires : $($sig.SignerCertificate.NotAfter)"
Write-Host ""

if ($sig.Status -eq "Valid") {
    Write-Host "Done! The installer is signed. Users will see a normal UAC prompt" -ForegroundColor Cyan
    Write-Host "instead of the SmartScreen block." -ForegroundColor Cyan
} else {
    Write-Host "Warning: Signature status is '$($sig.Status)'" -ForegroundColor Yellow
}
