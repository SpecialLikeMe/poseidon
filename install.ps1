$InstallDir = [System.IO.Path]::GetFullPath($PSScriptRoot)

# ── Create virtualenv if missing ────────────────────────────────────────────
$VenvPython = [System.IO.Path]::Combine($InstallDir, "venv\Scripts\python.exe")
if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    python -m venv "$InstallDir\venv"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: 'python' not found. Install Python 3.10+ and try again." -ForegroundColor Red
        exit 1
    }
}

# ── Install / upgrade dependencies ──────────────────────────────────────────
Write-Host "Installing dependencies..." -ForegroundColor Cyan
& "$InstallDir\venv\Scripts\pip.exe" install -r "$InstallDir\requirements.txt" --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: pip install failed." -ForegroundColor Red
    exit 1
}
Write-Host "Dependencies installed." -ForegroundColor Green

# ── Write pos.bat (uses venv python) ────────────────────────────────────────
$BatchFile = [System.IO.Path]::Combine($InstallDir, "pos.bat")
$BatchContent = "@echo off`r`ncall `"%~dp0venv\Scripts\activate.bat`"`r`npython `"%~dp0main.py`" %*`r`n"
[System.IO.File]::WriteAllText($BatchFile, $BatchContent)
Write-Host "Created 'pos.bat' launcher." -ForegroundColor Green

# ── Add install dir to user PATH ─────────────────────────────────────────────
$OldPath = [Environment]::GetEnvironmentVariable("Path", "User")
$Paths = if ($OldPath) { $OldPath -split ';' | Where-Object { $_.Trim() -ne "" } } else { @() }
$Normalized = $Paths | ForEach-Object { try { [System.IO.Path]::GetFullPath($_) } catch { $_ } }

if ($Normalized -notcontains $InstallDir) {
    $NewPath = ($Paths + $InstallDir) -join ';'
    [Environment]::SetEnvironmentVariable("Path", $NewPath, "User")
    Write-Host "Added to user PATH. Restart your terminal to use 'pos'." -ForegroundColor Green
} else {
    Write-Host "'pos' already on PATH." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Done! Run: pos" -ForegroundColor Cyan
