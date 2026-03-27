param(
    [switch]$RemoveVenv
)

# Repair script for uv virtual environments on Windows
# - If a package's .dist-info/RECORD file is missing (e.g. numpy), uv may warn that uninstall failed
# - This script cleans corrupted dist-info directories or removes the whole .venv if -RemoveVenv is specified

$ErrorActionPreference = 'Stop'

function Remove-CorruptedDistInfo {
    param(
        [string]$SitePackagesPath
    )

    if (-not (Test-Path $SitePackagesPath)) {
        Write-Host "Site-packages not found at: $SitePackagesPath" -ForegroundColor Yellow
        return
    }

    $targets = Get-ChildItem -Path $SitePackagesPath -Directory -Filter "*.dist-info" -ErrorAction SilentlyContinue
    $removed = 0
    foreach ($dir in $targets) {
        $record = Join-Path $dir.FullName "RECORD"
        if (-not (Test-Path $record)) {
            Write-Host "Removing corrupted dist-info: $($dir.FullName) (missing RECORD)" -ForegroundColor Yellow
            Remove-Item -Recurse -Force -LiteralPath $dir.FullName
            $removed++
        }
    }

    if ($removed -eq 0) {
        Write-Host "No corrupted dist-info directories found." -ForegroundColor Green
    } else {
        Write-Host "Removed $removed corrupted dist-info directorie(s)." -ForegroundColor Green
    }
}

# Determine .venv path relative to repo root
$venvPath = Join-Path (Get-Location) ".venv"

if ($RemoveVenv) {
    if (Test-Path $venvPath) {
        Write-Host "Removing entire virtual environment at: $venvPath" -ForegroundColor Yellow
        Remove-Item -Recurse -Force -LiteralPath $venvPath
        Write-Host ".venv removed. You can recreate it with: uv venv && uv sync" -ForegroundColor Green
    } else {
        Write-Host ".venv not found at: $venvPath" -ForegroundColor Yellow
    }
    exit 0
}

# Try to locate site-packages inside .venv (Windows layout)
$sitePackages = Join-Path $venvPath "Lib\site-packages"
Remove-CorruptedDistInfo -SitePackagesPath $sitePackages

Write-Host "Now you can run: uv sync --reinstall" -ForegroundColor Green
