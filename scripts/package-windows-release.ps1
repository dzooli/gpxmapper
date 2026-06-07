#!/usr/bin/env pwsh
<#
.SYNOPSIS
  Stage the Windows release bundle and create a ZIP (same layout as CI).

.DESCRIPTION
  Copies dist\gpxmapper.exe, README, LICENSE, CHANGELOG, Nominatim helpers, and
  doc\USER_GUIDE.md into a staging folder, then Compress-Archive.

  Run **python build_exe.py** first so dist\gpxmapper.exe exists.

.PARAMETER ZipPath
  Output ZIP file path. Default: gpxmapper-release.zip in the repository root.

.PARAMETER StagingDirectory
  Staging folder under the repo root (recreated each run). Default: release
  (matches .github/workflows/build.yml).
#>
param(
    [string]$ZipPath = "",
    [string]$StagingDirectory = "release"
)

$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $RepoRoot

$exe = Join-Path $RepoRoot 'dist\gpxmapper.exe'
if (-not (Test-Path -LiteralPath $exe)) {
    Write-Error "dist\gpxmapper.exe not found. From the repo root run: python build_exe.py"
}

if ([string]::IsNullOrWhiteSpace($ZipPath)) {
    $ZipPath = Join-Path $RepoRoot 'gpxmapper-release.zip'
}
elseif (-not [System.IO.Path]::IsPathRooted($ZipPath)) {
    $ZipPath = Join-Path $RepoRoot $ZipPath
}

$stage = Join-Path $RepoRoot $StagingDirectory
if (Test-Path -LiteralPath $stage) {
    Remove-Item -LiteralPath $stage -Recurse -Force
}
New-Item -ItemType Directory -Path $stage -Force | Out-Null
$docDir = Join-Path $stage 'doc'
New-Item -ItemType Directory -Path $docDir -Force | Out-Null

Copy-Item -LiteralPath $exe -Destination $stage
Copy-Item -LiteralPath (Join-Path $RepoRoot 'README.md') -Destination $stage
Copy-Item -LiteralPath (Join-Path $RepoRoot 'LICENSE') -Destination $stage
Copy-Item -LiteralPath (Join-Path $RepoRoot 'CHANGELOG.md') -Destination $stage
Copy-Item -LiteralPath (Join-Path $RepoRoot 'nominatim\start_server.sh') -Destination $stage
Copy-Item -LiteralPath (Join-Path $RepoRoot 'nominatim\verify_local_hungary.sh') -Destination $stage
Copy-Item -LiteralPath (Join-Path $RepoRoot 'nominatim\start_server.bat') -Destination $stage
Copy-Item -LiteralPath (Join-Path $RepoRoot 'install\doc\USER_GUIDE.md') -Destination $docDir

if (Test-Path -LiteralPath $ZipPath) {
    Remove-Item -LiteralPath $ZipPath -Force
}
Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $ZipPath

Write-Host "Packaged: $ZipPath" -ForegroundColor Green
Write-Host "Staging left at: $stage (delete when done if you do not need it)" -ForegroundColor DarkGray
