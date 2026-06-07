@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem GPXMapper — local Nominatim via Docker (Windows counterpart to nominatim/start_server.sh).
rem Run from the folder that contains this file (same folder as gpxmapper.exe in the release ZIP).
rem Override extract:  set PBF_URL=https://download.geofabrik.de/europe/...osm.pbf

if not defined PBF_URL set "PBF_URL=https://download.geofabrik.de/europe/hungary-latest.osm.pbf"
set "IMAGE=mediagis/nominatim:5.1"

echo === GPXMapper local Nominatim (Docker) ===
echo PBF_URL=%PBF_URL%
echo Image: %IMAGE%
echo.

where docker >nul 2>&1
if errorlevel 1 (
  echo ERROR: Docker not found. Install Docker Desktop, then retry.
  exit /b 1
)

docker info >nul 2>&1
if errorlevel 1 (
  echo ERROR: Docker is not running or not reachable. Start Docker Desktop.
  exit /b 1
)
echo OK  Docker is available.

docker image inspect %IMAGE% >nul 2>&1
if errorlevel 1 (
  echo Pulling Docker image %IMAGE% ...
  docker pull %IMAGE%
  if errorlevel 1 (
    echo ERROR: docker pull failed.
    exit /b 1
  )
) else (
  echo OK  Docker image %IMAGE% is present locally.
)

echo Checking PBF_URL with curl ...
curl -fsI "%PBF_URL%" >nul 2>&1
if errorlevel 1 (
  echo ERROR: Cannot reach PBF_URL. Check the URL and your network.
  exit /b 1
)
echo OK  PBF_URL is reachable.

echo.
echo === Starting container ===
echo First startup imports the PBF into the container — this can take a long time.
echo When import finishes, GPXMapper defaults to http://localhost:8080 for --geolocate.
echo Ensure port 8080 is free; if Docker reports the port is already allocated, stop the other service.
echo Optional checks:  gpxmapper.exe check-nominatim    or Git Bash:  bash verify_local_hungary.sh
echo.

docker run -it -e PBF_URL="%PBF_URL%" -p 8080:8080 %IMAGE%
if errorlevel 1 (
  echo ERROR: docker run failed.
  exit /b 1
)
exit /b 0
