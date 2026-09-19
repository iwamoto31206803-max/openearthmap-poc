@echo off
setlocal EnableExtensions
chcp 65001 >nul

REM ============================================================
REM Meta CHM downloader - user entry point
REM
REM Run this BAT from QGIS / OSGeo4W Shell.
REM ============================================================

set "PROJECT=%~dp0"
if "%PROJECT:~-1%"=="\" set "PROJECT=%PROJECT:~0,-1%"

REM ------------------------------------------------------------
REM Check OSGeo4W / GDAL environment
REM ------------------------------------------------------------
where gdalinfo >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] GDAL is not available.
    echo Please open QGIS OSGeo4W Shell and run:
    echo.
    echo   cd /d C:\tokyo-lidar-treeheight
    echo   download_meta_chm.bat
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Meta / WRI Canopy Height Map downloader
echo ============================================================
echo.
echo WGS84 decimal degrees are expected.
echo Example:
echo   Longitude: 139.58
echo   Latitude : 35.63
echo.

set /p "LON=Center longitude [deg]: "
set /p "LAT=Center latitude  [deg]: "
set /p "WIDTH=Width  [m]: "
set /p "HEIGHT=Height [m]: "

set "OUTNAME=Meta_CHM"
set /p "OUTNAME=Output base name [Meta_CHM]: "
if "%OUTNAME%"=="" set "OUTNAME=Meta_CHM"

echo.

call "%PROJECT%\scripts\download_meta_chm.cmd" ^
    "%PROJECT%" ^
    "%LON%" ^
    "%LAT%" ^
    "%WIDTH%" ^
    "%HEIGHT%" ^
    "%OUTNAME%"

set "RC=%ERRORLEVEL%"

echo.

if "%RC%"=="0" (
    echo [DONE] Processing completed successfully.
) else (
    echo [ERROR] Processing failed. Exit code: %RC%
)

echo.
pause
exit /b %RC%
