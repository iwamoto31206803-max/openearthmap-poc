@echo off
setlocal EnableExtensions
chcp 65001 >nul

REM ============================================================
REM Meta CHM downloader - user entry point
REM
REM Input:
REM   center longitude / latitude (WGS84)
REM   width / height in metres
REM
REM Requires:
REM   QGIS / OSGeo4W (GDAL/OGR)
REM   Windows PowerShell
REM   curl (included in current Windows 10/11)
REM ============================================================

set "PROJECT=%~dp0"
if "%PROJECT:~-1%"=="\" set "PROJECT=%PROJECT:~0,-1%"

REM ------------------------------------------------------------
REM Make GDAL/OGR available.
REM If already launched from OSGeo4W Shell, do nothing.
REM Otherwise try to find a standalone QGIS installation.
REM ------------------------------------------------------------
where gdalinfo >nul 2>&1
if errorlevel 1 (
    echo [INFO] GDAL is not on PATH. Searching for QGIS...

    set "QGISBAT="
    for /f "delims=" %%D in ('dir /b /ad /o-n "C:\Program Files\QGIS *" 2^>nul') do (
        if not defined QGISBAT (
            if exist "C:\Program Files\%%D\OSGeo4W.bat" (
                set "QGISBAT=C:\Program Files\%%D\OSGeo4W.bat"
            )
        )
    )

    if not defined QGISBAT (
        echo.
        echo [ERROR] QGIS / OSGeo4W could not be found.
        echo Please install QGIS or run this BAT from an OSGeo4W Shell.
        echo.
        pause
        exit /b 1
    )

    echo [INFO] Using: %QGISBAT%
    call "%QGISBAT%"
)

where gdalinfo >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] GDAL is still unavailable.
    echo Please run this BAT from an OSGeo4W Shell.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Meta / WRI Canopy Height Map downloader
echo ============================================================
echo.
echo WGS84 decimal degrees are expected.
echo Example: longitude 139.58   latitude 35.63
echo.

set /p "LON=Center longitude [deg]: "
set /p "LAT=Center latitude  [deg]: "
set /p "WIDTH=Width  [m]: "
set /p "HEIGHT=Height [m]: "

set "OUTNAME=Meta_CHM"
set /p "OUTNAME=Output base name [Meta_CHM]: "
if "%OUTNAME%"=="" set "OUTNAME=Meta_CHM"

echo.
call "%PROJECT%\scripts\download_meta_chm.cmd" "%PROJECT%" "%LON%" "%LAT%" "%WIDTH%" "%HEIGHT%" "%OUTNAME%"
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
