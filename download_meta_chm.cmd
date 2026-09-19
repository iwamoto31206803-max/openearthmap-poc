@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul

REM ============================================================
REM Meta CHM downloader - processing script
REM
REM Arguments:
REM   %1 Project root
REM   %2 Center longitude (WGS84)
REM   %3 Center latitude  (WGS84)
REM   %4 Width  [m]
REM   %5 Height [m]
REM   %6 Output base name
REM
REM Design:
REM   - No separately configured Python environment is required.
REM   - Tile search: OGR against tiles.geojson
REM   - Download: curl from public AWS S3
REM   - Mosaic: GDAL VRT (no raster merge into RAM)
REM   - Crop: gdal_translate, preserving Meta's native grid/CRS
REM ============================================================

set "PROJECT=%~1"
set "LON=%~2"
set "LAT=%~3"
set "WIDTH=%~4"
set "HEIGHT=%~5"
set "OUTNAME=%~6"

if "%PROJECT%"=="" exit /b 2
if "%LON%"=="" exit /b 2
if "%LAT%"=="" exit /b 2
if "%WIDTH%"=="" exit /b 2
if "%HEIGHT%"=="" exit /b 2
if "%OUTNAME%"=="" set "OUTNAME=Meta_CHM"

set "WORK=%PROJECT%\02_work\meta_chm"
set "CACHE=%WORK%\cache"
set "TMP=%WORK%\temp"
set "OUTPUT_DIR=%PROJECT%\03_output\meta_chm"

set "INDEX=%CACHE%\tiles.geojson"
set "SELECTED=%TMP%\selected_tiles.csv"
set "TILELIST=%TMP%\tile_list.txt"
set "VRT=%WORK%\meta_chm_selected.vrt"
set "OUTPUT=%OUTPUT_DIR%\%OUTNAME%_native.tif"

set "S3_BASE=https://dataforgood-fb-data.s3.amazonaws.com/forests/v2/global/dinov3_global_chm_v2_ml3"
set "INDEX_URL=%S3_BASE%/tiles.geojson"
set "CHM_URL=%S3_BASE%/chm"

if not exist "%WORK%" mkdir "%WORK%"
if not exist "%CACHE%" mkdir "%CACHE%"
if not exist "%TMP%" mkdir "%TMP%"
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM ------------------------------------------------------------
REM Check required commands
REM ------------------------------------------------------------
for %%C in (ogr2ogr gdalbuildvrt gdal_translate gdalinfo curl powershell) do (
    where %%C >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Required command not found: %%C
        exit /b 3
    )
)

echo.
echo [1/6] Computing WGS84 bounding box...

REM ------------------------------------------------------------
REM Convert center + ground width/height into an approximate
REM WGS84 bbox. For ordinary municipal / PoC extents this is
REM sufficiently accurate for tile selection and cropping.
REM ------------------------------------------------------------
for /f "tokens=1-4" %%A in ('powershell -NoProfile -Command "$ci=[Globalization.CultureInfo]::InvariantCulture; $lon=[double]::Parse('%LON%',$ci); $lat=[double]::Parse('%LAT%',$ci); $w=[double]::Parse('%WIDTH%',$ci); $h=[double]::Parse('%HEIGHT%',$ci); if([math]::Abs($lat)-ge 80){exit 9}; $rad=$lat*[math]::PI/180.0; $dlat=($h/2.0)/111320.0; $dlon=($w/2.0)/(111320.0*[math]::Cos($rad)); $west=$lon-$dlon; $east=$lon+$dlon; $south=$lat-$dlat; $north=$lat+$dlat; [Console]::WriteLine(('{0:R} {1:R} {2:R} {3:R}' -f $west,$south,$east,$north))"') do (
    set "WEST=%%A"
    set "SOUTH=%%B"
    set "EAST=%%C"
    set "NORTH=%%D"
)

if not defined WEST (
    echo [ERROR] Failed to calculate bounding box.
    echo Check longitude/latitude/width/height.
    exit /b 4
)

echo   West : !WEST!
echo   South: !SOUTH!
echo   East : !EAST!
echo   North: !NORTH!

REM ------------------------------------------------------------
REM Download tile index once and keep it in cache
REM ------------------------------------------------------------
echo.
echo [2/6] Preparing CHMv2 tile index...

if not exist "%INDEX%" (
    echo   Downloading tiles.geojson...
    curl -L --fail --retry 3 --retry-delay 2 -o "%INDEX%.part" "%INDEX_URL%"
    if errorlevel 1 (
        if exist "%INDEX%.part" del "%INDEX%.part"
        echo [ERROR] Failed to download tiles.geojson.
        exit /b 5
    )
    move /y "%INDEX%.part" "%INDEX%" >nul
) else (
    echo   Using cached tiles.geojson.
)

REM ------------------------------------------------------------
REM Spatially select intersecting tiles.
REM GeoJSON is treated as WGS84; -spat_srs makes the input clear.
REM ------------------------------------------------------------
echo.
echo [3/6] Selecting intersecting Meta CHM tiles...

if exist "%SELECTED%" del "%SELECTED%"

ogr2ogr -f CSV "%SELECTED%" "%INDEX%" ^
    -spat !WEST! !SOUTH! !EAST! !NORTH! ^
    -spat_srs EPSG:4326 ^
    -select tile

if errorlevel 1 (
    echo [ERROR] Tile selection failed.
    exit /b 6
)

if exist "%TILELIST%" del "%TILELIST%"

set /a TILECOUNT=0

REM ------------------------------------------------------------
REM Download selected COGs; keep originals untouched in cache.
REM Validate cached files before reuse.
REM ------------------------------------------------------------
echo.
echo [4/6] Downloading / validating selected CHMv2 tiles...

for /f "usebackq skip=1 tokens=1 delims=," %%T in ("%SELECTED%") do (
    set "TILE=%%~T"
    set "TILE=!TILE:"=!"
    if not "!TILE!"=="" (
        set /a TILECOUNT+=1
        set "LOCAL=%CACHE%\!TILE!.tif"

        if exist "!LOCAL!" (
            gdalinfo "!LOCAL!" >nul 2>&1
            if errorlevel 1 (
                echo   Cached !TILE!.tif is invalid; downloading again...
                del "!LOCAL!"
            ) else (
                echo   Using cached !TILE!.tif
            )
        )

        if not exist "!LOCAL!" (
            echo   Downloading !TILE!.tif
            curl -L --fail --retry 3 --retry-delay 2 ^
                -o "!LOCAL!.part" "%CHM_URL%/!TILE!.tif"
            if errorlevel 1 (
                if exist "!LOCAL!.part" del "!LOCAL!.part"
                echo [ERROR] Download failed for tile !TILE!
                exit /b 7
            )
            move /y "!LOCAL!.part" "!LOCAL!" >nul
        )

        >>"%TILELIST%" echo !LOCAL!
    )
)

if !TILECOUNT! LEQ 0 (
    echo [ERROR] No CHMv2 tiles intersect the requested area.
    echo Check the input coordinate and extent.
    exit /b 8
)

echo   Selected tiles: !TILECOUNT!

REM ------------------------------------------------------------
REM Create a VRT. This does not resample or merge the rasters
REM into one huge in-memory array.
REM ------------------------------------------------------------
echo.
echo [5/6] Building VRT...

gdalbuildvrt -overwrite ^
    -srcnodata 255 ^
    -vrtnodata 255 ^
    -input_file_list "%TILELIST%" ^
    "%VRT%"

if errorlevel 1 (
    echo [ERROR] VRT creation failed.
    exit /b 9
)

REM ------------------------------------------------------------
REM Crop in WGS84 coordinates while leaving the output in the
REM source CHM CRS (EPSG:3857) and preserving its native pixel grid.
REM Nearest-neighbour is implicit because no resampling/reprojection
REM is performed by gdal_translate.
REM ------------------------------------------------------------
echo.
echo [6/6] Cropping native-grid CHM...

if exist "%OUTPUT%" del "%OUTPUT%"

gdal_translate ^
    -projwin !WEST! !NORTH! !EAST! !SOUTH! ^
    -projwin_srs EPSG:4326 ^
    -a_nodata 255 ^
    -co COMPRESS=DEFLATE ^
    -co TILED=YES ^
    "%VRT%" ^
    "%OUTPUT%"

if errorlevel 1 (
    echo [ERROR] Final crop failed.
    exit /b 10
)

echo.
echo ============================================================
echo SUCCESS
echo ============================================================
echo Output:
echo   %OUTPUT%
echo.
echo Selected Meta tiles:
for /f "usebackq skip=1 tokens=1 delims=," %%T in ("%SELECTED%") do echo   %%~T
echo.
echo Output summary:
gdalinfo "%OUTPUT%" | findstr /C:"Size is" /C:"Pixel Size" /C:"NoData Value" /C:"COMPRESSION"
echo.
echo NOTE:
echo   The output keeps Meta CHM's native EPSG:3857 grid and pixel size.
echo   The requested ground width/height is converted to an approximate
echo   WGS84 rectangle around the input center coordinate.
echo ============================================================

exit /b 0
