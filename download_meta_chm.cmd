@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul

REM ============================================================
REM Meta CHM downloader - processing script
REM
REM Japan-wide version
REM
REM Arguments:
REM   %1 Project root
REM   %2 Center longitude (WGS84)
REM   %3 Center latitude  (WGS84)
REM   %4 Width  [m]
REM   %5 Height [m]
REM   %6 Output base name
REM
REM Spatial design:
REM
REM   WGS84 center coordinate
REM          |
REM          v
REM   local AEQD projection
REM   centered on requested point
REM          |
REM          v
REM   exact metric rectangle
REM   width x height [m]
REM          |
REM          v
REM   transform rectangle to WGS84
REM          |
REM          +--> select Meta tiles
REM          |
REM          +--> crop Meta native raster
REM
REM Meta output remains:
REM   CRS        : EPSG:3857
REM   Pixel size : Meta native (~1.194 m)
REM   Data type  : Byte
REM   NoData     : 255
REM
REM Requirements:
REM   QGIS / OSGeo4W
REM   GDAL / OGR
REM   curl
REM
REM Python and PowerShell are NOT required.
REM ============================================================


REM ============================================================
REM Arguments
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


REM ============================================================
REM Directories
REM ============================================================

set "WORK=%PROJECT%\02_work\meta_chm"
set "CACHE=%WORK%\cache"
set "TMP=%WORK%\temp"

set "OUTPUT_DIR=%PROJECT%\03_output\meta_chm"

set "INDEX=%CACHE%\tiles.geojson"

set "AREA_CSV=%TMP%\requested_area_local.csv"
set "AREA_WGS84=%TMP%\requested_area_wgs84.geojson"

set "SELECTED=%TMP%\selected_tiles.csv"
set "TILELIST=%TMP%\tile_list.txt"

set "VRT=%WORK%\meta_chm_selected.vrt"

set "OUTPUT=%OUTPUT_DIR%\%OUTNAME%_native.tif"


REM ============================================================
REM Meta CHM public AWS dataset
REM ============================================================

set "S3_BASE=https://dataforgood-fb-data.s3.amazonaws.com/forests/v2/global/dinov3_global_chm_v2_ml3"

set "INDEX_URL=%S3_BASE%/tiles.geojson"
set "CHM_URL=%S3_BASE%/chm"


REM ============================================================
REM Create directories
REM ============================================================

if not exist "%WORK%" mkdir "%WORK%"
if not exist "%CACHE%" mkdir "%CACHE%"
if not exist "%TMP%" mkdir "%TMP%"
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"


REM ============================================================
REM Check commands
REM ============================================================

for %%C in (
    ogr2ogr
    ogrinfo
    gdalinfo
    gdalbuildvrt
    gdal_translate
    curl
) do (

    where %%C >nul 2>&1

    if errorlevel 1 (
        echo.
        echo [ERROR] Required command not found: %%C
        echo.
        exit /b 3
    )
)


echo.
echo ============================================================
echo Meta CHM processing
echo ============================================================
echo.
echo Center:
echo   Longitude : %LON%
echo   Latitude  : %LAT%
echo.
echo Requested size:
echo   Width     : %WIDTH% m
echo   Height    : %HEIGHT% m


REM ============================================================
REM 1. Create local AEQD CRS
REM ============================================================

echo.
echo [1/8] Creating local metric coordinate system...


REM ------------------------------------------------------------
REM Local Azimuthal Equidistant projection.
REM
REM The requested center becomes approximately:
REM   X = 0
REM   Y = 0
REM
REM Distances around the center are expressed in metres.
REM ------------------------------------------------------------

set "LOCAL_SRS=+proj=aeqd +lat_0=%LAT% +lon_0=%LON% +datum=WGS84 +units=m +no_defs"


REM ------------------------------------------------------------
REM Windows SET /A uses integers.
REM Width and height are therefore treated as integer metres.
REM ------------------------------------------------------------

set /a HALF_WIDTH=%WIDTH% / 2
set /a HALF_HEIGHT=%HEIGHT% / 2

set /a XMIN=0-HALF_WIDTH
set /a XMAX=HALF_WIDTH

set /a YMIN=0-HALF_HEIGHT
set /a YMAX=HALF_HEIGHT


echo.
echo Local metric extent:
echo   XMIN : !XMIN!
echo   YMIN : !YMIN!
echo   XMAX : !XMAX!
echo   YMAX : !YMAX!


REM ============================================================
REM 2. Create requested-area polygon in local AEQD
REM ============================================================

echo.
echo [2/8] Creating requested-area polygon...


if exist "%AREA_CSV%" del "%AREA_CSV%"
if exist "%AREA_WGS84%" del "%AREA_WGS84%"


REM ------------------------------------------------------------
REM Write one WKT polygon to CSV.
REM ------------------------------------------------------------

echo WKT>"%AREA_CSV%"

echo "POLYGON ((!XMIN! !YMIN!, !XMIN! !YMAX!, !XMAX! !YMAX!, !XMAX! !YMIN!, !XMIN! !YMIN!))">>"%AREA_CSV%"


REM ------------------------------------------------------------
REM Convert local metric rectangle to WGS84.
REM ------------------------------------------------------------

ogr2ogr ^
    -f GeoJSON ^
    "%AREA_WGS84%" ^
    "%AREA_CSV%" ^
    -oo GEOM_POSSIBLE_NAMES=WKT ^
    -a_srs "%LOCAL_SRS%" ^
    -t_srs EPSG:4326


if errorlevel 1 (
    echo.
    echo [ERROR] Failed to create WGS84 requested-area polygon.
    exit /b 4
)


REM ============================================================
REM 3. Read WGS84 bounding box
REM ============================================================

echo.
echo [3/8] Reading requested-area WGS84 extent...


set "WEST="
set "SOUTH="
set "EAST="
set "NORTH="


REM ------------------------------------------------------------
REM ogrinfo normally returns:
REM
REM Extent: (xmin, ymin) - (xmax, ymax)
REM ------------------------------------------------------------

for /f "tokens=2,3,5,6 delims=(), " %%A in ('
    ogrinfo -so -al "%AREA_WGS84%" ^| findstr /B /C:"Extent:"
') do (

    set "WEST=%%A"
    set "SOUTH=%%B"
    set "EAST=%%C"
    set "NORTH=%%D"
)


if not defined WEST (
    echo.
    echo [ERROR] Failed to read requested-area extent.
    exit /b 5
)


echo.
echo WGS84 extent:
echo   West  : !WEST!
echo   South : !SOUTH!
echo   East  : !EAST!
echo   North : !NORTH!


REM ============================================================
REM 4. Prepare Meta tile index
REM ============================================================

echo.
echo [4/8] Preparing Meta CHM tile index...


if not exist "%INDEX%" (

    echo   Downloading tiles.geojson...

    curl ^
        -L ^
        --fail ^
        --retry 3 ^
        --retry-delay 2 ^
        -o "%INDEX%.part" ^
        "%INDEX_URL%"


    if errorlevel 1 (

        if exist "%INDEX%.part" del "%INDEX%.part"

        echo.
        echo [ERROR] Failed to download tiles.geojson.
        exit /b 6
    )


    move /y "%INDEX%.part" "%INDEX%" >nul

) else (

    echo   Using cached tiles.geojson.
)


REM ============================================================
REM 5. Detect tile ID field
REM ============================================================

echo.
echo [5/8] Detecting Meta tile identifier field...


set "TILE_FIELD="


ogrinfo -so -al "%INDEX%" | findstr /I /C:"tile:" >nul
if not errorlevel 1 set "TILE_FIELD=tile"


if not defined TILE_FIELD (

    ogrinfo -so -al "%INDEX%" | findstr /I /C:"quadkey:" >nul

    if not errorlevel 1 set "TILE_FIELD=quadkey"
)


if not defined TILE_FIELD (

    ogrinfo -so -al "%INDEX%" | findstr /I /C:"quad_key:" >nul

    if not errorlevel 1 set "TILE_FIELD=quad_key"
)


if not defined TILE_FIELD (

    ogrinfo -so -al "%INDEX%" | findstr /I /C:"tile_id:" >nul

    if not errorlevel 1 set "TILE_FIELD=tile_id"
)


if not defined TILE_FIELD (

    ogrinfo -so -al "%INDEX%" | findstr /I /C:"tileid:" >nul

    if not errorlevel 1 set "TILE_FIELD=tileid"
)


if not defined TILE_FIELD (

    echo.
    echo [ERROR] Could not identify tile ID field in tiles.geojson.
    echo.
    echo Please inspect:
    echo.
    echo   ogrinfo -so -al "%INDEX%"
    echo.
    exit /b 7
)


echo   Tile ID field: !TILE_FIELD!


REM ============================================================
REM 6. Select intersecting Meta tiles
REM ============================================================

echo.
echo [6/8] Selecting intersecting Meta CHM tiles...


if exist "%SELECTED%" del "%SELECTED%"


REM ------------------------------------------------------------
REM Use the actual requested polygon rather than only its bbox.
REM
REM This avoids needing to calculate intersections ourselves.
REM ------------------------------------------------------------

ogr2ogr ^
    -f CSV ^
    "%SELECTED%" ^
    "%INDEX%" ^
    -clipsrc "%AREA_WGS84%" ^
    -select !TILE_FIELD!


if errorlevel 1 (
    echo.
    echo [ERROR] Meta tile selection failed.
    exit /b 8
)


if exist "%TILELIST%" del "%TILELIST%"


set /a TILECOUNT=0


REM ============================================================
REM 7. Download selected CHM tiles
REM ============================================================

echo.
echo [7/8] Downloading / validating Meta CHM tiles...


for /f "usebackq skip=1 tokens=1 delims=," %%T in ("%SELECTED%") do (

    set "TILE=%%~T"

    set "TILE=!TILE:"=!"


    if not "!TILE!"=="" (

        set /a TILECOUNT+=1


        set "LOCAL=%CACHE%\!TILE!.tif"


        REM ----------------------------------------------------
        REM Reuse valid cached tile
        REM ----------------------------------------------------

        if exist "!LOCAL!" (

            gdalinfo "!LOCAL!" >nul 2>&1


            if errorlevel 1 (

                echo   Cached !TILE!.tif is invalid.
                echo   Removing invalid cache...

                del "!LOCAL!"

            ) else (

                echo   Using cached !TILE!.tif
            )
        )


        REM ----------------------------------------------------
        REM Download if not cached
        REM ----------------------------------------------------

        if not exist "!LOCAL!" (

            echo   Downloading !TILE!.tif


            curl ^
                -L ^
                --fail ^
                --retry 3 ^
                --retry-delay 2 ^
                -o "!LOCAL!.part" ^
                "%CHM_URL%/!TILE!.tif"


            if errorlevel 1 (

                if exist "!LOCAL!.part" del "!LOCAL!.part"

                echo.
                echo [ERROR] Download failed:
                echo   !TILE!

                exit /b 9
            )


            move /y "!LOCAL!.part" "!LOCAL!" >nul
        )


        >>"%TILELIST%" echo !LOCAL!
    )
)


if !TILECOUNT! LEQ 0 (

    echo.
    echo [ERROR] No Meta CHM tiles intersect the requested area.
    echo.
    echo Check the input coordinate and extent.
    echo.

    exit /b 10
)


echo.
echo   Selected tiles: !TILECOUNT!


REM ============================================================
REM 8a. Build VRT
REM ============================================================

echo.
echo [8/8] Building VRT and cropping Meta native CHM...


gdalbuildvrt ^
    -overwrite ^
    -srcnodata 255 ^
    -vrtnodata 255 ^
    -input_file_list "%TILELIST%" ^
    "%VRT%"


if errorlevel 1 (

    echo.
    echo [ERROR] VRT creation failed.
    exit /b 11
)


REM ============================================================
REM 8b. Crop native Meta grid
REM ============================================================

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

    echo.
    echo [ERROR] Final Meta CHM crop failed.
    exit /b 12
)


REM ============================================================
REM Finished
REM ============================================================

echo.
echo ============================================================
echo SUCCESS
echo ============================================================
echo.

echo Output:
echo   %OUTPUT%

echo.
echo Selected Meta tiles:


for /f "usebackq skip=1 tokens=1 delims=," %%T in ("%SELECTED%") do (
    echo   %%~T
)


echo.
echo Output summary:
echo.


gdalinfo "%OUTPUT%" | findstr ^
    /C:"Size is" ^
    /C:"Pixel Size" ^
    /C:"NoData Value" ^
    /C:"COMPRESSION"


echo.
echo ============================================================
echo NOTE
echo ============================================================
echo.
echo The output keeps the Meta CHM native raster grid:
echo.
echo   CRS        : EPSG:3857
echo   Pixel size : approximately 1.194 m
echo   Data type  : Byte
echo   NoData     : 255
echo.
echo The requested width / height are defined in a temporary
echo local Azimuthal Equidistant projection centered on the
echo input longitude / latitude.
echo.
echo Therefore this version is not tied to Tokyo EPSG:6677
echo and can be used throughout Japan.
echo.
echo ============================================================


exit /b 0
