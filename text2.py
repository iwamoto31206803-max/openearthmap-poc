cd /d C:\OpenEarthMap_PoC\oemsar_data

echo ===== ZIP =====
dir dfc25_track1_trainval.zip

echo.
echo ===== EXTRACT =====
powershell -NoProfile -Command "Expand-Archive -Path 'C:\OpenEarthMap_PoC\oemsar_data\dfc25_track1_trainval.zip' -DestinationPath 'C:\OpenEarthMap_PoC\oemsar_data\trainval' -Force"

echo.
echo ===== TOP LEVEL =====
dir C:\OpenEarthMap_PoC\oemsar_data\trainval

echo.
echo ===== DIRECTORIES =====
dir /s /b /ad C:\OpenEarthMap_PoC\oemsar_data\trainval | more

echo.
echo ===== TIFF COUNT =====
dir /s /b C:\OpenEarthMap_PoC\oemsar_data\trainval\*.tif | find /c /v ""

echo.
echo ===== LABEL-LIKE FOLDERS =====
dir /s /b /ad C:\OpenEarthMap_PoC\oemsar_data\trainval\*label* 2>nul

echo.
echo ===== RGB / SAR FOLDERS =====
dir /s /b /ad C:\OpenEarthMap_PoC\oemsar_data\trainval\*rgb* 2>nul
dir /s /b /ad C:\OpenEarthMap_PoC\oemsar_data\trainval\*sar* 2>nul

echo.
echo ===== TXT / CSV / JSON =====
dir /s /b C:\OpenEarthMap_PoC\oemsar_data\trainval\*.txt 2>nul
dir /s /b C:\OpenEarthMap_PoC\oemsar_data\trainval\*.csv 2>nul
dir /s /b C:\OpenEarthMap_PoC\oemsar_data\trainval\*.json 2>nul
