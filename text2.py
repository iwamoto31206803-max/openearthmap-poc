cd /d C:\OpenEarthMap_PoC\oemsar_data

powershell -NoProfile -Command "Expand-Archive -Path 'C:\OpenEarthMap_PoC\oemsar_data\dfc25_track1_trainval.zip' -DestinationPath 'C:\OpenEarthMap_PoC\oemsar_data\trainval' -Force"

cd /d C:\OpenEarthMap_PoC\oemsar_data\trainval

echo ===== TOP LEVEL =====
dir

echo.
echo ===== LABEL-LIKE FOLDERS =====
dir /s /b /ad *label* 2>nul

echo.
echo ===== RGB / SAR FOLDERS =====
dir /s /b /ad *rgb* 2>nul
dir /s /b /ad *sar* 2>nul

echo.
echo ===== TXT / CSV / JSON =====
dir /s /b *.txt 2>nul
dir /s /b *.csv 2>nul
dir /s /b *.json 2>nul
