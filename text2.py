cd /d C:\OpenEarthMap_PoC\oemsar_data\trainval

echo ===== COUNTS =====
echo train Labels:
dir /b train\Labels\*.tif | find /c /v ""

echo train RGB:
dir /b train\rgb_images\*.tif | find /c /v ""

echo train SAR:
dir /b train\sar_images\*.tif | find /c /v ""

echo val SAR:
dir /b val\sar_images\*.tif | find /c /v ""

echo.
echo ===== FIRST 20 LABELS =====
dir /b train\Labels\*.tif | more

echo.
echo ===== FIRST 20 RGB =====
dir /b train\rgb_images\*.tif | more

echo.
echo ===== FIRST 20 SAR =====
dir /b train\sar_images\*.tif | more
