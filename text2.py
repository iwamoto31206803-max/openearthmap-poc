cd /d C:\OpenEarthMap_PoC\oemsar_data

echo ===== DIRECTORY TREE =====
dir /ad /s /b val_labels

echo.
echo ===== EXACT LABEL COUNT =====
dir /b val_labels\val\labels\*.tif | find /c /v ""

echo.
echo ===== ALL TIFF PARENT FOLDERS =====
for /r val_labels %F in (*.tif) do @echo %~dpF
