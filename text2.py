cd /d C:\OpenEarthMap_PoC\oemsar_data

curl -L -o dfc25_track1_val_labels.zip ^
"https://zenodo.org/records/14950559/files/dfc25_track1_val_labels.zip?download=1"

certutil -hashfile dfc25_track1_val_labels.zip MD5

powershell -NoProfile -Command "Expand-Archive -Path 'C:\OpenEarthMap_PoC\oemsar_data\dfc25_track1_val_labels.zip' -DestinationPath 'C:\OpenEarthMap_PoC\oemsar_data\val_labels' -Force"

echo ===== VAL LABELS =====
dir /s /b C:\OpenEarthMap_PoC\oemsar_data\val_labels\*.tif | find /c /v ""

echo.
echo ===== FIRST 20 =====
dir /s /b C:\OpenEarthMap_PoC\oemsar_data\val_labels\*.tif | more
  
