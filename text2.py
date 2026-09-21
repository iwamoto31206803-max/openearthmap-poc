cd /d C:\OpenEarthMap_PoC\oemsar_data

echo ===== VAL SAR FIRST 20 =====
dir /b trainval\val\sar_images\*.tif | more

echo.
echo ===== VAL LABEL FIRST 20 =====
dir /b val_labels\val\labels\*.tif | more

echo.
echo ===== COUNTS =====
dir /b trainval\val\sar_images\*.tif | find /c /v ""
dir /b val_labels\val\labels\*.tif | find /c /v ""


gdalinfo "C:\OpenEarthMap_PoC\oemsar_data\val_labels\val\labels\ValArea_001.tif"

python -c "import rasterio, numpy as np; p=r'C:\OpenEarthMap_PoC\oemsar_data\val_labels\val\labels\ValArea_001.tif'; a=rasterio.open(p).read(1); print(np.unique(a, return_counts=True))"
