cd /d C:\OpenEarthMap_PoC

echo ===== 1. RGB_Real_5 checkpoint =====
dir /s /b RGB_Real_5_u-efficientnet-b4.pth 2>nul

echo.
echo ===== 2. OEM-SARらしいフォルダ =====
dir /s /b /ad *OpenEarthMap* 2>nul
dir /s /b /ad *OEM*SAR* 2>nul

echo.
echo ===== 3. train / val / labels / rgb_images folders =====
dir /s /b /ad train 2>nul
dir /s /b /ad val 2>nul
dir /s /b /ad labels 2>nul
dir /s /b /ad rgb_images 2>nul
dir /s /b /ad sar_images 2>nul

echo.
echo ===== 4. TrainArea / ValArea files =====
dir /s /b *TrainArea*.tif 2>nul
dir /s /b *ValArea*.tif 2>nul
