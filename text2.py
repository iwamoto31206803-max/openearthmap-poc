cd /d C:\OpenEarthMap_PoC\OpenEarthMap-SAR

echo ===== root =====
dir

echo.
echo ===== dataset folders =====
dir /s /b /ad dataset 2>nul
dir /s /b /ad data 2>nul

echo.
echo ===== tif count =====
dir /s /b *.tif 2>nul | find /c /v ""

echo.
echo ===== zip / tar / 7z =====
dir /s /b *.zip 2>nul
dir /s /b *.tar 2>nul
dir /s /b *.gz 2>nul
dir /s /b *.7z 2>nul


certutil -hashfile "C:\OpenEarthMap_PoC\OpenEarthMap-SAR\src\Semantic_Segemtation\pretrained\RGB_Real_5_u-efficientnet-b4.pth" SHA256
