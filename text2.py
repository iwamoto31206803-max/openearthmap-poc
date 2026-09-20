cd /d C:\OpenEarthMap_PoC

mkdir oemsar_data 2>nul
cd oemsar_data

curl -L -C - -o dfc25_track1_trainval.zip ^
"https://zenodo.org/records/14622048/files/dfc25_track1_trainval.zip?download=1"
