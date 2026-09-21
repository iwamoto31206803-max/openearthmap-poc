import glob
import math
import os

import rasterio
from rasterio.warp import transform_bounds

VAL_DIR = r"C:\OpenEarthMap_PoC\oemsar_data\trainval\val\sar_images"
TRAIN_RGB_DIR = r"C:\OpenEarthMap_PoC\oemsar_data\trainval\train\rgb_images"

# 日本 validation 54枚
val_files = sorted(glob.glob(os.path.join(VAL_DIR, "*.tif")))

vals = []

for p in val_files:
    with rasterio.open(p) as ds:
        b = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds)
        left, bottom, right, top = b
        lon = (left + right) / 2
        lat = (bottom + top) / 2

        if 122 <= lon <= 154 and 20 <= lat <= 46:
            vals.append({
                "file": os.path.basename(p),
                "lon": lon,
                "lat": lat,
            })

print("Japan validation:", len(vals))

# train RGB の中心座標を取得
rgb_files = sorted(glob.glob(os.path.join(TRAIN_RGB_DIR, "*.tif")))
rgbs = []

for i, p in enumerate(rgb_files, start=1):
    with rasterio.open(p) as ds:
        if ds.crs is None:
            continue

        b = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds)
        left, bottom, right, top = b

        rgbs.append({
            "file": os.path.basename(p),
            "lon": (left + right) / 2,
            "lat": (bottom + top) / 2,
        })

    if i % 500 == 0:
        print("read RGB:", i)

def distance_m(a, b):
    # この距離なら簡易近似で十分
    lat0 = math.radians((a["lat"] + b["lat"]) / 2)
    dx = (a["lon"] - b["lon"]) * 111320 * math.cos(lat0)
    dy = (a["lat"] - b["lat"]) * 110540
    return math.sqrt(dx * dx + dy * dy)

print()
print("=== NEAREST TRAIN RGB FOR EACH JAPAN VAL ===")

exact_like = 0

for v in vals:
    nearest = min(rgbs, key=lambda r: distance_m(v, r))
    d = distance_m(v, nearest)

    if d < 10:
        exact_like += 1

    print(
        f"{v['file']} -> {nearest['file']} "
        f"distance={d:.1f} m "
        f"val=({v['lat']:.6f},{v['lon']:.6f}) "
        f"rgb=({nearest['lat']:.6f},{nearest['lon']:.6f})"
    )

print()
print("within 10 m:", exact_like, "/", len(vals))
