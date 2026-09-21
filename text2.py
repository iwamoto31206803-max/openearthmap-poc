import glob
import math
import os

import rasterio
from rasterio.warp import transform_bounds

VAL_DIR = r"C:\OpenEarthMap_PoC\oemsar_data\trainval\val\sar_images"
TRAIN_SAR_DIR = r"C:\OpenEarthMap_PoC\oemsar_data\trainval\train\sar_images"
TRAIN_RGB_DIR = r"C:\OpenEarthMap_PoC\oemsar_data\trainval\train\rgb_images"

# -------------------------
# Japan validation 54枚
# -------------------------

val_files = sorted(glob.glob(os.path.join(VAL_DIR, "*.tif")))

vals = []

for p in val_files:
    with rasterio.open(p) as ds:
        if ds.crs is None:
            continue

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

# -------------------------
# Train SARから位置を取得
# 同名RGBが存在するものだけ使う
# -------------------------

sar_files = sorted(glob.glob(os.path.join(TRAIN_SAR_DIR, "*.tif")))

trains = []

for i, p in enumerate(sar_files, start=1):
    name = os.path.basename(p)

    rgb_path = os.path.join(TRAIN_RGB_DIR, name)

    if not os.path.exists(rgb_path):
        continue

    with rasterio.open(p) as ds:
        if ds.crs is None:
            continue

        b = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds)
        left, bottom, right, top = b

        trains.append({
            "file": name,
            "rgb_path": rgb_path,
            "lon": (left + right) / 2,
            "lat": (bottom + top) / 2,
        })

    if i % 500 == 0:
        print("read train SAR:", i)

print("georeferenced train pairs:", len(trains))

if not trains:
    raise RuntimeError("No georeferenced train SAR/RGB pairs found.")

# -------------------------
# 距離
# -------------------------

def distance_m(a, b):
    lat0 = math.radians((a["lat"] + b["lat"]) / 2)

    dx = (
        (a["lon"] - b["lon"])
        * 111320
        * math.cos(lat0)
    )

    dy = (
        (a["lat"] - b["lat"])
        * 110540
    )

    return math.sqrt(dx * dx + dy * dy)

# -------------------------
# 各Japan validationに
# 最も近いtrain RGBを探す
# -------------------------

print()
print("=== NEAREST TRAIN RGB FOR EACH JAPAN VAL ===")

within_10m = 0
within_100m = 0
within_1000m = 0

for v in vals:
    nearest = min(
        trains,
        key=lambda r: distance_m(v, r)
    )

    d = distance_m(v, nearest)

    if d < 10:
        within_10m += 1

    if d < 100:
        within_100m += 1

    if d < 1000:
        within_1000m += 1

    print(
        f"{v['file']} -> {nearest['file']} "
        f"distance={d:.1f} m "
        f"val=({v['lat']:.6f},{v['lon']:.6f}) "
        f"train=({nearest['lat']:.6f},{nearest['lon']:.6f})"
    )

print()
print("within 10 m :", within_10m, "/", len(vals))
print("within 100 m:", within_100m, "/", len(vals))
print("within 1 km :", within_1000m, "/", len(vals))
