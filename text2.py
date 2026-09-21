import csv
import glob
import os
from collections import Counter

import rasterio
from rasterio.warp import transform_bounds

files = sorted(
    glob.glob(
        r"C:\OpenEarthMap_PoC\oemsar_data\trainval\val\sar_images\*.tif"
    )
)

rows = []

for path in files:
    with rasterio.open(path) as ds:
        b = transform_bounds(
            ds.crs,
            "EPSG:4326",
            *ds.bounds
        )

        lon = (b.left + b.right) / 2
        lat = (b.bottom + b.top) / 2

        if 122 <= lon <= 154 and 20 <= lat <= 46:
            country = "Japan"
        elif -6 <= lon <= 10 and 41 <= lat <= 52:
            country = "France"
        elif -130 <= lon <= -60 and 20 <= lat <= 55:
            country = "USA"
        else:
            country = "Other"

        rows.append([
            os.path.basename(path),
            str(ds.crs),
            lon,
            lat,
            country,
        ])

out_csv = r"C:\OpenEarthMap_PoC\oemsar_data\val_geography.csv"

with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["file", "crs", "lon", "lat", "country"])
    writer.writerows(rows)

print("=== COUNTRY COUNTS ===")
print(Counter(r[4] for r in rows))

print()
print("=== JAPAN ===")
for r in rows:
    if r[4] == "Japan":
        print(r)

print()
print("CSV:", out_csv)
