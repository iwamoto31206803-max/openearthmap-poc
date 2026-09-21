import csv
from pathlib import Path

import rasterio

ROOT = Path(r"C:\OpenEarthMap_PoC\oemsar_data")

SAR_DIR = ROOT / "trainval" / "val" / "sar_images"
GT_DIR = ROOT / "val_labels" / "val" / "labels"
OUT_DIR = ROOT / "val_gt_georef"
GEO_CSV = ROOT / "val_geography.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# OEM8 style
# 0 = transparent
# 1-8 = OEM8 classes
# alpha=115 ≒ 45%
# ============================================================

ALPHA = 115

OEM8_COLORMAP = {
    0: (0, 0, 0, 0),
    1: (180, 80, 70, ALPHA),      # Bareland
    2: (110, 200, 70, ALPHA),     # Grass / Rangeland
    3: (190, 190, 190, ALPHA),    # Pavement / Developed space
    4: (235, 220, 190, ALPHA),    # Road
    5: (65, 125, 75, ALPHA),      # Tree
    6: (70, 110, 220, ALPHA),     # Water
    7: (125, 190, 115, ALPHA),    # Agriculture land
    8: (220, 75, 60, ALPHA),      # Building
}

# ============================================================
# Japan 54枚を抽出
# ============================================================

japan_files = set()

with open(GEO_CSV, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        if row["country"].strip() == "Japan":
            japan_files.add(row["file"].strip())

print("Japan files:", len(japan_files))

# ============================================================
# Georeference + style
# ============================================================

ok = 0
ng = 0

for filename in sorted(japan_files):

    sar_path = SAR_DIR / filename
    gt_path = GT_DIR / filename

    out_name = filename.replace(
        ".tif",
        "_gt_georef.tif"
    )

    out_path = OUT_DIR / out_name

    if not sar_path.exists():
        print("[NG] SAR missing:", sar_path)
        ng += 1
        continue

    if not gt_path.exists():
        print("[NG] GT missing:", gt_path)
        ng += 1
        continue

    try:
        with rasterio.open(sar_path) as sar:
            with rasterio.open(gt_path) as gt:

                if (
                    sar.width != gt.width
                    or sar.height != gt.height
                ):
                    raise RuntimeError(
                        f"size mismatch "
                        f"SAR={sar.width}x{sar.height}, "
                        f"GT={gt.width}x{gt.height}"
                    )

                profile = gt.profile.copy()

                profile.update(
                    driver="GTiff",
                    crs=sar.crs,
                    transform=sar.transform,
                    compress="lzw",
                    tiled=True,
                )

                data = gt.read()

                with rasterio.open(
                    out_path,
                    "w",
                    **profile
                ) as dst:
                    dst.write(data)

        print("[OK]", out_name)
        ok += 1

    except Exception as e:
        print("[NG]", filename, "->", e)
        ng += 1

print()
print("===== DONE =====")
print("Japan target :", len(japan_files))
print("OK           :", ok)
print("NG           :", ng)
print("OUT          :", OUT_DIR)
