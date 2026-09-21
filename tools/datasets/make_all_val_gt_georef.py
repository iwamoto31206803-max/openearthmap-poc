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
# Japan 54枚
# ============================================================

japan_files = []

with open(GEO_CSV, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        if row["country"].strip() == "Japan":
            japan_files.append(row["file"].strip())

japan_files = sorted(set(japan_files))

print("Japan target:", len(japan_files))

# ============================================================
# QGIS QML
# 0 = transparent
# 1-8 = OEM8
# opacity = 0.45
# ============================================================

QML = """<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.40" styleCategories="AllStyleCategories">
  <pipe>
    <rasterrenderer
        type="paletted"
        band="1"
        opacity="0.45"
        alphaBand="-1"
        nodataColor="">
      <colorPalette>
        <paletteEntry value="0" color="#000000" alpha="0" label="Background"/>
        <paletteEntry value="1" color="#b45046" alpha="255" label="Bareland"/>
        <paletteEntry value="2" color="#6ec846" alpha="255" label="Grass / Rangeland"/>
        <paletteEntry value="3" color="#bebebe" alpha="255" label="Pavement / Developed space"/>
        <paletteEntry value="4" color="#ebdcbe" alpha="255" label="Road"/>
        <paletteEntry value="5" color="#417d4b" alpha="255" label="Tree"/>
        <paletteEntry value="6" color="#466edc" alpha="255" label="Water"/>
        <paletteEntry value="7" color="#7dbe73" alpha="255" label="Agriculture land"/>
        <paletteEntry value="8" color="#dc4b3c" alpha="255" label="Building"/>
      </colorPalette>
    </rasterrenderer>
  </pipe>
</qgis>
"""

# ============================================================
# 一括生成
# ============================================================

ok = 0
ng = 0

for filename in japan_files:

    sar_path = SAR_DIR / filename
    gt_path = GT_DIR / filename

    stem = Path(filename).stem

    out_tif = OUT_DIR / f"{stem}_gt_georef.tif"
    out_qml = OUT_DIR / f"{stem}_gt_georef.qml"

    if not sar_path.exists():
        print("[NG] SAR missing:", sar_path)
        ng += 1
        continue

    if not gt_path.exists():
        print("[NG] GT missing:", gt_path)
        ng += 1
        continue

    try:
        with rasterio.open(sar_path) as sar, rasterio.open(gt_path) as gt:

            if (sar.width, sar.height) != (gt.width, gt.height):
                raise RuntimeError(
                    f"size mismatch: "
                    f"SAR={sar.width}x{sar.height}, "
                    f"GT={gt.width}x{gt.height}"
                )

            data = gt.read()

            # 元GTのprofileをそのまま使い、
            # SARの地理参照だけコピー
            profile = gt.profile.copy()

            profile.update(
                driver="GTiff",
                crs=sar.crs,
                transform=sar.transform,
            )

            # tiled/block設定は明示的に外す
            profile.pop("blockxsize", None)
            profile.pop("blockysize", None)
            profile.pop("tiled", None)

            with rasterio.open(out_tif, "w", **profile) as dst:
                dst.write(data)

        # QML sidecar
        out_qml.write_text(QML, encoding="utf-8")

        # 書いたTIFFを再オープンして健全性確認
        with rasterio.open(out_tif) as check:
            _ = check.read(1, window=((0, 1), (0, 1)))

        print("[OK]", out_tif.name)
        ok += 1

    except Exception as e:
        print("[NG]", filename, "->", e)

        # 失敗途中の壊れたファイルを残さない
        if out_tif.exists():
            try:
                out_tif.unlink()
            except Exception:
                pass

        if out_qml.exists():
            try:
                out_qml.unlink()
            except Exception:
                pass

        ng += 1

print()
print("===== DONE =====")
print("Japan target :", len(japan_files))
print("OK           :", ok)
print("NG           :", ng)
print("OUT          :", OUT_DIR)
