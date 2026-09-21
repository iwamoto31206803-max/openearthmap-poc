from pathlib import Path
import csv

import rasterio
from rasterio.transform import from_origin
from pyproj import Transformer

# =========================
# パス設定
# =========================
ROOT = Path(r"C:\OpenEarthMap_PoC")
GT_DIR = ROOT / "oemsar_data" / "val_gt"
OUT_DIR = ROOT / "oemsar_data" / "val_gt_georef"
GEO_CSV = ROOT / "oemsar_data" / "val_geography.csv"

# 既に作れている georef tif を1枚テンプレートとして使う
# ここは存在しているものに合わせてください
TEMPLATE_TIF = OUT_DIR / "ValArea_011_gt_georef.tif"

OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# geography csv を読む
# =========================
def clean(x):
    return str(x).strip().strip('"').strip("'")

def read_geography(csv_path):
    geo = {}

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        raise RuntimeError(f"CSV is empty: {csv_path}")

    # ヘッダ有無の両対応
    start_idx = 0
    first = [clean(v) for v in rows[0]]

    if len(first) >= 4 and first[0].lower() in ("file", "filename", "tile", "name"):
        start_idx = 1

    for row in rows[start_idx:]:
        if not row or len(row) < 4:
            continue

        row = [clean(v) for v in row]
        fname = row[0]
        epsg = row[1]
        lon = float(row[2])
        lat = float(row[3])

        geo[fname] = {
            "epsg": epsg,
            "lon": lon,
            "lat": lat,
        }

    return geo


# =========================
# テンプレートから解像度取得
# =========================
with rasterio.open(TEMPLATE_TIF) as src:
    xres = src.transform.a
    yres = abs(src.transform.e)

print(f"Template resolution: xres={xres}, yres={yres}")

geo = read_geography(GEO_CSV)
print(f"Geography rows: {len(geo)}")

# =========================
# 一括処理
# =========================
count_ok = 0
count_skip = 0
count_ng = 0

for gt_path in sorted(GT_DIR.glob("ValArea_*.tif")):
    fname = gt_path.name

    if fname not in geo:
        print(f"[SKIP] geography not found: {fname}")
        count_skip += 1
        continue

    meta_geo = geo[fname]
    epsg = meta_geo["epsg"]
    lon = meta_geo["lon"]
    lat = meta_geo["lat"]

    out_name = gt_path.stem + "_gt_georef.tif"
    out_path = OUT_DIR / out_name

    try:
        with rasterio.open(gt_path) as src:
            arr = src.read(1)
            profile = src.profile.copy()
            width = src.width
            height = src.height

        # 中心座標（lon, lat）を対象CRSへ変換
        transformer = Transformer.from_crs("EPSG:4326", epsg, always_xy=True)
        cx, cy = transformer.transform(lon, lat)

        # テンプレート解像度を用いて、中心から左上原点を計算
        left = cx - (width * xres) / 2.0
        top = cy + (height * yres) / 2.0

        transform = from_origin(left, top, xres, yres)

        profile.update(
            driver="GTiff",
            crs=epsg,
            transform=transform,
            compress="lzw",
            tiled=True,
            count=1
        )

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(arr, 1)

        print(f"[OK] {out_name}")
        count_ok += 1

    except Exception as e:
        print(f"[NG] {fname} -> {e}")
        count_ng += 1

print("---- done ----")
print(f"OK   : {count_ok}")
print(f"SKIP : {count_skip}")
print(f"NG   : {count_ng}")
print(f"OUT  : {OUT_DIR}")
