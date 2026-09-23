from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import xy
from shapely.geometry import box


# ============================================================
# Settings
# ============================================================

INPUT_CHM = Path(
    r"C:\tokyo-lidar-treeheight\02_work\09LD3441_CHM_tree_max_1m.tif"
)

OUTPUT_GPKG = Path(
    r"C:\tokyo-lidar-treeheight\03_output\09LD3441_canopy_cells_CR070.gpkg"
)

LAYER_NAME = "canopy_cells"

CROWN_RATIO = 0.70

# Ignore NoData and non-positive heights.
MIN_HEIGHT_M = 0.01


# ============================================================
# Read CHM
# ============================================================

with rasterio.open(INPUT_CHM) as src:
    chm = src.read(1)

    transform = src.transform
    crs = src.crs
    nodata = src.nodata

    pixel_width = abs(transform.a)
    pixel_height = abs(transform.e)

    print(f"Input CRS: {crs}")
    print(f"Pixel size: {pixel_width} x {pixel_height} m")
    print(f"NoData: {nodata}")
    print(f"Raster size: {src.width} x {src.height}")

    valid = np.isfinite(chm)

    if nodata is not None:
        valid &= chm != nodata

    valid &= chm >= MIN_HEIGHT_M

    rows, cols = np.where(valid)
    heights = chm[rows, cols].astype(float)


# ============================================================
# Build polygons
# ============================================================

geometries = []

half_w = pixel_width / 2
half_h = pixel_height / 2

for row, col in zip(rows, cols):
    x, y = xy(transform, row, col, offset="center")

    geom = box(
        x - half_w,
        y - half_h,
        x + half_w,
        y + half_h,
    )

    geometries.append(geom)


# ============================================================
# Attributes
# ============================================================

cbh = heights * (1.0 - CROWN_RATIO)
crown_depth = heights * CROWN_RATIO

gdf = gpd.GeoDataFrame(
    {
        "height_m": heights,
        "cr": CROWN_RATIO,
        "cbh_m": cbh,
        "crown_d_m": crown_depth,
    },
    geometry=geometries,
    crs=crs,
)


# ============================================================
# Write output
# ============================================================

OUTPUT_GPKG.parent.mkdir(parents=True, exist_ok=True)

gdf.to_file(
    OUTPUT_GPKG,
    layer=LAYER_NAME,
    driver="GPKG",
)

print()
print("DONE")
print(f"Cells: {len(gdf):,}")
print(f"Output: {OUTPUT_GPKG}")
print()
print(gdf[["height_m", "cbh_m", "crown_d_m"]].describe())
