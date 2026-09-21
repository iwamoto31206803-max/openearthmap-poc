import rasterio
from rasterio.warp import transform_bounds

p = r"C:\OpenEarthMap_PoC\oemsar_data\trainval\val\sar_images\ValArea_011.tif"

with rasterio.open(p) as ds:
    print("CRS:", ds.crs)
    print("size:", ds.width, ds.height)
    print("transform:", ds.transform)
    print("bounds_native:", ds.bounds)

    b = transform_bounds(
        ds.crs,
        "EPSG:4326",
        *ds.bounds,
        densify_pts=21,
    )

    left, bottom, right, top = b

    center_lon = (left + right) / 2
    center_lat = (bottom + top) / 2

    print()
    print("=== WGS84 ===")
    print("left  :", left)
    print("bottom:", bottom)
    print("right :", right)
    print("top   :", top)

    print()
    print("center lat/lon:", center_lat, center_lon)

    print()
    print("NW:", top, left)
    print("NE:", top, right)
    print("SW:", bottom, left)
    print("SE:", bottom, right)
