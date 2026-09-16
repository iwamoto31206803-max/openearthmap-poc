from pathlib import Path
import argparse
import math
import urllib.request
from io import BytesIO

import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_bounds

if __package__:
    from .config import GSI_SOURCE_NAME, GSI_SOURCE_NAME_JP, GSI_TILE_SIZE, GSI_TILE_URL
else:
    from config import GSI_SOURCE_NAME, GSI_SOURCE_NAME_JP, GSI_TILE_SIZE, GSI_TILE_URL

TILE_SIZE = GSI_TILE_SIZE
WEB_MERCATOR_RADIUS = 6378137.0
ORIGIN_SHIFT = math.pi * WEB_MERCATOR_RADIUS


def lonlat_to_tile(lon: float, lat: float, zoom: int):
    lat = max(min(lat, 85.05112878), -85.05112878)
    n = 2 ** zoom
    x = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def tile_bounds_3857(x: int, y: int, zoom: int):
    n = 2 ** zoom
    west = (x / n) * (2.0 * ORIGIN_SHIFT) - ORIGIN_SHIFT
    east = ((x + 1) / n) * (2.0 * ORIGIN_SHIFT) - ORIGIN_SHIFT
    north = ORIGIN_SHIFT - (y / n) * (2.0 * ORIGIN_SHIFT)
    south = ORIGIN_SHIFT - ((y + 1) / n) * (2.0 * ORIGIN_SHIFT)
    return west, south, east, north


def download_tile(z: int, x: int, y: int) -> Image.Image:
    url = GSI_TILE_URL.format(z=z, x=x, y=y)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "openearthmap-poc/0.1"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read()
    return Image.open(BytesIO(data)).convert("RGB")


def main():
    parser = argparse.ArgumentParser(
        description="Download GSI seamless aerial photo tiles and save as an RGB GeoTIFF."
    )
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--zoom", type=int, default=18)
    parser.add_argument("--tiles", type=int, default=3)
    parser.add_argument("--output", default="gsi_test_image.tif")
    args = parser.parse_args()

    if args.zoom < 14 or args.zoom > 18:
        raise ValueError("GSI seamlessphoto supports zoom levels 14-18.")
    if args.tiles < 1 or args.tiles % 2 == 0:
        raise ValueError("--tiles must be an odd integer such as 1, 3, or 5.")

    tile_x_f, tile_y_f = lonlat_to_tile(args.lon, args.lat, args.zoom)
    center_x = int(math.floor(tile_x_f))
    center_y = int(math.floor(tile_y_f))

    radius = args.tiles // 2
    min_x, max_x = center_x - radius, center_x + radius
    min_y, max_y = center_y - radius, center_y + radius

    width = args.tiles * TILE_SIZE
    height = args.tiles * TILE_SIZE
    mosaic = Image.new("RGB", (width, height))

    print("GSI seamless aerial photo -> GeoTIFF")
    print("-------------------------------------")
    print(f"Center lat/lon : {args.lat}, {args.lon}")
    print(f"Zoom           : {args.zoom}")
    print(f"Tile grid      : {args.tiles} x {args.tiles}")
    print(f"Output size    : {width} x {height} px")
    print()

    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            print(f"Downloading z={args.zoom}, x={x}, y={y}")
            tile = download_tile(args.zoom, x, y)
            mosaic.paste(
                tile,
                ((x - min_x) * TILE_SIZE, (y - min_y) * TILE_SIZE),
            )

    west, _, _, north = tile_bounds_3857(min_x, min_y, args.zoom)
    _, south, east, _ = tile_bounds_3857(max_x, max_y, args.zoom)

    transform = from_bounds(
        west, south, east, north, width, height
    )

    rgb = np.asarray(mosaic, dtype=np.uint8)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=3,
        dtype="uint8",
        crs="EPSG:3857",
        transform=transform,
        compress="deflate",
        photometric="RGB",
    ) as dst:
        dst.write(rgb[:, :, 0], 1)
        dst.write(rgb[:, :, 1], 2)
        dst.write(rgb[:, :, 2], 3)
        dst.set_band_description(1, "Red")
        dst.set_band_description(2, "Green")
        dst.set_band_description(3, "Blue")
        dst.update_tags(
            source=GSI_SOURCE_NAME,
            source_jp=GSI_SOURCE_NAME_JP,
            center_lat=str(args.lat),
            center_lon=str(args.lon),
            zoom=str(args.zoom),
        )

    print()
    print("-------------------------------------")
    print("GEOTIFF DOWNLOAD OK")
    print("Saved       :", output_path)
    print("CRS         : EPSG:3857")
    print("Attribution : 国土地理院 / GSI Tiles")


if __name__ == "__main__":
    main()
