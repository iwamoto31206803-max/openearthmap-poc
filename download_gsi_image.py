from pathlib import Path
import argparse
import math
import urllib.request
from io import BytesIO

from PIL import Image


TILE_SIZE = 256
GSI_TILE_URL = "https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg"


def lonlat_to_tile(lon: float, lat: float, zoom: int):
    """Convert WGS84 longitude/latitude to XYZ tile coordinates."""
    lat = max(min(lat, 85.05112878), -85.05112878)

    n = 2 ** zoom
    x = (lon + 180.0) / 360.0 * n

    lat_rad = math.radians(lat)
    y = (
        1.0
        - math.asinh(math.tan(lat_rad)) / math.pi
    ) / 2.0 * n

    return x, y


def download_tile(z: int, x: int, y: int) -> Image.Image:
    url = GSI_TILE_URL.format(z=z, x=x, y=y)

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "openearthmap-poc/0.1"
        },
    )

    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read()

    return Image.open(BytesIO(data)).convert("RGB")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Download a square RGB image around a longitude/latitude "
            "from GSI Seamless Aerial Photography tiles."
        )
    )

    parser.add_argument(
        "--lat",
        type=float,
        required=True,
        help="Center latitude in decimal degrees.",
    )

    parser.add_argument(
        "--lon",
        type=float,
        required=True,
        help="Center longitude in decimal degrees.",
    )

    parser.add_argument(
        "--zoom",
        type=int,
        default=18,
        help="XYZ zoom level. GSI seamlessphoto supports 14-18. Default: 18",
    )

    parser.add_argument(
        "--tiles",
        type=int,
        default=3,
        help=(
            "Number of tiles per side. Use an odd number such as 1, 3, or 5. "
            "Default: 3"
        ),
    )

    parser.add_argument(
        "--output",
        default="gsi_test_image.jpg",
        help="Output JPEG path.",
    )

    args = parser.parse_args()

    if args.zoom < 14 or args.zoom > 18:
        raise ValueError(
            "GSI seamlessphoto supports zoom levels 14-18."
        )

    if args.tiles < 1 or args.tiles % 2 == 0:
        raise ValueError(
            "--tiles must be an odd integer such as 1, 3, or 5."
        )

    tile_x_f, tile_y_f = lonlat_to_tile(
        args.lon,
        args.lat,
        args.zoom,
    )

    center_x = int(math.floor(tile_x_f))
    center_y = int(math.floor(tile_y_f))

    radius = args.tiles // 2

    min_x = center_x - radius
    max_x = center_x + radius
    min_y = center_y - radius
    max_y = center_y + radius

    mosaic = Image.new(
        "RGB",
        (
            args.tiles * TILE_SIZE,
            args.tiles * TILE_SIZE,
        ),
    )

    print("GSI seamless aerial photo download")
    print("----------------------------------")
    print(f"Center lat/lon : {args.lat}, {args.lon}")
    print(f"Zoom           : {args.zoom}")
    print(f"Center tile    : x={center_x}, y={center_y}")
    print(f"Tile grid      : {args.tiles} x {args.tiles}")
    print(
        f"Output size    : "
        f"{args.tiles * TILE_SIZE} x {args.tiles * TILE_SIZE} px"
    )
    print()

    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            print(f"Downloading z={args.zoom}, x={x}, y={y}")

            tile = download_tile(
                args.zoom,
                x,
                y,
            )

            paste_x = (x - min_x) * TILE_SIZE
            paste_y = (y - min_y) * TILE_SIZE

            mosaic.paste(
                tile,
                (paste_x, paste_y),
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    mosaic.save(
        output_path,
        format="JPEG",
        quality=95,
    )

    print()
    print("----------------------------------")
    print("DOWNLOAD OK")
    print("Saved:", output_path)
    print(
        "Source: GSI Tiles / Seamless Aerial Photography "
        "(全国最新写真（シームレス）)"
    )
    print(
        "Attribution: 国土地理院 / GSI Tiles"
    )


if __name__ == "__main__":
    main()
