from pathlib import Path
import argparse
import math

import numpy as np
import rasterio
from pyproj import CRS, Geod, Transformer
from rasterio.features import sieve
from rasterio.transform import xy
from shapely.geometry import Polygon
from shapely.ops import transform as shapely_transform


GEOD = Geod(ellps="WGS84")


def pixel_ground_area_m2(transform, crs, width, height) -> float:
    """
    Estimate the real ground area of one raster pixel at the raster center.

    This avoids treating EPSG:3857 projected square metres as true ground
    square metres.
    """
    col = width // 2
    row = height // 2

    x0, y0 = xy(transform, row, col, offset="ul")
    x1, y1 = xy(transform, row, col, offset="ur")
    x2, y2 = xy(transform, row, col, offset="lr")
    x3, y3 = xy(transform, row, col, offset="ll")

    pixel_poly = Polygon([
        (x0, y0),
        (x1, y1),
        (x2, y2),
        (x3, y3),
        (x0, y0),
    ])

    source_crs = CRS.from_user_input(crs)
    if source_crs.to_epsg() != 4326:
        transformer = Transformer.from_crs(
            source_crs,
            "EPSG:4326",
            always_xy=True,
        )
        pixel_poly = shapely_transform(transformer.transform, pixel_poly)

    area, _ = GEOD.geometry_area_perimeter(pixel_poly)
    return abs(float(area))


def build_output_path(
    input_path: Path,
    output_dir: Path,
    area_m2: float,
) -> Path:
    area_label = f"{area_m2:g}".replace(".", "p")
    return output_dir / f"{input_path.stem}_sieve_{area_label}m2.tif"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Apply raster sieve filtering to a land-cover class GeoTIFF. "
            "Thresholds are specified in approximate real ground area (m2) "
            "and converted to connected-pixel counts automatically."
        )
    )
    parser.add_argument(
        "classes",
        help="Path to *_classes.tif produced by predict_geotiff_tiled.py.",
    )
    parser.add_argument(
        "--areas",
        nargs="+",
        type=float,
        default=[2.0, 5.0, 10.0],
        help="Sieve area thresholds in m2. Default: 2 5 10",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        choices=[4, 8],
        default=8,
        help="Pixel connectivity for connected regions. Default: 8",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory. Default: same directory as input GeoTIFF."
        ),
    )
    args = parser.parse_args()

    class_path = Path(args.classes)
    if not class_path.exists():
        raise FileNotFoundError(f"Class GeoTIFF not found: {class_path}")

    if any(area <= 0 for area in args.areas):
        raise ValueError("All --areas values must be greater than 0.")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else class_path.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(class_path) as src:
        if src.count != 1:
            raise ValueError("Class GeoTIFF must have exactly one band.")
        if src.crs is None:
            raise ValueError("Class GeoTIFF has no CRS.")

        class_map = src.read(1)
        profile = src.profile.copy()
        tags = src.tags()
        colormap = None

        try:
            colormap = src.colormap(1)
        except ValueError:
            pass

        ground_pixel_area = pixel_ground_area_m2(
            src.transform,
            src.crs,
            src.width,
            src.height,
        )

        crs = src.crs
        width = src.width
        height = src.height

    print("Land-cover raster sieve")
    print("-------------------------------------")
    print("Input         :", class_path)
    print("Raster size   :", f"{width} x {height}")
    print("CRS           :", crs)
    print("Connectivity  :", args.connectivity)
    print(
        "Pixel area    :",
        f"{ground_pixel_area:.4f} m2 "
        "(estimated at raster center)",
    )
    print()

    unique_before = np.unique(class_map)
    print("Classes before:", ", ".join(map(str, unique_before.tolist())))
    print()

    for area_m2 in args.areas:
        pixel_count = max(
            1,
            int(math.ceil(area_m2 / ground_pixel_area)),
        )

        cleaned = sieve(
            class_map,
            size=pixel_count,
            connectivity=args.connectivity,
        ).astype(class_map.dtype)

        changed_pixels = int(np.count_nonzero(cleaned != class_map))
        changed_area_m2 = changed_pixels * ground_pixel_area
        changed_pct = 100.0 * changed_pixels / class_map.size

        output_path = build_output_path(
            class_path,
            output_dir,
            area_m2,
        )

        out_profile = profile.copy()
        out_profile.update(
            count=1,
            dtype=str(cleaned.dtype),
            compress="deflate",
        )

        with rasterio.open(output_path, "w", **out_profile) as dst:
            dst.write(cleaned, 1)
            dst.set_band_description(
                1,
                f"Land-cover class ID after {area_m2:g} m2 sieve",
            )

            if colormap is not None:
                dst.write_colormap(1, colormap)

            dst.update_tags(
                **tags,
                postprocess="rasterio.features.sieve",
                sieve_area_m2=f"{area_m2:g}",
                sieve_pixel_count=str(pixel_count),
                sieve_connectivity=str(args.connectivity),
                estimated_ground_pixel_area_m2=(
                    f"{ground_pixel_area:.8f}"
                ),
                confidence_note=(
                    "No confidence raster is modified. "
                    "Original confidence refers to raw model prediction."
                ),
            )

        print(
            f"{area_m2:g} m2 -> "
            f"{pixel_count} connected pixels | "
            f"changed {changed_pixels} pixels "
            f"({changed_pct:.3f}%, "
            f"~{changed_area_m2:.1f} m2)"
        )
        print("  Output:", output_path)

    print()
    print("-------------------------------------")
    print("SIEVE OK")
    print()
    print(
        "Important: the original confidence raster is NOT a confidence "
        "measure for pixels whose class was changed by sieve."
    )
    print(
        "For structural comparison, polygonize the sieve outputs without "
        "--confidence."
    )


if __name__ == "__main__":
    main()
