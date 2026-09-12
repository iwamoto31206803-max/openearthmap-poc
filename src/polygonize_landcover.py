from pathlib import Path
import argparse
from contextlib import ExitStack

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import CRS, Geod, Transformer
from rasterio.features import geometry_mask, geometry_window, shapes
from shapely.geometry import mapping, shape
from shapely.ops import transform as shapely_transform


from config import CLASS_NAMES

GEOD = Geod(ellps="WGS84")


def geodesic_area_m2(geom, source_crs) -> float:
    """Return polygon area in square metres without relying on Web Mercator area."""
    crs = CRS.from_user_input(source_crs)

    if crs.to_epsg() == 4326:
        geom_wgs84 = geom
    else:
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        geom_wgs84 = shapely_transform(transformer.transform, geom)

    area, _ = GEOD.geometry_area_perimeter(geom_wgs84)
    return abs(float(area))


def mean_confidence_for_geometry(conf_src, class_src, geom) -> float:
    """Calculate mean confidence only inside one polygon's local raster window."""
    window = geometry_window(class_src, [mapping(geom)])
    confidence = conf_src.read(1, window=window)
    local_transform = class_src.window_transform(window)

    inside = geometry_mask(
        [mapping(geom)],
        out_shape=confidence.shape,
        transform=local_transform,
        invert=True,
    )

    values = confidence[inside]
    values = values[np.isfinite(values)]

    if values.size == 0:
        return float("nan")

    return float(values.mean())


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Polygonize an OpenEarthMap land-cover class GeoTIFF and write "
            "an editable GeoPackage for QGIS."
        )
    )
    parser.add_argument(
        "classes",
        help="Path to *_classes.tif produced by predict_geotiff_tiled.py.",
    )
    parser.add_argument(
        "--confidence",
        default=None,
        help="Optional matching *_confidence.tif.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output GeoPackage path. Default: <classes_stem>.gpkg",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=0.0,
        help=(
            "Drop polygons smaller than this geodesic area in m2. "
            "Default: 0 (keep everything for the first End-to-End test)."
        ),
    )
    parser.add_argument(
        "--include-background",
        action="store_true",
        help="Include class 0 Background / Unlabelled polygons.",
    )
    args = parser.parse_args()

    class_path = Path(args.classes)
    confidence_path = Path(args.confidence) if args.confidence else None
    output_path = (
        Path(args.output)
        if args.output
        else class_path.with_suffix(".gpkg")
    )

    if not class_path.exists():
        raise FileNotFoundError(f"Class GeoTIFF not found: {class_path}")
    if confidence_path is not None and not confidence_path.exists():
        raise FileNotFoundError(
            f"Confidence GeoTIFF not found: {confidence_path}"
        )
    if args.min_area < 0:
        raise ValueError("--min-area must be 0 or greater.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with ExitStack() as stack:
        class_src = stack.enter_context(rasterio.open(class_path))
        conf_src = (
            stack.enter_context(rasterio.open(confidence_path))
            if confidence_path is not None
            else None
        )

        if class_src.count != 1:
            raise ValueError("Class GeoTIFF must have exactly one band.")
        if class_src.crs is None:
            raise ValueError("Class GeoTIFF has no CRS.")

        if conf_src is not None:
            if conf_src.count != 1:
                raise ValueError("Confidence GeoTIFF must have exactly one band.")
            if (
                conf_src.width != class_src.width
                or conf_src.height != class_src.height
                or conf_src.crs != class_src.crs
                or conf_src.transform != class_src.transform
            ):
                raise ValueError(
                    "Class and confidence GeoTIFFs must have identical "
                    "size, CRS, and transform."
                )

        class_map = class_src.read(1)

        valid = np.isin(class_map, list(CLASS_NAMES.keys()))
        if not args.include_background:
            valid &= class_map != 0

        records = []
        polygon_no = 0

        print("Land-cover polygonization")
        print("-------------------------------------")
        print("Classes    :", class_path)
        print("Confidence :", confidence_path if confidence_path else "not used")
        print("CRS        :", class_src.crs)
        print("Min area   :", f"{args.min_area:.2f} m2")
        print()

        for geom_json, value in shapes(
            class_map,
            mask=valid,
            transform=class_src.transform,
            connectivity=8,
        ):
            class_id = int(value)
            geom = shape(geom_json)

            if geom.is_empty:
                continue
            if not geom.is_valid:
                geom = geom.buffer(0)
            if geom.is_empty:
                continue

            area_m2 = geodesic_area_m2(geom, class_src.crs)
            if area_m2 < args.min_area:
                continue

            polygon_no += 1
            mean_conf = (
                mean_confidence_for_geometry(conf_src, class_src, geom)
                if conf_src is not None
                else float("nan")
            )

            records.append(
                {
                    "polygon_id": polygon_no,
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    "area_m2": area_m2,
                    "mean_conf": mean_conf,
                    "geometry": geom,
                }
            )

        source_crs = class_src.crs

    if not records:
        raise RuntimeError(
            "No polygons remained after masking/filtering. "
            "Try --min-area 0 or --include-background."
        )

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=source_crs)
    gdf = gdf.sort_values(["class_id", "polygon_id"]).reset_index(drop=True)

    if output_path.exists():
        output_path.unlink()

    gdf.to_file(
        output_path,
        layer="landcover",
        driver="GPKG",
        index=False,
    )

    print("Polygon summary")
    print("-------------------------------------")
    for class_id in sorted(gdf["class_id"].unique()):
        subset = gdf[gdf["class_id"] == class_id]
        print(
            f"{class_id}: {CLASS_NAMES[class_id]:28s} "
            f"polygons={len(subset):5d}  "
            f"area={subset['area_m2'].sum():10.1f} m2"
        )

    print()
    print("-------------------------------------")
    print("POLYGONIZATION OK")
    print("GeoPackage :", output_path)
    print("Layer      : landcover")
    print("Polygons   :", len(gdf))
    print()
    print("Open the GeoPackage in QGIS and inspect/edit the polygons.")


if __name__ == "__main__":
    main()
