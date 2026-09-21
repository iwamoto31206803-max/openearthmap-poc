"""Build the year-matched GSI RGB / OEM-SAR validation-GT dataset.

The label raster is the authority for every aspect of the output grid.  This
module never reprojects or otherwise modifies label pixel values.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from PIL import Image
import rasterio
from rasterio.crs import CRS
from rasterio.warp import Resampling, reproject, transform_bounds

TILE_SIZE = 256
ORIGIN_SHIFT = math.pi * 6378137.0
DEFAULT_GT_ROOT = Path(r"C:\OpenEarthMap_PoC\oemsar_data\val_gt_georef")
DEFAULT_MANIFEST = Path(r"C:\OpenEarthMap_PoC\oemsar_data\manifests\val_gt_georef.csv")
DEFAULT_OUTPUT_ROOT = Path(r"C:\OpenEarthMap_PoC\oemsar_data\gsi_val_gt54")
PILOT_IDS = (
    "ValArea_011", "ValArea_008", "ValArea_016",
    "ValArea_038", "ValArea_075", "ValArea_110",
)

# GSI's published year-specific orthophoto layer identifiers. Deliberately no
# seamlessphoto fallback: a missing year/tile is a hard failure.
GSI_YEAR_SOURCES = {
    year: f"https://cyberjapandata.gsi.go.jp/xyz/nendophoto{year}/{{z}}/{{x}}/{{y}}.png"
    for year in (2007, 2017, 2018, 2019, 2020, 2021)
}


@dataclass(frozen=True)
class Item:
    valarea: str
    year: int
    region: str


def source_for_year(year: int) -> str:
    try:
        return GSI_YEAR_SOURCES[year]
    except KeyError as exc:
        raise ValueError(f"No verified GSI XYZ source configured for year {year}") from exc


def read_items(path: Path) -> list[Item]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"valarea", "year", "region"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(f"Manifest must contain columns: {', '.join(sorted(required))}")
            result = []
            for line, row in enumerate(reader, 2):
                try:
                    valarea, region = row["valarea"].strip(), row["region"].strip()
                    year = int(row["year"])
                except (AttributeError, TypeError, ValueError) as exc:
                    raise ValueError(f"Malformed manifest row {line}") from exc
                if not valarea.startswith("ValArea_") or not region:
                    raise ValueError(f"Malformed manifest row {line}")
                source_for_year(year)
                result.append(Item(valarea, year, region))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Manifest not found: {path}") from exc
    if not result:
        raise ValueError("Manifest contains no data rows")
    if len({item.valarea for item in result}) != len(result):
        raise ValueError("Manifest contains duplicate valarea values")
    return result


def _tile_xy(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    lat = max(min(lat, 85.05112878), -85.05112878)
    n = 2**zoom
    return ((lon + 180) / 360 * n,
            (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)


def _tile_bounds(x: int, y: int, zoom: int) -> tuple[float, float, float, float]:
    n = 2**zoom
    west = x / n * 2 * ORIGIN_SHIFT - ORIGIN_SHIFT
    east = (x + 1) / n * 2 * ORIGIN_SHIFT - ORIGIN_SHIFT
    north = ORIGIN_SHIFT - y / n * 2 * ORIGIN_SHIFT
    south = ORIGIN_SHIFT - (y + 1) / n * 2 * ORIGIN_SHIFT
    return west, south, east, north


def tiles_for_grid(crs: CRS, bounds, zoom: int) -> list[tuple[int, int]]:
    west, south, east, north = transform_bounds(crs, "EPSG:4326", *bounds, densify_pts=21)
    x0, y0 = _tile_xy(west, north, zoom)
    x1, y1 = _tile_xy(east, south, zoom)
    # nextafter keeps an extent exactly on a tile edge from selecting its neighbour.
    return [(x, y) for y in range(math.floor(y0), math.floor(np.nextafter(y1, -math.inf)) + 1)
            for x in range(math.floor(x0), math.floor(np.nextafter(x1, -math.inf)) + 1)]


def download_tile(url: str, timeout: float, retries: int) -> np.ndarray:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "openearthmap-poc/1.0"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}: {url}")
                image = Image.open(BytesIO(response.read())).convert("RGB")
            if image.size != (TILE_SIZE, TILE_SIZE):
                raise RuntimeError(f"Unexpected tile size {image.size}: {url}")
            return np.asarray(image, dtype=np.uint8)
        except (OSError, urllib.error.URLError, RuntimeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2**attempt, 4))
    raise RuntimeError(f"GSI tile download failed (no fallback): {url}: {last_error}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_pair(rgb_path: Path, label_path: Path) -> dict[str, str]:
    with rasterio.open(rgb_path) as rgb, rasterio.open(label_path) as label:
        if rgb.count != 3:
            raise ValueError(f"RGB band count is {rgb.count}, expected 3")
        if label.count != 1:
            raise ValueError(f"GT band count is {label.count}, expected 1")
        if rgb.width != label.width or rgb.height != label.height:
            raise ValueError("RGB/GT size mismatch")
        if rgb.crs != label.crs:
            raise ValueError("RGB/GT CRS mismatch")
        # Exact equality is expected because the label transform is written directly.
        if rgb.transform != label.transform or rgb.bounds != label.bounds:
            raise ValueError("RGB/GT pixel-grid mismatch")
        labels = label.read(1)
        values = np.unique(labels)
        if np.any((values < 0) | (values > 8)):
            raise ValueError(f"GT contains values outside OEM8 range 0-8: {values.tolist()}")
        pixels = rgb.read()
        if not np.any(pixels) or (rgb.nodata is not None and np.all(pixels == rgb.nodata)):
            raise ValueError("RGB is all-zero or all-nodata")
        return {
            "crs": rgb.crs.to_string(), "width": str(rgb.width), "height": str(rgb.height),
            "transform": repr(tuple(rgb.transform)), "bounds": repr(tuple(rgb.bounds)),
            "rgb_dtype": rgb.dtypes[0], "label_dtype": label.dtypes[0],
        }


def build_item(item: Item, gt_root: Path, output_root: Path, zoom: int = 18,
               timeout: float = 30, retries: int = 3, overwrite: bool = False,
               fetch: Callable[[str, float, int], np.ndarray] = download_tile) -> dict[str, str]:
    gt_source = gt_root / f"{item.valarea}_gt_georef.tif"
    if not gt_source.is_file():
        raise FileNotFoundError(f"GT not found: {gt_source}")
    rgb_path = output_root / "rgb_images" / f"{item.valarea}.tif"
    label_path = output_root / "labels" / f"{item.valarea}.tif"
    template = source_for_year(item.year)
    if not overwrite and rgb_path.exists() and label_path.exists():
        details = validate_pair(rgb_path, label_path)
        return _row(item, template, rgb_path, label_path, zoom, 0, details)

    rgb_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_rgb, tmp_label = rgb_path.with_suffix(".tif.partial"), label_path.with_suffix(".tif.partial")
    for path in (tmp_rgb, tmp_label):
        path.unlink(missing_ok=True)
    try:
        with rasterio.open(gt_source) as gt:
            if gt.count != 1 or gt.crs is None:
                raise ValueError(f"GT must be a georeferenced one-band raster: {gt_source}")
            tiles = tiles_for_grid(gt.crs, gt.bounds, zoom)
            xs, ys = zip(*tiles)
            mosaic = np.empty(((max(ys)-min(ys)+1)*TILE_SIZE,
                               (max(xs)-min(xs)+1)*TILE_SIZE, 3), dtype=np.uint8)
            for x, y in tiles:
                url = template.format(z=zoom, x=x, y=y)
                mosaic[(y-min(ys))*TILE_SIZE:(y-min(ys)+1)*TILE_SIZE,
                       (x-min(xs))*TILE_SIZE:(x-min(xs)+1)*TILE_SIZE] = fetch(url, timeout, retries)
            west, _, _, north = _tile_bounds(min(xs), min(ys), zoom)
            _, south, east, _ = _tile_bounds(max(xs), max(ys), zoom)
            src_transform = rasterio.transform.from_bounds(west, south, east, north,
                                                            mosaic.shape[1], mosaic.shape[0])
            output = np.zeros((3, gt.height, gt.width), dtype=np.uint8)
            for band in range(3):
                reproject(mosaic[:, :, band], output[band], src_transform=src_transform,
                          src_crs="EPSG:3857", dst_transform=gt.transform, dst_crs=gt.crs,
                          resampling=Resampling.bilinear)
            profile = gt.profile.copy()
            profile.update(driver="GTiff", count=3, dtype="uint8", nodata=None,
                           compress="deflate", photometric="RGB")
            with rasterio.open(tmp_rgb, "w", **profile) as dst:
                dst.write(output)
                dst.update_tags(gsi_year=str(item.year), gsi_xyz_template=template,
                                download_zoom=str(zoom))
        shutil.copyfile(gt_source, tmp_label)
        details = validate_pair(tmp_rgb, tmp_label)
        os.replace(tmp_rgb, rgb_path)
        os.replace(tmp_label, label_path)
        return _row(item, template, rgb_path, label_path, zoom, len(tiles), details)
    except Exception:
        tmp_rgb.unlink(missing_ok=True)
        tmp_label.unlink(missing_ok=True)
        raise


def _row(item, template, rgb_path, label_path, zoom, count, details):
    return {"valarea": item.valarea, "region": item.region, "gsi_year": str(item.year),
            "gsi_xyz_template": template, "rgb_path": str(rgb_path), "label_path": str(label_path),
            **details, "download_zoom": str(zoom), "source_tile_count": str(count),
            "rgb_sha256": sha256(rgb_path), "label_sha256": sha256(label_path),
            "alignment_ok": "True", "error": ""}


FIELDS = ("valarea", "region", "gsi_year", "gsi_xyz_template", "rgb_path", "label_path",
          "crs", "width", "height", "transform", "bounds", "rgb_dtype", "label_dtype",
          "download_zoom", "source_tile_count", "rgb_sha256", "label_sha256", "alignment_ok", "error")


def write_manifest(path: Path, rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.partial")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--ids", nargs="+")
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--zoom", type=int, default=18)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    items = read_items(args.manifest)
    if args.ids:
        known = {x.valarea: x for x in items}
        missing = sorted(set(args.ids) - known.keys())
        if missing:
            parser.error(f"IDs absent from manifest: {', '.join(missing)}")
        items = [known[x] for x in args.ids]
    rows, failed = [], 0
    for item in items:
        try:
            rows.append(build_item(item, args.gt_root, args.output_root, args.zoom,
                                   args.timeout, args.retries, args.overwrite))
            print(f"PASS {item.valarea}")
        except Exception as exc:
            failed += 1
            rows.append({**{field: "" for field in FIELDS}, "valarea": item.valarea,
                         "region": item.region, "gsi_year": str(item.year),
                         "gsi_xyz_template": source_for_year(item.year),
                         "alignment_ok": "False", "error": str(exc)})
            print(f"FAIL {item.valarea}: {exc}")
    write_manifest(args.output_root / "manifest.csv", rows)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
