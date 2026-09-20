"""Compare two aligned OEM8 classification GeoTIFFs."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds, transform as window_transform

from src.config import CLASS_NAMES

MAJOR_TRANSITIONS = (
    (3, 4), (4, 3), (8, 5), (8, 4), (5, 4), (2, 4),
    (7, 4), (6, 5), (7, 5), (5, 8),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two aligned OEM8 classification GeoTIFFs.")
    parser.add_argument("source", type=Path, help="Earlier classification GeoTIFF.")
    parser.add_argument("target", type=Path, help="Later classification GeoTIFF.")
    parser.add_argument("--label", required=True, help="Comparison label, for example 02_to_03.")
    parser.add_argument("--output-dir", type=Path, default=Path("transition_diagnostics"))
    aoi = parser.add_mutually_exclusive_group()
    aoi.add_argument("--bbox", nargs=4, type=float, metavar=("LEFT", "BOTTOM", "RIGHT", "TOP"),
                     help="AOI in the raster CRS. Pixel-aligned bounds are required.")
    aoi.add_argument("--rows", nargs=2, type=int, metavar=("START", "STOP"),
                     help="Half-open row range; use together with --cols.")
    parser.add_argument("--cols", nargs=2, type=int, metavar=("START", "STOP"),
                        help="Half-open column range; use together with --rows.")
    parser.add_argument("--changed-geotiff", action="store_true",
                        help="Write changed_only.tif (from_id * 10 + to_id; 0 is unchanged).")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    for value in (args.source, args.target):
        if not value.is_file():
            parser.error(f"file not found: {value}")
    if not args.label or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.label):
        parser.error("--label may contain only letters, digits, '-' and '_'")
    if (args.rows is None) != (args.cols is None):
        parser.error("--rows and --cols must be specified together")


def _window(args: argparse.Namespace, src: rasterio.io.DatasetReader) -> Window:
    if args.bbox:
        left, bottom, right, top = args.bbox
        if left >= right or bottom >= top:
            raise ValueError("bbox must satisfy LEFT < RIGHT and BOTTOM < TOP")
        raw = from_bounds(left, bottom, right, top, src.transform)
        rounded = raw.round_offsets().round_lengths()
        if not np.allclose(tuple(raw), tuple(rounded), atol=1e-6):
            raise ValueError("bbox must follow raster pixel boundaries")
        window = rounded
    elif args.rows is not None:
        row_start, row_stop = args.rows
        col_start, col_stop = args.cols
        window = Window(col_start, row_start, col_stop - col_start, row_stop - row_start)
    else:
        window = Window(0, 0, src.width, src.height)
    if (window.col_off < 0 or window.row_off < 0 or window.width <= 0 or window.height <= 0
            or window.col_off + window.width > src.width or window.row_off + window.height > src.height):
        raise ValueError("AOI is empty or outside the raster extent")
    return window


def read_pair(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    with rasterio.open(args.source) as source, rasterio.open(args.target) as target:
        if source.count != 1 or target.count != 1:
            raise ValueError("classification rasters must have exactly one band")
        fields = ("width", "height", "crs", "transform")
        mismatch = [field for field in fields if getattr(source, field) != getattr(target, field)]
        if mismatch:
            raise ValueError("raster alignment mismatch: " + ", ".join(mismatch))
        window = _window(args, source)
        source_values = source.read(1, window=window)
        target_values = target.read(1, window=window)
        profile = source.profile.copy()
        profile.update(width=int(window.width), height=int(window.height),
                       transform=window_transform(window, source.transform))
        aoi = {"row_start": int(window.row_off), "row_stop": int(window.row_off + window.height),
               "col_start": int(window.col_off), "col_stop": int(window.col_off + window.width),
               "bounds": list(rasterio.windows.bounds(window, source.transform)),
               "crs": str(source.crs) if source.crs else None}
    for label, values in (("source", source_values), ("target", target_values)):
        if values.size == 0 or np.any(values < 0) or np.any(values > 8):
            raise ValueError(f"{label} contains class IDs outside OEM8 range 0..8")
    return source_values, target_values, profile, aoi


def summarize(source: np.ndarray, target: np.ndarray, label: str, aoi: dict) -> dict:
    matrix = np.zeros((9, 9), dtype=np.int64)
    np.add.at(matrix, (source.astype(int), target.astype(int)), 1)
    total = int(source.size)
    unchanged = int(np.trace(matrix))
    changed = total - unchanged

    def item(from_id: int, to_id: int) -> dict:
        count = int(matrix[from_id, to_id])
        return {"from_id": from_id, "from_name": CLASS_NAMES[from_id],
                "to_id": to_id, "to_name": CLASS_NAMES[to_id], "pixel_count": count,
                "percent_of_all": 100.0 * count / total,
                "percent_of_changed": (100.0 * count / changed if changed else 0.0),
                "changed": from_id != to_id}

    transitions = [item(i, j) for i in range(9) for j in range(9)]
    major = [item(i, j) for i, j in MAJOR_TRANSITIONS]
    background = [item(0, j) for j in range(9)]
    return {"schema_version": 1, "comparison_label": label, "class_names": {str(k): v for k, v in CLASS_NAMES.items()},
            "aoi": aoi, "total_pixel_count": total, "unchanged_pixel_count": unchanged,
            "unchanged_percent": 100.0 * unchanged / total, "changed_pixel_count": changed,
            "changed_percent": 100.0 * changed / total, "transitions": transitions,
            "changed_only_transitions": [row for row in transitions if row["changed"]],
            "nonzero_changed_transitions": [row for row in transitions if row["changed"] and row["pixel_count"] > 0],
            "major_transitions": major, "background_transitions": background}


def write_outputs(args: argparse.Namespace, summary: dict, source: np.ndarray,
                  target: np.ndarray, profile: dict) -> Path:
    output = args.output_dir / args.label
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {output}; use --overwrite")
    if output.exists() and args.overwrite:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output / "transition_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        columns = ("from_id", "from_name", "to_id", "to_name", "pixel_count",
                   "percent_of_all", "percent_of_changed", "changed")
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader(); writer.writerows(summary["transitions"])
    if args.changed_geotiff:
        changed = np.where(source == target, 0, source.astype(np.uint16) * 10 + target).astype(np.uint8)
        profile.update(count=1, dtype="uint8", nodata=None, compress="deflate")
        with rasterio.open(output / "changed_only.tif", "w", **profile) as dst:
            dst.write(changed, 1)
            dst.set_band_description(1, "changed: from class * 10 + to class; 0 means unchanged")
    return output


def print_summary(summary: dict) -> None:
    print(f"Comparison: {summary['comparison_label']}")
    print(f"Pixels: {summary['total_pixel_count']:,} | changed: {summary['changed_pixel_count']:,} "
          f"({summary['changed_percent']:.4f}%) | unchanged: {summary['unchanged_pixel_count']:,} "
          f"({summary['unchanged_percent']:.4f}%)")
    print("Major transitions:")
    for row in summary["major_transitions"]:
        print(f"  {row['from_name']} -> {row['to_name']}: {row['pixel_count']:,} "
              f"({row['percent_of_all']:.4f}% all, {row['percent_of_changed']:.4f}% changed)")
    print("Background -> *:")
    for row in summary["background_transitions"]:
        print(f"  Background/Unlabelled -> {row['to_name']}: {row['pixel_count']:,} ({row['percent_of_all']:.4f}%)")


def run(args: argparse.Namespace) -> dict:
    source, target, profile, aoi = read_pair(args)
    summary = summarize(source, target, args.label, aoi)
    summary["source_path"] = str(args.source.resolve())
    summary["target_path"] = str(args.target.resolve())
    write_outputs(args, summary, source, target, profile)
    print_summary(summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser(); args = parser.parse_args(argv); validate_args(parser, args)
    try:
        run(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
