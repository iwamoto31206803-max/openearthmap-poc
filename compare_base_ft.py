"""Reproducible, same-input comparison of OEM8 base and fine-tuned models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
import torch

from src.config import (
    CLASS_NAMES,
    DEFAULT_OVERLAP,
    DEFAULT_SIEVE_AREA_M2,
    INFERENCE_TILE_SIZE,
)
from src.model import PREPROCESSING
from src.qgis_styles import write_road_change_style


ROAD_CHANGE_NAMES = {
    0: "Other / not focused",
    1: "Pavement -> Road",
    2: "Road -> Road",
    3: "Road -> Pavement",
    4: "Other -> Road",
    5: "Road -> Other",
}
SITE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class Site:
    name: str
    lat: float
    lon: float


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_command(command: Sequence[str]) -> None:
    print("\n$", " ".join(map(str, command)))
    subprocess.run(list(command), check=True)


def validate_site(site: Site) -> Site:
    if not SITE_NAME_RE.fullmatch(site.name):
        raise ValueError("site name may contain only letters, digits, '-' and '_'")
    if not -90 <= site.lat <= 90:
        raise ValueError("latitude must be between -90 and 90")
    if not -180 <= site.lon <= 180:
        raise ValueError("longitude must be between -180 and 180")
    return site


def load_sites_csv(path: Path) -> list[Site]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["name", "lat", "lon"]:
            raise ValueError("--sites CSV header must be exactly: name,lat,lon")
        rows = list(reader)
    if not rows:
        raise ValueError("--sites CSV must contain at least one site")
    sites: list[Site] = []
    for line, row in enumerate(rows, start=2):
        try:
            site = validate_site(Site(row["name"], float(row["lat"]), float(row["lon"])))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid --sites CSV row {line}: {exc}") from exc
        sites.append(site)
    names = [site.name for site in sites]
    if len(names) != len(set(names)):
        raise ValueError("--sites CSV site names must be unique")
    return sites


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare base and fine-tuned OEM8 predictions on one shared GSI GeoTIFF."
    )
    parser.add_argument("--lat", type=float)
    parser.add_argument("--lon", type=float)
    parser.add_argument("--name")
    parser.add_argument("--sites", type=Path)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--fine-tuned-model", type=Path, required=True)
    parser.add_argument("--zoom", type=int, default=18)
    parser.add_argument("--tiles", type=int, default=3)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument("--sieve-area", type=float, default=DEFAULT_SIEVE_AREA_M2)
    parser.add_argument("--output-dir", type=Path, default=Path("comparison"))
    parser.add_argument("--polygonize-raw", action="store_true")
    parser.add_argument("--skip-vector", action="store_true")
    parser.add_argument("--keep-existing-input", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> list[Site]:
    single_values = (args.name, args.lat, args.lon)
    if args.sites is not None:
        if any(value is not None for value in single_values):
            parser.error("--sites is mutually exclusive with --name/--lat/--lon")
        sites = load_sites_csv(args.sites)
    else:
        if any(value is None for value in single_values):
            parser.error("single-site mode requires --name, --lat, and --lon")
        sites = [validate_site(Site(args.name, args.lat, args.lon))]
    if not 14 <= args.zoom <= 18:
        parser.error("--zoom must be between 14 and 18")
    if args.tiles < 1 or args.tiles % 2 == 0:
        parser.error("--tiles must be a positive odd integer")
    if not 0 <= args.overlap < INFERENCE_TILE_SIZE:
        parser.error(f"--overlap must be between 0 and {INFERENCE_TILE_SIZE - 1}")
    if args.sieve_area <= 0:
        parser.error("--sieve-area must be greater than zero")
    for model in (args.base_model, args.fine_tuned_model):
        if not model.is_file():
            parser.error(f"model file not found: {model}")
    return sites


def aligned_rasters(base_path: Path, ft_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    with rasterio.open(base_path) as base, rasterio.open(ft_path) as ft:
        properties = ("width", "height", "crs", "transform")
        mismatches = [key for key in properties if getattr(base, key) != getattr(ft, key)]
        if mismatches:
            raise ValueError("Base/FT raster alignment mismatch: " + ", ".join(mismatches))
        if base.count != 1 or ft.count != 1:
            raise ValueError("Base/FT class rasters must each contain exactly one band")
        return base.read(1), ft.read(1), base.profile.copy()


def transition_code(base: np.ndarray, ft: np.ndarray) -> np.ndarray:
    if base.shape != ft.shape:
        raise ValueError("Base/FT arrays must have identical shapes")
    return base.astype(np.uint16) * 10 + ft.astype(np.uint16)


def road_change_map(base: np.ndarray, ft: np.ndarray) -> np.ndarray:
    if base.shape != ft.shape:
        raise ValueError("Base/FT arrays must have identical shapes")
    result = np.zeros(base.shape, dtype=np.uint8)
    result[(base == 3) & (ft == 4)] = 1
    result[(base == 4) & (ft == 4)] = 2
    result[(base == 4) & (ft == 3)] = 3
    result[(base != 3) & (base != 4) & (ft == 4)] = 4
    result[(base == 4) & (ft != 3) & (ft != 4)] = 5
    return result


def comparison_summary(base: np.ndarray, ft: np.ndarray) -> dict:
    if base.shape != ft.shape:
        raise ValueError("Base/FT arrays must have identical shapes")
    total = int(base.size)
    changed = int(np.count_nonzero(base != ft))

    def distribution(values: np.ndarray) -> dict:
        return {
            str(class_id): {
                "name": CLASS_NAMES[class_id],
                "pixel_count": int(np.count_nonzero(values == class_id)),
                "percent": 100.0 * int(np.count_nonzero(values == class_id)) / total,
            }
            for class_id in range(9)
        }

    matrix = np.zeros((9, 9), dtype=np.int64)
    valid = (base >= 0) & (base <= 8) & (ft >= 0) & (ft <= 8)
    np.add.at(matrix, (base[valid].astype(int), ft[valid].astype(int)), 1)
    road = road_change_map(base, ft)
    focused_counts = {
        ROAD_CHANGE_NAMES[value]: int(np.count_nonzero(road == value))
        for value in range(1, 6)
    }
    return {
        "total_pixel_count": total,
        "changed_pixel_count": changed,
        "changed_pixel_percent": 100.0 * changed / total,
        "base_class_distribution": distribution(base),
        "fine_tuned_class_distribution": distribution(ft),
        "selected_transition_counts": {
            "3 -> 4": int(matrix[3, 4]), "4 -> 4": int(matrix[4, 4]),
            "4 -> 3": int(matrix[4, 3]), "7 -> 4": int(matrix[7, 4]),
            "6 -> 4": int(matrix[6, 4]), "5 -> 4": int(matrix[5, 4]),
            "8 -> 4": int(matrix[8, 4]),
            "all other -> 4": int(matrix[[0, 1, 2], 4].sum()),
            "4 -> all other": int(matrix[4, [0, 1, 2, 5, 6, 7, 8]].sum()),
        },
        "transition_matrix": {
            str(source): {str(target): int(matrix[source, target]) for target in range(9)}
            for source in range(9)
        },
        "road_focused_counts": focused_counts,
    }


def write_comparison_rasters(base_path: Path, ft_path: Path, transition_path: Path,
                             road_path: Path) -> dict:
    base, ft, profile = aligned_rasters(base_path, ft_path)
    transition_path.parent.mkdir(parents=True, exist_ok=True)
    transition_profile = profile.copy()
    transition_profile.update(count=1, dtype="uint16", compress="deflate")
    with rasterio.open(transition_path, "w", **transition_profile) as dst:
        dst.write(transition_code(base, ft), 1)
        dst.set_band_description(1, "Base class * 10 + fine-tuned class")
    road_profile = profile.copy()
    road_profile.update(count=1, dtype="uint8", compress="deflate")
    with rasterio.open(road_path, "w", **road_profile) as dst:
        dst.write(road_change_map(base, ft), 1)
        dst.set_band_description(1, "Road-focused Base to fine-tuned change")
        dst.update_tags(**{f"class_{key}": value for key, value in ROAD_CHANGE_NAMES.items()})
    write_road_change_style(road_path)
    return comparison_summary(base, ft)


def software_versions() -> dict:
    return {"python": platform.python_version(), "torch": torch.__version__,
            "rasterio": rasterio.__version__, "numpy": np.__version__}


def process_site(site: Site, args: argparse.Namespace, root: Path) -> dict:
    started = utc_now()
    run_dir = args.output_dir / site.name
    input_dir, base_dir = run_dir / "input", run_dir / "base"
    ft_dir, diff_dir = run_dir / "fine_tuned", run_dir / "diff"
    for directory in (input_dir, base_dir, ft_dir, diff_dir):
        directory.mkdir(parents=True, exist_ok=True)
    input_path = input_dir / "gsi_rgb.tif"
    area_label = f"{args.sieve_area:g}".replace(".", "p")
    raw_name, confidence_name = "gsi_rgb_classes.tif", "gsi_rgb_confidence.tif"
    sieve_name = f"gsi_rgb_classes_sieve_{area_label}m2.tif"
    output_paths = {
        "input": str(input_path),
        "base": {"raw_classes": str(base_dir / raw_name), "confidence": str(base_dir / confidence_name),
                 "sieve_classes": str(base_dir / sieve_name)},
        "fine_tuned": {"raw_classes": str(ft_dir / raw_name), "confidence": str(ft_dir / confidence_name),
                       "sieve_classes": str(ft_dir / sieve_name)},
        "diff": {name: str(diff_dir / name) for name in (
            "transition_raw.tif", "transition_sieve.tif", "road_change_raw.tif",
            "road_change_sieve.tif", "summary.json")},
    }
    manifest = {
        "status": "running", "run_timestamp": started, "start_timestamp": started,
        "end_timestamp": None, "name": site.name, "lat": site.lat, "lon": site.lon,
        "zoom": args.zoom, "tiles": args.tiles, "overlap": args.overlap,
        "sieve_area_m2": args.sieve_area, "sieve_connectivity": 8,
        "input_geotiff": {"path": str(input_path), "sha256": None},
        "base_model": {"path": str(args.base_model.resolve()), "sha256": sha256_file(args.base_model)},
        "fine_tuned_model": {"path": str(args.fine_tuned_model.resolve()), "sha256": sha256_file(args.fine_tuned_model)},
        "class_names": {str(key): value for key, value in CLASS_NAMES.items()},
        "preprocessing": PREPROCESSING,
        "pipeline_steps": ["single GSI download", "base inference", "fine-tuned inference",
                           "8-connectivity sieve", "optional polygonize", "diff rasters", "QGIS styles"],
        "output_paths": output_paths, "software": software_versions(),
    }
    manifest_path = run_dir / "manifest.json"

    def save_manifest() -> None:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    save_manifest()
    src = root / "src"
    try:
        if not (args.keep_existing_input and input_path.is_file()):
            run_command([sys.executable, str(src / "download_gsi_geotiff.py"), "--lat", str(site.lat),
                         "--lon", str(site.lon), "--zoom", str(args.zoom), "--tiles", str(args.tiles),
                         "--output", str(input_path)])
        manifest["input_geotiff"]["sha256"] = sha256_file(input_path)
        for directory, model in ((base_dir, args.base_model), (ft_dir, args.fine_tuned_model)):
            run_command([sys.executable, str(src / "predict_geotiff_tiled.py"), str(input_path),
                         "--model", str(model), "--overlap", str(args.overlap),
                         "--output-dir", str(directory)])
            run_command([sys.executable, str(src / "sieve_landcover.py"), str(directory / raw_name),
                         "--areas", str(args.sieve_area), "--connectivity", "8",
                         "--output-dir", str(directory)])
            if not args.skip_vector:
                run_command([sys.executable, str(src / "polygonize_landcover.py"),
                             str(directory / sieve_name), "--output", str(directory / "landcover_sieve.gpkg")])
                output_paths["base" if directory == base_dir else "fine_tuned"]["sieve_gpkg"] = str(directory / "landcover_sieve.gpkg")
                if args.polygonize_raw:
                    run_command([sys.executable, str(src / "polygonize_landcover.py"), str(directory / raw_name),
                                 "--confidence", str(directory / confidence_name),
                                 "--output", str(directory / "landcover_raw.gpkg")])
                    output_paths["base" if directory == base_dir else "fine_tuned"]["raw_gpkg"] = str(directory / "landcover_raw.gpkg")

        raw_summary = write_comparison_rasters(base_dir / raw_name, ft_dir / raw_name,
            diff_dir / "transition_raw.tif", diff_dir / "road_change_raw.tif")
        sieve_summary = write_comparison_rasters(base_dir / sieve_name, ft_dir / sieve_name,
            diff_dir / "transition_sieve.tif", diff_dir / "road_change_sieve.tif")
        (diff_dir / "summary.json").write_text(
            json.dumps({"raw": raw_summary, "sieve": sieve_summary}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        style_command = [sys.executable, str(src / "qgis_styles.py"), "--classes",
                         str(base_dir / raw_name), str(base_dir / sieve_name), str(ft_dir / raw_name),
                         str(ft_dir / sieve_name), "--confidence", str(base_dir / confidence_name),
                         str(ft_dir / confidence_name)]
        if not args.skip_vector:
            style_command += ["--vectors", str(base_dir / "landcover_sieve.gpkg"),
                              str(ft_dir / "landcover_sieve.gpkg")]
            if args.polygonize_raw:
                style_command += [str(base_dir / "landcover_raw.gpkg"), str(ft_dir / "landcover_raw.gpkg")]
        run_command(style_command)
        manifest["status"] = "complete"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["end_timestamp"] = utc_now()
        save_manifest()
    return {"name": site.name, "status": "complete", "manifest": str(manifest_path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    sites = validate_args(parser, args)
    root = Path(__file__).resolve().parent
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for site in sites:
        try:
            results.append(process_site(site, args, root))
        except Exception as exc:
            results.append({"name": site.name, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            print(f"ERROR: site {site.name} failed: {exc}", file=sys.stderr)
            if args.fail_fast or len(sites) == 1:
                break
    if args.sites is not None:
        (args.output_dir / "batch_summary.json").write_text(
            json.dumps({"created_utc": utc_now(), "sites": results}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
    return 1 if any(result["status"] == "failed" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
