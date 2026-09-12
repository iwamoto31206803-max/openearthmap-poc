from __future__ import annotations

from pathlib import Path
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone


DEFAULT_MODEL = (
    Path(__file__).resolve().parent.parent
    / "OpenEarthMap-SAR"
    / "src"
    / "Semantic_Segemtation"
    / "pretrained"
    / "RGB_Real_5_u-efficientnet-b4.pth"
)


def run_command(command: list[str]) -> None:
    print()
    print("$", " ".join(str(x) for x in command))
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the current OpenEarthMap PoC end-to-end while preserving "
            "intermediate products in a structured run directory."
        )
    )
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--name", default="site")
    parser.add_argument("--zoom", type=int, default=18)
    parser.add_argument("--tiles", type=int, default=3)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--sieve-area", type=float, default=5.0)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument(
        "--polygonize-raw",
        action="store_true",
        help="Also save a raw GeoPackage with mean confidence before sieve.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(
        c if c.isalnum() or c in "-_" else "_" for c in args.name
    )
    run_dir = Path(args.runs_dir) / f"{timestamp}_{safe_name}"

    input_dir = run_dir / "01_input"
    prediction_dir = run_dir / "02_prediction"
    postprocess_dir = run_dir / "03_postprocess"
    vector_dir = run_dir / "04_vector"
    analysis_dir = run_dir / "05_analysis"
    metadata_dir = run_dir / "metadata"

    for directory in [
        input_dir,
        prediction_dir,
        postprocess_dir,
        vector_dir,
        analysis_dir,
        metadata_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    rgb_path = input_dir / "gsi_rgb.tif"

    run_command([
        sys.executable,
        str(root / "download_gsi_geotiff.py"),
        "--lat", str(args.lat),
        "--lon", str(args.lon),
        "--zoom", str(args.zoom),
        "--tiles", str(args.tiles),
        "--output", str(rgb_path),
    ])

    run_command([
        sys.executable,
        str(root / "predict_geotiff_tiled.py"),
        str(rgb_path),
        "--model", str(args.model),
        "--overlap", str(args.overlap),
        "--output-dir", str(prediction_dir),
    ])

    raw_classes = prediction_dir / "gsi_rgb_classes.tif"
    raw_confidence = prediction_dir / "gsi_rgb_confidence.tif"

    run_command([
        sys.executable,
        str(root / "sieve_landcover.py"),
        str(raw_classes),
        "--areas", str(args.sieve_area),
        "--output-dir", str(postprocess_dir),
    ])

    area_label = f"{args.sieve_area:g}".replace(".", "p")
    cleaned_classes = (
        postprocess_dir / f"gsi_rgb_classes_sieve_{area_label}m2.tif"
    )
    cleaned_gpkg = vector_dir / "landcover_sieve.gpkg"

    run_command([
        sys.executable,
        str(root / "polygonize_landcover.py"),
        str(cleaned_classes),
        "--output", str(cleaned_gpkg),
    ])

    raw_gpkg = None
    if args.polygonize_raw:
        raw_gpkg = vector_dir / "landcover_raw.gpkg"
        run_command([
            sys.executable,
            str(root / "polygonize_landcover.py"),
            str(raw_classes),
            "--confidence", str(raw_confidence),
            "--output", str(raw_gpkg),
        ])

    analysis_inputs = [str(cleaned_gpkg)]
    if raw_gpkg is not None:
        analysis_inputs.insert(0, str(raw_gpkg))

    run_command([
        sys.executable,
        str(root / "analyze_landcover_gpkg.py"),
        *analysis_inputs,
        "--output-dir", str(analysis_dir),
    ])

    manifest = {
        "poc_version": "0.1",
        "created_local": datetime.now().astimezone().isoformat(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "site": {
            "name": args.name,
            "center_lat": args.lat,
            "center_lon": args.lon,
            "zoom": args.zoom,
            "tile_grid": args.tiles,
        },
        "model": {
            "path": str(Path(args.model)),
            "expected_weight": "RGB_Real_5_u-efficientnet-b4.pth",
            "license_status": "under confirmation; PoC baseline only",
        },
        "processing": {
            "tile_size_px": 512,
            "overlap_px": args.overlap,
            "sieve_area_m2": args.sieve_area,
            "sieve_connectivity": 8,
        },
        "products": {
            "input_rgb_geotiff": str(rgb_path),
            "raw_class_geotiff": str(raw_classes),
            "raw_confidence_geotiff": str(raw_confidence),
            "cleaned_class_geotiff": str(cleaned_classes),
            "cleaned_landcover_gpkg": str(cleaned_gpkg),
            "raw_landcover_gpkg": str(raw_gpkg) if raw_gpkg else None,
            "analysis_dir": str(analysis_dir),
        },
        "notes": [
            "Confidence is maximum softmax probability, not empirical accuracy.",
            "Raw confidence is not reassigned to pixels changed by sieve.",
            "The pretrained model weight is currently used for PoC evaluation only.",
        ],
    }

    manifest_path = metadata_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("============================================================")
    print("POC RUN COMPLETE")
    print("Run directory :", run_dir)
    print("Final GPKG    :", cleaned_gpkg)
    print("Manifest      :", manifest_path)
    print("============================================================")


if __name__ == "__main__":
    main()
