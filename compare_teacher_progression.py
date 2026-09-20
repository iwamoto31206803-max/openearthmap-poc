"""Diagnose OEM8 prediction changes as teachers are added stage by stage."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio

from compare_base_ft import run_command, sha256_file, software_versions, utc_now
from src.config import CLASS_NAMES, DEFAULT_OVERLAP, DEFAULT_SIEVE_AREA_M2, INFERENCE_TILE_SIZE
from src.model import PREPROCESSING
from src.qgis_styles import write_class_raster_style, write_transition_style

STAGES = (
    ("base", "00_base", "base_model"),
    ("paddy", "01_paddy", "paddy_model"),
    ("paddy_water", "02_paddy_water", "water_model"),
    ("paddy_water_road", "03_paddy_water_road", "road_model"),
)
WATCHED_TRANSITIONS = ((3, 4), (4, 3), (8, 5), (5, 8), (8, 4),
                       (5, 4), (2, 4), (7, 4), (6, 5), (7, 5))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare the fixed Base -> Paddy -> Paddy+Water -> Paddy+Water+Road progression."
    )
    parser.add_argument("--input", type=Path, required=True, help="Existing GSI RGB GeoTIFF.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--paddy-model", type=Path, required=True)
    parser.add_argument("--water-model", type=Path, required=True,
                        help="Paddy + Water model.")
    parser.add_argument("--road-model", type=Path, required=True,
                        help="Paddy + Water + Road model.")
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument("--sieve-area", type=float, default=DEFAULT_SIEVE_AREA_M2)
    parser.add_argument("--output-dir", type=Path, default=Path("teacher_progression"))
    parser.add_argument("--focus-transition", action="append", type=int, default=[])
    parser.add_argument("--polygonize", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.input.is_file():
        parser.error(f"input file not found: {args.input}")
    for _, _, attribute in STAGES:
        path = getattr(args, attribute)
        if not path.is_file():
            parser.error(f"model file not found: {path}")
    if not args.name or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in args.name):
        parser.error("--name may contain only letters, digits, '-' and '_'")
    if not 0 <= args.overlap < INFERENCE_TILE_SIZE:
        parser.error(f"--overlap must be between 0 and {INFERENCE_TILE_SIZE - 1}")
    if args.sieve_area <= 0:
        parser.error("--sieve-area must be greater than zero")
    if any(code < 1 or code > 88 or code // 10 > 8 or code % 10 > 8 or code // 10 == code % 10
           for code in args.focus_transition):
        parser.error("--focus-transition must encode two different OEM8 classes (for example 85)")


def validate_classes(values: np.ndarray, label: str) -> None:
    if values.size == 0 or np.any(values < 0) or np.any(values > 8):
        raise ValueError(f"{label} contains invalid class ID outside 0..8")


def aligned_stage_rasters(paths: Sequence[Path]) -> tuple[list[np.ndarray], dict]:
    arrays, reference = [], None
    for path in paths:
        with rasterio.open(path) as src:
            if src.count != 1:
                raise ValueError(f"stage class raster must have one band: {path}")
            signature = (src.width, src.height, src.crs, src.transform)
            if reference is None:
                reference = signature
            elif signature != reference:
                labels = ("width/height", "width/height", "CRS", "transform")
                mismatch = [labels[i] for i, (left, right) in enumerate(zip(reference, signature)) if left != right]
                raise ValueError("stage raster alignment mismatch: " + ", ".join(dict.fromkeys(mismatch)))
            values = src.read(1)
            validate_classes(values, str(path))
            arrays.append(values)
            profile = src.profile.copy()
    return arrays, profile


def validate_output_alignment(input_path: Path, output_path: Path) -> None:
    """Require a prediction to preserve the shared input grid exactly."""
    with rasterio.open(input_path) as input_raster, rasterio.open(output_path) as output:
        properties = ("width", "height", "crs", "transform")
        mismatches = [name for name in properties
                      if getattr(input_raster, name) != getattr(output, name)]
        if mismatches:
            raise ValueError(f"stage output alignment mismatch ({output_path}): " + ", ".join(mismatches))
        if output.count != 1:
            raise ValueError(f"stage output must have exactly one band: {output_path}")


def transition_code(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if source.shape != target.shape:
        raise ValueError("source and target arrays must have identical shapes")
    validate_classes(source, "source")
    validate_classes(target, "target")
    return source.astype(np.uint16) * 10 + target.astype(np.uint16)


def changed_only(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    codes = transition_code(source, target)
    return np.where(source == target, 0, codes).astype(np.uint16)


def class_distribution(values: np.ndarray) -> dict:
    total = int(values.size)
    return {str(i): {"name": CLASS_NAMES[i], "pixel_count": int(np.count_nonzero(values == i)),
                     "percent": 100.0 * int(np.count_nonzero(values == i)) / total}
            for i in range(9)}


def comparison_summary(source: np.ndarray, target: np.ndarray) -> dict:
    codes = transition_code(source, target)
    total = int(source.size)
    matrix = np.zeros((9, 9), dtype=np.int64)
    np.add.at(matrix, (source.astype(int), target.astype(int)), 1)
    transitions = []
    changed = int(np.count_nonzero(source != target))
    for source_id in range(9):
        for target_id in range(9):
            if source_id != target_id and matrix[source_id, target_id]:
                count = int(matrix[source_id, target_id])
                transitions.append({"code": source_id * 10 + target_id,
                                    "source_class": source_id, "target_class": target_id,
                                    "source_name": CLASS_NAMES[source_id], "target_name": CLASS_NAMES[target_id],
                                    "pixel_count": count, "percent": 100.0 * count / total})
    by_count = sorted(transitions, key=lambda item: (-item["pixel_count"], item["code"]))[:20]
    by_percent = sorted(transitions, key=lambda item: (-item["percent"], item["code"]))[:20]
    source_dist, target_dist = class_distribution(source), class_distribution(target)
    drift = {str(i): {"name": CLASS_NAMES[i],
                      "source_pixel_count": source_dist[str(i)]["pixel_count"],
                      "target_pixel_count": target_dist[str(i)]["pixel_count"],
                      "delta_pixels": target_dist[str(i)]["pixel_count"] - source_dist[str(i)]["pixel_count"],
                      "delta_percentage_points": target_dist[str(i)]["percent"] - source_dist[str(i)]["percent"]}
             for i in range(9)}
    return {"total_pixel_count": total, "changed_pixel_count": changed,
            "changed_pixel_percent": 100.0 * changed / total,
            "source_class_distribution": source_dist, "target_class_distribution": target_dist,
            "transition_matrix": {str(i): {str(j): int(matrix[i, j]) for j in range(9)} for i in range(9)},
            "top_transitions_by_pixel_count": by_count, "top_transitions_by_percent": by_percent,
            "watched_transitions": {f"{a} -> {b}": {"code": a * 10 + b,
                                      "pixel_count": int(np.count_nonzero(codes == a * 10 + b)),
                                      "percent": 100.0 * int(np.count_nonzero(codes == a * 10 + b)) / total}
                                    for a, b in WATCHED_TRANSITIONS},
            "class_drift": drift}


def write_raster(path: Path, values: np.ndarray, profile: dict, description: str) -> None:
    output_profile = profile.copy()
    output_profile.update(count=1, dtype=str(values.dtype), compress="deflate", nodata=None)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **output_profile) as dst:
        dst.write(values, 1)
        dst.set_band_description(1, description)


def focus_filename(code: int) -> str:
    source, target = divmod(code, 10)
    slug = lambda value: "_".join(CLASS_NAMES[value].split(" /")[0].lower().split())
    return f"focus_{code}_{slug(source)}_to_{slug(target)}.tif"


def prepare_run_directory(run_dir: Path, overwrite: bool) -> None:
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(f"output directory already exists: {run_dir}; use --overwrite")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)


def process(args: argparse.Namespace, root: Path) -> dict:
    # Hash all external files before creating output so unreadable inputs fail cleanly.
    input_source = args.input.resolve()
    input_hash = sha256_file(input_source)
    models = [(label, dirname, Path(getattr(args, attr)).resolve(), sha256_file(Path(getattr(args, attr))))
              for label, dirname, attr in STAGES]
    run_dir = args.output_dir / args.name
    prepare_run_directory(run_dir, args.overwrite)
    started = utc_now()
    input_path = run_dir / "input" / "gsi_rgb.tif"
    input_path.parent.mkdir(parents=True)
    shutil.copy2(input_source, input_path)
    if sha256_file(input_path) != input_hash:
        raise OSError("copied input SHA256 does not match source")
    area = f"{args.sieve_area:g}".replace(".", "p")
    stage_outputs, transition_outputs = {}, {}
    manifest = {"status": "running", "run_timestamp": started, "start_timestamp": started,
                "end_timestamp": None, "site": args.name,
                "input_geotiff": {"source_path": str(input_source), "path": str(input_path), "sha256": input_hash},
                "model_stages": [{"index": index, "label": label, "meaning": label.replace("_", " + "),
                                  "path": str(model), "sha256": digest}
                                 for index, (label, _, model, digest) in enumerate(models)],
                "overlap": args.overlap, "sieve_area_m2": args.sieve_area, "sieve_connectivity": 8,
                "class_names": {str(k): v for k, v in CLASS_NAMES.items()}, "preprocessing": PREPROCESSING,
                "software_versions": software_versions(), "stage_output_paths": stage_outputs,
                "transition_output_paths": transition_outputs}
    manifest_path = run_dir / "manifest.json"
    save = lambda: manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    save()
    try:
        raw_paths, sieve_paths = [], []
        for label, dirname, model, _ in models:
            directory = run_dir / "stages" / dirname
            directory.mkdir(parents=True)
            # predict/sieve retain their established names, then normalize the public progression names.
            run_command([sys.executable, str(root / "src/predict_geotiff_tiled.py"), str(input_path),
                         "--model", str(model), "--overlap", str(args.overlap), "--output-dir", str(directory)])
            generated_raw = directory / "gsi_rgb_classes.tif"
            generated_confidence = directory / "gsi_rgb_confidence.tif"
            raw, confidence = directory / "classes_raw.tif", directory / "confidence.tif"
            generated_raw.replace(raw); generated_confidence.replace(confidence)
            run_command([sys.executable, str(root / "src/sieve_landcover.py"), str(raw), "--areas",
                         str(args.sieve_area), "--connectivity", "8", "--output-dir", str(directory)])
            generated_sieve = directory / f"classes_raw_sieve_{area}m2.tif"
            sieve = directory / f"classes_sieve_{area}m2.tif"
            generated_sieve.replace(sieve)
            for path in (raw, confidence, sieve):
                validate_output_alignment(input_path, path)
            for path in (raw, sieve): write_class_raster_style(path)
            if args.polygonize:
                gpkg = directory / "landcover_sieve.gpkg"
                run_command([sys.executable, str(root / "src/polygonize_landcover.py"), str(sieve), "--output", str(gpkg)])
            stage_outputs[label] = {"raw_classes": str(raw), "confidence": str(confidence),
                                    "sieve_classes": str(sieve), **({"sieve_gpkg": str(gpkg)} if args.polygonize else {})}
            raw_paths.append(raw); sieve_paths.append(sieve)
        raw_arrays, raw_profile = aligned_stage_rasters(raw_paths)
        sieve_arrays, sieve_profile = aligned_stage_rasters(sieve_paths)
        steps = []
        for index in range(3):
            source_label, target_label = models[index][0], models[index + 1][0]
            step_name = f"{index:02d}_{source_label}_to_{index + 1:02d}_{target_label}"
            directory = run_dir / "transitions" / step_name
            paths = {}
            summaries = {}
            for kind, arrays, profile in (("raw", raw_arrays, raw_profile), ("sieve", sieve_arrays, sieve_profile)):
                source, target = arrays[index], arrays[index + 1]
                transition = directory / f"transition_{kind}.tif"
                changed = directory / f"changed_only_{kind}.tif"
                write_raster(transition, transition_code(source, target), profile, "source class * 10 + target class")
                write_raster(changed, changed_only(source, target), profile, "changed pixels; 0 means unchanged")
                write_transition_style(changed)
                paths[f"transition_{kind}"] = str(transition); paths[f"changed_only_{kind}"] = str(changed)
                for code in args.focus_transition:
                    focused = np.where(changed_only(source, target) == code, code, 0).astype(np.uint16)
                    focus_path = directory / kind / focus_filename(code)
                    write_raster(focus_path, focused, profile, f"focused transition {code}; 0 means other")
                    write_transition_style(focus_path)
                    paths.setdefault("focus", {}).setdefault(kind, {})[str(code)] = str(focus_path)
                summaries[kind] = comparison_summary(source, target)
            summary_path = directory / "summary.json"
            summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            paths["summary"] = str(summary_path); transition_outputs[step_name] = paths
            steps.append({"step": step_name, "source_stage": source_label, "target_stage": target_label,
                          "raw": summaries["raw"], "sieve": summaries["sieve"]})
        progression = {"schema_version": 1, "site": args.name,
                       "stage_order": [model[0] for model in models], "steps": steps}
        (run_dir / "progression_summary.json").write_text(
            json.dumps(progression, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest["status"] = "complete"
    except Exception as exc:
        manifest["status"] = "failed"; manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["end_timestamp"] = utc_now(); save()
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser(); args = parser.parse_args(argv); validate_args(parser, args)
    try:
        process(args, Path(__file__).resolve().parent)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr); return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
