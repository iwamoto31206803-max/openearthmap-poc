"""SACLAJ Evaluation v0.1: paired center-point inference on GSI latest photos."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import sys
from urllib.error import HTTPError, URLError
import uuid

import numpy as np
from PIL import Image, UnidentifiedImageError
import torch

from src import model as shared_model
from src.config import GSI_SOURCE_NAME_JP, GSI_TILE_SIZE, GSI_TILE_URL, repo_dir
from src.download_gsi_geotiff import download_tile, lonlat_to_tile
from src.evaluation.saclaj import (
    JAPAN_FILTER, LIMITATIONS, ValidationError, aggregate, load_mapping, read_csv, select_sites, sha256,
)
from src.training.train_gsi_paddy import resolve_device, seed_everything, write_json

PATCH_SIZE = 512
CENTER = PATCH_SIZE // 2
SITE_FIELDS = [
    "ID", "Latitude", "Longitude", "Date", "Diameter(m)", "Category_ID", "Category_detail",
    "subtype", "expected_class", "status", "reason", "patch_file", "patch_sha256", "acquired_at_utc",
] + [f"{model}_{field}" for model in ("base", "fine_tuned") for field in (
    "predicted_class", "expected_probability", "agriculture_probability",
)]


class AcquisitionFailure(Exception):
    def __init__(self, status: str, reason: str):
        self.status, self.reason = status, reason
        super().__init__(reason)  # No URL, coordinates, or raw exception text.


def acquire_patch(site, zoom=18):
    """Reuse GSI XYZ conversion/downloader; no resize, fill, or temporal search.

    The pixel containing the WGS84 point is floor(global XYZ pixel position).
    Crop so that pixel is at [256, 256]. All intersecting tiles must be present.
    """
    tile_x, tile_y = lonlat_to_tile(site.longitude, site.latitude, zoom)
    pixel_x, pixel_y = math.floor(tile_x * GSI_TILE_SIZE), math.floor(tile_y * GSI_TILE_SIZE)
    left, top = pixel_x - CENTER, pixel_y - CENTER
    right, bottom = left + PATCH_SIZE, top + PATCH_SIZE
    patch = np.empty((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
    for y in range(top // GSI_TILE_SIZE, (bottom - 1) // GSI_TILE_SIZE + 1):
        for x in range(left // GSI_TILE_SIZE, (right - 1) // GSI_TILE_SIZE + 1):
            try:
                tile = download_tile(zoom, x, y)
                if tile.size != (GSI_TILE_SIZE, GSI_TILE_SIZE):
                    raise AcquisitionFailure("error", "invalid_tile_dimensions")
                pixels = np.asarray(tile, dtype=np.uint8)
                if pixels.shape != (GSI_TILE_SIZE, GSI_TILE_SIZE, 3):
                    raise AcquisitionFailure("error", "invalid_tile_channels")
            except HTTPError as exc:
                if exc.code in (404, 410):
                    raise AcquisitionFailure("skipped", "imagery_unavailable") from None
                raise AcquisitionFailure("error", "http_error") from None
            except (URLError, TimeoutError, ConnectionError):
                raise AcquisitionFailure("error", "network_failure") from None
            except (UnidentifiedImageError, OSError, ValueError):
                raise AcquisitionFailure("error", "invalid_image_or_io") from None
            west, north = max(left, x * GSI_TILE_SIZE), max(top, y * GSI_TILE_SIZE)
            east, south = min(right, (x + 1) * GSI_TILE_SIZE), min(bottom, (y + 1) * GSI_TILE_SIZE)
            patch[north - top:south - top, west - left:east - left] = pixels[
                north - y * GSI_TILE_SIZE:south - y * GSI_TILE_SIZE,
                west - x * GSI_TILE_SIZE:east - x * GSI_TILE_SIZE,
            ]
    return patch


def predict_pair(patch, base, fine_tuned, expected_class, device):
    if patch.shape != (PATCH_SIZE, PATCH_SIZE, 3) or patch.dtype != np.uint8:
        raise ValueError("Expected a 512x512 uint8 RGB patch")
    # One common transform. Clones isolate any accidental in-place model operation.
    tensor = shared_model.rgb_to_tensor(patch).unsqueeze(0).to(device)
    predictions = {}
    with torch.inference_mode():
        for name, model in (("base", base), ("fine_tuned", fine_tuned)):
            model.eval()
            logits = model(tensor.clone())
            if logits.shape != (1, 9, PATCH_SIZE, PATCH_SIZE):
                raise ValueError("Model must return NCHW logits with 9 classes at native resolution")
            point = logits[0, :, CENTER, CENTER]
            if not torch.isfinite(point).all():
                raise ValueError("Non-finite point logits")
            probabilities = point.softmax(0).cpu()
            predictions[name] = {
                "predicted_class": int(probabilities.argmax()),
                "expected_probability": float(probabilities[expected_class]),
                "agriculture_probability": float(probabilities[7]),
            }
    return predictions


def require_external(path: Path):
    """Resolve symlinks/junctions and reject repo-contained sensitive input/output."""
    resolved = path.resolve()
    if resolved == repo_dir().resolve() or repo_dir().resolve() in resolved.parents:
        raise ValidationError("SACLAJ CSV and result directories must be outside this repository")
    if any((parent / ".git").exists() for parent in (resolved, *resolved.parents)):
        raise ValidationError("SACLAJ CSV and result directories must be outside every Git checkout")
    return resolved


def make_manifest(args, counts, mapping, run_id, timestamp, hashes, device, diameter_column):
    return {
        "schema_version": "saclaj-evaluation-0.1", "run_id": run_id,
        "timestamp_utc": timestamp, "status": "running",
        **hashes, "category_mapping_version": mapping["mapping_version"],
        "category_definition_sha256": mapping["definition_sha256"],
        "category_definition_verified": mapping["definition_verified"],
        "japan_filter": JAPAN_FILTER, "sampling_seed": args.seed,
        "max_samples_per_category": args.max_samples_per_category,
        "sampling_method": "per subtype, SHA256 of compact UTF-8 JSON [seed, subtype, ID], ascending; ID tie-break",
        "counts": counts, "csv_encoding": args.csv_encoding,
        "csv_resolved_columns": {"diameter": diameter_column},
        "csv_specification": "comma-delimited, exact headers, WGS84 decimal degrees; Date retained without parsing",
        "gsi_imagery_source": GSI_SOURCE_NAME_JP,
        "acquisition": {
            "url_template": GSI_TILE_URL, "zoom": args.zoom, "xyz_tile_size": GSI_TILE_SIZE,
            "patch_shape": [PATCH_SIZE, PATCH_SIZE, 3], "patch_format": "lossless RGB PNG",
            "projection": "EPSG:3857 XYZ grid from WGS84 longitude/latitude",
            "point_pixel": [CENTER, CENTER],
            "alignment": "floor global XYZ pixel; crop from point_pixel minus 256; no resampling",
            "missing_tile_policy": "skip 404/410; other network/decode failures are errors; no filling or replacement sampling",
            "timeout_seconds_per_tile": 30, "retries": 0, "temporal_matching": False,
        },
        "architecture": {"library": "segmentation_models_pytorch", "model": "Unet", **shared_model.MODEL_KWARGS},
        "preprocessing": shared_model.PREPROCESSING,
        "point_prediction": "argmax and softmax of 9 center-pixel logits; no overlap averaging/sieve/neighborhood",
        "paired_input": "same saved patch, transformed once through src.model.rgb_to_tensor",
        "device": device, "num_threads": args.num_threads, "deterministic_algorithms": True,
        "versions": {"python": platform.python_version(), "torch": str(torch.__version__),
                     "numpy": np.__version__, "Pillow": version("Pillow"),
                     "segmentation_models_pytorch": version("segmentation-models-pytorch")},
        "outcomes": {"success": 0, "skipped": 0, "error": 0},
        "output_files": ["evaluation_manifest.json", "aggregate_summary.json", "site_results.csv", "patches/"],
        "known_limitations": list(LIMITATIONS),
    }


def run(args):
    if not 14 <= args.zoom <= 18 or args.num_threads < 1:
        raise ValidationError("zoom must be 14..18 and num-threads positive")
    csv_path = require_external(args.saclaj_csv)
    output = require_external(args.output_dir)
    mapping = load_mapping(args.mapping)
    csv_data = read_csv(csv_path, args.csv_encoding)
    references, counts = select_sites(csv_data.sites, mapping, args.max_samples_per_category, args.seed)
    if not counts["sampled_by_subtype"]["rice_paddy"]:
        raise ValidationError("No eligible rice_paddy points; inspect the official mapping and CSV locally")
    if not any(ref.expected_class != 7 for ref in references):
        raise ValidationError("No non-Agriculture points; agriculture leakage cannot be evaluated")
    hashes = {
        "base_checkpoint_sha256": sha256(args.base_model),
        "fine_tuned_checkpoint_sha256": sha256(args.fine_tuned_model),
        "saclaj_csv_sha256": sha256(csv_path), "category_mapping_sha256": sha256(args.mapping),
    }
    seed_everything(args.seed)
    torch.set_num_threads(args.num_threads)
    device = resolve_device(args.device)
    base = shared_model.build_model(args.base_model, device=device)
    fine_tuned = shared_model.build_model(args.fine_tuned_model, device=device)
    # Both strict loaders and native-resolution forward passes checked offline.
    predict_pair(np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8), base, fine_tuned, 7, device)
    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid.uuid4().hex[:8]
    run_dir = output / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / ".gitignore").write_text("*\n", encoding="utf-8")
    manifest = make_manifest(args, counts, mapping, run_id, timestamp.isoformat(), hashes, device,
                             csv_data.diameter_column)
    manifest_path = run_dir / "evaluation_manifest.json"
    if args.preflight:
        manifest.update(status="preflight_passed", output_files=["evaluation_manifest.json"],
                        preflight="Schema, mapping, sampling, hashes, both strict model loads and zero-RGB forward passes; no network or site predictions")
        write_json(manifest_path, manifest)
        print(f"Preflight passed: sampled={len(references)}; network not tested. Run ID: {run_id}")
        return run_dir
    write_json(manifest_path, manifest)
    (run_dir / "patches").mkdir()
    results = []
    try:
        with (run_dir / "site_results.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=SITE_FIELDS)
            writer.writeheader()
            for index, ref in enumerate(references, start=1):
                site = ref.site
                row = {"ID": site.site_id, "Latitude": site.latitude, "Longitude": site.longitude,
                       "Date": site.date, "Diameter(m)": site.diameter, "Category_ID": site.category_id,
                       "Category_detail": site.detail, "subtype": ref.subtype,
                       "expected_class": ref.expected_class, "status": "error", "reason": ""}
                try:
                    patch = acquire_patch(site, args.zoom)
                except AcquisitionFailure as exc:
                    row.update(status=exc.status, reason=exc.reason)
                else:
                    patch_name = f"patches/{index:06d}.png"
                    Image.fromarray(patch).save(run_dir / patch_name)
                    row.update(patch_file=patch_name, patch_sha256=sha256(run_dir / patch_name),
                               acquired_at_utc=datetime.now(timezone.utc).isoformat())
                    try:
                        row.update(predict_pair(patch, base, fine_tuned, ref.expected_class, device))
                        row["status"] = "success"
                    except Exception:
                        row["reason"] = "prediction_failure"
                        # Systematic model/device failures stop rather than silently dropping a subset.
                        results.append(row)
                        writer.writerow({key: row.get(key, "") for key in SITE_FIELDS})
                        stream.flush()
                        raise
                results.append(row)
                flat = {key: row.get(key, "") for key in SITE_FIELDS}
                for model in ("base", "fine_tuned"):
                    for key, value in row.get(model, {}).items():
                        flat[f"{model}_{key}"] = value
                writer.writerow(flat)
                stream.flush()
                manifest["outcomes"][row["status"]] += 1
                manifest["processed"] = index
                write_json(manifest_path, manifest)
                print(f"Processed {index}/{len(references)}; success={manifest['outcomes']['success']}, "
                      f"skipped={manifest['outcomes']['skipped']}, error={manifest['outcomes']['error']}", flush=True)
        summary = aggregate(results, counts)
        primary_available = (summary["rice_agreement_percent"]["n"] > 0
                             and summary["agriculture_leakage_percent"]["n"] > 0)
        manifest["status"] = "completed" if primary_available else "insufficient_pairs"
    except BaseException as exc:
        manifest.update(status="failed", failure_type=type(exc).__name__)
        raise
    finally:
        summary = aggregate(results, counts)
        write_json(run_dir / "aggregate_summary.json", summary)
        manifest.update(outcomes=summary["outcomes"], processed=len(results),
                        unprocessed=len(references) - len(results),
                        finished_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(manifest_path, manifest)
    print(f"Evaluation {manifest['status']}. Run ID: {run_id}")
    return run_dir


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saclaj-csv", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True,
                        help="Reviewed explicit mapping JSON transcribed from official SACLAJ definitions")
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--fine-tuned-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--csv-encoding", default="utf-8-sig")
    parser.add_argument("--max-samples-per-category", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--zoom", type=int, default=18)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--preflight", action="store_true", help="Offline schema, mapping and checkpoint checks only")
    return parser


def main():
    args = make_parser().parse_args()
    try:
        directory = run(args)
        status = json.loads((directory / "evaluation_manifest.json").read_text(encoding="utf-8"))["status"]
        return 0 if status in {"preflight_passed", "completed"} else 2
    except ValidationError as exc:
        print(f"Preflight/evaluation validation failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        # Do not leak file paths, input values, URL tile coordinates, or site IDs.
        print(f"Evaluation stopped ({type(exc).__name__}). Check schema/mapping/files locally; "
              "if a run was created, inspect its local manifest.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
