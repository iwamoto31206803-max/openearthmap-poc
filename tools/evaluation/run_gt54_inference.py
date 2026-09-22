"""Run the fixed A/B/C/D checkpoints on the formal GT54 RGB inventory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import rasterio

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.evaluation.gt54_preflight import (  # noqa: E402
    inventory_dataset, load_manifest, validate_local_manifest,
)
from src.model import MODEL_KWARGS, PREPROCESSING, build_model  # noqa: E402
from src.predict_geotiff_tiled import predict_tiled  # noqa: E402


CHECKPOINT_SHA256 = {
    "A": "852cd4f27627a8b0b34fe35618fabafc85e1ff5025eadc259176ca4ecc23a81c",
    "B": "e536052223f2989ef382fd7d1bbfaa0d75662c0757c8362b574b7c28df4d0172",
    "C": "ff721d91959da847adee2d26639384f03bfa4ded5ea118e40ae16576260c4b6e",
    "D": "b4abe64d86e5b4350be2107d154a547f6b87b48e884ea87eac9e088cec201ed9",
}
TILE_SIZE = 512
OVERLAP = 128


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=REPOSITORY_ROOT / "manifests" / "val_gt_georef.csv")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    for model_id in CHECKPOINT_SHA256:
        parser.add_argument(f"--checkpoint-{model_id.lower()}", type=Path, required=True)
    parser.add_argument("--smoke-items", type=int,
                        help="Infer only the first N numeric ValAreas after full formal preflight.")
    parser.add_argument("--expected-items", type=int, default=54,
                        help="Expected manifest count (only for unit-test fixtures).")
    parser.add_argument("--expected-regions", type=int, default=8,
                        help="Expected region count (only for unit-test fixtures).")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _preflight_checkpoints(checkpoints: dict[str, Path]) -> None:
    errors = []
    for model_id, path in checkpoints.items():
        if not path.is_file():
            errors.append(f"checkpoint {model_id} not found: {path}")
        else:
            actual = _sha256(path)
            if actual != CHECKPOINT_SHA256[model_id]:
                errors.append(
                    f"checkpoint {model_id} SHA256 mismatch: expected "
                    f"{CHECKPOINT_SHA256[model_id]}, found {actual}"
                )
    if errors:
        raise ValueError("; ".join(errors))


def _write_prediction(path: Path, prediction: np.ndarray, metadata) -> None:
    profile = {
        "driver": "GTiff", "width": metadata.width, "height": metadata.height,
        "count": 1, "dtype": "uint8", "crs": metadata.crs,
        "transform": metadata.transform, "compress": "deflate",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as output:
        output.write(prediction, 1)
        output.set_band_description(1, "Land-cover class ID")


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {args.output_dir}; use --overwrite")
    if args.smoke_items is not None and args.smoke_items < 1:
        raise ValueError("--smoke-items must be at least 1")

    # Finish every dataset and checkpoint check before constructing a model or
    # invoking the predictor. This makes malformed inputs fail closed globally.
    items = load_manifest(args.manifest, expected_items=args.expected_items,
                          expected_regions=args.expected_regions)
    validate_local_manifest(args.dataset_root / "manifest.csv", items)
    inventory = inventory_dataset(items, args.dataset_root)
    checkpoints = {model_id: getattr(args, f"checkpoint_{model_id.lower()}")
                   for model_id in CHECKPOINT_SHA256}
    _preflight_checkpoints(checkpoints)

    selected = inventory if args.smoke_items is None else inventory[:args.smoke_items]
    if args.smoke_items is not None and args.smoke_items > len(inventory):
        raise ValueError("--smoke-items cannot exceed the formal inventory size")
    selected_ids = [entry.item.valarea for entry in selected]
    smoke = args.smoke_items is not None
    config = {
        "run_kind": "smoke_subset" if smoke else "formal_full",
        "smoke_subset": smoke,
        "smoke_items": args.smoke_items,
        "formal_inventory_items": len(inventory),
        "inference_items": len(selected),
        "selected_valareas": selected_ids,
        "models": list(CHECKPOINT_SHA256),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "device": "cpu", "tile_size": TILE_SIZE, "overlap": OVERLAP,
        "merge": "mean_logits", "prediction": "argmax",
        "preprocessing": PREPROCESSING, "model_kwargs": MODEL_KWARGS,
        "prediction_nodata": None, "resampling": None, "reprojection": None,
    }

    staging = args.output_dir.with_name(args.output_dir.name + ".tmp")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        (staging / "inference_config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with (staging / "inference_manifest.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("valarea", "region", "year", "subset_status"))
            for metadata in selected:
                writer.writerow((metadata.item.valarea, metadata.item.region,
                                 metadata.item.year, config["run_kind"]))
        with (staging / "inference_qc.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("check", "status", "value"))
            writer.writerow(("formal_inventory_preflight", "PASS", len(inventory)))
            writer.writerow(("all_checkpoint_sha256", "PASS", 4))
            writer.writerow(("subset_status", "SMOKE" if smoke else "FULL", config["run_kind"]))
            writer.writerow(("selected_valareas", "PASS", ";".join(selected_ids)))

        for model_id, checkpoint in checkpoints.items():
            model = build_model(checkpoint, device="cpu")
            for metadata in selected:
                with rasterio.open(metadata.rgb_path) as source:
                    # dtype was checked against the source bands during the full
                    # inventory preflight; intentionally do not cast here.
                    rgb = np.moveaxis(source.read([1, 2, 3]), 0, -1)
                prediction, _confidence = predict_tiled(rgb, model, OVERLAP)
                _write_prediction(staging / model_id / f"{metadata.item.valarea}.tif",
                                  prediction, metadata)
        if args.output_dir.exists():
            shutil.rmtree(args.output_dir)
        staging.rename(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return config


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = run(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"GT54 inference PASS: {config['run_kind']}; "
          f"{config['inference_items']} ValAreas x 4 checkpoints")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
