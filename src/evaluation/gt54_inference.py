"""Fail-closed, fair A/B/C/D prediction generation for GT54 (Step 4A)."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Callable

import numpy as np
import rasterio
import torch

from src.evaluation.gt54_preflight import ManifestItem, RasterMetadata, inspect_item, load_manifest
from src.model import MODEL_KWARGS, PREPROCESSING, build_model
from src.predict_geotiff_tiled import DEVICE, TILE_SIZE, predict_tiled


MODEL_ORDER = ("A", "B", "C", "D")
EXPECTED_CHECKPOINT_SHA256 = {
    "A": "852cd4f27627a8b0b34fe35618fabafc85e1ff5025eadc259176ca4ecc23a81c",
    "B": "e536052223f2989ef382fd7d1bbfaa0d75662c0757c8362b574b7c28df4d0172",
    "C": "ff721d91959da847adee2d26639384f03bfa4ded5ea118e40ae16576260c4b6e",
    "D": "b4abe64d86e5b4350be2107d154a547f6b87b48e884ea87eac9e088cec201ed9",
}
CLASS_COUNT = 9
DEFAULT_OVERLAP = 128


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Checkpoint:
    model_id: str
    path: Path
    sha256: str
    expected_sha256: str


def preflight_checkpoints(paths: dict[str, Path], *,
                          expected: dict[str, str] = EXPECTED_CHECKPOINT_SHA256,
                          loader: Callable[[Path, str], torch.nn.Module] = build_model
                          ) -> tuple[Checkpoint, ...]:
    """Hash and strict-load every checkpoint before any inference is allowed."""
    if tuple(paths) != MODEL_ORDER:
        raise ValueError("checkpoint paths must be supplied in A, B, C, D order")
    checked = []
    for model_id in MODEL_ORDER:
        path = Path(paths[model_id]).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint {model_id} not found: {path}")
        actual = sha256_file(path)
        wanted = expected[model_id]
        if actual != wanted:
            raise ValueError(f"checkpoint {model_id} SHA256 mismatch: {actual} != {wanted}")
        model = loader(path, DEVICE)  # model.load_state_dict is strict by default
        del model
        checked.append(Checkpoint(model_id, path, actual, wanted))
    return tuple(checked)


def inference_config(overlap: int, implementation: str) -> dict[str, object]:
    if overlap < 0 or overlap >= TILE_SIZE:
        raise ValueError(f"overlap must be between 0 and {TILE_SIZE - 1}")
    return {
        "architecture": MODEL_KWARGS,
        "preprocessing": PREPROCESSING,
        "device": DEVICE,
        "tile_size": TILE_SIZE,
        "overlap": overlap,
        "stride": TILE_SIZE - overlap,
        "padding_mode": "edge",
        "tile_traversal": "y-major, then x-major; terminal window anchored to edge",
        "overlap_merge_rule": "float32 logit sum divided by float32 visit count",
        "argmax_rule": "numpy.argmax(axis=0), first class wins ties",
        "class_count": CLASS_COUNT,
        "output_dtype": "uint8",
        "prediction_nodata": None,
        "implementation": implementation,
        "determinism": (
            "model.eval() + torch.inference_mode() on CPU; no stochastic preprocessing "
            "or seed-dependent operation"
        ),
    }


def implementation_id(repository_root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository_root, check=True,
            capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def validate_prediction(path: Path, source: RasterMetadata) -> tuple[int, int]:
    with rasterio.open(path) as pred:
        if pred.count != 1 or pred.dtypes != ("uint8",):
            raise ValueError(f"{path}: prediction must be one-band uint8")
        if pred.nodata is not None:
            raise ValueError(f"{path}: prediction must not define NoData")
        fields = ("width", "height", "crs", "transform", "bounds")
        mismatch = [field for field in fields if getattr(pred, field) != getattr(source, field)]
        if mismatch:
            raise ValueError(f"{path}: prediction grid mismatch: {', '.join(mismatch)}")
        values = pred.read(1)
    minimum, maximum = int(values.min()), int(values.max())
    if minimum < 0 or maximum >= CLASS_COUNT:
        raise ValueError(f"{path}: prediction classes outside 0..8")
    return minimum, maximum


def _write_prediction(path: Path, values: np.ndarray, source: RasterMetadata) -> None:
    if values.shape != (source.height, source.width) or values.dtype != np.uint8:
        raise ValueError("inference returned an invalid class raster shape or dtype")
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", width=source.width, height=source.height,
                       count=1, dtype="uint8", crs=source.crs, transform=source.transform,
                       compress="deflate") as dst:
        dst.write(values, 1)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_cross_model_identity(rows: list[dict[str, object]]) -> None:
    """Validate that recorded input/grid evidence is identical for A/B/C/D."""
    identity_fields = ("rgb_sha256", "width", "height", "crs", "transform", "bounds")
    by_item: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_item.setdefault(str(row["valarea"]), []).append(row)
    for valarea, item_rows in by_item.items():
        if [row["model_id"] for row in item_rows] != list(MODEL_ORDER):
            raise RuntimeError(f"{valarea}: model order/inventory invariant failed")
        if any(len({row[field] for row in item_rows}) != 1 for field in identity_fields):
            raise RuntimeError(f"{valarea}: cross-model RGB/grid identity invariant failed")


def run_gt54_inference(*, manifest: Path, dataset_root: Path,
                       checkpoint_paths: dict[str, Path], output_root: Path,
                       overlap: int = DEFAULT_OVERLAP, overwrite: bool = False,
                       smoke_items: int | None = None,
                       expected_items: int = 54, expected_regions: int = 8,
                       expected_hashes: dict[str, str] = EXPECTED_CHECKPOINT_SHA256,
                       loader: Callable[[Path, str], torch.nn.Module] = build_model,
                       predictor: Callable = predict_tiled) -> list[dict[str, object]]:
    """Generate all predictions only after dataset and all-model preflight passes."""
    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is not empty: {output_root}; use --overwrite")
    items = load_manifest(Path(manifest), expected_items=expected_items,
                          expected_regions=expected_regions)
    inventory = [inspect_item(item, Path(dataset_root)) for item in items]
    checkpoints = preflight_checkpoints(checkpoint_paths, expected=expected_hashes, loader=loader)
    config = inference_config(overlap, implementation_id(Path(__file__).resolve().parents[2]))
    if smoke_items is not None and not 1 <= smoke_items <= len(inventory):
        raise ValueError(f"smoke_items must be between 1 and {len(inventory)}")
    selected_inventory = inventory if smoke_items is None else inventory[:smoke_items]
    selected_valareas = [source.item.valarea for source in selected_inventory]
    run_identity = {
        "run_mode": "formal_full" if smoke_items is None else "smoke_subset",
        "inventory_item_count": len(inventory),
        "selected_item_count": len(selected_inventory),
        "selected_valareas": selected_valareas,
    }

    # Destructive action occurs only after every RGB/GT and checkpoint passes.
    if output_root.exists() and overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    config_payload = {"schema_version": 1, "inference": config, "run": run_identity,
                      "models": [{**asdict(cp), "path": str(cp.path)} for cp in checkpoints]}
    (output_root / "inference_config.json").write_text(
        json.dumps(config_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []
    for cp in checkpoints:
        model = loader(cp.path, DEVICE)
        for source in selected_inventory:
            with rasterio.open(source.rgb_path) as rgb_src:
                rgb = np.moveaxis(rgb_src.read([1, 2, 3]), 0, -1)
            classes, _confidence = predictor(rgb, model, overlap)
            prediction_path = output_root / cp.model_id / "predictions" / f"{source.item.valarea}.tif"
            _write_prediction(prediction_path, classes, source)
            minimum, maximum = validate_prediction(prediction_path, source)
            rows.append({
                "model_id": cp.model_id, "valarea": source.item.valarea,
                "region": source.item.region, "rgb_path": str(source.rgb_path),
                "rgb_sha256": source.rgb_sha256, "prediction_path": str(prediction_path.resolve()),
                "prediction_sha256": sha256_file(prediction_path), "width": source.width,
                "height": source.height, "crs": str(source.crs),
                "transform": repr(source.transform), "bounds": repr(source.bounds),
                "prediction_class_min": minimum, "prediction_class_max": maximum, "status": "PASS",
                "run_mode": run_identity["run_mode"],
                "inventory_item_count": len(inventory),
                "selected_item_count": len(selected_inventory),
                "selected_valareas": json.dumps(selected_valareas),
            })
        del model
        model_rows.append({
            "model_id": cp.model_id, "checkpoint_path": str(cp.path),
            "checkpoint_sha256": cp.sha256, "expected_checkpoint_sha256": cp.expected_sha256,
            **{key: json.dumps(value, sort_keys=True) if isinstance(value, dict) else value
               for key, value in config.items()},
            "item_count": len(selected_inventory), "inventory_item_count": len(inventory),
            "selected_item_count": len(selected_inventory),
            "run_mode": run_identity["run_mode"],
            "selected_valareas": json.dumps(selected_valareas), "status": "PASS",
        })

    # The same ordered inventory/config is used by construction; verify the recorded evidence too.
    validate_cross_model_identity(rows)
    _write_csv(output_root / "inference_manifest.csv", rows)
    _write_csv(output_root / "inference_qc.csv", model_rows)
    return rows
