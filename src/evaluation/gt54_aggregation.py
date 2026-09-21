"""Ownership-aware Step-3 aggregation for one GT54 prediction set."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from src.config import CLASS_NAMES
from src.evaluation.gt54_metrics import (
    EvaluationResult, MetricResult, TileResult, confusion_from_arrays,
    load_raster_pair, metrics_from_confusion, write_evaluation_outputs,
)
from src.evaluation.gt54_preflight import ManifestItem


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class OwnedTileResult:
    item: ManifestItem
    metrics: MetricResult
    raw_metrics: MetricResult
    ownership_true_pixels: int
    hash_status: str


@dataclass(frozen=True)
class RegionResult:
    region: str
    metrics: MetricResult
    tile_count: int


@dataclass(frozen=True)
class EqualRegionClassMetric:
    class_id: int
    supported_region_count: int
    equal_region_iou: float | None
    gt_support_total: int


@dataclass(frozen=True)
class OwnershipEvaluationResult:
    model_id: str
    raw: EvaluationResult
    tiles: tuple[OwnedTileResult, ...]
    regions: tuple[RegionResult, ...]
    global_deduplicated: MetricResult
    equal_region_classes: tuple[EqualRegionClassMetric, ...]
    equal_region_macro_miou_8: float | None
    mean_region_miou_8: float | None


def _read_qc(ownership_dir: Path) -> dict[str, dict[str, str]] | None:
    path = ownership_dir / "ownership_qc.csv"
    if not path.exists():
        return None
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or "valarea" not in (rows[0].keys() if rows else ()):
        raise ValueError("ownership_qc.csv is empty or lacks valarea")
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        valarea = row["valarea"]
        if not valarea or valarea in indexed:
            raise ValueError(f"invalid duplicate/empty ownership QC valarea: {valarea!r}")
        indexed[valarea] = row
    return indexed


def load_ownership_mask(ownership_dir: Path, item: ManifestItem,
                        gt_shape: tuple[int, ...],
                        qc_rows: dict[str, dict[str, str]] | None = None,
                        ) -> tuple[np.ndarray, str]:
    """Load and fail-closed validate an existing Step-1 bool mask."""
    path = ownership_dir / "ownership_masks" / f"{item.valarea}.npy"
    if not path.is_file() or path.suffix != ".npy":
        raise FileNotFoundError(f"ownership mask not found: {path}")
    try:
        mask = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"{item.valarea}: invalid ownership .npy") from exc
    if mask.dtype != np.dtype(bool):
        raise ValueError(f"{item.valarea}: ownership mask dtype must be bool")
    if mask.shape != gt_shape:
        raise ValueError(f"{item.valarea}: ownership mask shape mismatch: {mask.shape} != {gt_shape}")
    if mask.size == 0:
        raise ValueError(f"{item.valarea}: ownership mask must be non-empty")
    true_count = int(np.count_nonzero(mask))
    if true_count > mask.size:  # explicit protocol invariant
        raise ValueError(f"{item.valarea}: impossible ownership True count")
    if qc_rows is None:
        return mask, "qc_not_available"
    if item.valarea not in qc_rows:
        raise ValueError(f"{item.valarea}: missing row in ownership_qc.csv")
    row = qc_rows[item.valarea]
    if not row.get("owned_pixel_count", ""):
        raise ValueError(f"{item.valarea}: ownership QC lacks owned_pixel_count")
    try:
        expected_count = int(row["owned_pixel_count"])
    except ValueError as exc:
        raise ValueError(f"{item.valarea}: invalid owned_pixel_count in ownership QC") from exc
    if expected_count != true_count:
        raise ValueError(f"{item.valarea}: ownership True count mismatch")
    expected_hash = row.get("owned_mask_sha256", "").strip().lower()
    if expected_hash:
        actual_hash = hashlib.sha256(mask.tobytes(order="C")).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"{item.valarea}: ownership mask hash mismatch")
        return mask, "verified"
    return mask, "hash_not_provided"


def _equal_region(regions: tuple[RegionResult, ...]) -> tuple[
        tuple[EqualRegionClassMetric, ...], float | None, float | None]:
    classes: list[EqualRegionClassMetric] = []
    for class_id in range(1, 9):
        per_region = [region.metrics.class_metrics[class_id - 1] for region in regions]
        supported = [metric for metric in per_region if metric.gt_support > 0]
        classes.append(EqualRegionClassMetric(
            class_id=class_id, supported_region_count=len(supported),
            equal_region_iou=(float(np.mean([metric.iou for metric in supported]))
                              if supported else None),
            gt_support_total=sum(metric.gt_support for metric in per_region),
        ))
    supported_classes = [metric.equal_region_iou for metric in classes
                         if metric.supported_region_count > 0]
    region_mious = [region.metrics.miou_8 for region in regions
                    if region.metrics.miou_8 is not None]
    return (tuple(classes),
            float(np.mean(supported_classes)) if supported_classes else None,
            float(np.mean(region_mious)) if region_mious else None)


def evaluate_prediction_set_with_ownership(
        items: Iterable[ManifestItem], dataset_root: Path, prediction_dir: Path,
        ownership_dir: Path, *, model_id: str = "") -> OwnershipEvaluationResult:
    """Evaluate raw and ownership surfaces without any grid transformation."""
    ordered = sorted(items, key=lambda item: (item.numeric_id, item.valarea))
    qc_rows = _read_qc(ownership_dir)
    owned_tiles: list[OwnedTileResult] = []
    raw_tiles: list[TileResult] = []
    seen: set[str] = set()
    for item in ordered:
        if item.valarea in seen:
            raise ValueError(f"duplicate manifest valarea: {item.valarea}")
        gt, prediction, valid_gt = load_raster_pair(
            dataset_root / "labels" / f"{item.valarea}.tif",
            prediction_dir / f"{item.valarea}.tif")
        ownership, hash_status = load_ownership_mask(
            ownership_dir, item, gt.shape, qc_rows)
        seen.add(item.valarea)
        raw = metrics_from_confusion(confusion_from_arrays(gt, prediction, valid_gt))
        owned = metrics_from_confusion(confusion_from_arrays(
            gt, prediction, valid_gt & ownership))
        if owned.valid_gt_pixels > raw.valid_gt_pixels or owned.semantic_gt_pixels > raw.semantic_gt_pixels:
            raise AssertionError(f"{item.valarea}: owned counts exceed raw counts")
        raw_tiles.append(TileResult(item, raw))
        owned_tiles.append(OwnedTileResult(item, owned, raw, int(ownership.sum()), hash_status))
    raw_pool = sum((tile.metrics.confusion for tile in raw_tiles),
                   np.zeros((9, 9), dtype=np.int64))
    raw_result = EvaluationResult(model_id, tuple(raw_tiles), metrics_from_confusion(raw_pool))
    region_results: list[RegionResult] = []
    for region in sorted({tile.item.region for tile in owned_tiles}):
        members = [tile for tile in owned_tiles if tile.item.region == region]
        pooled = sum((tile.metrics.confusion for tile in members),
                     np.zeros((9, 9), dtype=np.int64))
        region_results.append(RegionResult(region, metrics_from_confusion(pooled), len(members)))
    regions = tuple(region_results)
    global_confusion = sum((region.metrics.confusion for region in regions),
                           np.zeros((9, 9), dtype=np.int64))
    global_owned = metrics_from_confusion(global_confusion)
    tile_confusion = sum((tile.metrics.confusion for tile in owned_tiles),
                         np.zeros((9, 9), dtype=np.int64))
    if not np.array_equal(tile_confusion, global_confusion):
        raise AssertionError("region confusion sum differs from global owned confusion")
    if global_owned.valid_gt_pixels != sum(tile.metrics.valid_gt_pixels for tile in owned_tiles):
        raise AssertionError("global owned count differs from tile sum")
    equal_classes, equal_macro, mean_region = _equal_region(regions)
    return OwnershipEvaluationResult(model_id, raw_result, tuple(owned_tiles), regions,
                                     global_owned, equal_classes, equal_macro, mean_region)


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _metric_rows(result: OwnershipEvaluationResult, scope: str, region: str,
                 metrics: MetricResult) -> list[dict[str, object]]:
    return [{"schema_version": SCHEMA_VERSION, "model_id": result.model_id,
             "scope": scope, "region": region, "class_id": metric.class_id,
             "class_name": CLASS_NAMES[metric.class_id], "tp": metric.tp,
             "fp": metric.fp, "fn": metric.fn, "gt_support": metric.gt_support,
             "prediction_support": metric.prediction_support, "iou": metric.iou,
             "precision": metric.precision, "recall": metric.recall,
             "undefined_reason": metric.undefined_reason}
            for metric in metrics.class_metrics]


def _summary(result: OwnershipEvaluationResult, scope: str, region: str,
             metrics: MetricResult, tile_count: int) -> dict[str, object]:
    return {"schema_version": SCHEMA_VERSION, "model_id": result.model_id,
            "scope": scope, "region": region,
            "valid_owned_gt_pixels": metrics.valid_gt_pixels,
            "semantic_owned_gt_pixels": metrics.semantic_gt_pixels,
            "semantic_pixel_accuracy": metrics.semantic_pixel_accuracy,
            "miou_8": metrics.miou_8,
            "supported_class_count": metrics.supported_class_count,
            "tile_count": tile_count}


def _confusion(result: OwnershipEvaluationResult, scope: str, region: str,
               metrics: MetricResult) -> list[dict[str, object]]:
    return [{"schema_version": SCHEMA_VERSION, "model_id": result.model_id,
             "scope": scope, "region": region, "gt_class_id": gt,
             "prediction_class_id": prediction,
             "pixel_count": int(metrics.confusion[gt, prediction])}
            for gt in range(9) for prediction in range(9)]


def write_ownership_evaluation_outputs(output_dir: Path,
                                       result: OwnershipEvaluationResult) -> None:
    """Write Step-2 outputs plus deterministic Step-3 formal surfaces and QC."""
    write_evaluation_outputs(output_dir, result.raw)
    region_classes = [row for region in result.regions for row in
                      _metric_rows(result, "region_deduplicated", region.region, region.metrics)]
    region_summaries = [_summary(result, "region_deduplicated", region.region,
                                 region.metrics, region.tile_count) for region in result.regions]
    region_confusion = [row for region in result.regions for row in
                        _confusion(result, "region_deduplicated", region.region, region.metrics)]
    global_classes = _metric_rows(result, "global_deduplicated", "",
                                  result.global_deduplicated)
    global_summary = [_summary(result, "global_deduplicated", "",
                               result.global_deduplicated, len(result.tiles))]
    global_confusion = _confusion(result, "global_deduplicated", "",
                                  result.global_deduplicated)
    equal_classes = [{"schema_version": SCHEMA_VERSION, "model_id": result.model_id,
                      "class_id": metric.class_id, "class_name": CLASS_NAMES[metric.class_id],
                      "supported_region_count": metric.supported_region_count,
                      "equal_region_iou": metric.equal_region_iou,
                      "gt_support_total": metric.gt_support_total}
                     for metric in result.equal_region_classes]
    equal_summary = [{"schema_version": SCHEMA_VERSION, "model_id": result.model_id,
                      "equal_region_macro_miou_8": result.equal_region_macro_miou_8,
                      "supported_class_count": sum(m.supported_region_count > 0
                                                   for m in result.equal_region_classes),
                      "mean_region_miou_8": result.mean_region_miou_8,
                      "region_count": len(result.regions)}]
    qc_rows: list[dict[str, object]] = []
    for tile in result.tiles:
        qc_rows.append({"schema_version": SCHEMA_VERSION, "model_id": result.model_id,
                        "scope": "tile", "region": tile.item.region,
                        "valarea": tile.item.valarea,
                        "raw_valid_gt_pixels": tile.raw_metrics.valid_gt_pixels,
                        "owned_valid_gt_pixels": tile.metrics.valid_gt_pixels,
                        "raw_semantic_gt_pixels": tile.raw_metrics.semantic_gt_pixels,
                        "owned_semantic_gt_pixels": tile.metrics.semantic_gt_pixels,
                        "ownership_true_pixels": tile.ownership_true_pixels,
                        "dropped_valid_gt_pixels_due_to_ownership": tile.raw_metrics.valid_gt_pixels - tile.metrics.valid_gt_pixels,
                        "dropped_semantic_gt_pixels_due_to_ownership": tile.raw_metrics.semantic_gt_pixels - tile.metrics.semantic_gt_pixels,
                        "tile_count": 1, "supported_class_count": tile.metrics.supported_class_count,
                        "raw_minus_owned": tile.raw_metrics.valid_gt_pixels - tile.metrics.valid_gt_pixels,
                        "ownership_mask_hash_verification_status": tile.hash_status})
    for region in result.regions:
        qc_rows.append({"schema_version": SCHEMA_VERSION, "model_id": result.model_id,
                        "scope": "region", "region": region.region, "valarea": "",
                        "raw_valid_gt_pixels": "", "owned_valid_gt_pixels": region.metrics.valid_gt_pixels,
                        "raw_semantic_gt_pixels": "", "owned_semantic_gt_pixels": region.metrics.semantic_gt_pixels,
                        "ownership_true_pixels": "", "dropped_valid_gt_pixels_due_to_ownership": "",
                        "dropped_semantic_gt_pixels_due_to_ownership": "", "tile_count": region.tile_count,
                        "supported_class_count": region.metrics.supported_class_count,
                        "raw_minus_owned": "", "ownership_mask_hash_verification_status": ""})
    statuses = {tile.hash_status for tile in result.tiles}
    qc_rows.append({"schema_version": SCHEMA_VERSION, "model_id": result.model_id,
                    "scope": "global", "region": "", "valarea": "",
                    "raw_valid_gt_pixels": result.raw.global_raw.valid_gt_pixels,
                    "owned_valid_gt_pixels": result.global_deduplicated.valid_gt_pixels,
                    "raw_semantic_gt_pixels": result.raw.global_raw.semantic_gt_pixels,
                    "owned_semantic_gt_pixels": result.global_deduplicated.semantic_gt_pixels,
                    "ownership_true_pixels": sum(tile.ownership_true_pixels for tile in result.tiles),
                    "dropped_valid_gt_pixels_due_to_ownership": result.raw.global_raw.valid_gt_pixels - result.global_deduplicated.valid_gt_pixels,
                    "dropped_semantic_gt_pixels_due_to_ownership": result.raw.global_raw.semantic_gt_pixels - result.global_deduplicated.semantic_gt_pixels,
                    "tile_count": len(result.tiles),
                    "supported_class_count": result.global_deduplicated.supported_class_count,
                    "raw_minus_owned": result.raw.global_raw.valid_gt_pixels - result.global_deduplicated.valid_gt_pixels,
                    "ownership_mask_hash_verification_status": (statuses.pop() if len(statuses) == 1 else "mixed")})
    for filename, rows in (
        ("region_deduplicated_class_metrics.csv", region_classes),
        ("region_deduplicated_summary.csv", region_summaries),
        ("confusion_region_deduplicated.csv", region_confusion),
        ("global_deduplicated_class_metrics.csv", global_classes),
        ("global_deduplicated_summary.csv", global_summary),
        ("confusion_global_deduplicated.csv", global_confusion),
        ("equal_region_class_metrics.csv", equal_classes),
        ("equal_region_summary.csv", equal_summary),
        ("ownership_evaluation_qc.csv", qc_rows),
    ):
        _write(output_dir / filename, rows)
