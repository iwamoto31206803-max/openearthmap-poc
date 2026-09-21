"""Single-model GT54 metrics on native, exactly aligned raster grids.

Step 2 deliberately performs raw pooling only.  It does not consume the
geographic ownership masks produced by Step 1.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio

from src.config import CLASS_NAMES
from src.evaluation.gt54_preflight import ManifestItem, VALID_CLASSES, _nodata_is_semantic


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ClassMetric:
    class_id: int
    tp: int
    fp: int
    fn: int
    gt_support: int
    prediction_support: int
    iou: float | None
    precision: float | None
    recall: float | None
    undefined_reason: str

    @property
    def iou_defined(self) -> bool:
        return self.iou is not None

    @property
    def precision_defined(self) -> bool:
        return self.precision is not None

    @property
    def recall_defined(self) -> bool:
        return self.recall is not None


@dataclass(frozen=True)
class MetricResult:
    confusion: np.ndarray
    valid_gt_pixels: int
    semantic_gt_pixels: int
    semantic_pixel_accuracy: float | None
    miou_8: float | None
    class_metrics: tuple[ClassMetric, ...]

    @property
    def supported_class_count(self) -> int:
        return sum(metric.gt_support > 0 for metric in self.class_metrics)


@dataclass(frozen=True)
class TileResult:
    item: ManifestItem
    metrics: MetricResult


@dataclass(frozen=True)
class EvaluationResult:
    model_id: str
    tiles: tuple[TileResult, ...]
    global_raw: MetricResult


def confusion_from_arrays(gt: np.ndarray, prediction: np.ndarray,
                          valid_gt_mask: np.ndarray) -> np.ndarray:
    """Return the GT-row/prediction-column OEM8 confusion matrix."""
    if gt.shape != prediction.shape or gt.shape != valid_gt_mask.shape:
        raise ValueError("GT, prediction, and valid mask shapes must match")
    selected_gt = gt[valid_gt_mask]
    selected_prediction = prediction[valid_gt_mask]
    if not np.isin(selected_prediction, tuple(VALID_CLASSES)).all():
        raise ValueError("prediction contains class IDs outside 0..8")
    encoded = selected_gt.astype(np.int64) * 9 + selected_prediction.astype(np.int64)
    return np.bincount(encoded, minlength=81).reshape(9, 9)


def metrics_from_confusion(confusion: np.ndarray) -> MetricResult:
    """Apply the protocol's absent-class conventions to a pooled confusion."""
    matrix = np.asarray(confusion, dtype=np.int64)
    if matrix.shape != (9, 9) or np.any(matrix < 0):
        raise ValueError("confusion must be a non-negative 9x9 matrix")
    class_metrics: list[ClassMetric] = []
    for class_id in range(1, 9):
        tp = int(matrix[class_id, class_id])
        gt_support = int(matrix[class_id, :].sum())
        prediction_support = int(matrix[:, class_id].sum())
        fn, fp = gt_support - tp, prediction_support - tp
        if gt_support == 0:
            iou = recall = None
            precision = 0.0 if prediction_support > 0 else None
            reason = "gt_support_zero"
        else:
            iou = tp / (tp + fp + fn)
            recall = tp / gt_support
            precision = tp / prediction_support if prediction_support > 0 else None
            reason = "prediction_support_zero" if prediction_support == 0 else ""
        class_metrics.append(ClassMetric(
            class_id, tp, fp, fn, gt_support, prediction_support,
            iou, precision, recall, reason,
        ))
    semantic_count = int(matrix[1:, :].sum())
    semantic_correct = int(np.trace(matrix[1:, 1:]))
    supported_ious = [metric.iou for metric in class_metrics if metric.gt_support > 0]
    return MetricResult(
        confusion=matrix.copy(),
        valid_gt_pixels=int(matrix.sum()),
        semantic_gt_pixels=semantic_count,
        semantic_pixel_accuracy=(semantic_correct / semantic_count if semantic_count else None),
        miou_8=(float(np.mean(supported_ious)) if supported_ious else None),
        class_metrics=tuple(class_metrics),
    )


def evaluate_arrays(gt: np.ndarray, prediction: np.ndarray,
                    valid_gt_mask: np.ndarray) -> MetricResult:
    """Evaluate arrays; class 0 remains in confusion but not primary metrics."""
    return metrics_from_confusion(confusion_from_arrays(gt, prediction, valid_gt_mask))


def _gt_valid_mask(dataset: rasterio.io.DatasetReader, values: np.ndarray) -> np.ndarray:
    """Build the GT mask and reject non-NoData values outside OEM8."""
    valid = dataset.read_masks(1) != 0
    if dataset.nodata is not None:
        nodata = dataset.nodata
        valid &= ~np.isnan(values) if np.isnan(nodata) else values != nodata
    valid_values = values[valid]
    invalid_classes = np.unique(valid_values[~np.isin(valid_values, tuple(VALID_CLASSES))])
    if invalid_classes.size:
        raise ValueError(
            f"valid GT pixels contain classes outside 0..8: {invalid_classes.tolist()}"
        )
    return valid


def evaluate_raster_pair(gt_path: Path, prediction_path: Path) -> MetricResult:
    """Validate and evaluate one pair without reprojection or resampling."""
    if not gt_path.is_file():
        raise FileNotFoundError(f"GT raster not found: {gt_path}")
    if not prediction_path.is_file():
        raise FileNotFoundError(f"prediction raster not found: {prediction_path}")
    with rasterio.open(gt_path) as gt, rasterio.open(prediction_path) as prediction:
        if gt.count != 1 or prediction.count != 1:
            raise ValueError("GT and prediction rasters must each have exactly one band")
        if gt.crs is None or prediction.crs is None:
            raise ValueError("GT and prediction rasters must have a CRS")
        mismatches = [field for field in ("width", "height", "crs", "transform", "bounds")
                      if getattr(gt, field) != getattr(prediction, field)]
        if mismatches:
            raise ValueError(f"GT/prediction grid mismatch: {', '.join(mismatches)}")
        if prediction.nodata is not None:
            raise ValueError("prediction raster must not declare NoData")
        if _nodata_is_semantic(gt.nodata):
            raise ValueError("GT NoData collides with OEM8 class 0..8")
        gt_values = gt.read(1)
        prediction_values = prediction.read(1)
        if not np.isin(prediction_values, tuple(VALID_CLASSES)).all():
            raise ValueError("prediction contains class IDs outside 0..8")
        valid_gt = _gt_valid_mask(gt, gt_values)
        return evaluate_arrays(gt_values, prediction_values, valid_gt)


def evaluate_prediction_set(items: Iterable[ManifestItem], dataset_root: Path,
                            prediction_dir: Path, *, model_id: str = "") -> EvaluationResult:
    ordered = sorted(items, key=lambda item: (item.numeric_id, item.valarea))
    tiles = tuple(TileResult(item, evaluate_raster_pair(
        dataset_root / "labels" / f"{item.valarea}.tif",
        prediction_dir / f"{item.valarea}.tif",
    )) for item in ordered)
    pooled = sum((tile.metrics.confusion for tile in tiles), np.zeros((9, 9), dtype=np.int64))
    return EvaluationResult(model_id, tiles, metrics_from_confusion(pooled))


def _base(result: EvaluationResult, scope: str, item: ManifestItem | None) -> dict[str, object]:
    return {"schema_version": SCHEMA_VERSION, "model_id": result.model_id, "scope": scope,
            "region": item.region if item else "", "valarea": item.valarea if item else ""}


def _class_rows(result: EvaluationResult, scope: str, item: ManifestItem | None,
                metrics: MetricResult) -> list[dict[str, object]]:
    return [{**_base(result, scope, item), "class_id": metric.class_id,
             "class_name": CLASS_NAMES[metric.class_id], "tp": metric.tp, "fp": metric.fp,
             "fn": metric.fn, "gt_support": metric.gt_support,
             "prediction_support": metric.prediction_support, "iou": metric.iou,
             "precision": metric.precision, "recall": metric.recall,
             "iou_defined": metric.iou_defined, "precision_defined": metric.precision_defined,
             "recall_defined": metric.recall_defined,
             "undefined_reason": metric.undefined_reason}
            for metric in metrics.class_metrics]


def _summary_row(result: EvaluationResult, scope: str, item: ManifestItem | None,
                 metrics: MetricResult) -> dict[str, object]:
    return {**_base(result, scope, item), "valid_gt_pixels": metrics.valid_gt_pixels,
            "semantic_gt_pixels": metrics.semantic_gt_pixels,
            "semantic_pixel_accuracy": metrics.semantic_pixel_accuracy,
            "miou_8": metrics.miou_8,
            "supported_class_count": metrics.supported_class_count}


def _confusion_rows(result: EvaluationResult, scope: str, item: ManifestItem | None,
                    metrics: MetricResult) -> list[dict[str, object]]:
    return [{**_base(result, scope, item), "gt_class_id": gt_id,
             "prediction_class_id": prediction_id,
             "pixel_count": int(metrics.confusion[gt_id, prediction_id])}
            for gt_id in range(9) for prediction_id in range(9)]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_evaluation_outputs(output_dir: Path, result: EvaluationResult) -> None:
    """Write deterministic long-form Step-2 CSV outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tile_classes = [row for tile in result.tiles
                    for row in _class_rows(result, "tile", tile.item, tile.metrics)]
    tile_summaries = [_summary_row(result, "tile", tile.item, tile.metrics)
                      for tile in result.tiles]
    tile_confusion = [row for tile in result.tiles
                      for row in _confusion_rows(result, "tile", tile.item, tile.metrics)]
    global_classes = _class_rows(result, "global_raw", None, result.global_raw)
    global_summary = [_summary_row(result, "global_raw", None, result.global_raw)]
    global_confusion = _confusion_rows(result, "global_raw", None, result.global_raw)
    evaluation_summary = [{**global_summary[0], "status": "PASS",
                           "tile_count": len(result.tiles),
                           "aggregation": "pooled_raw_confusion_with_spatial_overlap"}]
    for filename, rows in (
        ("tile_class_metrics.csv", tile_classes), ("tile_summary.csv", tile_summaries),
        ("global_raw_class_metrics.csv", global_classes),
        ("global_raw_summary.csv", global_summary), ("confusion_tile.csv", tile_confusion),
        ("confusion_global_raw.csv", global_confusion),
        ("evaluation_summary.csv", evaluation_summary),
    ):
        _write_csv(output_dir / filename, rows)
