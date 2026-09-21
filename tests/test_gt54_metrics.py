"""Synthetic tests for GT54 Step-2 single-model metrics."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.evaluation.gt54_metrics import (
    evaluate_arrays,
    evaluate_prediction_set,
    evaluate_raster_pair,
    metrics_from_confusion,
    write_evaluation_outputs,
)
from src.evaluation.gt54_preflight import ManifestItem


def metric(result, class_id):
    return result.class_metrics[class_id - 1]


def write_raster(path: Path, values: np.ndarray, *, nodata=None,
                 transform=None, mask=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    transform = transform or from_origin(0, values.shape[0], 1, 1)
    with rasterio.open(path, "w", driver="GTiff", width=values.shape[1],
                       height=values.shape[0], count=1, dtype=values.dtype,
                       crs="EPSG:3857", transform=transform, nodata=nodata) as dst:
        dst.write(values, 1)
        if mask is not None:
            dst.write_mask(mask)


def test_perfect_prediction_has_unit_semantic_metrics():
    values = np.arange(9, dtype=np.uint8).reshape(3, 3)
    result = evaluate_arrays(values, values, np.ones_like(values, dtype=bool))
    assert result.miou_8 == result.semantic_pixel_accuracy == 1.0
    assert all((m.iou, m.precision, m.recall) == (1.0, 1.0, 1.0)
               for m in result.class_metrics)


def test_background_is_in_confusion_but_excluded_from_semantic_denominator():
    result = evaluate_arrays(np.array([[0, 1]]), np.array([[8, 1]]),
                             np.array([[True, True]]))
    assert result.confusion[0, 8] == 1
    assert result.valid_gt_pixels == 2
    assert result.semantic_gt_pixels == 1
    assert result.semantic_pixel_accuracy == 1.0
    assert metric(result, 8).fp == 1


def test_semantic_prediction_as_background_is_valid_misclassification():
    result = evaluate_arrays(np.array([[4]]), np.array([[0]]), np.array([[True]]))
    assert result.confusion[4, 0] == 1
    assert result.semantic_pixel_accuracy == 0
    assert (metric(result, 4).iou, metric(result, 4).recall) == (0, 0)
    assert metric(result, 4).precision is None


def test_absent_class_rules_preserve_false_positives():
    absent = metrics_from_confusion(np.zeros((9, 9), dtype=np.int64))
    assert (metric(absent, 6).iou, metric(absent, 6).recall,
            metric(absent, 6).precision) == (None, None, None)
    confusion = np.zeros((9, 9), dtype=np.int64); confusion[0, 6] = 3
    predicted = metrics_from_confusion(confusion)
    assert (metric(predicted, 6).iou, metric(predicted, 6).recall,
            metric(predicted, 6).precision) == (None, None, 0)
    assert metric(predicted, 6).fp == metric(predicted, 6).prediction_support == 3


def test_miou_uses_only_gt_supported_semantic_classes():
    confusion = np.zeros((9, 9), dtype=np.int64)
    confusion[0, 0] = 100; confusion[1, 1] = 2; confusion[2, 2] = 1; confusion[2, 0] = 1
    result = metrics_from_confusion(confusion)
    assert result.supported_class_count == 2
    assert result.miou_8 == pytest.approx((1 + 0.5) / 2)


def test_gt_nodata_mask_is_distinct_from_background(tmp_path):
    gt = np.array([[255, 0], [1, 1]], dtype=np.uint8)
    prediction = np.array([[8, 8], [1, 1]], dtype=np.uint8)
    gt_path, prediction_path = tmp_path / "gt.tif", tmp_path / "prediction.tif"
    write_raster(gt_path, gt, nodata=255); write_raster(prediction_path, prediction)
    result = evaluate_raster_pair(gt_path, prediction_path)
    assert result.valid_gt_pixels == 3
    assert result.confusion[0, 8] == 1
    assert result.confusion[8, 8] == 0


def test_prediction_nodata_metadata_fails_closed(tmp_path):
    gt_path, prediction_path = tmp_path / "gt.tif", tmp_path / "prediction.tif"
    write_raster(gt_path, np.ones((2, 2), dtype=np.uint8))
    write_raster(prediction_path, np.ones((2, 2), dtype=np.uint8), nodata=255)
    with pytest.raises(ValueError, match="must not declare NoData"):
        evaluate_raster_pair(gt_path, prediction_path)


def test_prediction_outside_oem8_fails_closed(tmp_path):
    gt_path, prediction_path = tmp_path / "gt.tif", tmp_path / "prediction.tif"
    write_raster(gt_path, np.ones((2, 2), dtype=np.uint8))
    write_raster(prediction_path, np.array([[1, 1], [1, 9]], dtype=np.uint8))
    with pytest.raises(ValueError, match="outside 0..8"):
        evaluate_raster_pair(gt_path, prediction_path)


def test_grid_mismatch_fails_closed(tmp_path):
    gt_path, prediction_path = tmp_path / "gt.tif", tmp_path / "prediction.tif"
    write_raster(gt_path, np.ones((2, 2), dtype=np.uint8))
    write_raster(prediction_path, np.ones((2, 2), dtype=np.uint8),
                 transform=from_origin(1, 2, 1, 1))
    with pytest.raises(ValueError, match="grid mismatch"):
        evaluate_raster_pair(gt_path, prediction_path)


def test_global_raw_pools_confusion_instead_of_averaging_tile_metrics(tmp_path):
    root, predictions = tmp_path / "dataset", tmp_path / "predictions"
    items = (ManifestItem("ValArea_010", 2020, "B"),
             ManifestItem("ValArea_002", 2019, "A"))
    # Small tile is perfect; large tile has one correct pixel out of nine.
    write_raster(root / "labels/ValArea_002.tif", np.ones((1, 1), dtype=np.uint8))
    write_raster(predictions / "ValArea_002.tif", np.ones((1, 1), dtype=np.uint8))
    write_raster(root / "labels/ValArea_010.tif", np.ones((3, 3), dtype=np.uint8))
    pred = np.zeros((3, 3), dtype=np.uint8); pred[0, 0] = 1
    write_raster(predictions / "ValArea_010.tif", pred)
    result = evaluate_prediction_set(items, root, predictions, model_id="single")
    assert [tile.item.valarea for tile in result.tiles] == ["ValArea_002", "ValArea_010"]
    assert result.global_raw.semantic_pixel_accuracy == pytest.approx(0.2)
    assert result.global_raw.semantic_pixel_accuracy != pytest.approx((1 + 1 / 9) / 2)


def test_csv_order_is_numeric_valarea_then_fixed_class_order(tmp_path):
    root, predictions = tmp_path / "dataset", tmp_path / "predictions"
    items = (ManifestItem("ValArea_010", 2020, "B"),
             ManifestItem("ValArea_002", 2019, "A"))
    for item in items:
        write_raster(root / "labels" / f"{item.valarea}.tif", np.ones((1, 1), dtype=np.uint8))
        write_raster(predictions / f"{item.valarea}.tif", np.ones((1, 1), dtype=np.uint8))
    result = evaluate_prediction_set(items, root, predictions)
    output = tmp_path / "output"; write_evaluation_outputs(output, result)
    with (output / "tile_class_metrics.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [(row["valarea"], int(row["class_id"])) for row in rows] == [
        (valarea, class_id) for valarea in ("ValArea_002", "ValArea_010")
        for class_id in range(1, 9)
    ]
    with (output / "confusion_tile.csv").open(newline="", encoding="utf-8") as stream:
        confusion = list(csv.DictReader(stream))
    assert [(int(row["gt_class_id"]), int(row["prediction_class_id"]))
            for row in confusion[:81]] == [(gt, pred) for gt in range(9) for pred in range(9)]
    assert {path.name for path in output.iterdir()} == {
        "tile_class_metrics.csv", "tile_summary.csv", "global_raw_class_metrics.csv",
        "global_raw_summary.csv", "confusion_tile.csv", "confusion_global_raw.csv",
        "evaluation_summary.csv",
    }
