"""Synthetic acceptance tests for formal GT54 Step-4B comparison."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.evaluation.gt54_comparison import (
    CATEGORIES, COMPARISON_PAIRS, analyze_cubes, change_category, cube_from_arrays,
    evaluate_model_comparisons, validate_inference_provenance,
    write_comparison_outputs,
)
from src.evaluation.gt54_inference import EXPECTED_CHECKPOINT_SHA256, MODEL_ORDER
from src.evaluation.gt54_metrics import metrics_from_confusion
from src.evaluation.gt54_preflight import ManifestItem


def _all_pair_cubes(cube: np.ndarray):
    return {pair: cube.copy() for pair in COMPARISON_PAIRS}


def test_five_change_categories_are_exclusive_and_exhaustive():
    gt = np.array([[1, 1, 1, 1, 1]])
    before = np.array([[1, 2, 1, 2, 2]])
    after = np.array([[1, 1, 2, 3, 2]])
    cube = cube_from_arrays(gt, before, after, np.ones(gt.shape, dtype=bool))
    result = analyze_cubes(_all_pair_cubes(cube))
    summary = result.summaries[0]
    assert [summary[name] for name in CATEGORIES] == [1, 1, 1, 1, 1]
    assert sum(summary[name] for name in CATEGORIES) == summary["formal_valid_pixels"]


def test_background_transitions_retained_but_excluded_from_semantic_summary():
    gt = np.array([[0, 1, 0]])
    before = np.array([[0, 1, 2]])
    after = np.array([[2, 0, 2]])
    result = analyze_cubes(_all_pair_cubes(cube_from_arrays(
        gt, before, after, np.ones(gt.shape, dtype=bool))))
    assert result.summaries[0]["formal_valid_pixels"] == 3
    assert result.summaries[0]["semantic_gt_pixels"] == 1
    rows = result.cubes[:729]
    assert next(r for r in rows if r["gt_class_id"] == 0 and
                r["before_class_id"] == 0 and r["after_class_id"] == 2)["pixel_count"] == 1
    assert next(r for r in result.transitions[:81] if r["before_class_id"] == 1 and
                r["after_class_id"] == 0)["pixel_count"] == 1


def test_class_rates_absence_and_metric_deltas_reuse_metric_core():
    gt = np.array([[1, 1, 2, 2]])
    before = np.array([[2, 1, 2, 1]])
    after = np.array([[1, 2, 2, 2]])
    cube = cube_from_arrays(gt, before, after, np.ones(gt.shape, dtype=bool))
    result = analyze_cubes(_all_pair_cubes(cube))
    class1 = result.correction_regression[0]
    assert class1["corrected_pixels"] == class1["regressed_pixels"] == 1
    assert class1["net_correct_change"] == 0
    assert class1["corrected_rate"] == class1["regressed_rate"] == .5
    absent = result.correction_regression[2]
    assert absent["gt_support"] == 0 and absent["net_correct_rate"] is None
    expected_before = metrics_from_confusion(cube.sum(axis=2)).class_metrics[0]
    expected_after = metrics_from_confusion(cube.sum(axis=1)).class_metrics[0]
    metric = result.metrics[0]
    assert metric["delta_iou"] == pytest.approx(expected_after.iou - expected_before.iou)
    assert metric["delta_precision"] == pytest.approx(expected_after.precision - expected_before.precision)
    assert metric["delta_recall"] == pytest.approx(expected_after.recall - expected_before.recall)


def test_complete_matrices_fixed_diagnostics_and_top_tie_order():
    gt = np.array([[4, 8, 4, 6]])
    before = np.array([[8, 8, 6, 6]])
    after = np.array([[4, 4, 4, 4]])
    result = analyze_cubes(_all_pair_cubes(cube_from_arrays(
        gt, before, after, np.ones(gt.shape, dtype=bool))))
    assert len(result.transitions) == 5 * 81
    assert len(result.cubes) == 5 * 729
    fixed = [r for r in result.fixed_transitions if r["comparison"] == "A->B"
             and r["before_class_id"] == 8 and r["after_class_id"] == 4]
    assert len(result.fixed_transitions) == 5 * 6
    assert fixed[0]["transition_pixel_count"] == 2
    assert next(r for r in result.cubes if r["comparison"] == "A->B"
                and r["gt_class_id"] == 4 and r["before_class_id"] == 8
                and r["after_class_id"] == 4)["pixel_count"] == 1
    water = next(r for r in result.fixed_transitions if r["comparison"] == "A->B"
                 and r["before_class_id"] == 6 and r["after_class_id"] == 4)
    assert water["transition_pixel_count"] == 2
    top = [r for r in result.top_transitions if r["comparison"] == "A->B"]
    assert [(r["before_class_id"], r["after_class_id"]) for r in top] == [(6, 4), (8, 4)]
    assert all(r["before_class_id"] != r["after_class_id"] for r in top)


def test_mask_excludes_ownership_false_and_invalid_gt():
    gt = np.array([[1, 2, 255]])
    before = np.array([[0, 2, 1]])
    after = np.array([[1, 3, 2]])
    formal = np.array([[True, False, False]])
    cube = cube_from_arrays(gt, before, after, formal)
    assert cube.sum() == 1 and cube[1, 0, 1] == 1


def test_pair_order_and_category_function():
    assert COMPARISON_PAIRS == (("A", "B"), ("A", "C"), ("A", "D"),
                                ("B", "C"), ("C", "D"))
    assert change_category(1, 1, 1) == "unchanged_correct"
    assert change_category(1, 2, 1) == "correction"
    assert change_category(1, 1, 2) == "regression"
    assert change_category(1, 2, 3) == "changed_still_wrong"
    assert change_category(1, 2, 2) == "unchanged_wrong"


def _raster(path: Path, values: np.ndarray, *, nodata=None, x=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", width=values.shape[1],
                       height=values.shape[0], count=1, dtype=values.dtype,
                       crs="EPSG:3857", transform=from_origin(x, values.shape[0], 1, 1),
                       nodata=nodata) as dst:
        dst.write(values, 1)


def _integration_case(tmp_path: Path):
    item = ManifestItem("ValArea_001", 2020, "Region")
    gt = np.array([[1, 2, 255]], dtype=np.uint8)
    root, inference, ownership = tmp_path / "data", tmp_path / "inference", tmp_path / "own"
    _raster(root / "labels" / f"{item.valarea}.tif", gt, nodata=255)
    for index, model in enumerate(MODEL_ORDER):
        _raster(inference / model / "predictions" / f"{item.valarea}.tif",
                np.array([[1, index, 0]], dtype=np.uint8))
    mask = np.array([[True, False, True]])
    path = ownership / "ownership_masks" / f"{item.valarea}.npy"
    path.parent.mkdir(parents=True); np.save(path, mask, allow_pickle=False)
    with (ownership / "ownership_qc.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["valarea", "owned_pixel_count", "owned_mask_sha256"])
        writer.writeheader(); writer.writerow({"valarea": item.valarea, "owned_pixel_count": 2,
            "owned_mask_sha256": hashlib.sha256(mask.tobytes(order="C")).hexdigest()})
    return item, root, inference, ownership


def test_raster_evaluation_reuses_owned_valid_surface_and_regions(tmp_path):
    item, root, inference, ownership = _integration_case(tmp_path)
    result = evaluate_model_comparisons([item], root, inference, ownership,
                                        expected_items=1, validate_provenance=False)
    assert all(row["formal_valid_pixels"] == 1 for row in result.summaries)
    assert len(result.region_metrics) == 5 * 8
    assert all(row["status"] == "PASS" for row in result.qc)
    output = tmp_path / "output"; write_comparison_outputs(output, result)
    assert len(list(output.glob("*.csv"))) == 9


def test_prediction_grid_and_class_validation_fail_closed(tmp_path):
    item, root, inference, ownership = _integration_case(tmp_path)
    bad = inference / "D" / "predictions" / f"{item.valarea}.tif"
    _raster(bad, np.array([[9, 0, 0]], dtype=np.uint8))
    with pytest.raises(ValueError, match="outside 0..8"):
        evaluate_model_comparisons([item], root, inference, ownership,
                                   expected_items=1, validate_provenance=False)
    _raster(bad, np.array([[1, 0, 0]], dtype=np.uint8), x=1)
    with pytest.raises(ValueError, match="grid mismatch"):
        evaluate_model_comparisons([item], root, inference, ownership,
                                   expected_items=1, validate_provenance=False)


def _provenance(tmp_path: Path, *, mode="formal_full", missing_model=False):
    item = ManifestItem("ValArea_001", 2020, "R")
    root = tmp_path / "inference"; root.mkdir(parents=True)
    models = [{"model_id": m, "sha256": EXPECTED_CHECKPOINT_SHA256[m],
               "expected_sha256": EXPECTED_CHECKPOINT_SHA256[m]} for m in MODEL_ORDER]
    (root / "inference_config.json").write_text(json.dumps({"run": {
        "run_mode": mode, "inventory_item_count": 1, "selected_item_count": 1},
        "models": models}), encoding="utf-8")
    manifest = []
    for model in MODEL_ORDER:
        if missing_model and model == "D": continue
        prediction = root / model / "predictions" / f"{item.valarea}.tif"
        prediction.parent.mkdir(parents=True); prediction.touch()
        manifest.append({"model_id": model, "valarea": item.valarea, "status": "PASS",
            "run_mode": mode, "inventory_item_count": 1, "selected_item_count": 1,
            "rgb_sha256": "same", "width": "1", "height": "1", "crs": "c",
            "transform": "t", "bounds": "b"})
    with (root / "inference_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0])); writer.writeheader(); writer.writerows(manifest)
    qc = [{"model_id": m, "checkpoint_path": m,
           "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256[m],
           "expected_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256[m],
           "run_mode": mode, "status": "PASS", "item_count": "1"} for m in MODEL_ORDER]
    with (root / "inference_qc.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(qc[0])); writer.writeheader(); writer.writerows(qc)
    return root, item


def test_smoke_and_missing_prediction_provenance_fail_closed(tmp_path):
    root, item = _provenance(tmp_path / "smoke", mode="smoke_subset")
    with pytest.raises(ValueError, match="formal_full"):
        validate_inference_provenance(root, [item], expected_items=1)
    root, item = _provenance(tmp_path / "missing", missing_model=True)
    with pytest.raises(ValueError, match="expected_items x 4 rows"):
        validate_inference_provenance(root, [item], expected_items=1)
