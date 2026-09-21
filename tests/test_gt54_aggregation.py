"""Synthetic tests for GT54 Step-3 ownership-aware aggregation."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.evaluation.gt54_aggregation import (
    evaluate_prediction_set_with_ownership, load_ownership_mask,
    write_ownership_evaluation_outputs,
)
from src.evaluation.gt54_preflight import ManifestItem


def _raster(path: Path, values: np.ndarray, *, nodata=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", width=values.shape[1],
                       height=values.shape[0], count=1, dtype=values.dtype,
                       crs="EPSG:3857", transform=from_origin(0, values.shape[0], 1, 1),
                       nodata=nodata) as dst:
        dst.write(values, 1)


def _case(tmp_path: Path, specifications):
    root, predictions, ownership = (tmp_path / "dataset", tmp_path / "predictions",
                                    tmp_path / "ownership")
    rows = []
    items = []
    for number, (region, gt, prediction, mask) in enumerate(specifications, 1):
        valarea = f"ValArea_{number:03d}"
        item = ManifestItem(valarea, 2020, region); items.append(item)
        _raster(root / "labels" / f"{valarea}.tif", gt, nodata=255)
        _raster(predictions / f"{valarea}.tif", prediction)
        path = ownership / "ownership_masks" / f"{valarea}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, mask, allow_pickle=False)
        rows.append({"valarea": valarea, "owned_pixel_count": int(mask.sum()),
                     "owned_mask_sha256": hashlib.sha256(mask.tobytes(order="C")).hexdigest()})
    with (ownership / "ownership_qc.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return items, root, predictions, ownership


def test_all_true_ownership_exactly_matches_step2_global_raw(tmp_path):
    gt = np.array([[0, 1], [2, 255]], dtype=np.uint8)
    pred = np.array([[8, 1], [0, 7]], dtype=np.uint8)
    args = _case(tmp_path, [("B", gt, pred, np.ones((2, 2), dtype=bool))])
    result = evaluate_prediction_set_with_ownership(*args, model_id="A")
    assert np.array_equal(result.raw.global_raw.confusion,
                          result.global_deduplicated.confusion)
    assert result.global_deduplicated.valid_gt_pixels == 3
    assert result.global_deduplicated.semantic_gt_pixels == 2
    assert result.global_deduplicated.confusion[0, 8] == 1


def test_two_tile_overlap_counts_non_owner_only_once_and_pools_regions(tmp_path):
    gt = np.array([[1, 2]], dtype=np.uint8)
    pred = gt.copy()
    args = _case(tmp_path, [
        ("OwnerRegion", gt, pred, np.array([[True, True]])),
        ("OtherRegion", gt, pred, np.array([[False, False]])),
    ])
    result = evaluate_prediction_set_with_ownership(*args)
    assert result.raw.global_raw.valid_gt_pixels == 4
    assert result.raw.global_raw.valid_gt_pixels > result.global_deduplicated.valid_gt_pixels
    assert result.global_deduplicated.valid_gt_pixels == 2
    assert np.array_equal(sum((r.metrics.confusion for r in result.regions),
                              np.zeros((9, 9), dtype=np.int64)),
                          result.global_deduplicated.confusion)
    by_region = {region.region: region for region in result.regions}
    assert by_region["OwnerRegion"].metrics.valid_gt_pixels == 2
    assert by_region["OtherRegion"].metrics.valid_gt_pixels == 0
    assert by_region["OwnerRegion"].metrics.confusion[1, 1] == 1
    assert by_region["OwnerRegion"].metrics.confusion[2, 2] == 1
    assert by_region["OtherRegion"].metrics.confusion[1, 1] == 0
    assert by_region["OtherRegion"].metrics.confusion[2, 2] == 0


@pytest.mark.parametrize("mask,message", [
    (np.ones((1, 1), dtype=bool), "shape mismatch"),
    (np.ones((1, 2), dtype=np.uint8), "dtype must be bool"),
])
def test_mask_shape_and_dtype_fail_closed(tmp_path, mask, message):
    item = ManifestItem("ValArea_001", 2020, "A")
    directory = tmp_path / "ownership" / "ownership_masks"; directory.mkdir(parents=True)
    np.save(directory / "ValArea_001.npy", mask, allow_pickle=False)
    with pytest.raises(ValueError, match=message):
        load_ownership_mask(tmp_path / "ownership", item, (1, 2))


def test_missing_and_hash_mismatched_masks_fail_closed(tmp_path):
    item = ManifestItem("ValArea_001", 2020, "A")
    with pytest.raises(FileNotFoundError):
        load_ownership_mask(tmp_path, item, (1, 1))
    directory = tmp_path / "ownership_masks"; directory.mkdir()
    np.save(directory / "ValArea_001.npy", np.ones((1, 1), dtype=bool))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_ownership_mask(tmp_path, item, (1, 1), {item.valarea: {
            "owned_pixel_count": "1", "owned_mask_sha256": "0" * 64}})


def test_nodata_and_ownership_false_are_both_excluded(tmp_path):
    gt = np.array([[255, 1, 2]], dtype=np.uint8)
    pred = np.array([[8, 1, 2]], dtype=np.uint8)
    mask = np.array([[True, False, True]])
    result = evaluate_prediction_set_with_ownership(*_case(tmp_path, [("A", gt, pred, mask)]))
    assert result.raw.global_raw.valid_gt_pixels == 2
    assert result.global_deduplicated.valid_gt_pixels == 1
    assert result.global_deduplicated.confusion[2, 2] == 1


def test_class_first_equal_region_macro_excludes_unsupported_pairs(tmp_path):
    # Region A supports classes 1 and 2 perfectly; B supports only class 1, at IoU 0.
    # Class-first = mean(mean(1, 0), mean(1)) = .75; mean region mIoU = .5.
    args = _case(tmp_path, [
        ("A", np.array([[1, 2]], dtype=np.uint8), np.array([[1, 2]], dtype=np.uint8),
         np.ones((1, 2), dtype=bool)),
        ("B", np.array([[1]], dtype=np.uint8), np.array([[0]], dtype=np.uint8),
         np.ones((1, 1), dtype=bool)),
    ])
    result = evaluate_prediction_set_with_ownership(*args)
    assert result.equal_region_macro_miou_8 == pytest.approx(0.75)
    assert result.mean_region_miou_8 == pytest.approx(0.5)
    assert result.equal_region_classes[0].supported_region_count == 2
    assert result.equal_region_classes[1].supported_region_count == 1
    assert result.equal_region_classes[2].equal_region_iou is None


def test_deterministic_region_class_order_and_complete_csv_outputs(tmp_path):
    args = _case(tmp_path, [
        ("Z", np.array([[1]], dtype=np.uint8), np.array([[1]], dtype=np.uint8),
         np.ones((1, 1), dtype=bool)),
        ("A", np.array([[2]], dtype=np.uint8), np.array([[2]], dtype=np.uint8),
         np.ones((1, 1), dtype=bool)),
    ])
    result = evaluate_prediction_set_with_ownership(*args, model_id="single")
    assert [region.region for region in result.regions] == ["A", "Z"]
    output = tmp_path / "output"; write_ownership_evaluation_outputs(output, result)
    with (output / "region_deduplicated_class_metrics.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [(row["region"], int(row["class_id"])) for row in rows] == [
        (region, class_id) for region in ("A", "Z") for class_id in range(1, 9)]
    required = {"region_deduplicated_class_metrics.csv", "region_deduplicated_summary.csv",
                "confusion_region_deduplicated.csv", "global_deduplicated_class_metrics.csv",
                "global_deduplicated_summary.csv", "confusion_global_deduplicated.csv",
                "equal_region_class_metrics.csv", "equal_region_summary.csv",
                "ownership_evaluation_qc.csv"}
    assert required <= {path.name for path in output.iterdir()}
    with (output / "ownership_evaluation_qc.csv").open(newline="", encoding="utf-8") as stream:
        qc = list(csv.DictReader(stream))
    assert qc[-1]["ownership_mask_hash_verification_status"] == "verified"
    assert int(qc[-1]["raw_valid_gt_pixels"]) >= int(qc[-1]["owned_valid_gt_pixels"])
