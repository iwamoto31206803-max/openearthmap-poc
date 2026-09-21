"""Synthetic tests for GT54 Step-1 inventory and grid preflight."""

from __future__ import annotations

import csv
from pathlib import Path

from affine import Affine
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.evaluation.gt54_preflight import (
    GridTolerances,
    inventory_dataset,
    load_manifest,
    preflight_grids,
    validate_local_manifest,
)
from src.evaluation.gt54_ownership import build_ownership_masks
from tools.evaluation import preflight_gt54


def make_dataset(tmp_path: Path, specifications: list[dict]) -> tuple[Path, Path]:
    root = tmp_path / "dataset"
    manifest = tmp_path / "formal.csv"
    formal_rows, local_rows = [], []
    for index, spec in enumerate(specifications, 1):
        valarea = spec.get("valarea", f"ValArea_{index:03d}")
        region = spec.get("region", f"Region_{index}")
        year = spec.get("year", 2019)
        transform = spec["transform"]
        crs = spec.get("crs", "EPSG:3857")
        width, height = spec.get("width", 4), spec.get("height", 4)
        nodata = spec.get("nodata")
        rgb_path = root / "rgb_images" / f"{valarea}.tif"
        gt_path = root / "labels" / f"{valarea}.tif"
        rgb_path.parent.mkdir(parents=True, exist_ok=True)
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(rgb_path, "w", driver="GTiff", count=3, width=width, height=height,
                           dtype="uint8", crs=crs, transform=transform) as dst:
            dst.write(np.ones((3, height, width), dtype=np.uint8))
        with rasterio.open(gt_path, "w", driver="GTiff", count=1, width=width, height=height,
                           dtype="uint8", crs=crs, transform=transform, nodata=nodata) as dst:
            dst.write(np.ones((1, height, width), dtype=np.uint8))
            if "mask" in spec:
                dst.write_mask(spec["mask"])
        formal_rows.append((valarea, year, region))
        local_rows.append({"valarea": valarea, "region": region, "gsi_year": year,
                           "alignment_ok": "True", "error": ""})
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(("valarea", "year", "region")); writer.writerows(formal_rows)
    with (root / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=local_rows[0].keys())
        writer.writeheader(); writer.writerows(local_rows)
    return manifest, root


def inspect(tmp_path: Path, specifications: list[dict]):
    manifest, root = make_dataset(tmp_path, specifications)
    regions = len({spec.get("region", f"Region_{index}")
                   for index, spec in enumerate(specifications, 1)})
    items = load_manifest(manifest, expected_items=len(specifications), expected_regions=regions)
    validate_local_manifest(root / "manifest.csv", items)
    return preflight_grids(inventory_dataset(items, root))


def test_perfectly_aligned_overlap_passes_with_integer_placement(tmp_path):
    result = inspect(tmp_path, [
        {"transform": from_origin(0, 4, 1, 1), "region": "Same"},
        {"transform": from_origin(2, 4, 1, 1), "region": "Same"},
    ])
    assert result.passed
    assert [(p.valarea, p.grid_group_id, p.global_row_offset, p.global_col_offset)
            for p in result.placements] == [
                ("ValArea_001", "grid_000", 0, 0),
                ("ValArea_002", "grid_000", 0, 2),
            ]
    masks, rows = build_ownership_masks(result.inventory)
    assert sum(int(mask.sum()) for mask in masks.values()) < sum(mask.size for mask in masks.values())
    assert all(row["status"] == "PASS" for row in rows)


def test_edge_contact_is_not_overlap(tmp_path):
    result = inspect(tmp_path, [
        {"transform": from_origin(0, 4, 1, 1)},
        {"transform": from_origin(4, 4, 1, 1)},
    ])
    assert result.passed
    assert not [row for row in result.grid_rows if row["record_type"] == "overlap_pair"]
    assert len({placement.grid_group_id for placement in result.placements}) == 2
    masks, _ = build_ownership_masks(result.inventory)
    assert all(mask.all() for mask in masks.values())


@pytest.mark.parametrize(("transform", "reason"), [
    (from_origin(2.5, 4, 1, 1), "phase"),
    (from_origin(2, 4, 1.01, 1), "pixel size"),
    (Affine(1, 0, 2, 0.01, -1, 4), "rotation"),
    (Affine(1, 0.01, 2, 0, -1, 4), "shear"),
])
def test_incompatible_overlapping_grids_fail(tmp_path, transform, reason):
    result = inspect(tmp_path, [
        {"transform": from_origin(0, 4, 1, 1), "region": "Same"},
        {"transform": transform, "region": "Same"},
    ])
    assert result.passed, reason
    assert not result.pixel_identical_dedup_available
    assert result.geographic_ownership_available
    assert result.errors
    assert result.placements == ()
    masks, _ = build_ownership_masks(result.inventory)
    assert masks


def test_tolerance_boundary_for_metadata_noise(tmp_path):
    tolerances = GridTolerances()
    passing = inspect(tmp_path / "pass", [
        {"transform": from_origin(0, 4, 1, 1), "region": "Same"},
        {"transform": from_origin(2 + tolerances.phase_pixels / 2, 4, 1, 1), "region": "Same"},
    ])
    failing = inspect(tmp_path / "fail", [
        {"transform": from_origin(0, 4, 1, 1), "region": "Same"},
        {"transform": from_origin(2 + tolerances.phase_pixels * 2, 4, 1, 1), "region": "Same"},
    ])
    assert passing.passed
    assert failing.passed
    assert not failing.pixel_identical_dedup_available


def test_different_non_overlapping_grid_groups_are_allowed(tmp_path):
    result = inspect(tmp_path, [
        {"transform": from_origin(0, 4, 1, 1), "crs": "EPSG:3857"},
        {"transform": from_origin(130, 40, .001, .001), "crs": "EPSG:4326"},
    ])
    assert result.passed
    assert len({placement.grid_group_id for placement in result.placements}) == 2
    masks, _ = build_ownership_masks(result.inventory)
    assert all(mask.all() for mask in masks.values())


def test_incompatible_overlapping_crs_groups_fail(tmp_path):
    result = inspect(tmp_path, [
        {"transform": from_origin(0, 400, 100, 100), "crs": "EPSG:3857", "region": "Same"},
        {"transform": from_origin(0, .0036, .0009, .0009), "crs": "EPSG:4326", "region": "Same"},
    ])
    assert result.passed
    assert not result.pixel_identical_dedup_available
    assert any("incompatible overlapping" in error for error in result.errors)
    masks, rows = build_ownership_masks(result.inventory)
    assert masks and all(row["status"] == "PASS" for row in rows)


def test_placement_is_deterministic_by_numeric_valarea_id(tmp_path):
    result = inspect(tmp_path, [
        {"valarea": "ValArea_010", "transform": from_origin(2, 4, 1, 1), "region": "Same"},
        {"valarea": "ValArea_002", "transform": from_origin(0, 4, 1, 1), "region": "Same"},
    ])
    assert [placement.valarea for placement in result.placements] == ["ValArea_002", "ValArea_010"]
    assert [placement.global_col_offset for placement in result.placements] == [0, 2]


def test_gt_nodata_cannot_collide_with_background_or_semantic_class(tmp_path):
    manifest, root = make_dataset(tmp_path, [
        {"transform": from_origin(0, 4, 1, 1), "nodata": 0},
    ])
    items = load_manifest(manifest, expected_items=1, expected_regions=1)
    with pytest.raises(ValueError, match="collides with OEM8"):
        inventory_dataset(items, root)


def test_gt_mask_inventory_is_separate_from_class_zero(tmp_path):
    mask = np.full((4, 4), 255, dtype=np.uint8); mask[0, 0] = 0
    manifest, root = make_dataset(tmp_path, [
        {"transform": from_origin(0, 4, 1, 1), "nodata": 255, "mask": mask},
    ])
    items = load_manifest(manifest, expected_items=1, expected_regions=1)
    metadata = inventory_dataset(items, root)[0]
    assert metadata.gt_invalid_mask_pixels == 1
    assert metadata.gt_valid_pixels == 15
    assert metadata.gt_class_histogram[0] == 0
    assert metadata.gt_class_histogram[1] == 15


def test_cli_continues_when_pixel_identical_preflight_is_unavailable(tmp_path):
    manifest, root = make_dataset(tmp_path, [
        {"transform": from_origin(0, 4, 1, 1), "region": "Same"},
        {"transform": from_origin(2.5, 4, 1, 1), "region": "Same"},
    ])
    output = tmp_path / "output"
    assert preflight_gt54.main([
        "--manifest", str(manifest), "--dataset-root", str(root), "--output-dir", str(output),
        "--expected-items", "2", "--expected-regions", "1",
    ]) == 0
    assert {path.name for path in output.iterdir()} == {
        "evaluation_summary.csv", "dataset_qc.csv", "grid_preflight_qc.csv",
        "ownership_qc.csv", "ownership_masks",
    }
    assert not list(output.glob("*deduplicated*"))


def test_cli_fails_closed_when_geographic_ownership_is_impossible(tmp_path, monkeypatch):
    manifest, root = make_dataset(tmp_path, [
        {"transform": from_origin(0, 4, 1, 1)},
    ])
    output = tmp_path / "output"

    def impossible(_inventory):
        raise ValueError("unusable transformed footprint")

    monkeypatch.setattr(preflight_gt54, "build_ownership_masks", impossible)
    assert preflight_gt54.main([
        "--manifest", str(manifest), "--dataset-root", str(root), "--output-dir", str(output),
        "--expected-items", "1", "--expected-regions", "1",
    ]) == 1
    assert {path.name for path in output.iterdir()} == {
        "evaluation_summary.csv", "dataset_qc.csv", "grid_preflight_qc.csv"
    }


def test_ownership_is_independent_of_gt_values_and_deterministic(tmp_path):
    result = inspect(tmp_path, [
        {"valarea": "ValArea_010", "transform": from_origin(0, 4, 1, 1), "region": "Same"},
        {"valarea": "ValArea_002", "transform": from_origin(0, 4, 1, 1), "region": "Same"},
    ])
    first, _ = build_ownership_masks(result.inventory, chunk_rows=1)
    # Identical footprints are an all-pixel tie; the smaller numeric suffix owns all.
    assert first["ValArea_002"].all()
    assert not first["ValArea_010"].any()
    for metadata in result.inventory:
        with rasterio.open(metadata.gt_path, "r+") as raster:
            raster.write(np.full((1, metadata.height, metadata.width), 8, dtype=np.uint8))
    second, _ = build_ownership_masks(result.inventory, chunk_rows=3)
    assert all(np.array_equal(first[name], second[name]) for name in first)


def test_ownership_does_not_modify_or_resample_native_rasters(tmp_path):
    result = inspect(tmp_path, [
        {"transform": from_origin(0, 4, 1, 1), "region": "Same"},
        {"transform": from_origin(2.5, 4, 1.00002, 1.00002), "region": "Same"},
    ])
    before = {meta.item.valarea: (meta.gt_path.read_bytes(), meta.transform, meta.width, meta.height)
              for meta in result.inventory}
    masks, _ = build_ownership_masks(result.inventory)
    after = {meta.item.valarea: (meta.gt_path.read_bytes(), meta.transform, meta.width, meta.height)
             for meta in result.inventory}
    assert before == after
    assert {name: mask.shape for name, mask in masks.items()} == {
        meta.item.valarea: (meta.height, meta.width) for meta in result.inventory
    }
