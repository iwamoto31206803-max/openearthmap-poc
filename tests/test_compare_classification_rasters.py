"""Tests for direct OEM8 raster transition diagnostics."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.evaluation import compare_classification_progression as progression
from tools.evaluation import compare_classification_rasters as comparison


def write_raster(path: Path, values: np.ndarray, *, origin_x: float = 0) -> None:
    with rasterio.open(path, "w", driver="GTiff", width=values.shape[1], height=values.shape[0],
                       count=1, dtype="uint8", crs="EPSG:3857",
                       transform=from_origin(origin_x, 4, 1, 1)) as dst:
        dst.write(values, 1)


def test_outputs_include_full_matrix_major_transitions_and_changed_raster(tmp_path, capsys):
    source = np.array([[3, 4, 8, 8], [5, 2, 7, 6], [7, 5, 0, 0], [7, 1, 1, 1]], dtype=np.uint8)
    target = np.array([[4, 3, 5, 4], [4, 4, 4, 5], [5, 8, 6, 0], [7, 1, 1, 1]], dtype=np.uint8)
    first, second = tmp_path / "first.tif", tmp_path / "second.tif"
    write_raster(first, source); write_raster(second, target)
    output = tmp_path / "out"

    assert comparison.main([str(first), str(second), "--label", "02_to_03",
                            "--output-dir", str(output), "--changed-geotiff"]) == 0
    summary = json.loads((output / "02_to_03/summary.json").read_text())
    assert summary["total_pixel_count"] == 16
    assert summary["changed_pixel_count"] == 11
    assert summary["unchanged_pixel_count"] == 5
    assert len(summary["transitions"]) == 81
    assert len(summary["changed_only_transitions"]) == 72
    assert {(row["from_id"], row["to_id"]): row["pixel_count"]
            for row in summary["major_transitions"]} == {
                (3, 4): 1, (4, 3): 1, (8, 5): 1, (8, 4): 1, (5, 4): 1,
                (2, 4): 1, (7, 4): 1, (6, 5): 1, (7, 5): 1, (5, 8): 1}
    assert len(summary["nonzero_changed_transitions"]) == 11
    assert all(row["changed"] and row["pixel_count"] > 0
               for row in summary["nonzero_changed_transitions"])
    assert len(summary["background_transitions"]) == 9
    with (output / "02_to_03/transition_matrix.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 81
    with rasterio.open(output / "02_to_03/changed_only.tif") as changed:
        assert changed.read(1).tolist() == [
            [34, 43, 85, 84], [54, 24, 74, 65], [75, 58, 6, 0], [0, 0, 0, 0]]
        assert changed.transform == from_origin(0, 4, 1, 1)
    terminal = capsys.readouterr().out
    assert "Building -> Tree" not in terminal  # OEM8 canonical name is Buildings.
    assert "Buildings -> Tree: 1" in terminal
    assert "Background/Unlabelled -> Water: 1" in terminal


def test_row_col_aoi_updates_counts_and_georeferencing(tmp_path):
    source = np.zeros((4, 4), dtype=np.uint8)
    target = source.copy(); target[1:3, 1:4] = 6
    first, second = tmp_path / "first.tif", tmp_path / "second.tif"
    write_raster(first, source); write_raster(second, target)
    parser = comparison.build_parser()
    args = parser.parse_args([str(first), str(second), "--label", "aoi", "--output-dir", str(tmp_path),
                              "--rows", "1", "3", "--cols", "1", "4", "--changed-geotiff"])
    comparison.validate_args(parser, args)
    summary = comparison.run(args)
    assert summary["total_pixel_count"] == 6
    assert summary["changed_pixel_count"] == 6
    assert summary["aoi"]["row_start"] == 1
    with rasterio.open(tmp_path / "aoi/changed_only.tif") as output:
        assert output.shape == (2, 3)
        assert output.transform == from_origin(1, 3, 1, 1)


def test_progression_wrapper_creates_three_labeled_comparisons(tmp_path):
    paths = []
    for stage in range(4):
        path = tmp_path / f"{stage}.tif"
        write_raster(path, np.full((2, 2), stage, dtype=np.uint8))
        paths.append(path)
    output = tmp_path / "result"
    assert progression.main(["--stage-00", str(paths[0]), "--stage-01", str(paths[1]),
                             "--stage-02", str(paths[2]), "--stage-03", str(paths[3]),
                             "--output-dir", str(output)]) == 0
    assert {path.name for path in output.iterdir()} == {"00_to_01", "01_to_02", "02_to_03"}
    assert json.loads((output / "02_to_03/summary.json").read_text())["comparison_label"] == "02_to_03"
