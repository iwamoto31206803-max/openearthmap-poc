"""Synthetic-only tests for teacher progression diagnostics."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.evaluation import compare_teacher_progression as progression


def write_raster(path: Path, values: np.ndarray, *, origin: float = 0, bands: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = values if bands == 1 else np.stack([values] * bands)
    with rasterio.open(path, "w", driver="GTiff", width=values.shape[1], height=values.shape[0],
                       count=bands, dtype=str(values.dtype), crs="EPSG:3857",
                       transform=from_origin(origin, 100, 1, 1)) as dst:
        dst.write(data, 1) if bands == 1 else dst.write(data)


def test_transition_changed_focus_matrix_top_and_drift():
    source = np.array([[8, 3, 5], [4, 7, 6], [2, 1, 0]], dtype=np.uint8)
    target = np.array([[5, 4, 5], [3, 5, 5], [4, 1, 0]], dtype=np.uint8)
    codes = progression.transition_code(source, target)
    assert codes.tolist() == [[85, 34, 55], [43, 75, 65], [24, 11, 0]]
    changed = progression.changed_only(source, target)
    assert changed.tolist() == [[85, 34, 0], [43, 75, 65], [24, 0, 0]]
    assert np.where(changed == 85, changed, 0).tolist()[0][0] == 85
    assert np.where(changed == 34, changed, 0).tolist()[0][1] == 34

    summary = progression.comparison_summary(source, target)
    assert len(summary["transition_matrix"]) == 9
    assert all(len(row) == 9 for row in summary["transition_matrix"].values())
    assert summary["transition_matrix"]["8"]["5"] == 1
    assert all(item["source_class"] != item["target_class"]
               for item in summary["top_transitions_by_pixel_count"])
    assert summary["class_drift"]["8"] == {
        "name": progression.CLASS_NAMES[8], "source_pixel_count": 1,
        "target_pixel_count": 0, "delta_pixels": -1,
        "delta_percentage_points": pytest.approx(-100 / 9),
    }
    assert summary["watched_transitions"]["8 -> 5"]["pixel_count"] == 1


def test_alignment_and_invalid_classes_fail_fast(tmp_path):
    first, second = tmp_path / "a.tif", tmp_path / "b.tif"
    write_raster(first, np.zeros((2, 2), dtype=np.uint8))
    write_raster(second, np.zeros((2, 2), dtype=np.uint8), origin=1)
    with pytest.raises(ValueError, match="transform"):
        progression.aligned_stage_rasters([first, second])
    write_raster(second, np.full((2, 2), 9, dtype=np.uint8))
    with pytest.raises(ValueError, match="outside 0..8"):
        progression.aligned_stage_rasters([first, second])


def make_args(tmp_path: Path):
    input_path = tmp_path / "external.tif"
    write_raster(input_path, np.zeros((2, 3), dtype=np.uint8), bands=3)
    models = []
    for name in ("base", "paddy", "water", "road"):
        path = tmp_path / f"{name}.pth"; path.write_bytes(name.encode()); models.append(path)
    parser = progression.build_parser()
    args = parser.parse_args(["--input", str(input_path), "--name", "synthetic",
                              "--base-model", str(models[0]), "--paddy-model", str(models[1]),
                              "--water-model", str(models[2]), "--road-model", str(models[3]),
                              "--output-dir", str(tmp_path / "out"),
                              "--focus-transition", "85", "--focus-transition", "34"])
    progression.validate_args(parser, args)
    return args, models


def test_existing_run_requires_overwrite(tmp_path):
    args, _ = make_args(tmp_path)
    run_dir = args.output_dir / args.name; run_dir.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="--overwrite"):
        progression.process(args, ROOT)
    stale = run_dir / "stale.txt"; stale.write_text("stale")
    progression.prepare_run_directory(run_dir, True)
    assert run_dir.is_dir() and not stale.exists()


def test_mock_pipeline_uses_one_input_and_builds_three_steps(tmp_path, monkeypatch):
    args, models = make_args(tmp_path)
    calls = []

    def fake_run(command):
        calls.append(list(command))
        script = Path(command[1]).name
        directory = Path(command[command.index("--output-dir") + 1])
        if script == "predict_geotiff_tiled.py":
            stage = models.index(Path(command[command.index("--model") + 1]))
            values = np.array([[8, 3, 5], [4, 7, stage]], dtype=np.uint8)
            write_raster(directory / "gsi_rgb_classes.tif", values)
            write_raster(directory / "gsi_rgb_confidence.tif", values.astype(np.float32))
        elif script == "sieve_landcover.py":
            source = Path(command[2])
            shutil.copyfile(source, directory / "classes_raw_sieve_5m2.tif")

    monkeypatch.setattr(progression, "run_command", fake_run)
    manifest = progression.process(args, ROOT)
    prediction_calls = [call for call in calls if Path(call[1]).name == "predict_geotiff_tiled.py"]
    assert len(prediction_calls) == 4
    shared_inputs = {call[2] for call in prediction_calls}
    assert shared_inputs == {str(args.output_dir / "synthetic/input/gsi_rgb.tif")}
    assert manifest["status"] == "complete"
    assert len(manifest["stage_output_paths"]) == 4
    assert len(manifest["transition_output_paths"]) == 3
    summary = json.loads((args.output_dir / "synthetic/progression_summary.json").read_text())
    assert summary["schema_version"] == 1
    assert summary["stage_order"] == ["base", "paddy", "paddy_water", "paddy_water_road"]
    assert len(summary["steps"]) == 3
    for step in manifest["transition_output_paths"].values():
        assert Path(step["changed_only_raw"]).exists()
        assert Path(step["focus"]["raw"]["85"]).name == "focus_85_buildings_to_tree.tif"
        assert Path(step["focus"]["raw"]["34"]).name == "focus_34_pavement_to_road.tif"
    assert progression.sha256_file(args.input) == manifest["input_geotiff"]["sha256"]
