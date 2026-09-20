"""Synthetic-only tests for the Base/FT comparison CLI."""

from __future__ import annotations

import json
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location("compare_base_ft", ROOT / "compare_base_ft.py")
assert SPEC is not None and SPEC.loader is not None
comparison = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = comparison
SPEC.loader.exec_module(comparison)


def write_raster(path: Path, values: np.ndarray, *, x_origin: float = 0) -> None:
    with rasterio.open(
        path, "w", driver="GTiff", width=values.shape[1], height=values.shape[0],
        count=1, dtype=str(values.dtype), crs="EPSG:3857",
        transform=from_origin(x_origin, 100, 1, 1),
    ) as dst:
        dst.write(values, 1)


def write_existing_input(path: Path, *, lat=35.0, lon=140.0, zoom=18,
                         width=256, height=256, crs="EPSG:3857") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", width=width, height=height, count=3,
        dtype="uint8", crs=crs, transform=from_origin(0, height, 1, 1),
    ) as dst:
        dst.write(np.zeros((3, height, width), dtype=np.uint8))
        dst.update_tags(center_lat=str(lat), center_lon=str(lon), zoom=str(zoom))


def model_files(tmp_path: Path) -> tuple[Path, Path]:
    base, ft = tmp_path / "base.pth", tmp_path / "ft.pth"
    base.write_bytes(b"synthetic base")
    ft.write_bytes(b"synthetic ft")
    return base, ft


def test_alignment_check_accepts_equal_and_rejects_transform(tmp_path):
    values = np.array([[3, 4]], dtype=np.uint8)
    base, ft = tmp_path / "base.tif", tmp_path / "ft.tif"
    write_raster(base, values)
    write_raster(ft, values)
    actual_base, actual_ft, profile = comparison.aligned_rasters(base, ft)
    np.testing.assert_array_equal(actual_base, values)
    np.testing.assert_array_equal(actual_ft, values)
    assert profile["width"] == 2

    write_raster(ft, values, x_origin=1)
    with pytest.raises(ValueError, match="transform"):
        comparison.aligned_rasters(base, ft)


def test_transition_and_road_focused_codes():
    base = np.array([[3, 4, 4, 7, 4, 2]], dtype=np.uint8)
    ft = np.array([[4, 4, 3, 4, 8, 2]], dtype=np.uint8)
    np.testing.assert_array_equal(
        comparison.transition_code(base, ft),
        np.array([[34, 44, 43, 74, 48, 22]], dtype=np.uint16),
    )
    np.testing.assert_array_equal(
        comparison.road_change_map(base, ft),
        np.array([[1, 2, 3, 4, 5, 0]], dtype=np.uint8),
    )


def test_transition_matrix_and_summary_json(tmp_path):
    base = np.array([[3, 4, 4], [7, 6, 5], [8, 0, 1]], dtype=np.uint8)
    ft = np.array([[4, 4, 3], [4, 4, 4], [4, 4, 1]], dtype=np.uint8)
    summary = comparison.comparison_summary(base, ft)
    assert summary["total_pixel_count"] == 9
    assert summary["changed_pixel_count"] == 7
    assert summary["transition_matrix"]["3"]["4"] == 1
    assert summary["transition_matrix"]["1"]["1"] == 1
    selected = summary["selected_transition_counts"]
    assert selected["7 -> 4"] == selected["6 -> 4"] == selected["5 -> 4"] == 1
    assert selected["8 -> 4"] == selected["all other -> 4"] == 1
    assert summary["road_focused_counts"] == {
        "Pavement -> Road": 1, "Road -> Road": 1, "Road -> Pavement": 1,
        "Other -> Road": 5, "Road -> Other": 0,
    }
    output = tmp_path / "summary.json"
    output.write_text(json.dumps({"raw": summary, "sieve": summary}), encoding="utf-8")
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["raw"]["base_class_distribution"]["4"]["pixel_count"] == 2
    assert loaded["sieve"]["changed_pixel_percent"] == pytest.approx(700 / 9)


def test_single_site_validation_and_mutual_exclusion(tmp_path):
    base, ft = model_files(tmp_path)
    parser = comparison.build_parser()
    args = parser.parse_args(["--name", "synthetic_01", "--lat", "35", "--lon", "140",
                              "--base-model", str(base), "--fine-tuned-model", str(ft)])
    assert comparison.validate_args(parser, args) == [comparison.Site("synthetic_01", 35, 140)]

    csv_path = tmp_path / "sites.csv"
    csv_path.write_text("name,lat,lon\na,35,140\n", encoding="utf-8")
    args = parser.parse_args(["--sites", str(csv_path), "--lat", "35",
                              "--base-model", str(base), "--fine-tuned-model", str(ft)])
    with pytest.raises(SystemExit):
        comparison.validate_args(parser, args)


def test_existing_site_requires_overwrite(tmp_path):
    base, ft = model_files(tmp_path)
    output_dir = tmp_path / "comparison"
    (output_dir / "synthetic").mkdir(parents=True)
    parser = comparison.build_parser()
    args = parser.parse_args([
        "--name", "synthetic", "--lat", "35", "--lon", "140",
        "--base-model", str(base), "--fine-tuned-model", str(ft),
        "--output-dir", str(output_dir),
    ])
    with pytest.raises(FileExistsError, match="--overwrite"):
        comparison.process_site(comparison.Site("synthetic", 35, 140), args, ROOT)


def test_overwrite_reuses_only_matching_input_and_cleans_outputs(tmp_path):
    run_dir = tmp_path / "comparison" / "synthetic"
    input_path = run_dir / "input" / "gsi_rgb.tif"
    write_existing_input(input_path)
    original_hash = comparison.sha256_file(input_path)
    for relative in (
        "input/obsolete.txt", "base/landcover_raw.gpkg",
        "fine_tuned/old.tif", "diff/old.json", "manifest.json",
    ):
        path = run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stale", encoding="utf-8")

    reused = comparison.prepare_site_directory(
        run_dir, True, comparison.Site("synthetic", 35, 140), 18, 1
    )
    assert reused is True
    assert comparison.sha256_file(input_path) == original_hash
    assert list((run_dir / "input").iterdir()) == [input_path]
    assert not (run_dir / "base").exists()
    assert not (run_dir / "fine_tuned").exists()
    assert not (run_dir / "diff").exists()
    assert not (run_dir / "manifest.json").exists()


@pytest.mark.parametrize(
    "overrides, requested, mismatch",
    [
        ({"lat": 35.1}, (35.0, 140.0, 18, 1), "center_lat"),
        ({"lon": 140.1}, (35.0, 140.0, 18, 1), "center_lon"),
        ({"zoom": 17}, (35.0, 140.0, 18, 1), "zoom"),
        ({"width": 255}, (35.0, 140.0, 18, 1), "width"),
        ({"height": 255}, (35.0, 140.0, 18, 1), "height"),
    ],
)
def test_existing_input_mismatch_is_rejected(tmp_path, overrides, requested, mismatch):
    input_path = tmp_path / "input" / "gsi_rgb.tif"
    write_existing_input(input_path, **overrides)
    lat, lon, zoom, tiles = requested
    with pytest.raises(ValueError, match=mismatch):
        comparison.validate_existing_input(
            input_path, comparison.Site("synthetic", lat, lon), zoom, tiles
        )


@pytest.mark.parametrize(
    "content, message",
    [("site,lat,lon\na,35,140\n", "header"),
     ("name,lat,lon\na,not-a-number,140\n", "row 2"),
     ("name,lat,lon\na,35,140\na,36,141\n", "unique")],
)
def test_batch_csv_schema_validation(tmp_path, content, message):
    path = tmp_path / "sites.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        comparison.load_sites_csv(path)


def test_cli_help_and_existing_run_poc_regression():
    root = ROOT
    for script in ("compare_base_ft.py", "run_poc.py"):
        result = subprocess.run(
            [sys.executable, str(root / script), "--help"], text=True,
            capture_output=True, check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "--lat" in result.stdout


def test_mocked_single_site_pipeline_downloads_once_and_reuses_input(tmp_path, monkeypatch):
    base_model, ft_model = model_files(tmp_path)
    output_dir = tmp_path / "comparison"
    parser = comparison.build_parser()
    args = parser.parse_args([
        "--name", "synthetic", "--lat", "35", "--lon", "140",
        "--base-model", str(base_model), "--fine-tuned-model", str(ft_model),
        "--output-dir", str(output_dir), "--skip-vector",
    ])
    site = comparison.validate_args(parser, args)[0]
    calls = []

    def fake_run(command):
        calls.append(list(command))
        script = Path(command[1]).name
        if script == "download_gsi_geotiff.py":
            output = Path(command[command.index("--output") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            rgb = np.zeros((3, 2, 3), dtype=np.uint8)
            with rasterio.open(
                output, "w", driver="GTiff", width=3, height=2, count=3,
                dtype="uint8", crs="EPSG:3857", transform=from_origin(0, 2, 1, 1),
            ) as dst:
                dst.write(rgb)
        elif script == "predict_geotiff_tiled.py":
            image = Path(command[2])
            directory = Path(command[command.index("--output-dir") + 1])
            model = Path(command[command.index("--model") + 1])
            values = np.full((2, 3), 3 if model == base_model else 4, dtype=np.uint8)
            write_raster(directory / f"{image.stem}_classes.tif", values)
            write_raster(directory / f"{image.stem}_confidence.tif", values.astype(np.float32))
        elif script == "sieve_landcover.py":
            source = Path(command[2])
            directory = Path(command[command.index("--output-dir") + 1])
            shutil.copyfile(source, directory / "gsi_rgb_classes_sieve_5m2.tif")

    monkeypatch.setattr(comparison, "run_command", fake_run)
    result = comparison.process_site(site, args, ROOT)
    assert result["status"] == "complete"
    downloads = [call for call in calls if Path(call[1]).name == "download_gsi_geotiff.py"]
    predictions = [call for call in calls if Path(call[1]).name == "predict_geotiff_tiled.py"]
    assert len(downloads) == 1
    assert len(predictions) == 2
    assert predictions[0][2] == predictions[1][2] == str(output_dir / "synthetic/input/gsi_rgb.tif")
    manifest = json.loads((output_dir / "synthetic/manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["input_geotiff"]["sha256"] == comparison.sha256_file(
        output_dir / "synthetic/input/gsi_rgb.tif"
    )
    summary = json.loads((output_dir / "synthetic/diff/summary.json").read_text(encoding="utf-8"))
    assert summary["raw"]["selected_transition_counts"]["3 -> 4"] == 6
