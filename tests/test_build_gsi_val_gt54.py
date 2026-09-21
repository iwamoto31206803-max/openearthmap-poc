import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.build_gsi_val_gt54 import (
    Item, build_item, read_items, source_for_year, tiles_for_grid, validate_pair,
)


def raster(path, data, transform=from_origin(15500000, 4300000, 1, 1), crs="EPSG:3857"):
    path.parent.mkdir(parents=True, exist_ok=True)
    count, height, width = data.shape
    with rasterio.open(path, "w", driver="GTiff", count=count, width=width, height=height,
                       dtype=data.dtype, transform=transform, crs=crs) as dst:
        dst.write(data)


def test_manifest_parsing_and_year_resolution(tmp_path):
    manifest = tmp_path / "input.csv"
    manifest.write_text("valarea,year,region\nValArea_011,2019,Tokyo\n", encoding="utf-8")
    assert read_items(manifest) == [Item("ValArea_011", 2019, "Tokyo")]
    assert source_for_year(2019).endswith("nendophoto2019/{z}/{x}/{y}.png")
    with pytest.raises(ValueError, match="No verified"):
        source_for_year(2016)


@pytest.mark.parametrize("contents", [
    "valarea,year\nValArea_011,2019\n",
    "valarea,year,region\nValArea_011,nope,Tokyo\n",
    "valarea,year,region\nwrong,2019,Tokyo\n",
])
def test_malformed_csv(contents, tmp_path):
    path = tmp_path / "bad.csv"; path.write_text(contents)
    with pytest.raises(ValueError):
        read_items(path)


def test_target_grid_tile_selection():
    tiles = tiles_for_grid(rasterio.crs.CRS.from_epsg(3857),
                           (15500000, 4299990, 15500010, 4300000), 18)
    assert tiles and len(set(tiles)) == len(tiles)


def test_build_alignment_resume_and_overwrite_safety(tmp_path):
    gt_root, output = tmp_path / "gt", tmp_path / "out"
    gt = gt_root / "ValArea_011_gt_georef.tif"
    raster(gt, np.ones((1, 8, 9), dtype=np.uint8))
    calls = []

    def fetch(url, timeout, retries):
        calls.append(url)
        return np.full((256, 256, 3), 80, dtype=np.uint8)

    item = Item("ValArea_011", 2019, "Tokyo")
    row = build_item(item, gt_root, output, fetch=fetch)
    assert row["alignment_ok"] == "True" and calls
    rgb, label = output / "rgb_images/ValArea_011.tif", output / "labels/ValArea_011.tif"
    assert validate_pair(rgb, label)["width"] == "9"
    mtime = rgb.stat().st_mtime_ns
    calls.clear()
    build_item(item, gt_root, output, fetch=fetch)
    assert calls == [] and rgb.stat().st_mtime_ns == mtime


def test_missing_gt(tmp_path):
    with pytest.raises(FileNotFoundError, match="GT not found"):
        build_item(Item("ValArea_008", 2018, "Fukushima"), tmp_path, tmp_path / "out")


def test_grid_mismatch(tmp_path):
    rgb, label = tmp_path / "rgb.tif", tmp_path / "label.tif"
    raster(rgb, np.ones((3, 4, 4), dtype=np.uint8))
    raster(label, np.ones((1, 4, 4), dtype=np.uint8), from_origin(15500001, 4300000, 1, 1))
    with pytest.raises(ValueError, match="pixel-grid mismatch"):
        validate_pair(rgb, label)


def test_partial_cleanup_on_download_failure(tmp_path):
    gt_root, output = tmp_path / "gt", tmp_path / "out"
    raster(gt_root / "ValArea_011_gt_georef.tif", np.ones((1, 2, 2), dtype=np.uint8))

    def fail(*args):
        raise RuntimeError("missing tile")

    with pytest.raises(RuntimeError, match="missing tile"):
        build_item(Item("ValArea_011", 2019, "Tokyo"), gt_root, output, fetch=fail)
    assert not list(output.rglob("*.partial"))
    assert not list(output.rglob("*.tif"))
