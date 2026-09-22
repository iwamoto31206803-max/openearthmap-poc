import csv
import hashlib
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
import torch

from src.evaluation.gt54_inference import (
    MODEL_ORDER, inference_config, preflight_checkpoints, run_gt54_inference,
    validate_cross_model_identity, validate_prediction,
)
from src.evaluation.gt54_preflight import inspect_item, load_manifest


def _raster(path, data, *, transform=None, nodata=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    transform = transform or from_origin(10, 20, 1, 1)
    count, height, width = data.shape
    with rasterio.open(path, "w", driver="GTiff", width=width, height=height,
                       count=count, dtype=data.dtype, crs="EPSG:3857", transform=transform,
                       nodata=nodata) as dst:
        dst.write(data)


def _dataset(tmp_path, names=("ValArea_002", "ValArea_001")):
    root = tmp_path / "dataset"
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(("valarea", "year", "region"))
        for name in names:
            writer.writerow((name, 2020, "region"))
            _raster(root / "rgb_images" / f"{name}.tif",
                    np.arange(3 * 4 * 5, dtype=np.uint8).reshape(3, 4, 5))
            _raster(root / "labels" / f"{name}.tif", np.zeros((1, 4, 5), dtype=np.uint8))
    return root, manifest


def _checkpoints(tmp_path):
    paths, hashes = {}, {}
    for model_id in MODEL_ORDER:
        path = tmp_path / f"{model_id}.pth"; path.write_bytes(model_id.encode())
        paths[model_id] = path
        hashes[model_id] = hashlib.sha256(model_id.encode()).hexdigest()
    return paths, hashes


def _loader(path, device):
    return torch.nn.Identity()


def _predictor(rgb, model, overlap):
    values = (rgb[:, :, 0] % 9).astype(np.uint8)
    return values, np.ones(values.shape, dtype=np.float32)


def test_checkpoint_hash_match_and_model_order(tmp_path):
    paths, hashes = _checkpoints(tmp_path)
    result = preflight_checkpoints(paths, expected=hashes, loader=_loader)
    assert tuple(item.model_id for item in result) == MODEL_ORDER


def test_checkpoint_hash_mismatch_fails_before_load(tmp_path):
    paths, hashes = _checkpoints(tmp_path); hashes["C"] = "0" * 64
    loaded = []
    with pytest.raises(ValueError, match="C SHA256 mismatch"):
        preflight_checkpoints(paths, expected=hashes,
                              loader=lambda path, device: loaded.append(path))
    assert [path.name for path in loaded] == ["A.pth", "B.pth"]


def test_missing_checkpoint(tmp_path):
    paths, hashes = _checkpoints(tmp_path); paths["D"].unlink()
    with pytest.raises(FileNotFoundError, match="checkpoint D"):
        preflight_checkpoints(paths, expected=hashes, loader=_loader)


def test_missing_rgb(tmp_path):
    root, manifest = _dataset(tmp_path, ("ValArea_001",))
    (root / "rgb_images" / "ValArea_001.tif").unlink()
    item = load_manifest(manifest, expected_items=1, expected_regions=1)[0]
    with pytest.raises(FileNotFoundError, match="pair missing"):
        inspect_item(item, root)


def test_rgb_gt_grid_mismatch(tmp_path):
    root, manifest = _dataset(tmp_path, ("ValArea_001",))
    _raster(root / "labels" / "ValArea_001.tif", np.zeros((1, 4, 5), dtype=np.uint8),
            transform=from_origin(11, 20, 1, 1))
    item = load_manifest(manifest, expected_items=1, expected_regions=1)[0]
    with pytest.raises(ValueError, match="alignment mismatch"):
        inspect_item(item, root)


def test_manifest_numeric_order(tmp_path):
    _, manifest = _dataset(tmp_path)
    assert [x.valarea for x in load_manifest(manifest, expected_items=2, expected_regions=1)] == [
        "ValArea_001", "ValArea_002"]


def test_inference_config_is_stable():
    assert inference_config(128, "abc") == inference_config(128, "abc")
    assert inference_config(128, "abc")["stride"] == 384


def test_cross_model_rgb_identity_invariant():
    rows = [{"model_id": m, "valarea": "ValArea_001", "rgb_sha256": "same", "width": 2,
             "height": 2, "crs": "x", "transform": "t", "bounds": "b"} for m in MODEL_ORDER]
    validate_cross_model_identity(rows)
    rows[-1]["rgb_sha256"] = "stale"
    with pytest.raises(RuntimeError, match="identity invariant"):
        validate_cross_model_identity(rows)


def test_prediction_contract_and_invalid_class(tmp_path):
    root, manifest = _dataset(tmp_path, ("ValArea_001",))
    source = inspect_item(load_manifest(manifest, expected_items=1, expected_regions=1)[0], root)
    good = tmp_path / "good.tif"; _raster(good, np.full((1, 4, 5), 8, dtype=np.uint8))
    assert validate_prediction(good, source) == (8, 8)
    with rasterio.open(good) as src:
        assert src.count == 1 and src.dtypes == ("uint8",) and src.nodata is None
        assert (src.width, src.height, src.crs, src.transform, src.bounds) == (
            source.width, source.height, source.crs, source.transform, source.bounds)
    bad = tmp_path / "bad.tif"; _raster(bad, np.full((1, 4, 5), 9, dtype=np.uint8))
    with pytest.raises(ValueError, match="outside 0..8"):
        validate_prediction(bad, source)


def test_existing_output_safety_happens_before_checkpoint_work(tmp_path):
    root, manifest = _dataset(tmp_path, ("ValArea_001",)); paths, hashes = _checkpoints(tmp_path)
    output = tmp_path / "out"; output.mkdir(); (output / "old").write_text("x")
    with pytest.raises(FileExistsError):
        run_gt54_inference(manifest=manifest, dataset_root=root, checkpoint_paths=paths,
                           output_root=output, expected_items=1, expected_regions=1,
                           expected_hashes=hashes, loader=_loader, predictor=_predictor)


def test_end_to_end_order_contract_and_determinism(tmp_path):
    root, manifest = _dataset(tmp_path); paths, hashes = _checkpoints(tmp_path)
    kwargs = dict(manifest=manifest, dataset_root=root, checkpoint_paths=paths,
                  expected_items=2, expected_regions=1, expected_hashes=hashes,
                  loader=_loader, predictor=_predictor)
    rows1 = run_gt54_inference(output_root=tmp_path / "one", **kwargs)
    rows2 = run_gt54_inference(output_root=tmp_path / "two", **kwargs)
    assert [(r["model_id"], r["valarea"]) for r in rows1] == [
        (m, v) for m in MODEL_ORDER for v in ("ValArea_001", "ValArea_002")]
    assert [r["prediction_sha256"] for r in rows1] == [r["prediction_sha256"] for r in rows2]
    for row in rows1:
        assert validate_prediction(Path(row["prediction_path"]),
            inspect_item(next(x for x in load_manifest(manifest, expected_items=2, expected_regions=1)
                              if x.valarea == row["valarea"]), root))[1] <= 8
