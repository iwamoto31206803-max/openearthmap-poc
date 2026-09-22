"""Tests for GT54 Step-4A formal inference orchestration."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from tools.evaluation import run_gt54_inference


def make_fixture(tmp_path: Path, count: int, *, rgb_dtype: str = "uint8"):
    root = tmp_path / "dataset"
    (root / "rgb_images").mkdir(parents=True)
    (root / "labels").mkdir()
    formal_rows, local_rows = [], []
    # Reverse lexical/numeric order to ensure selection uses manifest loader's
    # numeric ValArea ordering rather than the CSV's order.
    numbers = list(range(1, count + 1))[::-1]
    for number in numbers:
        valarea = f"ValArea_{number:03d}"
        region = f"Region_{number % 8}"
        transform = from_origin(number * 10, 4, 1, 1)
        with rasterio.open(root / "rgb_images" / f"{valarea}.tif", "w", driver="GTiff",
                           count=3, width=4, height=4, dtype=rgb_dtype,
                           crs="EPSG:3857", transform=transform) as output:
            output.write(np.ones((3, 4, 4), dtype=rgb_dtype))
        with rasterio.open(root / "labels" / f"{valarea}.tif", "w", driver="GTiff",
                           count=1, width=4, height=4, dtype="uint8",
                           crs="EPSG:3857", transform=transform) as output:
            output.write(np.ones((1, 4, 4), dtype=np.uint8))
        formal_rows.append((valarea, 2019, region))
        local_rows.append((valarea, region, 2019, "True", ""))
    manifest = tmp_path / "formal.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(("valarea", "year", "region")); writer.writerows(formal_rows)
    with (root / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("valarea", "region", "gsi_year", "alignment_ok", "error"))
        writer.writerows(local_rows)
    checkpoints = {}
    for model_id in "ABCD":
        path = tmp_path / f"{model_id}.pth"; path.write_bytes(model_id.encode())
        checkpoints[model_id] = path
    return manifest, root, checkpoints


def args_for(manifest, root, checkpoints, output, count, smoke_items=None):
    argv = ["--manifest", str(manifest), "--dataset-root", str(root),
            "--output-dir", str(output), "--expected-items", str(count),
            "--expected-regions", str(min(count, 8))]
    for model_id, path in checkpoints.items():
        argv += [f"--checkpoint-{model_id.lower()}", str(path)]
    if smoke_items is not None:
        argv += ["--smoke-items", str(smoke_items)]
    return run_gt54_inference.build_parser().parse_args(argv)


def patch_inference(monkeypatch, checkpoints, calls):
    monkeypatch.setattr(run_gt54_inference, "CHECKPOINT_SHA256", {
        model_id: hashlib.sha256(path.read_bytes()).hexdigest()
        for model_id, path in checkpoints.items()
    })
    monkeypatch.setattr(run_gt54_inference, "build_model",
                        lambda path, device="cpu": path.stem)

    def predict(rgb, model, overlap):
        calls.append((model, rgb.dtype, overlap))
        return np.ones(rgb.shape[:2], dtype=np.uint8), np.ones(rgb.shape[:2], dtype=np.float32)

    monkeypatch.setattr(run_gt54_inference, "predict_tiled", predict)


def test_uint8_smoke_selects_first_numeric_valarea_for_all_models(tmp_path, monkeypatch):
    manifest, root, checkpoints = make_fixture(tmp_path, 54)
    calls = []; patch_inference(monkeypatch, checkpoints, calls)
    output = tmp_path / "output"
    config = run_gt54_inference.run(args_for(manifest, root, checkpoints, output, 54, 1))
    assert len(calls) == 4
    assert all(dtype == np.dtype("uint8") for _, dtype, _ in calls)
    assert config["smoke_subset"] is True
    assert config["selected_valareas"] == ["ValArea_001"]
    assert sorted(path.relative_to(output).as_posix() for path in output.glob("?/*.tif")) == [
        f"{model}/ValArea_001.tif" for model in "ABCD"
    ]
    saved = json.loads((output / "inference_config.json").read_text())
    assert saved["run_kind"] == "smoke_subset"
    assert saved["selected_valareas"] == ["ValArea_001"]
    assert "SMOKE" in (output / "inference_qc.csv").read_text()


def test_uint16_rgb_fails_before_predictor_is_called(tmp_path, monkeypatch):
    manifest, root, checkpoints = make_fixture(tmp_path, 1, rgb_dtype="uint16")
    calls = []; patch_inference(monkeypatch, checkpoints, calls)
    args = args_for(manifest, root, checkpoints, tmp_path / "output", 1)
    try:
        run_gt54_inference.run(args)
    except ValueError as exc:
        assert "must all have dtype uint8" in str(exc)
    else:
        raise AssertionError("uint16 formal RGB was accepted")
    assert calls == []


def test_without_smoke_option_processes_entire_inventory(tmp_path, monkeypatch):
    manifest, root, checkpoints = make_fixture(tmp_path, 8)
    calls = []; patch_inference(monkeypatch, checkpoints, calls)
    output = tmp_path / "output"
    config = run_gt54_inference.run(args_for(manifest, root, checkpoints, output, 8))
    assert len(calls) == 8 * 4
    assert config["run_kind"] == "formal_full"
    assert config["smoke_subset"] is False
    assert len(config["selected_valareas"]) == 8
