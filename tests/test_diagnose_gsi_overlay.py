import csv
import json

import numpy as np
from PIL import Image

from src.training.diagnose_gsi_overlay import diagnose_gsi_overlay


def _save(path, pixels):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(pixels, dtype=np.uint8), mode="RGB").save(path)


def test_diagnosis_counts_differences_and_fits_alpha_without_exporting_paths(tmp_path):
    root = tmp_path / "private" / "paddy"
    org = np.array([[[0, 100, 200], [40, 80, 120], [20, 20, 20]]], dtype=np.uint8)
    # Exact alpha=0.5 blend with overlay=(100, 200, 240); final pixel is unchanged.
    val = np.array([[[50, 150, 220], [70, 140, 180], [20, 20, 20]]], dtype=np.uint8)
    _save(root / "org/1.png", org)
    _save(root / "val/1.png", val)

    result = diagnose_gsi_overlay(root, tmp_path / "report", representative_ids=["1"])

    assert result["image_count"] == 1
    assert result["identical_pixel_count"] == 1
    assert result["different_pixel_count"] == 2
    assert result["representative_images"]["1"]["different_pixel_ratio"] == 2 / 3
    assert result["channel_delta_histograms"]["red"] == {"0": 1, "30": 1, "50": 1}
    fit = result["alpha_blend_fit"]
    assert fit["alpha"] == 0.5
    assert np.allclose(fit["overlay_rgb"], [100, 200, 240])
    assert fit["rmse_rgb_levels"] < 1e-6
    serialized = (tmp_path / "report/summary.json").read_text(encoding="utf-8")
    assert str(root) not in serialized
    assert json.loads(serialized)["top_exact_delta_patterns"]
    with (tmp_path / "report/per_image.csv").open(encoding="utf-8", newline="") as stream:
        assert list(csv.DictReader(stream))[0]["image_id"] == "1"


def test_diagnosis_processes_nested_pairs_and_reports_missing_representative(tmp_path):
    root = tmp_path / "dataset"
    pixels = np.zeros((2, 2, 3), dtype=np.uint8)
    _save(root / "org/nested/a.png", pixels)
    _save(root / "val/nested/a.png", pixels)

    result = diagnose_gsi_overlay(
        root, tmp_path / "report", representative_ids=["nested/a", "missing"]
    )

    assert result["identical_pixel_ratio"] == 1.0
    assert result["alpha_blend_fit"] is None
    assert result["missing_representative_ids"] == ["missing"]
