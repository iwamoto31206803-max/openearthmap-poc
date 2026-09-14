import csv
import json

import numpy as np
from PIL import Image
import pytest

from src.config import CLASS_NAMES
from src.training.prepare_gsi_labels import IGNORE_INDEX, prepare_gsi_dataset


def save_rgb(path, pixels):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(pixels, dtype=np.uint8), mode="RGB").save(path)


def run_prepare(tmp_path, pairs, class_id=5):
    root = tmp_path / "dataset"
    for name, org, val in pairs:
        save_rgb(root / "org" / name, org)
        save_rgb(root / "val" / name, val)
    output = tmp_path / "output"
    manifest = prepare_gsi_dataset(
        root, output, gsi_category="tree", oem_class_id=class_id
    )
    with (output / "audit.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return output, manifest, rows


def test_false_image_has_zero_ratio_and_only_ignore_pixels(tmp_path):
    black = np.zeros((2, 3, 3), dtype=np.uint8)
    output, manifest, rows = run_prepare(tmp_path, [("false.png", black, black)])

    assert rows[0]["positive_pixel_ratio"] == "0.0"
    assert rows[0]["is_false_image"] == "True"
    assert manifest["false_image_count"] == 1
    assert np.all(np.asarray(Image.open(output / "labels/false.png")) == IGNORE_INDEX)


def test_full_and_partial_images_preserve_oem_class_id(tmp_path):
    black = np.zeros((2, 2, 3), dtype=np.uint8)
    red = np.full((2, 2, 3), (255, 0, 0), dtype=np.uint8)
    partial = black.copy()
    partial[0, 1] = (255, 0, 0)
    output, manifest, rows = run_prepare(
        tmp_path,
        [("nested/full.png", black, red), ("partial.png", black, partial)],
        class_id=8,
    )

    assert rows[0]["positive_pixel_ratio"] == "1.0"
    assert rows[1]["positive_pixel_ratio"] == "0.25"
    assert rows[1]["oem_class_id"] == "8"
    assert rows[1]["oem_class_name"] == CLASS_NAMES[8]
    assert manifest["oem_class_id"] == 8
    assert set(np.asarray(Image.open(output / "labels/partial.png")).flat) == {8, 255}


def test_missing_pair_is_an_error(tmp_path):
    root = tmp_path / "dataset"
    save_rgb(root / "org/only.png", np.zeros((1, 1, 3)))
    (root / "val").mkdir()

    with pytest.raises(ValueError, match="missing from val: only.png"):
        prepare_gsi_dataset(root, tmp_path / "out", gsi_category="tree", oem_class_id=5)


def test_size_mismatch_is_an_error(tmp_path):
    root = tmp_path / "dataset"
    save_rgb(root / "org/1.png", np.zeros((1, 2, 3)))
    save_rgb(root / "val/1.png", np.zeros((2, 2, 3)))

    with pytest.raises(ValueError, match="Image size mismatch for 1.png"):
        prepare_gsi_dataset(root, tmp_path / "out", gsi_category="tree", oem_class_id=5)


def test_late_size_mismatch_creates_no_output(tmp_path):
    root = tmp_path / "dataset"
    matching = np.zeros((2, 2, 3), dtype=np.uint8)
    save_rgb(root / "org/a.png", matching)
    save_rgb(root / "val/a.png", matching)
    save_rgb(root / "org/z.png", np.zeros((1, 2, 3)))
    save_rgb(root / "val/z.png", np.zeros((2, 2, 3)))
    output = tmp_path / "out"

    with pytest.raises(ValueError, match="Image size mismatch for z.png"):
        prepare_gsi_dataset(root, output, gsi_category="tree", oem_class_id=5)

    assert not output.exists()


def test_non_label_pixel_difference_is_audited(tmp_path):
    org = np.zeros((1, 2, 3), dtype=np.uint8)
    val = org.copy()
    val[0, 0] = (0, 255, 0)
    output, manifest, rows = run_prepare(tmp_path, [("1.png", org, val)])

    assert rows[0]["non_label_mismatch_count"] == "1"
    assert manifest["inspection"]["non_label_mismatch_count"] == 1
    assert np.all(np.asarray(Image.open(output / "labels/1.png")) == IGNORE_INDEX)
    assert json.loads((output / "manifest.json").read_text())["image_count"] == 1


def test_custom_label_color(tmp_path):
    green = np.full((1, 1, 3), (0, 255, 0), dtype=np.uint8)
    root = tmp_path / "dataset"
    save_rgb(root / "org/1.png", np.zeros((1, 1, 3)))
    save_rgb(root / "val/1.png", green)
    prepare_gsi_dataset(
        root,
        tmp_path / "out",
        gsi_category="water",
        oem_class_id=6,
        label_color=(0, 255, 0),
    )
    assert np.asarray(Image.open(tmp_path / "out/labels/1.png"))[0, 0] == 6
