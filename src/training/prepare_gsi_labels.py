"""Prepare partial OEM8 labels from paired GSI ``org`` and ``val`` PNGs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from src.config import CLASS_NAMES

IGNORE_INDEX = 255
DEFAULT_LABEL_COLOR = (255, 0, 0)

CSV_FIELDS = (
    "source_image_id",
    "org_path",
    "val_path",
    "width",
    "height",
    "positive_pixel_count",
    "total_pixel_count",
    "positive_pixel_ratio",
    "is_false_image",
    "non_label_mismatch_count",
    "org_exact_label_color_count",
    "org_sha256",
    "val_sha256",
    "gsi_category",
    "oem_class_id",
    "oem_class_name",
    "capture_date",
    "capture_date_precision",
    "capture_district",
)


def _png_index(directory: Path, name: str) -> dict[str, Path]:
    if not directory.is_dir():
        raise ValueError(f"Required directory does not exist: {directory}")
    index: dict[str, Path] = {}
    normalized: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() != ".png":
            continue
        relative = path.relative_to(directory).as_posix()
        key = relative.casefold()
        if key in normalized:
            raise ValueError(
                f"Duplicate {name} PNG path (case-insensitive): "
                f"{normalized[key]} and {relative}"
            )
        normalized[key] = relative
        index[relative] = path
    return index


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_color(value: str) -> tuple[int, int, int]:
    try:
        color = tuple(int(component.strip()) for component in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("color must be R,G,B integers") from exc
    if len(color) != 3 or any(component < 0 or component > 255 for component in color):
        raise argparse.ArgumentTypeError("color must contain three values from 0 to 255")
    return color  # type: ignore[return-value]


def prepare_gsi_dataset(
    dataset_root: Path | str,
    output_dir: Path | str,
    *,
    gsi_category: str,
    oem_class_id: int,
    label_color: Sequence[int] = DEFAULT_LABEL_COLOR,
) -> dict[str, object]:
    """Create partial label PNGs, an audit CSV, and a dataset manifest.

    All pairs are validated before output is created. Pixels exactly matching
    ``label_color`` in ``val`` receive ``oem_class_id``; every other pixel is
    assigned :data:`IGNORE_INDEX` and is therefore not a negative example.
    """
    root = Path(dataset_root)
    output = Path(output_dir)
    if oem_class_id not in CLASS_NAMES:
        raise ValueError(f"Unknown OEM8 class ID: {oem_class_id}")
    if oem_class_id == 0:
        raise ValueError("OEM8 class 0 is not a positive training class")
    color = tuple(label_color)
    if len(color) != 3 or any(not isinstance(v, int) or v < 0 or v > 255 for v in color):
        raise ValueError("label_color must contain three integers from 0 to 255")

    org_files = _png_index(root / "org", "org")
    val_files = _png_index(root / "val", "val")
    missing_val = sorted(org_files.keys() - val_files.keys())
    missing_org = sorted(val_files.keys() - org_files.keys())
    if missing_val or missing_org:
        details = []
        if missing_val:
            details.append(f"missing from val: {', '.join(missing_val)}")
        if missing_org:
            details.append(f"missing from org: {', '.join(missing_org)}")
        raise ValueError("PNG pairing failed; " + "; ".join(details))
    if not org_files:
        raise ValueError("No PNG pairs found")

    pairs: list[tuple[str, Path, Path, tuple[int, int]]] = []
    for relative in sorted(org_files):
        org_path, val_path = org_files[relative], val_files[relative]
        with Image.open(org_path) as image:
            org_size = image.size
        with Image.open(val_path) as image:
            val_size = image.size
        if org_size != val_size:
            raise ValueError(
                f"Image size mismatch for {relative}: "
                f"org={org_size[0]}x{org_size[1]}, val={val_size[0]}x{val_size[1]}"
            )
        pairs.append((relative, org_path, val_path, org_size))

    labels_dir = output / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    label_rgb = np.asarray(color, dtype=np.uint8)
    for relative, org_path, val_path, (width, height) in pairs:
        with Image.open(org_path) as image:
            org = np.asarray(image.convert("RGB"))
        with Image.open(val_path) as image:
            val = np.asarray(image.convert("RGB"))
        positive = np.all(val == label_rgb, axis=2)
        mismatch = np.any(org != val, axis=2)
        non_label_mismatch = mismatch & ~positive
        label = np.full(positive.shape, IGNORE_INDEX, dtype=np.uint8)
        label[positive] = oem_class_id
        label_path = labels_dir / relative
        label_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(label, mode="L").save(label_path)

        positive_count = int(positive.sum())
        total_count = int(positive.size)
        records.append({
            "source_image_id": Path(relative).with_suffix("").as_posix(),
            "org_path": (Path("org") / relative).as_posix(),
            "val_path": (Path("val") / relative).as_posix(),
            "width": width,
            "height": height,
            "positive_pixel_count": positive_count,
            "total_pixel_count": total_count,
            "positive_pixel_ratio": positive_count / total_count,
            "is_false_image": positive_count == 0,
            "non_label_mismatch_count": int(non_label_mismatch.sum()),
            "org_exact_label_color_count": int(np.all(org == label_rgb, axis=2).sum()),
            "org_sha256": _sha256(org_path),
            "val_sha256": _sha256(val_path),
            "gsi_category": gsi_category,
            "oem_class_id": oem_class_id,
            "oem_class_name": CLASS_NAMES[oem_class_id],
            "capture_date": "",
            "capture_date_precision": "unknown",
            "capture_district": "",
        })
        del org, val, positive, mismatch, non_label_mismatch, label

    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)

    positive_total = sum(int(row["positive_pixel_count"]) for row in records)
    pixel_total = sum(int(row["total_pixel_count"]) for row in records)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "image_count": len(records),
        "positive_pixel_count": positive_total,
        "total_pixel_count": pixel_total,
        "positive_pixel_ratio": positive_total / pixel_total,
        "false_image_count": sum(bool(row["is_false_image"]) for row in records),
        "inspection": {
            "pairing_valid": True,
            "size_mismatch_count": 0,
            "non_label_mismatch_count": sum(
                int(row["non_label_mismatch_count"]) for row in records
            ),
            "org_exact_label_color_count": sum(
                int(row["org_exact_label_color_count"]) for row in records
            ),
        },
        "label_color_rgb": list(color),
        "ignore_index": IGNORE_INDEX,
        "gsi_category": gsi_category,
        "oem_class_id": oem_class_id,
        "oem_class_name": CLASS_NAMES[oem_class_id],
        "audit_csv": csv_path.name,
        "labels_directory": labels_dir.name,
    }
    with (output / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--gsi-category", required=True)
    parser.add_argument(
        "--oem-class-id",
        required=True,
        type=int,
        choices=sorted(class_id for class_id in CLASS_NAMES if class_id != 0),
    )
    parser.add_argument("--label-color", type=_parse_color, default=DEFAULT_LABEL_COLOR)
    args = parser.parse_args()
    prepare_gsi_dataset(
        args.dataset_root,
        args.output_dir,
        gsi_category=args.gsi_category,
        oem_class_id=args.oem_class_id,
        label_color=args.label_color,
    )


if __name__ == "__main__":
    main()
