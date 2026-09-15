"""Diagnose how paired GSI ``org``/``val`` PNGs encode an overlay.

This command is deliberately read-only with respect to the dataset.  It loads one
pair at a time and writes only aggregate statistics (never pixels or input paths).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


DELTA_MIN = -255
DELTA_MAX = 255
MAGNITUDE_MAX = math.ceil(math.sqrt(3 * 255**2))
CSV_FIELDS = (
    "image_id",
    "width",
    "height",
    "pixel_count",
    "identical_pixel_count",
    "identical_pixel_ratio",
    "different_pixel_count",
    "different_pixel_ratio",
)


def _png_index(directory: Path, name: str) -> dict[str, Path]:
    if not directory.is_dir():
        raise ValueError(f"Required directory does not exist: {directory}")
    result: dict[str, Path] = {}
    folded: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() != ".png":
            continue
        relative = path.relative_to(directory).as_posix()
        key = relative.casefold()
        if key in folded:
            raise ValueError(
                f"Duplicate {name} PNG path (case-insensitive): "
                f"{folded[key]} and {relative}"
            )
        folded[key] = relative
        result[relative] = path
    return result


def _nonzero_histogram(histogram: np.ndarray) -> dict[str, int]:
    return {str(index): int(value) for index, value in enumerate(histogram) if value}


def _signed_histogram(histogram: np.ndarray) -> dict[str, int]:
    return {
        str(index + DELTA_MIN): int(value)
        for index, value in enumerate(histogram)
        if value
    }


def _otsu_threshold(histogram: np.ndarray) -> dict[str, float | int] | None:
    """Return Otsu's split and between-class variance for an integer histogram."""
    values = np.arange(histogram.size, dtype=np.float64)
    total = int(histogram.sum())
    if total == 0 or np.count_nonzero(histogram) < 2:
        return None
    weighted_total = float(np.dot(values, histogram))
    left_count = 0
    left_weight = 0.0
    best_threshold = 0
    best_variance = -1.0
    for threshold in range(histogram.size - 1):
        left_count += int(histogram[threshold])
        left_weight += threshold * int(histogram[threshold])
        right_count = total - left_count
        if left_count == 0 or right_count == 0:
            continue
        mean_left = left_weight / left_count
        mean_right = (weighted_total - left_weight) / right_count
        variance = left_count * right_count * (mean_left - mean_right) ** 2
        if variance > best_variance:
            best_threshold = threshold
            best_variance = variance
    return {"threshold": best_threshold, "between_class_variance": best_variance}


def _mode_analysis(histogram: np.ndarray) -> dict[str, object]:
    """Describe strong local modes and the lowest bin between the top two."""
    peaks = []
    for index, count in enumerate(histogram):
        left = histogram[index - 1] if index else -1
        right = histogram[index + 1] if index + 1 < histogram.size else -1
        if count and count >= left and count >= right:
            peaks.append((index, int(count)))
    peaks.sort(key=lambda item: (-item[1], item[0]))
    strongest = peaks[:5]
    valley = None
    if len(strongest) >= 2:
        start, end = sorted((strongest[0][0], strongest[1][0]))
        if end - start > 1:
            valley_bin = start + 1 + int(np.argmin(histogram[start + 1 : end]))
            valley = {"bin": valley_bin, "pixel_count": int(histogram[valley_bin])}
    return {
        "strongest_local_peaks": [
            {"bin": index, "pixel_count": count} for index, count in strongest
        ],
        "valley_between_two_strongest_peaks": valley,
        "note": "A valley is evidence to inspect, not proof of class separability.",
    }


def _fit_alpha_blend(stats: dict[str, np.ndarray | int]) -> dict[str, object] | None:
    count = int(stats["count"])
    if count < 2:
        return None
    sum_x = np.asarray(stats["sum_x"], dtype=np.float64)
    sum_y = np.asarray(stats["sum_y"], dtype=np.float64)
    sum_xx = np.asarray(stats["sum_xx"], dtype=np.float64)
    sum_xy = np.asarray(stats["sum_xy"], dtype=np.float64)
    denominator = float(np.sum(sum_xx - sum_x * sum_x / count))
    if denominator <= 0:
        return None
    slope = float(np.sum(sum_xy - sum_x * sum_y / count) / denominator)
    intercept = (sum_y - slope * sum_x) / count
    alpha = 1.0 - slope
    overlay = intercept / alpha if abs(alpha) > 1e-12 else None
    sum_yy = np.asarray(stats["sum_yy"], dtype=np.float64)
    squared_error = np.sum(
        sum_yy
        + slope**2 * sum_xx
        + count * intercept**2
        - 2 * slope * sum_xy
        - 2 * intercept * sum_y
        + 2 * slope * intercept * sum_x
    )
    rmse = math.sqrt(max(0.0, float(squared_error)) / (count * 3))
    plausible = bool(
        overlay is not None
        and 0 < alpha <= 1
        and np.all((overlay >= 0) & (overlay <= 255))
    )
    return {
        "equation": "val = alpha * overlay_rgb + (1 - alpha) * org",
        "fitted_different_pixels": count,
        "alpha": alpha,
        "overlay_rgb": None if overlay is None else [float(value) for value in overlay],
        "channel_intercepts": [float(value) for value in intercept],
        "rmse_rgb_levels": rmse,
        "parameters_in_physical_range": plausible,
    }


def _iter_pairs(root: Path) -> Iterable[tuple[str, Path, Path]]:
    org = _png_index(root / "org", "org")
    val = _png_index(root / "val", "val")
    missing_val = sorted(org.keys() - val.keys())
    missing_org = sorted(val.keys() - org.keys())
    if missing_val or missing_org:
        details = []
        if missing_val:
            details.append(f"missing from val: {', '.join(missing_val)}")
        if missing_org:
            details.append(f"missing from org: {', '.join(missing_org)}")
        raise ValueError("PNG pairing failed; " + "; ".join(details))
    if not org:
        raise ValueError("No PNG pairs found")
    for relative in sorted(org):
        yield relative, org[relative], val[relative]


def diagnose_gsi_overlay(
    dataset_root: Path | str,
    output_dir: Path | str,
    *,
    representative_ids: Iterable[str] = ("1", "1300", "2600"),
    top_patterns: int = 20,
) -> dict[str, object]:
    """Stream through PNG pairs and write aggregate JSON plus image-level CSV."""
    if top_patterns < 1:
        raise ValueError("top_patterns must be positive")
    root, output = Path(dataset_root), Path(output_dir)
    pairs = list(_iter_pairs(root))  # paths only; image arrays are never retained
    requested = {
        value.replace("\\", "/").removesuffix(".png")
        for value in representative_ids
    }
    delta_hist = np.zeros((3, DELTA_MAX - DELTA_MIN + 1), dtype=np.int64)
    magnitude_hist = np.zeros(MAGNITUDE_MAX + 1, dtype=np.int64)
    pattern_counts: Counter[tuple[int, int, int]] = Counter()
    blend: dict[str, np.ndarray | int] = {
        "count": 0,
        "sum_x": np.zeros(3),
        "sum_y": np.zeros(3),
        "sum_xx": np.zeros(3),
        "sum_xy": np.zeros(3),
        "sum_yy": np.zeros(3),
    }
    rows: list[dict[str, object]] = []
    representative: dict[str, object] = {}
    for relative, org_path, val_path in pairs:
        with Image.open(org_path) as image:
            org = np.asarray(image.convert("RGB"), dtype=np.uint8)
        with Image.open(val_path) as image:
            val = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if org.shape != val.shape:
            raise ValueError(
                f"Image size mismatch for {relative}: org={org.shape[1]}x{org.shape[0]}, "
                f"val={val.shape[1]}x{val.shape[0]}"
            )
        delta = val.astype(np.int16) - org.astype(np.int16)
        different = np.any(delta != 0, axis=2)
        different_count = int(different.sum())
        pixel_count = int(different.size)
        for channel in range(3):
            delta_hist[channel] += np.bincount(
                delta[:, :, channel].ravel() - DELTA_MIN,
                minlength=delta_hist.shape[1],
            )
        squared = np.sum(delta.astype(np.int32) ** 2, axis=2)
        magnitude_bins = np.rint(np.sqrt(squared)).astype(np.int16)
        magnitude_hist += np.bincount(
            magnitude_bins.ravel(), minlength=magnitude_hist.size
        )
        if different_count:
            x = org[different].astype(np.float64)
            y = val[different].astype(np.float64)
            blend["count"] = int(blend["count"]) + different_count
            for key, values in (
                ("sum_x", x), ("sum_y", y), ("sum_xx", x * x),
                ("sum_xy", x * y), ("sum_yy", y * y),
            ):
                blend[key] = np.asarray(blend[key]) + values.sum(axis=0)
            unique, counts = np.unique(
                delta[different].reshape(-1, 3), axis=0, return_counts=True
            )
            pattern_counts.update(
                {tuple(map(int, item)): int(count) for item, count in zip(unique, counts)}
            )
        image_id = Path(relative).with_suffix("").as_posix()
        row = {
            "image_id": image_id,
            "width": org.shape[1],
            "height": org.shape[0],
            "pixel_count": pixel_count,
            "identical_pixel_count": pixel_count - different_count,
            "identical_pixel_ratio": (pixel_count - different_count) / pixel_count,
            "different_pixel_count": different_count,
            "different_pixel_ratio": different_count / pixel_count,
        }
        rows.append(row)
        if image_id in requested:
            representative[image_id] = row.copy()
        del org, val, delta, different, squared, magnitude_bins

    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_image.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    total = sum(int(row["pixel_count"]) for row in rows)
    different_total = sum(int(row["different_pixel_count"]) for row in rows)
    nonzero_magnitude = magnitude_hist.copy()
    nonzero_magnitude[0] = 0
    result: dict[str, object] = {
        "schema_version": 1,
        "method": {
            "delta": "signed int16 val - org, exact RGB values",
            "magnitude": "round(sqrt(delta_R^2 + delta_G^2 + delta_B^2))",
            "alpha_fit": "shared least-squares slope and per-channel intercept on different pixels",
        },
        "image_count": len(rows),
        "pixel_count": total,
        "identical_pixel_count": total - different_total,
        "identical_pixel_ratio": (total - different_total) / total,
        "different_pixel_count": different_total,
        "different_pixel_ratio": different_total / total,
        "representative_images": representative,
        "missing_representative_ids": sorted(requested - representative.keys()),
        "channel_delta_histograms": {
            channel: _signed_histogram(delta_hist[index])
            for index, channel in enumerate(("red", "green", "blue"))
        },
        "rgb_difference_magnitude_histogram": _nonzero_histogram(magnitude_hist),
        "top_exact_delta_patterns": [
            {"delta_rgb": list(pattern), "pixel_count": count}
            for pattern, count in pattern_counts.most_common(top_patterns)
        ],
        "alpha_blend_fit": _fit_alpha_blend(blend),
        "threshold_analysis": {
            "otsu_all_pixels": _otsu_threshold(magnitude_hist),
            "otsu_different_pixels_only": _otsu_threshold(nonzero_magnitude),
            "magnitude_modes": _mode_analysis(magnitude_hist),
            "note": "Otsu is descriptive only; no ground-truth mask is available to estimate errors.",
        },
        "interpretation": {
            "fixed_label_color": "Unsafe unless val contains a verified exact sentinel RGB.",
            "simple_nonmatch": "Captures every edit; safe only if unchanged areas are pixel-identical.",
            "rgb_threshold": "Prefer only if the magnitude histogram has a stable valley above noise.",
            "alpha_blend": "Prefer only if parameters are physical and residual RMSE is small.",
            "selection_rule": (
                "Validate the candidate rule on manually inspected masks before changing label generation; "
                "this diagnostic alone has no ground truth for false-positive/false-negative rates."
            ),
        },
    }
    with (output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path, help="directory containing org/ and val/")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--representative",
        default="1,1300,2600",
        help="comma-separated relative image IDs (without .png)",
    )
    parser.add_argument("--top-patterns", type=int, default=20)
    args = parser.parse_args()
    diagnose_gsi_overlay(
        args.dataset_root,
        args.output_dir,
        representative_ids=(
            value.strip()
            for value in args.representative.split(",")
            if value.strip()
        ),
        top_patterns=args.top_patterns,
    )


if __name__ == "__main__":
    main()
