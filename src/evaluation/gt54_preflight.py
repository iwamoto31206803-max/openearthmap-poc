"""GT54 inventory and pixel-identical grid compatibility diagnostic.

This module only inspects raster metadata and GT masks. It never reprojects or
resamples raster values. A failed common-lattice diagnostic does not prevent
the separate native-grid geographic ownership workflow from continuing.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.warp import transform_bounds


EXPECTED_ITEM_COUNT = 54
EXPECTED_REGION_COUNT = 8
VALID_CLASSES = frozenset(range(9))
VALAREA_RE = re.compile(r"^ValArea_(\d+)$")


@dataclass(frozen=True)
class GridTolerances:
    """Explicit, pixel-normalized tolerances used by grid preflight."""

    rotation_shear_pixels: float = 1e-9
    pixel_size_relative: float = 1e-9
    phase_pixels: float = 1e-6
    bounds_pixels: float = 1e-6

    def as_dict(self) -> dict[str, float]:
        return {
            "rotation_shear_pixels": self.rotation_shear_pixels,
            "pixel_size_relative": self.pixel_size_relative,
            "phase_pixels": self.phase_pixels,
            "bounds_pixels": self.bounds_pixels,
        }


@dataclass(frozen=True)
class ManifestItem:
    valarea: str
    year: int
    region: str

    @property
    def numeric_id(self) -> int:
        match = VALAREA_RE.fullmatch(self.valarea)
        if not match:  # guarded by manifest validation
            raise ValueError(f"invalid ValArea ID: {self.valarea}")
        return int(match.group(1))


@dataclass(frozen=True)
class RasterMetadata:
    item: ManifestItem
    rgb_path: Path
    gt_path: Path
    crs: CRS
    transform: Affine
    width: int
    height: int
    bounds: tuple[float, float, float, float]
    gt_nodata: float | None
    gt_invalid_mask_pixels: int
    gt_valid_pixels: int
    gt_class_histogram: tuple[int, ...]
    rgb_sha256: str
    gt_sha256: str


@dataclass(frozen=True)
class Placement:
    valarea: str
    grid_group_id: str
    global_row_offset: int
    global_col_offset: int


@dataclass(frozen=True)
class PreflightResult:
    pixel_identical_dedup_available: bool
    geographic_ownership_available: bool
    inventory: tuple[RasterMetadata, ...]
    placements: tuple[Placement, ...]
    grid_rows: tuple[dict[str, object], ...]
    errors: tuple[str, ...]
    tolerances: GridTolerances

    @property
    def passed(self) -> bool:
        """The evaluator may continue when geographic ownership is available."""
        return self.geographic_ownership_available


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path, *, expected_items: int = EXPECTED_ITEM_COUNT,
                  expected_regions: int = EXPECTED_REGION_COUNT) -> list[ManifestItem]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["valarea", "year", "region"]:
            raise ValueError("manifest header must be exactly: valarea,year,region")
        items: list[ManifestItem] = []
        for line, row in enumerate(reader, 2):
            try:
                item = ManifestItem(row["valarea"].strip(), int(row["year"]), row["region"].strip())
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid manifest row {line}") from exc
            if not VALAREA_RE.fullmatch(item.valarea) or not item.region:
                raise ValueError(f"invalid manifest row {line}")
            items.append(item)
    if len(items) != expected_items:
        raise ValueError(f"manifest item count is {len(items)}, expected {expected_items}")
    if len({item.valarea for item in items}) != len(items):
        raise ValueError("manifest ValArea IDs must be unique")
    if len({item.numeric_id for item in items}) != len(items):
        raise ValueError("manifest ValArea numeric suffixes must be unique")
    regions = {item.region for item in items}
    if len(regions) != expected_regions:
        raise ValueError(f"manifest region count is {len(regions)}, expected {expected_regions}")
    return sorted(items, key=lambda item: (item.numeric_id, item.valarea))


def validate_local_manifest(path: Path, items: Iterable[ManifestItem]) -> None:
    """Validate the builder inventory without trusting its stored raster metadata."""
    expected = {item.valarea: item for item in items}
    if not path.is_file():
        raise FileNotFoundError(f"local dataset manifest not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"valarea", "region", "gsi_year", "alignment_ok", "error"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("local dataset manifest lacks required inventory columns")
        rows = list(reader)
    if len(rows) != len(expected):
        raise ValueError("local dataset manifest item count does not match formal manifest")
    seen: set[str] = set()
    for line, row in enumerate(rows, 2):
        valarea = row["valarea"].strip()
        if valarea in seen or valarea not in expected:
            raise ValueError(f"unexpected or duplicate local manifest ValArea at row {line}: {valarea}")
        seen.add(valarea)
        item = expected[valarea]
        if row["region"].strip() != item.region or row["gsi_year"].strip() != str(item.year):
            raise ValueError(f"local manifest metadata mismatch for {valarea}")
        if row["alignment_ok"].strip().lower() != "true" or row["error"].strip():
            raise ValueError(f"local manifest does not record a successful aligned item: {valarea}")


def _nodata_is_semantic(nodata: float | None) -> bool:
    return (nodata is not None and math.isfinite(float(nodata))
            and float(nodata).is_integer() and int(nodata) in VALID_CLASSES)


def inspect_item(item: ManifestItem, dataset_root: Path) -> RasterMetadata:
    rgb_path = dataset_root / "rgb_images" / f"{item.valarea}.tif"
    gt_path = dataset_root / "labels" / f"{item.valarea}.tif"
    if not rgb_path.is_file() or not gt_path.is_file():
        raise FileNotFoundError(f"RGB/GT pair missing for {item.valarea}")
    with rasterio.open(rgb_path) as rgb, rasterio.open(gt_path) as gt:
        if rgb.count != 3 or gt.count != 1:
            raise ValueError(f"{item.valarea}: expected 3-band RGB and 1-band GT")
        if gt.crs is None:
            raise ValueError(f"{item.valarea}: GT has no CRS")
        fields = ("width", "height", "crs", "transform", "bounds")
        mismatches = [name for name in fields if getattr(rgb, name) != getattr(gt, name)]
        if mismatches:
            raise ValueError(f"{item.valarea}: RGB/GT alignment mismatch: {', '.join(mismatches)}")
        nodata = gt.nodata
        if _nodata_is_semantic(nodata):
            raise ValueError(
                f"{item.valarea}: GT NoData value {nodata:g} collides with OEM8 class 0..8"
            )
        values = gt.read(1)
        valid_mask = gt.read_masks(1) != 0
        valid_values = values[valid_mask]
        if valid_values.size == 0:
            raise ValueError(f"{item.valarea}: GT has no valid pixels")
        invalid_classes = np.unique(valid_values[(valid_values < 0) | (valid_values > 8)])
        if invalid_classes.size:
            raise ValueError(f"{item.valarea}: valid GT pixels contain classes outside 0..8")
        histogram = np.bincount(valid_values.astype(np.int64), minlength=9)[:9]
        bounds = tuple(float(value) for value in gt.bounds)
        metadata = RasterMetadata(
            item=item, rgb_path=rgb_path.resolve(), gt_path=gt_path.resolve(), crs=gt.crs,
            transform=gt.transform, width=gt.width, height=gt.height, bounds=bounds,
            gt_nodata=nodata, gt_invalid_mask_pixels=int(valid_mask.size - valid_mask.sum()),
            gt_valid_pixels=int(valid_mask.sum()),
            gt_class_histogram=tuple(int(value) for value in histogram),
            rgb_sha256="", gt_sha256="",
        )
    return RasterMetadata(**{**metadata.__dict__, "rgb_sha256": sha256_file(rgb_path),
                             "gt_sha256": sha256_file(gt_path)})


def inventory_dataset(items: Iterable[ManifestItem], dataset_root: Path) -> list[RasterMetadata]:
    return [inspect_item(item, dataset_root) for item in items]


def _normalized_bounds(meta: RasterMetadata) -> tuple[float, float, float, float]:
    try:
        return tuple(float(value) for value in transform_bounds(
            meta.crs, "EPSG:4326", *meta.bounds, densify_pts=21
        ))
    except Exception as exc:
        raise ValueError(f"{meta.item.valarea}: cannot transform bounds for overlap detection") from exc


def _bounds_overlap(left: tuple[float, float, float, float],
                    right: tuple[float, float, float, float], tolerance: float = 0.0) -> bool:
    return (min(left[2], right[2]) - max(left[0], right[0]) > tolerance
            and min(left[3], right[3]) - max(left[1], right[1]) > tolerance)


def _tiles_overlap(left: RasterMetadata, right: RasterMetadata,
                   geographic: dict[str, tuple[float, float, float, float]]) -> bool:
    if left.crs == right.crs:
        return _bounds_overlap(left.bounds, right.bounds)
    # Coordinate transformation is used only for conservative footprint
    # discovery.  Pixel placement never transforms or resamples either raster.
    return _bounds_overlap(geographic[left.item.valarea], geographic[right.item.valarea], 1e-12)


def _north_up(meta: RasterMetadata, tolerance: GridTolerances) -> tuple[bool, float, float]:
    transform = meta.transform
    scale = max(abs(transform.a), abs(transform.e))
    if not math.isfinite(scale) or scale <= 0:
        return False, math.inf, math.inf
    rotation = abs(transform.d) / scale
    shear = abs(transform.b) / scale
    return (transform.a > 0 and transform.e < 0
            and rotation <= tolerance.rotation_shear_pixels
            and shear <= tolerance.rotation_shear_pixels), rotation, shear


def _pixel_sizes_compatible(a: RasterMetadata, b: RasterMetadata,
                            tolerance: GridTolerances) -> bool:
    for left, right in ((abs(a.transform.a), abs(b.transform.a)),
                        (abs(a.transform.e), abs(b.transform.e))):
        scale = max(left, right)
        if scale <= 0 or abs(left - right) / scale > tolerance.pixel_size_relative:
            return False
    return True


def _integer_offset(value: float, origin: float, pixel_size: float,
                    tolerance_pixels: float) -> tuple[int, float, bool]:
    raw = (value - origin) / pixel_size
    integer = round(raw)
    residual = abs(raw - integer)
    return integer, residual, residual <= tolerance_pixels


def _placement(reference: RasterMetadata, meta: RasterMetadata,
               tolerance: GridTolerances) -> tuple[int, int, float, float, bool]:
    col, col_residual, col_ok = _integer_offset(
        meta.transform.c, reference.transform.c, abs(reference.transform.a), tolerance.phase_pixels
    )
    row, row_residual, row_ok = _integer_offset(
        reference.transform.f, meta.transform.f, abs(reference.transform.e), tolerance.phase_pixels
    )
    expected_right = reference.transform.c + (col + meta.width) * abs(reference.transform.a)
    expected_bottom = reference.transform.f - (row + meta.height) * abs(reference.transform.e)
    right_residual = abs(meta.bounds[2] - expected_right) / abs(reference.transform.a)
    bottom_residual = abs(meta.bounds[1] - expected_bottom) / abs(reference.transform.e)
    bounds_ok = max(right_residual, bottom_residual) <= tolerance.bounds_pixels
    return row, col, max(row_residual, col_residual), max(right_residual, bottom_residual), (
        row_ok and col_ok and bounds_ok
    )


def preflight_grids(inventory: Iterable[RasterMetadata],
                    tolerances: GridTolerances = GridTolerances()) -> PreflightResult:
    metadata = sorted(inventory, key=lambda value: (value.item.numeric_id, value.item.valarea))
    errors: list[str] = []
    rows: list[dict[str, object]] = []
    geographic = {meta.item.valarea: _normalized_bounds(meta) for meta in metadata}
    adjacency: dict[str, set[str]] = {meta.item.valarea: set() for meta in metadata}
    by_name = {meta.item.valarea: meta for meta in metadata}

    north_up: dict[str, bool] = {}
    for meta in metadata:
        ok, rotation, shear = _north_up(meta, tolerances)
        north_up[meta.item.valarea] = ok
        if not ok:
            errors.append(f"{meta.item.valarea}: rotation/shear exceeds tolerance")
        rows.append({"record_type": "tile", "tile_a": meta.item.valarea, "tile_b": "",
                     "overlaps": "", "crs_equal": "", "pixel_size_equal": "",
                     "rotation_pixels": rotation, "shear_pixels": shear,
                     "phase_residual_pixels": "", "bounds_residual_pixels": "",
                     "canonical_mapping_ok": ok, "grid_group_id": "",
                     "global_row_offset": "", "global_col_offset": "",
                     "status": "PASS" if ok else "FAIL",
                     "failure_reason": "" if ok else "rotation_or_shear"})

    for index, left in enumerate(metadata):
        for right in metadata[index + 1:]:
            overlaps = _tiles_overlap(left, right, geographic)
            if not overlaps:
                continue
            crs_equal = left.crs == right.crs
            size_equal = crs_equal and _pixel_sizes_compatible(left, right, tolerances)
            compatible = crs_equal and size_equal and north_up[left.item.valarea] and north_up[right.item.valarea]
            phase_residual = bounds_residual = math.inf
            if compatible:
                _, _, phase_residual, bounds_residual, compatible = _placement(left, right, tolerances)
            if compatible:
                adjacency[left.item.valarea].add(right.item.valarea)
                adjacency[right.item.valarea].add(left.item.valarea)
            else:
                errors.append(f"incompatible overlapping tiles: {left.item.valarea}, {right.item.valarea}")
            rows.append({"record_type": "overlap_pair", "tile_a": left.item.valarea,
                         "tile_b": right.item.valarea, "overlaps": True,
                         "crs_equal": crs_equal, "pixel_size_equal": size_equal,
                         "rotation_pixels": "", "shear_pixels": "",
                         "phase_residual_pixels": phase_residual,
                         "bounds_residual_pixels": bounds_residual,
                         "canonical_mapping_ok": compatible, "grid_group_id": "",
                         "global_row_offset": "", "global_col_offset": "",
                         "status": "PASS" if compatible else "FAIL",
                         "failure_reason": "" if compatible else "incompatible_overlap"})

    # Connected overlap components are independent namespaces.  Isolated tiles
    # remain separate groups, so incompatible non-overlapping grids are allowed.
    placements: list[Placement] = []
    visited: set[str] = set()
    group_number = 0
    if not errors:
        for start in sorted(by_name, key=lambda name: (by_name[name].item.numeric_id, name)):
            if start in visited:
                continue
            stack, component = [start], []
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current); component.append(current)
                stack.extend(sorted(adjacency[current], reverse=True))
            component.sort(key=lambda name: (by_name[name].item.numeric_id, name))
            reference = by_name[component[0]]
            group_id = f"grid_{group_number:03d}"
            group_number += 1
            for name in component:
                meta = by_name[name]
                row, col, phase, bounds, ok = _placement(reference, meta, tolerances)
                if not ok:
                    errors.append(f"{name}: canonical placement residual exceeds tolerance")
                    continue
                placements.append(Placement(name, group_id, row, col))
                rows.append({"record_type": "placement", "tile_a": name, "tile_b": "",
                             "overlaps": "", "crs_equal": True, "pixel_size_equal": True,
                             "rotation_pixels": "", "shear_pixels": "",
                             "phase_residual_pixels": phase, "bounds_residual_pixels": bounds,
                             "canonical_mapping_ok": True, "grid_group_id": group_id,
                             "global_row_offset": row, "global_col_offset": col,
                             "status": "PASS", "failure_reason": ""})
    return PreflightResult(not errors, True, tuple(metadata), tuple(placements), tuple(rows),
                           tuple(errors), tolerances)


def dataset_qc_rows(inventory: Iterable[RasterMetadata]) -> list[dict[str, object]]:
    rows = []
    for meta in inventory:
        row: dict[str, object] = {
            "valarea": meta.item.valarea, "region": meta.item.region, "year": meta.item.year,
            "rgb_path": str(meta.rgb_path), "gt_path": str(meta.gt_path),
            "rgb_sha256": meta.rgb_sha256, "gt_sha256": meta.gt_sha256,
            "crs": meta.crs.to_string(), "width": meta.width, "height": meta.height,
            "transform": repr(tuple(meta.transform)), "bounds": repr(meta.bounds),
            "gt_nodata": "" if meta.gt_nodata is None else meta.gt_nodata,
            "gt_invalid_mask_pixels": meta.gt_invalid_mask_pixels,
            "gt_valid_pixels": meta.gt_valid_pixels, "alignment_ok": True, "status": "PASS",
        }
        row.update({f"gt_class_{class_id}_pixels": count
                    for class_id, count in enumerate(meta.gt_class_histogram)})
        rows.append(row)
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames))
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def write_preflight_outputs(output_dir: Path, result: PreflightResult,
                            *, error: str = "") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_rows = dataset_qc_rows(result.inventory)
    dataset_fields = (["valarea", "region", "year", "rgb_path", "gt_path", "rgb_sha256",
                       "gt_sha256", "crs", "width", "height", "transform", "bounds",
                       "gt_nodata", "gt_invalid_mask_pixels", "gt_valid_pixels"]
                      + [f"gt_class_{value}_pixels" for value in range(9)]
                      + ["alignment_ok", "status"])
    write_csv(output_dir / "dataset_qc.csv", dataset_rows, dataset_fields)
    tolerance_values = result.tolerances.as_dict()
    grid_rows = [{**row, **tolerance_values} for row in result.grid_rows]
    grid_fields = ["record_type", "tile_a", "tile_b", "overlaps", "crs_equal",
                   "pixel_size_equal", "rotation_pixels", "shear_pixels",
                   "phase_residual_pixels", "bounds_residual_pixels", "canonical_mapping_ok",
                   "grid_group_id", "global_row_offset", "global_col_offset", "status",
                   "failure_reason", *tolerance_values.keys()]
    write_csv(output_dir / "grid_preflight_qc.csv", grid_rows, grid_fields)
    summary = [{"status": "PASS" if result.passed and not error else "FAIL",
                "pixel_identical_dedup_available": result.pixel_identical_dedup_available,
                "geographic_ownership_available": result.geographic_ownership_available,
                "item_count": len(result.inventory),
                "region_count": len({meta.item.region for meta in result.inventory}),
                "grid_group_count": len({placement.grid_group_id for placement in result.placements}),
                "placement_count": len(result.placements),
                "pixel_identical_issue_count": len(result.errors),
                "fatal_error_count": int(bool(error)),
                "pixel_identical_issues": " | ".join(result.errors),
                "fatal_error": error,
                **tolerance_values}]
    write_csv(output_dir / "evaluation_summary.csv", summary, summary[0].keys())
