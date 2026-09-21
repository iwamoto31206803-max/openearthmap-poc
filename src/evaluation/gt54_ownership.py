"""Deterministic geographic ownership masks for native-grid GT54 rasters.

Only pixel-center coordinates and tile footprint geometries are transformed.
Raster arrays are never reprojected, resampled, or used to select an owner.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
from pyproj import Transformer
import rasterio
import shapely
from shapely.geometry import Polygon
from shapely.ops import transform as transform_geometry

from src.evaluation.gt54_preflight import RasterMetadata


OWNERSHIP_CRS = "EPSG:3857"
DISTANCE_TIE_TOLERANCE = 1e-9


def _native_footprint(meta: RasterMetadata) -> Polygon:
    transform = meta.transform
    corners = [(transform.a * col + transform.b * row + transform.c,
                transform.d * col + transform.e * row + transform.f)
               for col, row in ((0, 0), (meta.width, 0),
                                (meta.width, meta.height), (0, meta.height))]
    polygon = Polygon(corners)
    if not polygon.is_valid or polygon.area <= 0:
        raise ValueError(f"{meta.item.valarea}: invalid raster footprint")
    return polygon


def _ownership_geometry(meta: RasterMetadata) -> Polygon:
    transformer = Transformer.from_crs(meta.crs, OWNERSHIP_CRS, always_xy=True)
    try:
        polygon = transform_geometry(transformer.transform, _native_footprint(meta))
    except Exception as exc:
        raise ValueError(f"{meta.item.valarea}: footprint transformation failed") from exc
    if not polygon.is_valid or polygon.area <= 0 or not np.all(np.isfinite(polygon.bounds)):
        raise ValueError(f"{meta.item.valarea}: unusable transformed footprint")
    return polygon


def _positive_area_overlap(left: Polygon, right: Polygon) -> bool:
    return left.intersection(right).area > 0.0


def _pixel_centers(meta: RasterMetadata, row_start: int, row_stop: int) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.meshgrid(np.arange(row_start, row_stop, dtype=np.float64),
                             np.arange(meta.width, dtype=np.float64), indexing="ij")
    cols += 0.5; rows += 0.5
    transform = meta.transform
    x = transform.a * cols + transform.b * rows + transform.c
    y = transform.d * cols + transform.e * rows + transform.f
    transformer = Transformer.from_crs(meta.crs, OWNERSHIP_CRS, always_xy=True)
    world_x, world_y = transformer.transform(x, y)
    world_x, world_y = np.asarray(world_x), np.asarray(world_y)
    if not np.all(np.isfinite(world_x)) or not np.all(np.isfinite(world_y)):
        raise ValueError(f"{meta.item.valarea}: non-finite transformed pixel center")
    return world_x, world_y


def build_ownership_masks(inventory: Iterable[RasterMetadata], *, chunk_rows: int = 1024
                          ) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    """Assign each native-grid pixel center to one geometry-only owner tile."""
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    metadata = sorted(inventory, key=lambda value: (value.item.numeric_id, value.item.valarea))
    geometries = {meta.item.valarea: _ownership_geometry(meta) for meta in metadata}
    ranks = {meta.item.valarea: meta.item.numeric_id for meta in metadata}
    candidates: dict[str, list[RasterMetadata]] = {}
    for meta in metadata:
        candidates[meta.item.valarea] = [
            other for other in metadata
            if _positive_area_overlap(geometries[meta.item.valarea], geometries[other.item.valarea])
            or other.item.valarea == meta.item.valarea
        ]

    masks: dict[str, np.ndarray] = {}
    qc_rows: list[dict[str, object]] = []
    for meta in metadata:
        own_rank = ranks[meta.item.valarea]
        owned = np.zeros((meta.height, meta.width), dtype=bool)
        overlap_centers = 0
        candidate_tiles = candidates[meta.item.valarea]
        for row_start in range(0, meta.height, chunk_rows):
            row_stop = min(row_start + chunk_rows, meta.height)
            x, y = _pixel_centers(meta, row_start, row_stop)
            shape = x.shape
            best_distance = np.full(shape, -np.inf, dtype=np.float64)
            best_rank = np.full(shape, np.iinfo(np.int64).max, dtype=np.int64)
            membership_count = np.zeros(shape, dtype=np.uint16)
            points = shapely.points(x, y)
            for candidate in candidate_tiles:
                geometry = geometries[candidate.item.valarea]
                inside = shapely.intersects_xy(geometry, x, y)
                membership_count += inside
                distances = shapely.distance(geometry.boundary, points)
                better = inside & (distances > best_distance + DISTANCE_TIE_TOLERANCE)
                tied = inside & (np.abs(distances - best_distance) <= DISTANCE_TIE_TOLERANCE)
                choose = better | (tied & (ranks[candidate.item.valarea] < best_rank))
                best_distance[choose] = distances[choose]
                best_rank[choose] = ranks[candidate.item.valarea]
            if np.any(best_rank == np.iinfo(np.int64).max):
                raise ValueError(f"{meta.item.valarea}: pixel center outside every candidate footprint")
            owned[row_start:row_stop] = best_rank == own_rank
            overlap_centers += int(np.count_nonzero(membership_count > 1))
        masks[meta.item.valarea] = owned
        qc_rows.append({
            "valarea": meta.item.valarea, "region": meta.item.region,
            "ownership_crs": OWNERSHIP_CRS, "candidate_tile_count": len(candidate_tiles),
            "raw_pixel_count": int(owned.size), "owned_pixel_count": int(owned.sum()),
            "dropped_pixel_count": int(owned.size - owned.sum()),
            "overlap_center_count": overlap_centers,
            "owned_mask_sha256": hashlib.sha256(owned.tobytes(order="C")).hexdigest(),
            "tie_tolerance_crs_units": DISTANCE_TIE_TOLERANCE,
            "status": "PASS",
        })
    return masks, qc_rows


def write_ownership_outputs(output_dir: Path, masks: dict[str, np.ndarray],
                            qc_rows: list[dict[str, object]]) -> None:
    mask_dir = output_dir / "ownership_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    for valarea in sorted(masks):
        np.save(mask_dir / f"{valarea}.npy", masks[valarea], allow_pickle=False)
    from src.evaluation.gt54_preflight import write_csv
    fields = (qc_rows[0].keys() if qc_rows else (
        "valarea", "region", "ownership_crs", "candidate_tile_count", "raw_pixel_count",
        "owned_pixel_count", "dropped_pixel_count", "overlap_center_count",
        "owned_mask_sha256", "tie_tolerance_crs_units", "status"))
    write_csv(output_dir / "ownership_qc.csv", qc_rows, fields)


def verify_native_rasters_unchanged(inventory: Iterable[RasterMetadata]) -> None:
    """Verify files still have their inventoried hash after ownership generation."""
    from src.evaluation.gt54_preflight import sha256_file
    for meta in inventory:
        if sha256_file(meta.rgb_path) != meta.rgb_sha256 or sha256_file(meta.gt_path) != meta.gt_sha256:
            raise ValueError(f"{meta.item.valarea}: raster changed during ownership generation")
