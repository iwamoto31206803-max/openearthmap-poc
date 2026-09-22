"""Formal ownership-aware A/B/C/D model comparison for GT54 (Step 4B).

The module is deliberately descriptive: it emits corrections, regressions and
prediction transitions, but never ranks or selects a model.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from src.config import CLASS_NAMES
from src.evaluation.gt54_aggregation import _read_qc, load_ownership_mask
from src.evaluation.gt54_inference import (
    EXPECTED_CHECKPOINT_SHA256, MODEL_ORDER, sha256_file, validate_cross_model_identity,
)
from src.evaluation.gt54_metrics import load_raster_pair, metrics_from_confusion
from src.evaluation.gt54_preflight import ManifestItem


SCHEMA_VERSION = 1
COMPARISON_PAIRS = (("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("C", "D"))
FIXED_TRANSITIONS = ((8, 4), (5, 4), (6, 4), (7, 4), (3, 4), (4, 3))
CATEGORIES = ("unchanged_correct", "correction", "regression",
              "changed_still_wrong", "unchanged_wrong")


@dataclass(frozen=True)
class ComparisonResult:
    metrics: tuple[dict[str, object], ...]
    correction_regression: tuple[dict[str, object], ...]
    summaries: tuple[dict[str, object], ...]
    transitions: tuple[dict[str, object], ...]
    cubes: tuple[dict[str, object], ...]
    fixed_transitions: tuple[dict[str, object], ...]
    top_transitions: tuple[dict[str, object], ...]
    region_metrics: tuple[dict[str, object], ...]
    qc: tuple[dict[str, object], ...]


def comparison_name(before: str, after: str) -> str:
    return f"{before}->{after}"


def change_category(gt: int, before: int, after: int) -> str:
    if before == gt:
        return "unchanged_correct" if after == gt else "regression"
    if after == gt:
        return "correction"
    return "unchanged_wrong" if before == after else "changed_still_wrong"


def cube_from_arrays(gt: np.ndarray, before: np.ndarray, after: np.ndarray,
                     formal_mask: np.ndarray) -> np.ndarray:
    """Build a complete GT/before/after 9x9x9 count cube."""
    if not (gt.shape == before.shape == after.shape == formal_mask.shape):
        raise ValueError("GT, predictions, and formal mask shapes must match")
    encoded = (gt[formal_mask].astype(np.int64) * 81
               + before[formal_mask].astype(np.int64) * 9
               + after[formal_mask].astype(np.int64))
    return np.bincount(encoded, minlength=729).reshape(9, 9, 9)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Step 4A provenance file not found: {path}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Step 4A provenance file is empty: {path.name}")
    return rows


def validate_inference_provenance(inference_root: Path, items: Iterable[ManifestItem],
                                  *, expected_items: int = 54) -> None:
    """Fail closed unless the input is a complete formal Step-4A run."""
    config_path = inference_root / "inference_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Step 4A provenance file not found: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    run = payload.get("run", {})
    if (run.get("run_mode") != "formal_full"
            or run.get("inventory_item_count") != expected_items
            or run.get("selected_item_count") != expected_items):
        raise ValueError("Step 4A provenance is not a complete formal_full run")
    models = payload.get("models", [])
    if [model.get("model_id") for model in models] != list(MODEL_ORDER):
        raise ValueError("Step 4A config must contain models A, B, C, D in order")
    for model in models:
        model_id = model["model_id"]
        if (model.get("sha256") != EXPECTED_CHECKPOINT_SHA256[model_id]
                or model.get("expected_sha256") != EXPECTED_CHECKPOINT_SHA256[model_id]):
            raise ValueError(f"checkpoint {model_id} does not match formal expected SHA256")

    manifest_rows = _read_csv(inference_root / "inference_manifest.csv")
    qc_rows = _read_csv(inference_root / "inference_qc.csv")
    expected_valareas = {item.valarea for item in items}
    if len(manifest_rows) != expected_items * len(MODEL_ORDER):
        raise ValueError("Step 4A inference manifest does not contain expected_items x 4 rows")
    if len(qc_rows) != len(MODEL_ORDER):
        raise ValueError("Step 4A inference QC does not contain four model rows")
    for row in manifest_rows:
        if (row.get("status") != "PASS" or row.get("run_mode") != "formal_full"
                or int(row.get("inventory_item_count", -1)) != expected_items
                or int(row.get("selected_item_count", -1)) != expected_items):
            raise ValueError("Step 4A inference manifest contains a non-formal/non-PASS row")
    by_model = {model: [] for model in MODEL_ORDER}
    for row in manifest_rows:
        if row.get("model_id") not in by_model:
            raise ValueError("Step 4A inference manifest contains an unexpected model")
        by_model[row["model_id"]].append(row)
    if any({row["valarea"] for row in rows} != expected_valareas or len(rows) != expected_items
           for rows in by_model.values()):
        raise ValueError("Step 4A model prediction inventories are incomplete or duplicated")
    # validate_cross_model_identity expects A/B/C/D order within each ValArea.
    ordered_identity = [next(row for row in by_model[m] if row["valarea"] == item.valarea)
                        for item in items for m in MODEL_ORDER]
    validate_cross_model_identity(ordered_identity)

    ignored = {"model_id", "checkpoint_path", "checkpoint_sha256",
               "expected_checkpoint_sha256"}
    identities = [{k: v for k, v in row.items() if k not in ignored} for row in qc_rows]
    if [row.get("model_id") for row in qc_rows] != list(MODEL_ORDER):
        raise ValueError("Step 4A inference QC model order/inventory is invalid")
    if any(row.get("status") != "PASS" or row.get("run_mode") != "formal_full"
           for row in qc_rows) or any(identity != identities[0] for identity in identities[1:]):
        raise ValueError("Step 4A inference configuration identity/status mismatch")
    for row in qc_rows:
        model = row["model_id"]
        if (row.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256[model]
                or row.get("expected_checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256[model]):
            raise ValueError(f"checkpoint {model} QC SHA256 mismatch")

    for model in MODEL_ORDER:
        prediction_dir = inference_root / model / "predictions"
        paths = {path.stem: path for path in prediction_dir.glob("*.tif")}
        if set(paths) != expected_valareas:
            raise ValueError(f"model {model} prediction files do not exactly match the manifest")
        for row in by_model[model]:
            expected_hash = row.get("prediction_sha256", "").lower()
            if not expected_hash or sha256_file(paths[row["valarea"]]) != expected_hash:
                raise ValueError(f"{model}/{row['valarea']}: prediction SHA256 mismatch")


def _category_counts(cube: np.ndarray, gt_ids=range(9)) -> dict[str, int]:
    counts = dict.fromkeys(CATEGORIES, 0)
    for gt in gt_ids:
        for before in range(9):
            for after in range(9):
                counts[change_category(gt, before, after)] += int(cube[gt, before, after])
    return counts


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _delta(before: float | None, after: float | None) -> float | None:
    return after - before if before is not None and after is not None else None


def _class_rows(name: str, before_model: str, after_model: str,
                cube: np.ndarray) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    before_result = metrics_from_confusion(cube.sum(axis=2))
    after_result = metrics_from_confusion(cube.sum(axis=1))
    metrics, changes = [], []
    for class_id in range(1, 9):
        bm, am = before_result.class_metrics[class_id - 1], after_result.class_metrics[class_id - 1]
        metrics.append({"comparison": name, "before_model": before_model,
                        "after_model": after_model, "class_id": class_id,
                        "class_name": CLASS_NAMES[class_id], "gt_support": bm.gt_support,
                        "before_iou": bm.iou, "after_iou": am.iou,
                        "delta_iou": _delta(bm.iou, am.iou),
                        "before_precision": bm.precision, "after_precision": am.precision,
                        "delta_precision": _delta(bm.precision, am.precision),
                        "before_recall": bm.recall, "after_recall": am.recall,
                        "delta_recall": _delta(bm.recall, am.recall),
                        "before_prediction_support": bm.prediction_support,
                        "after_prediction_support": am.prediction_support})
        counts = _category_counts(cube, (class_id,))
        support = int(cube[class_id].sum())
        net = counts["correction"] - counts["regression"]
        changes.append({"comparison": name, "scope": "global_deduplicated",
                        "region": "", "class_id": class_id,
                        "class_name": CLASS_NAMES[class_id], "gt_support": support,
                        "unchanged_correct_pixels": counts["unchanged_correct"],
                        "corrected_pixels": counts["correction"],
                        "regressed_pixels": counts["regression"],
                        "changed_still_wrong_pixels": counts["changed_still_wrong"],
                        "unchanged_wrong_pixels": counts["unchanged_wrong"],
                        "corrected_rate": _rate(counts["correction"], support),
                        "regressed_rate": _rate(counts["regression"], support),
                        "net_correct_change": net, "net_correct_rate": _rate(net, support)})
    return metrics, changes


def analyze_cubes(cubes: dict[tuple[str, str], np.ndarray],
                  region_cubes: dict[tuple[str, str, str], np.ndarray] | None = None,
                  *, item_count: int = 0, ownership_status: str = "verified") -> ComparisonResult:
    """Create all deterministic output tables and assert protocol identities."""
    if tuple(cubes) != COMPARISON_PAIRS:
        raise ValueError("comparison pairs must be exactly A->B, A->C, A->D, B->C, C->D")
    all_metrics, all_changes, summaries, transitions, cube_rows = [], [], [], [], []
    fixed, tops, region_rows, qc = [], [], [], []
    for before_model, after_model in COMPARISON_PAIRS:
        name = comparison_name(before_model, after_model)
        cube = np.asarray(cubes[(before_model, after_model)], dtype=np.int64)
        if cube.shape != (9, 9, 9) or np.any(cube < 0):
            raise ValueError("comparison cube must be a non-negative 9x9x9 array")
        formal = int(cube.sum())
        direct = _category_counts(cube)
        semantic = _category_counts(cube, range(1, 9))
        semantic_support = int(cube[1:].sum())
        metric_rows, change_rows = _class_rows(name, before_model, after_model, cube)
        all_metrics.extend(metric_rows); all_changes.extend(change_rows)
        net = semantic["correction"] - semantic["regression"]
        summaries.append({"comparison": name, "formal_valid_pixels": formal,
                          "semantic_gt_pixels": semantic_support, **direct,
                          "semantic_net_correct_change": net,
                          "semantic_corrected_rate": _rate(semantic["correction"], semantic_support),
                          "semantic_regressed_rate": _rate(semantic["regression"], semantic_support),
                          "semantic_net_correct_rate": _rate(net, semantic_support)})
        prediction_matrix = cube.sum(axis=0)
        for before in range(9):
            for after in range(9):
                transitions.append({"comparison": name, "before_class_id": before,
                                    "before_class_name": CLASS_NAMES[before],
                                    "after_class_id": after,
                                    "after_class_name": CLASS_NAMES[after],
                                    "pixel_count": int(prediction_matrix[before, after])})
        for gt in range(9):
            for before in range(9):
                for after in range(9):
                    cube_rows.append({"comparison": name, "gt_class_id": gt,
                                      "gt_class_name": CLASS_NAMES[gt],
                                      "before_class_id": before,
                                      "before_class_name": CLASS_NAMES[before],
                                      "after_class_id": after,
                                      "after_class_name": CLASS_NAMES[after],
                                      "pixel_count": int(cube[gt, before, after]),
                                      "change_category": change_category(gt, before, after)})
        for before, after in FIXED_TRANSITIONS:
            count = int(prediction_matrix[before, after])
            support = int(prediction_matrix[before].sum())
            fixed.append({"comparison": name, "before_class_id": before,
                          "before_class_name": CLASS_NAMES[before],
                          "after_class_id": after, "after_class_name": CLASS_NAMES[after],
                          "transition_pixel_count": count,
                          "rate_among_formal_valid_pixels": _rate(count, formal),
                          "before_prediction_support": support,
                          "rate_among_before_prediction_support": _rate(count, support)})
        changed_total = int(prediction_matrix.sum() - np.trace(prediction_matrix))
        candidates = [(int(prediction_matrix[b, a]), b, a) for b in range(9)
                      for a in range(9) if b != a and prediction_matrix[b, a] > 0]
        candidates.sort(key=lambda value: (-value[0], value[1], value[2]))
        for rank, (count, before, after) in enumerate(candidates[:10], 1):
            tops.append({"comparison": name, "rank": rank, "before_class_id": before,
                         "before_class_name": CLASS_NAMES[before], "after_class_id": after,
                         "after_class_name": CLASS_NAMES[after], "pixel_count": count,
                         "rate_among_changed_predictions": _rate(count, changed_total),
                         "rate_among_formal_valid_pixels": _rate(count, formal)})

        # Independent invariants, including confusion reconstruction and semantic identity.
        before_conf, after_conf = cube.sum(axis=2), cube.sum(axis=1)
        before_correct = int(sum(before_conf[c, c] for c in range(1, 9)))
        after_correct = int(sum(after_conf[c, c] for c in range(1, 9)))
        checks = {
            "categories_exhaustive": sum(direct.values()) == formal,
            "semantic_support_reconciled": sum(r["gt_support"] for r in change_rows) == semantic_support,
            "prediction_transition_reconciled": int(prediction_matrix.sum()) == formal,
            "cube_reconciled": int(cube.sum()) == formal,
            "cube_categories_reconciled": _category_counts(cube) == direct,
            "before_confusion_reconstructed": int(before_conf.sum()) == formal,
            "after_confusion_reconstructed": int(after_conf.sum()) == formal,
            "per_class_net_identity": all(r["net_correct_change"] == r["corrected_pixels"] - r["regressed_pixels"] for r in change_rows),
            "semantic_net_identity": net == after_correct - before_correct,
        }
        if not all(checks.values()):
            raise AssertionError(f"{name}: comparison invariant failed: {checks}")
        qc.append({"comparison": name, "status": "PASS", "item_count_per_model": item_count,
                   "ownership_mask_status": ownership_status, "formal_valid_pixels": formal,
                   **checks})

        if region_cubes:
            for region in sorted({key[0] for key in region_cubes}):
                rcube = region_cubes[(region, before_model, after_model)]
                rm, rchanges = _class_rows(name, before_model, after_model, rcube)
                for metric, change in zip(rm, rchanges):
                    region_rows.append({"comparison": name, "region": region,
                                        "class_id": metric["class_id"],
                                        "class_name": metric["class_name"],
                                        "gt_support": metric["gt_support"],
                                        "before_iou": metric["before_iou"],
                                        "after_iou": metric["after_iou"],
                                        "delta_iou": metric["delta_iou"],
                                        "corrected_pixels": change["corrected_pixels"],
                                        "regressed_pixels": change["regressed_pixels"],
                                        "net_correct_change": change["net_correct_change"],
                                        "net_correct_rate": change["net_correct_rate"]})
            if sum(int(region_cubes[(r, before_model, after_model)].sum())
                   for r in sorted({key[0] for key in region_cubes})) != formal:
                raise AssertionError(f"{name}: region totals do not reconcile with global")
    return ComparisonResult(*(tuple(rows) for rows in
        (all_metrics, all_changes, summaries, transitions, cube_rows, fixed, tops, region_rows, qc)))


def evaluate_model_comparisons(items: Iterable[ManifestItem], dataset_root: Path,
                               inference_root: Path, ownership_dir: Path,
                               *, expected_items: int = 54,
                               validate_provenance: bool = True) -> ComparisonResult:
    ordered = tuple(sorted(items, key=lambda item: (item.numeric_id, item.valarea)))
    if len(ordered) != expected_items:
        raise ValueError(f"comparison inventory has {len(ordered)} items, expected {expected_items}")
    if validate_provenance:
        validate_inference_provenance(inference_root, ordered, expected_items=expected_items)
    ownership_qc = _read_qc(ownership_dir)
    cubes = {pair: np.zeros((9, 9, 9), dtype=np.int64) for pair in COMPARISON_PAIRS}
    regions = sorted({item.region for item in ordered})
    region_cubes = {(region, *pair): np.zeros((9, 9, 9), dtype=np.int64)
                    for region in regions for pair in COMPARISON_PAIRS}
    ownership_statuses: set[str] = set()
    for item in ordered:
        gt = valid = None
        predictions: dict[str, np.ndarray] = {}
        for model in MODEL_ORDER:
            loaded_gt, prediction, loaded_valid = load_raster_pair(
                dataset_root / "labels" / f"{item.valarea}.tif",
                inference_root / model / "predictions" / f"{item.valarea}.tif")
            if gt is None:
                gt, valid = loaded_gt, loaded_valid
            elif not np.array_equal(gt, loaded_gt) or not np.array_equal(valid, loaded_valid):
                raise AssertionError(f"{item.valarea}: GT identity differs between models")
            predictions[model] = prediction
        ownership, status = load_ownership_mask(ownership_dir, item, gt.shape, ownership_qc)
        ownership_statuses.add(status)
        formal_mask = valid & ownership
        for pair in COMPARISON_PAIRS:
            tile_cube = cube_from_arrays(gt, predictions[pair[0]], predictions[pair[1]], formal_mask)
            cubes[pair] += tile_cube
            region_cubes[(item.region, *pair)] += tile_cube
    status = ownership_statuses.pop() if len(ownership_statuses) == 1 else "mixed"
    return analyze_cubes(cubes, region_cubes, item_count=len(ordered), ownership_status=status)


def _write(path: Path, rows: tuple[dict[str, object], ...],
           fieldnames: tuple[str, ...] | None = None) -> None:
    if not rows and fieldnames is None:
        raise ValueError(f"cannot infer columns for empty CSV: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else list(fieldnames or ()))
        writer.writeheader(); writer.writerows(rows)


def write_comparison_outputs(output_dir: Path, result: ComparisonResult) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in (
        ("model_comparison_metrics.csv", result.metrics),
        ("correction_regression.csv", result.correction_regression),
        ("comparison_summary.csv", result.summaries),
        ("prediction_transitions.csv", result.transitions),
        ("gt_conditioned_transitions.csv", result.cubes),
        ("fixed_transition_diagnostics.csv", result.fixed_transitions),
        ("top_prediction_transitions.csv", result.top_transitions),
        ("region_comparison_metrics.csv", result.region_metrics),
        ("comparison_qc.csv", result.qc),
    ):
        top_fields = ("comparison", "rank", "before_class_id", "before_class_name",
                      "after_class_id", "after_class_name", "pixel_count",
                      "rate_among_changed_predictions", "rate_among_formal_valid_pixels")
        _write(output_dir / filename, rows,
               top_fields if filename == "top_prediction_transitions.csv" else None)
