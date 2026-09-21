"""Inventory GT54 and validate lossless cross-tile canonical grid placement."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.evaluation.gt54_preflight import (
    GridTolerances,
    PreflightResult,
    inventory_dataset,
    load_manifest,
    preflight_grids,
    validate_local_manifest,
    write_preflight_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=REPOSITORY_ROOT / "manifests" / "val_gt_georef.csv")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-items", type=int, default=54,
                        help="Expected formal-manifest item count (tests may override).")
    parser.add_argument("--expected-regions", type=int, default=8,
                        help="Expected formal-manifest region count (tests may override).")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run(args: argparse.Namespace) -> PreflightResult:
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {args.output_dir}; use --overwrite")
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tolerances = GridTolerances()
    try:
        items = load_manifest(args.manifest, expected_items=args.expected_items,
                              expected_regions=args.expected_regions)
        validate_local_manifest(args.dataset_root / "manifest.csv", items)
        inventory = inventory_dataset(items, args.dataset_root)
        result = preflight_grids(inventory, tolerances)
        write_preflight_outputs(args.output_dir, result)
        return result
    except Exception as exc:
        # Inventory errors can occur before a safe grid result exists.  Emit the
        # three required Step-1 diagnostics and no downstream/empty metric CSVs.
        result = PreflightResult(False, tuple(), tuple(), tuple(), (str(exc),), tolerances)
        write_preflight_outputs(args.output_dir, result)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = build_parser(); args = parser.parse_args(argv)
    try:
        result = run(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if not result.passed:
        print("ERROR: grid preflight failed: " + " | ".join(result.errors), file=sys.stderr)
        return 1
    print(f"GT54 grid preflight PASS: {len(result.inventory)} items, "
          f"{len({item.grid_group_id for item in result.placements})} grid groups")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
