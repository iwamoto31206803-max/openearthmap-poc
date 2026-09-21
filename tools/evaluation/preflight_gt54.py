"""Inventory GT54, diagnose grids, and build native-grid ownership masks."""

from __future__ import annotations

import argparse
from dataclasses import replace
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
from src.evaluation.gt54_ownership import (
    build_ownership_masks,
    verify_native_rasters_unchanged,
    write_ownership_outputs,
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
        try:
            masks, ownership_rows = build_ownership_masks(inventory)
            verify_native_rasters_unchanged(inventory)
        except Exception:
            result = replace(result, geographic_ownership_available=False)
            raise
        write_preflight_outputs(args.output_dir, result)
        write_ownership_outputs(args.output_dir, masks, ownership_rows)
        return result
    except Exception as exc:
        # Inventory errors can occur before a safe grid result exists.  Emit the
        # three required Step-1 diagnostics and no downstream/empty metric CSVs.
        if "result" not in locals():
            result = PreflightResult(False, False, tuple(), tuple(), tuple(), tuple(), tolerances)
        write_preflight_outputs(args.output_dir, result, error=str(exc))
        for path in (args.output_dir / "ownership_qc.csv", args.output_dir / "ownership_masks"):
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        raise


def main(argv: list[str] | None = None) -> int:
    parser = build_parser(); args = parser.parse_args(argv)
    try:
        result = run(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"GT54 geographic ownership PASS: {len(result.inventory)} items; "
          f"pixel-identical dedup available={result.pixel_identical_dedup_available}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
