"""Evaluate one aligned GT54 prediction set using Step-2/Step-3 metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.evaluation.gt54_aggregation import (
    evaluate_prediction_set_with_ownership, write_ownership_evaluation_outputs,
)
from src.evaluation.gt54_preflight import load_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=REPOSITORY_ROOT / "manifests" / "val_gt_georef.csv")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--ownership-dir", type=Path, required=True,
                        help="Step-1 directory containing ownership_masks/ and optional ownership_qc.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="")
    parser.add_argument("--expected-items", type=int, default=54,
                        help="Expected manifest item count (tests may override).")
    parser.add_argument("--expected-regions", type=int, default=8,
                        help="Expected manifest region count (tests may override).")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run(args: argparse.Namespace):
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {args.output_dir}; use --overwrite")
    items = load_manifest(args.manifest, expected_items=args.expected_items,
                          expected_regions=args.expected_regions)
    # Validate every raster before replacing a prior report, so failure cannot
    # leave a partial result that looks like a successful evaluation.
    result = evaluate_prediction_set_with_ownership(
        items, args.dataset_root, args.prediction_dir, args.ownership_dir,
        model_id=args.model_id)
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    write_ownership_evaluation_outputs(args.output_dir, result)
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"GT54 Step 3 PASS: {len(result.tiles)} tiles; "
          f"global deduplicated mIoU-8={result.global_deduplicated.miou_8}; "
          f"equal-region macro mIoU-8={result.equal_region_macro_miou_8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
