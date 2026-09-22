"""Create formal descriptive A/B/C/D comparisons on the owned GT54 surface."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.evaluation.gt54_comparison import evaluate_model_comparisons, write_comparison_outputs
from src.evaluation.gt54_preflight import load_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=REPOSITORY_ROOT / "manifests" / "val_gt_georef.csv")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--inference-root", type=Path, required=True)
    parser.add_argument("--ownership-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-items", type=int, default=54,
                        help=argparse.SUPPRESS)
    parser.add_argument("--expected-regions", type=int, default=8,
                        help=argparse.SUPPRESS)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run(args: argparse.Namespace):
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {args.output_dir}; use --overwrite")
    items = load_manifest(args.manifest, expected_items=args.expected_items,
                          expected_regions=args.expected_regions)
    # Complete validation happens before an old report can be removed.
    result = evaluate_model_comparisons(
        items, args.dataset_root, args.inference_root, args.ownership_dir,
        expected_items=args.expected_items)
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    write_comparison_outputs(args.output_dir, result)
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"GT54 Step 4B PASS: {len(result.summaries)} formal descriptive comparisons")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
