"""Generate fair, auditable A/B/C/D GT54 prediction sets (Step 4A)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.evaluation.gt54_inference import MODEL_ORDER, run_gt54_inference


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=REPOSITORY_ROOT / "manifests" / "val_gt_georef.csv")
    parser.add_argument("--dataset-root", type=Path, required=True)
    for model_id in MODEL_ORDER:
        parser.add_argument(f"--model-{model_id.lower()}", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--smoke-items", type=int,
                        help=("After full formal inventory/checkpoint preflight, infer only the "
                              "first N items in numeric ValArea order."))
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace the entire output root after successful preflight.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = {model_id: getattr(args, f"model_{model_id.lower()}") for model_id in MODEL_ORDER}
    try:
        rows = run_gt54_inference(
            manifest=args.manifest, dataset_root=args.dataset_root, checkpoint_paths=paths,
            output_root=args.output_root, overlap=args.overlap, overwrite=args.overwrite,
            smoke_items=args.smoke_items)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"GT54 Step 4A PASS: {len(rows)} predictions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
