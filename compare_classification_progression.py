"""Run the standard 00 -> 01 -> 02 -> 03 classification comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from compare_classification_rasters import build_parser as comparison_parser
from compare_classification_rasters import run, validate_args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare Base, Paddy, Water, and Road OEM8 rasters in sequence.")
    parser.add_argument("--stage-00", type=Path, required=True, help="Original Base classification.")
    parser.add_argument("--stage-01", type=Path, required=True, help="Base + Paddy classification.")
    parser.add_argument("--stage-02", type=Path, required=True, help="Base + Paddy + Water classification.")
    parser.add_argument("--stage-03", type=Path, required=True, help="Base + Paddy + Water + Road classification.")
    parser.add_argument("--output-dir", type=Path, default=Path("transition_diagnostics"))
    aoi = parser.add_mutually_exclusive_group()
    aoi.add_argument("--bbox", nargs=4, type=float, metavar=("LEFT", "BOTTOM", "RIGHT", "TOP"))
    aoi.add_argument("--rows", nargs=2, type=int, metavar=("START", "STOP"))
    parser.add_argument("--cols", nargs=2, type=int, metavar=("START", "STOP"))
    parser.add_argument("--changed-geotiff", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stages = (args.stage_00, args.stage_01, args.stage_02, args.stage_03)
    labels = ("00_to_01", "01_to_02", "02_to_03")
    for source, target, label in zip(stages, stages[1:], labels):
        forwarded = [str(source), str(target), "--label", label, "--output-dir", str(args.output_dir)]
        if args.bbox: forwarded += ["--bbox", *map(str, args.bbox)]
        if args.rows: forwarded += ["--rows", *map(str, args.rows), "--cols", *map(str, args.cols or ())]
        if args.changed_geotiff: forwarded.append("--changed-geotiff")
        if args.overwrite: forwarded.append("--overwrite")
        parser = comparison_parser(); comparison_args = parser.parse_args(forwarded)
        validate_args(parser, comparison_args)
        run(comparison_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
