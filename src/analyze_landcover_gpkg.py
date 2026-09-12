from pathlib import Path
import argparse

import geopandas as gpd
import numpy as np
import pandas as pd


DEFAULT_THRESHOLDS = [1.0, 2.0, 5.0, 10.0]


def read_landcover_gpkg(path: Path, layer: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path, layer=layer)

    required = {"class_id", "class_name", "area_m2"}
    missing = required - set(gdf.columns)
    if missing:
        raise ValueError(
            f"{path.name}: required fields are missing: {sorted(missing)}"
        )

    gdf = gdf.copy()
    gdf["area_m2"] = pd.to_numeric(gdf["area_m2"], errors="coerce")

    # mean_conf is optional. Sieve outputs are intentionally polygonized
    # without confidence because the raw model confidence is no longer a
    # confidence measure for pixels whose class was changed by sieve.
    if "mean_conf" in gdf.columns:
        gdf["mean_conf"] = pd.to_numeric(
            gdf["mean_conf"], errors="coerce"
        )
    else:
        gdf["mean_conf"] = np.nan

    gdf = gdf[np.isfinite(gdf["area_m2"])]
    return gdf


def overall_summary(gdf: gpd.GeoDataFrame, source_name: str) -> dict:
    total_area = float(gdf["area_m2"].sum())
    conf = gdf["mean_conf"].dropna()

    return {
        "source": source_name,
        "has_confidence": bool(conf.notna().any()),
        "polygon_count": int(len(gdf)),
        "total_area_m2": total_area,
        "median_area_m2": float(gdf["area_m2"].median()),
        "mean_area_m2": float(gdf["area_m2"].mean()),
        "mean_conf": float(conf.mean()) if len(conf) else np.nan,
        "median_conf": float(conf.median()) if len(conf) else np.nan,
        "min_conf": float(conf.min()) if len(conf) else np.nan,
        "max_conf": float(conf.max()) if len(conf) else np.nan,
    }


def threshold_summary(
    gdf: gpd.GeoDataFrame,
    source_name: str,
    thresholds: list[float],
) -> list[dict]:
    total_count = len(gdf)
    total_area = float(gdf["area_m2"].sum())
    rows = []

    for threshold in thresholds:
        subset = gdf[gdf["area_m2"] < threshold]
        subset_area = float(subset["area_m2"].sum())
        conf = subset["mean_conf"].dropna()

        rows.append(
            {
                "source": source_name,
                "area_threshold_m2": threshold,
                "polygon_count_below": int(len(subset)),
                "polygon_count_pct": (
                    100.0 * len(subset) / total_count if total_count else np.nan
                ),
                "area_below_m2": subset_area,
                "area_below_pct": (
                    100.0 * subset_area / total_area if total_area else np.nan
                ),
                "mean_conf_below": (
                    float(conf.mean()) if len(conf) else np.nan
                ),
                "median_conf_below": (
                    float(conf.median()) if len(conf) else np.nan
                ),
            }
        )

    return rows


def class_summary(
    gdf: gpd.GeoDataFrame,
    source_name: str,
) -> list[dict]:
    rows = []

    for (class_id, class_name), group in gdf.groupby(
        ["class_id", "class_name"],
        dropna=False,
    ):
        conf = group["mean_conf"].dropna()
        total_area = float(group["area_m2"].sum())

        lt5 = group[group["area_m2"] < 5.0]
        lt10 = group[group["area_m2"] < 10.0]

        rows.append(
            {
                "source": source_name,
                "class_id": int(class_id),
                "class_name": class_name,
                "polygon_count": int(len(group)),
                "total_area_m2": total_area,
                "median_area_m2": float(group["area_m2"].median()),
                "mean_area_m2": float(group["area_m2"].mean()),
                "mean_conf": float(conf.mean()) if len(conf) else np.nan,
                "median_conf": float(conf.median()) if len(conf) else np.nan,
                "count_lt5m2": int(len(lt5)),
                "count_lt5m2_pct": (
                    100.0 * len(lt5) / len(group) if len(group) else np.nan
                ),
                "area_lt5m2_m2": float(lt5["area_m2"].sum()),
                "area_lt5m2_pct": (
                    100.0 * float(lt5["area_m2"].sum()) / total_area
                    if total_area else np.nan
                ),
                "count_lt10m2": int(len(lt10)),
                "count_lt10m2_pct": (
                    100.0 * len(lt10) / len(group) if len(group) else np.nan
                ),
            }
        )

    return rows


def joint_area_conf_summary(
    gdf: gpd.GeoDataFrame,
    source_name: str,
) -> list[dict]:
    area_bins = [-np.inf, 1, 2, 5, 10, 50, 200, np.inf]
    area_labels = [
        "<1",
        "1-2",
        "2-5",
        "5-10",
        "10-50",
        "50-200",
        ">=200",
    ]

    conf_bins = [-np.inf, 0.4, 0.5, 0.6, 0.7, 0.8, np.inf]
    conf_labels = [
        "<0.4",
        "0.4-0.5",
        "0.5-0.6",
        "0.6-0.7",
        "0.7-0.8",
        ">=0.8",
    ]

    temp = gdf.copy()
    temp["area_bin"] = pd.cut(
        temp["area_m2"],
        bins=area_bins,
        labels=area_labels,
        right=False,
    )
    temp["conf_bin"] = pd.cut(
        temp["mean_conf"],
        bins=conf_bins,
        labels=conf_labels,
        right=False,
    )

    rows = []

    for (class_id, class_name, area_bin, conf_bin), group in temp.groupby(
        ["class_id", "class_name", "area_bin", "conf_bin"],
        observed=True,
        dropna=False,
    ):
        rows.append(
            {
                "source": source_name,
                "class_id": int(class_id),
                "class_name": class_name,
                "area_bin_m2": str(area_bin),
                "confidence_bin": str(conf_bin),
                "polygon_count": int(len(group)),
                "total_area_m2": float(group["area_m2"].sum()),
            }
        )

    return rows


def candidate_summary(
    gdf: gpd.GeoDataFrame,
    source_name: str,
    max_area: float,
    max_conf: float,
) -> pd.DataFrame:
    subset = gdf[
        (gdf["area_m2"] < max_area)
        & (gdf["mean_conf"] < max_conf)
    ].copy()

    if subset.empty:
        return pd.DataFrame(
            columns=[
                "source",
                "polygon_id",
                "class_id",
                "class_name",
                "area_m2",
                "mean_conf",
            ]
        )

    cols = [
        c
        for c in [
            "polygon_id",
            "class_id",
            "class_name",
            "area_m2",
            "mean_conf",
        ]
        if c in subset.columns
    ]

    out = subset[cols].copy()
    out.insert(0, "source", source_name)
    return out.sort_values(["class_id", "area_m2", "mean_conf"])


def write_text_report(
    output_path: Path,
    overall_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    class_df: pd.DataFrame,
):
    lines = []
    lines.append("OpenEarthMap End-to-End PoC: GeoPackage analysis")
    lines.append("=" * 60)
    lines.append("")

    for source in overall_df["source"]:
        overall = overall_df[overall_df["source"] == source].iloc[0]
        th = threshold_df[threshold_df["source"] == source]
        cls = class_df[class_df["source"] == source]

        lines.append(f"[{source}]")
        lines.append(
            f"Polygons      : {int(overall['polygon_count'])}"
        )
        lines.append(
            f"Total area    : {overall['total_area_m2']:.1f} m2"
        )
        lines.append(
            f"Median area   : {overall['median_area_m2']:.2f} m2"
        )
        if pd.notna(overall["median_conf"]):
            lines.append(
                f"Median conf   : {overall['median_conf']:.3f}"
            )
        else:
            lines.append("Median conf   : n/a (structural analysis)")
        lines.append("")

        lines.append("Small-polygon thresholds")
        for _, row in th.iterrows():
            lines.append(
                f"  < {row['area_threshold_m2']:g} m2 : "
                f"{int(row['polygon_count_below'])} polygons "
                f"({row['polygon_count_pct']:.1f}%), "
                f"{row['area_below_m2']:.1f} m2 "
                f"({row['area_below_pct']:.3f}% of area)"
            )
        lines.append("")

        lines.append("Class summary")
        for _, row in cls.sort_values("class_id").iterrows():
            conf_text = (
                f"{row['median_conf']:.3f}"
                if pd.notna(row["median_conf"])
                else "n/a"
            )
            lines.append(
                f"  {int(row['class_id'])}: {row['class_name']} | "
                f"n={int(row['polygon_count'])}, "
                f"area={row['total_area_m2']:.1f} m2, "
                f"median area={row['median_area_m2']:.2f} m2, "
                f"median conf={conf_text}, "
                f"<5m2={row['count_lt5m2_pct']:.1f}%"
            )
        lines.append("")
        lines.append("-" * 60)
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze one or more land-cover GeoPackages created by "
            "polygonize_landcover.py."
        )
    )
    parser.add_argument(
        "gpkg",
        nargs="+",
        help="One or more GeoPackage files.",
    )
    parser.add_argument(
        "--layer",
        default="landcover",
        help="Layer name. Default: landcover",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis_results",
        help="Output directory. Default: analysis_results",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=DEFAULT_THRESHOLDS,
        help="Area thresholds in m2. Default: 1 2 5 10",
    )
    parser.add_argument(
        "--candidate-max-area",
        type=float,
        default=5.0,
        help="Max area for low-area/low-confidence candidate list. Default: 5",
    )
    parser.add_argument(
        "--candidate-max-conf",
        type=float,
        default=0.5,
        help="Max mean confidence for candidate list. Default: 0.5",
    )
    args = parser.parse_args()

    gpkg_paths = [Path(p) for p in args.gpkg]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    overall_rows = []
    threshold_rows = []
    class_rows = []
    joint_rows = []
    candidate_frames = []

    for path in gpkg_paths:
        if not path.exists():
            raise FileNotFoundError(f"GeoPackage not found: {path}")

        print(f"Reading: {path}")
        gdf = read_landcover_gpkg(path, args.layer)
        source_name = path.stem

        overall_rows.append(overall_summary(gdf, source_name))
        threshold_rows.extend(
            threshold_summary(gdf, source_name, args.thresholds)
        )
        class_rows.extend(class_summary(gdf, source_name))
        joint_rows.extend(joint_area_conf_summary(gdf, source_name))
        candidate_frames.append(
            candidate_summary(
                gdf,
                source_name,
                args.candidate_max_area,
                args.candidate_max_conf,
            )
        )

    overall_df = pd.DataFrame(overall_rows)
    threshold_df = pd.DataFrame(threshold_rows)
    class_df = pd.DataFrame(class_rows)
    joint_df = pd.DataFrame(joint_rows)
    candidate_df = pd.concat(candidate_frames, ignore_index=True)
    has_any_confidence = bool(overall_df["has_confidence"].any())

    overall_path = output_dir / "overall_summary.csv"
    threshold_path = output_dir / "threshold_summary.csv"
    class_path = output_dir / "class_summary.csv"
    joint_path = output_dir / "area_confidence_by_class.csv"
    candidate_path = output_dir / "low_area_low_conf_candidates.csv"
    report_path = output_dir / "report.txt"

    overall_df.to_csv(overall_path, index=False, encoding="utf-8-sig")
    threshold_df.to_csv(threshold_path, index=False, encoding="utf-8-sig")
    class_df.to_csv(class_path, index=False, encoding="utf-8-sig")
    if has_any_confidence:
        joint_df.to_csv(joint_path, index=False, encoding="utf-8-sig")
        candidate_df.to_csv(
            candidate_path, index=False, encoding="utf-8-sig"
        )

    write_text_report(
        report_path,
        overall_df,
        threshold_df,
        class_df,
    )

    print()
    print("ANALYSIS OK")
    print("-------------------------------------")
    print("Overall summary       :", overall_path)
    print("Threshold summary     :", threshold_path)
    print("Class summary         :", class_path)
    if has_any_confidence:
        print("Area x confidence     :", joint_path)
        print("Candidate polygons    :", candidate_path)
    else:
        print("Confidence outputs    : skipped (mean_conf not present)")
    print("Text report           :", report_path)
    if has_any_confidence:
        print()
        print(
            "Candidate rule        : "
            f"area < {args.candidate_max_area:g} m2 AND "
            f"mean_conf < {args.candidate_max_conf:g}"
        )


if __name__ == "__main__":
    main()
