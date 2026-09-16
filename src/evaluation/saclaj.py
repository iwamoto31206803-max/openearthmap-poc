"""Strict local CSV ingestion, explicit mapping, sampling and paired aggregates."""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re

REQUIRED_COLUMNS = (
    "ID", "Category_ID", "Latitude", "Longitude", "Date", "Diameter（m）",
    "Category_detail", "Note",
)
DIAMETER_COLUMNS = ("Diameter（m）", "Diameter(m)")
# Keys are deliberately fixed: no free-form input content reaches aggregates.
SUBTYPES = {
    "rice_paddy": 7, "other_crop": 7,
    "broadleaf": 5, "needleleaf": 5, "mixed": 5, "water": 6,
    "needleleaf_evergreen": 5, "broadleaf_evergreen": 5,
    "needleleaf_deciduous": 5, "broadleaf_deciduous": 5,
}
JAPAN_FILTER = {
    "method": "inclusive WGS84 bounding box",
    "west": 122.0, "east": 154.0, "south": 20.0, "north": 46.0,
    "limitation": "Approximate Japan extent; includes foreign territory/ocean inside the box; no border polygon test.",
}
LIMITATIONS = [
    "Independent point reference, not pixel-perfect ground truth or production accuracy evidence.",
    "SACLAJ observation dates may differ from the acquisition dates of GSI latest imagery; actual land-cover change can appear as evaluation mismatch.",
    JAPAN_FILTER["limitation"],
    "Category mapping is selective and semantic; urban/built-up and ambiguous categories are excluded.",
    "Diameter is metadata only, not a ground-truth circle; one containing pixel is evaluated without neighborhood voting.",
    "Category-stratified capped samples are not representative of Japan-wide prevalence; failed acquisitions can bias the evaluated subset.",
    "SACLAJ is not used for this fine-tuning; geographic overlap with training imagery and base pretraining data has not been verified.",
    "GSI latest imagery can change between runs; saved patches and their hashes preserve the actual paired inputs.",
    "Softmax probabilities are not calibrated accuracy; exact numerics across hardware/library versions are not guaranteed.",
]


class ValidationError(ValueError):
    """Validation diagnostic containing only controlled text, never input values."""


@dataclass(frozen=True)
class Site:
    site_id: str
    category_id: str
    latitude: float
    longitude: float
    date: str
    diameter: float | None
    detail: str


@dataclass(frozen=True)
class Reference:
    site: Site
    subtype: str
    expected_class: int


@dataclass(frozen=True)
class CsvData:
    sites: list[Site]
    diameter_column: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path, encoding: str = "utf-8-sig") -> CsvData:
    """Comma-delimited CSV with exact headers, decimal WGS84, unique nonempty IDs.

    Date is retained as a nonempty opaque string (no temporal matching).
    Diameter may be blank, otherwise finite/nonnegative. Note is not copied.
    Errors name row numbers/fields only, never input values or locations.
    """
    sites, seen = [], set()
    with Path(path).open(encoding=encoding, newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        headers = reader.fieldnames or []
        required = set(REQUIRED_COLUMNS) - {DIAMETER_COLUMNS[0]}
        if len(headers) != len(set(headers)) or not required <= set(headers):
            raise ValidationError("CSV requires unique exact headers: " + ", ".join(REQUIRED_COLUMNS))
        diameter_columns = [column for column in DIAMETER_COLUMNS if column in headers]
        if len(diameter_columns) != 1:
            raise ValidationError("CSV requires exactly one Diameter column: Diameter（m） or Diameter(m)")
        diameter_column = diameter_columns[0]
        for row_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValidationError(f"CSV row {row_number}: incorrect column count")
            values = {key: row[key].strip() for key in required}
            diameter_value = row[diameter_column].strip()
            if not values["ID"] or values["ID"] in seen:
                raise ValidationError(f"CSV row {row_number}: empty or duplicate ID")
            if not re.fullmatch(r"[0-9]+", values["Category_ID"]):
                raise ValidationError(f"CSV row {row_number}: Category_ID must be a nonnegative integer")
            if not values["Date"]:
                raise ValidationError(f"CSV row {row_number}: Date is required")
            try:
                lat, lon = float(values["Latitude"]), float(values["Longitude"])
                diameter = float(diameter_value) if diameter_value else None
            except ValueError:
                raise ValidationError(f"CSV row {row_number}: invalid numeric field") from None
            if not (math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
                raise ValidationError(f"CSV row {row_number}: invalid WGS84 coordinates")
            if diameter is not None and (not math.isfinite(diameter) or diameter < 0):
                raise ValidationError(f"CSV row {row_number}: invalid Diameter(m)")
            seen.add(values["ID"])
            sites.append(Site(values["ID"], str(int(values["Category_ID"])), lat, lon,
                              values["Date"], diameter, values["Category_detail"]))
    if not sites:
        raise ValidationError("CSV contains no data rows")
    return CsvData(sites, diameter_column)


def load_mapping(path: Path) -> dict:
    """No guessed SACLAJ IDs. Require a reviewed transcription of official definitions.

    Category_ID alone determines eligibility and subtype. Category_detail is
    optional free-form metadata, never a category name or mapping input.
    A null subtype is an explicit exclusion. Unknown IDs are also excluded.
    """
    table = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if (not isinstance(table, dict) or table.get("schema_version") != 2
            or table.get("definition_verified") is not True
            or not isinstance(table.get("mapping_version"), str)
            or not table["mapping_version"].strip()
            or not isinstance(table.get("definition_source"), str)
            or not table["definition_source"].strip()
            or not re.fullmatch(r"[0-9a-f]{64}", table.get("definition_sha256", ""))):
        raise ValidationError("Mapping requires version, verified official definition source and its SHA256")
    entries = table.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValidationError("Mapping requires explicit entries from the official SACLAJ definition")
    categories, accepted = set(), set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"category_id", "subtype"}:
            raise ValidationError("Mapping entries require category_id and subtype only")
        category, subtype = entry["category_id"], entry["subtype"]
        if (not isinstance(category, str) or not re.fullmatch(r"0|[1-9][0-9]*", category)
                or (subtype is not None and (not isinstance(subtype, str) or subtype not in SUBTYPES))):
            raise ValidationError("Invalid mapping entry; supported evaluation subtypes only")
        if category in categories:
            raise ValidationError("Duplicate/ambiguous mapping Category_ID")
        categories.add(category)
        accepted.add(subtype)
    if "rice_paddy" not in accepted:
        raise ValidationError("Mapping must include the primary rice_paddy category")
    if not accepted.intersection({subtype for subtype, expected in SUBTYPES.items() if expected != 7}):
        raise ValidationError("Mapping needs non-Agriculture references to assess leakage")
    return table


def in_japan(site: Site) -> bool:
    return (JAPAN_FILTER["south"] <= site.latitude <= JAPAN_FILTER["north"]
            and JAPAN_FILTER["west"] <= site.longitude <= JAPAN_FILTER["east"])


def subtype_counts(references) -> dict:
    counts = Counter(ref.subtype for ref in references)
    return {subtype: counts[subtype] for subtype in SUBTYPES}


def select_sites(sites: list[Site], mapping: dict, maximum: int = 100, seed: int = 42):
    if maximum < 1 or not 0 <= seed < 2**32:
        raise ValidationError("maximum must be positive and seed in [0, 2**32)")
    lookup = {e["category_id"]: e["subtype"] for e in mapping["entries"]}
    domestic = [s for s in sites if in_japan(s)]
    eligible = []
    for site in domestic:
        subtype = lookup.get(site.category_id)
        if subtype is not None:
            eligible.append(Reference(site, subtype, SUBTYPES[subtype]))
    # SHA256 ranking is independent of CSV row order and Python RNG versions.
    sampled = []
    for subtype in SUBTYPES:
        group = [ref for ref in eligible if ref.subtype == subtype]
        def rank(ref):
            key = json.dumps([seed, subtype, ref.site.site_id], ensure_ascii=False, separators=(",", ":"))
            return hashlib.sha256(key.encode("utf-8")).hexdigest(), ref.site.site_id
        sampled.extend(sorted(group, key=rank)[:maximum])
    counts = {
        "input": len(sites), "after_japan_filter": len(domestic),
        "excluded_outside_japan_bbox": len(sites) - len(domestic),
        "excluded_unmapped_or_ambiguous": len(domestic) - len(eligible),
        "eligible": len(eligible), "before_sampling_by_subtype": subtype_counts(eligible),
        "sampled": len(sampled), "sampled_by_subtype": subtype_counts(sampled),
    }
    return sampled, counts


def paired_metric(rows, field, target=None, percent=False):
    values = {}
    for model in ("base", "fine_tuned"):
        samples = [row[model][field] for row in rows]
        if target is not None:
            samples = [float(value == target) for value in samples]
        values[model] = sum(samples) / len(samples) * (100 if percent else 1) if samples else None
    delta = values["fine_tuned"] - values["base"] if rows else None
    return {"n": len(rows), **values, "delta_pp" if percent else "delta": delta}


def aggregate(results: list[dict], counts: dict) -> dict:
    """Build a fresh allowlisted aggregate; never copy IDs, coordinates or errors."""
    paired = [row for row in results if row["status"] == "success"]
    categories = {}
    for subtype, expected in SUBTYPES.items():
        rows = [row for row in paired if row["subtype"] == subtype]
        categories[subtype] = {
            "expected_class": expected,
            "agreement_percent": paired_metric(rows, "predicted_class", expected, True),
            "mean_expected_probability": paired_metric(rows, "expected_probability"),
            "mean_agriculture_probability": paired_metric(rows, "agriculture_probability"),
            "skipped": sum(r["subtype"] == subtype and r["status"] == "skipped" for r in results),
            "error": sum(r["subtype"] == subtype and r["status"] == "error" for r in results),
        }
    # Reconstruct the counts, too: arbitrary caller keys cannot enter sanitized JSON.
    clean_counts = {key: int(counts[key]) for key in (
        "input", "after_japan_filter", "excluded_outside_japan_bbox",
        "excluded_unmapped_or_ambiguous", "eligible", "sampled",
    )}
    for key in ("before_sampling_by_subtype", "sampled_by_subtype"):
        clean_counts[key] = {subtype: int(counts[key][subtype]) for subtype in SUBTYPES}
    return {
        "schema_version": "saclaj-evaluation-0.1",
        "description": "Sanitized aggregate without coordinates, IDs or site-level results",
        "counts": clean_counts, "japan_filter": JAPAN_FILTER,
        "outcomes": {status: sum(r["status"] == status for r in results)
                     for status in ("success", "skipped", "error")},
        "rice_agreement_percent": categories["rice_paddy"]["agreement_percent"],
        "agriculture_leakage_percent": paired_metric(
            [r for r in paired if r["expected_class"] != 7], "predicted_class", 7, True),
        "categories": categories, "known_limitations": list(LIMITATIONS),
    }
