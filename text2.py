from pathlib import Path
import csv
import math


ROOT = Path(r"C:\OpenEarthMap_PoC\data\saclaj\results")

OLD = ROOT / "20260919T101351_262021Z_48be0033" / "site_results.csv"
NEW = ROOT / "20260920T021202_280390Z_b4c1cedf" / "site_results.csv"


def read_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    result = {}

    for row in rows:
        site_id = row["ID"]

        if site_id in result:
            raise RuntimeError(f"duplicate ID: {site_id}")

        result[site_id] = row

    return result


def same_float(a, b, tol=1e-12):
    try:
        x = float(a)
        y = float(b)
    except (TypeError, ValueError):
        return a == b

    if math.isnan(x) and math.isnan(y):
        return True

    return abs(x - y) <= tol


old = read_rows(OLD)
new = read_rows(NEW)

old_ids = set(old)
new_ids = set(new)
common = sorted(old_ids & new_ids)

print("=" * 72)
print("SACLAJ INPUT IDENTITY CHECK")
print("=" * 72)

print("old rows =", len(old))
print("new rows =", len(new))
print("common IDs =", len(common))
print("old only =", len(old_ids - new_ids))
print("new only =", len(new_ids - old_ids))
print()

subtype_mismatch = 0
expected_class_mismatch = 0
status_mismatch = 0

common_success = 0

patch_sha_mismatch = 0
base_predicted_class_mismatch = 0
base_expected_probability_mismatch = 0
base_agriculture_probability_mismatch = 0

for site_id in common:
    a = old[site_id]
    b = new[site_id]

    if a["subtype"] != b["subtype"]:
        subtype_mismatch += 1

    if a["expected_class"] != b["expected_class"]:
        expected_class_mismatch += 1

    if a["status"] != b["status"]:
        status_mismatch += 1

    if a["status"] == "success" and b["status"] == "success":
        common_success += 1

        if a["patch_sha256"] != b["patch_sha256"]:
            patch_sha_mismatch += 1

        if a["base_predicted_class"] != b["base_predicted_class"]:
            base_predicted_class_mismatch += 1

        if not same_float(
            a["base_expected_probability"],
            b["base_expected_probability"],
        ):
            base_expected_probability_mismatch += 1

        if not same_float(
            a["base_agriculture_probability"],
            b["base_agriculture_probability"],
        ):
            base_agriculture_probability_mismatch += 1


print("subtype mismatch =", subtype_mismatch)
print("expected-class mismatch =", expected_class_mismatch)
print("status mismatch =", status_mismatch)
print()

print("common-success =", common_success)
print("patch SHA256 mismatch =", patch_sha_mismatch)
print("Base predicted-class mismatch =", base_predicted_class_mismatch)
print(
    "Base expected-probability mismatch =",
    base_expected_probability_mismatch,
)
print(
    "Base agriculture-probability mismatch =",
    base_agriculture_probability_mismatch,
)

print()
print("=" * 72)

perfect = (
    len(old) == 1000
    and len(new) == 1000
    and len(common) == 1000
    and not old_ids - new_ids
    and not new_ids - old_ids
    and subtype_mismatch == 0
    and expected_class_mismatch == 0
    and status_mismatch == 0
    and common_success == 1000
    and patch_sha_mismatch == 0
    and base_predicted_class_mismatch == 0
    and base_expected_probability_mismatch == 0
    and base_agriculture_probability_mismatch == 0
)

print("RESULT =", "PERFECT IDENTITY" if perfect else "NOT PERFECT IDENTITY")
print("=" * 72)
