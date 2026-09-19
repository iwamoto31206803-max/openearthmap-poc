from pathlib import Path
import csv
import math

RESULTS_DIR = Path(r"C:\OpenEarthMap_PoC\data\saclaj\results")

V03_RUN_ID = "20260918T000738_565461Z_3d6b2f0b"
V02_RUN_ID = "20260919T101351_262021Z_48be0033"

FLOAT_TOL = 1e-12


def load_site_results(run_id):
    path = RESULTS_DIR / run_id / "site_results.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    rows = {}

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            site_id = row["ID"]

            if site_id in rows:
                raise RuntimeError(f"Duplicate ID found: {site_id}")

            rows[site_id] = row

    return rows


def same_float(a, b, tol=FLOAT_TOL):
    try:
        x = float(a)
        y = float(b)
    except (TypeError, ValueError):
        return a == b

    if not math.isfinite(x) or not math.isfinite(y):
        return x == y

    return abs(x - y) <= tol


v03 = load_site_results(V03_RUN_ID)
v02 = load_site_results(V02_RUN_ID)

v03_ids = set(v03)
v02_ids = set(v02)
common_ids = v03_ids & v02_ids

only_v03 = v03_ids - v02_ids
only_v02 = v02_ids - v03_ids

subtype_mismatch = 0
expected_class_mismatch = 0
status_mismatch = 0

patch_sha_mismatch = 0
base_predicted_class_mismatch = 0
base_expected_probability_mismatch = 0
base_agriculture_probability_mismatch = 0

v03_success = {
    site_id for site_id, row in v03.items()
    if row["status"] == "success"
}

v02_success = {
    site_id for site_id, row in v02.items()
    if row["status"] == "success"
}

common_success = v03_success & v02_success


for site_id in common_ids:
    a = v03[site_id]
    b = v02[site_id]

    if a["subtype"] != b["subtype"]:
        subtype_mismatch += 1

    if a["expected_class"] != b["expected_class"]:
        expected_class_mismatch += 1

    if a["status"] != b["status"]:
        status_mismatch += 1


for site_id in common_success:
    a = v03[site_id]
    b = v02[site_id]

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


print()
print("=" * 72)
print("SACLAJ INPUT IDENTITY VERIFICATION")
print("Phase A v0.3 vs Phase B v0.2")
print("=" * 72)

print()
print("RUNS")
print("v0.3 =", V03_RUN_ID)
print("v0.2 =", V02_RUN_ID)

print()
print("SAMPLED SET")
print("v0.3 rows =", len(v03))
print("v0.2 rows =", len(v02))
print("common IDs =", len(common_ids))
print("only v0.3 =", len(only_v03))
print("only v0.2 =", len(only_v02))

print()
print("REFERENCE IDENTITY")
print("subtype mismatches =", subtype_mismatch)
print("expected-class mismatches =", expected_class_mismatch)

print()
print("STATUS")
print("v0.3 success =", len(v03_success))
print("v0.2 success =", len(v02_success))
print("common success =", len(common_success))
print("status mismatches =", status_mismatch)

print()
print("COMMON-SUCCESS INPUT IDENTITY")
print("patch SHA256 mismatches =", patch_sha_mismatch)

print()
print("COMMON-SUCCESS BASE OUTPUT IDENTITY")
print(
    "Base predicted-class mismatches =",
    base_predicted_class_mismatch,
)
print(
    "Base expected-probability mismatches =",
    base_expected_probability_mismatch,
)
print(
    "Base agriculture-probability mismatches =",
    base_agriculture_probability_mismatch,
)

print()
print("=" * 72)

perfect = (
    len(v03) == 1000
    and len(v02) == 1000
    and len(common_ids) == 1000
    and not only_v03
    and not only_v02
    and subtype_mismatch == 0
    and expected_class_mismatch == 0
    and status_mismatch == 0
    and len(v03_success) == 1000
    and len(v02_success) == 1000
    and patch_sha_mismatch == 0
    and base_predicted_class_mismatch == 0
    and base_expected_probability_mismatch == 0
    and base_agriculture_probability_mismatch == 0
)

if perfect:
    print("RESULT: PERFECT IDENTITY")
    print(
        "All 1000 sampled sites, patches, reference labels, statuses, "
        "and Base outputs are identical."
    )
else:
    print("RESULT: DIFFERENCES DETECTED")
    print("Inspect the counts above before attributing differences to the model.")
