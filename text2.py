from pathlib import Path
import csv
import json
import math


RESULTS_DIR = Path(
    r"C:\OpenEarthMap_PoC\data\saclaj\results"
)

V03_CHECKPOINT_SHA256 = (
    "e536052223f2989ef382fd7d1bbfaa0d75662c0757c8362b574b7c28df4d0172"
)

PHASE_B_RUN_ID = "20260919T010127_502237Z_1e06ae40"

FLOAT_TOL = 1e-12


def load_manifest(run_dir):
    path = run_dir / "evaluation_manifest.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def find_v03_run():
    matches = []

    for run_dir in RESULTS_DIR.iterdir():
        if not run_dir.is_dir():
            continue

        manifest = load_manifest(run_dir)
        if not manifest:
            continue

        if (
            manifest.get("fine_tuned_checkpoint_sha256", "").lower()
            == V03_CHECKPOINT_SHA256.lower()
        ):
            matches.append(run_dir)

    if not matches:
        raise RuntimeError(
            "Could not find SACLAJ v0.3 evaluation run."
        )

    if len(matches) > 1:
        print("WARNING: multiple v0.3 runs found:")
        for path in matches:
            print(" ", path.name)

        matches.sort(
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        print("Using newest:", matches[0].name)

    return matches[0]


def load_site_results(run_dir):
    path = run_dir / "site_results.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    rows = {}

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            site_id = row["ID"]

            if site_id in rows:
                raise RuntimeError(
                    "Duplicate site ID in local result"
                )

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


v03_run = find_v03_run()

phase_b_run = RESULTS_DIR / PHASE_B_RUN_ID

if not phase_b_run.exists():
    raise FileNotFoundError(phase_b_run)


old = load_site_results(v03_run)
new = load_site_results(phase_b_run)


old_ids = set(old)
new_ids = set(new)

common_ids = old_ids & new_ids


sample_id_only_old = old_ids - new_ids
sample_id_only_new = new_ids - old_ids


status_mismatch = 0

old_success = {
    site_id
    for site_id, row in old.items()
    if row["status"] == "success"
}

new_success = {
    site_id
    for site_id, row in new.items()
    if row["status"] == "success"
}

common_success = old_success & new_success


patch_sha_mismatch = 0
base_predicted_class_mismatch = 0
base_expected_probability_mismatch = 0
base_agriculture_probability_mismatch = 0

subtype_mismatch = 0
expected_class_mismatch = 0


for site_id in common_ids:

    a = old[site_id]
    b = new[site_id]

    if a["status"] != b["status"]:
        status_mismatch += 1

    if a["subtype"] != b["subtype"]:
        subtype_mismatch += 1

    if a["expected_class"] != b["expected_class"]:
        expected_class_mismatch += 1


for site_id in common_success:

    a = old[site_id]
    b = new[site_id]

    if a["patch_sha256"] != b["patch_sha256"]:
        patch_sha_mismatch += 1

    if (
        a["base_predicted_class"]
        != b["base_predicted_class"]
    ):
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
print("=" * 72)

print("v0.3 run =", v03_run.name)
print("Phase B run =", phase_b_run.name)

print()
print("SAMPLED SET")
print("v0.3 rows =", len(old))
print("Phase B rows =", len(new))
print("common IDs =", len(common_ids))
print("only v0.3 =", len(sample_id_only_old))
print("only Phase B =", len(sample_id_only_new))

print()
print("REFERENCE IDENTITY")
print("subtype mismatches =", subtype_mismatch)
print("expected-class mismatches =", expected_class_mismatch)

print()
print("STATUS")
print("v0.3 success =", len(old_success))
print("Phase B success =", len(new_success))
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

if (
    len(old_ids) == len(new_ids)
    and not sample_id_only_old
    and not sample_id_only_new
    and subtype_mismatch == 0
    and expected_class_mismatch == 0
    and patch_sha_mismatch == 0
    and base_predicted_class_mismatch == 0
    and base_expected_probability_mismatch == 0
    and base_agriculture_probability_mismatch == 0
):
    print(
        "RESULT: sampled set is identical, and all common-success "
        "patch/Base outputs are identical."
    )

    if len(old_success) != len(new_success):
        print(
            "NOTE: success sets differ, so full 1000-point "
            "input identity is incomplete."
        )
else:
    print(
        "RESULT: identity differences were detected. "
        "Inspect locally before attributing model differences."
    )
