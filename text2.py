from pathlib import Path
import csv


PADDY_AUDIT = Path(
    r"C:\OpenEarthMap_PoC\data\gsi\prepared\paddy_572\audit.csv"
)

WATER_AUDIT = Path(
    r"C:\OpenEarthMap_PoC\data\gsi\prepared\water_572\audit.csv"
)


def load_by_sha(path):
    by_sha = {}

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            sha = row["org_sha256"].strip().lower()

            by_sha.setdefault(sha, []).append(
                {
                    "source_image_id": row["source_image_id"],
                    "positive_pixel_count": int(
                        row["positive_pixel_count"]
                    ),
                    "is_false_image": row["is_false_image"],
                }
            )

    return by_sha


paddy = load_by_sha(PADDY_AUDIT)
water = load_by_sha(WATER_AUDIT)

paddy_shas = set(paddy)
water_shas = set(water)

common = sorted(paddy_shas & water_shas)

print("paddy images =", sum(len(v) for v in paddy.values()))
print("water images =", sum(len(v) for v in water.values()))

print()
print("unique paddy org SHA256 =", len(paddy_shas))
print("unique water org SHA256 =", len(water_shas))

print()
print("byte-identical org images across datasets =", len(common))

both_positive = 0
paddy_positive_water_false = 0
paddy_false_water_positive = 0
both_false = 0

examples = []

for sha in common:
    for p in paddy[sha]:
        for w in water[sha]:

            p_pos = p["positive_pixel_count"] > 0
            w_pos = w["positive_pixel_count"] > 0

            if p_pos and w_pos:
                both_positive += 1
                category = "both_positive"

            elif p_pos and not w_pos:
                paddy_positive_water_false += 1
                category = "paddy_positive_water_false"

            elif not p_pos and w_pos:
                paddy_false_water_positive += 1
                category = "paddy_false_water_positive"

            else:
                both_false += 1
                category = "both_false"

            if len(examples) < 50:
                examples.append(
                    {
                        "paddy_id": p["source_image_id"],
                        "water_id": w["source_image_id"],
                        "paddy_positive_pixels": p["positive_pixel_count"],
                        "water_positive_pixels": w["positive_pixel_count"],
                        "category": category,
                    }
                )


print()
print("OVERLAP BREAKDOWN")
print("both positive =", both_positive)
print(
    "paddy positive / water all-ignore =",
    paddy_positive_water_false,
)
print(
    "paddy all-ignore / water positive =",
    paddy_false_water_positive,
)
print("both all-ignore =", both_false)

if examples:
    print()
    print("FIRST OVERLAP EXAMPLES")

    for x in examples:
        print(
            x["category"],
            "| paddy_id =", x["paddy_id"],
            "| water_id =", x["water_id"],
            "| paddy_positive =", x["paddy_positive_pixels"],
            "| water_positive =", x["water_positive_pixels"],
        )
