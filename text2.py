from pathlib import Path
import csv

audit = Path(
    r"C:\OpenEarthMap_PoC\data\gsi\prepared\water_572\audit.csv"
)

rows = []

with audit.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        n = int(row["non_label_mismatch_count"])

        if n > 0:
            rows.append(
                (
                    row["source_image_id"],
                    n,
                    row["positive_pixel_count"],
                    row["org_exact_label_color_count"],
                )
            )

print("images with non-label mismatch =", len(rows))
print("total non-label mismatch =", sum(x[1] for x in rows))

for row in rows:
    print(
        "image =", row[0],
        "non_label_mismatch =", row[1],
        "positive_pixels =", row[2],
        "org_exact_label_color =", row[3],
    )
