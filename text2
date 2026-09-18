from pathlib import Path
from PIL import Image

root = Path(r"C:\OpenEarthMap_PoC\data\gsi\raw\water_572")
org_dir = root / "org"
val_dir = root / "val"

mismatches = []
missing_val = []
checked = 0

for org_path in sorted(org_dir.glob("*.png")):
    val_path = val_dir / org_path.name

    if not val_path.exists():
        missing_val.append(org_path.name)
        continue

    with Image.open(org_path) as img:
        org_size = img.size

    with Image.open(val_path) as img:
        val_size = img.size

    checked += 1

    if org_size != val_size:
        mismatches.append(
            {
                "name": org_path.name,
                "org_size": org_size,
                "val_size": val_size,
            }
        )

print("pairs checked =", checked)
print("missing val =", len(missing_val))
print("size mismatches =", len(mismatches))

if missing_val:
    print("\nMissing val files:")
    for name in missing_val[:50]:
        print(name)

if mismatches:
    print("\nSize mismatches:")
    for item in mismatches[:50]:
        print(
            item["name"],
            "org =", item["org_size"],
            "val =", item["val_size"],
        )
