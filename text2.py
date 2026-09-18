from pathlib import Path
from PIL import Image
import hashlib
import json
import os


src = Path(r"C:\OpenEarthMap_PoC\data\gsi\raw\water_572")
dst = Path(r"C:\OpenEarthMap_PoC\data\gsi\working\water_572_fixed")

repair_file = "554.png"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


for subdir in ["org", "val"]:
    (dst / subdir).mkdir(parents=True, exist_ok=True)

    for src_path in sorted((src / subdir).glob("*.png")):
        dst_path = dst / subdir / src_path.name

        if dst_path.exists():
            continue

        # val/554.png だけ修正
        if subdir == "val" and src_path.name == repair_file:
            with Image.open(src_path) as img:
                img = img.convert("RGB")

                if img.size != (574, 574):
                    raise ValueError(
                        f"Unexpected source size for {src_path}: {img.size}"
                    )

                corrected = img.crop((0, 0, 572, 572))
                corrected.save(dst_path)

        else:
            # 同一ドライブなのでhard linkを使用。
            # rawデータそのものは変更しない。
            os.link(src_path, dst_path)


src_org = src / "org" / repair_file
src_val = src / "val" / repair_file
dst_org = dst / "org" / repair_file
dst_val = dst / "val" / repair_file

with Image.open(src_val) as img:
    original_val_size = img.size

with Image.open(dst_val) as img:
    corrected_val_size = img.size

with Image.open(dst_org) as img:
    corrected_org_size = img.size


repair_manifest = {
    "source_dataset": str(src),
    "working_dataset": str(dst),
    "repair": {
        "file": repair_file,
        "reason": "GSI distributed val image is 574x574 while paired org is 572x572",
        "source_val_size": list(original_val_size),
        "corrected_val_size": list(corrected_val_size),
        "crop": {
            "left": 0,
            "top": 0,
            "right_exclusive": 572,
            "bottom_exclusive": 572,
        },
        "diagnostic": {
            "non_label_mismatch_after_crop": 0,
            "blue_positive_pixel_count": 12570,
        },
        "source_org_sha256": sha256(src_org),
        "source_val_sha256": sha256(src_val),
        "corrected_val_sha256": sha256(dst_val),
    },
}

with (dst / "repair_manifest.json").open(
    "w", encoding="utf-8"
) as f:
    json.dump(repair_manifest, f, ensure_ascii=False, indent=2)
    f.write("\n")


# 最終サイズ監査
mismatches = []

for org_path in sorted((dst / "org").glob("*.png")):
    val_path = dst / "val" / org_path.name

    with Image.open(org_path) as o:
        org_size = o.size

    with Image.open(val_path) as v:
        val_size = v.size

    if org_size != val_size:
        mismatches.append(
            (org_path.name, org_size, val_size)
        )


print("working dataset =", dst)
print("repair file =", repair_file)
print("original val size =", original_val_size)
print("corrected val size =", corrected_val_size)
print("org size =", corrected_org_size)
print("size mismatches after repair =", len(mismatches))
print("repair manifest =", dst / "repair_manifest.json")

if mismatches:
    print(mismatches[:20])
