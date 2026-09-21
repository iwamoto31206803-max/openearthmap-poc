import os
import rasterio

base = r"C:\OpenEarthMap_PoC\oemsar_data"

ids = [
    "ValArea_011",
    "ValArea_065",
    "ValArea_008",
    "ValArea_016",
    "ValArea_038",
]

sar_dir = os.path.join(
    base,
    "trainval",
    "val",
    "sar_images",
)

gt_dir = os.path.join(
    base,
    "val_labels",
    "val",
    "labels",
)

out_dir = os.path.join(
    base,
    "val_gt_georef",
)

os.makedirs(out_dir, exist_ok=True)

for area_id in ids:
    sar = os.path.join(
        sar_dir,
        area_id + ".tif",
    )

    gt = os.path.join(
        gt_dir,
        area_id + ".tif",
    )

    out = os.path.join(
        out_dir,
        area_id + "_gt_georef.tif",
    )

    with rasterio.open(sar) as s, rasterio.open(gt) as g:
        if (s.width, s.height) != (g.width, g.height):
            raise RuntimeError(
                f"Size mismatch: {area_id}"
            )

        profile = g.profile.copy()
        profile.update(
            crs=s.crs,
            transform=s.transform,
        )

        with rasterio.open(out, "w", **profile) as dst:
            dst.write(g.read())

    print("created:", out)

print("done")
