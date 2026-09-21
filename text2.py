import rasterio

sar = r"C:\OpenEarthMap_PoC\oemsar_data\trainval\val\sar_images\ValArea_011.tif"
gt = r"C:\OpenEarthMap_PoC\oemsar_data\val_labels\val\labels\ValArea_011.tif"
out = r"C:\OpenEarthMap_PoC\oemsar_data\ValArea_011_gt_georef.tif"

with rasterio.open(sar) as s, rasterio.open(gt) as g:
    profile = g.profile.copy()
    profile.update(
        crs=s.crs,
        transform=s.transform
    )

    with rasterio.open(out, "w", **profile) as dst:
        dst.write(g.read())

print(out)
