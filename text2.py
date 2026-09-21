import rasterio

p = r"C:\OpenEarthMap_PoC\oemsar_data\trainval\val\sar_images\ValArea_016.tif"

with rasterio.open(p) as ds:
    print("CRS:", ds.crs)
    print("bounds:", ds.bounds)
    print("profile:", ds.profile)

    print("\n=== TAGS ===")
    for k, v in ds.tags().items():
        print(k, "=", v)

    print("\n=== BAND TAGS ===")
    for i in range(1, ds.count + 1):
        print("Band", i, ds.tags(i))
