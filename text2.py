import csv
import math
from collections import defaultdict

csv_path = r"C:\OpenEarthMap_PoC\oemsar_data\val_geography.csv"

rows = []

with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        if r["country"] == "Japan":
            rows.append(
                {
                    "file": r["file"],
                    "crs": r["crs"],
                    "lon": float(r["lon"]),
                    "lat": float(r["lat"]),
                }
            )

def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0088
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)

    a = (
        math.sin(dp / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    )

    return 2 * R * math.asin(math.sqrt(a))

threshold_km = 30.0

# 単純な連結成分クラスタリング
n = len(rows)
visited = [False] * n
clusters = []

for i in range(n):
    if visited[i]:
        continue

    stack = [i]
    visited[i] = True
    members = []

    while stack:
        j = stack.pop()
        members.append(j)

        for k in range(n):
            if visited[k]:
                continue

            d = haversine_km(
                rows[j]["lon"],
                rows[j]["lat"],
                rows[k]["lon"],
                rows[k]["lat"],
            )

            if d <= threshold_km:
                visited[k] = True
                stack.append(k)

    clusters.append(members)

# 大きい順
clusters.sort(key=len, reverse=True)

print("=== JAPAN REGION CANDIDATES ===")
print("threshold_km =", threshold_km)
print("num_clusters =", len(clusters))
print()

for idx, members in enumerate(clusters, start=1):
    lons = [rows[i]["lon"] for i in members]
    lats = [rows[i]["lat"] for i in members]

    center_lon = sum(lons) / len(lons)
    center_lat = sum(lats) / len(lats)

    print(
        f"Region {idx}: n={len(members)}, "
        f"center=({center_lat:.6f}, {center_lon:.6f})"
    )

    for i in sorted(members, key=lambda x: rows[x]["file"]):
        r = rows[i]
        print(
            f"  {r['file']} "
            f"lat={r['lat']:.6f} lon={r['lon']:.6f} "
            f"crs={r['crs']}"
        )

    print()

# CSV出力
out_csv = r"C:\OpenEarthMap_PoC\oemsar_data\val_japan_regions.csv"

with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(
        ["region_id", "file", "crs", "lon", "lat"]
    )

    for idx, members in enumerate(clusters, start=1):
        for i in members:
            r = rows[i]
            writer.writerow(
                [
                    idx,
                    r["file"],
                    r["crs"],
                    r["lon"],
                    r["lat"],
                ]
            )

print("CSV:", out_csv)
