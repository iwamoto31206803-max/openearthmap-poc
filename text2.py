from pathlib import Path
import hashlib

import numpy as np
from PIL import Image


ROOT = Path(r"C:\OpenEarthMap_PoC\data\gsi")

PADDY_ORG = ROOT / "raw" / "paddy_572" / "org"
PADDY_LABEL = ROOT / "prepared" / "paddy_572" / "labels"

WATER_ORG = ROOT / "working" / "water_572_fixed" / "org"
WATER_LABEL = ROOT / "prepared" / "water_572" / "labels"

ROAD_ORG = ROOT / "raw" / "road_572" / "org"
ROAD_LABEL = ROOT / "prepared" / "road_572" / "labels"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def index_by_hash(org_dir, label_dir):
    result = {}

    labels = {
        p.relative_to(label_dir): p
        for p in label_dir.rglob("*.png")
    }

    for image in org_dir.rglob("*.png"):
        rel = image.relative_to(org_dir)

        if rel not in labels:
            raise RuntimeError(f"Missing label: {rel}")

        digest = sha256(image)

        result[digest] = {
            "image": image,
            "label": labels[rel],
            "id": rel.as_posix(),
        }

    return result


paddy = index_by_hash(PADDY_ORG, PADDY_LABEL)
water = index_by_hash(WATER_ORG, WATER_LABEL)
road = index_by_hash(ROAD_ORG, ROAD_LABEL)


def inspect_overlap(name, left, left_class, right, right_class):

    shared = sorted(set(left) & set(right))

    print()
    print("=" * 72)
    print(name)
    print("=" * 72)
    print("shared images =", len(shared))

    both_positive_images = 0
    total_left_positive = 0
    total_right_positive = 0
    total_pixel_overlap = 0

    for digest in shared:

        a = left[digest]
        b = right[digest]

        with Image.open(a["label"]) as im:
            la = np.asarray(im, dtype=np.uint8)

        with Image.open(b["label"]) as im:
            lb = np.asarray(im, dtype=np.uint8)

        if la.shape != lb.shape:
            raise RuntimeError("Shared source image has different label dimensions")

        pa = la == left_class
        pb = lb == right_class

        na = int(pa.sum())
        nb = int(pb.sum())
        overlap = int((pa & pb).sum())

        total_left_positive += na
        total_right_positive += nb
        total_pixel_overlap += overlap

        if na > 0 and nb > 0:
            both_positive_images += 1

        print(
            f"{a['id']} <-> {b['id']}: "
            f"left_positive={na}, "
            f"right_positive={nb}, "
            f"pixel_overlap={overlap}"
        )

    print()
    print("both-positive images =", both_positive_images)
    print("left positive pixels =", total_left_positive)
    print("right positive pixels =", total_right_positive)
    print("positive-mask pixel overlap =", total_pixel_overlap)


inspect_overlap(
    "PADDY vs ROAD",
    paddy,
    7,
    road,
    4,
)

inspect_overlap(
    "WATER vs ROAD",
    water,
    6,
    road,
    4,
)
