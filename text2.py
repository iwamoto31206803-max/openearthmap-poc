from pathlib import Path
import hashlib
import random


SEED = 42
TRAIN_RATIO = 0.8

ROOT = Path(r"C:\OpenEarthMap_PoC\data\gsi")

PADDY_ORG = ROOT / "raw" / "paddy_572" / "org"
WATER_ORG = ROOT / "working" / "water_572_fixed" / "org"
ROAD_ORG = ROOT / "raw" / "road_572" / "org"

ROAD_LABELS = ROOT / "prepared" / "road_572" / "labels"

ROAD_CLASS = 4
IGNORE = 255


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def image_index(directory):
    return sorted(
        p for p in directory.rglob("*.png")
        if p.is_file()
    )


# ------------------------------------------------------------
# Road positive-bearing / all-ignore split
# ------------------------------------------------------------

from PIL import Image
import numpy as np

road_images = {
    p.relative_to(ROAD_ORG): p
    for p in image_index(ROAD_ORG)
}

road_labels = {
    p.relative_to(ROAD_LABELS): p
    for p in image_index(ROAD_LABELS)
}

if road_images.keys() != road_labels.keys():
    raise RuntimeError("Road image/label pairing mismatch")

positive_ids = []
all_ignore_ids = []

for rel in sorted(road_images):
    with Image.open(road_labels[rel]) as im:
        label = np.asarray(im, dtype=np.uint8)

    if np.any(label == ROAD_CLASS):
        positive_ids.append(rel.as_posix())
    else:
        all_ignore_ids.append(rel.as_posix())


ids = sorted(positive_ids)
rng = random.Random(SEED)
rng.shuffle(ids)

n_train = int(len(ids) * TRAIN_RATIO)
n_train = max(1, min(n_train, len(ids) - 1))

road_train = ids[:n_train]
road_val = ids[n_train:]

# Pilot subset: exactly one Road sample per 1028 logical steps.
PILOT_ROAD_COUNT = 1028

if len(road_train) < PILOT_ROAD_COUNT:
    raise RuntimeError("Road train pool smaller than pilot target")

pilot_rng = random.Random(SEED)
road_pilot_train = sorted(
    pilot_rng.sample(road_train, PILOT_ROAD_COUNT)
)


print("=" * 72)
print("ROAD SPLIT")
print("=" * 72)
print("positive-bearing =", len(positive_ids))
print("all-ignore =", len(all_ignore_ids))
print("train pool =", len(road_train))
print("validation =", len(road_val))
print("Pilot Road samples =", len(road_pilot_train))
print("unused train-pool samples in 1-epoch Pilot =", len(road_train) - len(road_pilot_train))


# ------------------------------------------------------------
# Byte-identical source-image overlap
# ------------------------------------------------------------

def hash_set(directory):
    files = image_index(directory)
    hashes = {sha256(p) for p in files}
    return len(files), hashes


print()
print("=" * 72)
print("SOURCE IMAGE SHA256 OVERLAP")
print("=" * 72)

paddy_n, paddy_hash = hash_set(PADDY_ORG)
water_n, water_hash = hash_set(WATER_ORG)
road_n, road_hash = hash_set(ROAD_ORG)

print("Paddy images =", paddy_n, "unique SHA =", len(paddy_hash))
print("Water images =", water_n, "unique SHA =", len(water_hash))
print("Road images =", road_n, "unique SHA =", len(road_hash))

print()
print("Paddy ∩ Road =", len(paddy_hash & road_hash))
print("Water ∩ Road =", len(water_hash & road_hash))
print("Paddy ∩ Water =", len(paddy_hash & water_hash))

print()
print("=" * 72)
