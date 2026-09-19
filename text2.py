from pathlib import Path
import json
import hashlib

ROOT = Path(r"C:\OpenEarthMap_PoC")

GSI = ROOT / "data" / "gsi"

PADDY_ORG = GSI / "raw" / "paddy_572" / "org"
ROAD_ORG = GSI / "raw" / "road_572" / "org"

V02_RUN = (
    ROOT
    / "openearthmap-poc"
    / "training_outputs"
    / "gsi_phase_b_v01"
    / "20260919T053943_488504Z_000833f3"
)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_ids(filename):
    data = json.loads((V02_RUN / filename).read_text(encoding="utf-8"))

    # 保存形式がlistそのものでもdictでも対応
    if isinstance(data, list):
        return set(data)

    for key in ("ids", "sample_ids", "image_ids"):
        if key in data:
            return set(data[key])

    raise RuntimeError(
        f"Unknown ID-file structure: {filename}"
    )


paddy_positive_train = load_ids("paddy_train_ids.json")
paddy_positive_val = load_ids("paddy_validation_ids.json")
paddy_replay_train = load_ids("paddy_replay_train_ids.json")
paddy_replay_val = load_ids("paddy_replay_validation_ids.json")


def index(directory):
    result = {}

    for p in directory.rglob("*.png"):
        rel = p.relative_to(directory).with_suffix("").as_posix()

        result[sha256(p)] = {
            "id": rel,
            "path": p,
        }

    return result


paddy = index(PADDY_ORG)
road = index(ROAD_ORG)

shared = sorted(set(paddy) & set(road))

print("=" * 72)
print("PADDY / ROAD BYTE-IDENTICAL IMAGE SPLIT AUDIT")
print("=" * 72)
print("shared images =", len(shared))
print()

summary = {
    "paddy_positive_train": 0,
    "paddy_positive_val": 0,
    "paddy_replay_train": 0,
    "paddy_replay_val": 0,
    "not_found": 0,
}

for digest in shared:

    p = paddy[digest]
    r = road[digest]

    pid = p["id"]

    if pid in paddy_positive_train:
        split = "paddy_positive_train"

    elif pid in paddy_positive_val:
        split = "paddy_positive_val"

    elif pid in paddy_replay_train:
        split = "paddy_replay_train"

    elif pid in paddy_replay_val:
        split = "paddy_replay_val"

    else:
        split = "not_found"

    summary[split] += 1

    print(
        f"Paddy {p['id']} <-> Road {r['id']} : {split}"
    )


print()
print("SUMMARY")

for key, value in summary.items():
    print(key, "=", value)

print()
print("=" * 72)
