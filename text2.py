from pathlib import Path
import json
from collections import Counter

import torch
from torch.utils.data import DataLoader

from src.model import build_model
from src.training.train_gsi_phase_b import (
    ROAD_CLASS,
    RoadDataset,
    scan_road_dataset,
)
from src.training import replay_preservation as replay


ROOT = Path(r"C:\OpenEarthMap_PoC")

ROAD_ORG = ROOT / "data" / "gsi" / "raw" / "road_572" / "org"
ROAD_PREPARED = ROOT / "data" / "gsi" / "prepared" / "road_572"

RUN = (
    ROOT
    / "openearthmap-poc"
    / "training_outputs"
    / "gsi_phase_b_v01"
    / "20260919T151035_437811Z_0fd54ade"
)

VALIDATION_IDS = RUN / "road_positive_validation_ids.json"

BASE = (
    ROOT
    / "OpenEarthMap-SAR"
    / "src"
    / "Semantic_Segemtation"
    / "pretrained"
    / "RGB_Real_5_u-efficientnet-b4.pth"
)

FT = RUN / "checkpoints" / "best.pth"

DEVICE = "cpu"
NUM_THREADS = 2


CLASS_NAMES = {
    0: "Background",
    1: "Bareland",
    2: "Grass",
    3: "Pavement",
    4: "Road",
    5: "Tree",
    6: "Water",
    7: "Agriculture",
    8: "Buildings",
}


torch.set_num_threads(NUM_THREADS)


# ------------------------------------------------------------
# Reconstruct exactly the Road validation set used in the Pilot
# ------------------------------------------------------------

validation_ids = set(
    json.loads(VALIDATION_IDS.read_text(encoding="utf-8"))
)

road_positive, road_all_ignore = scan_road_dataset(
    ROAD_ORG,
    ROAD_PREPARED / "labels",
)

validation_samples = [
    sample
    for sample in road_positive
    if sample.source_image_id in validation_ids
]

validation_samples = sorted(
    validation_samples,
    key=lambda x: x.source_image_id,
)

if len(validation_samples) != len(validation_ids):
    raise RuntimeError(
        f"Validation reconstruction failed: "
        f"{len(validation_samples)} samples vs "
        f"{len(validation_ids)} IDs"
    )

if len(validation_samples) != 327:
    raise RuntimeError(
        f"Expected 327 Road validation samples, "
        f"got {len(validation_samples)}"
    )


loader = DataLoader(
    RoadDataset(validation_samples),
    batch_size=1,
    shuffle=False,
    num_workers=0,
    collate_fn=replay.checked_collate,
)


# ------------------------------------------------------------
# Models
# ------------------------------------------------------------

print("Loading Original Base...")
base_model = build_model(BASE, device=DEVICE)
base_model.eval()

print("Loading Phase B v0.3 Road...")
ft_model = build_model(FT, device=DEVICE)
ft_model.eval()


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

positive_pixels = 0

base_road_correct = 0
ft_road_correct = 0

base_road_prob_sum = 0.0
ft_road_prob_sum = 0.0

base_distribution = Counter()
ft_distribution = Counter()

transition = Counter()


with torch.inference_mode():

    for index, batch in enumerate(loader, start=1):

        images, labels, image_mask = batch

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        image_mask = image_mask.to(DEVICE)

        positive = image_mask & (labels == ROAD_CLASS)

        n_positive = int(positive.sum())

        if n_positive == 0:
            raise RuntimeError(
                "All-ignore sample reached Road validation"
            )

        base_logits = base_model(images)
        ft_logits = ft_model(images)

        base_prob = torch.softmax(base_logits, dim=1)
        ft_prob = torch.softmax(ft_logits, dim=1)

        base_pred = base_logits.argmax(dim=1)
        ft_pred = ft_logits.argmax(dim=1)

        positive_pixels += n_positive

        base_road_correct += int(
            ((base_pred == ROAD_CLASS) & positive).sum()
        )

        ft_road_correct += int(
            ((ft_pred == ROAD_CLASS) & positive).sum()
        )

        base_road_prob_sum += float(
            base_prob[:, ROAD_CLASS][positive]
            .double()
            .sum()
        )

        ft_road_prob_sum += float(
            ft_prob[:, ROAD_CLASS][positive]
            .double()
            .sum()
        )

        base_values = base_pred[positive].cpu().tolist()
        ft_values = ft_pred[positive].cpu().tolist()

        base_distribution.update(base_values)
        ft_distribution.update(ft_values)

        transition.update(zip(base_values, ft_values))

        if index % 25 == 0 or index == len(validation_samples):
            print(
                f"Processed {index}/{len(validation_samples)}"
            )


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

base_agreement = base_road_correct / positive_pixels
ft_agreement = ft_road_correct / positive_pixels

base_mean_prob = base_road_prob_sum / positive_pixels
ft_mean_prob = ft_road_prob_sum / positive_pixels


print()
print("=" * 78)
print("ROAD VALIDATION — ORIGINAL BASE vs PHASE B v0.3")
print("=" * 78)

print("validation images =", len(validation_samples))
print("Road positive pixels =", positive_pixels)

print()
print("ROAD AGREEMENT")
print(
    f"Base = {base_agreement:.6f} "
    f"({base_agreement * 100:.3f}%)"
)
print(
    f"FT   = {ft_agreement:.6f} "
    f"({ft_agreement * 100:.3f}%)"
)
print(
    f"delta = "
    f"{(ft_agreement - base_agreement) * 100:+.3f} pp"
)

print()
print("ROAD MEAN PROBABILITY")
print(f"Base = {base_mean_prob:.6f}")
print(f"FT   = {ft_mean_prob:.6f}")
print(f"delta = {ft_mean_prob - base_mean_prob:+.6f}")


print()
print("=" * 78)
print("BASE ARGMAX ON ROAD POSITIVE PIXELS")
print("=" * 78)

for cls in range(9):
    count = base_distribution[cls]
    pct = count / positive_pixels * 100

    print(
        f"{cls} {CLASS_NAMES[cls]:12s}: "
        f"{count:10d}  {pct:8.4f}%"
    )


print()
print("=" * 78)
print("FT ARGMAX ON ROAD POSITIVE PIXELS")
print("=" * 78)

for cls in range(9):
    count = ft_distribution[cls]
    pct = count / positive_pixels * 100

    print(
        f"{cls} {CLASS_NAMES[cls]:12s}: "
        f"{count:10d}  {pct:8.4f}%"
    )


# ------------------------------------------------------------
# Important transitions
# ------------------------------------------------------------

print()
print("=" * 78)
print("IMPORTANT TRANSITIONS")
print("=" * 78)


def show_transition(src, dst):
    count = transition[(src, dst)]
    pct = count / positive_pixels * 100

    print(
        f"{src} {CLASS_NAMES[src]} "
        f"-> "
        f"{dst} {CLASS_NAMES[dst]}: "
        f"{count} ({pct:.4f}%)"
    )


show_transition(3, 4)  # Pavement -> Road
show_transition(4, 4)  # Road -> Road
show_transition(4, 3)  # Road -> Pavement
show_transition(7, 4)  # Agriculture -> Road
show_transition(6, 4)  # Water -> Road
show_transition(5, 4)  # Tree -> Road
show_transition(8, 4)  # Buildings -> Road


print()
print("=" * 78)
print("ALL BASE -> FT TRANSITIONS TO ROAD")
print("=" * 78)

for src in range(9):

    count = transition[(src, ROAD_CLASS)]

    if count == 0:
        continue

    pct = count / positive_pixels * 100

    print(
        f"{src} {CLASS_NAMES[src]:12s} "
        f"-> Road: "
        f"{count:10d}  {pct:8.4f}%"
    )


print()
print("=" * 78)
print("CHECK")
print("=" * 78)

print(
    "Expected FT validation agreement from run_manifest "
    "≈ 0.878671"
)

print(
    "Expected FT validation mean probability from run_manifest "
    "≈ 0.701173"
)

print()
print("Done.")
