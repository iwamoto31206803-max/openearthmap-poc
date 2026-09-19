from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image
import torch

from src.model import build_model, rgb_to_tensor


ORG_DIR = Path(
    r"C:\OpenEarthMap_PoC\data\gsi\raw\road_572\org"
)

LABEL_DIR = Path(
    r"C:\OpenEarthMap_PoC\data\gsi\prepared\road_572\labels"
)

BASE_MODEL = Path(
    r"C:\OpenEarthMap_PoC\OpenEarthMap-SAR\src\Semantic_Segemtation"
    r"\pretrained\RGB_Real_5_u-efficientnet-b4.pth"
)

DEVICE = "cpu"
NUM_THREADS = 2
TARGET_CLASS = 4

CLASS_NAMES = {
    0: "Background / Unlabelled",
    1: "Bareland",
    2: "Grass / Rangeland",
    3: "Pavement / Developed space",
    4: "Road",
    5: "Tree",
    6: "Water",
    7: "Agriculture",
    8: "Buildings",
}


torch.set_num_threads(NUM_THREADS)

model = build_model(BASE_MODEL, DEVICE)
model.eval()

image_paths = {
    p.relative_to(ORG_DIR): p
    for p in ORG_DIR.rglob("*.png")
}

label_paths = {
    p.relative_to(LABEL_DIR): p
    for p in LABEL_DIR.rglob("*.png")
}

if image_paths.keys() != label_paths.keys():
    missing_labels = image_paths.keys() - label_paths.keys()
    missing_images = label_paths.keys() - image_paths.keys()

    raise RuntimeError(
        f"Pairing mismatch: "
        f"missing_labels={len(missing_labels)}, "
        f"missing_images={len(missing_images)}"
    )


argmax_counts = Counter()

probability_sums = np.zeros(9, dtype=np.float64)

positive_pixels = 0
positive_images = 0


with torch.no_grad():

    for index, relative in enumerate(sorted(image_paths), start=1):

        with Image.open(image_paths[relative]) as image:
            rgb = np.asarray(
                image.convert("RGB"),
                dtype=np.uint8,
            )

        with Image.open(label_paths[relative]) as image:
            label = np.asarray(
                image,
                dtype=np.uint8,
            )

        positive = label == TARGET_CLASS

        count = int(positive.sum())

        if count == 0:
            continue

        positive_images += 1
        positive_pixels += count

        tensor = (
            rgb_to_tensor(rgb)
            .unsqueeze(0)
            .to(DEVICE)
        )

        logits = model(tensor)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0]

        predicted = torch.argmax(
            logits,
            dim=1,
        )[0]

        positive_t = torch.from_numpy(
            positive
        ).to(DEVICE)

        selected_predictions = predicted[
            positive_t
        ]

        counts = torch.bincount(
            selected_predictions,
            minlength=9,
        ).cpu().numpy()

        for class_id, class_count in enumerate(counts):
            argmax_counts[class_id] += int(class_count)

        for class_id in range(9):
            probability_sums[class_id] += (
                probabilities[class_id][positive_t]
                .double()
                .sum()
                .item()
            )

        if index % 100 == 0:
            print(
                f"processed {index}/{len(image_paths)} images; "
                f"positive images={positive_images}; "
                f"positive pixels={positive_pixels}"
            )


print()
print("=" * 72)
print("GSI ROAD POSITIVE — ORIGINAL BASE CONFLICT DIAGNOSIS")
print("=" * 72)

print()
print("INVENTORY")
print("image count =", len(image_paths))
print("positive-bearing images =", positive_images)
print("positive pixels =", positive_pixels)

print()
print("BASE ARGMAX ON ROAD POSITIVE PIXELS")

for class_id in range(9):

    count = argmax_counts[class_id]

    percentage = (
        100.0 * count / positive_pixels
        if positive_pixels
        else 0.0
    )

    print(
        f"{class_id} {CLASS_NAMES[class_id]}: "
        f"{count:,} "
        f"({percentage:.4f}%)"
    )


print()
print("MEAN BASE PROBABILITY ON ROAD POSITIVE PIXELS")

for class_id in range(9):

    mean_probability = (
        probability_sums[class_id] / positive_pixels
        if positive_pixels
        else 0.0
    )

    print(
        f"{class_id} {CLASS_NAMES[class_id]}: "
        f"{mean_probability:.6f}"
    )


print()
print("=" * 72)

print(
    "Road argmax percent =",
    100.0 * argmax_counts[TARGET_CLASS] / positive_pixels,
)

print(
    "Road mean probability =",
    probability_sums[TARGET_CLASS] / positive_pixels,
)

print("=" * 72)
