from pathlib import Path
import json

import numpy as np
from PIL import Image
import torch

from src.config import CLASS_NAMES
from src.model import build_model, rgb_to_tensor


ORG_DIR = Path(
    r"C:\OpenEarthMap_PoC\data\gsi\working\water_572_fixed\org"
)

LABEL_DIR = Path(
    r"C:\OpenEarthMap_PoC\data\gsi\prepared\water_572\labels"
)

BASE_MODEL = Path(
    r"C:\OpenEarthMap_PoC\OpenEarthMap-SAR\src\Semantic_Segemtation"
    r"\pretrained\RGB_Real_5_u-efficientnet-b4.pth"
)

OUTPUT_JSON = Path(
    r"C:\OpenEarthMap_PoC\data\gsi\prepared\water_572"
    r"\base_conflict_diagnosis.json"
)

TARGET_CLASS = 6
AGRICULTURE_CLASS = 7
IGNORE_INDEX = 255


device = "cpu"

torch.set_num_threads(2)

print("Loading Base model...")
model = build_model(BASE_MODEL, device=device)

argmax_counts = np.zeros(9, dtype=np.int64)

positive_pixel_count = 0
sum_water_probability = 0.0
sum_agriculture_probability = 0.0

positive_image_count = 0
all_ignore_image_count = 0
processed = 0


label_paths = sorted(LABEL_DIR.rglob("*.png"))

print("label files =", len(label_paths))


with torch.no_grad():

    for label_path in label_paths:

        relative = label_path.relative_to(LABEL_DIR)
        org_path = ORG_DIR / relative

        if not org_path.exists():
            raise FileNotFoundError(org_path)

        with Image.open(label_path) as img:
            label = np.asarray(img, dtype=np.uint8)

        positive_mask_np = label == TARGET_CLASS

        n_positive = int(positive_mask_np.sum())

        if n_positive == 0:
            all_ignore_image_count += 1
            processed += 1
            continue

        positive_image_count += 1

        with Image.open(org_path) as img:
            rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)

        if rgb.shape[:2] != label.shape:
            raise ValueError(
                f"Size mismatch: {relative}: "
                f"image={rgb.shape[:2]}, label={label.shape}"
            )

        tensor = rgb_to_tensor(rgb).unsqueeze(0).to(device)

        logits = model(tensor)

        probs = torch.softmax(logits, dim=1)[0]

        pred = torch.argmax(probs, dim=0)

        mask = torch.from_numpy(positive_mask_np).to(device)

        pred_positive = pred[mask]

        counts = torch.bincount(
            pred_positive,
            minlength=9
        ).cpu().numpy()

        argmax_counts += counts

        n = int(mask.sum().item())

        positive_pixel_count += n

        sum_water_probability += float(
            probs[TARGET_CLASS][mask].sum().item()
        )

        sum_agriculture_probability += float(
            probs[AGRICULTURE_CLASS][mask].sum().item()
        )

        processed += 1

        if processed % 50 == 0:
            print(
                f"processed={processed}/{len(label_paths)} "
                f"positive_images={positive_image_count} "
                f"positive_pixels={positive_pixel_count}"
            )


if positive_pixel_count == 0:
    raise RuntimeError("No Water positive pixels found")


argmax_percent = (
    argmax_counts.astype(np.float64)
    / positive_pixel_count
    * 100.0
)

mean_water_probability = (
    sum_water_probability
    / positive_pixel_count
)

mean_agriculture_probability = (
    sum_agriculture_probability
    / positive_pixel_count
)


print()
print("=" * 72)
print("GSI WATER POSITIVE x ORIGINAL BASE DIAGNOSIS")
print("=" * 72)

print("images processed =", processed)
print("positive-bearing images =", positive_image_count)
print("all-ignore images =", all_ignore_image_count)
print("Water positive pixels =", positive_pixel_count)

print()
print("Base argmax distribution on GSI Water positive pixels:")
print()

for class_id in range(9):
    print(
        f"{class_id}: {CLASS_NAMES[class_id]:28s} "
        f"{argmax_counts[class_id]:12d} "
        f"{argmax_percent[class_id]:8.4f}%"
    )

print()
print(
    "mean Base Water probability =",
    mean_water_probability
)

print(
    "mean Base Agriculture probability =",
    mean_agriculture_probability
)


result = {
    "target": {
        "gsi_category": "water",
        "oem_class_id": TARGET_CLASS,
        "oem_class_name": CLASS_NAMES[TARGET_CLASS],
    },
    "counts": {
        "images_processed": processed,
        "positive_bearing_images": positive_image_count,
        "all_ignore_images": all_ignore_image_count,
        "positive_pixel_count": positive_pixel_count,
    },
    "base_argmax": {
        str(class_id): {
            "class_name": CLASS_NAMES[class_id],
            "pixel_count": int(argmax_counts[class_id]),
            "percent": float(argmax_percent[class_id]),
        }
        for class_id in range(9)
    },
    "probabilities": {
        "mean_base_water_probability": mean_water_probability,
        "mean_base_agriculture_probability": mean_agriculture_probability,
    },
    "notes": [
        "Read-only diagnostic.",
        "Only GSI Water positive pixels are evaluated.",
        "All-ignore pixels are not treated as Water-negative.",
        "No training is performed.",
    ],
}


with OUTPUT_JSON.open("w", encoding="utf-8") as f:
    json.dump(
        result,
        f,
        ensure_ascii=False,
        indent=2,
    )
    f.write("\n")


print()
print("saved =", OUTPUT_JSON)
