from pathlib import Path

import numpy as np
from PIL import Image


root = Path(r"C:\OpenEarthMap_PoC\data\gsi\raw\water_572")

org_path = root / "org" / "554.png"
val_path = root / "val" / "554.png"

with Image.open(org_path) as img:
    org = np.asarray(img.convert("RGB"))

with Image.open(val_path) as img:
    val = np.asarray(img.convert("RGB"))

print("org shape =", org.shape)
print("val shape =", val.shape)
print()

label_color = np.array([0, 0, 255], dtype=np.uint8)

results = []

# 574 -> 572 なので、開始位置は縦横それぞれ 0, 1, 2 の9通り
for top in range(3):
    for left in range(3):
        crop = val[top:top + 572, left:left + 572]

        positive = np.all(crop == label_color, axis=2)

        # GSI valはラベル部分以外ではorgと同じであることを期待する。
        mismatch = np.any(org != crop, axis=2)

        # 青ラベル以外でorgと異なる画素数
        non_label_mismatch = mismatch & ~positive

        positive_count = int(positive.sum())
        mismatch_count = int(mismatch.sum())
        non_label_mismatch_count = int(non_label_mismatch.sum())

        results.append(
            (
                non_label_mismatch_count,
                top,
                left,
                positive_count,
                mismatch_count,
            )
        )

results.sort()

print("Candidates sorted by non-label mismatch:")
print()
print(
    "non_label_mismatch | top | left | blue_positive | all_mismatch"
)

for (
    non_label_mismatch_count,
    top,
    left,
    positive_count,
    mismatch_count,
) in results:
    print(
        f"{non_label_mismatch_count:18d} |"
        f" {top:3d} |"
        f" {left:4d} |"
        f" {positive_count:13d} |"
        f" {mismatch_count:12d}"
    )

best = results[0]

print()
print("BEST CANDIDATE")
print("top =", best[1])
print("left =", best[2])
print("non_label_mismatch =", best[0])
print("blue_positive =", best[3])
