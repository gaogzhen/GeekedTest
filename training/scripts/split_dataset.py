# training/scripts/split_dataset.py
import os
import random
import shutil

from training.config import (
    IMAGES_ALL, LABELS_ALL,
    IMAGES_TRAIN, IMAGES_VAL,
    LABELS_TRAIN, LABELS_VAL,
)

VAL_RATIO = 0.2
random.seed(42)

# 确保目标目录存在
for d in [IMAGES_TRAIN, IMAGES_VAL, LABELS_TRAIN, LABELS_VAL]:
    os.makedirs(d, exist_ok=True)

# 收集有效样本（非空标签对应的图片）
valid_stems = []
for f in os.listdir(LABELS_ALL):
    if f.endswith(".txt") and os.path.getsize(os.path.join(LABELS_ALL, f)) > 0:
        valid_stems.append(os.path.splitext(f)[0])

print(f"有效样本: {len(valid_stems)}")

random.shuffle(valid_stems)
split = int(len(valid_stems) * (1 - VAL_RATIO))


def copy_pair(stem, img_dst, lbl_dst):
    for ext in [".jpg", ".jpeg", ".png"]:
        src = os.path.join(IMAGES_ALL, stem + ext)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(img_dst, stem + ext))
            break
    shutil.copy(os.path.join(LABELS_ALL, stem + ".txt"),
                os.path.join(lbl_dst, stem + ".txt"))


for s in valid_stems[:split]:
    copy_pair(s, IMAGES_TRAIN, LABELS_TRAIN)
for s in valid_stems[split:]:
    copy_pair(s, IMAGES_VAL, LABELS_VAL)

print(f"train: {split}, val: {len(valid_stems) - split}")