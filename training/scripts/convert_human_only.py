# training/scripts/convert_human_only.py
import os
import json

from config import IMAGES_ALL, LABELS_ALL, DIRECTIONS, HUMAN_END

JSON_DIR = IMAGES_ALL
TXT_DIR  = LABELS_ALL
DIR2ID   = {d: i for i, d in enumerate(DIRECTIONS)}

# 清空旧的 txt（避免残留污染）
if os.path.isdir(TXT_DIR):
    for f in os.listdir(TXT_DIR):
        if f.endswith(".txt"):
            os.remove(os.path.join(TXT_DIR, f))
    print(f"已清空旧 txt: {TXT_DIR}")

os.makedirs(TXT_DIR, exist_ok=True)

converted = 0
skipped_no_shapes = 0
unknown = set()

for fname in sorted(os.listdir(JSON_DIR)):
    if not fname.endswith(".json"):
        continue

    stem = os.path.splitext(fname)[0]

    # 只处理 0 ~ HUMAN_END 的编号
    try:
        if int(stem) > HUMAN_END:
            continue
    except ValueError:
        continue

    json_path = os.path.join(JSON_DIR, fname)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_w = data.get("imageWidth")
    img_h = data.get("imageHeight")
    shapes = data.get("shapes", [])

    if not img_w or not img_h or not shapes:
        skipped_no_shapes += 1
        continue

    lines = []
    for shape in shapes:
        label = shape.get("label", "")

        # 兼容 "turtle_ld" 和 "ld" 两种格式
        parts = label.split("_")
        direction = parts[-1]
        if direction not in DIR2ID:
            direction = label if label in DIR2ID else None

        if not direction:
            unknown.add(label)
            continue

        cls_id = DIR2ID[direction]
        points = shape.get("points", [])
        if not points:
            continue

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        x_center = (xmin + xmax) / 2.0 / img_w
        y_center = (ymin + ymax) / 2.0 / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h

        # 边界裁剪
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))

        lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

    txt_path = os.path.join(TXT_DIR, stem + ".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    if lines:
        converted += 1

print(f"转换成功: {converted}")
print(f"跳过（无标注框）: {skipped_no_shapes}")
if unknown:
    print(f"未识别标签: {unknown}")