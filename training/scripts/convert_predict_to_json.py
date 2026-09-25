# training/scripts/convert_predict_to_json.py
"""
把 YOLO 预测结果（txt）转成 X-AnyLabeling 的 JSON 格式。

用法：
    python -m training.scripts.convert_predict_to_json
    python -m training.scripts.convert_predict_to_json icon_v8_auto
"""
import os
import sys
import json
from PIL import Image

from config import (
    IMAGES_ALL, DIRECTIONS, HUMAN_END,
    PREDICT_BASE, PREDICT_NAME,
)

# 支持命令行覆盖 PREDICT_NAME
if len(sys.argv) > 1:
    PREDICT_NAME = sys.argv[1]

PRED_DIR = os.path.join(PREDICT_BASE, PREDICT_NAME, "labels")
IMG_DIR = IMAGES_ALL


def is_human(stem):
    try:
        return int(stem) <= HUMAN_END
    except ValueError:
        return False


def main():
    if not os.path.isdir(PRED_DIR):
        print(f"❌ 预测目录不存在: {PRED_DIR}")
        print(f"   请检查 config.py 里的 PREDICT_NAME 或命令行参数")
        return

    converted = 0
    skipped_human = 0
    skipped_missing_img = 0
    skipped_empty = 0

    for fname in sorted(os.listdir(PRED_DIR)):
        if not fname.endswith(".txt"):
            continue

        stem = os.path.splitext(fname)[0]

        if is_human(stem):
            skipped_human += 1
            continue

        txt_path = os.path.join(PRED_DIR, fname)

        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            p = os.path.join(IMG_DIR, stem + ext)
            if os.path.exists(p):
                img_path = p
                break
        if not img_path:
            skipped_missing_img += 1
            continue

        with Image.open(img_path) as im:
            img_w, img_h = im.size

        # 读取预测
        items = []
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                cls_id = int(parts[0])
                xc, yc, w, h = map(float, parts[1:5])
                conf = float(parts[5])
                items.append((cls_id, xc, yc, w, h, conf))

        # 按置信度降序，最多保留 4 个
        items.sort(key=lambda x: x[5], reverse=True)
        items = items[:4]

        if not items:
            skipped_empty += 1
            continue

        shapes = []
        for cls_id, xc, yc, w, h, conf in items:
            if cls_id >= len(DIRECTIONS):
                continue

            x1 = (xc - w / 2) * img_w
            y1 = (yc - h / 2) * img_h
            x2 = (xc + w / 2) * img_w
            y2 = (yc + h / 2) * img_h

            x1 = max(0, min(img_w, x1))
            y1 = max(0, min(img_h, y1))
            x2 = max(0, min(img_w, x2))
            y2 = max(0, min(img_h, y2))

            shapes.append({
                "label": DIRECTIONS[cls_id],
                "score": conf,
                "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                "group_id": None,
                "description": "",
                "difficult": False,
                "shape_type": "rectangle",
                "flags": {},
                "attributes": {},
                "kie_linking": [],
            })

        data = {
            "version": "4.0.6",
            "flags": {},
            "checked": False,
            "shapes": shapes,
            "imagePath": os.path.basename(img_path),
            "imageData": None,
            "imageHeight": img_h,
            "imageWidth": img_w,
            "description": "",
        }

        json_path = os.path.join(IMG_DIR, stem + ".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        converted += 1

    print("=" * 60)
    print(f"预测目录: {PRED_DIR}")
    print(f"人工标注上界: {HUMAN_END:06d}")
    print("=" * 60)
    print(f"更新 JSON: {converted}")
    print(f"跳过人工标注: {skipped_human}")
    print(f"跳过空预测: {skipped_empty}")
    print(f"找不到图片: {skipped_missing_img}")


if __name__ == "__main__":
    main()