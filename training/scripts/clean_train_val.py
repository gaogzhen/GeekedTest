# training/scripts/clear_train_val.py
import os
import sys
import shutil

# 把 training 加入 sys.path，导入 config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import IMAGES_TRAIN, IMAGES_VAL, LABELS_TRAIN, LABELS_VAL

DIRS = [IMAGES_TRAIN, IMAGES_VAL, LABELS_TRAIN, LABELS_VAL]

total_files = 0
total_dirs = 0

for d in DIRS:
    if not os.path.isdir(d):
        print(f"[不存在] {d}")
        continue

    removed_files = 0
    removed_dirs = 0

    for f in os.listdir(d):
        p = os.path.join(d, f)
        if os.path.isfile(p):
            os.remove(p)
            removed_files += 1
        elif os.path.isdir(p):
            shutil.rmtree(p)
            removed_dirs += 1

    print(f"[已清空] {d}  文件: {removed_files}, 子目录: {removed_dirs}")
    total_files += removed_files
    total_dirs += removed_dirs

print(f"\n共删除文件: {total_files}, 子目录: {total_dirs}")