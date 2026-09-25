# training/scripts/filter_4_boxes.py
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.config import LABELS_ALL

fixed_more = 0      # 超过 4 个，截断
kept_less = 0       # 少于 4 个，保留
ok = 0              # 正好 4 个

for fname in os.listdir(LABELS_ALL):
    if not fname.endswith(".txt"):
        continue
    path = os.path.join(LABELS_ALL, fname)

    with open(path, "r") as fp:
        lines = [l.strip() for l in fp if l.strip()]

    n = len(lines)

    if n == 4:
        ok += 1
        continue

    if n > 4:
        # 按置信度降序排序
        def get_conf(line):
            parts = line.split()
            return float(parts[5]) if len(parts) >= 6 else 1.0

        lines.sort(key=get_conf, reverse=True)
        lines = lines[:4]

        with open(path, "w") as fp:
            fp.write("\n".join(lines))
        fixed_more += 1
    else:
        # 少于 4 个，保留现状，后续人工补
        kept_less += 1

print(f"正好 4 个: {ok}")
print(f"超过 4 个已截断: {fixed_more}")
print(f"少于 4 个（需人工补）: {kept_less}")