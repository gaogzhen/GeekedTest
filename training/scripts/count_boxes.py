# training/scripts/count_boxes.py
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LABELS_ALL

counter = Counter()
for f in os.listdir(LABELS_ALL):
    if not f.endswith(".txt"):
        continue
    with open(os.path.join(LABELS_ALL, f)) as fp:
        n = sum(1 for line in fp if line.strip())
    counter[n] += 1

print("框数量分布:")
for n in sorted(counter.keys()):
    print(f"  {n} 个框: {counter[n]} 张")