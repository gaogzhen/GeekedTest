# test_yolo.py
from geeked.yolo_server import YoloService
import requests

yolo = YoloService()

url = "https://static.geetest.com/captcha_v4/policy/87d2c0d959/icon/310610/2026-09-24T22/3b6597262f284e8a9b2006a02b235d84.jpg"
img_bytes = requests.get(url).content

dets = yolo.detection(img_bytes)
print(f"检测到 {len(dets)} 个框")
for d in dets:
    print(f"  {d['direction']}  bbox={d['bbox']}  conf={d['conf']:.3f}")