# geeked/yolo_server.py
import os
from typing import List

import cv2
import numpy as np
import onnxruntime as ort

from config import (
    DIRECTIONS,
    YOLO_IMG_SIZE,
    YOLO_CONF_THRES,
    YOLO_IOU_THRES,
    YOLO_MAX_DET,
)
from utils import get_logger

logger = get_logger(__name__)


class YoloService:
    def __init__(self, model_path: str = None):
        if model_path is None:
            here = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(here, "models", "icon_yolo.onnx")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型不存在: {model_path}")

        logger.info("加载 YOLO 模型: %s", model_path)

        self.session = ort.InferenceSession(
            model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.img_size = YOLO_IMG_SIZE
        self.conf_thres = YOLO_CONF_THRES
        self.iou_thres = YOLO_IOU_THRES
        self.max_det = YOLO_MAX_DET

    # ---------------- 预处理 ----------------

    def _letterbox(self, img, new_shape: int = 640):
        """保持宽高比的缩放 + 填充"""
        h, w = img.shape[:2]
        r = min(new_shape / h, new_shape / w)
        new_unpad = (int(round(w * r)), int(round(h * r)))
        dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]
        dw, dh = dw // 2, dh // 2

        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = dh, new_shape - new_unpad[1] - dh
        left, right = dw, new_shape - new_unpad[0] - dw
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        return img, r, (left, top)

    # ---------------- 推理 ----------------

    def detection(self, image_bytes: bytes) -> List[dict]:
        """
        输入: 图片字节流
        输出: [{"direction": "ld", "bbox": [x1,y1,x2,y2], "conf": 0.9}, ...]
              按置信度降序，最多 4 个
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            logger.error("图片解码失败，返回空结果")
            return []

        h0, w0 = img.shape[:2]

        # 预处理
        img_lb, r, (pad_w, pad_h) = self._letterbox(img, self.img_size)
        blob = img_lb[:, :, ::-1].transpose(2, 0, 1)
        blob = np.ascontiguousarray(blob, dtype=np.float32) / 255.0
        blob = blob[None, ...]

        # 推理
        outputs = self.session.run(None, {self.input_name: blob})
        # YOLOv8 输出形状: (1, 4+nc, 8400)，nc=8 时为 (1, 12, 8400)
        pred = outputs[0][0]  # (12, 8400)
        nc = pred.shape[0] - 4

        # 解析
        boxes, scores, class_ids = [], [], []
        for i in range(pred.shape[1]):
            cls_scores = pred[4:4 + nc, i]
            cls_id = int(np.argmax(cls_scores))
            conf = float(cls_scores[cls_id])
            if conf < self.conf_thres:
                continue

            cx, cy, bw, bh = pred[0, i], pred[1, i], pred[2, i], pred[3, i]
            x1 = (cx - bw / 2 - pad_w) / r
            y1 = (cy - bh / 2 - pad_h) / r
            x2 = (cx + bw / 2 - pad_w) / r
            y2 = (cy + bh / 2 - pad_h) / r

            boxes.append([float(x1), float(y1),
                          float(x2 - x1), float(y2 - y1)])
            scores.append(conf)
            class_ids.append(cls_id)

        # NMS 去重
        idx = cv2.dnn.NMSBoxes(
            boxes, scores, self.conf_thres, self.iou_thres
        )

        results = []
        if len(idx) > 0:
            for i in idx.flatten():
                x, y, w, h = boxes[i]
                results.append({
                    "direction": DIRECTIONS[class_ids[i]],
                    "bbox": [
                        max(0, int(x)),
                        max(0, int(y)),
                        min(w0, int(x + w)),
                        min(h0, int(y + h)),
                    ],
                    "conf": scores[i],
                })

        # 按置信度降序，最多 4 个
        results.sort(key=lambda d: d["conf"], reverse=True)
        results = results[:self.max_det]

        logger.debug("YOLO 检测: %d 个框 (过滤后)", len(results))

        return results