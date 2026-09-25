# geeked/icon.py
from typing import List, Dict, Any, Optional

import requests

from config import DIRECTIONS, ICON_MAPPING, IMG_W, IMG_H
from utils import get_logger

logger = get_logger(__name__)


class IconSolver:
    # 8 方向环形顺序（从 config 导入）
    DIRS = DIRECTIONS

    # 背景图尺寸（从 config 导入）
    IMG_W = IMG_W
    IMG_H = IMG_H

    def __init__(self, imgs: str, ques: List[str]):
        self.imgs = self.load_image(f'https://static.geetest.com/{imgs}')
        self.ques = ques

        logger.debug("ques 原始: %s", ques)
        for q in ques:
            fname = q.split('/')[-1]
            direction = ICON_MAPPING.get(fname, '未命中')
            logger.debug("  %s → %s", fname, direction)

    # ---------------- 静态工具 ----------------

    @staticmethod
    def load_image(url: str) -> bytes:
        """从 URL 下载图片，返回字节流"""
        logger.debug("下载图片: %s", url)
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.content

    @staticmethod
    def test() -> None:
        """独立测试入口"""
        identifier = IconSolver(
            imgs="captcha_v4/policy/87d2c0d959/icon/163935/2025-04-20T17/3f99716d17324c9ba3eec402121eb1d9.jpg",
            ques=[
                "nerualpic/original_icon_pic/icon_20201215/315ce8665e781dabcd1eb09d3e604803.png",
                "nerualpic/original_icon_pic/icon_20201215/38bd9dda695098c7dfad74c921923a7d.png",
                "nerualpic/original_icon_pic/icon_20201215/cb0eaa639b2117a69a81af3d8c1496a1.png",
            ],
        )
        result = identifier.find_icon_position()
        logger.info("Result: %s", result)

    # ---------------- 内部方法 ----------------

    def _get_directions(self) -> List[Dict[str, Any]]:
        """把 ques 里的图标文件名转换成方向列表"""
        return [
            {'direction': ICON_MAPPING.get(q.split('/')[-1], '')}
            for q in self.ques
        ]

    @classmethod
    def _direction_distance(cls, d1: str, d2: str) -> int:
        """计算两个方向在 8 方向环上的最短距离（0=相同，4=相反）"""
        if d1 not in cls.DIRS or d2 not in cls.DIRS:
            return 999
        i1, i2 = cls.DIRS.index(d1), cls.DIRS.index(d2)
        diff = abs(i1 - i2)
        return min(diff, 8 - diff)

    # ---------------- 主流程 ----------------

    def find_icon_position(self) -> List[List[float]]:
        """
        1. 用 YOLO 检测背景图中所有图标及方向
        2. 按 ques 顺序精确匹配方向
        3. 缺失的方向用「方向最接近」的候选兜底
        4. bbox 中心坐标转 10000 归一化

        返回: [[x1, y1], [x2, y2], [x3, y3]]
        """
        from .yolo_server import YoloService

        yolo = YoloService()
        detections = yolo.detection(self.imgs)

        logger.debug("检测结果:")
        for d in detections:
            logger.debug(
                "  %s  bbox=%s  conf=%.3f",
                d['direction'], d['bbox'], d['conf']
            )

        box_directions = self._get_directions()
        logger.debug("ques 需要:")
        for boxd in box_directions:
            logger.debug("  %s", boxd['direction'])

        # 按置信度降序，最多保留 4 个
        detections.sort(key=lambda d: d["conf"], reverse=True)
        detections = detections[:4]

        # -------- 第一步：精确匹配 --------
        used = set()
        results: List[Optional[List[int]]] = []

        for boxd in box_directions:
            need = boxd["direction"]
            matched = None
            for i, det in enumerate(detections):
                if i in used:
                    continue
                if det["direction"] == need:
                    matched = det
                    used.add(i)
                    break
            results.append(matched["bbox"] if matched else None)

        # -------- 第二步：兜底 --------
        unused = [det for i, det in enumerate(detections) if i not in used]
        for i, r in enumerate(results):
            if r is None and unused:
                need = box_directions[i]["direction"]
                best = min(
                    unused,
                    key=lambda d: self._direction_distance(
                        d["direction"], need
                    )
                )
                results[i] = best["bbox"]
                unused.remove(best)
                logger.warning(
                    "兜底: 需要 %s, 用 %s 填补", need, best["direction"]
                )

        # -------- 第三步：bbox 转 center（10000 归一化）--------
        final: List[List[float]] = []
        for r in results:
            if r is None:
                continue
            x1, y1, x2, y2 = r
            center = [
                (x1 + (x2 - x1) / 2) * (10000 / self.IMG_W),
                (y1 + (y2 - y1) / 2) * (10000 / self.IMG_H),
            ]
            final.append(center)

        return final


if __name__ == '__main__':
    IconSolver.test()