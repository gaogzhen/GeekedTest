# config.py
"""
项目全局配置。
运行时代码（geeked/）和训练代码（training/）共用。
"""
import os

# ==================== 项目根目录 ====================
ROOT = os.path.dirname(os.path.abspath(__file__))

# ==================== 方向定义 ====================
# 8 方向环形顺序（用于兜底时计算方向距离）
DIRECTIONS = ["u", "d", "l", "r", "lu", "ld", "ru", "rd"]

# 方向名 -> 类别 ID
DIR2ID = {d: i for i, d in enumerate(DIRECTIONS)}
ID2DIR = {i: d for i, d in enumerate(DIRECTIONS)}

# ==================== 极验图标配置 ====================
# 背景图尺寸（用于坐标归一化到 10000 坐标系）
IMG_W = 300
IMG_H = 200

# 问题图标文件名 -> 方向
ICON_MAPPING = {
    '8da090c135ff029f3b5e19f4c44f73c8.png': 'u',
    'cb0eaa639b2117a69a81af3d8c1496a1.png': 'd',
    '315ce8665e781dabcd1eb09d3e604803.png': 'l',
    '38bd9dda695098c7dfad74c921923a7d.png': 'lu',
    '502e51dbabf411beba2dcd55fd38ebbd.png': 'ld',
    '2b2387f566f6a03ed594d4d7cfda471f.png': 'r',
    '78dc29045d587ad054c7353732df53c5.png': 'ru',
    '23ef93e6b0e0df0e15b66667c99a5fb4.png': 'rd',
}

# ==================== 训练配置 ====================
TRAINING_DIR = os.path.join(ROOT, "training")
DATASET_DIR  = os.path.join(TRAINING_DIR, "dataset")
IMAGES_ALL   = os.path.join(DATASET_DIR, "images", "all")
IMAGES_TRAIN = os.path.join(DATASET_DIR, "images", "train")
IMAGES_VAL   = os.path.join(DATASET_DIR, "images", "val")
LABELS_ALL   = os.path.join(DATASET_DIR, "labels", "all")
LABELS_TRAIN = os.path.join(DATASET_DIR, "labels", "train")
LABELS_VAL   = os.path.join(DATASET_DIR, "labels", "val")
DATA_YAML    = os.path.join(DATASET_DIR, "data.yaml")

RUNS_DIR     = os.path.join(TRAINING_DIR, "runs")
PREDICT_BASE = os.path.join(RUNS_DIR, "predict")
MODELS_DIR   = os.path.join(TRAINING_DIR, "models")
WEIGHTS_DIR  = os.path.join(TRAINING_DIR, "weights")

# 人工标注的编号上界（000000 ~ HUMAN_END 是人工标注）
HUMAN_END = 999

# 预测任务名（对应 training/runs/predict/ 下的子目录）
PREDICT_NAME = "icon_v11b_auto"

# ==================== 运行时配置 ====================
GEEKED_MODELS = os.path.join(ROOT, "geeked", "models")
ONNX_PATH     = os.path.join(GEEKED_MODELS, "icon_yolo.onnx")

# ==================== YOLO 推理配置 ====================
YOLO_IMG_SIZE   = 640
YOLO_CONF_THRES = 0.10
YOLO_IOU_THRES  = 0.45
YOLO_MAX_DET    = 4          # 每张图最多保留 4 个目标