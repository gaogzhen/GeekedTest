# 训练自定义 YOLO 模型

本文档详细说明如何训练 Geetest v4 图标点选验证码的专用检测模型。

> 项目总览和快速使用请见 [readme.md](readme.md)。

> **路径约定**：文档中所有命令假设当前工作目录为**项目根目录**（含 `config.py`、`main.py` 的目录）。绝对路径统一用 `<项目根目录>` 占位，实际执行时替换为你的本地路径。相对路径（如 `training/dataset/...`）直接使用，无需修改。

---

## 目录

- [总览](#总览)
- [前置准备](#前置准备)
- [阶段 1：数据采集](#阶段-1数据采集)
- [阶段 2：人工标注](#阶段-2人工标注)
- [阶段 3：JSON → YOLO txt](#阶段-3json--yolo-txt)
- [阶段 4：训练模型](#阶段-4训练模型)
- [阶段 5：评估模型](#阶段-5评估模型)
- [阶段 6：导出 ONNX](#阶段-6导出-onnx)
- [阶段 7：集成到运行时](#阶段-7集成到运行时)
- [阶段 8：自动预标注迭代](#阶段-8自动预标注迭代)
- [迭代记录参考](#迭代记录参考)
- [脚本速查](#脚本速查)
- [常见错误](#常见错误)
- [调优建议](#调优建议)

---

## 总览

### 迭代闭环

```
┌──────────────────────────────────────────────────────────────────┐
│                      训练迭代闭环                                 │
│                                                                    │
│   ① 采集背景图                                                      │
│        ↓                                                           │
│   ② 人工标注（8 方向）                                             │
│        ↓                                                           │
│   ③ 训练 v1                                                        │
│        ↓                                                           │
│   ④ 自动预标注剩余图片                                             │
│        ↓                                                           │
│   ⑤ 人工复核预标注结果                                             │
│        ↓                                                           │
│   ⑥ 扩充数据集，训练 vN                                            │
│        ↓                                                           │
│   ⑦ 评估实测通过率                                                 │
│        ↓                                                           │
│   通过率 > 90% ? ── 否 ──→ 回到 ④                                  │
│        │                                                           │
│        是                                                          │
│        ↓                                                           │
│      定稿，集成到 geeked/models/icon_yolo.onnx                     │
└──────────────────────────────────────────────────────────────────┘
```

### 核心要点

| 要点 | 说明 |
|---|---|
| **必须关闭 fliplr** | `fliplr=0.0`，否则方向混淆（详见下方） |
| **实测通过率是唯一标准** | 验证集 mAP 会骗人 |
| **每张图恰好 4 框** | 极验固定约束 |
| **8 方向分类** | `u/d/l/r/lu/ld/ru/rd` |
| **数据质量 > 数据量** | 1000 张精标 > 2000 张乱标 |

### ⚠️ 关键陷阱：关闭水平翻转

**Ultralytics 默认 `fliplr=0.5`**，训练时 50% 概率水平翻转图片，但 **YOLO 标签不跟着翻转**。

| 原图 | 标签 | 翻转后 | 标签（未变） | 模型学到 |
|---|---|---|---|---|
| 鱼头朝左 | `l` | 鱼头朝右 | `l`（错） | 朝右也是 `l` |

**结果**：模型分不清 `l`/`r`、`lu`/`ru`、`ld`/`rd`。

**训练时必须显式关闭**：

```bash
fliplr=0.0
flipud=0.0
```

**其他增强可保留**（不影响方向）：

| 增强 | 默认值 | 保留 | 原因 |
|---|---|---|---|
| `mosaic` | 1.0 | ✅ | 拼接多图，不改变方向 |
| `mixup` | 0.0 | ✅ | 图像混合 |
| `hsv_h/s/v` | 0.015/0.7/0.4 | ✅ | 颜色抖动 |
| `scale` | 0.5 | ✅ | 缩放 |
| `translate` | 0.1 | ✅ | 平移 |
| **`fliplr`** | **0.5** | ❌ | **破坏方向** |
| `flipud` | 0.0 | ❌ | 保险关闭 |

---

## 前置准备

### 1. 环境检查

```bash
# Python 版本
python --version
# 应为 3.10+

# GPU 可用性
python -c "import torch; print(torch.cuda.is_available())"
# 应输出 True

# GPU 信息
nvidia-smi
```

### 2. 安装依赖

```bash
# 进入项目根目录
cd <项目根目录>

# 运行时依赖
pip install -r requirements.txt

# 训练依赖（RTX 50 系需要 cu128）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-train.txt
```

### 3. 确认目录结构

```bash
# 进入项目根目录
cd <项目根目录>

# 查看目录
ls        # Linux/macOS
dir       # Windows
```

**应该看到**：

```
config.py
utils/
geeked/
test/
training/
weights/
main.py
requirements.txt
requirements-train.txt
```

### 4. 关键文件就位

```bash
# AMP 检查模型（amp=True 时需要）
ls weights/yolo26n.pt

# 如果不存在，从 training/weights 复制
cp training/weights/yolo26n.pt weights/yolo26n.pt    # Linux/macOS
copy training\weights\yolo26n.pt weights\yolo26n.pt  # Windows
```

---

## 阶段 1：数据采集

### 目标

收集大量极验图标点选背景图，作为训练数据。

### 脚本

`training/scripts/fetch_bg.py`

### 原理

1. 用 `curl_cffi` 模拟 Chrome TLS 指纹
2. 请求极验 `load` 接口，获取背景图路径
3. 用同一 Session 下载到本地
4. 每次新建 `Geeked` 实例，生成新 `challenge`，拿到不同图片

### 执行

```bash
# 进入项目根目录
cd <项目根目录>

# 运行采集脚本
python training/scripts/fetch_bg.py
```

### 配置

在 `training/scripts/fetch_bg.py` 顶部调整：

```python
CAPTCHA_ID = "你的_captcha_id"
RISK_TYPE = "icon"
SAVE_DIR = "training/dataset/images/all"    # 输出目录
TOTAL = 2000                                  # 目标采集数量
SLEEP_RANGE = (2, 4)                          # 每次请求间隔（秒）
```

### 断点续采

脚本支持断点续采：启动时扫描 `SAVE_DIR`，从最大编号 +1 开始。

```
检测到已有 1500 个文件，从 001500.jpg 继续
```

### 产出

```
training/dataset/images/all/
├── 000000.jpg
├── 000001.jpg
├── 000002.jpg
└── ...
```

### 数据量建议

| 阶段 | 数量 | 说明 |
|---|---|---|
| 初始 | 2000 张 | 够训练 + 验证 + 迭代 |
| 迭代 | +500 张/轮 | 用新模型预标注 + 人工复核 |

---

## 阶段 2：人工标注

### 工具

**X-AnyLabeling v4.0.6+**（推荐）

```bash
pip install anylabeling
# 或
pip install x-anylabeling-cvhub
```

### 步骤

1. 打开 X-AnyLabeling
2. `File` → `Open Dir` → 选 `training/dataset/images/all`
3. 每张图：
   - 用矩形框圈出**每一个图标**
   - 标签为 8 方向之一
   - **每张图恰好 4 个框**
4. `Ctrl+S` 保存（保存为 X-AnyLabeling JSON）

### 方向判定标准

| 方向 | 图标头部朝向 | 角度 |
|---|---|---|
| `u` | 正上 | 90° |
| `d` | 正下 | 270° |
| `l` | 正左 | 180° |
| `r` | 正右 | 0° |
| `lu` | 左上 | 135° |
| `ld` | 左下 | 225° |
| `ru` | 右上 | 45° |
| `rd` | 右下 | 315° |

**关键**：以**图标头部尖端**为准，用 45° 扇区划分。

**边界模糊时**：看鱼头最尖端落在哪个扇区。

### 标注技巧

| 技巧 | 说明 |
|---|---|
| 开启自动保存 | `File` → `Save Automatically` |
| 快捷键 `W` | 画框 |
| 快捷键 `D` | 下一张 |
| 快捷键 `Delete` | 删除选中框 |
| 快捷键 `Ctrl+S` | 手动保存 |
| **每张必检 4 框** | 少补多删 |

### 初期建议

标注 **500–1000 张**，覆盖 8 个方向，每个方向至少 50 个样本。

### 产出

```
training/dataset/images/all/
├── 000000.jpg
├── 000000.json     ← X-AnyLabeling 标注（含 shapes）
├── 000001.jpg
├── 000001.json
└── ...
```

### JSON 格式示例

```json
{
  "version": "4.0.6",
  "shapes": [
    {
      "label": "lu",
      "score": null,
      "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
      "shape_type": "rectangle"
    }
  ],
  "imagePath": "000000.jpg",
  "imageHeight": 200,
  "imageWidth": 300
}
```

- `score: null` → 人工标注
- `score: 0.95` → 自动预标注

---

## 阶段 3：JSON → YOLO txt

### 配置人工标注上界

编辑根 `config.py`：

```python
# 人工标注的编号上界（000000 ~ HUMAN_END 是人工标注）
HUMAN_END = 999
```

### 脚本清单

| 脚本 | 用途 | 命令 |
|---|---|---|
| `clean_train_val.py` | 清空旧 train/val | `python -m training.scripts.clean_train_val` |
| `convert_human_only.py` | 只转人工标注 | `python -m training.scripts.convert_human_only` |
| `convert_json_to_yolo.py` | 全量转 | `python -m training.scripts.convert_json_to_yolo` |
| `clean_empty_labels.py` | 清理空标签 | `python -m training.scripts.clean_empty_labels` |
| `split_dataset.py` | 划分 train/val | `python -m training.scripts.split_dataset` |

### 完整执行

```bash
# 进入项目根目录
cd <项目根目录>

# 1. 清空旧 train/val
python -m training.scripts.clean_train_val

# 2. 只转人工标注（0 ~ HUMAN_END）
python -m training.scripts.convert_human_only
# 预期输出: 转换成功: 1000

# 3. 清理空标签
python -m training.scripts.clean_empty_labels
# 预期输出: 删除空文件: 0

# 4. 划分 train/val（8:2）
python -m training.scripts.split_dataset
# 预期输出: train: 800, val: 200
```

### 验证

```bash
# Linux/macOS
ls training/dataset/images/train | wc -l
ls training/dataset/labels/train | wc -l
ls training/dataset/images/val | wc -l
ls training/dataset/labels/val | wc -l

# Windows PowerShell
(Get-ChildItem training\dataset\images\train).Count
(Get-ChildItem training\dataset\labels\train).Count
(Get-ChildItem training\dataset\images\val).Count
(Get-ChildItem training\dataset\labels\val).Count
```

**期望**：图片数与标签数一致。

### 产出

```
training/dataset/
├── images/
│   ├── all/        ← 全量图
│   ├── train/      ← 80%
│   └── val/        ← 20%
└── labels/
    ├── all/        ← 全量标签（YOLO txt）
    ├── train/
    └── val/
```

### data.yaml 内容

```yaml
train: ./images/train
val: ./images/val

nc: 8
names: ['u', 'd', 'l', 'r', 'lu', 'ld', 'ru', 'rd']
```

---

## 阶段 4：训练模型

### 首次训练（从 COCO 预训练权重开始）

```bash
# 进入项目根目录
cd <项目根目录>

yolo detect train \
  data=training/dataset/data.yaml \
  model=yolov8n.pt \
  epochs=200 \
  imgsz=640 \
  batch=64 \
  device=0 \
  amp=True \
  cache=True \
  fliplr=0.0 \
  flipud=0.0 \
  patience=50 \
  project=training/runs \
  name=icon_dir_v1
```

**Windows 用户用反引号续行**：

```powershell
yolo detect train `
  data=training/dataset/data.yaml `
  model=yolov8n.pt `
  epochs=200 `
  imgsz=640 `
  batch=64 `
  device=0 `
  amp=True `
  cache=True `
  fliplr=0.0 `
  flipud=0.0 `
  patience=50 `
  project=training/runs `
  name=icon_dir_v1
```

### 迭代训练（从上一版最佳权重开始）

```bash
yolo detect train \
  data=training/dataset/data.yaml \
  model=training/runs/icon_dir_v11b/weights/best.pt \
  epochs=100 \
  imgsz=640 \
  batch=64 \
  device=0 \
  amp=True \
  cache=True \
  fliplr=0.0 \
  flipud=0.0 \
  project=training/runs \
  name=icon_dir_v12
```

### 参数详解

| 参数 | 首次 | 迭代 | 说明 |
|---|---|---|---|
| `data` | `training/dataset/data.yaml` | 同 | 数据集配置 |
| `model` | `yolov8n.pt` | `best.pt` | 起点权重 |
| `epochs` | 200 | 100 | 训练轮数 |
| `imgsz` | 640 | 640 | 输入尺寸 |
| `batch` | 64 | 64 | 批大小 |
| `device` | 0 | 0 | GPU 编号 |
| `amp` | True | True | 混合精度 |
| `cache` | True | True | 缓到内存 |
| **`fliplr`** | **0.0** | **0.0** | **必须关闭** |
| **`flipud`** | **0.0** | **0.0** | **必须关闭** |
| `patience` | 50 | — | 早停（50 轮无提升） |
| `project` | `training/runs` | 同 | 输出根目录 |
| `name` | `icon_dir_vN` | 同 | 本次名称 |

### 显存参考

| batch | 显存 | 速度 | 说明 |
|---|---|---|---|
| 16 | ~2 GB | 1x | 浪费 GPU |
| 32 | ~4 GB | 1.5x | 保守 |
| **64** | **~6–8 GB** | **2–3x** | **推荐（12GB 显存）** |
| 96 | ~9 GB | 3x | 激进 |
| 128 | ~10–11 GB | 3–4x | 极限 |

### 训练日志

```
Ultralytics 8.4.160 🚀 Python-3.11.x torch-2.7.0+cu128 CUDA:0 (NVIDIA GeForce RTX 5070, 12227MiB)
...
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
  1/100   6.5G      1.234      2.345      1.456        320         640
```

**关键指标**：
- `CUDA:0` → GPU 已启用
- `GPU_mem` → 显存占用
- `box_loss` / `cls_loss` → 应持续下降

### 训练时间参考（RTX 5070）

| 数据量 | batch | epochs | 耗时 |
|---|---|---|---|
| 500 | 64 | 200 | ~5 分钟 |
| 1000 | 64 | 100 | ~8 分钟 |
| 2000 | 64 | 100 | ~15 分钟 |

### 产出

```
training/runs/icon_dir_v1/
├── weights/
│   ├── best.pt        ← 最佳权重
│   └── last.pt        ← 最后一轮
├── results.csv        ← 每轮指标
├── confusion_matrix.png
├── train_batch0.jpg
└── val_batch0_labels.jpg
```

---

## 阶段 5：评估模型

### 方式 1：验证集指标（快）

```bash
# 进入项目根目录
cd <项目根目录>

yolo detect val \
  model=training/runs/icon_dir_v1/weights/best.pt \
  data=training/dataset/data.yaml
```

**输出示例**：

```
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
                   all        200        801      0.897      0.911      0.958      0.956
                     u        104        106      0.944      0.991      0.994      0.993
                     d        104        104          1      0.956      0.994      0.991
                     l         97         97      0.861      0.907      0.936      0.934
                     r        103        103      0.884      0.893      0.959      0.955
                    lu         97         97      0.866      0.918      0.943      0.943
                    ld         86         87      0.864      0.877      0.933      0.927
                    ru        101        103      0.874      0.874      0.958      0.958
                    rd        103        104      0.886      0.875      0.944      0.944
```

**关注指标**：

| 指标 | 含义 | 目标 |
|---|---|---|
| `mAP50` | IoU=0.5 平均精度 | > 0.94 |
| `mAP50-95` | 多阈值平均 | > 0.92 |
| `P` | 精确率 | > 0.85 |
| `R` | 召回率 | > 0.85 |

### 方式 2：实测通过率（更可靠）

```bash
# 进入 test 目录
cd <项目根目录>/test

python eval_pass_rate.py
```

**输出示例**：

```
开始评估: 20 次
============================================================
[ 1/20] ✅ 成功  (2.33s)
[ 2/20] ✅ 成功  (2.04s)
...
[18/20] ❌ 失败  (2.14s)  原因: verify_fail
[19/20] ✅ 成功  (2.61s)
[20/20] ✅ 成功  (1.82s)
============================================================
总次数: 20
成功: 19
失败: 1
通过率: 95.0%

失败原因分布:
  verify_fail: 1 次
```

### ⚠️ 验证集 mAP 会骗人

| 版本 | 数据量 | fliplr | 验证集 mAP50 | 实测通过率 |
|---|---|---|---|---|
| v7 | 500 | 0.5 | 0.929 | 40–65% |
| v8 | 600 | 0.5 | **0.962** | **45%** |
| v10 | 1000 | 0.5 | 0.958 | 40–50% |
| v11b | 1000 | **0.0** | — | **95%** |

**v8 验证集 mAP 最高（0.962）但实测最差（45%）**。

**原因**：验证集和训练集同分布，都受 `fliplr` 影响，所以验证集看起来正常。但实测遇到新图片时，模型对左右方向判断错误。

**结论**：**必须以实测通过率为准**。

---

## 阶段 6：导出 ONNX

### 标准导出

```bash
# 进入项目根目录
cd <项目根目录>

yolo export \
  model=training/runs/icon_dir_v1/weights/best.pt \
  format=onnx \
  opset=12 \
  simplify=True
```

**产出**：同目录生成 `best.onnx`。

### 如果实测置信度偏低

试 `simplify=False`：

```bash
yolo export \
  model=training/runs/icon_dir_v1/weights/best.pt \
  format=onnx \
  opset=12 \
  simplify=False
```

### 验证输出形状

```python
import onnxruntime as ort

sess = ort.InferenceSession(
    "training/runs/icon_dir_v1/weights/best.onnx",
    providers=['CPUExecutionProvider']
)
print("输入:", [(i.name, i.shape) for i in sess.get_inputs()])
print("输出:", [(o.name, o.shape) for o in sess.get_outputs()])
```

**期望**：

```
输入: [('images', [1, 3, 640, 640])]
输出: [('output0', [1, 12, 8400])]
```

`12 = 4 (bbox) + 8 (方向类别)`，说明是 8 方向检测器。

### 单测 ONNX

```bash
# 进入项目根目录
cd <项目根目录>

python test/test_yolo.py
```

**期望输出**：

```
检测到 4 个框
  u  bbox=[...]  conf=0.97
  d  bbox=[...]  conf=0.95
  lu bbox=[...]  conf=0.94
  rd bbox=[...]  conf=0.93
```

**如果置信度只有 0.5–0.7**：ONNX 导出有问题，试 `simplify=False`。

---

## 阶段 7：集成到运行时

### 复制 ONNX

```bash
# 进入项目根目录
cd <项目根目录>

# Linux/macOS
cp training/runs/icon_dir_v1/weights/best.onnx geeked/models/icon_yolo.onnx
cp training/runs/icon_dir_v1/weights/best.onnx training/models/icon_yolo.onnx

# Windows
copy training\runs\icon_dir_v1\weights\best.onnx geeked\models\icon_yolo.onnx
copy training\runs\icon_dir_v1\weights\best.onnx training\models\icon_yolo.onnx
```

### 测试

```bash
# 单测
python main.py

# 通过率评估
cd test
python eval_pass_rate.py
```

**期望**：

```
总次数: 20
成功: 19
失败: 1
通过率: 95.0%
```

---

## 阶段 8：自动预标注迭代

### 目标

用训练好的模型预标注剩余图片，人工只修正错误，快速扩充数据集。

### 步骤 1：预测全量图

```bash
# 进入项目根目录
cd <项目根目录>

yolo detect predict \
  model=training/runs/icon_dir_v1/weights/best.pt \
  source=training/dataset/images/all \
  save=True \
  save_txt=True \
  save_conf=True \
  conf=0.10 \
  iou=0.45 \
  project=training/runs/predict \
  name=icon_v1_auto \
  exist_ok=True
```

**关键参数**：

| 参数 | 值 | 说明 |
|---|---|---|
| `conf` | **0.10** | 低阈值，保留所有候选 |
| `iou` | 0.45 | NMS 去重 |
| `save_txt` | True | 保存 YOLO txt |
| `save_conf` | True | 保存置信度 |

**产出**：

```
training/runs/predict/detect/icon_v1_auto/
├── labels/       ← YOLO txt
└── *.jpg         ← 可视化图
```

**确认路径**：

```bash
# Linux/macOS
ls training/runs/predict/detect/icon_v1_auto/labels/

# Windows
dir training\runs\predict\detect\icon_v1_auto\labels\
```

**如果路径不同**，用以下命令查找：

```bash
# Linux/macOS
find training/runs -type d -name "icon_v1_auto"

# Windows PowerShell
Get-ChildItem -Recurse -Path training\runs -Filter "icon_v1_auto" | Select-Object FullName
```

### 步骤 2：转成 X-AnyLabeling JSON

```bash
# 进入项目根目录
cd <项目根目录>

python -m training.scripts.convert_predict_to_json icon_v1_auto
```

**关键逻辑**：

1. 读取预测的 YOLO txt
2. 跳过已人工标注的图片（编号 ≤ `HUMAN_END`）
3. 按置信度排序，每张图保留前 4 个框
4. 生成 X-AnyLabeling JSON，覆盖到 `images/all/`

**预期输出**：

```
预测目录: ...\training\runs\predict\detect\icon_v1_auto\labels
人工标注上界: 000999
============================================================
更新 JSON: 1101
跳过人工标注: 1000
跳过空预测: 0
找不到图片: 0
```

### 步骤 3：人工复核

```bash
xanylabeling
# Open Dir: training/dataset/images/all
# 从 HUMAN_END+1 开始，逐张复核
```

**复核操作**：

| 操作 | 快捷键 | 说明 |
|---|---|---|
| 下一张 | `D` | — |
| 上一张 | `A` | — |
| 删误检框 | 选中 + `Delete` | — |
| 补漏检框 | `R` | 画框 |
| 修改标签 | 选中框，右侧改 | — |
| 保存 | `Ctrl+S` | — |

**复核重点**：

| 检查项 | 处理 |
|---|---|
| 误检框（背景被当成图标） | 删除 |
| 漏检框（图标没被框出） | 补框 |
| 方向标错 | 修正 |
| 框数 ≠ 4 | 补或删 |
| 框不贴合 | 拖动边缘 |

**优先复核弱项方向**（从验证集结果看哪类 mAP 最低）：

| 方向 | 常见问题 |
|---|---|
| `ru` | 被误判成 `rd` |
| `lu` | 被误判成 `ru` 或 `u` |
| `ld` | 被误判成 `rd` 或 `d` |

### 步骤 4：扩充数据集，训练新版

**更新配置**：

编辑根 `config.py`：

```python
# 从 999 改成新的上界
HUMAN_END = 1499    # 新增 500 张人工标注
```

**执行**：

```bash
# 进入项目根目录
cd <项目根目录>

# 1. 清空旧 train/val
python -m training.scripts.clean_train_val

# 2. 转换人工标注
python -m training.scripts.convert_human_only

# 3. 清理空标签
python -m training.scripts.clean_empty_labels

# 4. 重新划分
python -m training.scripts.split_dataset

# 5. 训练新版（用上一版作为起点）
yolo detect train \
  data=training/dataset/data.yaml \
  model=training/runs/icon_dir_v1/weights/best.pt \
  epochs=100 \
  imgsz=640 \
  batch=64 \
  device=0 \
  amp=True \
  cache=True \
  fliplr=0.0 \
  flipud=0.0 \
  project=training/runs \
  name=icon_dir_v2
```

### 步骤 5：评估 + 重复

```bash
# 进入项目根目录
cd <项目根目录>

# 评估
yolo detect val \
  model=training/runs/icon_dir_v2/weights/best.pt \
  data=training/dataset/data.yaml

# 导出
yolo export \
  model=training/runs/icon_dir_v2/weights/best.pt \
  format=onnx opset=12 simplify=True

# 集成（Linux/macOS）
cp training/runs/icon_dir_v2/weights/best.onnx geeked/models/icon_yolo.onnx

# 测通过率
cd test
python eval_pass_rate.py
```

**重复阶段 4–8，直到通过率 > 90%。**

---

## 迭代记录参考

### 完整版本历史

| 版本 | 数据量 | fliplr | epochs | 验证集 mAP50 | 实测通过率 | 状态 |
|---|---|---|---|---|---|---|
| v6 | 400 | 0.5 | 200 | 0.940 | 50% | 基线 |
| v7 | 500 | 0.5 | 200 | 0.929 | 40–65% | 波动大 |
| v8 | 600 | 0.5 | 200 | 0.962 | 45% | 验证集虚高 |
| v9 | 800 | 0.5 | 100 | 0.943 | — | — |
| v10 | 1000 | 0.5 | 100 | 0.958 | 40–50% | 仍未突破 |
| **v11a** | **1000** | **0.0** | 100 | — | **95%** | ⭐ 迁移 v7 |
| **v11b** | **1000** | **0.0** | 200 | — | **95%** | ⭐ 从零训练 |

### 关键转折点

**v10 → v11：关闭 fliplr**

| 项 | v10 | v11 |
|---|---|---|
| 数据量 | 1000 | 1000 |
| **fliplr** | **0.5** | **0.0** |
| 通过率 | 40–50% | **95%** |

**数据量相同，仅关掉 fliplr，通过率翻倍。**

### 验证集 vs 实测对比

| 版本 | 验证集 mAP50 | 实测通过率 | 差异 |
|---|---|---|---|
| v7 | 0.929 | 40–65% | 一致 |
| v8 | **0.962** | **45%** | ❌ 反向 |
| v10 | 0.958 | 40–50% | ❌ 反向 |
| v11b | — | **95%** | — |

**结论**：验证集 mAP 高 ≠ 实测通过率高。**以实测为准。**

---

## 脚本速查

### 数据处理

| 脚本 | 命令 | 用途 |
|---|---|---|
| `fetch_bg.py` | `python training/scripts/fetch_bg.py` | 采集背景图 |
| `convert_human_only.py` | `python -m training.scripts.convert_human_only` | 只转人工标注 |
| `convert_json_to_yolo.py` | `python -m training.scripts.convert_json_to_yolo` | 全量转 YOLO |
| `convert_predict_to_json.py` | `python -m training.scripts.convert_predict_to_json <name>` | 预测结果转 JSON |
| `clean_empty_labels.py` | `python -m training.scripts.clean_empty_labels` | 清理空标签 |
| `clean_train_val.py` | `python -m training.scripts.clean_train_val` | 清空 train/val |
| `split_dataset.py` | `python -m training.scripts.split_dataset` | 划分数据集 |
| `count_boxes.py` | `python -m training.scripts.count_boxes` | 统计每张框数 |

### 训练 / 评估 / 导出

| 任务 | 命令 |
|---|---|
| 训练 | `yolo detect train data=... model=... fliplr=0.0 ...` |
| 评估 | `yolo detect val model=... data=...` |
| 导出 | `yolo export model=... format=onnx opset=12 simplify=True` |
| 预测 | `yolo detect predict model=... source=... conf=0.10 ...` |
| 通过率测试 | `cd test && python eval_pass_rate.py` |
| YOLO 单测 | `python test/test_yolo.py` |

---

## 常见错误

### 训练阶段

| 错误 | 原因 | 解决 |
|---|---|---|
| `fliplr` 未关 | 方向混淆 | 加 `fliplr=0.0` `flipud=0.0` |
| `No labels found` | train/val 为空 | 先跑 `convert_human_only` + `split_dataset` |
| `Label class X exceeds nc=8` | 非法类别 ID | 检查标签格式（应为 0–7） |
| `CUDA out of memory` | batch 太大 | 降到 32 或 16 |
| `torch.cuda.is_available()` False | torch 版本不对 | 装 cu128 版 |
| `AMP: downloading yolo26n.pt` | 缺 AMP 检查模型 | 放 `weights/yolo26n.pt` 或 `amp=False` |
| `data.yaml not found` | 路径错误 | 用相对路径或检查工作目录 |
| 训练一直不收敛 | 学习率问题 | 检查 `lr0`，默认 0.01 |

### 推理 / 集成阶段

| 错误 | 原因 | 解决 |
|---|---|---|
| ONNX 置信度偏低 | simplify 问题 | `simplify=False` 重新导出 |
| 检测框数 ≠ 4 | 模型没学好 | 检查训练数据每张是否 4 框 |
| 方向全错 | fliplr 问题 | 重训，`fliplr=0.0` |
| `ru` 偶发缺失 | 样本不足 | 补标 `ru` 样本 |
| 坐标偏 | 缩放系数错 | 确认用 `10000/300`、`10000/200` |
| `ModuleNotFoundError: No module named 'config'` | sys.path 未加 | 在 `geeked/__init__.py` 加 |
| `CUDAExecutionProvider` 警告 | CPU 版 onnxruntime | 忽略，或装 `onnxruntime-gpu` |

---

## 调优建议

### 1. 提高通过率

| 措施 | 效果 | 成本 |
|---|---|---|
| 补标弱项方向（`ru`、`lu`） | ⭐⭐⭐ | 中 |
| 增加数据量到 2000+ | ⭐⭐ | 高 |
| 换 YOLOv8s（更大模型） | ⭐⭐ | 中 |
| 调整兜底逻辑 | ⭐ | 低 |
| 优化坐标精度 | ⭐ | 低 |

### 2. 加速训练

| 措施 | 效果 |
|---|---|
| `batch=64`（当前 16） | 2–3x |
| `cache=True` | 消除 IO 瓶颈 |
| `imgsz=512`（当前 640） | 1.3x |
| `amp=True` | 1.1–1.2x |

### 3. 防止过拟合

| 措施 | 说明 |
|---|---|
| `epochs=100`（不是 200） | 小数据集 |
| `patience=50` | 早停 |
| 保留 `mosaic`、`mixup` | 数据增强 |
| 增加数据量 | 根本解决 |

### 4. 数据质量 > 数据量

**1000 张精标 > 2000 张乱标**。

**优先做的事**：
1. 每张图**恰好 4 框**
2. 方向标签**准确**（头部朝向）
3. 框**贴合**图标边缘
4. 弱项方向**多补样本**

### 5. 关键参数速查

| 参数 | 推荐值 | 说明 |
|---|---|---|
| `fliplr` | **0.0** | **必须** |
| `flipud` | **0.0** | **必须** |
| `batch` | 64 | 12GB 显存 |
| `imgsz` | 640 | — |
| `epochs` | 100–200 | 首次 200，迭代 100 |
| `patience` | 50 | 早停 |
| `cache` | True | 加速 |
| `amp` | True | 加速 |
| `mosaic` | 1.0 | 保留 |
| `mixup` | 0.0 | 保留 |
| `hsv_h/s/v` | 0.015/0.7/0.4 | 保留 |
| `scale` | 0.5 | 保留 |
| `translate` | 0.1 | 保留 |

---

## 附录：目录结构

```
<项目根目录>/
├── config.py                  # 全局配置（含 HUMAN_END）
├── utils/
│   └── logger.py
├── geeked/
│   ├── geeked.py
│   ├── icon.py
│   ├── yolo_server.py
│   ├── sign.py
│   └── models/
│       └── icon_yolo.onnx     # 最终集成的模型
├── test/
│   ├── eval_pass_rate.py      # 通过率评估
│   └── test_yolo.py           # YOLO 单测
├── training/
│   ├── __init__.py
│   ├── config.py              # 转发根 config
│   ├── scripts/               # 训练脚本
│   ├── dataset/               # 数据集（不上传）
│   ├── runs/                  # 训练产物（不上传）
│   ├── models/                # ONNX 备份
│   └── weights/               # 预训练权重
├── weights/
│   └── yolo26n.pt             # AMP 检查用
├── main.py
├── requirements.txt
├── requirements-train.txt
├── readme.md
├── training.md                # 本文件
└── LICENSE
```

---

## 附录：训练流程图（详细）

```
┌─────────────────────────────────────────────────────────────────────┐
│                         完整训练流程                                 │
└─────────────────────────────────────────────────────────────────────┘

【准备】
    │
    ├─ pip install -r requirements.txt
    ├─ pip install torch torchvision torchaudio --index-url .../cu128
    ├─ pip install -r requirements-train.txt
    └─ 确认 weights/yolo26n.pt 存在

【阶段 1：采集】
    │
    ├─ python training/scripts/fetch_bg.py
    └─ 产出: training/dataset/images/all/*.jpg（2000 张）

【阶段 2：标注】
    │
    ├─ X-AnyLabeling 打开 training/dataset/images/all
    ├─ 每张图画 4 个框，标签为 8 方向之一
    └─ 产出: training/dataset/images/all/*.json（1000 张）

【阶段 3：转换】
    │
    ├─ 改 config.py: HUMAN_END = 999
    ├─ python -m training.scripts.clean_train_val
    ├─ python -m training.scripts.convert_human_only
    ├─ python -m training.scripts.clean_empty_labels
    ├─ python -m training.scripts.split_dataset
    └─ 产出: training/dataset/{images,labels}/{train,val}/

【阶段 4：训练】
    │
    ├─ yolo detect train ... fliplr=0.0 flipud=0.0
    └─ 产出: training/runs/icon_dir_v1/weights/best.pt

【阶段 5：评估】
    │
    ├─ yolo detect val（验证集）
    ├─ cd test && python eval_pass_rate.py（实测）
    └─ 判断: 通过率 > 90%?

【阶段 6：导出】
    │
    ├─ yolo export ... format=onnx
    └─ 产出: training/runs/icon_dir_v1/weights/best.onnx

【阶段 7：集成】
    │
    ├─ cp best.onnx geeked/models/icon_yolo.onnx
    └─ python main.py（测试）

【阶段 8：迭代】
    │
    ├─ yolo detect predict（预标注剩余图）
    ├─ python -m training.scripts.convert_predict_to_json
    ├─ X-AnyLabeling 人工复核
    ├─ 改 config.py: HUMAN_END += 500
    ├─ 重新转换、划分
    └─ 训练 v2 → 回到阶段 5
```

---

**文档版本**: 1.0  
**最后更新**: 2026-09-25  
**适用版本**: GeekedTest v1.9.7+ 定制版