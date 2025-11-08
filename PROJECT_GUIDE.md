# 动物目标检测比赛 - 完整项目指南

## 📋 项目概述

本项目用于参加 AI Studio 动物目标检测比赛，使用 PaddleDetection 框架训练轻量级目标检测模型。

**数据集信息：**
- 总图像数：2,475 张
- 类别：3 类（monkey, panda, wolf）
- 图像尺寸：3840x2160 (4K)
- 数据格式：Pascal VOC XML

**比赛要求：**
- 预测速度：V100 上 ≥ 20 FPS
- 模型大小：< 200MB
- 提交格式：submission.zip

---

## 📂 项目结构

```
/home/user/ai/
├── data/
│   ├── raw/
│   │   └── wild_animal/          # 原始数据集
│   └── processed/
│       └── animals/               # 处理后的数据集
│           ├── train/            # 训练集 (1730 张, 69.9%)
│           ├── val/              # 验证集 (493 张, 19.9%)
│           └── test/             # 测试集 (252 张, 10.2%)
├── scripts/
│   ├── analyze_dataset.py        # 数据集分析脚本
│   └── split_dataset.py          # 数据集划分脚本
├── predict.py                    # 预测脚本 (提交用)
├── PROJECT_GUIDE.md              # 本文档
└── README.md

```

---

## 🚀 快速开始

### 步骤 1: 数据集准备 ✅ (已完成)

数据集已下载、分析并划分完成：
- 训练集：1,730 张
- 验证集：493 张
- 测试集：252 张

数据位置：`/home/user/ai/data/processed/animals/`

### 步骤 2: 在 AI Studio 上训练模型

⚠️ **重要**：当前环境是 CPU，无法高效训练。建议在 AI Studio 上使用免费 GPU 进行训练。

#### 2.1 创建 AI Studio 项目

1. 访问 https://aistudio.baidu.com/
2. 创建新项目（选择GPU环境）
3. 上传数据集（已处理好的数据）

#### 2.2 推荐模型选择

根据 20 FPS 要求，推荐以下轻量级模型：

| 模型 | 速度 (V100) | 精度 (COCO) | 模型大小 | 推荐度 |
|------|------------|------------|----------|--------|
| **PP-YOLOE-s** | ~60 FPS | 43.4 mAP | ~7.9MB | ⭐⭐⭐⭐⭐ |
| PP-PicoDet-s | ~150 FPS | 32.0 mAP | ~4.9MB | ⭐⭐⭐⭐ |
| YOLOv3-MobileNetV3 | ~45 FPS | 31.6 mAP | ~9MB | ⭐⭐⭐ |

**最佳选择：PP-YOLOE-s**
- 速度快（~60 FPS）
- 精度高
- 模型小（< 200MB）

#### 2.3 训练命令示例

在 AI Studio 环境中：

```bash
# 1. 克隆 PaddleDetection
git clone https://github.com/PaddlePaddle/PaddleDetection.git
cd PaddleDetection

# 2. 安装依赖
pip install -r requirements.txt
python setup.py install

# 3. 下载预训练模型
mkdir -p pretrain_models
cd pretrain_models
wget https://paddledet.bj.bcebos.com/models/ppyoloe_crn_s_300e_coco.pdparams
cd ..

# 4. 准备配置文件 (见下方)
# 创建 configs/animals/ppyoloe_s_animals.yml

# 5. 开始训练
python tools/train.py -c configs/animals/ppyoloe_s_animals.yml \
    --use_vdl=True \
    --vdl_log_dir=vdl_log_dir/animals \
    --eval

# 6. 导出推理模型
python tools/export_model.py \
    -c configs/animals/ppyoloe_s_animals.yml \
    -o weights=output/ppyoloe_s_animals/best_model \
    --output_dir=inference_model
```

#### 2.4 配置文件示例

创建 `configs/animals/ppyoloe_s_animals.yml`:

```yaml
_BASE_: [
  '../datasets/voc.yml',
  '../runtime.yml',
  '../ppyoloe/_base_/ppyoloe_crn.yml',
  '../ppyoloe/_base_/optimizer_300e.yml',
  '../ppyoloe/_base_/ppyoloe_reader.yml',
]

pretrain_weights: pretrain_models/ppyoloe_crn_s_300e_coco.pdparams
depth_mult: 0.33
width_mult: 0.50

VOCDataSet:
  dataset_dir: /path/to/your/data/processed/animals
  anno_path: train/annotations
  label_list: label_list.txt
  data_fields: ['image', 'gt_bbox', 'gt_class']

metric: VOC
map_type: 11point
num_classes: 3

epoch: 100
LearningRate:
  base_lr: 0.001
  schedulers:
  - !CosineDecay
    max_epochs: 100
  - !LinearWarmup
    start_factor: 0.
    epochs: 5

TrainReader:
  batch_size: 8

EvalReader:
  batch_size: 1
```

### 步骤 3: 准备提交文件

训练完成后，导出的模型文件会在 `inference_model/` 目录下：
- `model.pdmodel` - 模型结构
- `model.pdiparams` - 模型参数

#### 3.1 提交包结构

```
submission.zip
├── model/                  # 模型目录
│   ├── model.pdmodel
│   ├── model.pdiparams
│   └── infer_cfg.yml
├── predict.py              # 预测脚本 (已准备好)
└── env/                    # 可选：依赖库
    └── paddledet/
```

#### 3.2 创建提交包

```bash
# 在 AI Studio 上执行
mkdir -p submission/model
mkdir -p submission/env

# 复制模型文件
cp inference_model/model.pdmodel submission/model/
cp inference_model/model.pdiparams submission/model/
cp inference_model/infer_cfg.yml submission/model/

# 复制预测脚本
cp /path/to/predict.py submission/

# 打包
cd submission
zip -r submission.zip .
```

---

## 📝 predict.py 使用说明

`predict.py` 已经按照比赛要求编写完成，接受两个参数：

```bash
python predict.py data.txt result.json
```

**参数说明：**
- `data.txt`: 包含图像路径列表的文本文件（每行一个路径）
- `result.json`: 输出的检测结果（JSON格式）

**输出格式：**
```json
{
  "image_path": [
    {
      "bbox": [x1, y1, x2, y2],
      "score": 0.95,
      "category": "monkey",
      "category_id": 0
    }
  ]
}
```

---

## 🔧 本地测试

虽然本地是 CPU 环境，但可以测试代码逻辑：

```bash
# 创建测试数据列表
echo "/path/to/test/image1.jpg" > test_data.txt
echo "/path/to/test/image2.jpg" >> test_data.txt

# 运行预测（需要先有训练好的模型）
python predict.py test_data.txt test_result.json
```

---

## 📊 性能优化建议

### 达到 20 FPS 的关键：

1. **模型选择**：
   - 使用 PP-YOLOE-s 或更轻量的模型
   - 避免使用 ResNet50 等重量级 backbone

2. **输入尺寸**：
   - 图像 resize 到 640x640 或更小
   - 权衡速度和精度

3. **推理优化**：
   - 启用 GPU 加速
   - 使用 Paddle Inference（已在 predict.py 中实现）
   - 考虑使用 TensorRT 加速

4. **代码优化**：
   - 减少不必要的数据复制
   - 批量处理（如果允许）

---

## ✅ 检查清单

提交前请确认：

- [ ] 模型文件大小 < 200MB
- [ ] predict.py 可正常运行
- [ ] 在 V100 上测试 FPS ≥ 20
- [ ] submission.zip 结构正确
- [ ] result.json 格式正确

---

## 🆘 常见问题

### Q1: 模型太大超过 200MB 怎么办？
A: 使用更轻量的模型（PP-PicoDet）或减小输入尺寸。

### Q2: FPS 达不到 20 怎么办？
A: 减小图像输入尺寸，或使用更快的模型。

### Q3: 在哪里可以免费训练？
A: AI Studio 提供免费 GPU 算力，注册后可直接使用。

---

## 📚 参考资料

- PaddleDetection 官方文档：https://github.com/PaddlePaddle/PaddleDetection
- PP-YOLOE 模型介绍：https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/ppyoloe
- AI Studio 平台：https://aistudio.baidu.com/

---

## 📧 项目信息

- 创建日期：2025-11-08
- 数据集：wild_animals (2,475 张图像)
- 目标：野生动物检测（monkey, panda, wolf）
- 框架：PaddleDetection + PaddlePaddle

祝你比赛顺利！🎉
