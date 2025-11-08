# AI Studio 完整使用指南

## 🎯 快速开始

本指南帮助你在 AI Studio 上快速训练动物检测模型。

---

## 📋 准备工作

### 1. 注册 AI Studio 账号
访问：https://aistudio.baidu.com/
使用百度账号登录

### 2. 创建项目
1. 点击「项目」→「创建项目」
2. 选择「Notebook」
3. 环境配置：
   - **算力**：选择 GPU（免费）
   - **镜像**：PaddlePaddle 2.6
   - **存储**：至少 10GB

---

## 📦 步骤1：上传数据集

### 方法 A：使用已有数据集（推荐）
从百度网盘或者之前下载的位置获取 `wild_animal` 数据集。

### 方法 B：从本地上传
1. 在本地打包数据：
   ```bash
   cd /home/user/ai/data/processed
   zip -r animals_dataset.zip VOCdevkit/
   ```
2. 在 AI Studio 项目中：
   - 点击「数据」→「上传数据」
   - 选择 `animals_dataset.zip`
   - 解压到项目目录

---

## 🚀 步骤2：设置训练环境

在 AI Studio Notebook 中执行：

```bash
# 1. 克隆 PaddleDetection
!git clone https://github.com/PaddlePaddle/PaddleDetection.git
%cd PaddleDetection

# 2. 安装依赖
!pip install -r requirements.txt

# 3. 安装 PaddleDetection
!python setup.py install
```

---

## ⚙️ 步骤3：准备数据和配置

### 3.1 上传数据集（如果还没上传）
```bash
# 上传你的数据集 zip 文件，然后解压
!unzip -q /home/aistudio/data/animals_dataset.zip -d /home/aistudio/work/
```

### 3.2 创建数据集配置文件

创建 `configs/animals/animals_dataset.yml`:

```yaml
metric: VOC
map_type: 11point
num_classes: 3

TrainDataset:
  name: VOCDataSet
  dataset_dir: /home/aistudio/work/VOCdevkit
  anno_path: VOC2007/ImageSets/Main/trainval.txt
  label_list: VOC2007/label_list.txt
  data_fields: ['image', 'gt_bbox', 'gt_class', 'difficult']

EvalDataset:
  name: VOCDataSet
  dataset_dir: /home/aistudio/work/VOCdevkit
  anno_path: VOC2007/ImageSets/Main/test.txt
  label_list: VOC2007/label_list.txt
  data_fields: ['image', 'gt_bbox', 'gt_class', 'difficult']
```

### 3.3 创建训练配置文件

创建 `configs/animals/ppyoloe_s_animals.yml`:

```yaml
_BASE_: [
  '../runtime.yml',
  '../ppyoloe/_base_/optimizer_80e.yml',
  '../ppyoloe/_base_/ppyoloe_crn.yml',
  '../ppyoloe/_base_/ppyoloe_reader.yml',
  './animals_dataset.yml',
]

log_iter: 10
snapshot_epoch: 5
weights: output/ppyoloe_crn_s_80e_animals/model_final

# PP-YOLOE-s 配置（轻量级，满足 20 FPS）
depth_mult: 0.33
width_mult: 0.50

# 训练配置
epoch: 80
LearningRate:
  base_lr: 0.01
  schedulers:
  - !CosineDecay
    max_epochs: 80
  - !LinearWarmup
    start_factor: 0.
    epochs: 5

# GPU 训练配置
TrainReader:
  batch_size: 16  # GPU 可以用更大的 batch
  shuffle: true
  drop_last: true

EvalReader:
  batch_size: 8

# 使用 GPU
use_gpu: true
```

---

## 🏋️ 步骤4：开始训练

```bash
# 在 PaddleDetection 目录下执行
!python tools/train.py \
    -c configs/animals/ppyoloe_s_animals.yml \
    --eval \
    --use_vdl=True \
    --vdl_log_dir=vdl_log/animals
```

**训练预计时间**（V100 GPU）：
- 每个 epoch: ~2-3 分钟
- 80 epochs: ~2-4 小时

---

## 📊 步骤5：监控训练

### 使用 VisualDL 查看训练曲线
```bash
# 新建终端执行
!visualdl --logdir vdl_log/animals --port 8080
```

然后在浏览器打开 VisualDL 界面查看：
- Loss 曲线
- mAP 指标
- 学习率变化

---

## 💾 步骤6：导出推理模型

训练完成后，导出推理模型：

```bash
!python tools/export_model.py \
    -c configs/animals/ppyoloe_s_animals.yml \
    -o weights=output/ppyoloe_crn_s_80e_animals/best_model \
    --output_dir=inference_model
```

导出的模型文件：
- `inference_model/model.pdmodel`
- `inference_model/model.pdiparams`
- `inference_model/infer_cfg.yml`

---

## 📦 步骤7：准备提交文件

### 7.1 复制 predict.py 到项目

将本项目的 `predict.py` 上传到 AI Studio，或者创建新的：

```python
# predict.py 的内容见项目根目录的 predict.py 文件
# 确保修改模型路径为: model_dir = './model'
```

### 7.2 创建提交目录结构

```bash
# 创建提交目录
!mkdir -p submission/model
!mkdir -p submission/env

# 复制模型文件
!cp inference_model/model.pdmodel submission/model/
!cp inference_model/model.pdiparams submission/model/
!cp inference_model/infer_cfg.yml submission/model/

# 复制预测脚本
!cp predict.py submission/

# 检查模型大小（必须 < 200MB）
!du -sh submission/model/
```

### 7.3 测试预测脚本

```bash
# 创建测试数据列表
!echo "/home/aistudio/work/VOCdevkit/VOC2007/JPEGImages/monkey0001.jpg" > test_data.txt

# 测试预测
%cd submission
!python predict.py test_data.txt result.json

# 查看结果
!cat result.json
```

### 7.4 打包提交

```bash
# 打包
%cd submission
!zip -r ../submission.zip .

# 检查大小
!ls -lh ../submission.zip
```

---

## ✅ 步骤8：提交比赛

1. 下载 `submission.zip` 到本地
2. 访问比赛页面：https://aistudio.baidu.com/competition/detail/XXX
3. 点击「提交结果」
4. 上传 `submission.zip`
5. 等待评测结果

---

## 🔧 性能优化建议

### 达到 20 FPS 的关键

1. **模型选择**
   - ✅ PP-YOLOE-s：~60 FPS，精度好
   - ✅ PP-PicoDet-s：~150 FPS，更快但精度略低

2. **输入尺寸**
   - 640x640：推荐（平衡速度和精度）
   - 512x512：更快但精度降低
   - 416x416：最快

3. **推理优化**
   - 使用 Paddle Inference
   - 启用 TensorRT（如果V100支持）
   - 减少后处理时间

4. **如果 FPS 不够**
   ```yaml
   # 修改配置文件，减小输入尺寸
   TrainReader:
     inputs_def:
       image_shape: [3, 512, 512]  # 从 640 改为 512
   ```

---

## ❓ 常见问题

### Q1: 训练时显存不足
```yaml
# 减小 batch_size
TrainReader:
  batch_size: 8  # 从 16 改为 8
```

### Q2: 训练太慢
- 检查是否使用了 GPU
- 减少 epoch 数进行快速验证
- 使用更小的模型（PP-PicoDet）

### Q3: mAP 太低
- 增加训练 epoch
- 调整学习率
- 增加数据增强

### Q4: 模型文件太大
- 使用 PP-YOLOE-s（~8MB）
- 不要使用 ResNet50 等重型 backbone

---

## 📚 参考资源

- PaddleDetection 文档：https://github.com/PaddlePaddle/PaddleDetection
- PP-YOLOE 介绍：https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/ppyoloe
- AI Studio 使用文档：https://ai.baidu.com/ai-doc/AISTUDIO/index

---

## 🎉 预期结果

使用 PP-YOLOE-s 模型：
- **训练时间**：2-4 小时
- **mAP**：预计 60-75%
- **FPS**（V100）：~60 FPS ✅
- **模型大小**：~8MB ✅

完全满足比赛要求！

祝你比赛成功！🏆
