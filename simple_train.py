#!/usr/bin/env python3
"""
简单直接的动物检测训练脚本
不依赖复杂的 PaddleDetection 配置
"""

import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import paddle
import paddle.nn as nn
from paddle.io import Dataset, DataLoader
from tqdm import tqdm
import json

class AnimalDataset(Dataset):
    """简单的动物检测数据集"""

    def __init__(self, image_dir, xml_dir, img_size=640):
        self.image_dir = image_dir
        self.xml_dir = xml_dir
        self.img_size = img_size

        # 类别映射
        self.classes = ['monkey', 'panda', 'wolf']
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # 加载所有样本
        self.samples = []
        for xml_file in os.listdir(xml_dir):
            if xml_file.endswith('.xml'):
                img_name = xml_file.replace('.xml', '.jpg')
                img_path = os.path.join(image_dir, img_name)
                xml_path = os.path.join(xml_dir, xml_file)

                if os.path.exists(img_path):
                    self.samples.append((img_path, xml_path))

        print(f"✅ 加载了 {len(self.samples)} 个样本")

    def __len__(self):
        return len(self.samples)

    def parse_xml(self, xml_path):
        """解析 XML 标注"""
        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        boxes = []
        labels = []

        for obj in root.findall('object'):
            name = obj.find('name').text
            if name in self.class_to_idx:
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)

                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(self.class_to_idx[name])

        return boxes, labels, (width, height)

    def __getitem__(self, idx):
        img_path, xml_path = self.samples[idx]

        # 读取图像
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 解析标注
        boxes, labels, (orig_w, orig_h) = self.parse_xml(xml_path)

        # Resize 图像
        img_resized = cv2.resize(img, (self.img_size, self.img_size))

        # 归一化
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_normalized = (img_normalized - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

        # 转换为 CHW 格式
        img_tensor = np.transpose(img_normalized, (2, 0, 1))

        # 调整 boxes 尺寸
        scale_x = self.img_size / orig_w
        scale_y = self.img_size / orig_h

        boxes_resized = []
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            boxes_resized.append([
                xmin * scale_x / self.img_size,
                ymin * scale_y / self.img_size,
                xmax * scale_x / self.img_size,
                ymax * scale_y / self.img_size
            ])

        # 如果没有目标，返回空
        if len(boxes_resized) == 0:
            boxes_resized = [[0, 0, 0, 0]]
            labels = [0]

        return img_tensor.astype(np.float32), np.array(boxes_resized[0], dtype=np.float32), labels[0]


class SimpleDetector(nn.Layer):
    """简单的检测模型"""

    def __init__(self, num_classes=3):
        super().__init__()

        # 简单的 CNN backbone
        self.features = nn.Sequential(
            nn.Conv2D(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2D(2),
            nn.Conv2D(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2D(2),
            nn.Conv2D(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2D(2),
            nn.Conv2D(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2D(2),
            nn.AdaptiveAvgPool2D(1)
        )

        # 分类和回归头
        self.classifier = nn.Linear(256, num_classes)
        self.regressor = nn.Linear(256, 4)  # x, y, w, h

    def forward(self, x):
        features = self.features(x)
        features = paddle.flatten(features, 1)

        cls_pred = self.classifier(features)
        box_pred = self.regressor(features)

        return cls_pred, box_pred


def train():
    """训练函数"""

    print("=" * 60)
    print("🚀 开始训练动物检测模型")
    print("=" * 60)

    # 数据路径
    train_img_dir = '/home/user/ai/data/processed/VOCdevkit/VOC2007/JPEGImages'
    train_xml_dir = '/home/user/ai/data/processed/VOCdevkit/VOC2007/Annotations'

    # 创建数据集
    print("\n📦 加载训练数据...")
    train_dataset = AnimalDataset(train_img_dir, train_xml_dir, img_size=640)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)

    # 创建模型
    print("\n🏗️  创建模型...")
    model = SimpleDetector(num_classes=3)

    # 优化器和损失函数
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=0.001)
    cls_loss_fn = nn.CrossEntropyLoss()
    box_loss_fn = nn.SmoothL1Loss()

    # 训练
    print(f"\n🏋️  开始训练 (共 {len(train_loader)} 个批次)...")
    print("⚠️  CPU 训练会比较慢，请耐心等待...\n")

    num_epochs = 3  # Demo版本：快速训练3个epoch

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_idx, (images, boxes, labels) in enumerate(pbar):
            # 前向传播
            cls_pred, box_pred = model(images)

            # 计算损失
            cls_loss = cls_loss_fn(cls_pred, labels)
            box_loss = box_loss_fn(box_pred, boxes)
            loss = cls_loss + box_loss

            # 反向传播
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            # 更新进度
            total_loss += float(loss.numpy())
            pbar.set_postfix({'loss': f'{float(loss.numpy()):.4f}'})

            # 每 50 个批次打印一次
            if (batch_idx + 1) % 50 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"  Batch {batch_idx+1}/{len(train_loader)}, Avg Loss: {avg_loss:.4f}")

        avg_epoch_loss = total_loss / len(train_loader)
        print(f"\n✅ Epoch {epoch+1} 完成, 平均损失: {avg_epoch_loss:.4f}\n")

    # 保存训练权重
    output_dir = '/home/user/ai/models'
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'animal_detector.pdparams')
    paddle.save(model.state_dict(), model_path)

    print("=" * 60)
    print(f"✅ 训练完成！训练权重已保存到: {model_path}")
    print("=" * 60)

    # 导出推理模型（比赛提交用）
    print("\n🔄 导出推理模型...")
    inference_dir = os.path.join(output_dir, 'inference')
    os.makedirs(inference_dir, exist_ok=True)

    # 切换到评估模式
    model.eval()

    # 创建示例输入用于导出
    dummy_input = paddle.randn([1, 3, 640, 640], dtype='float32')

    # 使用 jit.save 导出推理模型
    inference_model_path = os.path.join(inference_dir, 'model')
    paddle.jit.save(
        layer=model,
        path=inference_model_path,
        input_spec=[paddle.static.InputSpec(shape=[None, 3, 640, 640], dtype='float32', name='image')]
    )

    print(f"✅ 推理模型已导出到: {inference_dir}/")
    print(f"   - model.pdmodel")
    print(f"   - model.pdiparams")

    # 保存模型信息
    info = {
        'num_classes': 3,
        'classes': ['monkey', 'panda', 'wolf'],
        'img_size': 640,
        'epochs': num_epochs,
        'input_shape': [1, 3, 640, 640],
        'inference_model': inference_model_path
    }

    with open(os.path.join(output_dir, 'model_info.json'), 'w') as f:
        json.dump(info, f, indent=2)

    print(f"\n📄 模型信息已保存到: {output_dir}/model_info.json")
    print("\n🎉 训练和导出全部完成！可以用于比赛提交。")


if __name__ == '__main__':
    train()
