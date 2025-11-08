#!/usr/bin/env python3
"""
单独导出推理模型脚本
如果训练已完成，可以用这个脚本单独导出推理模型
"""

import paddle
import paddle.nn as nn
import os
import json


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


def export_inference_model(weights_path, output_dir):
    """导出推理模型"""

    print("=" * 60)
    print("🔄 开始导出推理模型")
    print("=" * 60)

    # 加载模型
    print(f"\n📥 加载训练权重: {weights_path}")
    model = SimpleDetector(num_classes=3)
    state_dict = paddle.load(weights_path)
    model.set_state_dict(state_dict)
    model.eval()

    print("✅ 模型加载成功")

    # 导出推理模型
    print("\n🔄 导出为推理格式...")
    os.makedirs(output_dir, exist_ok=True)

    inference_model_path = os.path.join(output_dir, 'model')
    paddle.jit.save(
        layer=model,
        path=inference_model_path,
        input_spec=[paddle.static.InputSpec(shape=[None, 3, 640, 640], dtype='float32', name='image')]
    )

    print("=" * 60)
    print(f"✅ 推理模型已导出到: {output_dir}/")
    print(f"   - model.pdmodel  (模型结构)")
    print(f"   - model.pdiparams  (模型参数)")
    print("=" * 60)

    # 检查文件大小
    pdmodel_size = os.path.getsize(os.path.join(output_dir, 'model.pdmodel'))
    pdiparams_size = os.path.getsize(os.path.join(output_dir, 'model.pdiparams'))
    total_size = pdmodel_size + pdiparams_size

    print(f"\n📊 模型大小:")
    print(f"   - model.pdmodel: {pdmodel_size / 1024:.2f} KB")
    print(f"   - model.pdiparams: {pdiparams_size / (1024*1024):.2f} MB")
    print(f"   - 总大小: {total_size / (1024*1024):.2f} MB")

    if total_size < 200 * 1024 * 1024:
        print(f"   ✅ 模型大小符合比赛要求（< 200MB）")
    else:
        print(f"   ⚠️  警告：模型大小超过 200MB 限制！")

    # 保存配置信息
    config = {
        'model_type': 'SimpleDetector',
        'num_classes': 3,
        'classes': ['monkey', 'panda', 'wolf'],
        'img_size': 640,
        'input_shape': [1, 3, 640, 640],
        'model_files': {
            'pdmodel': 'model.pdmodel',
            'pdiparams': 'model.pdiparams'
        }
    }

    config_path = os.path.join(output_dir, 'infer_cfg.yml')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n📄 配置文件已保存: {config_path}")
    print("\n🎉 导出完成！现在可以用于比赛提交。")


if __name__ == '__main__':
    # 默认路径
    weights_path = '/home/user/ai/models/animal_detector.pdparams'
    output_dir = '/home/user/ai/models/inference'

    if os.path.exists(weights_path):
        export_inference_model(weights_path, output_dir)
    else:
        print(f"❌ 错误：找不到训练权重文件: {weights_path}")
        print("请先完成训练，或指定正确的权重文件路径")
