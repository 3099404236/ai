#!/usr/bin/env python3
"""
目标检测预测脚本 - 符合 AI Studio 比赛提交要求

比赛要求:
1. 读取 data.txt (包含图像路径列表)
2. 使用训练好的模型进行预测
3. 将结果保存到 result.json
4. 预测速度需在 V100 上达到 20 FPS 以上

运行方式:
    python predict.py data.txt result.json
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
import paddle
from paddle import inference


class AnimalDetector:
    """动物目标检测器"""

    def __init__(self, model_dir, use_gpu=True):
        """
        初始化检测器

        Args:
            model_dir: 模型目录路径
            use_gpu: 是否使用 GPU
        """
        self.model_dir = Path(model_dir)
        self.use_gpu = use_gpu

        # 类别标签
        self.labels = ['monkey', 'panda', 'wolf']

        # 加载模型
        self.predictor = self._load_model()

    def _load_model(self):
        """加载 Paddle 推理模型"""
        model_file = str(self.model_dir / 'model.pdmodel')
        params_file = str(self.model_dir / 'model.pdiparams')

        # 配置推理
        config = inference.Config(model_file, params_file)

        if self.use_gpu:
            config.enable_use_gpu(1000, 0)  # GPU 显存 1000MB, 设备ID 0
        else:
            config.disable_gpu()
            config.set_cpu_math_library_num_threads(4)

        # 启用内存优化
        config.enable_memory_optim()

        # 创建预测器
        predictor = inference.create_predictor(config)

        return predictor

    def preprocess(self, image_path, target_size=640):
        """
        图像预处理

        Args:
            image_path: 图像路径
            target_size: 目标尺寸

        Returns:
            处理后的图像数组和原始尺寸
        """
        import cv2

        # 读取图像
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        orig_h, orig_w = img.shape[:2]

        # Resize 保持宽高比
        scale = min(target_size / orig_w, target_size / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        img_resized = cv2.resize(img, (new_w, new_h))

        # Padding to target_size
        img_padded = np.ones((target_size, target_size, 3), dtype=np.uint8) * 114
        img_padded[:new_h, :new_w, :] = img_resized

        # 转换为 RGB 并归一化
        img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0

        # 转换为 CHW 格式
        img_chw = np.transpose(img_normalized, (2, 0, 1))

        # 添加 batch 维度
        img_batch = np.expand_dims(img_chw, axis=0)

        return img_batch, (orig_w, orig_h), scale

    def postprocess(self, outputs, orig_size, scale, conf_thresh=0.5):
        """
        后处理预测结果

        Args:
            outputs: 模型输出
            orig_size: 原始图像尺寸 (width, height)
            scale: 缩放比例
            conf_thresh: 置信度阈值

        Returns:
            检测结果列表
        """
        # 根据实际模型输出格式调整
        # 这里假设输出格式: [num_boxes, 6] (x1, y1, x2, y2, conf, class_id)

        results = []

        if len(outputs) == 0:
            return results

        boxes = outputs[0]  # 获取第一个输出（boxes）

        orig_w, orig_h = orig_size

        for box in boxes:
            if len(box) < 6:
                continue

            x1, y1, x2, y2, conf, class_id = box[:6]

            if conf < conf_thresh:
                continue

            # 转换回原始图像坐标
            x1 = int(x1 / scale)
            y1 = int(y1 / scale)
            x2 = int(x2 / scale)
            y2 = int(y2 / scale)

            # 确保坐标在图像范围内
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))

            class_id = int(class_id)
            class_name = self.labels[class_id] if class_id < len(self.labels) else 'unknown'

            results.append({
                'bbox': [x1, y1, x2, y2],
                'score': float(conf),
                'category': class_name,
                'category_id': class_id
            })

        return results

    def predict(self, image_path, conf_thresh=0.5):
        """
        预测单张图像

        Args:
            image_path: 图像路径
            conf_thresh: 置信度阈值

        Returns:
            检测结果列表
        """
        # 预处理
        img_batch, orig_size, scale = self.preprocess(image_path)

        # 获取输入输出 tensor
        input_names = self.predictor.get_input_names()
        input_handle = self.predictor.get_input_handle(input_names[0])

        # 设置输入
        input_handle.copy_from_cpu(img_batch)

        # 运行推理
        self.predictor.run()

        # 获取输出
        output_names = self.predictor.get_output_names()
        outputs = []
        for output_name in output_names:
            output_handle = self.predictor.get_output_handle(output_name)
            output_data = output_handle.copy_to_cpu()
            outputs.append(output_data)

        # 后处理
        results = self.postprocess(outputs, orig_size, scale, conf_thresh)

        return results


def main(data_file, output_file):
    """
    主函数

    Args:
        data_file: 输入数据文件路径 (包含图像路径列表)
        output_file: 输出结果文件路径 (JSON格式)
    """
    print(f"[INFO] 开始预测...")
    print(f"[INFO] 数据文件: {data_file}")
    print(f"[INFO] 输出文件: {output_file}")

    # 读取图像路径列表
    with open(data_file, 'r') as f:
        image_paths = [line.strip() for line in f if line.strip()]

    print(f"[INFO] 共 {len(image_paths)} 张图像待预测")

    # 初始化检测器
    model_dir = './model'  # 模型目录
    detector = AnimalDetector(model_dir, use_gpu=True)

    # 预测所有图像
    all_results = {}
    total_time = 0.0

    for i, img_path in enumerate(image_paths):
        print(f"[{i+1}/{len(image_paths)}] 处理: {img_path}")

        start_time = time.time()

        try:
            results = detector.predict(img_path, conf_thresh=0.5)
            all_results[img_path] = results

        except Exception as e:
            print(f"[ERROR] 预测失败: {img_path}, 错误: {e}")
            all_results[img_path] = []

        elapsed = time.time() - start_time
        total_time += elapsed

        print(f"  -> 检测到 {len(results)} 个目标, 用时: {elapsed:.3f}s")

    # 计算 FPS
    avg_time = total_time / len(image_paths)
    fps = 1.0 / avg_time if avg_time > 0 else 0

    print(f"\n[INFO] 预测完成!")
    print(f"[INFO] 平均每张用时: {avg_time:.3f}s")
    print(f"[INFO] 平均 FPS: {fps:.2f}")

    if fps < 20:
        print(f"[WARNING] FPS 低于要求的 20 FPS，请考虑使用更轻量的模型或优化代码")

    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"[INFO] 结果已保存到: {output_file}")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("用法: python predict.py <data.txt> <result.json>")
        print("示例: python predict.py data.txt result.json")
        sys.exit(1)

    data_file = sys.argv[1]
    output_file = sys.argv[2]

    main(data_file, output_file)
