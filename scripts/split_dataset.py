#!/usr/bin/env python3
"""
数据集划分脚本
将VOC格式数据集按7:2:1划分为训练集、验证集和测试集
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
import xml.etree.ElementTree as ET

def get_class_from_xml(xml_path):
    """从XML文件中获取类别"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    obj = root.find('object')
    if obj is not None:
        return obj.find('name').text
    return None

def split_dataset(xml_dir, image_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    """
    按类别分层划分数据集

    Args:
        xml_dir: XML标注文件目录
        image_dir: 图像文件目录
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
    """
    random.seed(seed)

    xml_dir = Path(xml_dir)
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)

    # 创建输出目录
    for split in ['train', 'val', 'test']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'annotations').mkdir(parents=True, exist_ok=True)

    # 按类别组织文件
    class_files = defaultdict(list)
    xml_files = list(xml_dir.glob('*.xml'))

    print(f"🔍 正在扫描 {len(xml_files)} 个标注文件...")

    for xml_file in xml_files:
        class_name = get_class_from_xml(xml_file)
        if class_name:
            # 获取对应的图像文件
            image_name = xml_file.stem + '.jpg'  # 假设是.jpg
            image_path = image_dir / image_name

            if not image_path.exists():
                # 尝试其他扩展名
                for ext in ['.png', '.jpeg', '.JPG', '.PNG']:
                    alt_path = image_dir / (xml_file.stem + ext)
                    if alt_path.exists():
                        image_path = alt_path
                        break

            if image_path.exists():
                class_files[class_name].append({
                    'xml': xml_file,
                    'image': image_path
                })
            else:
                print(f"⚠️  警告: 找不到图像文件 {image_name}")

    # 统计信息
    print(f"\n📊 按类别统计:")
    for class_name, files in class_files.items():
        print(f"  - {class_name}: {len(files)} 个样本")

    # 对每个类别进行分层划分
    splits = {
        'train': [],
        'val': [],
        'test': []
    }

    for class_name, files in class_files.items():
        random.shuffle(files)
        n = len(files)

        # 计算划分点
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        splits['train'].extend(files[:train_end])
        splits['val'].extend(files[train_end:val_end])
        splits['test'].extend(files[val_end:])

    # 打乱每个集合
    for split_files in splits.values():
        random.shuffle(split_files)

    # 复制文件到对应目录
    print(f"\n📦 正在复制文件...")
    for split_name, split_files in splits.items():
        print(f"\n{split_name.upper()} 集: {len(split_files)} 个样本")

        for file_info in split_files:
            # 复制XML
            dst_xml = output_dir / split_name / 'annotations' / file_info['xml'].name
            shutil.copy2(file_info['xml'], dst_xml)

            # 复制图像
            dst_image = output_dir / split_name / 'images' / file_info['image'].name
            shutil.copy2(file_info['image'], dst_image)

    # 生成标签文件
    label_file = output_dir / 'label_list.txt'
    with open(label_file, 'w') as f:
        for class_name in sorted(class_files.keys()):
            f.write(f"{class_name}\n")

    print(f"\n✅ 数据集划分完成！")
    print(f"\n最终统计:")
    print(f"  - 训练集: {len(splits['train'])} 个样本 ({len(splits['train'])/sum(len(s) for s in splits.values())*100:.1f}%)")
    print(f"  - 验证集: {len(splits['val'])} 个样本 ({len(splits['val'])/sum(len(s) for s in splits.values())*100:.1f}%)")
    print(f"  - 测试集: {len(splits['test'])} 个样本 ({len(splits['test'])/sum(len(s) for s in splits.values())*100:.1f}%)")
    print(f"\n输出目录: {output_dir}")
    print(f"标签文件: {label_file}")

if __name__ == '__main__':
    xml_dir = '/home/user/ai/data/raw/wild_animal/wild_animals/xml'
    image_dir = '/home/user/ai/data/raw/wild_animal/wild_animals/image'
    output_dir = '/home/user/ai/data/processed/animals'

    split_dataset(xml_dir, image_dir, output_dir)
