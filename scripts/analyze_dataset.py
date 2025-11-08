#!/usr/bin/env python3
"""
数据集分析脚本
分析VOC格式的动物检测数据集
"""

import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

def parse_xml(xml_path):
    """解析VOC格式的XML文件"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    info = {
        'filename': root.find('filename').text,
        'width': int(root.find('size/width').text),
        'height': int(root.find('size/height').text),
        'objects': []
    }

    for obj in root.findall('object'):
        obj_info = {
            'name': obj.find('name').text,
            'xmin': int(obj.find('bndbox/xmin').text),
            'ymin': int(obj.find('bndbox/ymin').text),
            'xmax': int(obj.find('bndbox/xmax').text),
            'ymax': int(obj.find('bndbox/ymax').text),
        }
        info['objects'].append(obj_info)

    return info

def analyze_dataset(xml_dir):
    """分析整个数据集"""
    xml_dir = Path(xml_dir)
    xml_files = list(xml_dir.glob('*.xml'))

    class_count = defaultdict(int)
    total_objects = 0
    image_sizes = []
    objects_per_image = []

    print(f"📊 开始分析数据集...")
    print(f"XML文件总数: {len(xml_files)}\n")

    for xml_file in xml_files:
        try:
            info = parse_xml(xml_file)
            image_sizes.append((info['width'], info['height']))
            objects_per_image.append(len(info['objects']))

            for obj in info['objects']:
                class_count[obj['name']] += 1
                total_objects += 1
        except Exception as e:
            print(f"警告: 解析 {xml_file.name} 失败: {e}")

    # 统计结果
    print("=" * 60)
    print("📈 数据集统计信息")
    print("=" * 60)
    print(f"总图像数: {len(xml_files)}")
    print(f"总目标数: {total_objects}")
    print(f"\n类别分布:")
    for cls, count in sorted(class_count.items()):
        percentage = (count / total_objects) * 100
        print(f"  - {cls}: {count} ({percentage:.2f}%)")

    # 图像尺寸统计
    print(f"\n图像尺寸分布:")
    size_count = defaultdict(int)
    for size in image_sizes:
        size_count[size] += 1

    for size, count in sorted(size_count.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {size[0]}x{size[1]}: {count} 张")

    # 每张图像的目标数统计
    avg_objects = sum(objects_per_image) / len(objects_per_image)
    print(f"\n平均每张图像目标数: {avg_objects:.2f}")
    print(f"最多目标数: {max(objects_per_image)}")
    print(f"最少目标数: {min(objects_per_image)}")

    print("=" * 60)

    return {
        'total_images': len(xml_files),
        'total_objects': total_objects,
        'class_count': dict(class_count),
        'avg_objects_per_image': avg_objects
    }

if __name__ == '__main__':
    xml_dir = '/home/user/ai/data/raw/wild_animal/wild_animals/xml'
    result = analyze_dataset(xml_dir)
