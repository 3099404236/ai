#!/usr/bin/env python3
"""
将划分好的数据集转换为 PaddleDetection VOC 格式
创建 trainval.txt 和 test.txt 文件列表
"""

import os
from pathlib import Path


def create_file_list(image_dir, annotation_dir, output_file):
    """创建文件列表"""
    image_dir = Path(image_dir)
    annotation_dir = Path(annotation_dir)

    image_files = sorted(image_dir.glob('*.jpg')) + sorted(image_dir.glob('*.png'))

    with open(output_file, 'w') as f:
        for img_file in image_files:
            # 检查对应的 XML 是否存在
            xml_file = annotation_dir / (img_file.stem + '.xml')
            if xml_file.exists():
                # 只写文件名（不含扩展名）
                f.write(f"{img_file.stem}\n")
            else:
                print(f"Warning: {xml_file} not found for {img_file}")

    print(f"Created {output_file} with {len(image_files)} images")


def organize_voc_dataset(base_dir, output_dir):
    """
    组织成 VOC 格式的目录结构

    VOC 格式要求:
    VOCdevkit/
    └── VOC2007/
        ├── JPEGImages/      # 所有图像
        ├── Annotations/     # 所有标注
        └── ImageSets/
            └── Main/
                ├── trainval.txt
                └── test.txt
    """
    base_dir = Path(base_dir)
    output_dir = Path(output_dir)

    # 创建 VOC 目录结构
    voc_dir = output_dir / 'VOCdevkit' / 'VOC2007'
    jpeg_dir = voc_dir / 'JPEGImages'
    anno_dir = voc_dir / 'Annotations'
    imagesets_dir = voc_dir / 'ImageSets' / 'Main'

    jpeg_dir.mkdir(parents=True, exist_ok=True)
    anno_dir.mkdir(parents=True, exist_ok=True)
    imagesets_dir.mkdir(parents=True, exist_ok=True)

    # 合并训练集和验证集作为 trainval
    print("Creating trainval set...")
    trainval_images = []
    trainval_annos = []

    for split in ['train', 'val']:
        img_dir = base_dir / split / 'images'
        ann_dir = base_dir / split / 'annotations'

        if img_dir.exists():
            for img_file in sorted(img_dir.glob('*')):
                xml_file = ann_dir / (img_file.stem + '.xml')
                if xml_file.exists():
                    # 复制文件到 VOC 目录（使用软链接更快）
                    dest_img = jpeg_dir / img_file.name
                    dest_xml = anno_dir / xml_file.name

                    if not dest_img.exists():
                        os.symlink(img_file.absolute(), dest_img)
                    if not dest_xml.exists():
                        os.symlink(xml_file.absolute(), dest_xml)

                    trainval_images.append(img_file.stem)

    # 创建 trainval.txt
    trainval_file = imagesets_dir / 'trainval.txt'
    with open(trainval_file, 'w') as f:
        for name in sorted(trainval_images):
            f.write(f"{name}\n")

    print(f"Created trainval.txt with {len(trainval_images)} images")

    # 处理测试集
    print("Creating test set...")
    test_images = []
    test_img_dir = base_dir / 'test' / 'images'
    test_ann_dir = base_dir / 'test' / 'annotations'

    if test_img_dir.exists():
        for img_file in sorted(test_img_dir.glob('*')):
            xml_file = test_ann_dir / (img_file.stem + '.xml')
            if xml_file.exists():
                dest_img = jpeg_dir / img_file.name
                dest_xml = anno_dir / xml_file.name

                if not dest_img.exists():
                    os.symlink(img_file.absolute(), dest_img)
                if not dest_xml.exists():
                    os.symlink(xml_file.absolute(), dest_xml)

                test_images.append(img_file.stem)

    # 创建 test.txt
    test_file = imagesets_dir / 'test.txt'
    with open(test_file, 'w') as f:
        for name in sorted(test_images):
            f.write(f"{name}\n")

    print(f"Created test.txt with {len(test_images)} images")

    # 创建 label_list.txt
    label_file = voc_dir / 'label_list.txt'
    with open(label_file, 'w') as f:
        f.write("monkey\n")
        f.write("panda\n")
        f.write("wolf\n")

    print(f"\n✅ VOC dataset prepared at: {voc_dir}")
    print(f"   - trainval: {len(trainval_images)} images")
    print(f"   - test: {len(test_images)} images")
    print(f"   - total: {len(trainval_images) + len(test_images)} images")

    return str(voc_dir)


if __name__ == '__main__':
    base_dir = '/home/user/ai/data/processed/animals'
    output_dir = '/home/user/ai/data/processed'

    voc_path = organize_voc_dataset(base_dir, output_dir)
    print(f"\nDataset ready for PaddleDetection training!")
    print(f"Use dataset_dir: {output_dir}/VOCdevkit")
