#!/bin/bash
# 动物检测模型训练脚本

cd /home/user/ai/PaddleDetection

echo "======================================"
echo "开始训练动物检测模型"
echo "======================================"
echo "数据集: 2475 张动物图像 (monkey, panda, wolf)"
echo "模型: PP-YOLOE-s"
echo "Epochs: 50"
echo "Batch size: 2"
echo "Image size: 640x640"
echo "环境: CPU (训练会较慢)"
echo "======================================"
echo ""

# 训练
python tools/train.py \
    -c configs/animals/ppyoloe_s_animals.yml \
    --eval \
    --use_vdl=True \
    --vdl_log_dir=vdl_dir/animals

echo ""
echo "======================================"
echo "训练完成！"
echo "模型保存在: output/ppyoloe_crn_s_80e_animals/"
echo "======================================"
