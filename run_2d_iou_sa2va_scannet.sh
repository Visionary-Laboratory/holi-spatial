#!/bin/bash
# 运行 2d_iou_Sa2VA.py 处理 scannet_pgsr_eval 下的五个场景

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate Sa2VA

# 五个场景
scenes=(
    "scene0181_02"
    "scene0204_02"
    "scene0347_02"
    "scene0520_00"
)

# 输出目录
output_dir="2d_iou_sa2va_outputs_scannet"
mkdir -p "$output_dir"

# 日志目录
log_dir="$output_dir/logs"
mkdir -p "$log_dir"

# 依次运行每个场景
for scene in "${scenes[@]}"; do
    echo "=========================================="
    echo "处理场景: $scene"
    echo "=========================================="
    
    log_file="$log_dir/${scene}.log"
    
    python 2d_iou_Sa2VA.py \
        --scene "$scene" \
        --data-root scannet_pgsr_eval \
        --gt-json "mask_index_outputs_scannet/${scene}.json" \
        --output-dir "$output_dir" \
        2>&1 | tee "$log_file"
    
    echo "场景 $scene 处理完成，日志已保存到: $log_file"
    echo ""
done

echo "=========================================="
echo "所有场景处理完成！"
echo "=========================================="

