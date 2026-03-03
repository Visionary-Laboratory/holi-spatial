#!/bin/bash
# 批量运行 2d_iou.py 并统计平均 IoU

set -e

# 激活 conda 环境
CONDA_BASE=$(conda info --base 2>/dev/null || echo "/mnt/shared-storage-user/solution/liuyifei/miniconda3")
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate pgsr_sam3tracker
elif command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate pgsr_sam3tracker
else
    echo "警告: 未找到 conda，假设环境已激活"
fi

# 设置路径
DATA_ROOT="scannetppv2/data"
RRD_ROOT="output_3d_bounding_scannet_evaluation"
GT_JSON_ROOT="mask_index_outputs"
OUTPUT_DIR="2d_iou_outputs_gyn_seperate_instance_scene_label_sam"

# 获取所有场景
SCENES=()
for json_file in "${GT_JSON_ROOT}"/*.json; do
    if [ -f "$json_file" ]; then
        scene_name=$(basename "$json_file" .json)
        SCENES+=("$scene_name")
    fi
done

echo "找到 ${#SCENES[@]} 个场景: ${SCENES[*]}"
echo ""

# 运行每个场景
success_count=0
failed_scenes=()

for scene in "${SCENES[@]}"; do
    echo "=========================================="
    echo "处理场景: $scene"
    echo "=========================================="
    
    rrd_path="${RRD_ROOT}/${scene}.rrd"
    gt_json_path="${GT_JSON_ROOT}/${scene}.json"
    
    if [ ! -f "$rrd_path" ]; then
        echo "⚠️  警告: RRD 文件不存在: $rrd_path，跳过"
        failed_scenes+=("$scene")
        continue
    fi
    
    if [ ! -f "$gt_json_path" ]; then
        echo "⚠️  警告: GT JSON 文件不存在: $gt_json_path，跳过"
        failed_scenes+=("$scene")
        continue
    fi
    
    if python 2d_iou_gyn_seperate_instance_scene_label.py \
        --scene "$scene" \
        --data-root "$DATA_ROOT" \
        --rrd-path "$rrd_path" \
        --gt-json "$gt_json_path" \
        --output-dir "$OUTPUT_DIR"; then
        echo "✓ 场景 $scene 处理成功"
        success_count=$((success_count + 1))
    else
        echo "✗ 场景 $scene 处理失败"
        failed_scenes+=("$scene")
    fi
    
    echo ""
done

echo "=========================================="
echo "处理完成"
echo "=========================================="
echo "成功: $success_count / ${#SCENES[@]}"
if [ ${#failed_scenes[@]} -gt 0 ]; then
    echo "失败场景: ${failed_scenes[*]}"
fi
echo ""

# 统计平均 IoU
echo "=========================================="
echo "统计平均 IoU"
echo "=========================================="

python3 << 'PYTHON_SCRIPT'
import json
from pathlib import Path
import numpy as np

output_dir = Path("2d_iou_outputs")
scenes = []
for json_file in Path("mask_index_outputs").glob("*.json"):
    scenes.append(json_file.stem)

all_ious = []
scene_stats = []

for scene in scenes:
    json_path = output_dir / f"{scene}_2d_iou.json"
    if not json_path.exists():
        print(f"⚠️  场景 {scene} 的结果文件不存在: {json_path}")
        continue
    
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        stats = data.get("statistics", {})
        if stats:
            overall_mean_iou = stats.get("overall_mean_iou", 0.0)
            all_ious.append(overall_mean_iou)
            scene_stats.append({
                "scene": scene,
                "mean_iou": overall_mean_iou,
                "total_count": stats.get("total_count", 0),
                "total_images": stats.get("total_images", 0),
                "total_labels": stats.get("total_labels", 0)
            })
            print(f"场景 {scene}: 平均 IoU = {overall_mean_iou:.4f} (计算数: {stats.get('total_count', 0)})")
    except Exception as e:
        print(f"⚠️  读取场景 {scene} 结果失败: {e}")

if all_ious:
    print("")
    print("=" * 50)
    print(f"九个场景的平均 IoU: {np.mean(all_ious):.4f}")
    print(f"中位数 IoU: {np.median(all_ious):.4f}")
    print(f"最小 IoU: {np.min(all_ious):.4f}")
    print(f"最大 IoU: {np.max(all_ious):.4f}")
    print(f"标准差: {np.std(all_ious):.4f}")
    print("=" * 50)
    
    # 按场景排序输出
    scene_stats.sort(key=lambda x: x["mean_iou"], reverse=True)
    print("\n各场景详细统计（按 IoU 降序）:")
    for stat in scene_stats:
        print(f"  {stat['scene']:20s} | IoU: {stat['mean_iou']:.4f} | 图像: {stat['total_images']:3d} | 标签: {stat['total_labels']:3d} | 计算数: {stat['total_count']:4d}")
else:
    print("⚠️  没有找到任何有效的结果文件")
PYTHON_SCRIPT

