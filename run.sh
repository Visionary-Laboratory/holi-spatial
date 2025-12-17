#!/usr/bin/env bash
set -euo pipefail

SCENES=(
    "00dd871005"
    "033d0b9343"
    "020312de8d"
    "00a231a370"
    "02c2ddee2a"
)

DATA_ROOT="/home/liuyifei/code/posevlm/scannetppv2/data"
OUTPUT_ROOT="/home/liuyifei/code/posevlm/output"

for scene in "${SCENES[@]}"; do
    echo "=== Processing scene: ${scene} ==="
    
    # 临时关闭 -e，允许错误继续执行
    set +e

    # 在 mindcube 环境下跑 classic.py
    conda run -n mindcube python classic.py \
        --data-root "${DATA_ROOT}" --scene "${scene}"
    if [ $? -ne 0 ]; then
        echo "❌ Error: classic.py failed for scene ${scene}, continuing..."
    fi

    # 在 Octree 环境下跑 sam3 + 3d_bounding_instance_gs_rerun.py
    conda run -n mindcube python sam3.py \
        --scene-json "scene_objects_Qwen3-VL-30B-A3B-Instruct/${scene}.json"
    if [ $? -ne 0 ]; then
        echo "❌ Error: sam3.py failed for scene ${scene}, continuing..."
    fi

    conda run -n Octree python 3d_bounding_instance_gs_rerun.py \
        --scene "${scene}" -m "${OUTPUT_ROOT}/${scene}"
    if [ $? -ne 0 ]; then
        echo "❌ Error: 3d_bounding_instance_gs_rerun.py failed for scene ${scene}, continuing..."
    fi
    
    # 恢复 -e 设置
    set -e
    
    echo "✓ Completed scene: ${scene}"
    echo ""
done