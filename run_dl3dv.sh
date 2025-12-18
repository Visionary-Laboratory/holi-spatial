#!/usr/bin/env bash
set -euo pipefail

SCENES=(
    "1K/0a6c01ac3212768772f8f6eca86314c72d5ca320c3e3def148ddaceab23c07f4"
    # "033d0b9343"
    # "020312de8d"
    # "00a231a370"
    # "02c2ddee2a"
)

DATA_ROOT="DL3DV"
OUTPUT_ROOT="/home/liuyifei/code/posevlm/dl3dv"

for scene in "${SCENES[@]}"; do
    echo "=== Processing scene: ${scene} ==="
    
    # 临时关闭 -e，允许错误继续执行
    set +e

    # 在 mindcube 环境下跑 classic.py
    conda run -n mindcube python classic.py \
        --data-root "${DATA_ROOT}" --scene "${scene}" --output-dir /home/liuyifei/code/posevlm/dl3dv
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




python 3d_bounding_instance_gs_rerun.py --data-root DL3DV/1K/ --scene 0a6c01ac3212768772f8f6eca86314c72d5ca320c3e3def148ddaceab23c07f4 --mask-root sam_dl3dv --model-path output_DL3DV/1K/0a6c01ac3212768772f8f6eca86314c72d5ca320c3e3def148ddaceab23c07f4/ --output-dir output_yifei_dl3dv/ --erode-pixels 15
