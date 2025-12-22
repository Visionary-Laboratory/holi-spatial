#!/usr/bin/env bash
set -euo pipefail

# SCENES=(
# "0271889ec0"
# )

DATA_ROOT="/home/liuyifei/code/posevlm/scannetppv2/data"
OUTPUT_ROOT="/home/liuyifei/code/posevlm/output"
PROCESSED_ROOT="/home/liuyifei/code/posevlm/output_yifei"

# 自动收集需要处理的场景：必须存在 point_cloud/iteration_30000/point_cloud.ply，
# 并且 output_yifei 下还没有同名 json（视为已处理）。
readarray -t SCENES < <(
    OUTPUT_ROOT="${OUTPUT_ROOT}" PROCESSED_ROOT="${PROCESSED_ROOT}" python - <<'PY'
from pathlib import Path
import os

output_root = Path(os.environ["OUTPUT_ROOT"])
processed_root = Path(os.environ["PROCESSED_ROOT"])

scenes = []
for entry in sorted(output_root.iterdir()):
    if not entry.is_dir():
        continue
    ply_path = entry / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    if not ply_path.exists():
        continue
    if (processed_root / f"{entry.name}.json").exists():
        continue
    scenes.append(entry.name)

print("\n".join(scenes))
PY
)

echo "待处理场景数量: ${#SCENES[@]}"

if [ ${#SCENES[@]} -eq 0 ]; then
    echo "没有符合条件的场景需要处理，直接退出。"
    exit 0
fi



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