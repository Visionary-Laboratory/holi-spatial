#!/usr/bin/env bash
set -euo pipefail

# SCENES=(
# "0271889ec0"
# )







DATA_ROOT="/home/liuyifei/code/posevlm/scannetppv2/data"
OUTPUT_ROOT="/home/liuyifei/code/posevlm/output"
PROCESSED_ROOT="/home/liuyifei/code/posevlm/output_3d_bounding"


#run PGSR

## Scannetpp
# bash train_pgsr.sh
# ## DL3DV
# bash pgsr_train_dl3dv.sh

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

# 初始化状态记录文件
STATUS_JSON="3d_bounding_scannet.json"
export ALL_SCENES_STR="${SCENES[*]}"
python - <<PY
import json
import os
from pathlib import Path

scenes = os.environ.get("ALL_SCENES_STR", "").split()
status_file = "${STATUS_JSON}"

existing_status = {}
if os.path.exists(status_file):
    try:
        with open(status_file, "r") as f:
            existing_status = json.load(f)
    except Exception:
        pass

status_data = {}
# 首先同步当前发现的所有需要处理的场景
for s in scenes:
    if existing_status.get(s) == "completed":
        status_data[s] = "completed"
    else:
        status_data[s] = "pending"

# 保留旧状态文件中已完成但可能不在当前 SCENES 里的场景记录
for s, status in existing_status.items():
    if s not in status_data:
        status_data[s] = status

with open(status_file, "w") as f:
    json.dump(status_data, f, indent=2)
PY

# 更新状态的辅助函数
update_status() {
    local scene="$1"
    local status="$2"
    python - <<PY
import json
import os
with open("${STATUS_JSON}", "r+") as f:
    data = json.load(f)
    data["${scene}"] = "${status}"
    f.seek(0)
    json.dump(data, f, indent=2)
    f.truncate()
PY
}

# 并行配置
GPU_IDS=(0 1 2 3 4 5 6 7) # 指定可用的 GPU 编号
GPU_COUNT=${#GPU_IDS[@]}
MAX_PARALLEL=8   # 总并行进程数

# 初始化 GPU 令牌桶，用于管理每个 GPU 的任务分配
TOKEN_DIR=$(mktemp -d)
trap 'rm -rf "$TOKEN_DIR"' EXIT

# 创建总计 MAX_PARALLEL 个令牌，并循环分配 GPU ID
for ((i=0; i<MAX_PARALLEL; i++)); do
    gpu_id=${GPU_IDS[$((i % GPU_COUNT))]}
    touch "${TOKEN_DIR}/slot_${i}_gpu_${gpu_id}"
done

process_scene() {
    local scene="$1"
    
    # 原子获取一个可用的 GPU 令牌
    local token=""
    local gpu_id=""
    while true; do
        for f in "${TOKEN_DIR}"/slot_*; do
            [ -e "$f" ] || continue
            token=$(basename "$f")
            # 跳过已经在忙碌中的令牌
            if [[ "$token" == *.busy ]]; then continue; fi
            
            # 尝试抢占令牌
            if mv "$f" "${TOKEN_DIR}/${token}.busy" 2>/dev/null; then
                # 提取 GPU ID (只取数字部分)
                gpu_id=$(echo "$token" | sed 's/.*_gpu_\([0-9]*\).*/\1/')
                break 2
            fi
        done
        sleep 1
    done

    echo "=== Processing scene: ${scene} on GPU ${gpu_id} ==="
    update_status "${scene}" "running"
    
    # 设置当前进程使用的 GPU
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    
    local success=true

    # 既然环境已激活，直接使用 python 运行
    python classic.py --data-root "${DATA_ROOT}" --scene "${scene}"
    if [ $? -ne 0 ]; then
        echo "❌ Error: classic.py failed for scene ${scene} on GPU ${gpu_id}"
        success=false
    fi

    python sam3.py --scene-json "scene_objects_Qwen3-VL-30B-A3B-Instruct/${scene}.json"
    if [ $? -ne 0 ]; then
        echo "❌ Error: sam3.py failed for scene ${scene} on GPU ${gpu_id}"
        success=false
    fi

    python 3d_bounding_instance_gs_rerun.py --scene "${scene}" -m "${OUTPUT_ROOT}/${scene}"
    if [ $? -ne 0 ]; then
        echo "❌ Error: 3d_bounding_instance_gs_rerun.py failed for scene ${scene} on GPU ${gpu_id}"
        success=false
    fi
    
    if [ "$success" = true ]; then
        echo "✓ Completed scene: ${scene} on GPU ${gpu_id}"
        update_status "${scene}" "completed"
    else
        update_status "${scene}" "failed"
    fi

    # 释放令牌，归还到池中
    mv "${TOKEN_DIR}/${token}.busy" "${TOKEN_DIR}/${token}"
}

for scene in "${SCENES[@]}"; do
    # 检测 3d_bounding_scannet.json 中的状态，如果是 completed 则跳过
    status=$(python -c "import json; print(json.load(open('${STATUS_JSON}')).get('${scene}', 'pending'))")
    if [ "$status" = "completed" ]; then
        echo ">>> Skipping already completed scene (from status JSON): ${scene}"
        continue
    fi

    process_scene "${scene}" &
    
    # 限制总并行任务数量（等待令牌机制实际上已经限制了，但这里保留以保持逻辑清晰）
    while [ $(jobs -rp | wc -l) -ge ${MAX_PARALLEL} ]; do
        sleep 1
    done
done

# 等待所有剩余任务完成
wait
echo "所有任务已完成。"