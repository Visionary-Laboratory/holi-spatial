#!/bin/bash

# 单场景示例（保留作参考）
# python classic_vllm.py --data-root scannetv2/scans --scene scene0000_00 --api-base http://100.98.124.126:8000/v1 --output-dir Qwen3VL-32B-Scannetv2
# python sam3.py --data-root scannetv2/scans --scene-json Qwen3VL-32B-Scannetv2/scene0000_00.json --output-dir sam3_masks_scannetv2
# python 3d_bounding_instance_gs_rerun.py --scene scene0000_00 --data-root scannetv2/scans -m pgsr_result_scannetv2/scene0000_00/ --mask-root sam3_masks_scannetv2/ --output-dir output_scannetv2 --vllm-api-url http://10.102.204.50:8000/v1/chat/completions --scene-json-dir Qwen3VL-32B-Scannetv2/

# ========== 批量处理：所有有 point_cloud.ply 的场景，8 卡每卡 2 场景 = 16 并行 ==========

SCENES_DIR="pgsr_result_scannetv2"
OUTPUT_JSON_DIR="output_scannetv2"
PLY_PATH="point_cloud/iteration_30000/point_cloud.ply"
SCENES=()

for scene_dir in "$SCENES_DIR"/*; do
    if [ -d "$scene_dir" ]; then
        scene=$(basename "$scene_dir")
        if [ ! -f "$scene_dir/$PLY_PATH" ]; then
            continue
        fi
        if [ -f "$OUTPUT_JSON_DIR/$scene.json" ]; then
            echo "跳过（已有结果）: $scene"
            continue
        fi
        SCENES+=("$scene")
    fi
done

echo "待处理场景数: ${#SCENES[@]}"

process_scene() {
    local scene=$1
    local gpu_id=$2
    local slot=$3
    export CUDA_VISIBLE_DEVICES=$gpu_id
    echo "[GPU${gpu_id}-${slot}] 开始: $scene"
    python classic_vllm.py --data-root scannetv2/scans --scene "$scene" --api-base http://100.98.124.126:8000/v1 --output-dir Qwen3VL-32B-Scannetv2 || { echo "[GPU${gpu_id}-${slot}] $scene classic_vllm 失败"; return 1; }
    python sam3.py --data-root scannetv2/scans --scene-json "Qwen3VL-32B-Scannetv2/${scene}.json" --output-dir sam3_masks_scannetv2 || { echo "[GPU${gpu_id}-${slot}] $scene sam3 失败"; return 1; }
    python 3d_bounding_instance_gs_rerun.py --scene "$scene" --data-root scannetv2/scans --mask-root sam3_masks_scannetv2/ -m "pgsr_result_scannetv2/${scene}/" --output-dir output_scannetv2 --vllm-api-url http://10.102.204.50:8000/v1/chat/completions --scene-json-dir Qwen3VL-32B-Scannetv2/ || { echo "[GPU${gpu_id}-${slot}] $scene 3d_bounding 失败"; return 1; }
    echo "[GPU${gpu_id}-${slot}] 完成: $scene"
}

# 16 并行，每卡 2 个场景：用全局任务序号 task_index 分配 GPU，保证轮询 0~7
MAX_PARALLEL=8
NUM_GPUS=2
pids=()
task_index=0

for scene in "${SCENES[@]}"; do
    while [ ${#pids[@]} -ge $MAX_PARALLEL ]; do
        new_pids=()
        for i in "${!pids[@]}"; do
            if kill -0 "${pids[$i]}" 2>/dev/null; then
                new_pids+=("${pids[$i]}")
            fi
        done
        pids=("${new_pids[@]}")
        [ ${#pids[@]} -ge $MAX_PARALLEL ] && sleep 5
    done

    gpu_id=$((task_index % NUM_GPUS))
    process_scene "$scene" "$gpu_id" "$task_index" &
    pids+=($!)
    echo "启动 场景=$scene GPU=$gpu_id task_index=$task_index PID=$!"
    ((task_index++))
done

echo "等待所有任务结束..."
for pid in "${pids[@]}"; do
    wait "$pid"
done
echo "全部完成。"
