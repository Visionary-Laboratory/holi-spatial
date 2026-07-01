#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-scannetppv2/data}"
SCENES_DIR="${SCENES_DIR:-pgsr_scannetppv2_all}"
CLASS_JSON_DIR="${CLASS_JSON_DIR:-Qwen3VL-32B-Scannetppv2}"
MASK_ROOT="${MASK_ROOT:-sam3_masks_scannetppv2_new}"
OUTPUT_JSON_DIR="${OUTPUT_JSON_DIR:-output_scannetppv2_new}"
VLLM_API_BASE="${VLLM_API_BASE:-http://localhost:8000/v1}"
VLLM_CHAT_URL="${VLLM_CHAT_URL:-http://localhost:8000/v1/chat/completions}"
BBOX_SCRIPT="${BBOX_SCRIPT:-3d_bounding_instance_gs_rerun_da3.py}"
MAX_PARALLEL="${MAX_PARALLEL:-16}"
NUM_GPUS="${NUM_GPUS:-8}"
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
    python classic_vllm.py --data-root "$DATA_ROOT" --scene "$scene" --api-base "$VLLM_API_BASE" --output-dir "$CLASS_JSON_DIR" || { echo "[GPU${gpu_id}-${slot}] $scene classic_vllm 失败"; return 1; }
    python sam3.py --data-root "$DATA_ROOT" --scene-json "$CLASS_JSON_DIR/${scene}.json" --output-dir "$MASK_ROOT" || { echo "[GPU${gpu_id}-${slot}] $scene sam3 失败"; return 1; }
    python "$BBOX_SCRIPT" --scene "$scene" --data-root "$DATA_ROOT" --mask-root "$MASK_ROOT" -m "$SCENES_DIR/${scene}/" --output-dir "$OUTPUT_JSON_DIR" --vllm-api-url "$VLLM_CHAT_URL" --scene-json-dir "$CLASS_JSON_DIR" || { echo "[GPU${gpu_id}-${slot}] $scene 3d_bounding 失败"; return 1; }
    echo "[GPU${gpu_id}-${slot}] 完成: $scene"
}

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
