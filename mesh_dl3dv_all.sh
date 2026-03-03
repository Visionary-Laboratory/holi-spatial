#!/bin/bash

# 批量处理 DL3DV 1K 数据集：对 processed_dl3dv_ours/1K 下所有场景跑 render.py + mesh2mask.py
# 若场景已有 mesh/tsdf_fusion_post.ply 则跳过；使用多卡并行，每张卡处理一个场景

echo "开始批量处理 DL3DV 1K 下的所有场景..."

DATA_ROOT="processed_dl3dv_ours/1K"
OUTPUT_ROOT="pgsr_DL3DV_all/1K"

if [ ! -d "$DATA_ROOT" ]; then
    echo "错误: 数据目录不存在 $DATA_ROOT"
    exit 1
fi

# 收集需要处理的场景（无 mesh 的）
SCENES=()
for scene_dir in "${DATA_ROOT}"/*/; do
    if [ ! -d "$scene_dir" ]; then
        continue
    fi
    scene=$(basename "$scene_dir")
    SCENES+=("$scene")
done

echo "找到 ${#SCENES[@]} 个需要处理的场景"

# 处理单个场景（render + mesh2mask）
process_scene() {
    local scene=$1
    local gpu_id=$2

    echo "[GPU $gpu_id] 开始处理场景: $scene"

    export CUDA_VISIBLE_DEVICES=$gpu_id

    data_dir="${DATA_ROOT}/${scene}"
    mesh_dir="${OUTPUT_ROOT}/${scene}"
    mesh_file="${mesh_dir}/mesh/tsdf_fusion_post.ply"

    # 1. Render
    echo "[GPU $gpu_id] 场景 $scene: 开始 render..."
    python PGSR/render.py \
        -s "$data_dir" \
        -m "$mesh_dir" \
        --skip_test

    if [ $? -ne 0 ]; then
        echo "[GPU $gpu_id] 错误: 场景 $scene render 失败"
        return 1
    fi

    if [ ! -f "$mesh_file" ]; then
        echo "[GPU $gpu_id] 警告: 场景 $scene mesh 未生成，跳过 mesh2mask"
        return 1
    fi

    # 2. Mesh2Mask（直接运行，覆盖已有 mask）
    echo "[GPU $gpu_id] 场景 $scene: 开始 mesh2mask..."
    python PGSR/mesh2mask.py \
        -m "$mesh_dir" \
        -s "$data_dir" \
        --mesh_path "mesh/tsdf_fusion_post.ply"
    if [ $? -ne 0 ]; then
        echo "[GPU $gpu_id] 错误: 场景 $scene mesh2mask 失败"
        return 1
    fi

    echo "[GPU $gpu_id] 场景 $scene 处理完成！"
    return 0
}

# 多卡并行：8 张卡，每张卡跑 2 个场景，共 16 个并行任务
NUM_GPUS=8
PER_GPU=1
MAX_PARALLEL=3  # 16
current_slot=0
pids=()

for scene in "${SCENES[@]}"; do
    while [ ${#pids[@]} -ge $MAX_PARALLEL ]; do
        new_pids=()
        for pid in "${pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                new_pids+=("$pid")
            fi
        done
        pids=("${new_pids[@]}")
        if [ ${#pids[@]} -ge $MAX_PARALLEL ]; then
            sleep 5
        fi
    done

    gpu_id=$(( current_slot % NUM_GPUS ))
    process_scene "$scene" "$gpu_id" &
    pid=$!
    pids+=("$pid")
    echo "场景 $scene 已分配到 GPU $gpu_id (PID: $pid)"

    current_slot=$(( current_slot + 1 ))
done

echo "等待所有任务完成..."
for pid in "${pids[@]}"; do
    wait "$pid"
done

echo "所有场景处理完成！"
