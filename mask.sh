#!/bin/bash

# 对所有已有 mesh 的场景执行 mesh2mask
# 只处理存在 pgsr_scannetppv2_all/<scene>/mesh/tsdf_fusion_post.ply 的场景
# 8 张卡并行，每张卡处理一个场景

echo "开始批量生成 mask..."

SCENES_DIR="pgsr_scannetppv2_all"
SCENES=()

for scene_dir in "$SCENES_DIR"/*; do
    if [ -d "$scene_dir" ]; then
        scene=$(basename "$scene_dir")
        mesh_file="$scene_dir/mesh/tsdf_fusion_post.ply"
        if [ -f "$mesh_file" ]; then
            SCENES+=("$scene")
        fi
    fi
done

echo "找到 ${#SCENES[@]} 个有 mesh 的场景"

process_scene() {
    local scene=$1
    local gpu_id=$2
    echo "[GPU $gpu_id] 处理场景: $scene"
    export CUDA_VISIBLE_DEVICES=$gpu_id
    python PGSR/mesh2mask.py \
        -m "pgsr_scannetppv2_all/$scene" \
        -s "scannetppv2/data/$scene/dslr/nerfstudio" \
        --mesh_path "mesh/tsdf_fusion_post.ply"
    if [ $? -ne 0 ]; then
        echo "[GPU $gpu_id] 错误: 场景 $scene mask 生成失败"
        return 1
    fi
    echo "[GPU $gpu_id] 场景 $scene 完成"
    return 0
}

MAX_PARALLEL=8
current_gpu=0
pids=()

for scene in "${SCENES[@]}"; do
    while [ ${#pids[@]} -ge $MAX_PARALLEL ]; do
        new_pids=()
        for pid in "${pids[@]}"; do
            kill -0 "$pid" 2>/dev/null && new_pids+=("$pid")
        done
        pids=("${new_pids[@]}")
        [ ${#pids[@]} -ge $MAX_PARALLEL ] && sleep 5
    done

    process_scene "$scene" "$current_gpu" &
    pids+=($!)
    echo "场景 $scene -> GPU $current_gpu (PID: $!)"
    current_gpu=$(( (current_gpu + 1) % MAX_PARALLEL ))
done

echo "等待所有任务完成..."
for pid in "${pids[@]}"; do
    wait "$pid"
done
echo "全部完成！"



 python PGSR/render_cuda.py -s  processed_dl3dv_ours/1K/1b331d86ed9501a65135172a15348b6c823f6808ccc93fcc89d7c3b2eb19f6c5/  -m pgsr_DL3DV_all/1K/1b331d86ed9501a65135172a15348b6c823f6808ccc93fcc89d7c3b2eb19f6c5 --skip_test             