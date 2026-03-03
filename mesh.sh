#!/bin/bash

# 批量处理所有场景的脚本
# 自动遍历 pgsr_scannetppv2_all/ 下的所有场景
# 如果已存在 mesh/tsdf_fusion_post.ply 则跳过
# 使用8张GPU并行处理，每张卡处理一个场景

echo "开始批量处理所有场景..."

# 获取所有需要处理的场景列表
SCENES_DIR="pgsr_scannetppv2_all"
SCENES=()

# 遍历所有场景目录
for scene_dir in "$SCENES_DIR"/*; do
    if [ -d "$scene_dir" ]; then
        scene=$(basename "$scene_dir")
        mesh_file="$scene_dir/mesh/tsdf_fusion_post.ply"
        
        # 检查是否已存在 mesh 文件
        if [ -f "$mesh_file" ]; then
            echo "场景 $scene 已存在 mesh 文件，跳过: $mesh_file"
            continue
        fi
        
        # 检查必要的输入目录是否存在
        nerfstudio_dir="scannetppv2/data/$scene/dslr/nerfstudio"
        images_dir="scannetppv2/data/$scene/dslr/resized_undistorted_images"
        
        if [ ! -d "$nerfstudio_dir" ] || [ ! -d "$images_dir" ]; then
            echo "警告: 场景 $scene 缺少必要的输入目录，跳过"
            continue
        fi
        
        SCENES+=("$scene")
    fi
done

echo "找到 ${#SCENES[@]} 个需要处理的场景"

# 处理单个场景的函数
process_scene() {
    local scene=$1
    local gpu_id=$2
    
    echo "[GPU $gpu_id] 开始处理场景: $scene"
    
    # 设置使用的GPU
    export CUDA_VISIBLE_DEVICES=$gpu_id
    
    # 第一步: 渲染深度图
    echo "[GPU $gpu_id] 场景 $scene: 开始渲染深度图..."
    python PGSR/render.py \
        -s "scannetppv2/data/$scene/dslr/nerfstudio" \
        -m "pgsr_scannetppv2_all/$scene" \
        --skip_test \
        -i "scannetppv2/data/$scene/dslr/resized_undistorted_images"
    
    if [ $? -ne 0 ]; then
        echo "[GPU $gpu_id] 错误: 场景 $scene 渲染失败"
        return 1
    fi
    
    # 第二步: 生成 mask
    echo "[GPU $gpu_id] 场景 $scene: 开始生成 mask..."
    python PGSR/mesh2mask.py \
        -m "pgsr_scannetppv2_all/$scene" \
        -s "scannetppv2/data/$scene/dslr/nerfstudio" \
        --mesh_path "mesh/tsdf_fusion_post.ply"
    
    if [ $? -ne 0 ]; then
        echo "[GPU $gpu_id] 错误: 场景 $scene mask 生成失败"
        return 1
    fi
    
    echo "[GPU $gpu_id] 场景 $scene 处理完成！"
    return 0
}

# 使用8张GPU并行处理
MAX_PARALLEL=3
current_gpu=0
pids=()

# 处理所有场景
for scene in "${SCENES[@]}"; do
    # 等待有空闲的GPU
    while [ ${#pids[@]} -ge $MAX_PARALLEL ]; do
        # 检查已完成的进程
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
    
    # 在后台启动处理任务
    process_scene "$scene" "$current_gpu" &
    pid=$!
    pids+=("$pid")
    
    echo "场景 $scene 已分配到 GPU $current_gpu (PID: $pid)"
    
    # 更新GPU索引（循环使用0-7）
    current_gpu=$(( (current_gpu + 1) % MAX_PARALLEL ))
done

# 等待所有后台任务完成
echo "等待所有任务完成..."
for pid in "${pids[@]}"; do
    wait "$pid"
done

echo "所有场景处理完成！"
