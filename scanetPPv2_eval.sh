#!/bin/bash
# 批量运行 sam3_kuixuan.py 和 3d_bounding_instance_gs_rerun.py
# 使用四个GPU并行处理，每个GPU一个场景

# 不使用 set -e，因为需要处理并行任务中的失败情况

# 激活 conda 环境
CONDA_BASE=$(conda info --base 2>/dev/null || echo "/mnt/shared-storage-user/solution/liuyifei/miniconda3")
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate pgsr
elif command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate pgsr
else
    echo "警告: 未找到 conda，假设环境已激活"
fi

# 设置路径
GT_JSON_ROOT="mask_index_outputs"
OUTPUT_ROOT="output"

# 定义要处理的场景列表（排除 1cefb55d50）
SCENES=(
    “1cefb55d50”
    "027cd6ea0f"
    "09d6e808b4"
    "0a7cc12c0e"
    "0b031f3119"
    "0d8ead0038"
    "116456116b"
    "17a5e7d36c"
    "20871b98f3"
    "924b364b9f"
)

echo "找到 ${#SCENES[@]} 个场景: ${SCENES[*]}"
echo ""

# 运行场景的函数
run_scene() {
    local scene=$1
    local gpu_id=$2
    
    echo "[GPU $gpu_id] =========================================="
    echo "[GPU $gpu_id] 开始处理场景: $scene"
    echo "[GPU $gpu_id] =========================================="
    
    local scene_json="${GT_JSON_ROOT}/${scene}.json"
    
    if [ ! -f "$scene_json" ]; then
        echo "[GPU $gpu_id] ⚠️  警告: JSON 文件不存在: $scene_json，跳过"
        return 1
    fi
    
    # 设置GPU
    export CUDA_VISIBLE_DEVICES=$gpu_id
    
    # 第一步: 运行 sam3_kuixuan.py
    echo "[GPU $gpu_id] 步骤 1/2: 运行 sam3_kuixuan.py..."
    if ! python sam3_kuixuan.py --scene-json "$scene_json"; then
        echo "[GPU $gpu_id] ✗ 场景 $scene 的 sam3_kuixuan.py 处理失败"
        return 1
    fi
    echo "[GPU $gpu_id] ✓ 场景 $scene 的 sam3_kuixuan.py 处理成功"
    
    # 第二步: 运行 3d_bounding_instance_gs_rerun.py
    echo "[GPU $gpu_id] 步骤 2/2: 运行 3d_bounding_instance_gs_rerun.py..."
    local output_dir="${OUTPUT_ROOT}/${scene}"
    if ! python 3d_bounding_instance_gs_rerun.py --scene "$scene" -m "$output_dir"; then
        echo "[GPU $gpu_id] ✗ 场景 $scene 的 3d_bounding_instance_gs_rerun.py 处理失败"
        return 1
    fi
    echo "[GPU $gpu_id] ✓ 场景 $scene 的 3d_bounding_instance_gs_rerun.py 处理成功"
    
    echo "[GPU $gpu_id] =========================================="
    echo "[GPU $gpu_id] 场景 $scene 全部处理完成"
    echo "[GPU $gpu_id] =========================================="
    return 0
}

# 并行处理场景
scene_index=0
success_count=0
failed_scenes=()
declare -A gpu_pids
declare -A gpu_scenes  # 跟踪每个GPU当前处理的场景

# 初始化四个GPU的任务
for gpu in 0 1 2 3; do
    if [ $scene_index -lt ${#SCENES[@]} ]; then
        scene=${SCENES[$scene_index]}
        run_scene "$scene" $gpu &
        gpu_pids[$gpu]=$!
        gpu_scenes[$gpu]=$scene
        echo "启动 GPU $gpu 处理场景: $scene (PID: ${gpu_pids[$gpu]})"
        scene_index=$((scene_index + 1))
    fi
done

# 等待任务完成并分配新任务
while [ $scene_index -lt ${#SCENES[@]} ] || [ ${#gpu_pids[@]} -gt 0 ]; do
    for gpu in 0 1 2 3; do
        if [ -n "${gpu_pids[$gpu]}" ]; then
            # 检查进程是否还在运行
            if ! kill -0 ${gpu_pids[$gpu]} 2>/dev/null; then
                # 进程已完成，等待获取退出状态
                current_scene=${gpu_scenes[$gpu]}
                wait ${gpu_pids[$gpu]}
                exit_code=$?
                
                if [ $exit_code -eq 0 ]; then
                    success_count=$((success_count + 1))
                    echo "[GPU $gpu] ✓ 场景 $current_scene 处理成功"
                else
                    failed_scenes+=("$current_scene")
                    echo "[GPU $gpu] ✗ 场景 $current_scene 处理失败 (退出码: $exit_code)"
                fi
                
                # 分配新任务
                unset gpu_pids[$gpu]
                unset gpu_scenes[$gpu]
                if [ $scene_index -lt ${#SCENES[@]} ]; then
                    scene=${SCENES[$scene_index]}
                    run_scene "$scene" $gpu &
                    gpu_pids[$gpu]=$!
                    gpu_scenes[$gpu]=$scene
                    echo "启动 GPU $gpu 处理场景: $scene (PID: ${gpu_pids[$gpu]})"
                    scene_index=$((scene_index + 1))
                fi
            fi
        elif [ $scene_index -lt ${#SCENES[@]} ]; then
            # GPU空闲且有未处理场景，分配新任务
            scene=${SCENES[$scene_index]}
            run_scene "$scene" $gpu &
            gpu_pids[$gpu]=$!
            gpu_scenes[$gpu]=$scene
            echo "启动 GPU $gpu 处理场景: $scene (PID: ${gpu_pids[$gpu]})"
            scene_index=$((scene_index + 1))
        fi
    done
    
    # 短暂休眠避免CPU占用过高
    sleep 1
done

# 等待所有剩余任务完成
for gpu in 0 1 2 3; do
    if [ -n "${gpu_pids[$gpu]}" ]; then
        current_scene=${gpu_scenes[$gpu]}
        wait ${gpu_pids[$gpu]}
        exit_code=$?
        if [ $exit_code -eq 0 ]; then
            success_count=$((success_count + 1))
            echo "[GPU $gpu] ✓ 场景 $current_scene 处理成功"
        else
            failed_scenes+=("$current_scene")
            echo "[GPU $gpu] ✗ 场景 $current_scene 处理失败 (退出码: $exit_code)"
        fi
    fi
done

echo ""
echo "=========================================="
echo "所有场景处理完成"
echo "=========================================="
echo "成功: $success_count / ${#SCENES[@]}"
if [ ${#failed_scenes[@]} -gt 0 ]; then
    echo "失败场景: ${failed_scenes[*]}"
fi
