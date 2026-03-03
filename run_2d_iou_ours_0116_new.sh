#!/bin/bash
# 批量运行 2d_iou_gyn_seperate_instance_scene_label_gyy.py
# 支持指定GPU数量并行处理

# 解析命令行参数
NUM_GPUS=1  # 默认使用4个GPU
if [ $# -ge 1 ]; then
    NUM_GPUS=$1
fi

# 验证GPU数量
if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]] || [ "$NUM_GPUS" -le 0 ]; then
    echo "错误: GPU数量必须是正整数"
    echo "用法: $0 [GPU数量]"
    echo "示例: $0 4    # 使用4个GPU (默认)"
    echo "      $0 2    # 使用2个GPU"
    exit 1
fi

# 生成GPU ID列表
GPUS=()
for ((i=0; i<$NUM_GPUS; i++)); do
    GPUS+=($i)
done

echo "将使用 ${NUM_GPUS} 个GPU: ${GPUS[@]}"

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
OUTPUT_DIR="gyn_test_images/0116_new_ours"
RRD_ROOT="0116_scannetpp/output_3d_bounding_scannetppv2_evaluation"

# 定义 10 个场景
SCENES=(
    "027cd6ea0f"
    "09d6e808b4"
    "0a7cc12c0e"
    "0b031f3119"
    "0d8ead0038"
    "116456116b"
    "17a5e7d36c"
    "1cefb55d50"
    "20871b98f3"
    "924b364b9f"
)

echo "准备处理 ${#SCENES[@]} 个场景..."
mkdir -p "$OUTPUT_DIR"

# 运行场景的函数
run_scene() {
    local scene=$1
    local gpu_id=$2
    
    echo "[GPU $gpu_id] 开始处理场景: $scene"
    
    # 设置GPU
    export CUDA_VISIBLE_DEVICES=$gpu_id
    
    python 2d_iou_gyn_seperate_instance_scene_label.py \
        --scene "$scene" \
        --rrd-path "${RRD_ROOT}/${scene}.rrd" \
        --mask-selection "merge" \
        --output-dir "$OUTPUT_DIR" 
    local status=$?
    if [ $status -eq 0 ]; then
        echo "[GPU $gpu_id] ✓ 场景 $scene 处理成功"
    else
        echo "[GPU $gpu_id] ✗ 场景 $scene 处理失败 (退出码: $status)"
    fi
    return $status
}

# 并行逻辑
scene_index=0
declare -A gpu_pids
declare -A gpu_scenes

# 初始化 GPU 的任务
for gpu in "${GPUS[@]}"; do
    if [ $scene_index -lt ${#SCENES[@]} ]; then
        scene=${SCENES[$scene_index]}
        run_scene "$scene" $gpu &
        gpu_pids[$gpu]=$!
        gpu_scenes[$gpu]=$scene
        scene_index=$((scene_index + 1))
    fi
done

# 动态调度：当一个 GPU 跑完，立刻分下一个场景
while [ $scene_index -lt ${#SCENES[@]} ] || [ ${#gpu_pids[@]} -gt 0 ]; do
    for gpu in "${GPUS[@]}"; do
        if [ -n "${gpu_pids[$gpu]}" ]; then
            if ! kill -0 ${gpu_pids[$gpu]} 2>/dev/null; then
                wait ${gpu_pids[$gpu]}
                unset gpu_pids[$gpu]
                
                if [ $scene_index -lt ${#SCENES[@]} ]; then
                    scene=${SCENES[$scene_index]}
                    run_scene "$scene" $gpu &
                    gpu_pids[$gpu]=$!
                    gpu_scenes[$gpu]=$scene
                    scene_index=$((scene_index + 1))
                fi
            fi
        fi
    done
    sleep 1
done

echo "=========================================="
echo "所有场景已处理完成。"
echo "输出目录: $OUTPUT_DIR"
echo "=========================================="
