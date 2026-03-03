#!/bin/bash
set -euo pipefail

# 遍历 dl3dv_1k_eval 目录下的所有 JSON 文件
JSON_DIR="dl3dv_1k_eval"
MODEL_ROOT="output_DL3DV/1K"
OUTPUT_ROOT="rerun_output_dl3dv"
DATA_ROOT="DL3DV/1K"
INPUT_JSON_DIR="output_3d_bounding_dl3dv_1k"
LABEL_JSON_DIR="scene_objects_Qwen3-VL-30B-A3B-Instruct-DL3DV"

# 创建输出根目录
mkdir -p "${OUTPUT_ROOT}"

# 统计信息
total=0
success=0
failed=0

# 遍历所有 JSON 文件
for json_file in "${JSON_DIR}"/*.json; do
    if [ ! -f "$json_file" ]; then
        continue
    fi
    
    # 提取场景名称（去掉路径和扩展名）
    scene=$(basename "$json_file" .json)
    total=$((total + 1))
    
    echo "=========================================="
    echo "处理场景 $total: ${scene}"
    echo "=========================================="
    
    # 检查模型路径是否存在
    model_path="${MODEL_ROOT}/${scene}"
    if [ ! -d "$model_path" ]; then
        echo "⚠️  警告: 模型路径不存在: ${model_path}，跳过"
        failed=$((failed + 1))
        continue
    fi
    
    # 检查输入 JSON 文件（优先使用 dl3dv_1k_eval 下的，否则使用 output_3d_bounding_dl3dv_1k 下的）
    input_json="${json_file}"
    if [ ! -f "$input_json" ]; then
        input_json="${INPUT_JSON_DIR}/${scene}.json"
        if [ ! -f "$input_json" ]; then
            echo "⚠️  警告: 输入 JSON 文件不存在: ${input_json}，跳过"
            failed=$((failed + 1))
            continue
        fi
    fi
    
    # 设置输出目录
    output_dir="${OUTPUT_ROOT}/${scene}"
    
    # 运行 json2rerun.py
    echo "运行: python json2rerun.py ${input_json} -m ${model_path} --output-dir ${output_dir} --data-root ${DATA_ROOT}"
    if python json2rerun.py "${input_json}" -m "${model_path}" --output-dir "${output_dir}" --data-root "${DATA_ROOT}"; then
        echo "✓ 场景 ${scene} 处理成功"
        
        # 复制 JSON 文件到输出目录
        mkdir -p "${output_dir}"
        cp "${input_json}" "${output_dir}/${scene}.json"
        echo "✓ JSON 文件已复制到 ${output_dir}/${scene}.json"
        
        # 复制 label JSON 文件
        label_json="${LABEL_JSON_DIR}/${scene}.json"
        if [ -f "$label_json" ]; then
            cp "${label_json}" "${output_dir}/label.json"
            echo "✓ Label JSON 文件已复制到 ${output_dir}/label.json"
        else
            echo "⚠️  警告: Label JSON 文件不存在: ${label_json}"
        fi

        
        success=$((success + 1))
    else
        echo "✗ 场景 ${scene} 处理失败"
        failed=$((failed + 1))
    fi
    
    echo ""
done

echo "=========================================="
echo "处理完成"
echo "总计: ${total} 个场景"
echo "成功: ${success} 个"
echo "失败: ${failed} 个"
echo "=========================================="