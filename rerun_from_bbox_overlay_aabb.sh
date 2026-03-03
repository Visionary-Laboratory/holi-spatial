#!/usr/bin/env bash
set -euo pipefail

# 基于 bbox_overlay_test_outputs_aabb 目录里的场景，逐个调用 json2rerun.py。
#
# 示例：
#   bash rerun_from_bbox_overlay_aabb.sh
#   bash rerun_from_bbox_overlay_aabb.sh --scene-dir /path/to/bbox_overlay_test_outputs_aabb
#   bash rerun_from_bbox_overlay_aabb.sh --json-dir /path/to/jsons --output-root ./rerun_output_scannetppv2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCENE_DIR="${SCRIPT_DIR}/bbox_overlay_test_outputs_aabb"
JSON_DIR="${SCRIPT_DIR}/output_scannetppv2_new_aabb"
MODEL_ROOT="${SCRIPT_DIR}/pgsr_scannetppv2_all"
OUTPUT_ROOT="${SCRIPT_DIR}/rerun_output_scannetppv2"
DATA_ROOT="/mnt/shared-storage-user/solution-gpfs02/liuyifei/scannnetppv2_0117/data_all"
LABEL_JSON_DIR=""

print_help() {
  cat <<'EOF'
用法:
  bash rerun_from_bbox_overlay_aabb.sh [可选参数]

可选参数:
  --scene-dir <path>      场景目录（默认: ./bbox_overlay_test_outputs_aabb）
  --json-dir <path>       输入 JSON 目录（默认: ./output_scannetppv2_new_aabb）
  --model-root <path>     模型根目录（默认: ./pgsr_scannetppv2_all）
  --output-root <path>    输出根目录（默认: ./rerun_output_scannetppv2）
  --data-root <path>      数据根目录
  --label-json-dir <path> Label JSON 目录（可选，若提供会复制到输出）
  -h, --help              显示帮助
EOF
}

require_value() {
  local opt="$1"
  local val="${2:-}"
  if [[ -z "${val}" ]] || [[ "${val}" == --* ]]; then
    echo "错误: 选项 ${opt} 缺少参数值"
    print_help
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scene-dir)
      require_value "$1" "${2:-}"
      SCENE_DIR="${2:-}"
      shift 2
      ;;
    --scene-dir=*)
      SCENE_DIR="${1#*=}"
      shift 1
      ;;
    --json-dir)
      require_value "$1" "${2:-}"
      JSON_DIR="${2:-}"
      shift 2
      ;;
    --json-dir=*)
      JSON_DIR="${1#*=}"
      shift 1
      ;;
    --model-root)
      require_value "$1" "${2:-}"
      MODEL_ROOT="${2:-}"
      shift 2
      ;;
    --model-root=*)
      MODEL_ROOT="${1#*=}"
      shift 1
      ;;
    --output-root)
      require_value "$1" "${2:-}"
      OUTPUT_ROOT="${2:-}"
      shift 2
      ;;
    --output-root=*)
      OUTPUT_ROOT="${1#*=}"
      shift 1
      ;;
    --data-root)
      require_value "$1" "${2:-}"
      DATA_ROOT="${2:-}"
      shift 2
      ;;
    --data-root=*)
      DATA_ROOT="${1#*=}"
      shift 1
      ;;
    --label-json-dir)
      require_value "$1" "${2:-}"
      LABEL_JSON_DIR="${2:-}"
      shift 2
      ;;
    --label-json-dir=*)
      LABEL_JSON_DIR="${1#*=}"
      shift 1
      ;;
    -h|--help)
      print_help
      exit 0
      ;;
    *)
      echo "未知参数: $1"
      print_help
      exit 1
      ;;
  esac
done

if [[ ! -d "${SCENE_DIR}" ]]; then
  echo "错误: 场景目录不存在: ${SCENE_DIR}"
  exit 1
fi
if [[ ! -d "${JSON_DIR}" ]]; then
  echo "错误: JSON 目录不存在: ${JSON_DIR}"
  exit 1
fi
if [[ ! -d "${MODEL_ROOT}" ]]; then
  echo "错误: 模型根目录不存在: ${MODEL_ROOT}"
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

shopt -s nullglob
scene_dirs=( "${SCENE_DIR}"/* )
shopt -u nullglob

total=0
success=0
failed=0

for scene_path in "${scene_dirs[@]}"; do
  if [[ ! -d "${scene_path}" ]]; then
    continue
  fi

  scene="$(basename "${scene_path}")"
  if [[ "${scene}" == "videos" ]]; then
    continue
  fi

  total=$((total + 1))
  echo "=========================================="
  echo "处理场景 $total: ${scene}"
  echo "=========================================="

  input_json="${JSON_DIR}/${scene}.json"
  if [[ ! -f "${input_json}" ]]; then
    echo "⚠️  警告: 输入 JSON 文件不存在: ${input_json}，跳过"
    failed=$((failed + 1))
    continue
  fi

  model_path="${MODEL_ROOT}/${scene}"
  if [[ ! -d "${model_path}" ]]; then
    echo "⚠️  警告: 模型路径不存在: ${model_path}，跳过"
    failed=$((failed + 1))
    continue
  fi

  output_dir="${OUTPUT_ROOT}/${scene}"
  echo "运行: python json2rerun.py ${input_json} -m ${model_path} --output-dir ${output_dir} --data-root ${DATA_ROOT}"
  if python json2rerun.py "${input_json}" -m "${model_path}" --output-dir "${output_dir}" --data-root "${DATA_ROOT}"; then
    echo "✓ 场景 ${scene} 处理成功"

    mkdir -p "${output_dir}"
    cp "${input_json}" "${output_dir}/${scene}.json"

    if [[ -n "${LABEL_JSON_DIR}" ]]; then
      label_json="${LABEL_JSON_DIR}/${scene}.json"
      if [[ -f "${label_json}" ]]; then
        cp "${label_json}" "${output_dir}/label.json"
      else
        echo "⚠️  警告: Label JSON 文件不存在: ${label_json}"
      fi
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
echo "输出目录: ${OUTPUT_ROOT}"
echo "=========================================="
