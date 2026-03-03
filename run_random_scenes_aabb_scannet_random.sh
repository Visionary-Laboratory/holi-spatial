#!/usr/bin/env bash
set -u
set -o pipefail

# 随机抽样 N 个场景，先渲染 bbox 图片，再合成为视频。
#
# 示例：
#   bash run_random_scenes_aabb.sh -n 5
#   bash run_random_scenes_aabb.sh -n 10 --seed 42
#   bash run_random_scenes_aabb.sh -n 3 --json-dir /path/to/jsons --output-dir ./bbox_overlay_test_outputs_aabb

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NUM_SCENES=""
JSON_DIR="${SCRIPT_DIR}/output_scannet_new_aabb"
DATA_ROOT="/mnt/shared-storage-user/intern7shared/liuyifei/scannetv2/scans"
OUTPUT_DIR="${SCRIPT_DIR}/bbox_overlay_test_outputs_aabb_scannet_yyg"
VIDEO_DIR="${OUTPUT_DIR}/videos"
MODEL_PATH="/home/liuyifei/code/posevlm/pgsr_result_scannetv2"
EXCLUDE_LABELS="tile,tiles,shelf,floor,ceiling,drawe,can,lamp,wall,stool,light"
FPS=6
SEED=""

print_help() {
  cat <<'EOF'
用法:
  bash run_random_scenes_aabb.sh -n <场景数量> [可选参数]

可选参数:
  -n, --num-scenes <int>   随机抽取的场景数（必填）
      --json-dir <path>    场景 JSON 目录（默认: ./output_scannetppv2_new_aabb）
      --data-root <path>   数据根目录
      --output-dir <path>  渲染图片输出目录（默认: ./bbox_overlay_test_outputs_aabb）
      --video-dir <path>   视频输出目录（默认: <output-dir>/videos）
      --model-path <path>  3DGS 模型目录（默认: ./pgsr_scannetppv2_all）
      --fps <int>          输出视频帧率（默认: 12）
      --seed <int>         随机种子（给定后可复现）
  -h, --help               显示帮助
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--num-scenes)
      NUM_SCENES="${2:-}"
      shift 2
      ;;
    --json-dir)
      JSON_DIR="${2:-}"
      shift 2
      ;;
    --data-root)
      DATA_ROOT="${2:-}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    --video-dir)
      VIDEO_DIR="${2:-}"
      shift 2
      ;;
    --model-path)
      MODEL_PATH="${2:-}"
      shift 2
      ;;
    --fps)
      FPS="${2:-}"
      shift 2
      ;;
    --seed)
      SEED="${2:-}"
      shift 2
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

if [[ -z "${NUM_SCENES}" ]]; then
  echo "错误: 请通过 -n/--num-scenes 指定要抽样的场景数。"
  print_help
  exit 1
fi

if ! [[ "${NUM_SCENES}" =~ ^[0-9]+$ ]] || [[ "${NUM_SCENES}" -le 0 ]]; then
  echo "错误: --num-scenes 必须是正整数，当前值: ${NUM_SCENES}"
  exit 1
fi

if ! [[ "${FPS}" =~ ^[0-9]+$ ]] || [[ "${FPS}" -le 0 ]]; then
  echo "错误: --fps 必须是正整数，当前值: ${FPS}"
  exit 1
fi

if [[ -n "${SEED}" ]] && ! [[ "${SEED}" =~ ^[0-9]+$ ]]; then
  echo "错误: --seed 必须是非负整数，当前值: ${SEED}"
  exit 1
fi

if [[ ! -d "${JSON_DIR}" ]]; then
  echo "错误: JSON 目录不存在: ${JSON_DIR}"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}" "${VIDEO_DIR}"

shopt -s nullglob
json_files=( "${JSON_DIR}"/*.json )
shopt -u nullglob

total="${#json_files[@]}"
if [[ "${total}" -eq 0 ]]; then
  echo "错误: 未在 ${JSON_DIR} 找到任何 .json 文件。"
  exit 1
fi

if [[ "${NUM_SCENES}" -gt "${total}" ]]; then
  echo "警告: 请求抽样 ${NUM_SCENES} 个场景，但仅有 ${total} 个，自动改为 ${total}。"
  NUM_SCENES="${total}"
fi

selected_jsons=()
if [[ -n "${SEED}" ]]; then
  mapfile -t selected_jsons < <(
    python3 - "${NUM_SCENES}" "${SEED}" "${json_files[@]}" <<'PY'
import random
import sys

k = int(sys.argv[1])
seed = int(sys.argv[2])
items = sys.argv[3:]

rng = random.Random(seed)
for p in rng.sample(items, k):
    print(p)
PY
  )
else
  mapfile -t selected_jsons < <(printf '%s\n' "${json_files[@]}" | shuf -n "${NUM_SCENES}")
fi

echo "=============================================="
echo "随机场景批处理开始"
echo "JSON_DIR   : ${JSON_DIR}"
echo "DATA_ROOT  : ${DATA_ROOT}"
echo "OUTPUT_DIR : ${OUTPUT_DIR}"
echo "VIDEO_DIR  : ${VIDEO_DIR}"
echo "MODEL_PATH : ${MODEL_PATH}"
echo "FPS        : ${FPS}"
if [[ -n "${SEED}" ]]; then
  echo "SEED       : ${SEED}"
fi
echo "抽样数量   : ${NUM_SCENES}/${total}"
echo "=============================================="

ok_count=0
fail_count=0

for scene_json in "${selected_jsons[@]}"; do
  scene_name="$(basename "${scene_json}" .json)"
  scene_img_dir="${OUTPUT_DIR}/${scene_name}"
  video_path="${VIDEO_DIR}/out_${scene_name}.mp4"

  echo
  echo ">>> [场景] ${scene_name}"
  echo "    JSON: ${scene_json}"

  python3 "${SCRIPT_DIR}/json2bbox_images_depth_scannet.py" "${scene_json}" \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --render wire \
    --alpha 0.32 \
    --no-include-desc \
    --no-frustum-filter \
    --show-labels \
    -m "${MODEL_PATH}" \
    --gpu-edge-visibility \
    --depth-eps 0.06 \
    --depth-eps-scale 0.012 \
    --edge-sample-density 4.0 \
    --downscale 1 \
    --thickness 3 \
    --exclude-labels "$EXCLUDE_LABELS"
  render_exit=$?

  if [[ "${render_exit}" -ne 0 ]]; then
    echo "    [失败] 渲染退出码: ${render_exit}"
    fail_count=$((fail_count + 1))
    continue
  fi

  if [[ ! -d "${scene_img_dir}" ]]; then
    echo "    [失败] 渲染目录不存在: ${scene_img_dir}"
    fail_count=$((fail_count + 1))
    continue
  fi

  python3 "${SCRIPT_DIR}/imgseq_to_video.py" \
    --input_dir "${scene_img_dir}" \
    --output "${video_path}" \
    --fps "${FPS}"
  video_exit=$?

  if [[ "${video_exit}" -ne 0 ]]; then
    echo "    [失败] 合成视频退出码: ${video_exit}"
    fail_count=$((fail_count + 1))
    continue
  fi

  echo "    [完成] ${video_path}"
  ok_count=$((ok_count + 1))
done

echo
echo "=============================================="
echo "处理结束：成功 ${ok_count}，失败 ${fail_count}，总计 ${NUM_SCENES}"
echo "图片目录：${OUTPUT_DIR}"
echo "视频目录：${VIDEO_DIR}"
echo "=============================================="
