#!/usr/bin/env bash
set -u
set -o pipefail

# 渲染 bbox 图片并合成为视频。支持两种模式：
#   1) 指定场景：-s 6f1848d1e3
#   2) 随机抽样：-n 5
#
# 示例：
#   bash run_random_scenes_aabb.sh -s 6f1848d1e3
#   bash run_random_scenes_aabb.sh -s 6f1848d1e3 --exclude-ids 123 456
#   bash run_random_scenes_aabb.sh -n 5 --seed 42
#   bash run_random_scenes_aabb.sh -n 10 --exclude-ids 12,34

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCENES=()
NUM_SCENES=""
SEED=""
EXCLUDE_IDS=()
JSON_DIR="${SCRIPT_DIR}/output_scannetppv2_new_aabb"
DATA_ROOT="/mnt/shared-storage-gpfs2/solution-gpfs02/liuyifei/scannnetppv2_0117/data_all"
OUTPUT_DIR="${SCRIPT_DIR}/bbox_overlay_test_outputs_aabb_gyy"
VIDEO_DIR="${OUTPUT_DIR}/videos"
MODEL_PATH="/mnt/shared-storage-gpfs2/solution-gpfs02/liuyifei/pgsr_scannetppv2_all"
FPS=12
MAX_JOBS=5

print_help() {
  cat <<'EOF'
用法:
  bash run_random_scenes_aabb.sh -s <场景名> [<场景名> ...] [可选参数]
  bash run_random_scenes_aabb.sh -n <数量> [可选参数]

参数（二选一）:
  -s, --scene <名称>      指定场景名，可多个。如: -s 6f1848d1e3
  -n, --num-scenes <int>  随机抽取的场景数。如: -n 5

可选参数:
      --exclude-ids <id>  要过滤掉的 ins_id，可多个或逗号分隔
      --seed <int>        随机种子（仅 -n 模式，用于复现）
      --json-dir <path>   场景 JSON 目录
      --data-root <path>  数据根目录
      --output-dir <path> 渲染图片输出目录（默认: ./bbox_overlay_test_outputs_aabb）
      --video-dir <path>  视频输出目录（默认: <output-dir>/videos）
      --model-path <path> 3DGS 模型目录（默认: ./pgsr_scannetppv2_all）
      --fps <int>         输出视频帧率（默认: 12）
      --jobs <int>        并行场景数（默认: 5）
  -h, --help              显示帮助
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--scene)
      shift
      while [[ $# -gt 0 ]] && [[ "$1" != --* ]] && [[ "$1" != -s ]] && [[ "$1" != -n ]] && [[ "$1" != -h ]]; do
        [[ -n "$1" ]] && SCENES+=("$1")
        shift
      done
      ;;
    -n|--num-scenes)
      NUM_SCENES="${2:-}"
      shift 2
      ;;
    --seed)
      SEED="${2:-}"
      shift 2
      ;;
    --exclude-ids)
      # 支持 --exclude-ids 123 456 或 --exclude-ids 123,456
      shift
      while [[ $# -gt 0 ]] && [[ "$1" != --* ]] && [[ "$1" != -s ]] && [[ "$1" != -n ]] && [[ "$1" != -h ]]; do
        # 拆开逗号分隔的值
        for part in $(echo "$1" | tr ',' ' '); do
          [[ -n "${part}" ]] && EXCLUDE_IDS+=("${part}")
        done
        shift
      done
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
    --jobs)
      MAX_JOBS="${2:-}"
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

if [[ ${#SCENES[@]} -gt 0 ]] && [[ -n "${NUM_SCENES}" ]]; then
  echo "错误: -s/--scene 与 -n/--num-scenes 不能同时指定，请二选一。"
  print_help
  exit 1
fi

if [[ ${#SCENES[@]} -eq 0 ]] && [[ -z "${NUM_SCENES}" ]]; then
  echo "错误: 请指定 -s/--scene（场景名）或 -n/--num-scenes（随机数量）之一。"
  print_help
  exit 1
fi

if [[ -n "${NUM_SCENES}" ]]; then
  if ! [[ "${NUM_SCENES}" =~ ^[0-9]+$ ]] || [[ "${NUM_SCENES}" -le 0 ]]; then
    echo "错误: --num-scenes 必须是正整数，当前值: ${NUM_SCENES}"
    exit 1
  fi
  if [[ -n "${SEED}" ]] && ! [[ "${SEED}" =~ ^[0-9]+$ ]]; then
    echo "错误: --seed 必须是非负整数，当前值: ${SEED}"
    exit 1
  fi
fi

if ! [[ "${FPS}" =~ ^[0-9]+$ ]] || [[ "${FPS}" -le 0 ]]; then
  echo "错误: --fps 必须是正整数，当前值: ${FPS}"
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

selected_jsons=()
if [[ ${#SCENES[@]} -gt 0 ]]; then
  # 指定场景模式
  for name in "${SCENES[@]}"; do
    json_path="${JSON_DIR}/${name}.json"
    if [[ -f "${json_path}" ]]; then
      selected_jsons+=("${json_path}")
    else
      echo "警告: 未找到场景 JSON: ${json_path}，跳过"
    fi
  done
else
  # 随机抽样模式
  total="${#json_files[@]}"
  if [[ "${total}" -eq 0 ]]; then
    echo "错误: 未在 ${JSON_DIR} 找到任何 .json 文件。"
    exit 1
  fi
  if [[ "${NUM_SCENES}" -gt "${total}" ]]; then
    echo "警告: 请求抽样 ${NUM_SCENES} 个场景，但仅有 ${total} 个，自动改为 ${total}。"
    NUM_SCENES="${total}"
  fi
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
fi

if [[ ${#selected_jsons[@]} -eq 0 ]]; then
  echo "错误: 没有找到任何有效场景的 JSON 文件。"
  exit 1
fi

echo "=============================================="
if [[ ${#SCENES[@]} -gt 0 ]]; then
  echo "指定场景批处理开始"
else
  echo "随机场景批处理开始"
fi
echo "JSON_DIR   : ${JSON_DIR}"
echo "DATA_ROOT  : ${DATA_ROOT}"
echo "OUTPUT_DIR : ${OUTPUT_DIR}"
echo "VIDEO_DIR  : ${VIDEO_DIR}"
echo "MODEL_PATH : ${MODEL_PATH}"
echo "FPS        : ${FPS}"
echo "场景数量   : ${#selected_jsons[@]}"
if [[ -n "${NUM_SCENES}" ]] && [[ -n "${SEED}" ]]; then
  echo "SEED       : ${SEED}"
fi
if [[ ${#EXCLUDE_IDS[@]} -gt 0 ]]; then
  echo "排除 ID    : ${EXCLUDE_IDS[*]}"
fi
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

  exclude_args=()
  if [[ ${#EXCLUDE_IDS[@]} -gt 0 ]]; then
    exclude_args=(--exclude-ids "${EXCLUDE_IDS[@]}")
  fi

  python3 "${SCRIPT_DIR}/json2bbox_images_depth.py" "${scene_json}" \
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
    --exclude-labels tile,tiles,shelf,floor,ceiling,drawer,light \
    "${exclude_args[@]}"
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
echo "处理结束：成功 ${ok_count}，失败 ${fail_count}，总计 ${#selected_jsons[@]}"
echo "图片目录：${OUTPUT_DIR}"
echo "视频目录：${VIDEO_DIR}"
echo "=============================================="
