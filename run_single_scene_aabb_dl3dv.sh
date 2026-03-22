#!/usr/bin/env bash
set -u
set -o pipefail

# 处理单个 DL3DV 场景：渲染 bbox 图片 + 合成视频
#
# 示例：
#   bash run_single_scene_aabb_dl3dv.sh --scene-id 0a6c01ac3212768772f8f6eca86314c72d5ca320c3e3def148ddaceab23c07f4
#   bash run_single_scene_aabb_dl3dv.sh --scene-id <SCENE_ID> --fps 8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCENE_ID=""
JSON_DIR="${SCRIPT_DIR}/output_DL3DV_new/1K"
DATA_ROOT="${SCRIPT_DIR}/DL3DV/1K"
OUTPUT_DIR="${SCRIPT_DIR}/bbox_overlay_test_outputs_aabb_dl3dv_gyy"
VIDEO_DIR="${OUTPUT_DIR}/videos"
MODEL_PATH="/mnt/shared-storage-gpfs2/solution-gpfs02/liuyifei/pgsr_DL3DV_all/1K"
FPS=6

print_help() {
  cat <<'EOF'
用法:
  bash run_single_scene_aabb_dl3dv.sh --scene-id <场景ID> [可选参数]

必选参数:
      --scene-id <str>       场景 ID（例如：0a6c01ac3...07f4）

可选参数:
      --json-dir <path>      场景 JSON 目录（默认: output_DL3DV_new/1K）
      --data-root <path>     数据根目录（默认: DL3DV/1K）
      --output-dir <path>    渲染图片输出目录（默认: ./bbox_overlay_test_outputs_aabb_dl3dv_gyy）
      --video-dir <path>     视频输出目录（默认: <output-dir>/videos）
      --model-path <path>    3DGS 模型目录（默认: pgsr_DL3DV_all/1K）
      --fps <int>            输出视频帧率（默认: 6）
  -h, --help                 显示帮助
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scene-id)
      SCENE_ID="${2:-}"
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

if [[ -z "${SCENE_ID}" ]]; then
  echo "错误: 请通过 --scene-id 指定场景 ID。"
  print_help
  exit 1
fi

if ! [[ "${FPS}" =~ ^[0-9]+$ ]] || [[ "${FPS}" -le 0 ]]; then
  echo "错误: --fps 必须是正整数，当前值: ${FPS}"
  exit 1
fi

# JSON_DIR 可为相对路径（相对当前目录）或绝对路径
if [[ "${JSON_DIR}" != /* ]]; then
  JSON_DIR="${SCRIPT_DIR}/${JSON_DIR}"
fi
if [[ ! -d "${JSON_DIR}" ]]; then
  echo "错误: JSON 目录不存在: ${JSON_DIR}"
  exit 1
fi

if [[ "${DATA_ROOT}" != /* ]]; then
  DATA_ROOT="${SCRIPT_DIR}/${DATA_ROOT}"
fi

mkdir -p "${OUTPUT_DIR}" "${VIDEO_DIR}"

scene_json="${JSON_DIR}/${SCENE_ID}.json"
if [[ ! -f "${scene_json}" ]]; then
  echo "错误: 找不到场景 JSON 文件: ${scene_json}"
  exit 1
fi

scene_img_dir="${OUTPUT_DIR}/${SCENE_ID}"
video_path="${VIDEO_DIR}/out_${SCENE_ID}.mp4"

echo "=============================================="
echo "DL3DV 单场景处理开始"
echo "SCENE_ID   : ${SCENE_ID}"
echo "JSON       : ${scene_json}"
echo "JSON_DIR   : ${JSON_DIR}"
echo "DATA_ROOT  : ${DATA_ROOT}"
echo "OUTPUT_DIR : ${OUTPUT_DIR}"
echo "VIDEO_DIR  : ${VIDEO_DIR}"
echo "MODEL_PATH : ${MODEL_PATH}"
echo "FPS        : ${FPS}"
echo "=============================================="

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
  --exclude-labels tile,tiles,sign,notice,ceiling,wall,floor,ground,light,column,hallway,surveillance sign
render_exit=$?

if [[ "${render_exit}" -ne 0 ]]; then
  echo "[失败] 渲染退出码: ${render_exit}"
  exit "${render_exit}"
fi

if [[ ! -d "${scene_img_dir}" ]]; then
  echo "[失败] 渲染目录不存在: ${scene_img_dir}"
  exit 1
fi

python3 "${SCRIPT_DIR}/imgseq_to_video.py" \
  --input_dir "${scene_img_dir}" \
  --output "${video_path}" \
  --fps "${FPS}"
video_exit=$?

if [[ "${video_exit}" -ne 0 ]]; then
  echo "[失败] 合成视频退出码: ${video_exit}"
  exit "${video_exit}"
fi

echo "[完成] ${video_path}"
echo "图片目录：${scene_img_dir}"
echo "视频文件：${video_path}"

