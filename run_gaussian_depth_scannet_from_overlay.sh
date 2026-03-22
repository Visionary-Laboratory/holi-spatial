#!/usr/bin/env bash
# 基于 bbox_overlay_scannet_batch_outputs_aabb 中出现的所有 scene，
# 批量用 3D 高斯渲染深度并保存为彩色深度图。
#
# 用法：在项目根目录执行
#   bash run_gaussian_depth_scannet_from_overlay.sh
#
# 可通过环境变量覆盖：
#   DATA_ROOT   - ScanNet scans 根目录
#   MODEL_PATH  - 3DGS 模型根目录（多场景）
#   OVERLAY_DIR - 已有 bbox overlay 输出根目录
#   OUTPUT_DIR  - 深度图输出根目录

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATA_ROOT="${DATA_ROOT:-/mnt/shared-storage-user/intern7shared/liuyifei/scannetv2/scans}"
MODEL_PATH="${MODEL_PATH:-/home/liuyifei/code/posevlm/pgsr_result_scannetv2}"
OVERLAY_DIR="${OVERLAY_DIR:-./bbox_overlay_scannet_batch_outputs_aabb}"
OUTPUT_DIR="${OUTPUT_DIR:-./gaussian_depth_from_3dgs}"
VIDEO_DIR="${VIDEO_DIR:-${OUTPUT_DIR}/videos}"
FPS="${FPS:-20}"

LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR" "$VIDEO_DIR"

SCENES=()
for d in "${OVERLAY_DIR}"/scene*; do
  if [[ -d "$d" ]]; then
    SCENES+=("$(basename "$d")")
  fi
done

if [[ ${#SCENES[@]} -eq 0 ]]; then
  echo "在 ${OVERLAY_DIR} 下没有找到任何 scene*/ 子目录，检查路径是否正确。"
  exit 1
fi

echo "=============================================="
echo "将对 ${#SCENES[@]} 个场景渲染高斯深度"
echo "DATA_ROOT   : ${DATA_ROOT}"
echo "MODEL_PATH  : ${MODEL_PATH}"
echo "OVERLAY_DIR : ${OVERLAY_DIR}"
echo "OUTPUT_DIR  : ${OUTPUT_DIR}"
echo "=============================================="

for scene in "${SCENES[@]}"; do
  echo "[${scene}] 开始深度渲染..."
  python render_gaussian_depth_scannet.py "${scene}" \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    -m "${MODEL_PATH}" \
    &>"${LOG_DIR}/${scene}.log"
  echo "[${scene}] 深度渲染完成，开始合成视频..."

  scene_img_dir="${OUTPUT_DIR}/${scene}"
  video_path="${VIDEO_DIR}/${scene}.mp4"
  if [[ ! -d "$scene_img_dir" ]]; then
    echo "[${scene}] 跳过视频合成，目录不存在: ${scene_img_dir}"
  else
    python imgseq_to_video.py \
      --input_dir "$scene_img_dir" \
      --output "$video_path" \
      --fps "$FPS"
    echo "[${scene}] 视频完成: ${video_path}"
  fi

done

echo "=============================================="
echo "全部 ${#SCENES[@]} 个场景已完成深度渲染与视频合成"
echo "深度输出目录 : ${OUTPUT_DIR}"
echo "视频输出目录 : ${VIDEO_DIR}"
echo "日志目录     : ${LOG_DIR}"
echo "=============================================="

