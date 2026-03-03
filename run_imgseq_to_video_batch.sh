#!/usr/bin/env bash
# 对已渲染的 bbox overlay 图片目录并行合成视频（imgseq_to_video）
# 用法: 在项目根目录执行 bash run_imgseq_to_video_batch.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SCENES=(
  scene0009_00
  scene0052_02
  scene0087_00
  scene0109_01
  scene0118_01
  scene0174_00
  scene0198_00
)

INPUT_BASE="${INPUT_BASE:-./bbox_overlay_scannet_batch_outputs_aabb}"
VIDEO_DIR="${VIDEO_DIR:-./bbox_overlay_scannet_batch_outputs_aabb/videos}"
FPS="${FPS:-20}"

mkdir -p "$VIDEO_DIR"

echo "=============================================="
echo "并行合成 ${#SCENES[@]} 个场景视频"
echo "INPUT_BASE : ${INPUT_BASE}"
echo "VIDEO_DIR  : ${VIDEO_DIR}"
echo "FPS        : ${FPS}"
echo "=============================================="

for scene in "${SCENES[@]}"; do
  scene_img_dir="${INPUT_BASE}/${scene}"
  video_path="${VIDEO_DIR}/${scene}.mp4"
  if [[ ! -d "$scene_img_dir" ]]; then
    echo "[跳过] 目录不存在: $scene_img_dir"
    continue
  fi
  (
    echo "[${scene}] 合成视频..."
    python imgseq_to_video.py \
      --input_dir "$scene_img_dir" \
      --output "$video_path" \
      --fps "$FPS"
    echo "[${scene}] 完成: $video_path"
  ) &
done

wait
echo "=============================================="
echo "全部完成，视频目录: ${VIDEO_DIR}"
echo "=============================================="
