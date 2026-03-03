#!/usr/bin/env bash
# 并行处理多个 ScanNet 场景的 bbox overlay（与单场景 scene0029_00 相同参数）
# 用法: 在项目根目录执行 bash run_bbox_overlay_scannet_batch.sh

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

JSON_DIR="${JSON_DIR:-output_scannet_new_aabb}"
DATA_ROOT="${DATA_ROOT:-/mnt/shared-storage-user/intern7shared/liuyifei/scannetv2/scans}"
OUTPUT_DIR="${OUTPUT_DIR:-./bbox_overlay_scannet_batch_outputs_aabb}"
MODEL_PATH="${MODEL_PATH:-/home/liuyifei/code/posevlm/pgsr_result_scannetv2}"

EXCLUDE_LABELS="tile,tiles,shelf,floor,ceiling,drawe,can,lamp,wall,stool,paper,recycling bin"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"

echo "=============================================="
echo "并行处理 ${#SCENES[@]} 个场景"
echo "JSON_DIR   : ${JSON_DIR}"
echo "OUTPUT_DIR : ${OUTPUT_DIR}"
echo "MODEL_PATH : ${MODEL_PATH}"
echo "=============================================="

for scene in "${SCENES[@]}"; do
  json_path="${JSON_DIR}/${scene}.json"
  if [[ ! -f "$json_path" ]]; then
    echo "[跳过] JSON 不存在: $json_path"
    continue
  fi
  (
    echo "[${scene}] 开始..."
    python json2bbox_images_depth_scannet.py "$json_path" \
      --data-root "$DATA_ROOT" \
      --output-dir "$OUTPUT_DIR" \
      --render wire \
      --alpha 0.32 \
      --no-include-desc \
      --no-frustum-filter \
      --show-labels \
      -m "$MODEL_PATH" \
      --gpu-edge-visibility \
      --depth-eps 0.06 \
      --depth-eps-scale 0.012 \
      --edge-sample-density 4.0 \
      --downscale 1 \
      --thickness 3 \
      --exclude-labels "$EXCLUDE_LABELS"
    echo "[${scene}] 完成"
  ) &>"${LOG_DIR}/${scene}.log" &
done

wait
echo "=============================================="
echo "全部 ${#SCENES[@]} 个场景已处理完成"
echo "输出目录: ${OUTPUT_DIR}"
echo "日志目录: ${LOG_DIR}"
echo "=============================================="
