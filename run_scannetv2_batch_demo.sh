#!/usr/bin/env bash

# 按给定命令批量处理多个 ScanNetv2 场景：
# 1) 调用 2d_iou_gyn_seperate_instance_scene_label_demo.py 生成 JSON mask
# 2) 调用 visualize_seg.py 生成可视化分割图和图例

set -e

SCENES=(
  # "scene0000_00"
  # "scene0071_00"
  "scene0628_02"
  # "scene0659_01"
)

DATA_ROOT="scannetv2/scans"
RRD_ROOT="output_scannetv2"
MODEL_ROOT="pgsr_result_scannetv2"
SCENE_OBJS_ROOT="Qwen3VL-32B-Scannetv2"
OUT_ROOT="output_demo"

for SCENE in "${SCENES[@]}"; do
  echo "==== 处理场景: ${SCENE} ===="

  # 1) 先根据 rrd + 3DGS 生成该场景下所有图像的 2D mask JSON
  python 2d_iou_gyn_seperate_instance_scene_label_demo.py \
    --scene "${SCENE}" \
    --data-root "${DATA_ROOT}" \
    --rrd-path "${RRD_ROOT}/${SCENE}.rrd" \
    --model-path "${MODEL_ROOT}/${SCENE}" \
    --output-dir "${OUT_ROOT}/${SCENE}" \
    --scene-objects-json "${SCENE_OBJS_ROOT}/${SCENE}.json"

  # 2) 再对该场景的 JSON 做分割可视化
  python visualize_seg.py \
    "${DATA_ROOT}/${SCENE}/color" \
    "${OUT_ROOT}/${SCENE}/${SCENE}.json" \
    "${OUT_ROOT}/${SCENE}/masks"

  echo "==== 场景 ${SCENE} 处理完成 ===="
done

