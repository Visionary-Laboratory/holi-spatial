#!/bin/bash
# 运行 2d_iou_Sa2VA.py 处理 DL3DV 下的9个场景

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate Sa2VA

# 9个场景（从 mask_index_outputs_dl3dv 目录获取）
scenes=(
    "0bfab69b5b2b692d066b4435f2f6406357772aa53a3e17c3e0b8598083848a7a"
    "2d52e65f703cf307013273ffeea79b76db153d665294becb16f00477aac05a88"
    "5b1e56386e263f2324365d2c623d6b746280aa4f5a6257bb0ba66e5dff135c8a"
    "7ae35f524e63aa86d1985f1c59dfd7a29cbd6069770ee26506ac26705917d9ea"
    "7ed574ca0081fa9a8c879a4b360ec7b274b3a9d248c1a6e66bd1e1d1c47bea4b"
    "7f5f6eac88ce535147bb9a7740635901929af1b08fbf0dd1adf28bf8aeee380a"
    "b3d3847b167ee770b105de4388108a1b8f3bfaa430d4d6ff0ba910d271d8a487"
    "c07ac2cdc0dddc8fe07c42258725ca0a5b486dcaccaef4fcbe8f5311cdee46fc"
    "cd6c19a00fe93297c128769f81f7778f8c2b7a6ca7d1ca780b3f83803cdc2082"
)

# 输出目录
output_dir="2d_iou_sa2va_outputs_sa2va"
mkdir -p "$output_dir"

# 日志目录
log_dir="$output_dir/logs"
mkdir -p "$log_dir"

# 依次运行每个场景
for scene in "${scenes[@]}"; do
    echo "=========================================="
    echo "处理场景: $scene"
    echo "=========================================="
    
    log_file="$log_dir/${scene}.log"
    
    python 2d_iou_Sa2VA.py \
        --scene "$scene" \
        --data-root DL3DV/1K \
        --gt-json "mask_index_outputs_dl3dv/${scene}.json" \
        --output-dir "$output_dir" \
        2>&1 | tee "$log_file"
    
    echo "场景 $scene 处理完成，日志已保存到: $log_file"
    echo ""
done

echo "=========================================="
echo "所有场景处理完成！"
echo "=========================================="
echo "运行统计脚本查看结果:"
echo "python stat_2d_iou_sa2va_dl3dv.py"

