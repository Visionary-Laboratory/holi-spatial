#!/bin/bash
# 运行 3d_bounding_instance_gs_rerun.py 处理多个 scannetppv2 场景

scenes=(
    "027cd6ea0f"
    "09d6e808b4"
    "0a7cc12c0e"
    "0d8ead0038"
    "116456116b"
    "17a5e7d36c"
    "1cefb55d50"
    "20871b98f3"
    "924b364b9f"
)

for scene in "${scenes[@]}"; do
    echo "=========================================="
    echo "处理场景: $scene"
    echo "=========================================="
    
    python 3d_bounding_instance_gs_rerun.py \
        --scene "$scene" \
        --data-root scannetppv2/data/ \
        --mask-root /mnt/shared-storage-user/solution/liuyifei/code/posevlm/sam_masks_debug_scannet_old/ \
        -m "output_scannet/scannetppv2/${scene}/" \
        --output-dir output_3d_bounding_scannet_evaluation_new \
        --vllm-api-url http://100.102.160.232:8000/v1/chat/completions
    
    echo "场景 $scene 处理完成"
    echo ""
done

echo "=========================================="
echo "所有场景处理完成！"
echo "=========================================="

