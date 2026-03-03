#!/bin/bash
# 统计 pgsr_scannetppv2_all 下有 mesh/tsdf_fusion_post.ply 的场景个数
count=0
for d in pgsr_scannetppv2_all/*/; do
  if [ -f "${d}mesh/tsdf_fusion_post.ply" ]; then
    count=$((count + 1))
  fi
done
echo "有 mesh 的场景个数: $count"
