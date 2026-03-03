#!/bin/bash

# 批量处理 pgsr_scannetppv2_all 下所有场景的脚本
# 如果场景的 mesh/tsdf_fusion_post.ply 已存在，则跳过

echo "开始批量处理 pgsr_scannetppv2_all 下的所有场景..."

# 检查 pgsr_scannetppv2_all 目录是否存在
if [ ! -d "pgsr_scannetppv2_all" ]; then
    echo "错误: pgsr_scannetppv2_all 目录不存在"
    exit 1
fi

# 统计变量
total=0
skipped=0
processed=0
failed=0

# 遍历 pgsr_scannetppv2_all 下的所有场景目录
for scene_dir in pgsr_scannetppv2_all/*/; do
    # 获取场景名称（去掉路径和尾部斜杠）
    scene=$(basename "$scene_dir")
    
    # 检查是否是目录
    if [ ! -d "$scene_dir" ]; then
        continue
    fi
    
    total=$((total + 1))
    
    # 检查数据目录是否存在
    data_dir="scannetppv2/data/${scene}/dslr/nerfstudio"
    image_dir="scannetppv2/data/${scene}/dslr/resized_undistorted_images"
    
    if [ ! -d "$data_dir" ]; then
        echo "=========================================="
        echo "警告: 场景 $scene 的数据目录不存在: $data_dir"
        echo "跳过该场景"
        echo "=========================================="
        skipped=$((skipped + 1))
        continue
    fi
    
    if [ ! -d "$image_dir" ]; then
        echo "=========================================="
        echo "警告: 场景 $scene 的图像目录不存在: $image_dir"
        echo "跳过该场景"
        echo "=========================================="
        skipped=$((skipped + 1))
        continue
    fi
    
    # 处理场景
    echo "=========================================="
    echo "处理场景: $scene"
    echo "=========================================="
    
    # 检查 mesh 文件是否已存在
    mesh_file="${scene_dir}mesh/tsdf_fusion_post.ply"
    mesh_exists=false
    
    if [ -f "$mesh_file" ]; then
        echo "mesh 文件已存在，跳过 render.py 步骤"
        mesh_exists=true
    else
        # 执行 render.py 生成 mesh
        python PGSR/render.py \
            -s "$data_dir" \
            -m "pgsr_scannetppv2_all/${scene}" \
            --skip_test \
            -i "$image_dir"
        
        if [ $? -ne 0 ]; then
            echo "错误: 场景 $scene render.py 处理失败"
            failed=$((failed + 1))
            echo ""
            continue
        fi
        
        # 检查 mesh 文件是否生成
        if [ ! -f "$mesh_file" ]; then
            echo "警告: 场景 $scene mesh 文件未生成，跳过 mesh2mask 处理"
            failed=$((failed + 1))
            echo ""
            continue
        fi
    fi
    
    # 检查 mask 目录是否已存在
    mask_dir="scannetppv2/data/${scene}/mask"
    if [ -d "$mask_dir" ]; then
        echo "mask 目录已存在，跳过 mesh2mask.py 步骤"
        echo "场景 $scene 所有处理完成"
        processed=$((processed + 1))
    else
        # 执行 mesh2mask.py 命令
        echo "执行 mesh2mask.py..."
        python PGSR/mesh2mask.py \
            -m "pgsr_scannetppv2_all/${scene}" \
            -s "${data_dir}/" \
            --mesh_path "mesh/tsdf_fusion_post.ply" \
            -i "$image_dir"
        
        if [ $? -ne 0 ]; then
            echo "警告: 场景 $scene mesh2mask.py 处理失败"
            failed=$((failed + 1))
        else
            echo "场景 $scene 所有处理完成"
            processed=$((processed + 1))
        fi
    fi
    
    echo ""
done

# 输出统计信息
echo "=========================================="
echo "处理完成！"
echo "总场景数: $total"
echo "已处理: $processed"
echo "已跳过: $skipped"
echo "失败: $failed"
echo "=========================================="

