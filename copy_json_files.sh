#!/bin/bash

# 源目录
SOURCE_DIR="/mnt/shared-storage-user/intern7shared/liuyifei/code/posevlm/output_DL3DV/1K"

# 目标目录（如果未提供参数，使用当前目录下的copied_folders文件夹）
TARGET_DIR="${1:-$(pwd)/copied_folders}"

# 创建目标目录（如果不存在）
mkdir -p "$TARGET_DIR"

# 要拷贝的JSON文件列表（去掉扩展名即为文件夹名）
JSON_FILES=(
    "0bfab69b5b2b692d066b4435f2f6406357772aa53a3e17c3e0b8598083848a7a.json"
    "2d52e65f703cf307013273ffeea79b76db153d665294becb16f00477aac05a88.json"
    "5b1e56386e263f2324365d2c623d6b746280aa4f5a6257bb0ba66e5dff135c8a.json"
    "7ae35f524e63aa86d1985f1c59dfd7a29cbd6069770ee26506ac26705917d9ea.json"
    "7ed574ca0081fa9a8c879a4b360ec7b274b3a9d248c1a6e66bd1e1d1c47bea4b.json"
    "7f5f6eac88ce535147bb9a7740635901929af1b08fbf0dd1adf28bf8aeee380a.json"
    "b3d3847b167ee770b105de4388108a1b8f3bfaa430d4d6ff0ba910d271d8a487.jsonl"
    "c07ac2cdc0dddc8fe07c42258725ca0a5b486dcaccaef4fcbe8f5311cdee46fc.json"
    "cd6c19a00fe93297c128769f81f7778f8c2b7a6ca7d1ca780b3f83803cdc2082.json"
)

# 检查源目录是否存在
if [ ! -d "$SOURCE_DIR" ]; then
    echo "错误: 源目录不存在: $SOURCE_DIR"
    exit 1
fi

# 复制文件夹
SUCCESS_COUNT=0
FAIL_COUNT=0

echo "开始从 $SOURCE_DIR 复制文件夹到 $TARGET_DIR"
echo "----------------------------------------"

for json_file in "${JSON_FILES[@]}"; do
    # 去掉.json或.jsonl扩展名，得到文件夹名
    if [[ "$json_file" == *.jsonl ]]; then
        FOLDER_NAME="${json_file%.jsonl}"
    else
        FOLDER_NAME="${json_file%.json}"
    fi
    
    SOURCE_FOLDER="$SOURCE_DIR/$FOLDER_NAME"
    TARGET_FOLDER="$TARGET_DIR/$FOLDER_NAME"
    
    if [ -d "$SOURCE_FOLDER" ]; then
        cp -r "$SOURCE_FOLDER" "$TARGET_FOLDER"
        if [ $? -eq 0 ]; then
            echo "✓ 成功复制: $FOLDER_NAME/"
            ((SUCCESS_COUNT++))
        else
            echo "✗ 复制失败: $FOLDER_NAME/"
            ((FAIL_COUNT++))
        fi
    else
        echo "✗ 文件夹不存在: $FOLDER_NAME/"
        ((FAIL_COUNT++))
    fi
done

echo "----------------------------------------"
echo "完成! 成功: $SUCCESS_COUNT, 失败: $FAIL_COUNT"
echo "目标目录: $TARGET_DIR"

