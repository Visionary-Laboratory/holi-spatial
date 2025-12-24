import json
import numpy as np
import pycocotools.mask as mask_utils
from PIL import Image
from pathlib import Path
import argparse

def test_mask_decoding(json_path, output_dir="test_masks"):
    # 1. 加载 JSON 数据
    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    if not results:
        print("JSON 文件为空")
        return

    # 创建输出目录
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 2. 遍历实例并尝试解码第一个 mask
    for inst_idx, inst in enumerate(results):
        label = inst.get("label", "unknown")
        ins_id = inst.get("ins_id", str(inst_idx))
        mask_encodings = inst.get("mask_encodings", [])
        
        if not mask_encodings:
            print(f"实例 {ins_id} ({label}) 没有 mask 编码")
            continue

        # 取出第一个 mask 编码
        rle = mask_encodings[0]
        
        # 3. 解码为 numpy 数组 (0 或 1)
        # 注意：rle 必须包含 'size' 和 'counts' (如果是字符串，mask_utils 可以处理)
        try:
            mask = mask_utils.decode(rle)
            
            # 4. 转换为可视化的图片 (0 -> 0, 1 -> 255)
            mask_img = (mask * 255).astype(np.uint8)
            img = Image.fromarray(mask_img)
            
            # 保存
            save_name = f"mask_{label}_{ins_id}.png"
            img.save(out_path / save_name)
            print(f"成功保存: {out_path / save_name}")
            
            # 为了演示，只测试每个标签的前几个或整体前几个
            if inst_idx >= 5: # 限制一下，防止保存太多
                print("已达到测试数量限制，停止测试。")
                break
                
        except Exception as e:
            print(f"解码实例 {ins_id} 失败: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=str, help="JSON 结果文件的路径")
    parser.add_argument("--out", type=str, default="test_results", help="保存图片的目录")
    args = parser.parse_args()

    test_mask_decoding(args.json_path, args.out)