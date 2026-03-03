#!/usr/bin/env python3
"""
统计 2d_iou_scannet 和 2d_iou_scannet_sam3 目录下五个场景的平均 IoU
"""

import json
from pathlib import Path
from typing import Dict, List

def load_statistics(json_path: Path) -> Dict:
    """加载 JSON 文件的统计信息"""
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("statistics", {})

def process_directory(base_dir: Path, dir_name: str):
    """处理一个目录，统计五个场景的平均 IoU"""
    # 五个场景
    scenes = [
        "scene0124_00",
        "scene0181_02",
        "scene0204_02",
        "scene0347_02",
        "scene0520_00",
    ]
    
    all_stats = []
    
    print(f"\n{'='*60}")
    print(f"统计目录: {dir_name}")
    print(f"{'='*60}\n")
    
    for scene in scenes:
        json_path = base_dir / f"{scene}_2d_iou.json"
        if not json_path.exists():
            print(f"警告: {json_path} 不存在，跳过")
            continue
        
        stats = load_statistics(json_path)
        all_stats.append(stats)
        print(f"{scene}:")
        print(f"  overall_mean_iou: {stats.get('overall_mean_iou', 0):.6f}")
        print(f"  per_image_mean_iou: {stats.get('per_image_mean_iou', 0):.6f}")
        print(f"  per_label_mean_iou: {stats.get('per_label_mean_iou', 0):.6f}")
        print()
    
    if not all_stats:
        print("没有找到任何统计文件")
        return None
    
    # 计算平均值
    overall_mean_ious = [s.get("overall_mean_iou", 0) for s in all_stats]
    per_image_mean_ious = [s.get("per_image_mean_iou", 0) for s in all_stats]
    per_label_mean_ious = [s.get("per_label_mean_iou", 0) for s in all_stats]
    
    avg_overall_mean_iou = sum(overall_mean_ious) / len(overall_mean_ious)
    avg_per_image_mean_iou = sum(per_image_mean_ious) / len(per_image_mean_ious)
    avg_per_label_mean_iou = sum(per_label_mean_ious) / len(per_label_mean_ious)
    
    print("=" * 60)
    print(f"{dir_name} - 五个场景的平均 IoU:")
    print(f"  overall_mean_iou: {avg_overall_mean_iou:.6f}")
    print(f"  per_image_mean_iou: {avg_per_image_mean_iou:.6f}")
    print(f"  per_label_mean_iou: {avg_per_label_mean_iou:.6f}")
    print("=" * 60)
    
    return {
        "overall_mean_iou": avg_overall_mean_iou,
        "per_image_mean_iou": avg_per_image_mean_iou,
        "per_label_mean_iou": avg_per_label_mean_iou,
    }

def main():
    # 统计两个目录
    dirs = [
        ("2d_iou_scannet", Path("2d_iou_scannet")),
        ("2d_iou_scannet_sam3", Path("2d_iou_scannet_sam3")),
    ]
    
    results = {}
    for dir_name, base_dir in dirs:
        result = process_directory(base_dir, dir_name)
        if result:
            results[dir_name] = result
    
    # 打印总结
    if results:
        print(f"\n{'='*60}")
        print("总结对比:")
        print(f"{'='*60}")
        for dir_name, stats in results.items():
            print(f"\n{dir_name}:")
            print(f"  overall_mean_iou: {stats['overall_mean_iou']:.6f}")
            print(f"  per_image_mean_iou: {stats['per_image_mean_iou']:.6f}")
            print(f"  per_label_mean_iou: {stats['per_label_mean_iou']:.6f}")

if __name__ == "__main__":
    main()

