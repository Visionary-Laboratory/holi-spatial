#!/usr/bin/env python3
"""
统计 2d_iou_sa2va_outputs_scannet 目录下五个场景的平均 IoU
"""

import json
from pathlib import Path
from typing import Dict, List

def load_statistics(json_path: Path) -> Dict:
    """加载 JSON 文件的统计信息"""
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("statistics", {})

def parse_log_file(log_path: Path) -> Dict:
    """从日志文件中提取统计信息"""
    stats = {}
    if not log_path.exists():
        return stats
    
    with log_path.open("r", encoding="utf-8") as f:
        content = f.read()
        # 尝试从日志中提取关键信息
        if "整体平均IoU:" in content:
            for line in content.split("\n"):
                if "整体平均IoU:" in line:
                    try:
                        value = float(line.split("整体平均IoU:")[1].strip())
                        stats["overall_mean_iou"] = value
                    except:
                        pass
                if "每张图像平均IoU:" in line:
                    try:
                        value = float(line.split("每张图像平均IoU:")[1].strip())
                        stats["per_image_mean_iou"] = value
                    except:
                        pass
                if "每个标签平均IoU:" in line:
                    try:
                        value = float(line.split("每个标签平均IoU:")[1].strip())
                        stats["per_label_mean_iou"] = value
                    except:
                        pass
    return stats

def main():
    base_dir = Path("2d_iou_sa2va_outputs_scannet")
    log_dir = base_dir / "logs"
    
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
    print(f"统计目录: 2d_iou_sa2va_outputs_scannet")
    print(f"{'='*60}\n")
    
    for scene in scenes:
        json_path = base_dir / f"{scene}_2d_iou_sa2va.json"
        log_path = log_dir / f"{scene}.log"
        
        stats = None
        
        # 优先从 JSON 文件读取
        if json_path.exists():
            stats = load_statistics(json_path)
        # 如果 JSON 不存在，尝试从日志文件读取
        elif log_path.exists():
            stats = parse_log_file(log_path)
            if not stats:
                print(f"警告: {scene} 的 JSON 和日志文件都找不到统计信息")
                continue
        else:
            print(f"警告: {scene} 的 JSON 和日志文件都不存在，跳过")
            continue
        
        all_stats.append(stats)
        print(f"{scene}:")
        print(f"  overall_mean_iou: {stats.get('overall_mean_iou', 0):.6f}")
        print(f"  per_image_mean_iou: {stats.get('per_image_mean_iou', 0):.6f}")
        print(f"  per_label_mean_iou: {stats.get('per_label_mean_iou', 0):.6f}")
        print()
    
    if not all_stats:
        print("没有找到任何统计文件")
        return
    
    # 计算平均值
    overall_mean_ious = [s.get("overall_mean_iou", 0) for s in all_stats if s.get("overall_mean_iou") is not None]
    per_image_mean_ious = [s.get("per_image_mean_iou", 0) for s in all_stats if s.get("per_image_mean_iou") is not None]
    per_label_mean_ious = [s.get("per_label_mean_iou", 0) for s in all_stats if s.get("per_label_mean_iou") is not None]
    
    if overall_mean_ious:
        avg_overall_mean_iou = sum(overall_mean_ious) / len(overall_mean_ious)
        print("=" * 60)
        print("五个场景的平均 IoU:")
        print(f"  overall_mean_iou: {avg_overall_mean_iou:.6f}")
        if per_image_mean_ious:
            avg_per_image_mean_iou = sum(per_image_mean_ious) / len(per_image_mean_ious)
            print(f"  per_image_mean_iou: {avg_per_image_mean_iou:.6f}")
        if per_label_mean_ious:
            avg_per_label_mean_iou = sum(per_label_mean_ious) / len(per_label_mean_ious)
            print(f"  per_label_mean_iou: {avg_per_label_mean_iou:.6f}")
        print("=" * 60)
    else:
        print("无法计算平均 IoU：没有找到有效的统计信息")

if __name__ == "__main__":
    main()




