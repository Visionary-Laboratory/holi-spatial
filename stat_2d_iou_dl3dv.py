#!/usr/bin/env python3
"""
统计 2d_iou_dl3dv 目录下9个场景的平均 IoU
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
    base_dir = Path("2d_iou_dl3dv")
    log_dir = base_dir / "logs"
    
    # 9个场景
    scenes = [
        "0bfab69b5b2b692d066b4435f2f6406357772aa53a3e17c3e0b8598083848a7a",
        "2d52e65f703cf307013273ffeea79b76db153d665294becb16f00477aac05a88",
        "5b1e56386e263f2324365d2c623d6b746280aa4f5a6257bb0ba66e5dff135c8a",
        "7ae35f524e63aa86d1985f1c59dfd7a29cbd6069770ee26506ac26705917d9ea",
        "7ed574ca0081fa9a8c879a4b360ec7b274b3a9d248c1a6e66bd1e1d1c47bea4b",
        "7f5f6eac88ce535147bb9a7740635901929af1b08fbf0dd1adf28bf8aeee380a",
        "b3d3847b167ee770b105de4388108a1b8f3bfaa430d4d6ff0ba910d271d8a487",
        "c07ac2cdc0dddc8fe07c42258725ca0a5b486dcaccaef4fcbe8f5311cdee46fc",
        "cd6c19a00fe93297c128769f81f7778f8c2b7a6ca7d1ca780b3f83803cdc2082",
    ]
    
    all_stats = []
    
    print(f"\n{'='*60}")
    print(f"统计目录: 2d_iou_dl3dv")
    print(f"{'='*60}\n")
    
    for scene in scenes:
        json_path = base_dir / f"{scene}_2d_iou.json"
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
        scene_short = scene[:16] + "..."  # 显示前16个字符
        print(f"{scene_short}:")
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
        print(f"9个场景的平均 IoU:")
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




