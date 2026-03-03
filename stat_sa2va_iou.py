#!/usr/bin/env python3
"""
统计所有场景的 Sa2VA 2D IoU 结果
"""

import json
import sys
import numpy as np
from pathlib import Path
from typing import List, Dict


def stat_all_scenes(output_dir: Path, scenes: List[str]) -> Dict:
    """统计所有场景的 IoU"""
    all_ious = []
    scene_stats = {}
    
    for scene in scenes:
        json_file = output_dir / f"{scene}_2d_iou_sa2va.json"
        if not json_file.exists():
            print(f"警告: 场景 {scene} 的结果文件不存在: {json_file}", file=sys.stderr)
            continue
        
        try:
            with json_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            
            stats = data.get("statistics", {})
            results = data.get("results", [])
            
            if results:
                scene_ious = [r["iou"] for r in results]
                all_ious.extend(scene_ious)
                
                scene_stats[scene] = {
                    "count": len(scene_ious),
                    "mean_iou": stats.get("overall_mean_iou", 0.0),
                    "median_iou": stats.get("overall_median_iou", 0.0),
                    "min_iou": stats.get("overall_min_iou", 0.0),
                    "max_iou": stats.get("overall_max_iou", 0.0)
                }
        except Exception as e:
            print(f"错误: 读取场景 {scene} 的结果文件失败: {e}", file=sys.stderr)
            continue
    
    if not all_ious:
        print("错误: 没有找到任何 IoU 数据", file=sys.stderr)
        return None
    
    overall_mean = np.mean(all_ious)
    overall_median = np.median(all_ious)
    overall_min = np.min(all_ious)
    overall_max = np.max(all_ious)
    
    print("\n" + "="*60)
    print("总体统计结果")
    print("="*60)
    print(f"总场景数: {len(scene_stats)}")
    print(f"总计算数: {len(all_ious)}")
    print(f"总平均 IoU: {overall_mean:.4f}")
    print(f"总中位数 IoU: {overall_median:.4f}")
    print(f"总最小 IoU: {overall_min:.4f}")
    print(f"总最大 IoU: {overall_max:.4f}")
    print("\n" + "-"*60)
    print("各场景统计:")
    print("-"*60)
    for scene in sorted(scene_stats.keys()):
        stats = scene_stats[scene]
        print(f"{scene}:")
        print(f"  计算数: {stats['count']}")
        print(f"  平均 IoU: {stats['mean_iou']:.4f}")
        print(f"  中位数 IoU: {stats['median_iou']:.4f}")
        print(f"  最小 IoU: {stats['min_iou']:.4f}")
        print(f"  最大 IoU: {stats['max_iou']:.4f}")
        print()
    
    summary = {
        "total_scenes": len(scene_stats),
        "total_count": len(all_ious),
        "overall_mean_iou": float(overall_mean),
        "overall_median_iou": float(overall_median),
        "overall_min_iou": float(overall_min),
        "overall_max_iou": float(overall_max),
        "per_scene_stats": scene_stats
    }
    
    return summary


def main():
    if len(sys.argv) < 3:
        print("用法: python stat_sa2va_iou.py <output_dir> <scene1> [scene2] ...", file=sys.stderr)
        sys.exit(1)
    
    output_dir = Path(sys.argv[1])
    scenes = sys.argv[2:]
    
    summary = stat_all_scenes(output_dir, scenes)
    
    if summary:
        summary_file = output_dir / "summary_all_scenes.json"
        with summary_file.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n汇总结果已保存到: {summary_file}")


if __name__ == "__main__":
    main()


