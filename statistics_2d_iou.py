#!/usr/bin/env python3
"""
统计2D IoU输出文件中各项指标的平均值
分别统计带_coarse和不带_coarse的文件
"""

import json
import os
from pathlib import Path
from collections import defaultdict

def load_statistics(json_file):
    """从JSON文件中加载statistics数据"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data.get('statistics', {})

def calculate_averages(stats_list):
    """计算统计指标的平均值"""
    if not stats_list:
        return {}
    
    metrics = [
        'overall_mean_iou',
        'overall_median_iou',
        'overall_min_iou',
        'overall_max_iou',
        'per_image_mean_iou',
        'per_label_mean_iou'
    ]
    
    averages = {}
    for metric in metrics:
        values = [s.get(metric, 0) for s in stats_list if metric in s]
        if values:
            averages[metric] = sum(values) / len(values)
        else:
            averages[metric] = 0.0
    
    return averages

def main():
    # 设置输入目录
    input_dir = Path('/mnt/shared-storage-user/intern7shared/liuyifei/code/posevlm/gyn_test_images/2d_iou_outputs_gyn_new_kuixuan_new_1652_Ours_new_gt_sam3')
    
    # 分别收集带coarse和不带coarse的文件
    coarse_stats = []
    non_coarse_stats = []
    
    # 遍历目录中的所有JSON文件
    for json_file in sorted(input_dir.glob('*.json')):
        filename = json_file.name
        
        # 跳过非2d_iou文件
        if '_2d_iou' not in filename:
            continue
        
        try:
            stats = load_statistics(json_file)
            if not stats:
                print(f"警告: {filename} 中没有找到statistics数据")
                continue
            
            if '_coarse' in filename:
                coarse_stats.append(stats)
                print(f"已加载: {filename} (coarse)")
            else:
                non_coarse_stats.append(stats)
                print(f"已加载: {filename} (non-coarse)")
        
        except Exception as e:
            print(f"错误: 无法读取 {filename}: {e}")
    
    # 计算平均值
    coarse_averages = calculate_averages(coarse_stats)
    non_coarse_averages = calculate_averages(non_coarse_stats)
    
    # 打印结果
    print("\n" + "="*80)
    print("统计结果")
    print("="*80)
    
    print(f"\n不带_coarse的文件数量: {len(non_coarse_stats)}")
    print("-" * 80)
    for metric, value in non_coarse_averages.items():
        print(f"  {metric:30s}: {value:.6f}")
    
    print(f"\n带_coarse的文件数量: {len(coarse_stats)}")
    print("-" * 80)
    for metric, value in coarse_averages.items():
        print(f"  {metric:30s}: {value:.6f}")
    
    # 计算差值
    print(f"\n差值 (non-coarse - coarse):")
    print("-" * 80)
    for metric in non_coarse_averages.keys():
        diff = non_coarse_averages[metric] - coarse_averages[metric]
        print(f"  {metric:30s}: {diff:+.6f}")
    
    # 保存结果到JSON文件
    output_file = input_dir / 'statistics_summary.json'
    summary = {
        'non_coarse': {
            'file_count': len(non_coarse_stats),
            'averages': non_coarse_averages
        },
        'coarse': {
            'file_count': len(coarse_stats),
            'averages': coarse_averages
        },
        'differences': {
            metric: non_coarse_averages[metric] - coarse_averages[metric]
            for metric in non_coarse_averages.keys()
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n结果已保存到: {output_file}")

if __name__ == '__main__':
    main()

