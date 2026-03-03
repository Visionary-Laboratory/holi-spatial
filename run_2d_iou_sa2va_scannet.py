#!/usr/bin/env python3
"""
运行 2d_iou_Sa2VA.py 处理 scannet_pgsr_eval 下的五个场景
"""

import subprocess
import sys
from pathlib import Path

def main():
    # 五个场景
    scenes = [
        "scene0124_00",
        "scene0181_02",
        "scene0204_02",
        "scene0347_02",
        "scene0520_00",
    ]
    
    # 输出目录
    output_dir = Path("2d_iou_sa2va_outputs_scannet")
    output_dir.mkdir(exist_ok=True)
    
    # 日志目录
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # 依次运行每个场景
    for scene in scenes:
        print("=" * 60)
        print(f"处理场景: {scene}")
        print("=" * 60)
        
        log_file = log_dir / f"{scene}.log"
        
        cmd = [
            "python", "2d_iou_Sa2VA.py",
            "--scene", scene,
            "--data-root", "scannet_pgsr_eval",
            "--gt-json", f"mask_index_outputs_scannet/{scene}.json",
            "--output-dir", str(output_dir),
        ]
        
        print(f"运行命令: {' '.join(cmd)}")
        print(f"日志文件: {log_file}")
        
        with open(log_file, "w", encoding="utf-8") as f:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            f.write(result.stdout)
            if result.returncode != 0:
                print(f"场景 {scene} 处理失败，返回码: {result.returncode}")
                print(f"错误信息已保存到: {log_file}")
            else:
                print(f"场景 {scene} 处理完成，日志已保存到: {log_file}")
        
        print()
    
    print("=" * 60)
    print("所有场景处理完成！")
    print("=" * 60)
    print("\n运行统计脚本查看结果:")
    print("python stat_2d_iou_sa2va_scannet.py")

if __name__ == "__main__":
    main()




