#!/usr/bin/env python3
"""
运行 2d_iou_gyn_seperate_instance_scene_label_gyy_SAM3.py 处理 DL3DV 下的9个场景
"""

import subprocess
import sys
from pathlib import Path

def main():
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
    
    # 输出目录
    output_dir = Path("2d_iou_dl3dv_sam3")
    output_dir.mkdir(exist_ok=True)
    
    # 日志目录
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # 依次运行每个场景
    for scene in scenes:
        print("=" * 60)
        print(f"处理场景: {scene[:16]}...")
        print("=" * 60)
        
        log_file = log_dir / f"{scene}.log"
        
        cmd = [
            "python", "2d_iou_gyn_seperate_instance_scene_label_gyy_SAM3.py",
            "--scene", scene,
            "--data-root", "DL3DV/1K",
            "--rrd-path", f"output_3d_bounding_dl3dv_evaluation/{scene}.rrd",
            "--output-dir", str(output_dir),
            "--gt-json", f"mask_index_outputs_dl3dv/{scene}.json",
            "--model-path", f"output_DL3DV/1K/{scene}",
        ]
        
        print(f"运行命令: python 2d_iou_gyn_seperate_instance_scene_label_gyy_SAM3.py --scene {scene[:16]}...")
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
                print(f"场景 {scene[:16]}... 处理失败，返回码: {result.returncode}")
                print(f"错误信息已保存到: {log_file}")
            else:
                print(f"场景 {scene[:16]}... 处理完成，日志已保存到: {log_file}")
        
        print()
    
    print("=" * 60)
    print("所有场景处理完成！")
    print("=" * 60)
    print("\n运行统计脚本查看结果:")
    print("python stat_2d_iou_dl3dv_sam3.py")

if __name__ == "__main__":
    main()




