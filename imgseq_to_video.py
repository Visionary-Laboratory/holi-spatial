#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将一组按照文件名排序的图片合成为视频。

用法示例：
  python imgseq_to_video.py \
      --input_dir bbox_overlay_out/09d6e808b4 \
      --output out.mp4 \
      --fps 25

优先使用 ffmpeg (libx264 + yuv420p) 编码，便于 VS Code / 浏览器直接预览；
若未安装 ffmpeg 则回退到 OpenCV 编码。

依赖：
  pip install opencv-python
  可选：系统安装 ffmpeg（推荐，用于生成兼容性更好的 mp4）
"""

import argparse
import re
import shutil
import subprocess
from pathlib import Path
from typing import List

import cv2


def _natural_sort_key(path: Path):
    """自然排序：把文件名中的数字部分按数值比较，而非字典序。
    例如 1.png, 2.png, 10.png, 11.png 会正确排序为 1 < 2 < 10 < 11。"""
    name = path.name
    parts = re.split(r"(\d+)", name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def list_images(input_dir: Path, exts=(".png", ".jpg", ".jpeg")) -> List[Path]:
    files = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    # 按自然顺序排序（1, 2, ..., 9, 10, 11 而非 1, 10, 11, 2, 3...）
    files.sort(key=_natural_sort_key)
    return files


def images_to_video_ffmpeg(imgs: List[Path], output_path: Path, w: int, h: int, fps: int) -> bool:
    """用 ffmpeg libx264 + yuv420p 编码，便于 VS Code/浏览器播放。成功返回 True，否则 False。"""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-movflags", "+faststart",
        str(output_path),
    ]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for i, img_path in enumerate(imgs):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            if img.shape[0] != h or img.shape[1] != w:
                img = cv2.resize(img, (w, h))
            proc.stdin.write(img.tobytes())
            if (i + 1) % 50 == 0:
                print(f"已写入 {i + 1}/{len(imgs)} 帧...")
        proc.stdin.close()
        proc.wait()
        if proc.returncode != 0:
            return False
        return True
    except Exception:
        return False


def images_to_video(input_dir: Path, output_path: Path, fps: int) -> None:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    imgs = list_images(input_dir)
    if not imgs:
        raise RuntimeError(f"目录中没有找到图片: {input_dir}")

    # 读取第一张图片确定分辨率
    first = cv2.imread(str(imgs[0]))
    if first is None:
        raise RuntimeError(f"无法读取图片: {imgs[0]}")
    h, w = first.shape[:2]

    # 优先用 ffmpeg (libx264 + yuv420p)，便于 VS Code / 浏览器直接预览
    if images_to_video_ffmpeg(imgs, output_path, w, h, fps):
        print(f"完成（ffmpeg H.264）！输出视频: {output_path}，共 {len(imgs)} 帧，FPS={fps}")
        return

    # 回退到 OpenCV 编码
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    for codec_name, fourcc in [("avc1", "avc1"), ("H264", "H264"), ("mp4v", "mp4v")]:
        vw = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
        if vw.isOpened():
            writer = vw
            print(f"未找到 ffmpeg，使用 OpenCV 编码: {codec_name}")
            break
        vw.release()
    if writer is None:
        raise RuntimeError(
            f"视频写入失败: {output_path}。建议安装 ffmpeg 以生成 VS Code 可预览的 mp4。"
        )

    try:
        for i, img_path in enumerate(imgs):
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[警告] 跳过无法读取的图片: {img_path}")
                continue

            if img.shape[0] != h or img.shape[1] != w:
                img = cv2.resize(img, (w, h))

            writer.write(img)
            if (i + 1) % 50 == 0:
                print(f"已写入 {i + 1}/{len(imgs)} 帧...")
    finally:
        writer.release()

    print(f"完成！输出视频: {output_path}，共 {len(imgs)} 帧，FPS={fps}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将图片序列合成为视频")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="输入图片所在的目录，例如 bbox_overlay_out/09d6e808b4",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出视频路径，例如 out.mp4",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="视频帧率（默认为 25）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    images_to_video(input_dir, output_path, args.fps)


if __name__ == "__main__":
    main()

