#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Set, Dict

import cv2
import numpy as np
import torch
from tqdm import tqdm

from json2bbox_images_depth_scannet import (  # type: ignore
    load_transforms,
    render_depths_with_gaussians,
    normalize_image_name,
    setup_logger,
)


def visualize_scalars(scalar_tensor: torch.Tensor) -> np.ndarray:
    """
    将单通道标量张量可视化为伪彩色 RGB 图像。
    使用分位数归一化，增强对比度。
    """
    to_use = scalar_tensor.view(-1)
    while to_use.shape[0] > 2**24:
        to_use = to_use[::2]

    mi = torch.quantile(to_use, 0.05)
    ma = torch.quantile(to_use, 0.95)

    scalar_tensor = (scalar_tensor - mi) / max(ma - mi, 1e-8)  # normalize to 0~1
    scalar_tensor = scalar_tensor.clamp_(0, 1)

    scalar_tensor = ((1 - scalar_tensor) * 255).byte().cpu().numpy()  # inverse heatmap
    return cv2.cvtColor(
        cv2.applyColorMap(scalar_tensor, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB
    )


def render_depth_for_scene(
    scene: str,
    data_root: Path,
    model_path: Path,
    output_dir: Path,
    images: Optional[List[str]] = None,
    iteration: int = 30000,
) -> None:
    """
    对单个 ScanNet 场景使用 3DGS 渲染深度，并把深度可视化图保存到指定目录。
    """
    scene_dir = data_root / scene
    intr_map, c2w_map, _, size_map, path_map = load_transforms(scene_dir)

    if images:
        stems: List[str] = [normalize_image_name(x) for x in images]
        # 只保留在相机列表中的帧
        stems = [s for s in stems if s in c2w_map]
    else:
        stems = sorted(c2w_map.keys())

    if not stems:
        logging.warning("场景 %s 没有需要渲染的相机帧，直接返回。", scene)
        return

    depth_model_path = model_path
    if not (depth_model_path / "cfg_args").exists():
        scene_model = depth_model_path / scene
        if (scene_model / "cfg_args").exists():
            depth_model_path = scene_model
            logging.info("检测到多场景模型目录，自动使用: %s", depth_model_path)

    logging.info("开始渲染场景 %s 的高斯深度，共 %d 帧", scene, len(stems))

    # 渲染深度（以相机 z 为单位的平面深度）
    depth_map: Dict[str, np.ndarray] = render_depths_with_gaussians(
        model_path=depth_model_path,
        iteration=iteration,
        intr_map=intr_map,
        c2w_map=c2w_map,
        size_map={k: (int(v[0]), int(v[1])) for k, v in size_map.items()},
        path_map=path_map,
        image_names=set(stems),
    )

    out_scene_dir = output_dir / scene
    out_scene_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for stem in tqdm(stems, desc=f"Save depth {scene}", leave=True):
        depth = depth_map.get(stem)
        if depth is None:
            continue

        # depth: (H, W) float32, 单位为米，可能包含 0 或 inf
        depth_np = np.asarray(depth, dtype=np.float32)

        # 转成 torch 张量做可视化
        depth_t = torch.from_numpy(depth_np)
        if depth_t.ndim == 2:
            pass
        elif depth_t.ndim == 3 and depth_t.shape[0] == 1:
            depth_t = depth_t[0]
        else:
            depth_t = depth_t.squeeze()

        depth_vis = visualize_scalars(depth_t)  # RGB, uint8
        depth_bgr = cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR)

        out_path = out_scene_dir / f"{stem}.png"
        cv2.imwrite(str(out_path), depth_bgr)
        saved += 1

    logging.info("场景 %s 深度图保存完成: %d 张 -> %s", scene, saved, out_scene_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="只进行 3D 高斯深度渲染，并把每帧深度保存为彩色可视化 PNG。"
    )
    parser.add_argument("scene", type=str, help="ScanNet 场景名，例如 scene0477_00")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            "/mnt/shared-storage-user/intern7shared/liuyifei/scannetv2/scans"
        ),
        help="ScanNet scans 根目录（默认: scannetv2/scans）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./gaussian_depth_outputs_scannet"),
        help="深度可视化输出根目录（每个 scene 一个子目录）",
    )
    parser.add_argument(
        "--model-path",
        "-m",
        type=Path,
        required=True,
        help="3DGS 模型目录；如果是多场景目录，会自动拼接 scene 子目录",
    )
    parser.add_argument(
        "--images",
        nargs="*",
        default=None,
        help="只渲染指定图像（stem 或文件名），不指定则渲染该场景的所有相机帧",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=30000,
        help="3DGS iteration（默认 30000，与 json2bbox_images_depth_scannet.py 一致）",
    )

    args = parser.parse_args()
    setup_logger()

    render_depth_for_scene(
        scene=args.scene,
        data_root=args.data_root,
        model_path=args.model_path,
        output_dir=args.output_dir,
        images=args.images,
        iteration=args.iteration,
    )


if __name__ == "__main__":
    main()

