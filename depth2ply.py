#!/usr/bin/env python3
"""
从 GS 模型渲染深度，结合 mask 过滤，生成点云并保存为 PLY
"""

import argparse
import json
import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import trimesh

# 将 PGSR 路径加入 sys.path 以便导入其模块
PGSR_ROOT = Path(__file__).parent / "PGSR"
if str(PGSR_ROOT) not in sys.path:
    sys.path.append(str(PGSR_ROOT))

from arguments import ModelParams, PipelineParams  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from scene.gaussian_model import GaussianModel  # noqa: E402
from scene.cameras import Camera  # noqa: E402

MAX_POINT_COUNT = 4_000_000


def setup_logger() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def load_mask(mask_path: Path) -> np.ndarray:
    """加载 mask 图像，返回布尔数组"""
    mask = Image.open(mask_path).convert("L")
    return np.array(mask) > 0


def load_transforms(scene_dir: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], str, Dict[str, Tuple[int, int]], Dict[str, str]]:
    """
    读取 nerfstudio transforms_undistorted.json，返回
      intr_map[stem] = 3x3 K
      c2w_map[stem] = 4x4 camera-to-world
      scene_type: str
      size_map[stem] = (width, height)
      path_map[stem] = image_path
    """
    # scannetppv2
    if (scene_dir/"dslr").exists():
        scene_type = "scannetppv2"
        json_path = scene_dir / "dslr/nerfstudio/transforms_undistorted.json"
        with json_path.open("r", encoding="utf-8") as f:
            contents = json.load(f)

        fl_x, fl_y, cx, cy = contents["fl_x"], contents["fl_y"], contents["cx"], contents["cy"]
        w, h = contents.get("w"), contents.get("h")
        train_frames = contents.get("frames", [])
        test_frames = contents.get("test_frames", [])

        intr_map: Dict[str, np.ndarray] = {}
        c2w_map: Dict[str, np.ndarray] = {}
        size_map: Dict[str, Tuple[int, int]] = {}
        path_map: Dict[str, str] = {}
        
        for frame in [*train_frames, *test_frames]:
            K = np.array([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]], dtype=np.float32)
            c2w = np.array(frame["transform_matrix"], dtype=np.float32)
            # OpenGL -> COLMAP 约定
            c2w[:3, 1:3] *= -1
            stem = Path(frame["file_path"]).stem
            intr_map[stem] = K
            c2w_map[stem] = c2w
            
            frame_w = frame.get("w", w)
            frame_h = frame.get("h", h)
            size_map[stem] = (frame_w, frame_h)
            path_map[stem] = str(scene_dir / "dslr/" / "resized_undistorted_images" / frame["file_path"])
            
        logging.info("加载位姿完成，数量: %d", len(intr_map))
        return intr_map, c2w_map, scene_type, size_map, path_map
    elif (scene_dir/"cam").exists() and (scene_dir/"color").exists():
        scene_type = "scannet"
        cam_dir = scene_dir / "cam"
        color_dir = scene_dir / "color"
        cam_files = sorted([f for f in os.listdir(cam_dir) if f.endswith('.npz')])
        intr_map: Dict[str, np.ndarray] = {}
        c2w_map: Dict[str, np.ndarray] = {}
        size_map: Dict[str, Tuple[int, int]] = {}
        path_map: Dict[str, str] = {}
        
        for idx, cam_file in tqdm(enumerate(cam_files), total=len(cam_files), desc="加载 ScanNet 相机数据"):
            cam_file_path = cam_dir / cam_file
            cam_data = np.load(cam_file_path)
            
            # 读取内参
            if 'intrinsic' in cam_data:
                intrinsic = cam_data['intrinsic']
                fx = intrinsic[0, 0]
                fy = intrinsic[1, 1]
                cx = intrinsic[0, 2]
                cy = intrinsic[1, 2]
            elif 'intrinsics' in cam_data:
                intrinsics = cam_data['intrinsics']
                fx = intrinsics[0, 0]
                fy = intrinsics[1, 1]
                cx = intrinsics[0, 2]
                cy = intrinsics[1, 2]
            else:
                raise ValueError(f"No intrinsics found in {cam_file}. Available keys: {list(cam_data.keys())}")
            
            # 读取位姿
            if 'pose' in cam_data:
                c2w = cam_data['pose']
            else:
                raise ValueError(f"No pose found in {cam_file}. Available keys: {list(cam_data.keys())}")
            
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            stem = Path(cam_file).stem
            intr_map[stem] = K
            c2w_map[stem] = c2w
            
            # 查找对应的图像文件
            img_path = color_dir / f"{stem}.png"
            if not img_path.exists():
                img_path = color_dir / f"{stem}.jpg"
            if not img_path.exists():
                logging.warning(f"未找到图像文件: {stem}，跳过")
                continue
            
            # 获取图像尺寸
            if 'width' in cam_data and 'height' in cam_data:
                size_map[stem] = (int(cam_data['width']), int(cam_data['height']))
            else:
                # 从图像文件读取尺寸
                try:
                    with Image.open(img_path) as img:
                        size_map[stem] = img.size
                except Exception as e:
                    logging.warning(f"无法读取图像尺寸 {img_path}: {e}，跳过")
                    continue
            
            path_map[stem] = str(img_path)
        logging.info("加载位姿完成，数量: %d", len(intr_map))
        return intr_map, c2w_map, scene_type, size_map, path_map
    # dl3dv
    elif (scene_dir/"dense").exists():
        scene_type = "dl3dv"
        cam_dir = scene_dir / "dense" / "cam"
        rgb_dir = scene_dir / "dense" / "rgb"
        cam_files = sorted([f for f in os.listdir(cam_dir) if f.endswith('.npz')])
        intr_map: Dict[str, np.ndarray] = {}
        c2w_map: Dict[str, np.ndarray] = {}
        size_map: Dict[str, Tuple[int, int]] = {}
        path_map: Dict[str, str] = {}
        
        for idx, cam_file in tqdm(enumerate(cam_files), total=len(cam_files), desc="加载 DL3DV 相机数据"):
            cam_file_path = os.path.join(cam_dir, cam_file)
            cam_data = np.load(cam_file_path)
            if 'intrinsic' in cam_data:
                intrinsic = cam_data['intrinsic']
                fx = intrinsic[0, 0]
                fy = intrinsic[1, 1]
                cx = intrinsic[0, 2]
                cy = intrinsic[1, 2]
            elif 'intrinsics' in cam_data:
                intrinsics = cam_data['intrinsics']
                fx = intrinsics[0, 0]
                fy = intrinsics[1, 1]
                cx = intrinsics[0, 2]
                cy = intrinsics[1, 2]
            else:
                raise ValueError(f"No intrinsics found in {cam_file}. Available keys: {list(cam_data.keys())}")
            c2w = cam_data['pose']
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            stem = Path(cam_file).stem
            intr_map[stem] = K
            c2w_map[stem] = c2w
            
            # DL3DV 通常在 npz 中包含尺寸，或者需要读取图片
            img_path = rgb_dir / f"{stem}.png"
            if not img_path.exists():
                img_path = rgb_dir / f"{stem}.jpg"
            
            if 'width' in cam_data and 'height' in cam_data:
                size_map[stem] = (int(cam_data['width']), int(cam_data['height']))
            else:
                # 如果没有尺寸，尝试读取第一张图获取尺寸
                with Image.open(img_path) as img:
                    size_map[stem] = img.size
            path_map[stem] = str(img_path)
            
        logging.info("加载位姿完成，数量: %d", len(intr_map))
        return intr_map, c2w_map, scene_type, size_map, path_map
    else:
        raise NotImplementedError(f"不支持的场景格式: {scene_dir}, 目前只支持scannetppv2、scannet和dl3dv.")


def load_gaussian_cfg(model_path: Path) -> argparse.Namespace:
    cfg_path = model_path / "cfg_args"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg_ns = eval(f.read(), {"Namespace": argparse.Namespace})

    cfg = vars(cfg_ns).copy()
    cfg["model_path"] = str(model_path)

    repo_root = Path(__file__).parent
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    if "source_path" in cfg:
        cfg["source_path"] = os.path.join(curr_dir, cfg["source_path"])
    if "images" in cfg:
        cfg["images"] = os.path.join(curr_dir, cfg["images"])
    return argparse.Namespace(**cfg)


def build_pipeline_defaults() -> argparse.Namespace:
    pipeline_params = PipelineParams(argparse.ArgumentParser())
    vals = {k.lstrip("_"): v for k, v in vars(pipeline_params).items() if not k.startswith("__")}
    return argparse.Namespace(**vals)


def render_depths_with_gaussians(
    model_path: Path, 
    iteration: int, 
    intr_map: Dict[str, np.ndarray], 
    c2w_map: Dict[str, np.ndarray],
    size_map: Dict[str, Tuple[int, int]],
    path_map: Dict[str, str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    from utils.graphics_utils import focal2fov
    
    cfg = load_gaussian_cfg(model_path)
    dataset = ModelParams(argparse.ArgumentParser(), sentinel=True).extract(cfg)
    pipeline = PipelineParams(argparse.ArgumentParser()).extract(build_pipeline_defaults())

    gaussians = GaussianModel(dataset.sh_degree)
    ply_path = model_path / "point_cloud" / f"iteration_{iteration if iteration > 0 else 30000}" / "point_cloud.ply"
    if not ply_path.exists():
        # 尝试寻找最新的 iteration
        pc_dir = model_path / "point_cloud"
        iters = sorted([int(d.name.split('_')[-1]) for d in pc_dir.glob("iteration_*")])
        if iters:
            ply_path = pc_dir / f"iteration_{iters[-1]}" / "point_cloud.ply"
        else:
            raise FileNotFoundError(f"未找到 point_cloud.ply: {ply_path}")
            
    gaussians.load_ply(str(ply_path))
    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")

    depth_map: Dict[str, np.ndarray] = {}
    color_map: Dict[str, np.ndarray] = {}
    
    stems_to_render = list(c2w_map.keys())

    logging.info("准备渲染相机数量: %d", len(stems_to_render))

    with torch.no_grad():
        for stem in tqdm(stems_to_render, desc="Rendering depths"):
            c2w = c2w_map[stem]
            K = intr_map[stem]
            width, height = size_map[stem]
            image_path = path_map[stem]

            R_c2w = c2w[:3, :3]
            T_c2w = c2w[:3, 3]
            R_w2v = R_c2w.T
            T_w2v = -R_w2v @ T_c2w
            
            fx, fy = K[0, 0], K[1, 1]
            fovx = focal2fov(fx, width)
            fovy = focal2fov(fy, height)

            cam = Camera(
                colmap_id=0,
                R=R_c2w,
                T=T_w2v,
                FoVx=fovx,
                FoVy=fovy,
                image_width=width,
                image_height=height,
                image_path=image_path,
                image_name=stem,
                uid=0,
                preload_img=False,
                data_device="cuda"
            )

            out = render(cam, gaussians, pipeline, background)
            depth_np = out["plane_depth"].squeeze().detach().cpu().numpy()
            depth_map[stem] = depth_np
            color = out["render"].permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy()
            color_map[stem] = color

    return depth_map, color_map


def depth_to_points(
    depth: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
    mask: Optional[np.ndarray] = None,
    color: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    将深度图转换为点云
    mask: 布尔数组，True 表示保留该像素
    """
    H, W = depth.shape
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    d = depth.astype(np.float32).reshape(-1)
    valid = np.isfinite(d) & (d > 0)
    
    # 应用 mask 过滤
    if mask is not None:
        # 确保 mask 尺寸匹配
        if mask.shape[:2] != (H, W):
            mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
            mask_resized = mask_img.resize((W, H), resample=Image.NEAREST)
            mask = (np.array(mask_resized) > 0).astype(bool)
        mask_flat = mask.reshape(-1)
        valid = valid & mask_flat
    
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), None
    
    xs = xs.reshape(-1)[valid].astype(np.float32)
    ys = ys.reshape(-1)[valid].astype(np.float32)
    d = d[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (xs - cx) / fx * d
    y_cam = (ys - cy) / fy * d
    z_cam = d
    ones = np.ones_like(z_cam)
    cam_pts = np.stack([x_cam, y_cam, z_cam, ones], axis=1)
    world_pts = (cam_pts @ c2w.T)[:, :3]
    cols = None
    if color is not None:
        col = color.reshape(-1, color.shape[-1])[valid]
        if col.dtype != np.uint8:
            col = (col * 255.0).clip(0, 255).astype(np.uint8)
        cols = col[:, :3]
    return world_pts.astype(np.float32), cols


def load_masks(scene_dir: Path, stems: List[str], size_map: Dict[str, Tuple[int, int]]) -> Dict[str, np.ndarray]:
    """
    从 scene_dir/mask 目录加载所有 mask
    返回: {stem: mask_bool_array}
    """
    mask_dir = scene_dir / "mask"
    mask_map: Dict[str, np.ndarray] = {}
    
    if not mask_dir.exists():
        logging.warning(f"Mask 目录不存在: {mask_dir}，将不使用 mask 过滤")
        return mask_map
    
    for stem in tqdm(stems, desc="加载 masks"):
        mask_path = mask_dir / f"{stem}.png"
        if mask_path.exists():
            mask = load_mask(mask_path)
            # 调整 mask 尺寸以匹配图像尺寸
            if stem in size_map:
                frame_w, frame_h = size_map[stem]
                if mask.shape[:2] != (frame_h, frame_w):
                    mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
                    mask_resized = mask_img.resize((frame_w, frame_h), resample=Image.NEAREST)
                    mask = (np.array(mask_resized) > 0).astype(bool)
            mask_map[stem] = mask
        else:
            # 如果没有 mask，创建一个全为 True 的 mask（不过滤）
            if stem in size_map:
                frame_w, frame_h = size_map[stem]
                mask_map[stem] = np.ones((frame_h, frame_w), dtype=bool)
            else:
                logging.warning(f"未找到 mask 且无法确定尺寸: {stem}")
    
    logging.info("加载了 %d 个 masks", len(mask_map))
    return mask_map


def main():
    parser = argparse.ArgumentParser(description="从 GS 模型渲染深度，结合 mask 过滤，生成点云并保存为 PLY")
    parser.add_argument("--scene", type=str, required=True, help="场景名称，如 0a7cc12c0e")
    parser.add_argument("--data-root", type=Path, default=Path("scannetppv2/data"), help="数据根目录")
    parser.add_argument("--model-path", "-m", type=Path, required=True, help="3DGS 模型路径（包含 mask 目录）")
    parser.add_argument("--iteration", type=int, default=30000, help="GS 模型迭代次数")
    parser.add_argument("--output", type=Path, required=True, help="输出 PLY 文件路径")
    args = parser.parse_args()
    
    setup_logger()
    
    scene_dir = args.data_root / args.scene
    if not scene_dir.exists():
        logging.error(f"场景目录不存在: {scene_dir}")
        return
    
    if not args.model_path.exists():
        logging.error(f"模型路径不存在: {args.model_path}")
        return
    
    # 1. 加载位姿与内参
    logging.info("加载相机位姿和内参...")
    intr_map, c2w_map, scene_type, size_map, path_map = load_transforms(scene_dir)
    
    # 2. 加载 masks
    logging.info("加载 masks...")
    mask_map = load_masks(scene_dir, list(c2w_map.keys()), size_map)
    
    # 3. 3DGS 渲染深度与颜色
    logging.info("渲染深度和颜色...")
    depth_map, color_map = render_depths_with_gaussians(
        args.model_path, args.iteration, intr_map, c2w_map, size_map, path_map
    )
    
    # 4. 生成全局点云（使用 mask 过滤）
    logging.info("生成点云（应用 mask 过滤）...")
    all_points = []
    all_colors = []
    for stem, depth in tqdm(depth_map.items(), desc="投射点云"):
        color = color_map.get(stem)
        mask = mask_map.get(stem)
        pts, cols = depth_to_points(depth, intr_map[stem], c2w_map[stem], mask=mask, color=color)
        if pts.shape[0] > 0:
            all_points.append(pts)
            if cols is not None:
                all_colors.append(cols)
    
    if not all_points:
        logging.error("未生成任何有效点云")
        return
    
    pts_all = np.concatenate(all_points, axis=0)
    cols_all = np.concatenate(all_colors, axis=0) if all_colors else None
    
    logging.info("生成点云总数: %d", pts_all.shape[0])
    
    # 5. 下采样
    if pts_all.shape[0] > MAX_POINT_COUNT:
        logging.info("下采样点云: %d -> %d", pts_all.shape[0], MAX_POINT_COUNT)
        idx = np.random.choice(pts_all.shape[0], MAX_POINT_COUNT, replace=False)
        pts_all = pts_all[idx]
        if cols_all is not None:
            cols_all = cols_all[idx]
    
    # 6. 保存 PLY
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pc = trimesh.points.PointCloud(pts_all, colors=cols_all)
    pc.export(args.output)
    logging.info("点云已保存至: %s (点数: %d)", args.output, pts_all.shape[0])


if __name__ == "__main__":
    main()

