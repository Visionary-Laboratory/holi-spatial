from __future__ import annotations

import argparse
import json
import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Literal

import numpy as np
import torch
import cv2
from PIL import Image
from tqdm import tqdm
import trimesh
from scipy.spatial.transform import Rotation as R
import pycocotools.mask as mask_utils

# 将 PGSR 路径加入 sys.path 以便导入其模块
PGSR_ROOT = Path(__file__).parent / "PGSR"
if str(PGSR_ROOT) not in sys.path:
    sys.path.append(str(PGSR_ROOT))

from arguments import ModelParams, PipelineParams  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from scene.gaussian_model import GaussianModel  # noqa: E402
from scene.cameras import Camera  # noqa: E402

DEFAULT_DATA_ROOT = Path("scannet_pgsr_eval")
DEFAULT_OUTPUT_DIR = Path("rerun_output_scannet_pgsr_eval")
MAX_POINT_COUNT = 10_000_000

def setup_logger(log_path: Optional[Path] = None) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
        
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
        handlers=handlers
    )

def label_color(label: str) -> np.ndarray:
    """根据标签生成稳定的伪随机颜色。"""
    import hashlib
    h = hashlib.md5(label.encode("utf-8")).digest()
    return np.array([h[0], h[1], h[2], 255], dtype=np.uint8)

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
        json_path = scene_dir /    "dslr/nerfstudio/transforms_undistorted.json"
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
            path_map[stem] = str(scene_dir / "dslr/" / "resized_undistorted_images" /frame["file_path"])
            
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
        
        for idx, cam_file in tqdm(enumerate(cam_files),total=len(cam_files),):
            cam_file_path = os.path.join(cam_dir, cam_file)
            cam_data = np.load(cam_file_path)
            if 'intrinsic' in cam_data:
                intrinsic = cam_data['intrinsic']
                fx = intrinsic[0, 0]
                fy = intrinsic[1, 1]
                cx = intrinsic[0, 2]
                cy = intrinsic[1, 2]
            elif 'intrinsics' in cam_data:
                # Fallback for ScanNet-like format
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
    # scannet

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
    image_names: Optional[set[str]] = None
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
    
    if image_names is not None:
        target_stems = {Path(name).stem for name in image_names}
        stems_to_render = [s for s in target_stems if s in c2w_map]
    else:
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
    color: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    H, W = depth.shape
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    d = depth.astype(np.float32).reshape(-1)
    valid = np.isfinite(d) & (d > 0)
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

def _sanitize_entity_segment(value: str) -> str:
    # Avoid rerun entity path warnings by removing whitespace
    # and characters that can break path semantics.
    import re

    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("_")
    return cleaned or "unknown"

def log_rerun_scene(
    points_all: Optional[np.ndarray],
    colors_all: Optional[np.ndarray],
    bboxes: List[Dict],
    c2w_map: Dict[str, np.ndarray],
    intr_map: Dict[str, np.ndarray],
    color_map: Dict[str, np.ndarray],
    scene: str,
    scene_type: str,
    rerun_save_path: Optional[Path],
) -> None:
    try:
        import rerun as rr
    except ImportError:
        logging.warning("未安装 rerun，跳过可视化")
        return

    rr.init(f"json2rerun/{scene}", spawn=False)
    if scene_type == "scannetppv2":
        rr.log("/", rr.ViewCoordinates.RUB)

    if points_all is not None and points_all.shape[0] > 0:
        log_kwargs = {}
        if colors_all is not None and colors_all.shape[0] == points_all.shape[0]:
            log_kwargs["colors"] = colors_all
        rr.log("others", rr.Points3D(points_all, **log_kwargs))

    for box in bboxes:
        label = box["label"]
        ins_id = box["ins_id"]
        label_path = _sanitize_entity_segment(label)
        col = label_color(label)[:3]
        transform = np.array(box["obb_transform"], dtype=np.float32)
        extents = np.array(box["obb_extents"], dtype=np.float32)
        center = transform[:3, 3]
        quat_xyzw = R.from_matrix(transform[:3, :3]).as_quat().astype(np.float32)
        rr.log(
            f"instances/{label_path}/{ins_id}/bbox",
            rr.Boxes3D(
                centers=np.array([center], dtype=np.float32),
                half_sizes=np.array([extents * 0.5], dtype=np.float32),
                quaternions=[rr.Quaternion(xyzw=quat_xyzw)],
                colors=np.array([col], dtype=np.uint8),
                labels=[f"{label}:{ins_id}"],
            ),
        )

    if rerun_save_path:
        rr.save(str(rerun_save_path))
        logging.info("rerun 日志已保存到: %s", rerun_save_path)

def process_scene(
    input_json_path: Path,
    data_root: Path,
    model_path: Path,
    iteration: int,
    output_dir: Path,
) -> None:
    # 1. 解析场景名
    scene = input_json_path.stem
    scene_dir = data_root / scene
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. 加载位姿与内参
    intr_map, c2w_map, scene_type, size_map, path_map = load_transforms(scene_dir)
    
    # 3. 加载 BBox JSON
    with input_json_path.open("r", encoding="utf-8") as f:
        bboxes = json.load(f)
    logging.info("加载了 %d 个 3D bounding boxes", len(bboxes))
    
    # 4. 确定要渲染的 100 张图
    discovery_json_path = Path("scene_objects_Qwen3-VL-30B-A3B-Instruct") / f"{scene}.json"
    image_names = None
    if discovery_json_path.exists():
        with discovery_json_path.open("r", encoding="utf-8") as f:
            disc_data = json.load(f)
            image_names = set(disc_data.get("per_image", {}).keys())
        logging.info("从 %s 加载了 %d 张图片名", discovery_json_path, len(image_names))
    else:
        logging.warning("发现 JSON 不存在: %s，渲染全部相机", discovery_json_path)

    # 5. 3DGS 渲染深度与颜色
    depth_map, color_map = render_depths_with_gaussians(
        model_path, iteration, intr_map, c2w_map, size_map, path_map, image_names
    )
    
    # 6. 生成全局点云
    all_points = []
    all_colors = []
    for stem, depth in depth_map.items():
        color = color_map.get(stem)
        pts, cols = depth_to_points(depth, intr_map[stem], c2w_map[stem], color=color)
        if pts.shape[0] > 0:
            all_points.append(pts)
            if cols is not None:
                all_colors.append(cols)
    
    if not all_points:
        logging.error("未生成任何有效点云")
        return

    pts_all = np.concatenate(all_points, axis=0)
    cols_all = np.concatenate(all_colors, axis=0) if all_colors else None
    
    # 下采样
    if pts_all.shape[0] > MAX_POINT_COUNT:
        idx = np.random.choice(pts_all.shape[0], MAX_POINT_COUNT, replace=False)
        pts_all = pts_all[idx]
        if cols_all is not None:
            cols_all = cols_all[idx]
        logging.info("点云下采样: -> %d", pts_all.shape[0])

    # 7. 保存 PLY
    ply_save_path = output_dir / f"{scene}_points.ply"
    pc = trimesh.points.PointCloud(pts_all, colors=cols_all)
    pc.export(ply_save_path)
    logging.info("点云已保存至: %s", ply_save_path)
    
    # 8. 可视化与 RRD 保存
    rrd_save_path = output_dir / f"{scene}.rrd"
    log_rerun_scene(
        pts_all, cols_all, bboxes, c2w_map, intr_map, color_map, scene, scene_type, rrd_save_path
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", type=Path, help="输入 BBox JSON 路径，例如 output_3d_bounding/0a184cf634.json")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--model-path", "-m", type=Path, required=True, help="3DGS 模型路径")
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--output-dir", type=Path, default=Path("rerun_output"))
    args = parser.parse_args()
    
    setup_logger()
    process_scene(
        args.input_json, args.data_root, args.model_path, args.iteration, args.output_dir
    )

if __name__ == "__main__":
    main()
