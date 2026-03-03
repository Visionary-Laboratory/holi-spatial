from __future__ import annotations

import argparse
import json
import logging
import sys
import os
import random
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
MULTI_VIEW_ROOT_FALLBACK = Path("/mnt/shared-storage-gpfs2/solution-gpfs02/liuyifei/pgsr_scannetppv2_all")
DEFAULT_DESC_ROOT = Path("/mnt/shared-storage-gpfs2/solution-gpfs02/liuyifei/desc_copy/output_3d_bounding_with_descriptions")

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
    else:
        raise NotImplementedError(f"不支持的场景格式: {scene_dir}, 目前只支持scannetppv2和dl3dv.")

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

def extract_image_names_from_bboxes(bboxes: List[Dict]) -> set[str]:
    """
    从 bboxes JSON 中提取所有图像名称
    
    Args:
        bboxes: bounding boxes 列表
    
    Returns:
        图像名称集合（stem，不含扩展名）
    """
    image_names = set()
    for box in bboxes:
        images = box.get("images", [])
        for image_path in images:
            # 从路径中提取图像名称
            # 例如: /home/dengliyuan/code/posevlm/sam_masks_debug_scannet/0a7cc12c0e/DSC05827/bed.png
            # 路径结构: .../scene_id/image_name/label.png
            # 需要提取 image_name (DSC05827)
            path_parts = Path(image_path).parts
            # 查找场景ID后面的部分（图像名称）
            # 通常格式: .../sam_masks_debug_scannet/scene_id/image_name/label.png
            for i, part in enumerate(path_parts):
                # 检查下一部分是否是图像文件（.png, .jpg等）
                if i + 1 < len(path_parts) and path_parts[i + 1].endswith(('.png', '.jpg', '.JPG', '.PNG')):
                    # 当前部分就是图像名称
                    image_name = part
                    image_names.add(image_name)
                    break
    
    logging.info("从 bboxes 中提取了 %d 个图像名称: %s", len(image_names), sorted(list(image_names))[:10])
    return image_names

def normalize_image_name(name: str) -> str:
    """将输入的图像名归一化为 stem（去扩展名）。"""
    return Path(name).stem

def extract_image_stems_from_paths(image_paths: List[str]) -> set[str]:
    """从 mask 路径列表中提取图像 stem（例如 DSC06102）。"""
    stems = set()
    for image_path in image_paths:
        path_parts = Path(image_path).parts
        for i, part in enumerate(path_parts):
            if i + 1 < len(path_parts) and path_parts[i + 1].endswith(('.png', '.jpg', '.JPG', '.PNG')):
                stems.add(part)
                break
    return stems

def load_multi_view_relations(scene: str, model_path: Path) -> Dict[str, set[str]]:
    """加载 multi_view.json，返回 ref_name -> set(nearest_name) 映射。"""
    candidate_paths = [
        model_path / "multi_view.json",
        MULTI_VIEW_ROOT_FALLBACK / scene / "multi_view.json",
    ]
    for mv_path in candidate_paths:
        if not mv_path.exists():
            continue
        relations: Dict[str, set[str]] = {}
        with mv_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    # 兼容整文件是 JSON 列表的情况
                    try:
                        f.seek(0)
                        data = json.load(f)
                        for item in data:
                            ref = item.get("ref_name")
                            neighbors = item.get("nearest_name", [])
                            if ref:
                                relations[ref] = set(neighbors)
                        logging.info("加载 multi_view (list) 成功: %s (%d)", mv_path, len(relations))
                        return relations
                    except Exception as e:
                        logging.warning("解析 multi_view 失败: %s (%s)", mv_path, e)
                        break
                else:
                    ref = obj.get("ref_name")
                    neighbors = obj.get("nearest_name", [])
                    if ref:
                        relations[ref] = set(neighbors)
        if relations:
            logging.info("加载 multi_view 成功: %s (%d)", mv_path, len(relations))
            return relations
    logging.warning("未找到 multi_view.json，跳过关联筛选")
    return {}

def pick_related_images(
    rng: random.Random,
    candidates: List[str],
    relations: Dict[str, set[str]],
    k: int,
) -> List[str]:
    """从 candidates 中选择 k 张图，优先保证 multi_view 关联性。"""
    if k <= 0:
        return []
    if k == 1 or not relations:
        return rng.sample(candidates, k)
    candidate_set = set(candidates)
    rel_map: Dict[str, List[str]] = {}
    for ref, neighbors in relations.items():
        if ref not in candidate_set:
            continue
        valid_neighbors = [n for n in neighbors if n in candidate_set]
        if valid_neighbors:
            rel_map[ref] = valid_neighbors
    valid_refs = [ref for ref, neighbors in rel_map.items() if len(neighbors) >= k - 1]
    if not valid_refs:
        if rel_map:
            ref = rng.choice(list(rel_map.keys()))
            neighbors = rel_map[ref]
            chosen_neighbors = rng.sample(neighbors, min(k - 1, len(neighbors)))
            selected = [ref] + chosen_neighbors
            if len(selected) < k:
                remaining = list(candidate_set - set(selected))
                if len(remaining) >= (k - len(selected)):
                    selected += rng.sample(remaining, k - len(selected))
            rng.shuffle(selected)
            logging.warning("multi_view 关联不足以满足 k=%d，使用可用关联并补齐", k)
            return selected
        logging.warning("未找到 multi_view 关联，回退为随机选择")
        return rng.sample(candidates, k)
    ref = rng.choice(valid_refs)
    neighbors = rel_map[ref]
    selected = [ref] + rng.sample(neighbors, k - 1)
    rng.shuffle(selected)
    return selected

def merge_descriptions(bboxes: List[Dict], desc_path: Path) -> List[Dict]:
    """将 desc_path 中的 description 合并到当前 bboxes。"""
    if not desc_path.exists():
        logging.info("未找到描述文件，跳过: %s", desc_path)
        return bboxes
    try:
        with desc_path.open("r", encoding="utf-8") as f:
            desc_bboxes = json.load(f)
    except Exception as e:
        logging.warning("读取描述文件失败: %s (%s)", desc_path, e)
        return bboxes
    desc_map: Dict[Tuple[str, str], str] = {}
    for item in desc_bboxes:
        ins_id = str(item.get("ins_id", ""))
        label = str(item.get("label", ""))
        desc = item.get("description")
        if ins_id and label and isinstance(desc, str) and desc.strip():
            desc_map[(ins_id, label)] = desc.strip()
    if not desc_map:
        logging.info("描述文件中未找到有效 description: %s", desc_path)
        return bboxes
    merged = 0
    for box in bboxes:
        ins_id = str(box.get("ins_id", ""))
        label = str(box.get("label", ""))
        key = (ins_id, label)
        if key in desc_map:
            box["description"] = desc_map[key]
            merged += 1
    logging.info("合并 description 完成: %d/%d", merged, len(bboxes))
    return bboxes

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
    size_map: Optional[Dict[str, Tuple[int, int]]] = None,
    path_map: Optional[Dict[str, str]] = None,
    show_labels: bool = True,
    include_desc: bool = True,
    process_obb: bool = False,  # 是否处理 OBB（在相机坐标系中只保留 y 轴旋转，默认关闭）
) -> None:
    try:
        import rerun as rr
    except ImportError:
        logging.warning("未安装 rerun，跳过可视化")
        return

    rr.init(f"json2rerun/{scene}", spawn=False)
    if scene_type == "scannetppv2":
        rr.log("world", rr.ViewCoordinates.RUB)

    if points_all is not None and points_all.shape[0] > 0:
        log_kwargs = {}
        if colors_all is not None and colors_all.shape[0] == points_all.shape[0]:
            log_kwargs["colors"] = colors_all
        rr.log("world/others", rr.Points3D(points_all, **log_kwargs))

    # 获取参考相机（使用第一个可用的相机）
    ref_c2w = None
    ref_w2c_rot = None
    if process_obb and c2w_map:
        first_camera = next(iter(c2w_map.keys()))
        ref_c2w = c2w_map[first_camera]
        # 计算世界到相机的变换矩阵
        # c2w: camera to world, w2c: world to camera
        # w2c = c2w^(-1) = [R^T | -R^T @ t]
        ref_w2c_rot = ref_c2w[:3, :3].T  # 旋转部分的转置

    for box in bboxes:
        label = box["label"]
        ins_id = box["ins_id"]
        col = label_color(label)[:3]
        desc = box.get("description")
        if include_desc:
            if isinstance(desc, str) and desc.strip():
                label_text = f"{label}: {desc.strip()}"
            else:
                logging.warning("缺少 description: label=%s, ins_id=%s", label, ins_id)
                label_text = f"{label}: (no description)"
        else:
            label_text = f"{label}"
        transform = np.array(box["obb_transform"], dtype=np.float32)
        extents = np.array(box["obb_extents"], dtype=np.float32)
        center = transform[:3, 3]
        
        # 提取旋转矩阵
        rot_matrix = transform[:3, :3].copy()
        
        # 处理 OBB（如果需要）
        if process_obb and scene_type in ("scannet", "scannetppv2"):
            # 对于 scannet/scannetppv2，使用读取角点的方式处理
            # 1. 优先从 box 读取 8 个角点
            original_world_corners = None
            
            # 尝试从 bounding_box 字段读取
            if "bounding_box" in box:
                bb = box["bounding_box"]
                if isinstance(bb, list) and len(bb) == 8:
                    if isinstance(bb[0], dict):
                        original_world_corners = np.array([[float(p["x"]), float(p["y"]), float(p["z"])] for p in bb], dtype=np.float32)
                    elif isinstance(bb[0], (list, tuple)) and len(bb[0]) == 3:
                        original_world_corners = np.array([[float(p[0]), float(p[1]), float(p[2])] for p in bb], dtype=np.float32)
            
            # 尝试从 corners 字段读取
            if original_world_corners is None and "corners" in box:
                corners = box["corners"]
                if isinstance(corners, list) and len(corners) == 8:
                    if isinstance(corners[0], dict):
                        original_world_corners = np.array([[float(p["x"]), float(p["y"]), float(p["z"])] for p in corners], dtype=np.float32)
                    elif isinstance(corners[0], (list, tuple)) and len(corners[0]) == 3:
                        original_world_corners = np.array([[float(p[0]), float(p[1]), float(p[2])] for p in corners], dtype=np.float32)
            
            # 如果没有角点，从 transform 和 extents 计算
            if original_world_corners is None:
                half_extents = extents * 0.5
                local_corners = np.array([
                    [-half_extents[0], -half_extents[1], -half_extents[2]],
                    [ half_extents[0], -half_extents[1], -half_extents[2]],
                    [-half_extents[0],  half_extents[1], -half_extents[2]],
                    [ half_extents[0],  half_extents[1], -half_extents[2]],
                    [-half_extents[0], -half_extents[1],  half_extents[2]],
                    [ half_extents[0], -half_extents[1],  half_extents[2]],
                    [-half_extents[0],  half_extents[1],  half_extents[2]],
                    [ half_extents[0],  half_extents[1],  half_extents[2]],
                ], dtype=np.float32)
                original_world_corners = (rot_matrix @ local_corners.T).T + center
            
            # 2. 使用 upright_box_from_corners 处理角点
            try:
                # 使用 PCA 找到主轴
                c = original_world_corners.mean(axis=0)
                X = original_world_corners - c
                cov = (X.T @ X) / X.shape[0]
                w, v = np.linalg.eigh(cov)  # v: columns are eigenvectors
                order = np.argsort(w)[::-1]
                axes = v[:, order]  # columns are local axes in world
                # make right-handed
                if np.linalg.det(axes) < 0:
                    axes[:, 2] *= -1.0
                
                # 计算在 PCA 坐标系中的坐标和长度
                coords = X @ axes  # local coordinates
                mins = coords.min(axis=0)
                maxs = coords.max(axis=0)
                lengths = maxs - mins  # full edge lengths in each PCA axis
                
                # 选择最接近世界 Z 轴的轴作为垂直轴
                world_z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                align = np.abs(axes.T @ world_z)  # dot(axis_i, world_z)
                i_vert = int(np.argmax(align))
                
                idx = [0, 1, 2]
                idx.remove(i_vert)
                i_a, i_b = idx[0], idx[1]  # the two horizontal axes (order to be decided)
                
                # 在 XY 平面上投影水平轴，保留 yaw 角度
                def xy_norm(v: np.ndarray) -> float:
                    return np.hypot(float(v[0]), float(v[1]))
                
                a = axes[:, i_a]
                b = axes[:, i_b]
                # pick the horizontal axis with stronger XY projection as new_x direction
                if xy_norm(a) < xy_norm(b):
                    i_a, i_b = i_b, i_a
                    a, b = b, a
                
                new_z = world_z
                a_xy = np.array([a[0], a[1], 0.0], dtype=np.float32)
                n = np.linalg.norm(a_xy)
                
                if n < 1e-8:
                    b_xy = np.array([b[0], b[1], 0.0], dtype=np.float32)
                    n2 = np.linalg.norm(b_xy)
                    if n2 < 1e-8:
                        new_x = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                    else:
                        new_x = b_xy / n2
                        i_a, i_b = i_b, i_a
                else:
                    new_x = a_xy / n
                
                new_y = np.cross(new_z, new_x)
                ny = np.linalg.norm(new_y)
                if ny < 1e-8:
                    new_x = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                    new_y = np.cross(new_z, new_x)
                    ny = np.linalg.norm(new_y)
                new_y = new_y / ny
                new_x = np.cross(new_y, new_z)  # re-orthonormalize
                
                # 计算新的 extents
                Lx = float(lengths[i_a])
                Ly = float(lengths[i_b])
                Lz = float(lengths[i_vert])
                
                # 构建新的旋转矩阵和 transform
                new_rot_matrix = np.column_stack([new_x, new_y, new_z]).astype(np.float32)
                new_center = c.astype(np.float32)
                
                # 更新 transform 和 extents
                transform[:3, :3] = new_rot_matrix
                transform[:3, 3] = new_center
                extents = np.array([Lx, Ly, Lz], dtype=np.float32)
                center = new_center
                rot_matrix = new_rot_matrix
                
            except Exception as e:
                logging.warning("无法处理 OBB (label=%s, ins_id=%s): %s，保持原样", label, ins_id, e)
        elif process_obb:
            # 计算原始 OBB 的 8 个角点（在 OBB 局部坐标系中）
            # OBB 局部坐标系中，角点位于 ±extents/2 的位置
            half_extents = extents * 0.5
            local_corners = np.array([
                [-half_extents[0], -half_extents[1], -half_extents[2]],
                [ half_extents[0], -half_extents[1], -half_extents[2]],
                [-half_extents[0],  half_extents[1], -half_extents[2]],
                [ half_extents[0],  half_extents[1], -half_extents[2]],
                [-half_extents[0], -half_extents[1],  half_extents[2]],
                [ half_extents[0], -half_extents[1],  half_extents[2]],
                [-half_extents[0],  half_extents[1],  half_extents[2]],
                [ half_extents[0],  half_extents[1],  half_extents[2]],
            ], dtype=np.float32)
            
            # 将角点转换到世界坐标系
            original_world_corners = (rot_matrix @ local_corners.T).T + center
            
            # 根据场景类型选择处理方式
            if scene_type in ("scannet", "scannetppv2"):
                # ScanNet/ScanNet++ v2: 使用 PCA 方法创建一个有一个面平行于 xy 平面的新 OBB
                # 完全按照参考代码的 upright_box 方法
                try:
                    # 1. 使用 PCA 找到主轴
                    c = original_world_corners.mean(axis=0)
                    X = original_world_corners - c
                    cov = (X.T @ X) / X.shape[0]
                    w, v = np.linalg.eigh(cov)  # v: columns are eigenvectors
                    order = np.argsort(w)[::-1]
                    axes = v[:, order]  # columns are local axes in world
                    # make right-handed
                    if np.linalg.det(axes) < 0:
                        axes[:, 2] *= -1.0
                    
                    # 2. 计算在 PCA 坐标系中的坐标和长度
                    coords = X @ axes  # local coordinates
                    mins = coords.min(axis=0)
                    maxs = coords.max(axis=0)
                    lengths = maxs - mins  # full edge lengths in each PCA axis
                    
                    # 3. 选择最接近世界 Z 轴的轴作为垂直轴
                    world_z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                    align = np.abs(axes.T @ world_z)  # dot(axis_i, world_z)
                    i_vert = int(np.argmax(align))
                    
                    idx = [0, 1, 2]
                    idx.remove(i_vert)
                    i_a, i_b = idx[0], idx[1]  # the two horizontal axes (order to be decided)
                    
                    # 4. 在 XY 平面上投影水平轴，保留 yaw 角度
                    def xy_norm(v: np.ndarray) -> float:
                        return np.hypot(float(v[0]), float(v[1]))
                    
                    a = axes[:, i_a]
                    b = axes[:, i_b]
                    # pick the horizontal axis with stronger XY projection as new_x direction
                    if xy_norm(a) < xy_norm(b):
                        i_a, i_b = i_b, i_a
                        a, b = b, a
                    
                    new_z = world_z
                    a_xy = np.array([a[0], a[1], 0.0], dtype=np.float32)
                    n = np.linalg.norm(a_xy)
                    
                    if n < 1e-8:
                        b_xy = np.array([b[0], b[1], 0.0], dtype=np.float32)
                        n2 = np.linalg.norm(b_xy)
                        if n2 < 1e-8:
                            new_x = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                        else:
                            new_x = b_xy / n2
                            i_a, i_b = i_b, i_a
                    else:
                        new_x = a_xy / n
                    
                    new_y = np.cross(new_z, new_x)
                    ny = np.linalg.norm(new_y)
                    if ny < 1e-8:
                        new_x = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                        new_y = np.cross(new_z, new_x)
                        ny = np.linalg.norm(new_y)
                    new_y = new_y / ny
                    new_x = np.cross(new_y, new_z)  # re-orthonormalize
                    
                    # 5. 计算新的 extents
                    Lx = float(lengths[i_a])
                    Ly = float(lengths[i_b])
                    Lz = float(lengths[i_vert])
                    
                    # 6. 构建新的旋转矩阵和 transform
                    new_rot_matrix = np.column_stack([new_x, new_y, new_z]).astype(np.float32)
                    new_center = c.astype(np.float32)
                    
                    # 更新 transform 和 extents
                    transform[:3, :3] = new_rot_matrix
                    transform[:3, 3] = new_center
                    extents = np.array([Lx, Ly, Lz], dtype=np.float32)
                    center = new_center
                    rot_matrix = new_rot_matrix
                    
                except Exception as e:
                    logging.warning("无法处理 OBB (label=%s, ins_id=%s): %s，保持原样", label, ins_id, e)
            else:
                # DL3DV: 在相机坐标系中只保留 y 轴旋转
                if ref_c2w is not None:
                    # OBB 的旋转矩阵 R_world 表示从 OBB 局部坐标系到世界坐标系的旋转
                    # 要将这个旋转转换到相机坐标系：
                    # 如果一个向量在 OBB 局部坐标系中是 v_local，在世界坐标系中是 v_world = R_world @ v_local
                    # 在相机坐标系中是 v_cam = R_w2c @ v_world = R_w2c @ R_world @ v_local
                    # 所以，在相机坐标系中，OBB 的旋转矩阵是：R_cam = R_w2c @ R_world
                    cam_rot_matrix = ref_w2c_rot @ rot_matrix
                    
                    # 在相机坐标系中提取 y 轴旋转（只保留绕 y 轴的旋转）
                    try:
                        cam_euler = R.from_matrix(cam_rot_matrix).as_euler('xyz', degrees=False)
                        # 只保留 y 轴旋转（绕 y 轴旋转），x 和 z 轴旋转设为 0
                        new_cam_euler = np.array([0.0, cam_euler[1], 0.0])
                        new_cam_rot_matrix = R.from_euler('xyz', new_cam_euler, degrees=False).as_matrix().astype(np.float32)
                    except Exception as e:
                        logging.warning("无法提取相机坐标系 y 轴旋转 (label=%s, ins_id=%s): %s，使用单位矩阵", label, ins_id, e)
                        new_cam_rot_matrix = np.eye(3, dtype=np.float32)
                    
                    # 将相机坐标系的旋转矩阵转换回世界坐标系
                    # 我们需要找到 R_world_new，使得在相机坐标系中 OBB 的旋转是 R_cam_new
                    # 即：R_cam_new = R_w2c @ R_world_new
                    # 所以：R_world_new = R_w2c^T @ R_cam_new = R_c2w @ R_cam_new
                    new_rot_matrix = ref_w2c_rot.T @ new_cam_rot_matrix
                    
                    # 将原始 8 个角点转换到新的 OBB 局部坐标系
                    new_center = np.mean(original_world_corners, axis=0)
                    new_local_corners = (new_rot_matrix.T @ (original_world_corners - new_center).T).T
                    
                    # 在新局部坐标系中计算 AABB
                    min_corner = np.min(new_local_corners, axis=0)
                    max_corner = np.max(new_local_corners, axis=0)
                    new_extents = (max_corner - min_corner).astype(np.float32)
                    local_center_offset = (min_corner + max_corner) * 0.5
                    new_center = new_center + new_rot_matrix @ local_center_offset
                    
                    # 更新 transform 和 extents
                    transform[:3, :3] = new_rot_matrix
                    transform[:3, 3] = new_center
                    extents = new_extents
                    center = new_center
                    rot_matrix = new_rot_matrix
                else:
                    # 如果没有参考相机，回退到世界坐标系的处理
                    try:
                        euler = R.from_matrix(rot_matrix).as_euler('xyz', degrees=False)
                        new_euler = np.array([0.0, euler[1], 0.0])
                        new_rot_matrix = R.from_euler('xyz', new_euler, degrees=False).as_matrix().astype(np.float32)
                        
                        # 将原始 8 个角点转换到新的 OBB 局部坐标系
                        new_center = np.mean(original_world_corners, axis=0)
                        new_local_corners = (new_rot_matrix.T @ (original_world_corners - new_center).T).T
                        
                        # 在新局部坐标系中计算 AABB
                        min_corner = np.min(new_local_corners, axis=0)
                        max_corner = np.max(new_local_corners, axis=0)
                        new_extents = (max_corner - min_corner).astype(np.float32)
                        local_center_offset = (min_corner + max_corner) * 0.5
                        new_center = new_center + new_rot_matrix @ local_center_offset
                        
                        # 更新 transform 和 extents
                        transform[:3, :3] = new_rot_matrix
                        transform[:3, 3] = new_center
                        extents = new_extents
                        center = new_center
                        rot_matrix = new_rot_matrix
                    except Exception as e:
                        logging.warning("无法提取 y 轴旋转 (label=%s, ins_id=%s): %s，使用单位矩阵", label, ins_id, e)
                        new_rot_matrix = np.eye(3, dtype=np.float32)
        
        # 计算最终的 8 个角点（用于可视化）
        # 完全按照参考代码的方式生成角点
        hx, hy, hz = extents[0] / 2.0, extents[1] / 2.0, extents[2] / 2.0
        
        # canonical order: bottom 4 then top 4 (与参考代码一致)
        # 确保底面4个点z值相同，顶面4个点z值相同，且对应点在同一z轴连线上
        # 底面（z = -hz）: 0, 1, 2, 3
        # 顶面（z = +hz）: 4, 5, 6, 7
        # 对应关系：0-4, 1-5, 2-6, 3-7 在同一z轴连线上
        signs = [
            (-1, -1, -1),  # 0: 底面左下
            (+1, -1, -1),  # 1: 底面右下
            (-1, +1, -1),  # 2: 底面左上
            (+1, +1, -1),  # 3: 底面右上
            (-1, -1, +1),  # 4: 顶面左下（与0在同一z轴连线）
            (+1, -1, +1),  # 5: 顶面右下（与1在同一z轴连线）
            (-1, +1, +1),  # 6: 顶面左上（与2在同一z轴连线）
            (+1, +1, +1),  # 7: 顶面右上（与3在同一z轴连线）
        ]
        
        # 使用旋转矩阵的列向量作为新的轴方向
        new_x = rot_matrix[:, 0]
        new_y = rot_matrix[:, 1]
        new_z = rot_matrix[:, 2]
        
        # 按照参考代码的方式生成角点
        final_world_corners = np.array(
            [center + sx * hx * new_x + sy * hy * new_y + sz * hz * new_z for sx, sy, sz in signs],
            dtype=np.float32,
        )
        
        # 转换为四元数用于 rerun（用于 Boxes3D 作为备用）
        try:
            quat_xyzw = R.from_matrix(rot_matrix).as_quat().astype(np.float32)
        except Exception as e:
            logging.warning("无法转换为四元数 (label=%s, ins_id=%s): %s，使用单位四元数", label, ins_id, e)
            quat_xyzw = np.array([0, 0, 0, 1], dtype=np.float32)
        
        # 使用处理后的 8 个角点进行可视化
        # 定义 OBB 的 12 条边（连接 8 个角点）
        # 每个 OBB 有 12 条边：4 条垂直边 + 4 条底边 + 4 条顶边
        edge_indices = [
            [0, 1], [2, 3], [4, 5], [6, 7],  # 垂直边
            [0, 2], [1, 3], [4, 6], [5, 7],  # 底边和顶边
            [0, 4], [1, 5], [2, 6], [3, 7],  # 连接底和顶的边
        ]
        
        # 构建所有边的点序列
        edge_strips = [final_world_corners[edge_idx] for edge_idx in edge_indices]
        
        # 使用 LineStrips3D 可视化 OBB 的边（基于处理后的 8 个角点）
        rr.log(
            f"world/instances/{label}/{ins_id}/bbox/edges",
            rr.LineStrips3D(
                strips=edge_strips,
                colors=col,
                radii=0.01,
            ),
        )
        
        # 同时保留 Boxes3D 可视化（用于更好的显示效果）
        rr.log(
            f"world/instances/{label}/{ins_id}/bbox",
            rr.Boxes3D(
                centers=np.array([center], dtype=np.float32),
                half_sizes=np.array([extents * 0.5], dtype=np.float32),
                quaternions=[rr.Quaternion(xyzw=quat_xyzw)],
                colors=np.array([col], dtype=np.uint8),
                labels=[label_text] if show_labels else None,
            ),
        )

        if not show_labels:
            # 使用 Text3D 显示标签（仅文字，透明背景）
            label_pos = center + np.array([0.0, extents[1] * 0.5 + 0.1, 0.0], dtype=np.float32)
            try:
                rr.log(
                    f"world/instances/{label}/{ins_id}/label",
                    rr.Text3D(
                        text=label_text,
                        position=label_pos,
                        color=col,
                    ),
                    static=True,
                )
            except Exception:
                # 如果 Text3D 不可用，则不显示标签
                pass

    # 从 bboxes 中提取图像名称，只保存这些图像对应的相机
    bbox_image_names = extract_image_names_from_bboxes(bboxes)
    
    # 可视化用到的相机位置和姿态（只保存 JSON 中 images 出现过的相机）
    # 过滤出在 bboxes images 中出现的相机
    used_cameras = []
    for stem in color_map.keys():
        # 检查 stem 是否在 bbox_image_names 中
        if stem in bbox_image_names:
            used_cameras.append(stem)
        else:
            # 也检查大小写变体（例如 DSC05827 vs dsc05827）
            stem_upper = stem.upper()
            stem_lower = stem.lower()
            if any(name.upper() == stem_upper or name.lower() == stem_lower for name in bbox_image_names):
                used_cameras.append(stem)
    
    if used_cameras:
        logging.info("保存 %d 个相机（从 bboxes images 中提取）", len(used_cameras))
        # 为每个相机记录完整的变换和内参
        for idx, stem in enumerate(used_cameras):
            if stem not in c2w_map or stem not in intr_map:
                continue
            
            c2w = c2w_map[stem]
            K = intr_map[stem]
            
            # 修正旋转矩阵（确保是有效的旋转矩阵）
            rot_matrix = c2w[:3, :3].copy()
            try:
                U, _, Vt = np.linalg.svd(rot_matrix)
                rot_matrix = U @ Vt
                if np.linalg.det(rot_matrix) < 0:
                    U[:, -1] *= -1
                    rot_matrix = U @ Vt
            except Exception:
                # 如果 SVD 失败，使用单位矩阵
                rot_matrix = np.eye(3, dtype=np.float32)
            
            # 为每个相机使用唯一的实体路径，放在 world 路径下与其他元素保持一致
            cam_entity = f"world/cameras/{idx:04d}"
            
            # 记录相机位姿（使用 Transform3D，static=True）
            rr.log(
                cam_entity,
                rr.Transform3D(
                    translation=c2w[:3, 3],
                    mat3x3=rot_matrix,
                ),
                static=True,
            )
            
            # 加载并记录图像（仿照 json_visulize_anchor.py 的方式）
            # 先加载图像以确定下采样后的尺寸
            downscale_factor = 2  # 宽高各缩小2倍，总面积缩小4倍
            img_resized = None
            if path_map is not None and stem in path_map:
                image_path = path_map[stem]
                if image_path and os.path.exists(image_path):
                    try:
                        img = Image.open(image_path)
                        # 下采样4倍（宽高各缩小2倍）
                        original_size = img.size
                        new_size = (original_size[0] // downscale_factor, original_size[1] // downscale_factor)
                        img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
                        img_array = np.array(img_resized)
                        rr.log(cam_entity, rr.Image(img_array), static=True)
                        logging.debug("已记录相机 %d/%d 的图像: %s (原始尺寸: %s, 下采样后: %s)", 
                                     idx + 1, len(used_cameras), stem, original_size, new_size)
                    except Exception as e:
                        logging.warning("加载图像失败 %s: %s", image_path, e)
                else:
                    if image_path:
                        logging.warning("图像未找到: %s", image_path)
            
            # 记录相机内参（Pinhole）- 如果提供了 size_map
            # 内参需要根据图像下采样进行调整
            if size_map is not None and stem in size_map:
                fx, fy = K[0, 0], K[1, 1]
                cx, cy = K[0, 2], K[1, 2]
                width, height = size_map[stem]
                
                # 如果图像被下采样了，内参也需要相应调整
                if img_resized is not None:
                    fx = fx / downscale_factor
                    fy = fy / downscale_factor
                    cx = cx / downscale_factor
                    cy = cy / downscale_factor
                    width = width // downscale_factor
                    height = height // downscale_factor
                
                rr.log(
                    cam_entity,
                    rr.Pinhole(
                        resolution=[width, height],
                        focal_length=[fx, fy],
                        principal_point=[cx, cy],
                    ),
                    static=True,
                )
        
        logging.info("已记录 %d 个相机位置和姿态", len(used_cameras))

    if rerun_save_path:
        rr.save(str(rerun_save_path))
        logging.info("rerun 日志已保存到: %s", rerun_save_path)

def process_scene(
    input_json_path: Path,
    data_root: Path,
    model_path: Path,
    iteration: int,
    output_dir: Path,
    z_min: Optional[float] = None,
    selected_images: Optional[List[str]] = None,
    output_tag: Optional[str] = None,
    desc_root: Optional[Path] = None,
    show_labels: bool = True,
    include_desc: bool = True,
    process_obb: bool = False,  # 是否处理 OBB（在相机坐标系中只保留 y 轴旋转，默认关闭）
) -> None:
    # 1. 解析场景名
    scene = input_json_path.stem
    scene_dir = data_root / scene
    output_dir.mkdir(parents=True, exist_ok=True)
    

    print(f"scene_dir: {scene_dir}")
    # 2. 加载位姿与内参
    intr_map, c2w_map, scene_type, size_map, path_map = load_transforms(scene_dir)
    
    # 3. 加载 BBox JSON
    with input_json_path.open("r", encoding="utf-8") as f:
        bboxes = json.load(f)
    logging.info("加载了 %d 个 3D bounding boxes", len(bboxes))
    if desc_root is not None:
        desc_path = desc_root / f"{scene}.json"
        bboxes = merge_descriptions(bboxes, desc_path)

    selected_image_stems: Optional[set[str]] = None
    if selected_images:
        selected_image_stems = {normalize_image_name(name) for name in selected_images}
        logging.info("指定图像数量: %d (%s)", len(selected_image_stems), sorted(list(selected_image_stems))[:10])

        # 仅保留包含指定图像的 bbox
        filtered_bboxes = []
        for box in bboxes:
            image_stems = extract_image_stems_from_paths(box.get("images", []))
            if image_stems & selected_image_stems:
                filtered_bboxes.append(box)
        bboxes = filtered_bboxes
        logging.info("筛选后保留 %d 个 3D bounding boxes", len(bboxes))
    
    # 4. 确定要渲染的 100 张图
    discovery_json_path = Path("scene_objects_Qwen3-VL-30B-A3B-Instruct") / f"{scene}.json"
    image_names = None
    if selected_image_stems:
        image_names = set(selected_image_stems)
        logging.info("使用指定图像进行渲染，数量: %d", len(image_names))
    elif discovery_json_path.exists():
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
    
    # 6. 生成全局点云（直接投射，不做多视图过滤）
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
    
    # z 轴过滤（删除 z < z_min 的点）
    if z_min is not None:
        valid_mask = pts_all[:, 2] >= z_min
        original_count = pts_all.shape[0]
        pts_all = pts_all[valid_mask]
        if cols_all is not None:
            cols_all = cols_all[valid_mask]
        filtered_count = pts_all.shape[0]
        logging.info("z 轴过滤 (z_min=%.2f): %d -> %d 点", z_min, original_count, filtered_count)
    
    # 下采样
    if pts_all.shape[0] > MAX_POINT_COUNT:
        idx = np.random.choice(pts_all.shape[0], MAX_POINT_COUNT, replace=False)
        pts_all = pts_all[idx]
        if cols_all is not None:
            cols_all = cols_all[idx]
        logging.info("点云下采样: -> %d", pts_all.shape[0])

    # 7. 保存 PLY / 记录随机选择信息（如有）
    name_suffix = f"_{output_tag}" if output_tag else ""
    base_name = f"{scene}{name_suffix}"
    ply_save_path = output_dir / f"{base_name}_points.ply"
    pc = trimesh.points.PointCloud(pts_all, colors=cols_all)
    pc.export(ply_save_path)
    logging.info("点云已保存至: %s", ply_save_path)
    
    # 8. 可视化与 RRD 保存
    rrd_save_path = output_dir / f"{base_name}.rrd"
    if selected_images is not None:
        selected_save_path = output_dir / f"{base_name}_images.json"
        with selected_save_path.open("w", encoding="utf-8") as f:
            json.dump(sorted([normalize_image_name(x) for x in selected_images]), f, ensure_ascii=False, indent=2)
        logging.info("已保存图像列表: %s", selected_save_path)
    log_rerun_scene(
        pts_all,
        cols_all,
        bboxes,
        c2w_map,
        intr_map,
        color_map,
        scene,
        scene_type,
        rrd_save_path,
        size_map,
        path_map,
        show_labels,
        include_desc,
        process_obb=process_obb,
    )

def get_candidate_image_stems(input_json_path: Path, data_root: Path) -> List[str]:
    scene = input_json_path.stem
    scene_dir = data_root / scene
    intr_map, c2w_map, scene_type, size_map, path_map = load_transforms(scene_dir)
    with input_json_path.open("r", encoding="utf-8") as f:
        bboxes = json.load(f)
    available = set(c2w_map.keys())
    bbox_stems: set[str] = set()
    for box in bboxes:
        bbox_stems.update(extract_image_stems_from_paths(box.get("images", [])))
    if bbox_stems:
        candidate = sorted(available & bbox_stems)
        if not candidate:
            candidate = sorted(available)
    else:
        candidate = sorted(available)
    logging.info("候选图像数量: %d", len(candidate))
    return candidate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", type=Path, help="输入 BBox JSON 路径，例如 output_3d_bounding/0a184cf634.json")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--model-path", "-m", type=Path, required=True, help="3DGS 模型路径")
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--output-dir", type=Path, default=Path("rerun_output_demo_ours_scannet"))
    parser.add_argument("--z-min", type=float, default=None, help="点云 z 轴最小值过滤，删除 z < z_min 的点")
    parser.add_argument(
        "--images",
        nargs="*",
        default=None,
        help="仅使用指定图像名（可含扩展名），例如: DSC06102.jpg DSC06106.jpg",
    )
    parser.add_argument("--random-runs", type=int, default=0, help="随机选择图像运行的次数（>0 时忽略 --images）")
    parser.add_argument("--random-min", type=int, default=2, help="每次随机选择图像的最小数量")
    parser.add_argument("--random-max", type=int, default=3, help="每次随机选择图像的最大数量")
    # parser.add_argument("--seed", type=int, default=None, help="随机种子（可选）")
    parser.add_argument("--seed", type=int, default=32, help="随机种子（可选）")
    parser.add_argument(
        "--desc-root",
        type=Path,
        default=DEFAULT_DESC_ROOT,
        help="带 description 的 bbox JSON 根目录",
    )
    parser.add_argument(
        "--show-labels",
        dest="show_labels",
        action="store_true",
        default=True,
        help="在 Boxes3D 上显示标签（默认开启）",
    )
    parser.add_argument(
        "--no-show-labels",
        dest="show_labels",
        action="store_false",
        help="关闭 Boxes3D 标签显示（改用 Text3D）",
    )
    parser.add_argument(
        "--include-desc",
        dest="include_desc",
        action="store_true",
        default=True,
        help="label 中包含 description（默认开启）",
    )
    parser.add_argument(
        "--no-include-desc",
        dest="include_desc",
        action="store_false",
        help="label 仅显示物体名字（不含 description）",
    )
    parser.add_argument(
        "--process-obb",
        action="store_true",
        default=False,
        help="处理 OBB transform 和 extents（在相机坐标系中只保留 y 轴旋转，默认关闭）",
    )
    args = parser.parse_args()
    
    setup_logger()
    if args.random_runs and args.random_runs > 0:
        if args.images:
            logging.warning("已设置 --random-runs，忽略 --images")
        if args.random_min <= 0 or args.random_max <= 0:
            logging.error("--random-min/--random-max 需为正数")
            return
        if args.random_min > args.random_max:
            logging.error("--random-min 不能大于 --random-max")
            return
        candidates = get_candidate_image_stems(args.input_json, args.data_root)
        if not candidates:
            logging.error("未找到可用图像候选，无法随机选择")
            return
        relations = load_multi_view_relations(args.input_json.stem, args.model_path)
        rng = random.Random(args.seed)
        for run_idx in range(args.random_runs):
            k = rng.randint(args.random_min, args.random_max)
            k = min(k, len(candidates))
            selected = pick_related_images(rng, candidates, relations, k)
            tag = f"r{run_idx + 1:02d}_k{k}"
            logging.info("随机选择第 %d/%d 次: %s", run_idx + 1, args.random_runs, selected)
            process_scene(
                args.input_json,
                args.data_root,
                args.model_path,
                args.iteration,
                args.output_dir,
                args.z_min,
                selected,
                tag,
                args.desc_root,
                args.show_labels,
                args.include_desc,
                args.process_obb,
            )
        return

    process_scene(
        args.input_json,
        args.data_root,
        args.model_path,
        args.iteration,
        args.output_dir,
        args.z_min,
        args.images,
        None,
        args.desc_root,
        args.show_labels,
        args.include_desc,
        args.process_obb,
    )

if __name__ == "__main__":
    main()

