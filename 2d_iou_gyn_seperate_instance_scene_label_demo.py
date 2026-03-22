#!/usr/bin/env python3
"""
计算2D IoU：将3D点云投影到2D图像，生成mask，并与GT mask比较
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
from PIL import Image
import torch
from scipy.ndimage import binary_fill_holes
from tqdm import tqdm
import pycocotools.mask as mask_utils
import rerun.dataframe as rr_df


ENABLE_PURE_SAM3 = True

# 导入 SAM3
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# 导入 3d_bounding_instance_gs_rerun.py 中的深度渲染函数
from importlib import import_module
import importlib.util

sys.path.insert(0, str(Path(__file__).parent))

spec = importlib.util.spec_from_file_location(
    "bounding_module", 
    Path(__file__).parent / "3d_bounding_instance_gs_rerun.py"
)
bounding_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bounding_module)

render_depths_with_gaussians = bounding_module.render_depths_with_gaussians


def setup_logger():
    """设置日志"""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def decode_rle_mask(rle_dict: Dict) -> np.ndarray:
    """解码COCO RLE格式的mask，参考 3d_bounding_instance_gs_rerun.py 的实现"""
    # 直接使用 mask_utils.decode，参考 3d_bounding_instance_gs_rerun.py:946-950
    rle = rle_dict  # rle_dict 本身就是 RLE 格式
    mask = mask_utils.decode(rle).astype(bool)
    return mask.astype(np.uint8) * 255  # 转换为 uint8 格式 (0 或 255)


def load_gt_masks(gt_json_path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    """
    加载GT masks
    返回: {image_name: {label: mask_array}}
    """
    with gt_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    gt_masks = {}
    for scene_data in data:
        scene_name = scene_data.get("scene", "")
        items = scene_data.get("items", [])
        
        for item in items:
            image_name = item.get("image", "")
            label = item.get("label", "")
            mask_rle = item.get("mask_rle", {})
            
            if not image_name or not label or not mask_rle:
                continue
            
            # 跳过背景
            if label == "__background__":
                continue
            
            if image_name not in gt_masks:
                gt_masks[image_name] = {}
            
            try:
                mask = decode_rle_mask(mask_rle)
                gt_masks[image_name][label] = mask
            except Exception as e:
                logging.warning(f"解码mask失败 {image_name}/{label}: {e}")
                continue
    
    return gt_masks


def load_scene_objects_labels(scene_objects_json_path: Path) -> Dict[str, List[str]]:
    """
    加载 scene_objects JSON 文件中的每张图片的 label 列表
    返回: {image_name: [label1, label2, ...]}
    """
    with scene_objects_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    per_image_labels = {}
    per_image = data.get("per_image", {})
    
    for image_name, labels in per_image.items():
        if isinstance(labels, list):
            per_image_labels[image_name] = labels
        else:
            per_image_labels[image_name] = []
    
    return per_image_labels


def get_instance_points(rec, label: str, ins_id: str):
    ins_id = str(ins_id)

    # 生成多种可能的路径格式，以处理带空格的 label（如 "toilet paper"）
    # rerun 路径中的空格可能需要转义或替换
    label_variants = [
        label,  # 原始 label（带空格）
        label.replace(" ", "\\ "),  # 转义空格（rerun 可能使用这种格式）
        label.replace(" ", "_"),  # 下划线替换空格
        label.replace(" ", "-"),  # 连字符替换空格
    ]
    
    path_candidates = []
    for label_var in label_variants:
        path_candidates.extend([
            f"/instances/{label_var}/{ins_id}/points",
            f"instances/{label_var}/{ins_id}/points",
        ])

    # 至少把 log_tick 加进去；如果你 rrd 用了自定义时间轴（如 frame_nr），也要加进来
    index_candidates = ["log_time", "log_tick"]

    df = None
    for index_type in index_candidates:
        for entity_path in path_candidates:
            try:
                view = rec.view(index=index_type, contents=entity_path)
                data = view.select()
                tmp = data.read_pandas()
                if tmp is not None and not tmp.empty:
                    df = tmp
                    break
            except Exception:
                pass
        if df is not None:
            break

    if df is None or df.empty:
        return None

        
    # Find column that ends with positions (使用与 3d_2d.py 相同的方式)
    col = [c for c in df.columns if "positions" in c]
    if not col:
        return None
    val = df[col[0]].iloc[-1]
    
    if isinstance(val, np.ndarray) and val.dtype == object:
        # It's an array of arrays, stack them
        pts = np.stack(val)
    else:
        pts = np.array(val)
        
    if pts.ndim == 1:
        if pts.size % 3 == 0 and pts.size > 0:
            pts = pts.reshape(-1, 3)
        elif pts.size == 3:
            pts = pts.reshape(1, 3)
    return pts



def load_label_points_from_rrd(rrd_path: Path, instances_json: List[Dict]) -> Dict[str, List[np.ndarray]]:
    """
    从 rrd 文件读取每个 label 下的所有 instance 点云（不合并）
    返回: {label: [points_per_instance]}
    """
    try:
        archive = rr_df.load_archive(str(rrd_path))
        rec = archive.all_recordings()[0]
    except Exception as e:
        raise RuntimeError(f"无法加载 RRD 文件: {e}")
    
    label_points_dict: Dict[str, List[np.ndarray]] = {}
    
    for inst in instances_json:
        label = inst.get("label", "unknown")
        ins_id = inst.get("ins_id", "unknown")
        
        if label == "unknown" or ins_id == "unknown":
            continue
        
        # 确保 ins_id 是字符串（在调用函数前转换，而不是在函数内）
        ins_id = str(ins_id)
        # print(f"label: {label}, ins_id: {ins_id} (type: {type(ins_id).__name__})")
        
        points = get_instance_points(rec, label, ins_id)
        if points is None or points.size == 0:
            print(f"points is None or points.size == 0: {points}")
            continue
        
        if points.ndim != 2 or points.shape[1] != 3:
            print(f"points.ndim != 2 or points.shape[1] != 3: {points.ndim} {points.shape[1]}")
            continue
        
        
        if label not in label_points_dict:
            label_points_dict[label] = []
        label_points_dict[label].append(points)
    
    for label, points_list in label_points_dict.items():
        total_pts = sum(p.shape[0] for p in points_list)
        print(f"标签 {label} 共 {len(points_list)} 个 instance，合计 {total_pts} 个点")
        logging.info(f"标签 {label} 共 {len(points_list)} 个 instance，合计 {total_pts} 个点")
    
    return label_points_dict


def load_camera_params(scene_dir: Path):
    """
    加载相机参数，参考 3d_2d.py 的 load_camera_params 函数
    支持 scannetppv2 (dslr), scannet (cam/color), dl3dv (dense) 三种格式
    """
    # scannetppv2 格式
    if (scene_dir / "dslr").exists():
        json_path = scene_dir / "dslr" / "nerfstudio" / "transforms_undistorted.json"
        if not json_path.exists():
            logging.warning(f"Warning: {json_path} not found")
            return None, None, None
            
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        fl_x = data["fl_x"]
        fl_y = data["fl_y"]
        cx = data["cx"]
        cy = data["cy"]
        w = data.get("w", 1920)
        h = data.get("h", 1080)
        
        intr_map = {}
        c2w_map = {}
        size_map = {}
        
        frames = data.get("frames", []) + data.get("test_frames", [])
        
        for frame in frames:
            stem = Path(frame["file_path"]).stem
            c2w = np.array(frame["transform_matrix"], dtype=np.float32)
            # OpenGL (nerfstudio) to OpenCV/COLMAP
            c2w[:3, 1:3] *= -1
            
            K = np.array([
                [fl_x, 0, cx],
                [0, fl_y, cy],
                [0, 0, 1]
            ], dtype=np.float32)
            
            frame_w = frame.get("w", w)
            frame_h = frame.get("h", h)
            
            intr_map[stem] = K
            c2w_map[stem] = c2w
            size_map[stem] = (frame_w, frame_h)
        
        logging.info("加载位姿完成，数量: %d", len(intr_map))
        return intr_map, c2w_map, size_map
    
    # scannetv2 / ScanNet 官方结构:
    #   scene0000_00/
    #     intrinsic/*.txt  (如 intrinsic_color.txt，4x4 矩阵，前 3x3 为 K)
    #     pose/*.txt       (每帧 4x4 相机位姿矩阵)
    #     color/*.jpg/.png (每帧彩色图)
    elif (scene_dir / "intrinsic").exists() and (scene_dir / "pose").exists() and (scene_dir / "color").exists():
        intr_dir = scene_dir / "intrinsic"
        pose_dir = scene_dir / "pose"
        color_dir = scene_dir / "color"

        # 读取一次内参矩阵（对 ScanNet 来说所有帧共享同一个 intrinsic_color）
        intr_file = None
        if (intr_dir / "intrinsic_color.txt").exists():
            intr_file = intr_dir / "intrinsic_color.txt"
        elif (intr_dir / "intrinsic_depth.txt").exists():
            intr_file = intr_dir / "intrinsic_depth.txt"
        else:
            txts = sorted([f for f in os.listdir(intr_dir) if f.endswith(".txt")])
            if txts:
                intr_file = intr_dir / txts[0]

        if intr_file is None:
            raise FileNotFoundError(f"未在 {intr_dir} 找到任何 intrinsic *.txt 文件")

        intrinsic_mat = np.loadtxt(str(intr_file))
        if intrinsic_mat.shape[0] >= 3 and intrinsic_mat.shape[1] >= 3:
            K_full = intrinsic_mat[:3, :3]
        else:
            raise ValueError(f"内参文件格式异常: {intr_file}, shape={intrinsic_mat.shape}")

        fx = float(K_full[0, 0])
        fy = float(K_full[1, 1])
        cx = float(K_full[0, 2])
        cy = float(K_full[1, 2])
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        pose_files = sorted([f for f in os.listdir(pose_dir) if f.endswith(".txt")])
        intr_map: Dict[str, np.ndarray] = {}
        c2w_map: Dict[str, np.ndarray] = {}
        size_map: Dict[str, Tuple[int, int]] = {}

        for idx, pose_file in tqdm(
            list(enumerate(pose_files)),
            total=len(pose_files),
            desc="加载 ScanNetv2 相机数据",
        ):
            pose_path = pose_dir / pose_file
            pose_mat = np.loadtxt(str(pose_path))
            if pose_mat.shape != (4, 4):
                logging.warning(f"位姿矩阵尺寸异常 {pose_path}, shape={pose_mat.shape}，跳过")
                continue

            stem = Path(pose_file).stem  # 如 000000 / frame_00020 之类
            c2w = pose_mat.astype(np.float32)

            intr_map[stem] = K
            c2w_map[stem] = c2w

            # 从对应的 color 图像读取尺寸
            img_path = color_dir / f"{stem}.png"
            if not img_path.exists():
                img_path = color_dir / f"{stem}.jpg"
            if not img_path.exists():
                logging.warning(f"未找到图像文件: {stem}.png/jpg，跳过该帧")
                continue
            try:
                with Image.open(img_path) as img:
                    size_map[stem] = img.size  # (width, height)
            except Exception as e:
                logging.warning(f"无法读取图像尺寸 {img_path}: {e}，跳过该帧")
                continue

        logging.info("加载 ScanNetv2 位姿完成，数量: %d", len(intr_map))
        return intr_map, c2w_map, size_map
    
    # dl3dv 格式
    elif (scene_dir / "dense").exists():
        cam_dir = scene_dir / "dense" / "cam"
        rgb_dir = scene_dir / "dense" / "rgb"
        cam_files = sorted([f for f in os.listdir(cam_dir) if f.endswith('.npz')])
        intr_map = {}
        c2w_map = {}
        size_map = {}
        
        for idx, cam_file in tqdm(enumerate(cam_files), total=len(cam_files), desc="加载 DL3DV 相机数据"):
            cam_file_path = cam_dir / cam_file
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
                # 如果没有尺寸，尝试读取图片获取尺寸
                if img_path.exists():
                    try:
                        with Image.open(img_path) as img:
                            size_map[stem] = img.size
                    except Exception as e:
                        logging.warning(f"无法读取图像尺寸 {img_path}: {e}，跳过")
                        continue
                else:
                    logging.warning(f"未找到图像文件: {stem}，跳过")
                    continue
        
        logging.info("加载位姿完成，数量: %d", len(intr_map))
        return intr_map, c2w_map, size_map
    
    else:
        logging.warning(f"不支持的场景格式: {scene_dir}, 目前只支持 scannetppv2 (dslr), scannet (cam/color), dl3dv (dense)")
        return None, None, None


def project_points(points: np.ndarray, K: np.ndarray, c2w: np.ndarray, img_size: Tuple[int, int]) -> np.ndarray:
    """
    将3D点投影到2D图像平面，参考 3d_2d.py 的 project_points 函数
    points: (N, 3) in world coordinates
    c2w: (4, 4) world to camera transform
    返回: mask (height, width) uint8
    """
    # World to camera: pts_cam = inv(c2w) @ pts_world
    w2c = np.linalg.inv(c2w)
    
    # Convert points to homogeneous coordinates
    pts_homo = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    pts_cam_homo = pts_homo @ w2c.T
    pts_cam = pts_cam_homo[:, :3]
    
    # Filter points behind camera (z <= 0)
    mask_z = pts_cam[:, 2] > 0
    pts_cam = pts_cam[mask_z]
    
    if pts_cam.shape[0] == 0:
        return np.zeros((img_size[1], img_size[0]), dtype=np.uint8)
    
    # Project to image plane: p_img = K @ pts_cam
    pts_img_homo = pts_cam @ K.T
    pts_img = pts_img_homo[:, :2] / pts_img_homo[:, 2:3]
    
    u = np.round(pts_img[:, 0]).astype(int)
    v = np.round(pts_img[:, 1]).astype(int)
    
    # Filter points outside image boundaries
    valid_mask = (u >= 0) & (u < img_size[0]) & (v >= 0) & (v < img_size[1])
    u, v = u[valid_mask], v[valid_mask]
    
    mask = np.zeros((img_size[1], img_size[0]), dtype=np.uint8)
    mask[v, u] = 255
    
    return mask


def project_points_with_depth_filter(
    points: np.ndarray, 
    K: np.ndarray, 
    c2w: np.ndarray, 
    img_size: Tuple[int, int],
    depth_map: np.ndarray,
    depth_threshold: float = 0.1
) -> np.ndarray:
    """
    将3D点投影到2D图像平面，并考虑遮挡关系（深度过滤）
    points: (N, 3) in world coordinates
    c2w: (4, 4) world to camera transform
    depth_map: (height, width) 当前视角的深度图
    depth_threshold: 深度容差，只保留在 [depth, depth+depth_threshold] 范围内的点
    返回: mask (height, width) uint8
    """
    # World to camera: pts_cam = inv(c2w) @ pts_world
    w2c = np.linalg.inv(c2w)
    
    # Convert points to homogeneous coordinates
    pts_homo = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    pts_cam_homo = pts_homo @ w2c.T
    pts_cam = pts_cam_homo[:, :3]
    
    # Filter points behind camera (z <= 0)
    mask_z = pts_cam[:, 2] > 0
    pts_cam = pts_cam[mask_z]
    
    if pts_cam.shape[0] == 0:
        return np.zeros((img_size[1], img_size[0]), dtype=np.uint8)
    
    # Project to image plane: p_img = K @ pts_cam
    pts_img_homo = pts_cam @ K.T
    pts_img = pts_img_homo[:, :2] / pts_img_homo[:, 2:3]
    
    u = np.round(pts_img[:, 0]).astype(int)
    v = np.round(pts_img[:, 1]).astype(int)
    
    # Filter points outside image boundaries
    valid_mask = (u >= 0) & (u < img_size[0]) & (v >= 0) & (v < img_size[1])
    u, v = u[valid_mask], v[valid_mask]
    pts_cam = pts_cam[valid_mask]
    
    if len(u) == 0:
        return np.zeros((img_size[1], img_size[0]), dtype=np.uint8)
    
    # 获取对应像素的深度值
    point_depths = pts_cam[:, 2]  # 点在相机坐标系下的深度（z值）
    pixel_depths = depth_map[v, u]  # 深度图在该像素的深度值
    
    # 只保留在 [depth, depth+depth_threshold] 范围内的点（考虑遮挡）
    # 如果点的深度在深度图的深度范围内，说明点没有被遮挡
    depth_valid =  (point_depths <= pixel_depths + depth_threshold)
    
    u = u[depth_valid]
    v = v[depth_valid]
    
    if len(u) == 0:
        return np.zeros((img_size[1], img_size[0]), dtype=np.uint8)
    
    mask = np.zeros((img_size[1], img_size[0]), dtype=np.uint8)
    mask[v, u] = 255
    
    return mask


def create_sparse_mask(
    pixel_coords: np.ndarray,
    labels: np.ndarray,
    label_to_query: str,
    width: int,
    height: int
) -> np.ndarray:
    """
    根据投影的点创建稀疏的2D mask
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # 只保留匹配的label的点
    if labels.dtype == object:
        label_mask = np.array([str(l) == label_to_query for l in labels])
    else:
        label_mask = labels == label_to_query
    
    if not np.any(label_mask):
        return mask
    
    valid_coords = pixel_coords[label_mask]
    
    # 将点绘制到mask上（使用更粗的点以提高覆盖率）
    for coord in valid_coords:
        x, y = int(coord[0]), int(coord[1])
        if 0 <= x < width and 0 <= y < height:
            # 绘制一个小的圆形区域，而不是单个点
            cv2.circle(mask, (x, y), 2, 255, -1)
    
    return mask


def fill_sparse_mask(mask: np.ndarray, method: str = "dilation") -> np.ndarray:
    """
    填充稀疏的mask
    method: "fill_holes" 或 "dilation"
    """
    if method == "fill_holes":
        # 使用 binary_fill_holes 填充空洞
        filled = binary_fill_holes(mask > 0).astype(np.uint8) * 255
    elif method == "dilation":
        # 使用膨胀操作填充稀疏区域
        kernel = np.ones((5, 5), np.uint8)
        filled = cv2.dilate(mask, kernel, iterations=5)
        # 然后再填充空洞
        filled = binary_fill_holes(filled > 0).astype(np.uint8) * 255
        # 使用形态学闭运算平滑边界
        kernel = np.ones((3, 3), np.uint8)
        filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel, iterations=2)
    else:
        filled = mask
    
    return filled

def _sample_points_from_mask(mask: np.ndarray, num_points: int) -> List[Tuple[int, int]]:
    """在mask内部均匀抽样若干点（像素坐标）。"""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return []
    idxs = np.linspace(0, len(xs) - 1, num=min(num_points, len(xs)), dtype=int)
    pts = [(int(xs[i]), int(ys[i])) for i in idxs]
    return pts


def _sample_negative_points(mask: np.ndarray, num_points: int, dilate_size: int = 25) -> List[Tuple[int, int]]:
    """在mask边缘附近采样负点，优先选mask膨胀后的环形区域。"""
    if num_points <= 0:
        return []
    kernel = np.ones((dilate_size, dilate_size), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=1)
    ring = (dilated > 0) & (mask == 0)
    ys, xs = np.where(ring)
    if len(xs) == 0:
        # 如果没有环形区域，就在背景随机采样
        ys, xs = np.where(mask == 0)
    if len(xs) == 0:
        return []
    idxs = np.linspace(0, len(xs) - 1, num=min(num_points, len(xs)), dtype=int)
    pts = [(int(xs[i]), int(ys[i])) for i in idxs]
    return pts


def get_mask_from_sam(
    original_mask: np.ndarray,
    coarse_mask: np.ndarray,
    label_to_query: str,
    image_path: Optional[Path],
    sam3_processor: Optional[Sam3Processor],
    max_positive_points: int = 5,
    max_negative_points: int = 3,
    mask_selection: str = "merge",  # "merge" 加权融合；"best" 取最高分
) -> np.ndarray:
    """
    路线B：粗mask + 少量正负点，引导 SAM3 精修。
    - coarse_mask: 0/255 的粗分割（二值）。
    - 返回 uint8 0/255 的精修mask；失败时回退 coarse_mask。
    """
    # 粗mask用于提取 bbox 与回退
    coarse_mask = original_mask
    fallback_mask = coarse_mask if coarse_mask is not None else original_mask
    if sam3_processor is None:
        logging.warning("SAM3 未加载，直接返回粗mask")
        return fallback_mask
    if image_path is None or (not isinstance(image_path, Path)) or (not image_path.exists()):
        logging.warning("图像不存在，直接返回粗mask")
        return fallback_mask

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        logging.warning(f"读取图像失败 {image_path}: {e}")
        return fallback_mask

    # mask 无有效区域则直接返回
    if np.sum(fallback_mask > 0) == 0:
        return fallback_mask

    # 计算mask的bbox（归一化 cxcywh）
    ys, xs = np.where(fallback_mask > 0)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    w = x_max - x_min + 1
    h = y_max - y_min + 1
    cx = x_min + w / 2.0
    cy = y_min + h / 2.0
    img_w, img_h = image.size
    box_norm = [
        cx / img_w,
        cy / img_h,
        w / img_w,
        h / img_h,
    ]

    # 设置图像
    state = sam3_processor.set_image(image, state={})
    # SAM3 需要 language_features，否则 forward_grounding 会缺键
    try:
        state = sam3_processor.set_text_prompt(label_to_query or "visual", state)
    except Exception as e:
        logging.warning(f"SAM3 文本提示设置失败，回退粗mask: {e}")
        return fallback_mask

    # 构造几何提示：mask + box + points
    prompt = sam3_processor.model._get_dummy_prompt()

    # 使用bbox作为提示（不再传入mask）
    boxes = torch.tensor(box_norm, dtype=torch.float32, device=sam3_processor.device).view(1, 1, 4)
    box_labels = torch.ones((1, 1), dtype=torch.long, device=sam3_processor.device)
    box_mask = torch.zeros((1, 1), dtype=torch.bool, device=sam3_processor.device)
    if not ENABLE_PURE_SAM3:
        prompt.append_boxes(boxes, box_labels, box_mask)


    state["geometric_prompt"] = prompt

    try:
        state = sam3_processor._forward_grounding(state)
    except Exception as e:
        logging.warning(f"SAM3 推理失败，回退粗mask: {e}")
        return fallback_mask

    masks = state.get("masks")
    scores = state.get("scores")
    if masks is None or masks.numel() == 0:
        return fallback_mask

    # 根据策略融合或选择单一 mask
    masks_tensor = masks
    if masks_tensor.ndim == 4 and masks_tensor.shape[1] == 1:
        masks_tensor = masks_tensor[:, 0]
    elif masks_tensor.ndim == 3:
        pass
    else:
        return coarse_mask

    mask_selection = mask_selection.lower() if isinstance(mask_selection, str) else "merge"
    if mask_selection == "best":
        if scores is not None and scores.numel() > 0:
            best_idx = int(torch.argmax(scores).item())
        else:
            best_idx = 0
        refined = masks_tensor[best_idx].detach().cpu().numpy()
        refined_bin = (refined > 0.5).astype(np.uint8) * 255
        return refined_bin
    else:
        # merge: 对所有候选二值化后求并集
        bin_masks = (masks_tensor > 0.5)
        fused = torch.any(bin_masks, dim=0).float()
        refined = fused.detach().cpu().numpy()
        refined_bin = (refined > 0.5).astype(np.uint8) * 255
        return refined_bin


def compute_2d_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    计算两个2D mask的IoU
    基于目前解码的方式：GT mask 是 uint8 格式 (0 或 255)，pred_mask 也是 uint8 格式 (0 或 255)
    """
    # 确保两个mask都是二值格式
    mask1_binary = (mask1 > 0).astype(bool)
    mask2_binary = (mask2 > 0).astype(bool)
    
    # 计算交集和并集
    intersection = np.logical_and(mask1_binary, mask2_binary).sum()
    union = np.logical_or(mask1_binary, mask2_binary).sum()
    
    if union == 0:
        return 0.0
    
    iou = float(intersection) / float(union)
    return iou


def visualize_masks(
    image_path: Path,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    output_path: Path,
    original_output_path: Optional[Path] = None,
):
    """
    可视化预测mask和GT mask
    - 额外保存一张原图（若路径提供）
    - 底图先显著变暗，再用 1.0/1.0 权重叠加 GT/Pred，颜色更突出
    """
    # 加载原始图像
    if image_path.exists():
        image = np.array(Image.open(image_path))
    else:
        # 如果图像不存在，创建黑色背景
        h, w = pred_mask.shape
        image = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 保存原图（可选）
    if original_output_path is not None:
        try:
            Image.fromarray(image).save(original_output_path, "JPEG", quality=90)
        except Exception as e:
            logging.warning(f"保存原图失败 {original_output_path}: {e}")
    
    # 底图变暗（防止 1:1 叠加过曝）
    base = (image.astype(np.float32) * 0.1).astype(np.uint8)
    vis = base.copy()
    
    # 绘制GT mask（绿色）
    gt_overlay = vis.copy()
    gt_overlay[gt_mask > 0] = [0, 255, 0]
    vis = cv2.addWeighted(vis, 1.0, gt_overlay, 1.0, 0)
    
    # 绘制预测mask（红色）
    pred_overlay = vis.copy()
    pred_overlay[pred_mask > 0] = [255, 0, 0]
    vis = cv2.addWeighted(vis, 1.0, pred_overlay, 1.0, 0)
    
    # 保存叠加图
    try:
        Image.fromarray(vis).save(output_path, "JPEG", quality=85)
    except Exception as e:
        logging.warning(f"保存可视化图像失败 {output_path}: {e}")


def process_scene(
    scene: str,
    data_root: Path,
    rrd_path: Path,
    scene_objects_json_path: Optional[Path],
    output_dir: Path,
    model_path: Optional[Path] = None,
    iteration: int = 30000,
    sam3_model=None,
    sam3_processor=None,
    depth_threshold: float = 0.1,
    mask_selection: str = "merge",
    max_images: Optional[int] = None,
    coarse_only: bool = False,
):
    """处理单个场景"""
    logging.info(f"处理场景: {scene}")
    
    # 1. 加载相机参数（使用 3d_2d.py 的方式）
    scene_dir = data_root / scene
    intr_map, c2w_map, size_map = load_camera_params(scene_dir)
    if intr_map is None:
        logging.error(f"无法加载相机参数: {scene_dir}")
        return
    logging.info(f"加载了 {len(intr_map)} 个相机的参数")
    
    # 构建 path_map（用于深度渲染）
    path_map = {}
    # 根据场景类型构建 path_map
    if (scene_dir / "dslr").exists():
        # scannetppv2 格式
        for stem in intr_map.keys():
            image_path = scene_dir / "dslr" / "resized_undistorted_images" / f"{stem}.JPG"
            if not image_path.exists():
                image_path = scene_dir / "dslr" / "resized_undistorted_images" / f"{stem}.jpg"
            path_map[stem] = str(image_path) if image_path.exists() else ""
    elif (scene_dir / "cam").exists() and (scene_dir / "color").exists():
        # 旧 ScanNet 格式（cam/*.npz + color/*.png）
        color_dir = scene_dir / "color"
        for stem in intr_map.keys():
            image_path = color_dir / f"{stem}.png"
            if not image_path.exists():
                image_path = color_dir / f"{stem}.jpg"
            path_map[stem] = str(image_path) if image_path.exists() else ""
    elif (scene_dir / "intrinsic").exists() and (scene_dir / "pose").exists() and (scene_dir / "color").exists():
        # ScanNetv2 官方结构：intrinsic/ + pose/ + color/
        color_dir = scene_dir / "color"
        # 遍历 color 下的所有图片文件，全部加入 path_map
        for fname in sorted(os.listdir(color_dir)):
            if not (fname.endswith(".jpg") or fname.endswith(".png") or fname.endswith(".jpeg")):
                continue
            image_path = color_dir / fname
            stem = Path(fname).stem
            path_map[stem] = str(image_path)
    elif (scene_dir / "dense").exists():
        # dl3dv 格式
        rgb_dir = scene_dir / "dense" / "rgb"
        for stem in intr_map.keys():
            image_path = rgb_dir / f"{stem}.png"
            if not image_path.exists():
                image_path = rgb_dir / f"{stem}.jpg"
            path_map[stem] = str(image_path) if image_path.exists() else ""
    
    # 2. 加载 scene_objects 中的 label，并汇总成“全局标签集合”（所有 per_image 的 label 做并集）
    scene_objects_labels: Optional[Dict[str, List[str]]] = None
    if scene_objects_json_path is not None and scene_objects_json_path.exists():
        try:
            scene_objects_labels = load_scene_objects_labels(scene_objects_json_path)
            logging.info(f"加载了 {len(scene_objects_labels)} 张图像的 scene_objects labels")
        except Exception as e:
            logging.warning(f"加载 scene_objects 失败，将无法基于 scene_objects 统计 label: {e}")
    else:
        logging.warning(f"scene_objects JSON 不存在或未提供: {scene_objects_json_path}")

    all_labels_from_scene_objects: List[str] = []
    if scene_objects_labels:
        label_set = set()
        for labels in scene_objects_labels.values():
            for lbl in labels:
                if isinstance(lbl, str) and lbl:
                    label_set.add(lbl)
        all_labels_from_scene_objects = sorted(label_set)


    # 3. 获取 bounding box JSON 文件路径（包含 instances 信息）
    bbox_json_path = rrd_path.with_suffix(".json")   # ✅ 和 rrd 配套
    if not bbox_json_path.exists():
        bbox_json_path = Path("3d_bounding_scannet_evalation_ours") / f"{scene}.json"
    # 或者再加你自己的 kuixuan fallback

    
    with bbox_json_path.open("r", encoding="utf-8") as f:
        instances_json = json.load(f)
    print("loading label points from rrd file: "+str(rrd_path))
    # 4. 从 rrd 文件读取每个 label 的点云（不合并不同视角）
    label_points_dict = load_label_points_from_rrd(rrd_path, instances_json)
    logging.info(f"从 rrd 文件成功加载 {len(label_points_dict)} 个标签的点云")
    if not label_points_dict:
        logging.error("无法加载点云数据")
        return

    # 根据 scene_objects 和 rrd 实际可用标签，确定最终要处理的标签集合
    if all_labels_from_scene_objects:
        all_labels = [lbl for lbl in all_labels_from_scene_objects if lbl in label_points_dict]
    else:
        all_labels = sorted(label_points_dict.keys())
    if not all_labels:
        logging.error("没有可用的标签，无法生成 mask JSON")
        return
    
    # 5. 渲染深度图（如果提供了 model_path）
    depth_map_dict: Dict[str, np.ndarray] = {}
    if model_path and model_path.exists():
        logging.info("渲染深度图...")
        # 获取需要渲染的图像列表：场景下所有有相机参数的视角
        image_names_set = set(intr_map.keys())
        # 如果这里失败，希望直接抛出异常而不是静默继续
        depth_map_dict, color_map_dict = render_depths_with_gaussians(
            model_path, iteration,
            intr_map=intr_map, c2w_map=c2w_map,
            size_map=size_map, path_map=path_map,
            image_names=image_names_set
        )
        logging.info(f"渲染了 {len(depth_map_dict)} 张深度图")
    else:
        logging.warning("未提供 model_path，将不使用深度过滤")
    
    # 6. 为场景下的所有图像、所有标签生成预测 mask，并保存为 JSON
    items: List[Dict] = []

    # 这里不再依赖任何 JSON 中出现的图片列表，而是直接遍历
    # 我们在前面构建好的 path_map：也就是场景下所有存在的 color/dslr/dense 图像。
    for idx, (image_key, image_path_str) in enumerate(tqdm(list(path_map.items()), desc="处理图像")):
        if max_images is not None and idx >= max_images:
            logging.info(f"已处理前 {max_images} 张图像，提前结束。")
            break
        # 跳过没有实际图像路径的条目
        if not image_path_str:
            continue

        image_path = Path(image_path_str)
        if not image_path.exists():
            logging.warning(f"图像文件不存在: {image_path}，跳过")
            continue

        # 只处理有相机参数和尺寸的帧
        if image_key not in intr_map or image_key not in c2w_map or image_key not in size_map:
            logging.warning(f"图像 {image_key} 缺少 intr/c2w/size 之一，跳过")
            continue

        K = intr_map[image_key]
        c2w = c2w_map[image_key]
        width, height = size_map[image_key]

        # 用于 JSON 的 image 字段：直接使用实际文件名
        image_name = image_path.name

        for label in all_labels:
            instance_points_list = label_points_dict.get(label, [])
            if not instance_points_list:
                # 该 label 在点云中没有实例，直接跳过
                continue

            merged_pred_mask = np.zeros((height, width), dtype=np.uint8)
            merged_pred_mask_expanded = np.zeros((height, width), dtype=np.uint8)
            valid_instance_found = False

            for inst_idx, label_points in enumerate(instance_points_list):
                # 获取当前视角的深度图（如果可用）
                if image_key in depth_map_dict:
                    depth_map = depth_map_dict[image_key]
                    pred_mask_sparse = project_points_with_depth_filter(
                        label_points, K, c2w, (width, height), depth_map, depth_threshold
                    )
                else:
                    pred_mask_sparse = project_points(label_points, K, c2w, (width, height))

                if np.sum(pred_mask_sparse > 0) == 0:
                    if not ENABLE_PURE_SAM3:
                        continue

                valid_instance_found = True

                # 粗 mask（膨胀填充）
                pred_mask_expanded = fill_sparse_mask(pred_mask_sparse, method="dilation")

                if coarse_only:
                    # 只用粗 mask，不经过 SAM3 精修
                    pred_mask = pred_mask_expanded
                else:
                    # SAM3 精修
                    pred_mask = get_mask_from_sam(
                        pred_mask_sparse,
                        pred_mask_expanded,
                        label,
                        image_path if image_path else None,
                        sam3_processor,
                        mask_selection=mask_selection,
                    )

                merged_pred_mask_expanded = np.maximum(merged_pred_mask_expanded, pred_mask_expanded)
                merged_pred_mask = np.maximum(merged_pred_mask, pred_mask)

            if not valid_instance_found:
                logging.info(f"图像 {image_name} 标签 {label} 在当前视角没有有效 instance，跳过写入 JSON")
                continue

            # 将预测 mask 编码为 COCO RLE（0/1），并写入 JSON
            # 如果是 coarse_only，就直接保存粗 mask；否则保存 SAM3 精修后的 mask
            mask_to_encode = merged_pred_mask_expanded if coarse_only else merged_pred_mask
            mask_binary = (mask_to_encode > 0).astype(np.uint8)
            if np.sum(mask_binary) == 0:
                # 完全空的 mask 不写入，避免 JSON 体积过大
                continue

            # pycocotools 的 encode 期望 Fortran-order，size 是 [h, w]
            rle = mask_utils.encode(np.asfortranarray(mask_binary))
            if isinstance(rle.get("counts"), bytes):
                rle["counts"] = rle["counts"].decode("ascii")
            rle["size"] = [int(height), int(width)]

            items.append(
                {
                    "image": image_name,
                    "label": label,
                    "mask_rle": rle,
                }
            )

    # 输出 JSON，格式与 dl3dv 的 mask_index_outputs_* 一致：
    # [
    #   {
    #     "scene": "<scene_name>",
    #     "items": [ ... ]
    #   }
    # ]
    output_json = output_dir / f"{scene}.json"
    output_data = [
        {
            "scene": scene,
            "items": items,
        }
    ]
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    logging.info(f"预测 mask JSON 已保存到: {output_json}")


def main():
    parser = argparse.ArgumentParser(description="为场景下所有图像/标签生成2D mask JSON")
    parser.add_argument("--scene", type=str, required=True, help="场景名称，如 0a7cc12c0e")
    parser.add_argument("--data-root", type=Path, default=Path("scannet_pgsr_eval"), help="数据根目录")
    parser.add_argument("--rrd-path", type=Path, required=True, help="rerun文件路径，包含点云和标签信息，如 output_3d_bounding_scannet_evaluation/0a7cc12c0e.rrd")
    parser.add_argument("--scene-objects-json", type=Path,  help="scene_objects JSON文件路径，如 scene_objects_Qwen3-VL-30B-A3B-Instruct/0a7cc12c0e.json")
    parser.add_argument("--output-dir", type=Path, default=Path("gyn_test_images/2d_iou_outputs_gyn_cmp_sam3_inst_scene_files"), help="输出目录")
    parser.add_argument("--model-path", type=Path, help="3DGS模型路径，如 output/0a7cc12c0e，用于深度渲染和遮挡过滤")
    parser.add_argument("--iteration", type=int, default=30000, help="模型迭代次数")
    parser.add_argument("--depth-threshold", type=float, default=0.0005, help="深度容差，只保留在 [depth, depth+threshold] 范围内的点")
    parser.add_argument("--checkpoint-path", type=Path, default=Path("/mnt/shared-storage-user/solution/huggingface/hub/models--facebook--sam3/snapshots/2afe64078f4420bdfbc063162d1336003efadc81/sam3.pt"), help="SAM3 模型路径")
    parser.add_argument("--confidence-threshold", type=float, default=0.6, help="SAM3 置信度阈值")
    parser.add_argument("--mask-selection", type=str, choices=["merge", "best"], default="merge", help="SAM3 多候选 mask 的选择方式：merge=加权融合，best=取最高分")
    parser.add_argument("--max-images", type=int, default=1000, help="最多处理的图像数量（用于加速调试，默认只跑前10张）")
    parser.add_argument("--coarse-only", action="store_true", help="仅保存基于点云膨胀的粗 mask，不使用 SAM3 精修")
    
    args = parser.parse_args()
    
    setup_logger()
    
    # 设置默认路径
    if args.rrd_path is None:
        args.rrd_path = Path("output_3d_bounding_scannet_evaluation") / f"{args.scene}.rrd"
    
    if args.scene_objects_json is None:
        args.scene_objects_json = Path("scene_objects_Qwen3-VL-30B-A3B-Instruct") / f"{args.scene}.json"
    
    if not args.rrd_path.exists():
        logging.error(f"rerun 文件不存在: {args.rrd_path}")
        return
    if not args.scene_objects_json.exists():
        logging.warning(f"scene_objects JSON 文件不存在，将跳过过滤: {args.scene_objects_json}")
        args.scene_objects_json = None
    
    # 如果没有提供 model_path，尝试使用默认路径
    if args.model_path is None:
        default_model_path = Path("output") / args.scene
        if default_model_path.exists():
            args.model_path = default_model_path
            logging.info(f"使用默认模型路径: {args.model_path}")
        else:
            logging.warning(f"未找到模型路径，将不使用深度过滤: {default_model_path}")
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    

    # 加载 SAM3 模型（如果不是 coarse-only 模式）
    model = None
    processor = None
    if not args.coarse_only:
        logging.info("加载 SAM3 模型...")
        if not args.checkpoint_path.exists():
            print(f"SAM3 checkpoint 不存在: {args.checkpoint_path}")
            return
        try:
            model = build_sam3_image_model(checkpoint_path=str(args.checkpoint_path))
            processor = Sam3Processor(model, confidence_threshold=args.confidence_threshold)
            print("SAM3 模型加载成功")
        except Exception as e:
            print(f"SAM3 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            return



    process_scene(
        scene=args.scene,
        data_root=args.data_root,
        rrd_path=args.rrd_path,
        scene_objects_json_path=args.scene_objects_json,
        output_dir=args.output_dir,
        model_path=args.model_path,
        iteration=args.iteration,
        sam3_model=model,
        sam3_processor=processor,
        depth_threshold=args.depth_threshold,
        mask_selection=args.mask_selection,
        max_images=args.max_images,
        coarse_only=args.coarse_only,
    )


if __name__ == "__main__":
    main()
