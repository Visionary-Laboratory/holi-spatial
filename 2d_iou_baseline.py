#!/usr/bin/env python3
"""
计算2D IoU：将3D点云投影到2D图像，生成mask，并与GT mask比较
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import binary_fill_holes
from tqdm import tqdm
import pycocotools.mask as mask_utils
import rerun.dataframe as rr_df

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
        level=logging.INFO,
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


def get_instance_points(rec, label: str, ins_id: str):
    """
    从 rerun 文件读取单个 instance 的点云，参考 3d_2d.py 的 get_instance_points 函数
    """
    # Entity path in rerun. The leading slash is important depending on how it was logged.
    # In 3d_bounding_instance_gs_rerun.py, it was logged as f"instances/{label}/{ins_id}/points"
    # Dataframe might prepend a slash.
    entity_path = f"/instances/{label}/{ins_id}/points"
    try:
        view = rec.view(index="log_time", contents=entity_path)
        data = view.select()
        df = data.read_pandas()
        if df.empty:
            # Try without leading slash
            entity_path = f"instances/{label}/{ins_id}/points"
            view = rec.view(index="log_time", contents=entity_path)
            data = view.select()
            df = data.read_pandas()
            if df.empty:
                return None
        
        # Find column that ends with positions
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
    except Exception as e:
        logging.warning(f"Error reading points for {label}_{ins_id}: {e}")
        return None


def load_label_points_from_rrd(rrd_path: Path, instances_json: List[Dict]) -> Dict[str, np.ndarray]:
    """
    从 rrd 文件读取每个 label 的所有 instance 点云并合并
    返回: {label: merged_points}
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
        
        points = get_instance_points(rec, label, ins_id)
        if points is None or points.size == 0:
            continue
        
        if points.ndim != 2 or points.shape[1] != 3:
            continue
        
        if label not in label_points_dict:
            label_points_dict[label] = []
        label_points_dict[label].append(points)
    
    # 合并每个 label 的所有点云
    merged_label_points: Dict[str, np.ndarray] = {}
    for label, points_list in label_points_dict.items():
        if points_list:
            merged_label_points[label] = np.concatenate(points_list, axis=0)
            logging.info(f"标签 {label} 合并了 {len(points_list)} 个 instance，共 {len(merged_label_points[label])} 个点")
    
    return merged_label_points


def load_camera_params(scene_dir: Path):
    """
    加载相机参数，参考 3d_2d.py 的 load_camera_params 函数
    """
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
        
        intr_map[stem] = K
        c2w_map[stem] = c2w
    
    return intr_map, c2w_map, (w, h)


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
    output_path: Path
):
    """可视化预测mask和GT mask"""
    # 加载原始图像
    if image_path.exists():
        image = np.array(Image.open(image_path))
    else:
        # 如果图像不存在，创建黑色背景
        h, w = pred_mask.shape
        image = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 创建可视化
    vis = image.copy()
    
    # 绘制GT mask（绿色）
    gt_overlay = vis.copy()
    gt_overlay[gt_mask > 0] = [0, 255, 0]  # 绿色
    vis = cv2.addWeighted(vis, 0.7, gt_overlay, 0.3, 0)
    
    # 绘制预测mask（红色）
    pred_overlay = vis.copy()
    pred_overlay[pred_mask > 0] = [255, 0, 0]  # 红色
    vis = cv2.addWeighted(vis, 0.7, pred_overlay, 0.3, 0)
    
    # 保存
    try:
        Image.fromarray(vis).save(output_path, "JPEG", quality=85)
    except Exception as e:
        logging.warning(f"保存可视化图像失败 {output_path}: {e}")


def process_scene(
    scene: str,
    data_root: Path,
    rrd_path: Path,
    gt_json_path: Path,
    output_dir: Path,
    model_path: Optional[Path] = None,
    iteration: int = 30000,
    depth_threshold: float = 0.1
):
    """处理单个场景"""
    logging.info(f"处理场景: {scene}")
    
    # 1. 加载相机参数（使用 3d_2d.py 的方式）
    scene_dir = data_root / scene
    intr_map, c2w_map, img_size = load_camera_params(scene_dir)
    if intr_map is None:
        logging.error(f"无法加载相机参数: {scene_dir}")
        return
    logging.info(f"加载了 {len(intr_map)} 个相机的参数")
    width, height = img_size
    
    # 构建 size_map 和 path_map（用于深度渲染）
    size_map = {}
    path_map = {}
    for stem in intr_map.keys():
        size_map[stem] = (width, height)
        image_path = scene_dir / "dslr" / "resized_undistorted_images" / f"{stem}.JPG"
        if not image_path.exists():
            image_path = scene_dir / "dslr" / "resized_undistorted_images" / f"{stem}.jpg"
        path_map[stem] = str(image_path) if image_path.exists() else ""
    
    # 2. 加载GT masks 和 instances JSON
    gt_masks = load_gt_masks(gt_json_path)
    logging.info(f"加载了 {len(gt_masks)} 张图像的GT masks")
    
    # 获取 bounding box JSON 文件路径（包含 instances 信息）
    bbox_json_path = Path("output_3d_bounding_scannet_evaluation") / f"{scene}.json"
    if not bbox_json_path.exists():
        logging.error(f"找不到 bounding box JSON 文件: {bbox_json_path}")
        return
    
    with bbox_json_path.open("r", encoding="utf-8") as f:
        instances_json = json.load(f)
    
    # 3. 从 rrd 文件读取每个 label 的点云（合并同一 label 的所有 instance）
    try:
        label_points_dict = load_label_points_from_rrd(rrd_path, instances_json)
        logging.info(f"从 rrd 文件成功加载 {len(label_points_dict)} 个标签的点云")
    except Exception as e:
        logging.error(f"从 rrd 文件读取失败: {e}")
        return
    
    if not label_points_dict:
        logging.error("无法加载点云数据")
        return
    
    # 4. 渲染深度图（如果提供了 model_path）
    depth_map_dict: Dict[str, np.ndarray] = {}
    if model_path and model_path.exists():
        logging.info("渲染深度图...")
        # 获取需要渲染的图像列表（从 GT masks）
        image_names_set = set(gt_masks.keys())
        try:
            depth_map_dict, color_map_dict = render_depths_with_gaussians(
                model_path, iteration,
                intr_map=intr_map, c2w_map=c2w_map,
                size_map=size_map, path_map=path_map,
                image_names=image_names_set
            )
            logging.info(f"渲染了 {len(depth_map_dict)} 张深度图")
            
            # 保存深度图可视化
            depth_vis_dir = output_dir / "depth_visualizations" / scene
            try:
                depth_vis_dir.mkdir(parents=True, exist_ok=True)
                saved_count = 0
                for stem, depth_map in depth_map_dict.items():
                    try:
                        # 归一化深度图用于可视化
                        valid_depth = depth_map[depth_map > 0]
                        # if valid_depth.size > 0:
                        #     d_min, d_max = valid_depth.min(), valid_depth.max()
                        #     depth_vis = (depth_map - d_min) / (d_max - d_min + 1e-8)
                        #     depth_vis = np.clip(depth_vis, 0, 1)
                        #     # 转换为彩色深度图（使用 colormap）
                        #     depth_vis_uint8 = (depth_vis * 255).astype(np.uint8)
                        #     depth_colored = cv2.applyColorMap(depth_vis_uint8, cv2.COLORMAP_JET)
                        #     # 无效深度区域设为黑色
                        #     depth_colored[depth_map <= 0] = [0, 0, 0]
                            
                        #     # 使用 PIL 保存，更稳定
                        #     depth_colored_bgr = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
                        #     depth_img = Image.fromarray(depth_colored_bgr)
                        #     depth_path = depth_vis_dir / f"{stem}_depth.png"
                        #     depth_img.save(depth_path, "PNG", compress_level=1)
                            
                        #     # 同时保存灰度版本
                        #     depth_gray = (depth_vis * 255).astype(np.uint8)
                        #     depth_gray[depth_map <= 0] = 0
                        #     depth_gray_img = Image.fromarray(depth_gray)
                        #     depth_gray_path = depth_vis_dir / f"{stem}_depth_gray.png"
                        #     depth_gray_img.save(depth_gray_path, "PNG", compress_level=1)
                            
                        #     saved_count += 1
                    except Exception as e:
                        logging.warning(f"保存深度图 {stem} 失败: {e}")
                        continue
                logging.info(f"深度图可视化已保存 {saved_count} 张到: {depth_vis_dir}")
            except Exception as e:
                logging.warning(f"创建深度图可视化目录失败: {e}")
        except Exception as e:
            logging.warning(f"深度渲染失败: {e}，将不使用深度过滤")
            depth_map_dict = {}
    else:
        logging.warning("未提供 model_path，将不使用深度过滤")
    
    # 5. 为每张图像、每个label计算IoU
    results = []
    vis_dir = output_dir / "visualizations" / scene
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    for image_name, label_masks in tqdm(gt_masks.items(), desc="处理图像"):
        # 尝试匹配图像名称（可能带扩展名或不带扩展名）
        image_stem = Path(image_name).stem  # 去掉扩展名
        image_key = None
        
        # 先尝试完整名称
        if image_name in intr_map:
            image_key = image_name
        # 再尝试 stem（不含扩展名）
        elif image_stem in intr_map:
            image_key = image_stem
        else:
            logging.warning(f"图像 {image_name} (stem: {image_stem}) 没有相机参数，跳过")
            continue
        
        # 获取相机参数
        K = intr_map[image_key]
        c2w = c2w_map[image_key]
        
        # 尝试获取图像路径
        image_path = scene_dir / "dslr" / "resized_undistorted_images" / image_name
        if not image_path.exists():
            image_path = None
        
        # 为每个label计算IoU
        for label, gt_mask in label_masks.items():
            # 检查GT mask的尺寸是否匹配
            if gt_mask.shape[0] != height or gt_mask.shape[1] != width:
                # 调整GT mask尺寸
                gt_mask_resized = cv2.resize(gt_mask, (width, height), interpolation=cv2.INTER_NEAREST)
            else:
                gt_mask_resized = gt_mask
            
            # 从 label_points_dict 获取该 label 的点云（已经合并了所有 instance）
            if label not in label_points_dict:
                logging.warning(f"图像 {image_name} 标签 {label} 没有点云数据")
                continue
            
            label_points = label_points_dict[label]
            
            # 获取当前视角的深度图（如果可用）
            if image_key in depth_map_dict:
                depth_map = depth_map_dict[image_key]
                # 使用深度过滤的投射方式
                pred_mask_sparse = project_points_with_depth_filter(
                    label_points, K, c2w, (width, height), depth_map, depth_threshold
                )
            else:
                # 如果没有深度图，使用原来的方式（不考虑遮挡）
                pred_mask_sparse = project_points(label_points, K, c2w, (width, height))
            
            if np.sum(pred_mask_sparse > 0) == 0:
                logging.warning(f"图像 {image_name} 标签 {label} 投射后没有有效点")
                continue
            
            # 填充mask
            # pred_mask = fill_sparse_mask(pred_mask_sparse, method="dilation")
            pred_mask = pred_mask_sparse
            # 计算IoU
            iou = compute_2d_iou(pred_mask, gt_mask_resized)
            
            results.append({
                "image": image_name,
                "label": label,
                "iou": float(iou),
                "pred_mask_area": int(np.sum(pred_mask > 0)),
                "gt_mask_area": int(np.sum(gt_mask > 0))
            })
            
            # 保存可视化
            if image_path:
                try:
                    image_stem = Path(image_name).stem
                    vis_path = vis_dir / f"{image_stem}_{label}_iou_{iou:.3f}.jpg"
                    visualize_masks(
                        Path(image_path) if isinstance(image_path, str) else image_path,
                        pred_mask,
                        gt_mask_resized,
                        vis_path
                    )
                except Exception as e:
                    logging.warning(f"保存可视化失败 {image_name}_{label}: {e}")
                
                try:
                    # 保存 GT mask
                    image_stem = Path(image_name).stem
                    gt_mask_path = vis_dir / f"{image_stem}_{label}_gt_mask.png"
                    Image.fromarray(gt_mask_resized).save(gt_mask_path, "PNG", compress_level=1)
                except Exception as e:
                    logging.warning(f"保存 GT mask 失败 {image_name}_{label}: {e}")
                
                try:
                    # 保存预测 mask
                    image_stem = Path(image_name).stem
                    pred_mask_path = vis_dir / f"{image_stem}_{label}_pred_mask.png"
                    Image.fromarray(pred_mask).save(pred_mask_path, "PNG", compress_level=1)
                except Exception as e:
                    logging.warning(f"保存预测 mask 失败 {image_name}_{label}: {e}")
    
    # 5. 保存结果和统计信息
    output_json = output_dir / f"{scene}_2d_iou.json"
    
    if results:
        ious = [r["iou"] for r in results]
        
        # 按图像和标签分组计算统计
        per_image_stats = {}
        per_label_stats = {}
        
        for r in results:
            img = r["image"]
            lbl = r["label"]
            iou_val = r["iou"]
            
            if img not in per_image_stats:
                per_image_stats[img] = []
            per_image_stats[img].append(iou_val)
            
            if lbl not in per_label_stats:
                per_label_stats[lbl] = []
            per_label_stats[lbl].append(iou_val)
        
        logging.info(f"场景 {scene} 完成:")
        logging.info(f"  总计算数: {len(results)}")
        logging.info(f"  总图像数: {len(per_image_stats)}")
        logging.info(f"  总标签数: {len(per_label_stats)}")
        logging.info(f"  整体平均IoU: {np.mean(ious):.4f}")
        logging.info(f"  整体中位数IoU: {np.median(ious):.4f}")
        logging.info(f"  整体最小IoU: {np.min(ious):.4f}")
        logging.info(f"  整体最大IoU: {np.max(ious):.4f}")
        
        per_image_means = {img: np.mean(vals) for img, vals in per_image_stats.items()}
        logging.info(f"  每张图像平均IoU: {np.mean(list(per_image_means.values())):.4f}")
        
        per_label_means = {lbl: np.mean(vals) for lbl, vals in per_label_stats.items()}
        logging.info(f"  每个标签平均IoU: {np.mean(list(per_label_means.values())):.4f}")
        
        stats = {
            "total_count": len(results),
            "total_images": len(per_image_stats),
            "total_labels": len(per_label_stats),
            "overall_mean_iou": float(np.mean(ious)),
            "overall_median_iou": float(np.median(ious)),
            "overall_min_iou": float(np.min(ious)),
            "overall_max_iou": float(np.max(ious)),
            "per_image_mean_iou": float(np.mean(list(per_image_means.values()))),
            "per_label_mean_iou": float(np.mean(list(per_label_means.values()))),
            "per_image_stats": {img: float(mean) for img, mean in per_image_means.items()},
            "per_label_stats": {lbl: float(mean) for lbl, mean in per_label_means.items()}
        }
        
        output_data = {
            "statistics": stats,
            "results": results
        }
        
        with output_json.open("w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
    else:
        with output_json.open("w", encoding="utf-8") as f:
            json.dump({"statistics": {}, "results": []}, f, indent=2, ensure_ascii=False)
    
    logging.info(f"结果已保存到: {output_json}")


def main():
    parser = argparse.ArgumentParser(description="计算2D IoU")
    parser.add_argument("--scene", type=str, required=True, help="场景名称，如 0a7cc12c0e")
    parser.add_argument("--data-root", type=Path, default=Path("scannetppv2/data"), help="数据根目录")
    parser.add_argument("--rrd-path", type=Path, required=True, help="rerun文件路径，包含点云和标签信息，如 output_3d_bounding_scannet_evaluation/0a7cc12c0e.rrd")
    parser.add_argument("--gt-json", type=Path, help="GT mask JSON文件路径，如 mask_index_outputs/0a7cc12c0e.json")
    parser.add_argument("--output-dir", type=Path, default=Path("2d_iou_outputs_baseline"), help="输出目录")
    parser.add_argument("--model-path", type=Path, help="3DGS模型路径，如 output/0a7cc12c0e，用于深度渲染和遮挡过滤")
    parser.add_argument("--iteration", type=int, default=30000, help="模型迭代次数")
    parser.add_argument("--depth-threshold", type=float, default=0.0005, help="深度容差，只保留在 [depth, depth+threshold] 范围内的点")
    args = parser.parse_args()
    
    setup_logger()
    
    # 设置默认路径
    if args.rrd_path is None:
        args.rrd_path = Path("output_3d_bounding_scannet_evaluation") / f"{args.scene}.rrd"
    
    if args.gt_json is None:
        args.gt_json = Path("mask_index_outputs") / f"{args.scene}.json"
    
    if not args.rrd_path.exists():
        logging.error(f"rerun 文件不存在: {args.rrd_path}")
        return
    
    # 如果没有提供 model_path，尝试使用默认路径
    if args.model_path is None:
        default_model_path = Path("output") / args.scene
        if default_model_path.exists():
            args.model_path = default_model_path
            logging.info(f"使用默认模型路径: {args.model_path}")
        else:
            logging.warning(f"未找到模型路径，将不使用深度过滤: {default_model_path}")
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    process_scene(
        scene=args.scene,
        data_root=args.data_root,
        rrd_path=args.rrd_path,
        gt_json_path=args.gt_json,
        output_dir=args.output_dir,
        model_path=args.model_path,
        iteration=args.iteration,
        depth_threshold=args.depth_threshold
    )


if __name__ == "__main__":
    main()
