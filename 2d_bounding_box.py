from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image


DEFAULT_SCENE = "0a5c013435"
DEFAULT_DATA_ROOT = Path("/home/liuyifei/code/posevlm/scannetppv2/data")
DEFAULT_MASK_ROOT = Path("/home/liuyifei/code/posevlm/sam_masks")
DEFAULT_SCENE_OBJECTS_ROOT = Path("/home/liuyifei/code/posevlm/scene_objects_Qwen2.5-VL-7B-Instruct")
DEFAULT_DEPTH_ROOT = Path("/home/liuyifei/code/posevlm/DptV3/data")
DEFAULT_OUTPUT_DIR = Path("/home/liuyifei/code/posevlm/sam_masks")


def setup_logger() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def load_transforms(scannet_dir: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], int, int, List[Tuple[str, str]]]:
    """
    读取 nerfstudio transforms_undistorted.json
    返回 K、c2w、宽高以及 (stem, file_path) 列表。
    """
    json_path = scannet_dir / "dslr" / "nerfstudio" / "transforms_undistorted.json"
    with json_path.open("r", encoding="utf-8") as f:
        contents = json.load(f)

    fl_x, fl_y, cx, cy = contents["fl_x"], contents["fl_y"], contents["cx"], contents["cy"]
    w, h = contents["w"], contents["h"]
    frames = contents.get("frames", []) + contents.get("test_frames", [])

    intr_map: Dict[str, np.ndarray] = {}
    c2w_map: Dict[str, np.ndarray] = {}
    frame_list: List[Tuple[str, str]] = []

    for frame in frames:
        K = np.array([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]], dtype=np.float32)
        c2w = np.array(frame["transform_matrix"], dtype=np.float32)
        # OpenGL -> COLMAP
        c2w[:3, 1:3] *= -1
        stem = Path(frame["file_path"]).stem
        intr_map[stem] = K
        c2w_map[stem] = c2w
        frame_list.append((stem, frame["file_path"]))

    logging.info("加载位姿完成，帧数: %d", len(frame_list))
    return intr_map, c2w_map, w, h, frame_list


def project_bbox_to_image(
    corners: List[Dict[str, float]],
    K: np.ndarray,
    c2w: np.ndarray,
    img_w: int,
    img_h: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回投影点 (N,2)、对应深度 (N,) 以及有效掩码。"""
    pts = np.array([[c["x"], c["y"], c["z"], 1.0] for c in corners], dtype=np.float32)  # (8,4)
    w2c = np.linalg.inv(c2w)
    cam = (pts @ w2c.T)[:, :3]
    z = cam[:, 2]
    valid = z > 1e-4
    if not np.any(valid):
        return np.zeros((len(pts), 2), dtype=np.float32), z, valid

    cam_xy = cam[:, :2]
    uv = (cam_xy / z[:, None]) @ K[:2, :2].T + K[:2, 2]
    return uv, z, valid


def project_and_clip(
    corners: List[Dict[str, float]],
    K: np.ndarray,
    c2w: np.ndarray,
    img_w: int,
    img_h: int,
) -> Tuple[bool, Tuple[int, int, int, int]]:
    """只基于几何可见性（前方 + 与图像相交）计算 2D 框。"""
    projected, depths, valid_mask = project_bbox_to_image(corners, K, c2w, img_w, img_h)
    if not valid_mask.any():
        return False, (0, 0, 0, 0)

    valid_proj = projected[valid_mask]
    x_min = int(np.floor(valid_proj[:, 0].min()))
    x_max = int(np.ceil(valid_proj[:, 0].max()))
    y_min = int(np.floor(valid_proj[:, 1].min()))
    y_max = int(np.ceil(valid_proj[:, 1].max()))

    if x_max < 0 or x_min >= img_w or y_max < 0 or y_min >= img_h:
        return False, (0, 0, 0, 0)

    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img_w - 1, x_max)
    y_max = min(img_h - 1, y_max)
    if x_max <= x_min or y_max <= y_min:
        return False, (0, 0, 0, 0)
    return True, (x_min, y_min, x_max, y_max)


def load_per_image_labels(scene_objects_root: Path, scene: str) -> Dict[str, List[str]]:
    path = scene_objects_root / f"{scene}.json"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    per_image: Dict[str, List[str]] = data.get("per_image", {})
    return per_image


def check_bbox_visibility(
    corners: List[Dict[str, float]],
    depth_map: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
    img_w: int,
    img_h: int,
    depth_threshold: float = 0.5,
    occlusion_ratio_threshold: float = 0.3,
) -> Tuple[bool, Dict]:
    """
    利用深度判定遮挡：统计 ROI 中有多少像素比 bbox 最近深度更近。
    """
    projected, depths, valid_mask = project_bbox_to_image(corners, K, c2w, img_w, img_h)
    if not valid_mask.any():
        return False, {"reason": "behind_camera", "valid_points": 0, "occlusion_ratio": 1.0}

    valid_proj = projected[valid_mask]
    x_min = int(np.floor(valid_proj[:, 0].min()))
    x_max = int(np.ceil(valid_proj[:, 0].max()))
    y_min = int(np.floor(valid_proj[:, 1].min()))
    y_max = int(np.ceil(valid_proj[:, 1].max()))

    if x_max < 0 or x_min >= img_w or y_max < 0 or y_min >= img_h:
        return False, {"reason": "outside_image", "bbox_2d": [x_min, y_min, x_max, y_max], "valid_points": int(valid_mask.sum())}

    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img_w - 1, x_max)
    y_max = min(img_h - 1, y_max)
    if x_max <= x_min or y_max <= y_min:
        return False, {"reason": "too_small", "bbox_2d": [x_min, y_min, x_max, y_max]}

    valid_depths = depths[valid_mask]
    bbox_min_depth = valid_depths.min()

    roi_depth = depth_map[y_min : y_max + 1, x_min : x_max + 1]
    valid_roi_depth = roi_depth[(roi_depth > 0) & np.isfinite(roi_depth) & (roi_depth < 100)]

    if len(valid_roi_depth) == 0:
        return True, {
            "reason": "no_depth_in_roi",
            "bbox_2d": [x_min, y_min, x_max, y_max],
            "bbox_min_depth": float(bbox_min_depth),
            "valid_points": int(valid_mask.sum()),
        }

    occluded_pixels = np.sum(valid_roi_depth < (bbox_min_depth - depth_threshold))
    total_valid_pixels = len(valid_roi_depth)
    occlusion_ratio = occluded_pixels / total_valid_pixels
    is_visible = occlusion_ratio < occlusion_ratio_threshold

    info = {
        "bbox_2d": [int(x_min), int(y_min), int(x_max), int(y_max)],
        "bbox_min_depth": float(bbox_min_depth),
        "bbox_max_depth": float(valid_depths.max()),
        "roi_min_depth": float(valid_roi_depth.min()),
        "roi_median_depth": float(np.median(valid_roi_depth)),
        "occlusion_ratio": float(occlusion_ratio),
        "valid_points": int(valid_mask.sum()),
    }
    if not is_visible:
        info["reason"] = "occluded"
    return is_visible, info


def draw_boxes(image_path: Path, boxes: List[Tuple[str, Tuple[int, int, int, int]]], save_path: Path) -> None:
    img = cv2.imread(str(image_path))
    if img is None:
        logging.warning("无法读取图片: %s", image_path)
        return
    for label, (x0, y0, x1, y1) in boxes:
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(img, label, (x0, max(0, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), img)


def process_scene(
    scene: str,
    data_root: Path,
    mask_root: Path,
    scene_objects_root: Path,
    depth_root: Path,
    output_dir: Path,
    depth_threshold: float,
    occlusion_ratio_threshold: float,
) -> Path:
    scene_dir = data_root / scene
    intr_map, c2w_map, img_w, img_h, frame_list = load_transforms(scene_dir)

    bbox3d_path = mask_root / f"{scene}.json"
    with bbox3d_path.open("r", encoding="utf-8") as f:
        bbox_items = json.load(f)

    per_image_labels = load_per_image_labels(scene_objects_root, scene)

    image_root = scene_dir / "dslr" / "resized_undistorted_images"
    depth_dir = depth_root / scene / "depth_da3"
    vis_dir = output_dir / scene / "2d_bbox_vis"
    meta = {}

    for stem, rel_path in frame_list:
        if stem not in intr_map or stem not in c2w_map:
            continue
        depth_path = depth_dir / f"{stem}.npy"
        if not depth_path.exists():
            logging.warning("缺少深度，跳过: %s", depth_path)
            continue
        depth = np.load(depth_path)
        image_name = Path(rel_path).name
        valid_labels = set(per_image_labels.get(image_name, []))
        if not valid_labels:
            continue
        K = intr_map[stem]
        c2w = c2w_map[stem]
        boxes_2d: List[Tuple[str, Tuple[int, int, int, int]]] = []
        for item in bbox_items:
            label = item["label"]
            if label not in valid_labels:
                continue
            corners = item["bounding_box"]
            visible, info = check_bbox_visibility(
                corners,
                depth,
                K,
                c2w,
                img_w,
                img_h,
                depth_threshold=depth_threshold,
                occlusion_ratio_threshold=occlusion_ratio_threshold,
            )
            if visible:
                box = info["bbox_2d"]
                boxes_2d.append((label, tuple(box)))
        if not boxes_2d:
            continue

        img_path = image_root / rel_path
        save_path = vis_dir / f"{stem}_bbox.jpg"
        draw_boxes(img_path, boxes_2d, save_path)
        meta[stem] = [{"label": lbl, "bbox": box} for lbl, box in boxes_2d]
        logging.info("保存 2D 可视化: %s，框数: %d", save_path, len(boxes_2d))

    meta_path = vis_dir / "bboxes_per_image.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logging.info("写出 2D bbox 元数据: %s", meta_path)
    return meta_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 3D bbox 投影到图像并可视化 2D bbox")
    parser.add_argument("--scene", default=DEFAULT_SCENE, help="场景名，如 0a5c013435")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="scannetppv2/data 根目录")
    parser.add_argument("--mask-root", type=Path, default=DEFAULT_MASK_ROOT, help="sam_masks 根目录（含 <scene>.json 3D bbox）")
    parser.add_argument("--scene-objects-root", type=Path, default=DEFAULT_SCENE_OBJECTS_ROOT, help="scene_objects 分类结果目录（含 <scene>.json，提供 per_image 标签列表）")
    parser.add_argument("--depth-root", type=Path, default=DEFAULT_DEPTH_ROOT, help="深度 npy 根目录（scene/depth_da3）")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="输出目录，保存 2D 可视化")
    parser.add_argument("--depth-threshold", type=float, default=0.2, help="遮挡判定的深度差阈值（米）")
    parser.add_argument("--occlusion-ratio-threshold", type=float, default=0.3, help="遮挡比例阈值，超过则判为不可见")
    return parser.parse_args()


def main() -> None:
    setup_logger()
    args = parse_args()
    process_scene(
        scene=args.scene,
        data_root=args.data_root,
        mask_root=args.mask_root,
        scene_objects_root=args.scene_objects_root,
        depth_root=args.depth_root,
        output_dir=args.output_dir,
        depth_threshold=args.depth_threshold,
        occlusion_ratio_threshold=args.occlusion_ratio_threshold,
    )


if __name__ == "__main__":
    main()

