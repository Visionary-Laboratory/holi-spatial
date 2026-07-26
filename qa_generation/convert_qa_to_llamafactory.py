#!/usr/bin/env python3
"""
将QA JSON转换为llamafactory可训练的格式。

输入：
- --image-root: 原始图片路径（如 scannetppv2/data）
- --qa-json-dir: QA JSON目录（如 output_QA）
- --output-dir: 输出数据集目录

输出：
- images/: 生成的图片目录（按场景组织）
- dataset.jsonl: llamafactory格式的数据集文件

功能：
- 根据marker_type生成相应的图片（point_marker, white_mask, mask_covering_image, 2d_bbox）
- 如果不需要指代（如language_description或没有marker_type），使用原图
- 支持--scene-id参数，用于单个场景验证
"""

import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
import hashlib

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from pycocotools import mask as mask_util


# 颜色映射（BGR格式，用于OpenCV）
COLOR_MAP = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
}


def decode_rle_mask(mask_rle: Dict) -> Optional[np.ndarray]:
    """解码RLE格式的mask"""
    size = mask_rle.get("size", [])
    counts = mask_rle.get("counts", "")
    
    if isinstance(counts, str):
        rle = {"size": size, "counts": counts}
    else:
        rle = {"size": size, "counts": counts}
    
    mask = mask_util.decode(rle)
    return mask.astype(np.uint8) * 255


def _save_image(img: Image.Image, path: Path, quality: int = 95, skip_existing: bool = False) -> None:
    if skip_existing and path.exists():
        return
    img.save(path, quality=quality)


def _copy_image(src: Path, dst: Path, skip_existing: bool = False) -> None:
    if skip_existing and dst.exists():
        return
    shutil.copy2(src, dst)


def find_original_image_path(image_root: Path, scene_id: str, image_name: str) -> Optional[Path]:
    """查找原始图片路径
    
    scannetppv2的目录结构：
    scannetppv2/
      data/
        {scene_id}/
          dslr/
            resized_undistorted_images/
              {image_name}
    """
    # 尝试多种可能的路径结构
    possible_paths = [
        # scannetppv2: scannetppv2/data/{scene_id}/dslr/resized_undistorted_images/{image_name}
        image_root / "data" / scene_id / "dslr" / "resized_undistorted_images" / image_name,
        image_root / "data" / scene_id / "dslr" / "resized_undistorted_images" / image_name.upper(),
        image_root / "data" / scene_id / "dslr" / "resized_undistorted_images" / image_name.lower(),
        # 如果image_root已经是scannetppv2/data
        image_root / scene_id / "dslr" / "resized_undistorted_images" / image_name,
        image_root / scene_id / "dslr" / "resized_undistorted_images" / image_name.upper(),
        image_root / scene_id / "dslr" / "resized_undistorted_images" / image_name.lower(),
        # ScanNetV2: scannetv2/scans/{scene_id}/color/{image_name}
        image_root / "scans" / scene_id / "color" / image_name,
        image_root / "scans" / scene_id / "color" / image_name.upper(),
        image_root / "scans" / scene_id / "color" / image_name.lower(),
        # 如果image_root已经是scannetv2/scans
        image_root / scene_id / "color" / image_name,
        image_root / scene_id / "color" / image_name.upper(),
        image_root / scene_id / "color" / image_name.lower(),
        # 备用路径
        image_root / "data" / scene_id / image_name,
        image_root / "data" / scene_id / image_name.upper(),
        image_root / "data" / scene_id / image_name.lower(),
        image_root / scene_id / image_name,
        image_root / scene_id / image_name.upper(),
        image_root / scene_id / image_name.lower(),
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    return None


def draw_point_on_image(image_path: Path, point: Tuple[float, float], color: str = "red", radius: int = 5) -> Image.Image:
    """在原图上画点"""
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    x, y = int(point[0]), int(point[1])
    bbox = [x - radius, y - radius, x + radius, y + radius]
    
    # 使用PIL的颜色（RGB格式）
    rgb_color = {
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
    }.get(color, (255, 0, 0))
    
    draw.ellipse(bbox, fill=rgb_color, outline=rgb_color, width=2)
    return img


def draw_bbox_on_image(image_path: Path, bbox: Tuple[float, float, float, float], color: str = "red", line_width: int = 3) -> Image.Image:
    """在原图上画2D边界框"""
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    x_min, y_min, x_max, y_max = bbox
    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
    
    rgb_color = {
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
    }.get(color, (255, 0, 0))
    
    draw.rectangle([x_min, y_min, x_max, y_max], outline=rgb_color, width=line_width)
    return img


def draw_mask_overlay_on_image(image_path: Path, mask_rle: Dict, color: str = "red", line_width: int = 3) -> Image.Image:
    """在原图上画mask overlay（轮廓）"""
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    
    mask = decode_rle_mask(mask_rle)
    if mask is None:
        return img
    
    # 调整mask尺寸以匹配图像尺寸
    img_h, img_w = img_array.shape[:2]
    mask_h, mask_w = mask.shape
    if mask_h != img_h or mask_w != img_w:
        mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    
    # 将RGB转换为BGR（因为OpenCV函数期望BGR格式）
    masked_image_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 绘制轮廓（使用BGR颜色）
    bgr_color = COLOR_MAP.get(color, (0, 0, 255))
    cv2.drawContours(masked_image_bgr, contours, -1, bgr_color, line_width)
    
    # 转换回RGB格式
    masked_image_rgb = cv2.cvtColor(masked_image_bgr, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(masked_image_rgb)


def create_white_mask_image(mask_rle: Dict, image_size: Optional[Tuple[int, int]] = None) -> Optional[Image.Image]:
    """创建白色mask图片"""
    mask = decode_rle_mask(mask_rle)
    if mask is None:
        return None
    
    # 如果指定了图像尺寸，调整mask
    if image_size:
        h, w = image_size
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # 创建白色背景
    white_mask = np.ones((mask.shape[0], mask.shape[1], 3), dtype=np.uint8) * 255
    
    # 将mask区域设为白色（已经是白色了），非mask区域设为黑色
    mask_binary = (mask > 0).astype(np.uint8)
    white_mask[:, :, 0] = white_mask[:, :, 0] * mask_binary
    white_mask[:, :, 1] = white_mask[:, :, 1] * mask_binary
    white_mask[:, :, 2] = white_mask[:, :, 2] * mask_binary
    
    return Image.fromarray(white_mask)


def extract_mask_label(mask_key: str, image_key: str) -> str:
    """从mask key中提取A/B/C/1/2标签."""
    prefix = f"{image_key}_"
    if not mask_key.startswith(prefix):
        return ""
    suffix = mask_key[len(prefix):]
    if "_" not in suffix:
        return ""
    label = suffix.split("_")[-1]
    return label if label in {"A", "B", "C", "1", "2"} else ""


def mask_key_sorter(mask_key: str, image_key: str) -> Tuple[int, str]:
    """保证mask key顺序稳定且可读."""
    order = {"A": 0, "B": 1, "C": 2, "1": 3, "2": 4, "": 5}
    label = extract_mask_label(mask_key, image_key)
    return (order.get(label, 99), mask_key)


def generate_unique_suffix(entry: Dict[str, Any], marker_data: Dict[str, Any], image_name: str) -> str:
    """
    为每个QA条目生成唯一的后缀，避免文件名冲突。
    基于question_type、sub_question_type、marker_type、ins_id和颜色信息生成hash。
    """
    # 提取关键信息用于生成hash
    key_parts = []

    # 1. 基础类型信息
    question_type = entry.get("question_type", "")
    sub_question_type = entry.get("sub_question_type", "")
    marker_type = entry.get("marker_type", "")

    key_parts.extend([question_type, sub_question_type, marker_type])

    # 2. 根据image_name锁定image_a/image_b，只考虑匹配图片相关的数据
    image_key = None
    if image_name:
        if image_name == entry.get("image_a"):
            image_key = "image_a"
        elif image_name == entry.get("image_b"):
            image_key = "image_b"

    marker_data_filtered = marker_data
    entry_color_filtered: Dict[str, Any] = {}
    entry_image_payload: Dict[str, Any] = {}
    if image_key:
        marker_data_filtered = {k: v for k, v in marker_data.items() if k.startswith(f"{image_key}_")}
        entry_image_payload = {
            k: v for k, v in entry.items()
            if k.startswith(f"{image_key}_")
            and k not in {"image_a", "image_b"}
            and not k.endswith("_color")
        }
        entry_color_filtered = {
            k: v for k, v in entry.items()
            if k.startswith(f"{image_key}_") and k.endswith("_color")
        }
    else:
        entry_color_filtered = {k: v for k, v in entry.items() if k.endswith("_color")}

    # 3. 解析该图片涉及的对象标签 (A/B/C/1/2/空)
    marker_labels = set()
    prefix = f"{image_key}_" if image_key else ""
    for key in list(marker_data_filtered.keys()) + list(entry_image_payload.keys()):
        base_key = key[:-6] if key.endswith("_color") else key
        remainder = base_key[len(prefix):] if prefix and base_key.startswith(prefix) else base_key
        label = ""
        if "_" in remainder:
            candidate = remainder.split("_")[-1]
            if candidate in {"A", "B", "C", "1", "2"}:
                label = candidate
        marker_labels.add(label)
    marker_labels.discard("")

    # 4. 物体身份信息 (ins_id) - 仅考虑匹配图片涉及的对象
    ins_id_set = set()

    # inter_object_distance: object_1 / object_2
    for label in sorted(marker_labels):
        if label in {"1", "2"}:
            obj = entry.get(f"object_{label}")
            if isinstance(obj, dict):
                ins_id = obj.get("ins_id")
                if ins_id:
                    ins_id_set.add(str(ins_id))

    # object_distance/object_relpos: objects list/dict
    if "objects" in entry:
        if isinstance(entry["objects"], list):
            if marker_labels:
                for label in sorted(marker_labels):
                    if label.isdigit():
                        idx = int(label) - 1
                        if 0 <= idx < len(entry["objects"]):
                            ins_id = entry["objects"][idx].get("ins_id")
                            if ins_id:
                                ins_id_set.add(str(ins_id))
            if not ins_id_set and len(entry["objects"]) == 1:
                ins_id = entry["objects"][0].get("ins_id")
                if ins_id:
                    ins_id_set.add(str(ins_id))
        elif isinstance(entry["objects"], dict):
            if marker_labels:
                for label in sorted(marker_labels):
                    if label in entry["objects"]:
                        ins_id = entry["objects"][label].get("ins_id")
                        if ins_id:
                            ins_id_set.add(str(ins_id))
            if not ins_id_set and len(entry["objects"]) == 1:
                only_obj = next(iter(entry["objects"].values()))
                if isinstance(only_obj, dict):
                    ins_id = only_obj.get("ins_id")
                    if ins_id:
                        ins_id_set.add(str(ins_id))

    if ins_id_set:
        key_parts.append("_".join(sorted(ins_id_set)))

    # 5. 标记颜色信息 - 仅考虑匹配图片相关颜色
    color_parts = []
    for key, value in marker_data_filtered.items():
        if key.endswith("_color"):
            color_parts.append(f"{key}:{value}")
    for key, value in entry_color_filtered.items():
        if isinstance(value, str):
            color_parts.append(f"{key}:{value}")

    if color_parts:
        key_parts.append("_".join(sorted(color_parts)))

    # 6. marker_data内容（去掉颜色）哈希，避免同一图片不同坐标的覆盖
    marker_payload = {k: v for k, v in marker_data_filtered.items() if not k.endswith("_color")}
    if entry_image_payload:
        marker_payload.update(entry_image_payload)
    if marker_payload:
        marker_str = json.dumps(marker_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        marker_hash = hashlib.md5(marker_str.encode("utf-8")).hexdigest()[:8]
        key_parts.append(f"marker:{marker_hash}")

    # 生成hash
    key_str = "_".join(str(p) for p in key_parts if p)  # 过滤掉空字符串
    hash_obj = hashlib.md5(key_str.encode('utf-8'))
    hash_hex = hash_obj.hexdigest()[:8]  # 使用前8位作为后缀

    return hash_hex


def generate_image_for_entry(
    entry: Dict[str, Any],
    image_root: Path,
    scene_id: str,
    output_images_dir: Path,
    image_counter: Dict[str, int],
    skip_existing_images: bool = False,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    为QA条目生成所需的图片，返回图片路径列表和元数据。
    
    Returns:
        (image_paths, metadata): 图片路径列表（相对于output_images_dir）和元数据
    """
    marker_type = entry.get("marker_type")
    marker_data = entry.get("marker_data", {})
    image_a_name = entry.get("image_a", "")
    image_b_name = entry.get("image_b", "")
    
    image_paths = []
    metadata = {}
    
    # 查找原始图片路径
    image_a_path = find_original_image_path(image_root, scene_id, image_a_name) if image_a_name else None
    image_b_path = find_original_image_path(image_root, scene_id, image_b_name) if image_b_name else None
    
    if not image_a_path and not image_b_path:
        raise ValueError(f"无法找到原始图片: scene_id={scene_id}, image_a={image_a_name}, image_b={image_b_name}, image_a_path={image_a_path}, image_b_path={image_b_path}")
    
    # 创建场景图片目录
    scene_images_dir = output_images_dir / scene_id
    scene_images_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成唯一后缀，避免文件名冲突
    suffix_a = generate_unique_suffix(entry, marker_data, image_a_name) if image_a_name else ""
    suffix_b = generate_unique_suffix(entry, marker_data, image_b_name) if image_b_name else ""
    
    # 根据marker_type生成图片
    if marker_type == "point_marker":
        # 需要在原图上画点（如果marker_data中有对应图片的点标记）
        # 可能有多个点（如object_dist中的point_1, point_2）
        if image_a_path:
            # 查找所有image_a的点（可能是image_a_point、image_a_point_1等）
            point_keys_a = [k for k in marker_data.keys() if k.startswith("image_a_point_") and not k.endswith("_color")]
            if not point_keys_a:
                # 回退到单个点（如object_dpt）
                point_a = marker_data.get("image_a_point")
                if not point_a:
                    raise ValueError(f"未找到image_a的点标记: scene_id={scene_id}, marker_data keys={list(marker_data.keys())}")
                color_a = marker_data.get("image_a_point_color", "red")
                img_a = draw_point_on_image(image_a_path, point_a, color_a)
                img_a_filename = f"{Path(image_a_name).stem}_point_{suffix_a}.jpg"
                img_a_path_save = scene_images_dir / img_a_filename
                _save_image(img_a, img_a_path_save, quality=95, skip_existing=skip_existing_images)
                image_paths.append(f"images/{scene_id}/{img_a_filename}")
            else:
                # 多个点，在同一张图上画多个点
                img_a = Image.open(image_a_path).convert('RGB')
                img_a_array = np.array(img_a)
                # OpenCV使用BGR通道顺序，先转换以保证颜色正确
                img_a_bgr = cv2.cvtColor(img_a_array, cv2.COLOR_RGB2BGR)
                
                for point_key in point_keys_a:
                    point = marker_data.get(point_key)
                    if not point:
                        raise ValueError(f"点标记数据为空: scene_id={scene_id}, point_key={point_key}")
                    if not isinstance(point, (list, tuple)) or len(point) != 2:
                        raise ValueError(f"点标记格式错误: scene_id={scene_id}, point_key={point_key}, point={point}")
                    
                    color_key = f"{point_key}_color"
                    color = marker_data.get(color_key, "red")
                    x, y = int(point[0]), int(point[1])
                    bgr_color = COLOR_MAP.get(color, (0, 0, 255))
                    cv2.circle(img_a_bgr, (x, y), 5, bgr_color, -1)
                    cv2.circle(img_a_bgr, (x, y), 8, bgr_color, 2)
                
                img_a_array = cv2.cvtColor(img_a_bgr, cv2.COLOR_BGR2RGB)
                img_a = Image.fromarray(img_a_array)
                img_a_filename = f"{Path(image_a_name).stem}_point_{suffix_a}.jpg"
                img_a_path_save = scene_images_dir / img_a_filename
                _save_image(img_a, img_a_path_save, quality=95, skip_existing=skip_existing_images)
                image_paths.append(f"images/{scene_id}/{img_a_filename}")
        
        if image_b_path:
            # 查找所有image_b的点（可能是image_b_point、image_b_point_2等）
            point_keys_b = [k for k in marker_data.keys() if k.startswith("image_b_point_") and not k.endswith("_color")]
            if not point_keys_b:
                # 回退到单个点（如object_dpt）
                point_b = marker_data.get("image_b_point")
                if not point_b:
                    # 如果image_b_path存在但没有image_b_point，说明这个条目可能只需要image_a的点标记
                    # 但image_b仍然需要原图（问题中可能提到"Locate the same physical object in image B"）
                    img_b_filename = f"{Path(image_b_name).stem}.jpg"
                    img_b_path_save = scene_images_dir / img_b_filename
                    if not img_b_path_save.exists():
                        _copy_image(image_b_path, img_b_path_save, skip_existing=skip_existing_images)
                    image_paths.append(f"images/{scene_id}/{img_b_filename}")
                else:
                    color_b = marker_data.get("image_b_point_color", "red")
                    img_b = draw_point_on_image(image_b_path, point_b, color_b)
                    img_b_filename = f"{Path(image_b_name).stem}_point_{suffix_b}.jpg"
                    img_b_path_save = scene_images_dir / img_b_filename
                    _save_image(img_b, img_b_path_save, quality=95, skip_existing=skip_existing_images)
                    image_paths.append(f"images/{scene_id}/{img_b_filename}")
            else:
                # 多个点，在同一张图上画多个点
                img_b = Image.open(image_b_path).convert('RGB')
                img_b_array = np.array(img_b)
                # OpenCV使用BGR通道顺序，先转换以保证颜色正确
                img_b_bgr = cv2.cvtColor(img_b_array, cv2.COLOR_RGB2BGR)
                
                for point_key in point_keys_b:
                    point = marker_data.get(point_key)
                    if not point:
                        raise ValueError(f"点标记数据为空: scene_id={scene_id}, point_key={point_key}")
                    if not isinstance(point, (list, tuple)) or len(point) != 2:
                        raise ValueError(f"点标记格式错误: scene_id={scene_id}, point_key={point_key}, point={point}")
                    
                    color_key = f"{point_key}_color"
                    color = marker_data.get(color_key, "red")
                    x, y = int(point[0]), int(point[1])
                    bgr_color = COLOR_MAP.get(color, (0, 0, 255))
                    cv2.circle(img_b_bgr, (x, y), 5, bgr_color, -1)
                    cv2.circle(img_b_bgr, (x, y), 8, bgr_color, 2)
                
                img_b_array = cv2.cvtColor(img_b_bgr, cv2.COLOR_BGR2RGB)
                img_b = Image.fromarray(img_b_array)
                img_b_filename = f"{Path(image_b_name).stem}_point_{suffix_b}.jpg"
                img_b_path_save = scene_images_dir / img_b_filename
                _save_image(img_b, img_b_path_save, quality=95, skip_existing=skip_existing_images)
                image_paths.append(f"images/{scene_id}/{img_b_filename}")
    
    elif marker_type == "white_mask":
        # 需要单独的白色mask图片（如果marker_data中有对应图片的mask）
        # 处理object_relpos的多个mask（A, B, C）或object_dist的mask（1, 2）或其他情况的单个mask
        if image_a_path:
            img_a = Image.open(image_a_path)
            # 查找所有image_a的mask_rle（可能是image_a_mask_rle、image_a_mask_rle_A/B/C、image_a_mask_rle_1等）
            mask_keys_a = [k for k in entry.keys() if k.startswith("image_a_mask_rle") and not k.endswith("_color")]
            if not mask_keys_a:
                mask_keys_a = [k for k in marker_data.keys() if k.startswith("image_a_mask_rle") and not k.endswith("_color")]
            
            if mask_keys_a:
                img_a_masks = []
                for mask_key in sorted(mask_keys_a, key=lambda k: mask_key_sorter(k, "image_a")):
                    mask_rle = entry.get(mask_key) or marker_data.get(mask_key)
                    if not mask_rle:
                        continue
                    mask = decode_rle_mask(mask_rle)
                    if mask is None:
                        continue
                    img_h, img_w = img_a.size[1], img_a.size[0]
                    mask_h, mask_w = mask.shape
                    if mask_h != img_h or mask_w != img_w:
                        mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                    mask_binary = (mask > 0).astype(np.uint8)
                    white_mask_array = np.ones((mask_binary.shape[0], mask_binary.shape[1], 3), dtype=np.uint8) * 255
                    white_mask_array[:, :, 0] = white_mask_array[:, :, 0] * mask_binary
                    white_mask_array[:, :, 1] = white_mask_array[:, :, 1] * mask_binary
                    white_mask_array[:, :, 2] = white_mask_array[:, :, 2] * mask_binary
                    white_mask_a = Image.fromarray(white_mask_array)
                    label = extract_mask_label(mask_key, "image_a")
                    label_part = f"_{label}" if label else ""
                    img_a_filename = f"{Path(image_a_name).stem}_mask{label_part}_{suffix_a}.jpg"
                    img_a_path_save = scene_images_dir / img_a_filename
                    _save_image(white_mask_a, img_a_path_save, quality=95, skip_existing=skip_existing_images)
                    img_a_masks.append(f"images/{scene_id}/{img_a_filename}")
            
            # 还需要原图
            img_a_orig_filename = f"{Path(image_a_name).stem}.jpg"
            img_a_orig_path = scene_images_dir / img_a_orig_filename
            if not img_a_orig_path.exists():
                _copy_image(image_a_path, img_a_orig_path, skip_existing=skip_existing_images)
            image_paths.append(f"images/{scene_id}/{img_a_orig_filename}")
            if mask_keys_a:
                image_paths.extend(img_a_masks)
        
        if image_b_path:
            img_b = Image.open(image_b_path)
            # 查找所有image_b的mask_rle（可能是image_b_mask_rle、image_b_mask_rle_A/B/C、image_b_mask_rle_2等）
            mask_keys_b = [k for k in entry.keys() if k.startswith("image_b_mask_rle") and not k.endswith("_color")]
            if not mask_keys_b:
                mask_keys_b = [k for k in marker_data.keys() if k.startswith("image_b_mask_rle") and not k.endswith("_color")]
            
            if mask_keys_b:
                img_b_masks = []
                for mask_key in sorted(mask_keys_b, key=lambda k: mask_key_sorter(k, "image_b")):
                    mask_rle = entry.get(mask_key) or marker_data.get(mask_key)
                    if not mask_rle:
                        continue
                    mask = decode_rle_mask(mask_rle)
                    if mask is None:
                        continue
                    img_h, img_w = img_b.size[1], img_b.size[0]
                    mask_h, mask_w = mask.shape
                    if mask_h != img_h or mask_w != img_w:
                        mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                    mask_binary = (mask > 0).astype(np.uint8)
                    white_mask_array = np.ones((mask_binary.shape[0], mask_binary.shape[1], 3), dtype=np.uint8) * 255
                    white_mask_array[:, :, 0] = white_mask_array[:, :, 0] * mask_binary
                    white_mask_array[:, :, 1] = white_mask_array[:, :, 1] * mask_binary
                    white_mask_array[:, :, 2] = white_mask_array[:, :, 2] * mask_binary
                    white_mask_b = Image.fromarray(white_mask_array)
                    label = extract_mask_label(mask_key, "image_b")
                    label_part = f"_{label}" if label else ""
                    img_b_filename = f"{Path(image_b_name).stem}_mask{label_part}_{suffix_b}.jpg"
                    img_b_path_save = scene_images_dir / img_b_filename
                    _save_image(white_mask_b, img_b_path_save, quality=95, skip_existing=skip_existing_images)
                    img_b_masks.append(f"images/{scene_id}/{img_b_filename}")
            
            # 还需要原图
            img_b_orig_filename = f"{Path(image_b_name).stem}.jpg"
            img_b_orig_path = scene_images_dir / img_b_orig_filename
            if not img_b_orig_path.exists():
                _copy_image(image_b_path, img_b_orig_path, skip_existing=skip_existing_images)
            image_paths.append(f"images/{scene_id}/{img_b_orig_filename}")
            if mask_keys_b:
                image_paths.extend(img_b_masks)
    
    elif marker_type == "mask_covering_image":
        # 需要在原图上画mask overlay（如果marker_data中有对应图片的mask）
        if image_a_path:
            # 可能有多个mask（如object_relpos中的A, B, C）
            mask_keys = [k for k in marker_data.keys() if k.startswith("image_a_mask_rle_") and not k.endswith("_color")]
            if mask_keys:
                img_a = Image.open(image_a_path).convert('RGB')
                img_a_array = np.array(img_a)
                
                # 将RGB转换为BGR（因为OpenCV函数期望BGR格式）
                img_a_bgr = cv2.cvtColor(img_a_array, cv2.COLOR_RGB2BGR)
                
                for mask_key in mask_keys:
                    mask_rle = marker_data[mask_key]
                    color_key = f"{mask_key}_color"
                    color = marker_data.get(color_key, "red")
                    
                    mask = decode_rle_mask(mask_rle)
                    if mask is not None:
                        img_h, img_w = img_a_array.shape[:2]
                        mask_h, mask_w = mask.shape
                        if mask_h != img_h or mask_w != img_w:
                            mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                        
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        bgr_color = COLOR_MAP.get(color, (0, 0, 255))
                        cv2.drawContours(img_a_bgr, contours, -1, bgr_color, 3)
                
                # 转换回RGB格式
                img_a_array = cv2.cvtColor(img_a_bgr, cv2.COLOR_BGR2RGB)
                
                img_a = Image.fromarray(img_a_array)
                img_a_filename = f"{Path(image_a_name).stem}_mask_overlay_{suffix_a}.jpg"
                img_a_path_save = scene_images_dir / img_a_filename
                _save_image(img_a, img_a_path_save, quality=95, skip_existing=skip_existing_images)
                image_paths.append(f"images/{scene_id}/{img_a_filename}")
            else:
                # 回退到单个mask（如object_dpt）
                mask_rle_a = marker_data.get("image_a_mask_rle")
                if mask_rle_a:
                    img_a = draw_mask_overlay_on_image(image_a_path, mask_rle_a, marker_data.get("image_a_mask_rle_color", "red"))
                    img_a_filename = f"{Path(image_a_name).stem}_mask_overlay_{suffix_a}.jpg"
                    img_a_path_save = scene_images_dir / img_a_filename
                    _save_image(img_a, img_a_path_save, quality=95, skip_existing=skip_existing_images)
                    image_paths.append(f"images/{scene_id}/{img_a_filename}")
        
        if image_b_path:
            mask_keys = [k for k in marker_data.keys() if k.startswith("image_b_mask_rle_") and not k.endswith("_color")]
            if mask_keys:
                img_b = Image.open(image_b_path).convert('RGB')
                img_b_array = np.array(img_b)
                
                # 将RGB转换为BGR（因为OpenCV函数期望BGR格式）
                img_b_bgr = cv2.cvtColor(img_b_array, cv2.COLOR_RGB2BGR)
                
                for mask_key in mask_keys:
                    mask_rle = marker_data[mask_key]
                    color_key = f"{mask_key}_color"
                    color = marker_data.get(color_key, "red")
                    
                    mask = decode_rle_mask(mask_rle)
                    if mask is not None:
                        img_h, img_w = img_b_array.shape[:2]
                        mask_h, mask_w = mask.shape
                        if mask_h != img_h or mask_w != img_w:
                            mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                        
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        bgr_color = COLOR_MAP.get(color, (0, 0, 255))
                        cv2.drawContours(img_b_bgr, contours, -1, bgr_color, 3)
                
                # 转换回RGB格式
                img_b_array = cv2.cvtColor(img_b_bgr, cv2.COLOR_BGR2RGB)
                
                img_b = Image.fromarray(img_b_array)
                img_b_filename = f"{Path(image_b_name).stem}_mask_overlay_{suffix_b}.jpg"
                img_b_path_save = scene_images_dir / img_b_filename
                _save_image(img_b, img_b_path_save, quality=95, skip_existing=skip_existing_images)
                image_paths.append(f"images/{scene_id}/{img_b_filename}")
            else:
                mask_rle_b = marker_data.get("image_b_mask_rle")
                if mask_rle_b:
                    img_b = draw_mask_overlay_on_image(image_b_path, mask_rle_b, marker_data.get("image_b_mask_rle_color", "red"))
                    img_b_filename = f"{Path(image_b_name).stem}_mask_overlay_{suffix_b}.jpg"
                    img_b_path_save = scene_images_dir / img_b_filename
                    _save_image(img_b, img_b_path_save, quality=95, skip_existing=skip_existing_images)
                    image_paths.append(f"images/{scene_id}/{img_b_filename}")
                else:
                    # 如果image_b_path存在但没有image_b_mask_rle，说明这个条目可能只需要image_a的mask标记
                    # 但image_b仍然需要原图（问题中可能提到"Locate the same physical object in image B"）
                    img_b_filename = f"{Path(image_b_name).stem}.jpg"
                    img_b_path_save = scene_images_dir / img_b_filename
                    if not img_b_path_save.exists():
                        _copy_image(image_b_path, img_b_path_save, skip_existing=skip_existing_images)
                    image_paths.append(f"images/{scene_id}/{img_b_filename}")
    
    elif marker_type == "2d_bbox":
        # 需要在原图上画2D边界框（如果marker_data中有对应图片的bbox）
        if image_a_path:
            bbox_a = marker_data.get("image_a_bbox")
            color_a = marker_data.get("image_a_bbox_color", "red")
            
            # 可能有多个bbox（如object_relpos）
            bbox_keys = [k for k in marker_data.keys() if k.startswith("image_a_bbox_") and not k.endswith("_color")]
            
            if bbox_keys:
                img_a = Image.open(image_a_path).convert('RGB')
                draw = ImageDraw.Draw(img_a)
                
                for bbox_key in bbox_keys:
                    bbox = marker_data[bbox_key]
                    color_key = f"{bbox_key}_color"
                    color = marker_data.get(color_key, "red")
                    rgb_color = {
                        "red": (255, 0, 0),
                        "green": (0, 255, 0),
                        "blue": (0, 0, 255),
                    }.get(color, (255, 0, 0))
                    
                    x_min, y_min, x_max, y_max = bbox
                    draw.rectangle([int(x_min), int(y_min), int(x_max), int(y_max)], outline=rgb_color, width=3)
                
                img_a_filename = f"{Path(image_a_name).stem}_bbox_{suffix_a}.jpg"
                img_a_path_save = scene_images_dir / img_a_filename
                _save_image(img_a, img_a_path_save, quality=95, skip_existing=skip_existing_images)
                image_paths.append(f"images/{scene_id}/{img_a_filename}")
            elif bbox_a:
                img_a = draw_bbox_on_image(image_a_path, bbox_a, color_a)
                img_a_filename = f"{Path(image_a_name).stem}_bbox_{suffix_a}.jpg"
                img_a_path_save = scene_images_dir / img_a_filename
                _save_image(img_a, img_a_path_save, quality=95, skip_existing=skip_existing_images)
                image_paths.append(f"images/{scene_id}/{img_a_filename}")
        
        if image_b_path:
            bbox_b = marker_data.get("image_b_bbox")
            color_b = marker_data.get("image_b_bbox_color", "red")
            
            bbox_keys = [k for k in marker_data.keys() if k.startswith("image_b_bbox_") and not k.endswith("_color")]
            
            if bbox_keys:
                img_b = Image.open(image_b_path).convert('RGB')
                draw = ImageDraw.Draw(img_b)
                
                for bbox_key in bbox_keys:
                    bbox = marker_data[bbox_key]
                    color_key = f"{bbox_key}_color"
                    color = marker_data.get(color_key, "red")
                    rgb_color = {
                        "red": (255, 0, 0),
                        "green": (0, 255, 0),
                        "blue": (0, 0, 255),
                    }.get(color, (255, 0, 0))
                    
                    x_min, y_min, x_max, y_max = bbox
                    draw.rectangle([int(x_min), int(y_min), int(x_max), int(y_max)], outline=rgb_color, width=3)
                
                img_b_filename = f"{Path(image_b_name).stem}_bbox_{suffix_b}.jpg"
                img_b_path_save = scene_images_dir / img_b_filename
                _save_image(img_b, img_b_path_save, quality=95, skip_existing=skip_existing_images)
                image_paths.append(f"images/{scene_id}/{img_b_filename}")
            elif bbox_b:
                img_b = draw_bbox_on_image(image_b_path, bbox_b, color_b)
                img_b_filename = f"{Path(image_b_name).stem}_bbox_{suffix_b}.jpg"
                img_b_path_save = scene_images_dir / img_b_filename
                _save_image(img_b, img_b_path_save, quality=95, skip_existing=skip_existing_images)
                image_paths.append(f"images/{scene_id}/{img_b_filename}")
            else:
                # 如果image_b_path存在但没有image_b_bbox，说明这个条目可能只需要image_a的bbox标记
                # 但image_b仍然需要原图（问题中可能提到"Locate the same physical object in image B"）
                img_b_filename = f"{Path(image_b_name).stem}.jpg"
                img_b_path_save = scene_images_dir / img_b_filename
                if not img_b_path_save.exists():
                    _copy_image(image_b_path, img_b_path_save, skip_existing=skip_existing_images)
                image_paths.append(f"images/{scene_id}/{img_b_filename}")
    
    else:
        # language_description 或其他不需要特殊标记的情况，使用原图
        if image_a_path:
            img_a_filename = f"{Path(image_a_name).stem}.jpg"
            img_a_path = scene_images_dir / img_a_filename
            if not img_a_path.exists():
                _copy_image(image_a_path, img_a_path, skip_existing=skip_existing_images)
            image_paths.append(f"images/{scene_id}/{img_a_filename}")
        
        if image_b_path:
            img_b_filename = f"{Path(image_b_name).stem}.jpg"
            img_b_path = scene_images_dir / img_b_filename
            if not img_b_path.exists():
                _copy_image(image_b_path, img_b_path, skip_existing=skip_existing_images)
            image_paths.append(f"images/{scene_id}/{img_b_filename}")
    
    # 确保至少生成了一张图片
    if not image_paths:
        raise ValueError(f"未生成任何图片: scene_id={scene_id}, marker_type={marker_type}, image_a={image_a_name}, image_b={image_b_name}, image_a_path={image_a_path}, image_b_path={image_b_path}")
    
    return image_paths, metadata


def convert_qa_to_llamafactory_format(
    qa_json_path: Path,
    image_root: Path,
    output_dir: Path,
    scene_id: Optional[str] = None,
    entry_id_start: int = 0,
    jsonl_path: Optional[Path] = None,
    append: bool = True,
    show_progress: bool = True,
    skip_existing_images: bool = False,
) -> Tuple[int, int]:
    """转换单个QA JSON文件为llamafactory格式"""
    # 读取QA JSON
    with open(qa_json_path, 'r', encoding='utf-8') as f:
        qa_entries = json.load(f)
    
    if not isinstance(qa_entries, list):
        qa_entries = [qa_entries]
    
    # 创建输出目录
    output_images_dir = output_dir / "images"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    
    # 准备JSONL输出
    jsonl_path = jsonl_path or (output_dir / "dataset.jsonl")
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    entry_id = entry_id_start
    jsonl_file = open(jsonl_path, 'a' if append else 'w', encoding='utf-8')
    
    processed_count = 0
    
    for entry in tqdm(qa_entries, desc=f"处理 {qa_json_path.name}", disable=not show_progress):
        entry_scene_id = entry.get("scene_id", "")
        if scene_id and entry_scene_id != scene_id:
            continue
        
        # 生成图片（如果失败会直接抛出异常）
        image_paths, metadata = generate_image_for_entry(
            entry,
            image_root,
            entry_scene_id,
            output_images_dir,
            {},
            skip_existing_images=skip_existing_images,
        )
        
        # 构建conversations
        question = entry.get("question", "")
        answer = entry.get("answer", "")
        
        # 根据图片数量构建value：多张图片时每张图片前加<image>标记
        if len(image_paths) == 1:
            human_value = f"<image>\n{question}"
        else:
            # 多张图片：在问题中插入<image>标记
            # 假设问题中已经包含了图片的引用（如"In image A..."），我们只需要在开头添加<image>标记
            human_value = "\n".join(["<image>"] * len(image_paths)) + f"\n{question}"
        
        conversations = [
            {
                "from": "human",
                "value": human_value
            },
            {
                "from": "gpt",
                "value": answer
            }
        ]
        
        # 构建llamafactory格式的条目
        llamafactory_entry = {
            "id": entry_id,
            "conversations": conversations,
            "image": image_paths,
            "question_type": entry.get("question_type", ""),
            "sub_question_type": entry.get("sub_question_type", "")
        }
        
        # 写入JSONL
        jsonl_file.write(json.dumps(llamafactory_entry, ensure_ascii=False) + '\n')
        entry_id += 1
        processed_count += 1
    
    jsonl_file.close()
    return processed_count, entry_id


def merge_jsonl_parts(parts_dir: Path, qa_json_files: List[Path], output_jsonl_path: Path) -> int:
    """合并各场景临时jsonl，重排全局id."""
    entry_id = 0
    with open(output_jsonl_path, 'w', encoding='utf-8') as out_f:
        for qa_json_path in qa_json_files:
            part_path = parts_dir / f"{qa_json_path.stem}.jsonl"
            if not part_path.exists():
                print(f"未找到临时jsonl: {part_path}")
                continue
            with open(part_path, 'r', encoding='utf-8') as in_f:
                for line in in_f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    entry["id"] = entry_id
                    out_f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    entry_id += 1
    return entry_id


def main():
    parser = argparse.ArgumentParser(description="将QA JSON转换为llamafactory格式")
    parser.add_argument(
        "--image-root",
        type=Path,
        required=True,
        help="原始图片根目录（如 scannetppv2/data）"
    )
    parser.add_argument(
        "--qa-json-dir",
        type=Path,
        required=True,
        help="QA JSON目录（如 output_QA）"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="输出数据集目录"
    )
    parser.add_argument(
        "--scene-id",
        type=str,
        default=None,
        help="可选：只处理指定场景ID（用于验证）"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 4)),
        help="并行处理场景数（仅在处理全部场景时生效）"
    )
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not args.image_root.exists():
        raise ValueError(f"图片根目录不存在: {args.image_root}")
    if not args.qa_json_dir.exists():
        raise ValueError(f"QA JSON目录不存在: {args.qa_json_dir}")
    
    # 创建输出目录
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 删除已有的dataset.jsonl，避免重复追加
    jsonl_path = args.output_dir / "dataset.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()
        print(f"已删除已有的 {jsonl_path}")
    
    # 如果指定了scene_id，只处理该场景
    if args.scene_id:
        qa_json_path = args.qa_json_dir / f"{args.scene_id}.json"
        if not qa_json_path.exists():
            raise ValueError(f"QA JSON文件不存在: {qa_json_path}")
        
        print(f"处理单个场景: {args.scene_id}")
        count, _ = convert_qa_to_llamafactory_format(
            qa_json_path,
            args.image_root,
            args.output_dir,
            args.scene_id,
            entry_id_start=0,
            jsonl_path=jsonl_path,
            append=False,
        )
        print(f"处理完成，共 {count} 条QA")
    else:
        # 处理所有场景
        qa_json_files = sorted(args.qa_json_dir.glob("*.json"))
        print(f"找到 {len(qa_json_files)} 个QA JSON文件")
        
        total_count = 0
        entry_id = 0
        if args.num_workers <= 1 or len(qa_json_files) <= 1:
            for qa_json_path in tqdm(qa_json_files, desc="处理所有场景"):
                count, entry_id = convert_qa_to_llamafactory_format(
                    qa_json_path,
                    args.image_root,
                    args.output_dir,
                    None,
                    entry_id_start=entry_id,
                    jsonl_path=jsonl_path,
                    append=True,
                )
                total_count += count
        else:
            parts_dir = args.output_dir / "jsonl_parts"
            parts_dir.mkdir(parents=True, exist_ok=True)
            for part_file in parts_dir.glob("*.jsonl"):
                part_file.unlink()

            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                future_to_scene = {}
                for qa_json_path in qa_json_files:
                    part_path = parts_dir / f"{qa_json_path.stem}.jsonl"
                    future = executor.submit(
                        convert_qa_to_llamafactory_format,
                        qa_json_path,
                        args.image_root,
                        args.output_dir,
                        None,
                        0,
                        part_path,
                        False,
                        False,
                    )
                    future_to_scene[future] = qa_json_path

                for future in tqdm(as_completed(future_to_scene), total=len(future_to_scene), desc="并行处理"):
                    qa_json_path = future_to_scene[future]
                    try:
                        count, _ = future.result()
                        total_count += count
                    except Exception as exc:  # noqa: BLE001
                        raise RuntimeError(f"处理失败: {qa_json_path}") from exc

            entry_id = merge_jsonl_parts(parts_dir, qa_json_files, jsonl_path)
        
        print(f"处理完成，共 {total_count} 条QA")


if __name__ == "__main__":
    main()
