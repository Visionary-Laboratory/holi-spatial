#!/usr/bin/env python3
"""
使用 Sa2VA-Qwen2_5-VL-7B 模型进行 2D IoU 计算
1. 读取 GT JSON 文件
2. 根据 GT JSON 中的图片和 label，使用 Sa2VA 模型进行分割
3. 合并 Sa2VA 输出的同一 label 下所有实例的 mask 作为 pred_mask
4. 计算每个图每个 label 的 IoU
5. 计算整个场景的平均 IoU
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
from tqdm import tqdm
import pycocotools.mask as mask_utils
import torch

# 导入 Sa2VA
from Sa2VA import load_sa2va_model, predict_mask_sa2va, extract_masks_from_sa2va_output


def setup_logger():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def decode_rle_mask(rle_dict: Dict) -> np.ndarray:
    """解码COCO RLE格式的mask，参考 3d_bounding_instance_gs_rerun.py 的实现"""
    rle = rle_dict  # rle_dict 本身就是 RLE 格式
    mask = mask_utils.decode(rle).astype(bool)
    return mask.astype(np.uint8) * 255  # 转换为 uint8 格式 (0 或 255)


def encode_mask_to_rle(mask: np.ndarray) -> Dict:
    """
    将mask编码为COCO RLE格式
    mask: (height, width) uint8 格式 (0 或 255)
    返回: RLE 字典，格式为 {"size": [height, width], "counts": "..."}
    """
    # 转换为布尔值
    mask_binary = (mask > 0).astype(np.uint8)
    height, width = mask_binary.shape
    
    # 使用 pycocotools 编码
    rle = mask_utils.encode(np.asfortranarray(mask_binary))
    
    # 确保格式正确：size 应该是列表 [height, width]，counts 应该是字符串
    if 'size' in rle:
        size = rle['size']
        # 确保 size 是列表格式 [height, width]
        if isinstance(size, (list, tuple, np.ndarray)):
            size = [int(size[0]), int(size[1])]
        else:
            size = [int(height), int(width)]
    else:
        size = [int(height), int(width)]
    
    # 确保 counts 是字符串格式
    if 'counts' in rle:
        counts = rle['counts']
        if isinstance(counts, bytes):
            counts = counts.decode('utf-8')
        elif not isinstance(counts, str):
            counts = str(counts)
    else:
        counts = ""
    
    return {
        "size": size,
        "counts": counts
    }


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


def merge_masks(masks: List[np.ndarray]) -> np.ndarray:
    """
    合并多个 mask（同一 label 的所有实例）
    """
    if not masks:
        return np.array([])
    
    # 合并所有 mask（使用逻辑或）
    merged = np.zeros_like(masks[0], dtype=np.uint8)
    for mask in masks:
        merged = np.logical_or(merged, mask > 0).astype(np.uint8) * 255
    
    return merged


def get_image_path_map(scene_dir: Path) -> Dict[str, Path]:
    """
    根据场景类型获取图像路径映射
    支持 scannetppv2 (dslr), scannet (cam/color), dl3dv (dense) 三种格式
    返回: {image_stem: image_path}
    """
    image_path_map: Dict[str, Path] = {}
    
    # scannetppv2 格式
    if (scene_dir / "dslr").exists():
        image_base_dir = scene_dir / "dslr" / "resized_undistorted_images"
        if image_base_dir.exists():
            # 列出所有图像文件
            for img_file in image_base_dir.iterdir():
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    stem = img_file.stem
                    image_path_map[stem] = img_file
        logging.info(f"加载 scannetppv2 格式图像路径，数量: {len(image_path_map)}")
        return image_path_map
    
    # scannet 格式
    elif (scene_dir / "cam").exists() and (scene_dir / "color").exists():
        color_dir = scene_dir / "color"
        if color_dir.exists():
            # 列出所有图像文件
            for img_file in color_dir.iterdir():
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    stem = img_file.stem
                    image_path_map[stem] = img_file
        logging.info(f"加载 scannet 格式图像路径，数量: {len(image_path_map)}")
        return image_path_map
    
    # dl3dv 格式
    elif (scene_dir / "dense").exists():
        rgb_dir = scene_dir / "dense" / "rgb"
        if rgb_dir.exists():
            # 列出所有图像文件
            for img_file in rgb_dir.iterdir():
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    stem = img_file.stem
                    image_path_map[stem] = img_file
        logging.info(f"加载 dl3dv 格式图像路径，数量: {len(image_path_map)}")
        return image_path_map
    
    else:
        logging.warning(f"不支持的场景格式: {scene_dir}, 目前只支持 scannetppv2 (dslr), scannet (cam/color), dl3dv (dense)")
        return image_path_map


def process_scene(
    scene: str,
    data_root: Path,
    gt_json_path: Path,
    scene_objects_json_path: Optional[Path],
    output_dir: Path,
    sa2va_model,
    sa2va_tokenizer,
    sa2va_processor
):
    """处理单个场景"""
    logging.info(f"处理场景: {scene}")
    
    # 1. 加载GT masks（用于选择图片和获取GT mask）
    gt_masks = load_gt_masks(gt_json_path)
    logging.info(f"加载了 {len(gt_masks)} 张图像的GT masks")
    
    # 2. 加载 scene_objects JSON 中的 label 列表（用于确定每张图片要处理的 label）
    scene_objects_labels: Optional[Dict[str, List[str]]] = None
    if scene_objects_json_path is not None and scene_objects_json_path.exists():
        try:
            scene_objects_labels = load_scene_objects_labels(scene_objects_json_path)
            logging.info(f"加载了 {len(scene_objects_labels)} 张图像的 scene_objects labels")
        except Exception as e:
            logging.warning(f"加载 scene_objects 失败: {e}，将从 GT masks 加载全部 labels")
            scene_objects_labels = None
    
    # 如果没有 scene_objects JSON，从 GT masks 中提取所有 labels
    if scene_objects_labels is None:
        logging.info("scene_objects JSON 不存在，从 GT masks 加载全部 labels")
        scene_objects_labels = {}
        for image_name, gt_label_masks in gt_masks.items():
            # 提取该图像的所有 GT labels
            scene_objects_labels[image_name] = list(gt_label_masks.keys())
        logging.info(f"从 GT masks 提取了 {len(scene_objects_labels)} 张图像的 labels")
    
    # 3. 获取场景目录和图像路径映射
    scene_dir = data_root / scene
    image_path_map = get_image_path_map(scene_dir)
    
    # 4. 为每张图像、每个label计算IoU
    results = []
    pred_mask_rles = []  # 收集所有预测 mask 的 RLE
    
    # 遍历 GT masks 中的图片（通过 gt-json 选择图片）
    for image_name, gt_label_masks in tqdm(gt_masks.items(), desc="处理图像"):
        # 尝试匹配图像名称（可能带扩展名或不带扩展名）
        image_stem = Path(image_name).stem  # 去掉扩展名
        
        # 先尝试完整名称
        if image_name in image_path_map:
            image_path = image_path_map[image_name]
        # 再尝试 stem（不含扩展名）
        elif image_stem in image_path_map:
            image_path = image_path_map[image_stem]
        else:
            logging.warning(f"图像 {image_name} (stem: {image_stem}) 没有找到对应的图像文件，跳过")
            continue
        
        if not image_path.exists():
            logging.warning(f"图像文件不存在: {image_path}")
            continue
        
        # 加载图像
        try:
            image = Image.open(image_path).convert("RGB")
            image_size = (image.height, image.width)
        except Exception as e:
            logging.warning(f"加载图像失败 {image_name}: {e}")
            continue
        
        # 获取该图片在 scene_objects JSON 中的 label 列表
        if scene_objects_labels is not None:
            scene_labels = scene_objects_labels.get(image_name, [])
            scene_labels_set = set(scene_labels)
            logging.debug(f"图像 {image_name} 在 scene_objects 中有 {len(scene_labels)} 个 labels")
        else:
            # 如果没有 scene_objects，使用 GT masks 中的所有 labels
            scene_labels_set = set(gt_label_masks.keys())
            logging.debug(f"图像 {image_name} 使用 GT masks 中的所有 labels: {len(scene_labels_set)} 个")
        
        # 为每个 GT label 计算 IoU
        for gt_label, gt_mask in gt_label_masks.items():
            # 检查 scene_objects 中是否包含该 label（如果 scene_objects_labels 不为 None）
            if scene_objects_labels is not None and gt_label not in scene_labels_set:
                # 如果不包含，IoU 为 0，创建一个空 mask RLE
                empty_mask = np.zeros(gt_mask.shape, dtype=np.uint8)
                pred_mask_rle = encode_mask_to_rle(empty_mask)
                pred_mask_rles.append({
                    "image": image_name,
                    "label": gt_label,
                    "mask_rle": pred_mask_rle
                })
                results.append({
                    "image": image_name,
                    "label": gt_label,
                    "iou": 0.0,
                    "pred_mask_area": 0,
                    "gt_mask_area": int(np.sum(gt_mask > 0)),
                    "num_instances": 0,
                    "note": "label not in scene_objects"
                })
                continue
            
            # 如果包含，使用 Sa2VA 进行分割π
            try:
                # 构建文本提示
                text_prompt = f"<image>Please segment all the {gt_label} in the image."
                
                # 使用 Sa2VA 进行预测
                return_dict = predict_mask_sa2va(
                    sa2va_model,
                    sa2va_tokenizer,
                    sa2va_processor,
                    image,
                    text_prompt,
                    mask_prompts=None
                )
                
                # 从输出中提取 masks
                instance_masks = extract_masks_from_sa2va_output(return_dict, image_size)
                
                # 如果没有提取到 mask，尝试从 prediction 文本中解析
                if not instance_masks:
                    # 如果模型返回的是文本描述，可能需要其他方式处理
                    # 这里先创建一个空 mask
                    logging.warning(f"图像 {image_name} 标签 {gt_label} Sa2VA 未返回 mask，尝试其他方法")
                    # 可以尝试使用其他提示或方法
                    instance_masks = []
                
                # 合并同一 label 的所有实例 mask
                if instance_masks:
                    pred_mask = merge_masks(instance_masks)
                else:
                    # 如果没有检测到实例，创建空 mask
                    pred_mask = np.zeros(gt_mask.shape, dtype=np.uint8)
                    logging.warning(f"图像 {image_name} 标签 {gt_label} Sa2VA 未检测到实例")
                
                # 调整 pred_mask 尺寸以匹配 GT mask
                if pred_mask.shape[0] != gt_mask.shape[0] or pred_mask.shape[1] != gt_mask.shape[1]:
                    pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                # 计算IoU
                iou = compute_2d_iou(pred_mask, gt_mask)
                
                # 将预测 mask 编码为 RLE
                pred_mask_rle = encode_mask_to_rle(pred_mask)
                pred_mask_rles.append({
                    "image": image_name,
                    "label": gt_label,
                    "mask_rle": pred_mask_rle
                })
                
                results.append({
                    "image": image_name,
                    "label": gt_label,
                    "iou": float(iou),
                    "pred_mask_area": int(np.sum(pred_mask > 0)),
                    "gt_mask_area": int(np.sum(gt_mask > 0)),
                    "num_instances": len(instance_masks),
                    "note": "computed"
                })
                
            except Exception as e:
                logging.warning(f"处理图像 {image_name} 标签 {gt_label} 失败: {e}")
                import traceback
                traceback.print_exc()
                # 失败时也记录 IoU 为 0，创建一个空 mask RLE
                empty_mask = np.zeros(gt_mask.shape, dtype=np.uint8)
                pred_mask_rle = encode_mask_to_rle(empty_mask)
                pred_mask_rles.append({
                    "image": image_name,
                    "label": gt_label,
                    "mask_rle": pred_mask_rle
                })
                results.append({
                    "image": image_name,
                    "label": gt_label,
                    "iou": 0.0,
                    "pred_mask_area": 0,
                    "gt_mask_area": int(np.sum(gt_mask > 0)),
                    "num_instances": 0,
                    "note": f"error: {str(e)}"
                })
                continue
    
    # 5. 保存结果和统计信息
    output_json = output_dir / f"{scene}_2d_iou_sa2va.json"
    
    # 保存预测 mask RLE 的 JSON 文件（格式与 mask_index_outputs/0a7cc12c0e.json 相同）
    pred_mask_rle_json = output_dir / f"{scene}_pred_mask_rle.json"
    with pred_mask_rle_json.open("w", encoding="utf-8") as f:
        json.dump(pred_mask_rles, f, indent=2, ensure_ascii=False)
    logging.info(f"预测 mask RLE 已保存到: {pred_mask_rle_json} (共 {len(pred_mask_rles)} 个)")
    
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
    parser = argparse.ArgumentParser(description="使用 Sa2VA 计算2D IoU")
    parser.add_argument("--scene", type=str, required=True, help="场景名称，如 0a7cc12c0e")
    parser.add_argument("--data-root", type=Path, default=Path("scannetppv2/data"), help="数据根目录")
    parser.add_argument("--gt-json", type=Path, help="GT mask JSON文件路径，如 mask_index_outputs/0a7cc12c0e.json")
    parser.add_argument("--scene-objects-json", type=Path, help="scene_objects JSON文件路径，如 scene_objects_Qwen3-VL-30B-A3B-Instruct/0a7cc12c0e.json")
    parser.add_argument("--output-dir", type=Path, default=Path("2d_iou_sa2va_outputs"), help="输出目录")
    parser.add_argument("--model-path", type=Path, default=Path("/mnt/shared-storage-user/solution/huggingface/hub/models--ByteDance--Sa2VA-Qwen2_5-VL-7B/snapshots/4cd6709067cf257235e922c201a6f265292f5fc8"), help="Sa2VA 模型路径")
    args = parser.parse_args()
    
    setup_logger()
    
    # 设置默认路径
    if args.gt_json is None:
        args.gt_json = Path("mask_index_outputs") / f"{args.scene}.json"
    
    if args.scene_objects_json is None:
        args.scene_objects_json = Path("scene_objects_Qwen3-VL-30B-A3B-Instruct") / f"{args.scene}.json"
    
    if not args.gt_json.exists():
        logging.error(f"GT JSON文件不存在: {args.gt_json}")
        return
    
    # scene_objects JSON 是可选的，如果不存在会从 GT masks 加载全部 labels
    if args.scene_objects_json is not None and not args.scene_objects_json.exists():
        logging.warning(f"scene_objects JSON文件不存在: {args.scene_objects_json}，将从 GT masks 加载全部 labels")
        args.scene_objects_json = None
    
    if not args.model_path.exists():
        logging.error(f"Sa2VA 模型文件不存在: {args.model_path}")
        return
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载 Sa2VA 模型
    logging.info("加载 Sa2VA 模型...")
    try:
        model, tokenizer, processor = load_sa2va_model(str(args.model_path))
        logging.info("Sa2VA 模型加载成功")
    except Exception as e:
        logging.error(f"Sa2VA 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    process_scene(
        scene=args.scene,
        data_root=args.data_root,
        gt_json_path=args.gt_json,
        scene_objects_json_path=args.scene_objects_json,
        output_dir=args.output_dir,
        sa2va_model=model,
        sa2va_tokenizer=tokenizer,
        sa2va_processor=processor
    )


if __name__ == "__main__":
    main()


