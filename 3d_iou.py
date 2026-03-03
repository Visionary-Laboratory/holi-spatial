#!/usr/bin/env python3
"""
计算3D IoU：比较 recover_labels (GT) 和 output_3d_bounding (预测) 的bounding box
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import trimesh
from collections import defaultdict
from tqdm import tqdm

def corners_to_mesh(corners: List[Dict]) -> trimesh.Trimesh:
    """从8个角点创建trimesh mesh"""
    # 提取8个角点的坐标
    points = np.array([
        [corner["x"], corner["y"], corner["z"]]
        for corner in corners
    ], dtype=np.float32)
    
    if len(points) != 8:
        raise ValueError(f"Expected 8 corners, got {len(points)}")
    
    # 使用凸包创建mesh（8个角点应该构成一个OBB）
    try:
        # 使用trimesh的凸包功能
        mesh = trimesh.convex.convex_hull(points)
        return mesh
    except Exception:
        # 如果凸包失败，尝试使用bounds创建box（fallback）
        min_bounds = np.min(points, axis=0)
        max_bounds = np.max(points, axis=0)
        extents = max_bounds - min_bounds
        center = (min_bounds + max_bounds) / 2.0
        transform = np.eye(4)
        transform[:3, 3] = center
        box = trimesh.creation.box(extents=extents, transform=transform)
        return box

def point_in_obb(point: np.ndarray, transform: np.ndarray, extents: np.ndarray) -> bool:
    """检查点是否在OBB内（使用局部坐标系）"""
    # 将点转换到OBB的局部坐标系
    transform_inv = np.linalg.inv(transform)
    point_homo = np.append(point, 1.0)
    local_point = transform_inv @ point_homo
    local_point = local_point[:3]
    
    # 检查点是否在[-extents/2, extents/2]范围内
    half_extents = extents / 2.0
    return np.all(np.abs(local_point) <= half_extents + 1e-6)

def compute_3d_iou(box1: trimesh.Trimesh, box2: trimesh.Trimesh, num_samples: int = 5000) -> float:
    """使用采样方法计算两个OBB的3D IoU（不依赖trimesh.contains）"""
    try:
        # 从mesh中提取transform和extents（需要从创建时的参数获取）
        # 由于我们无法直接从mesh获取，我们需要传入transform和extents
        # 但这里我们使用bounds来估算
        
        # 计算两个box的体积
        vol1 = box1.volume
        vol2 = box2.volume
        
        if vol1 < 1e-9 or vol2 < 1e-9:
            return 0.0
        
        # 获取两个box的联合边界
        bounds1 = box1.bounds
        bounds2 = box2.bounds
        min_bounds = np.minimum(bounds1[0], bounds2[0])
        max_bounds = np.maximum(bounds1[1], bounds2[1])
        
        # 在联合边界内均匀采样
        dims = max_bounds - min_bounds
        if np.any(dims < 1e-6):
            return 0.0
        
        # 计算采样密度
        total_volume = np.prod(dims)
        samples_per_unit_volume = num_samples / total_volume
        
        # 计算每个维度的采样点数
        n_per_dim = int(np.ceil(np.power(samples_per_unit_volume * np.max(dims)**3, 1/3)))
        n_per_dim = max(10, min(n_per_dim, 30))  # 减少采样点以提高速度
        
        # 生成网格点
        x = np.linspace(min_bounds[0], max_bounds[0], n_per_dim)
        y = np.linspace(min_bounds[1], max_bounds[1], n_per_dim)
        z = np.linspace(min_bounds[2], max_bounds[2], n_per_dim)
        
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
        
        # 使用mesh的vertices和faces来检查点是否在内部
        # 使用ray casting方法（简化版）
        try:
            inside1 = box1.ray.contains_points(points)
            inside2 = box2.ray.contains_points(points)
        except Exception:
            # 如果ray casting失败，使用bounds作为fallback
            inside1 = np.all((points >= bounds1[0]) & (points <= bounds1[1]), axis=1)
            inside2 = np.all((points >= bounds2[0]) & (points <= bounds2[1]), axis=1)
        
        # 计算交集和并集
        inter_count = np.sum(inside1 & inside2)
        union_count = np.sum(inside1 | inside2)
        
        if union_count == 0:
            return 0.0
        
        # 使用计数比例估算体积
        sample_volume = total_volume / len(points)
        inter_volume = inter_count * sample_volume
        union_volume = union_count * sample_volume
        
        if union_volume < 1e-9:
            return 0.0
        
        iou = inter_volume / union_volume
        return float(max(0.0, min(1.0, iou)))
    except Exception:
        return 0.0

def load_bboxes(json_path: Path) -> List[Dict]:
    """加载bounding box JSON文件"""
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []

def labels_match(gt_label: str, pred_label: str) -> bool:
    """判断label是否匹配：gt标签是pred标签的子串或完全一致"""
    gl = (gt_label or "unknown").lower().strip()
    pl = (pred_label or "unknown").lower().strip()
    if gl == pl:
        return True
    if gl and gl in pl:
        return True
    return False


def match_boxes(gt_boxes: List[Dict], pred_boxes: List[Dict], iou_threshold: float = 0.1) -> List[Tuple[Dict, Dict, float]]:
    """
    匹配GT和预测的bounding box
    
    label匹配规则：只要GT的label是预测label的子串（或完全一致）就算同类。
    """
    matches = []

    # 预先构建mesh，避免重复计算
    gt_meshes = []
    for box in gt_boxes:
        try:
            if "bounding_box" not in box or len(box["bounding_box"]) != 8:
                gt_meshes.append(None)
                continue
            gt_meshes.append(corners_to_mesh(box["bounding_box"]))
        except Exception:
            gt_meshes.append(None)

    pred_meshes = []
    for box in pred_boxes:
        try:
            if "bounding_box" not in box or len(box["bounding_box"]) != 8:
                pred_meshes.append(None)
                continue
            pred_meshes.append(corners_to_mesh(box["bounding_box"]))
        except Exception:
            pred_meshes.append(None)

    # 计算IoU矩阵（仅当label匹配时）
    iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
    label_stats = defaultdict(lambda: {"gt": 0, "pred": 0, "max_iou": 0.0})

    for i, gt_box in enumerate(gt_boxes):
        gt_label = gt_box.get("label", "unknown")
        label_stats[gt_label]["gt"] += 1
        gt_mesh = gt_meshes[i]
        if gt_mesh is None:
            continue

        for j, pred_box in enumerate(pred_boxes):
            pred_label = pred_box.get("label", "unknown")
            if not labels_match(gt_label, pred_label):
                continue

            label_stats[gt_label]["pred"] += 1
            pred_mesh = pred_meshes[j]
            if pred_mesh is None:
                continue

            try:
                iou = compute_3d_iou(gt_mesh, pred_mesh)
                iou_matrix[i, j] = iou
                label_stats[gt_label]["max_iou"] = max(label_stats[gt_label]["max_iou"], iou)
            except Exception:
                continue

    # 调试信息：打印最大IoU
    for lbl, st in label_stats.items():
        if st["gt"] > 0 and st["pred"] > 0:
            print(f"  Label '{lbl}': {st['gt']} GT, {st['pred']} pred, max IoU: {st['max_iou']:.4f}")

    # 贪心匹配
    used_gt = set()
    used_pred = set()
    candidates = []
    for i in range(len(gt_boxes)):
        for j in range(len(pred_boxes)):
            if iou_matrix[i, j] >= iou_threshold:
                candidates.append((iou_matrix[i, j], i, j))

    candidates.sort(reverse=True)

    for iou, i, j in candidates:
        if i not in used_gt and j not in used_pred:
            matches.append((gt_boxes[i], pred_boxes[j], iou))
            used_gt.add(i)
            used_pred.add(j)

    return matches

def compute_scene_iou(gt_path: Path, pred_path: Path, iou_threshold: float = 0.1, precision_recall_threshold: float = 0.5) -> Dict:
    """计算单个场景的IoU统计，包括precision、recall和F1 score"""
    scene_name = gt_path.stem
    
    gt_boxes = load_bboxes(gt_path)
    pred_boxes = load_bboxes(pred_path)
    
    num_gt = len(gt_boxes)
    num_pred = len(pred_boxes)
    
    if not gt_boxes:
        return {
            "scene": scene_name,
            "num_gt": 0,
            "num_pred": num_pred,
            "num_matched": 0,
            "mean_iou": 0.0,
            "median_iou": 0.0,
            "iou_list": [],
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "tp": 0,
            "fp": num_pred,
            "fn": 0
        }
    
    if not pred_boxes:
        return {
            "scene": scene_name,
            "num_gt": num_gt,
            "num_pred": 0,
            "num_matched": 0,
            "mean_iou": 0.0,
            "median_iou": 0.0,
            "iou_list": [],
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": num_gt
        }
    
    # 使用较小的阈值进行匹配，以便计算所有可能的IoU
    matches = match_boxes(gt_boxes, pred_boxes, iou_threshold)
    
    if not matches:
        return {
            "scene": scene_name,
            "num_gt": num_gt,
            "num_pred": num_pred,
            "num_matched": 0,
            "mean_iou": 0.0,
            "median_iou": 0.0,
            "iou_list": [],
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "tp": 0,
            "fp": num_pred,
            "fn": num_gt
        }
    
    iou_values = [iou for _, _, iou in matches]
    
    # 计算 precision、recall 和 F1 score（使用 precision_recall_threshold）
    # TP: 匹配且IoU >= precision_recall_threshold 的实例数
    # FP: 未匹配的预测实例数 + IoU < precision_recall_threshold 的匹配数（这些匹配质量不够，算作FP）
    # FN: 未匹配的GT实例数 + IoU < precision_recall_threshold 的匹配数（这些匹配质量不够，算作FN）
    tp = sum(1 for iou in iou_values if iou >= precision_recall_threshold)
    low_iou_matches = sum(1 for iou in iou_values if iou < precision_recall_threshold)
    fp = (num_pred - len(matches)) + low_iou_matches  # 未匹配的预测 + 低IoU匹配
    fn = (num_gt - len(matches)) + low_iou_matches    # 未匹配的GT + 低IoU匹配
    
    # 计算 precision、recall 和 F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "scene": scene_name,
        "num_gt": num_gt,
        "num_pred": num_pred,
        "num_matched": len(matches),
        "mean_iou": float(np.mean(iou_values)),
        "median_iou": float(np.median(iou_values)),
        "std_iou": float(np.std(iou_values)),
        "min_iou": float(np.min(iou_values)),
        "max_iou": float(np.max(iou_values)),
        "iou_list": [float(x) for x in iou_values],
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "matches": [
            {
                "gt_label": gt["label"],
                "gt_ins_id": gt.get("ins_id", "unknown"),
                "pred_label": pred["label"],
                "pred_ins_id": pred.get("ins_id", "unknown"),
                "iou": float(iou)
            }
            for gt, pred, iou in matches
        ]
    }

def main():
    parser = argparse.ArgumentParser(description="计算3D IoU")
    parser.add_argument("--gt-dir", type=Path, default=Path("recover_labels"), help="GT bounding box目录")
    parser.add_argument("--pred-dir", type=Path, default=Path("output_3d_bounding"), help="预测bounding box目录")
    parser.add_argument("--output", type=Path, default=Path("3d_iou_results_scannet.json"), help="输出结果JSON文件")
    parser.add_argument("--iou-threshold", type=float, default=0.01, help="匹配的最小IoU阈值")
    parser.add_argument("--precision-recall-threshold", type=float, default=0.5, help="计算precision/recall/F1的最小IoU阈值")
    args = parser.parse_args()
    
    gt_dir = args.gt_dir
    pred_dir = args.pred_dir
    
    # 获取所有GT场景
    gt_files = sorted(gt_dir.glob("*.json"))
    
    if not gt_files:
        print(f"Error: 在 {gt_dir} 中未找到JSON文件")
        return
    
    print(f"找到 {len(gt_files)} 个GT场景")
    
    all_results = []
    
    for gt_path in tqdm(gt_files, desc="计算IoU"):
        scene_name = gt_path.stem
        pred_path = pred_dir / f"{scene_name}.json"
        
        if not pred_path.exists():
            print(f"Warning: 预测文件不存在: {pred_path}")
            all_results.append({
                "scene": scene_name,
                "num_gt": len(load_bboxes(gt_path)),
                "num_pred": 0,
                "num_matched": 0,
                "mean_iou": 0.0,
                "error": "prediction file not found"
            })
            continue
        
        try:
            result = compute_scene_iou(gt_path, pred_path, args.iou_threshold, args.precision_recall_threshold)
            all_results.append(result)
        except Exception as e:
            print(f"Error processing {scene_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "scene": scene_name,
                "error": str(e)
            })
    
    # 计算总体统计
    valid_results = [r for r in all_results if "mean_iou" in r and "error" not in r]
    
    if valid_results:
        all_ious = []
        for r in valid_results:
            all_ious.extend(r.get("iou_list", []))
        
        # 计算总体 precision、recall 和 F1
        total_tp = sum(r.get("tp", 0) for r in valid_results)
        total_fp = sum(r.get("fp", 0) for r in valid_results)
        total_fn = sum(r.get("fn", 0) for r in valid_results)
        
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
        
        # 计算每个场景的平均 precision、recall 和 F1
        avg_precision = float(np.mean([r.get("precision", 0.0) for r in valid_results]))
        avg_recall = float(np.mean([r.get("recall", 0.0) for r in valid_results]))
        avg_f1 = float(np.mean([r.get("f1_score", 0.0) for r in valid_results]))
        
        summary = {
            "total_scenes": len(all_results),
            "valid_scenes": len(valid_results),
            "total_gt_boxes": sum(r.get("num_gt", 0) for r in valid_results),
            "total_pred_boxes": sum(r.get("num_pred", 0) for r in valid_results),
            "total_matched": sum(r.get("num_matched", 0) for r in valid_results),
            "overall_mean_iou": float(np.mean(all_ious)) if all_ious else 0.0,
            "overall_median_iou": float(np.median(all_ious)) if all_ious else 0.0,
            "overall_std_iou": float(np.std(all_ious)) if all_ious else 0.0,
            "overall_min_iou": float(np.min(all_ious)) if all_ious else 0.0,
            "overall_max_iou": float(np.max(all_ious)) if all_ious else 0.0,
            "overall_precision": float(overall_precision),
            "overall_recall": float(overall_recall),
            "overall_f1_score": float(overall_f1),
            "average_precision": avg_precision,
            "average_recall": avg_recall,
            "average_f1_score": avg_f1,
            "total_tp": int(total_tp),
            "total_fp": int(total_fp),
            "total_fn": int(total_fn),
            "precision_recall_threshold": args.precision_recall_threshold,
        }
    else:
        summary = {
            "total_scenes": len(all_results),
            "valid_scenes": 0,
            "error": "No valid results"
        }
    
    output_data = {
        "summary": summary,
        "per_scene": all_results
    }
    
    # 保存结果
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {args.output}")
    print(f"\n总体统计:")
    if "overall_mean_iou" in summary:
        print(f"  总场景数: {summary['total_scenes']}")
        print(f"  有效场景数: {summary['valid_scenes']}")
        print(f"  总GT boxes: {summary['total_gt_boxes']}")
        print(f"  总预测boxes: {summary['total_pred_boxes']}")
        print(f"  总匹配数: {summary['total_matched']}")
        print(f"  平均IoU: {summary['overall_mean_iou']:.4f}")
        print(f"  中位数IoU: {summary['overall_median_iou']:.4f}")
        print(f"  IoU标准差: {summary['overall_std_iou']:.4f}")
        print(f"  最小IoU: {summary['overall_min_iou']:.4f}")
        print(f"  最大IoU: {summary['overall_max_iou']:.4f}")
        print(f"\n  Precision/Recall/F1 (IoU阈值={args.precision_recall_threshold}):")
        print(f"    总体 Precision: {summary['overall_precision']:.4f} (TP={summary['total_tp']}, FP={summary['total_fp']})")
        print(f"    总体 Recall: {summary['overall_recall']:.4f} (TP={summary['total_tp']}, FN={summary['total_fn']})")
        print(f"    总体 F1 Score: {summary['overall_f1_score']:.4f}")
        print(f"    平均 Precision: {summary['average_precision']:.4f}")
        print(f"    平均 Recall: {summary['average_recall']:.4f}")
        print(f"    平均 F1 Score: {summary['average_f1_score']:.4f}")

if __name__ == "__main__":
    main()

