import argparse
import itertools
import json
import random
import sys
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Set

import numpy as np
from scipy.spatial.transform import Rotation as R

# Ensure repository root on sys.path so intra-package imports work when run as a script.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from qa_generation.templates_cam_translation import (
    build_cam_translation_main_dir_entry,
    build_cam_translation_distance_exact_entry,
    build_cam_translation_distance_threshold_entry,
)
from qa_generation.templates_cam_rotation import build_cam_rotation_entry
from qa_generation.templates_object_dpt import build_object_dpt_entry, load_bbox_items
from qa_generation.templates_object_dist import build_object_dist_entry
from qa_generation.templates_object_relpos import build_object_relpos_entry


@dataclass
class FrameItem:
    idx: int
    file_name: str
    transform_matrix: List[List[float]]
    is_bad: bool


def load_transforms(scene_id: str, data_root: Path) -> Tuple[Dict[str, float], List[FrameItem], Path]:
    """Load intrinsics/extrinsics and frame list for a scene."""
    meta_path = data_root / scene_id / "dslr" / "nerfstudio" / "transforms_undistorted.json"
    with meta_path.open() as f:
        meta = json.load(f)

    frames_raw = meta.get("frames", [])
    test_frames_raw = meta.get("test_frames", [])
    all_frames_raw = frames_raw + test_frames_raw
    # Align order with covisibility index (sorted by file name)
    all_frames_raw = sorted(all_frames_raw, key=lambda f: f["file_path"])

    frames: List[FrameItem] = []
    for idx, fr in enumerate(all_frames_raw):
        frames.append(
            FrameItem(
                idx=idx,
                file_name=fr["file_path"],
                transform_matrix=fr["transform_matrix"],
                is_bad=bool(fr.get("is_bad", False)),
            )
        )

    images_dir = data_root / scene_id / "dslr" / "resized_undistorted_images"
    intrinsics = {
        "fl_x": meta["fl_x"],
        "fl_y": meta["fl_y"],
        "cx": meta["cx"],
        "cy": meta["cy"],
        "w": meta["w"],
        "h": meta["h"],
        "camera_model": meta.get("camera_model", "PINHOLE"),
    }
    return intrinsics, frames, images_dir


def load_covisibility(scene_id: str, wai_root: Path) -> np.ndarray | None:
    """Load the covisibility matrix; return None if missing or ambiguous."""
    cov_dir = wai_root / scene_id / "covisibility" / "v0"
    npy_files = sorted(cov_dir.glob("*.npy"))
    if len(npy_files) == 0:
        print(f"警告: 场景 {scene_id} 未找到 covisibility npy: {cov_dir}")
        return None
    if len(npy_files) != 1:
        print(f"警告: 场景 {scene_id} 期望该目录仅有一个npy文件: {cov_dir}")
        return None
    return np.load(npy_files[0])


def validate_images(frames: List[FrameItem], images_dir: Path) -> np.ndarray:
    """Return a boolean mask for frames that have existing JPG files and are not marked bad."""
    existing = {p.name for p in images_dir.glob("*.JPG")}
    mask = []
    for fr in frames:
        ok = (fr.file_name in existing) #and (not fr.is_bad)
        mask.append(ok)
    return np.array(mask, dtype=bool)


def sample_pairs(
    covis: np.ndarray,
    valid_mask: np.ndarray,
    covis_threshold: float,
    k: int,
    rng: random.Random,
    frames: List["FrameItem"] | None = None,
    min_translation_m: float | None = None,
    min_rotation_deg: float | None = None,
    bbox_items: List[Dict] | None = None,
    at_least_3_objs: bool = False,
):
    """Sample k index pairs with covisibility above covis_threshold and both valid, without repeating images.

    Optional motion filters (if frames provided):
      - min_translation_m: skip pairs whose translation norm is below this value (world c2w).
      - min_rotation_deg: skip pairs whose relative rotation angle is below this value.
    
    Optional instance coverage filter (if bbox_items and frames provided):
      - at_least_3_objs: if True, require the union of instances in the image pair to have at least 3 objects.
    """
    n = covis.shape[0]
    
    # 预先构建 stem_to_inst_ids 映射（只在 at_least_3_objs=True 的时候使用）
    stem_to_inst_ids: Dict[str, set] = {}
    if at_least_3_objs and bbox_items is not None and frames is not None:
        for inst in bbox_items:
            ins_id = str(inst.get("ins_id", ""))
            images = inst.get("images", [])
            for p_str in images:
                stem = Path(p_str).parent.name
                if stem not in stem_to_inst_ids:
                    stem_to_inst_ids[stem] = set()
                stem_to_inst_ids[stem].add(ins_id)
    
    candidates: List[Tuple[int, int, float]] = []
    for i in range(n):
        if not valid_mask[i]:
            continue
        for j in range(i + 1, n):
            if not valid_mask[j]:
                continue
            
            # 共视约束
            val = float(covis[i, j])
            if val <= covis_threshold:
                continue

            # 最小位移/旋转约束
            if frames is not None and (min_translation_m is not None or min_rotation_deg is not None):
                fi, fj = frames[i], frames[j]
                Ti = np.array(fi.transform_matrix, dtype=float)
                Tj = np.array(fj.transform_matrix, dtype=float)
                delta_t = Tj[:3, 3] - Ti[:3, 3]
                trans_norm = float(np.linalg.norm(delta_t))
                if min_translation_m is not None and trans_norm < min_translation_m:
                    continue
                if min_rotation_deg is not None:
                    Ri2w_w = Ti[:3, :3]
                    Rj2w_w = Tj[:3, :3]
                    R_rel = Ri2w_w.T @ Rj2w_w
                    yaw_deg, pitch_deg, roll_deg = R.from_matrix(R_rel).as_euler("YXZ", degrees=True)
                    if max(abs(yaw_deg), abs(pitch_deg), abs(roll_deg)) < min_rotation_deg:
                        continue

            # 约束：要求图对的实例并集至少包含 3 个对象
            if at_least_3_objs and bbox_items is not None and frames is not None:
                fi, fj = frames[i], frames[j]
                stem_i = Path(fi.file_name).stem
                stem_j = Path(fj.file_name).stem
                inst_ids_i = stem_to_inst_ids.get(stem_i, set())
                inst_ids_j = stem_to_inst_ids.get(stem_j, set())
                union_inst_ids = inst_ids_i | inst_ids_j
                if len(union_inst_ids) < 3:
                    continue

            # 随机交换图像对顺序
            if rng.random() < 0.5:
                i, j = j, i
            candidates.append((i, j, val))
    if not candidates:
        raise ValueError("没有满足阈值的图像对。")
    rng.shuffle(candidates)

    chosen: List[Tuple[int, int, float]] = []
    used = set()
    for pair in candidates:
        i, j, _ = pair
        if i in used or j in used:
            continue
        if i == j:
            continue
        chosen.append(pair)
        used.add(i)
        used.add(j)
        if len(chosen) >= k:
            break

    if len(chosen) < k:
        print(f"警告: 可用的非重复图像对不足，阈值 {covis_threshold}，请求 {k} 对，仅找到 {len(chosen)} 对。")
    return chosen


def ensure_output_dir(path: Path):
    """Ensure output is a directory (create if needed)."""
    if path.exists() and not path.is_dir():
        raise ValueError(f"--output 需要是目录路径，当前存在且非目录: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _build_stem_to_insts_map(
    bbox_items: List[Dict],
    frames: List["FrameItem"],
    ignore_labels: Set[str],
) -> Dict[str, List[Dict]]:
    """构建 stem -> instances 映射，过滤 ignore_labels。"""
    stem_to_insts = {}
    for inst in bbox_items:
        label = str(inst.get("label", ""))
        if label.lower() in ignore_labels:
            continue
        images = inst.get("images", [])
        for p_str in images:
            stem = Path(p_str).parent.name
            if stem not in stem_to_insts:
                stem_to_insts[stem] = []
            stem_to_insts[stem].append(inst)
    return stem_to_insts


def sample1obj_common(
    frames: List["FrameItem"],
    pairs: List[Tuple[int, int, float]],
    bbox_items: List[Dict],
    ignore_labels: Set[str],
) -> List[Tuple[Tuple[int, int, float], Dict]]:
    """采样在图像对的两个图像中同时出现的对象。返回 [(image_pair, instance), ...]"""
    results = []
    for pair in pairs:
        i, j, _ = pair
        fa, fb = frames[i], frames[j]
        stem_a = Path(fa.file_name).stem
        stem_b = Path(fb.file_name).stem

        for inst in bbox_items:
            label = str(inst.get("label", ""))
            if label.lower() in ignore_labels:
                continue
            images = inst.get("images", [])
            mask_map = {Path(p).parent.name: p for p in images}
            if stem_a in mask_map and stem_b in mask_map:
                results.append((pair, inst))
    return results


def sample2obj_across(
    frames: List["FrameItem"],
    pairs: List[Tuple[int, int, float]],
    bbox_items: List[Dict],
    ignore_labels: Set[str],
    rng: random.Random,
    max_inst_pairs_per_image_pair: int = 3,
) -> List[Tuple[Tuple[int, int, float], Tuple[Dict, Dict]]]:
    """从图像对的两个图像中分别采样所有不同的对象对。返回 [(image_pair, (inst1, inst2)), ...]
    inst1来自图像1, inst2来自图像2
    
    Args:
        max_inst_pairs_per_image_pair: 每个 image pair 最多返回的 instance pair 数量，超过则随机选择
    """
    results = []
    stem_to_insts = _build_stem_to_insts_map(bbox_items, frames, ignore_labels)
    
    for pair in pairs:
        i, j, _ = pair
        fa, fb = frames[i], frames[j]
        stem_a = Path(fa.file_name).stem
        stem_b = Path(fb.file_name).stem
        
        insts_a = stem_to_insts.get(stem_a, []).copy()
        insts_b = stem_to_insts.get(stem_b, []).copy()
        
        if not insts_a or not insts_b:
            continue
        
        # 打乱顺序以实现随机采样
        rng.shuffle(insts_a)
        rng.shuffle(insts_b)
        
        # 遍历并计数，达到限制就停止
        count = 0
        for inst1 in insts_a:
            if count >= max_inst_pairs_per_image_pair:
                break
            for inst2 in insts_b:
                if count >= max_inst_pairs_per_image_pair:
                    break
                if inst1.get("ins_id") == inst2.get("ins_id"):
                    continue
                if inst1.get("label") == inst2.get("label"):
                    continue
                results.append((pair, (inst1, inst2)))
                count += 1
    
    return results


def sample3obj(
    frames: List["FrameItem"],
    pairs: List[Tuple[int, int, float]],
    bbox_items: List[Dict],
    ignore_labels: Set[str],
    rng: random.Random,
) -> List[Tuple[Tuple[int, int, float], Tuple[Dict, Dict, Dict]]]:
    """从图像对的两个图像的实例并集中采样3个不同的对象。返回 [(image_pair, (instA, instB, instC)), ...]"""
    results = []
    stem_to_insts = _build_stem_to_insts_map(bbox_items, frames, ignore_labels)
    
    for pair in pairs:
        i, j, _ = pair
        fa, fb = frames[i], frames[j]
        stem_i = Path(fa.file_name).stem
        stem_j = Path(fb.file_name).stem
        
        insts_i = stem_to_insts.get(stem_i, [])
        insts_j = stem_to_insts.get(stem_j, [])
        all_insts = []
        seen_ids = set()
        for inst in insts_i + insts_j:
            ins_id = str(inst.get("ins_id", ""))
            if ins_id not in seen_ids:
                seen_ids.add(ins_id)
                all_insts.append(inst)
        
        if len(all_insts) < 3:
            continue
        
        # 构建 ins_id 到图像集合的映射，用于快速检查
        ins_id_to_stems = {}
        for inst in insts_i:
            ins_id = str(inst.get("ins_id", ""))
            if ins_id not in ins_id_to_stems:
                ins_id_to_stems[ins_id] = set()
            ins_id_to_stems[ins_id].add(stem_i)
        for inst in insts_j:
            ins_id = str(inst.get("ins_id", ""))
            if ins_id not in ins_id_to_stems:
                ins_id_to_stems[ins_id] = set()
            ins_id_to_stems[ins_id].add(stem_j)
        
        # 生成所有可能的3元组组合
        for instA, instB, instC in itertools.combinations(all_insts, 3):
            # 确保没有重复的 instance（通过 ins_id 判断，已在 all_insts 构建时去重）
            # 确保没有重复的 label
            labelA = instA.get("label")
            labelB = instB.get("label")
            labelC = instC.get("label")
            if labelA == labelB or labelA == labelC or labelB == labelC:
                continue
            
            # 确保3个物体不能同时被一张图片看到（避免退化成单图问题）
            ins_idA = str(instA.get("ins_id", ""))
            ins_idB = str(instB.get("ins_id", ""))
            ins_idC = str(instC.get("ins_id", ""))
            
            stems_A = ins_id_to_stems.get(ins_idA, set())
            stems_B = ins_id_to_stems.get(ins_idB, set())
            stems_C = ins_id_to_stems.get(ins_idC, set())
            
            # 检查是否有一张图像同时包含所有3个物体
            if stem_i in stems_A and stem_i in stems_B and stem_i in stems_C:
                continue
            if stem_j in stems_A and stem_j in stems_B and stem_j in stems_C:
                continue
            
            results.append((pair, (instA, instB, instC)))
    
    return results


def main():
    parser = argparse.ArgumentParser(description="为两帧构造3D QA占位数据（基于covisibility>阈值随机抽样）。")
    parser.add_argument("--scene-id", default="all", help="例如 0a5c013435")
    # parser.add_argument("--scene-id", default="00777c41d4", help="例如 0a5c013435")
    # parser.add_argument("--scene-id", default="0a5c013435", help="例如 0a5c013435")
    parser.add_argument("--data-root", default="scannetppv2/data", type=Path)
    parser.add_argument("--wai-root", default="scannetppv2_wai", type=Path)
    parser.add_argument("--covis-threshold", default=0.25, type=float, help="covisibility 阈值，用于选取图像对")
    parser.add_argument("--num", default=100, type=int, help="每类采样多少对（translation、rotation各一轮）")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--output",
        default="output_QA",
        type=Path,
        help="输出目录（文件名为 <scene_id>.json）未提供则打印到stdout",
    )
    parser.add_argument(
        "--bbox-json-folder",
        type=Path,
        default="output_yifei",
        help="3D bbox 结果所在目录，文件名为 <scene_id>.json；不存在则跳过该类问题",
    )

    args = parser.parse_args()

    def iter_scene_ids():
        if args.scene_id.lower() != "all":
            yield args.scene_id
            return
        root = args.data_root
        for p in sorted(root.iterdir()):
            if not p.is_dir():
                continue
            tf = p / "dslr" / "nerfstudio" / "transforms_undistorted.json"
            if tf.exists():
                yield p.name

    ensure_output_dir(args.output)
    all_entries: List[Dict] = []
    base_rng = random.Random(args.seed)

    for idx, scene_id in enumerate(iter_scene_ids()):
        print(f"处理场景 {scene_id} ...")
        rng = random.Random(base_rng.random() * 1e9)

        intrinsics, frames, images_dir = load_transforms(scene_id, args.data_root)
        covis = load_covisibility(scene_id, args.wai_root)
        if covis is None:
            print(f"跳过场景 {scene_id}: 缺少或存在多个 covisibility npy")
            continue

        if covis.shape[0] != covis.shape[1] or covis.shape[0] != len(frames):
            print(f"跳过场景 {scene_id}: covisibility矩阵与frame数量不匹配: covis {covis.shape}, frames {len(frames)}")
            continue

        # 图像的文件必须存在
        valid_mask = validate_images(frames, images_dir)

        # Cam Translation QA
        entries_translation: List[Dict] = []
        pairs = sample_pairs(covis, valid_mask, args.covis_threshold, args.num, rng, frames=frames, min_rotation_deg=0.35)
        entries_translation += [build_cam_translation_main_dir_entry(scene_id, intrinsics, frames, p, args.covis_threshold) for p in pairs]
        pairs = sample_pairs(covis, valid_mask, args.covis_threshold, args.num, rng, frames=frames, min_translation_m=0.35)
        entries_translation += [build_cam_translation_distance_exact_entry(scene_id, intrinsics, frames, p, args.covis_threshold) for p in pairs]
        pairs = sample_pairs(covis, valid_mask, args.covis_threshold, args.num, rng, frames=frames, min_translation_m=0.35)
        entries_translation += [build_cam_translation_distance_threshold_entry(scene_id, intrinsics, frames, p, args.covis_threshold, rng) for p in pairs]

        # Cam Rotation QA
        pairs = sample_pairs(covis, valid_mask, args.covis_threshold, args.num, rng, frames=frames, min_rotation_deg=15.0)
        entries_rotation = [entry for p in pairs if (entry := build_cam_rotation_entry(scene_id, intrinsics, frames, p, args.covis_threshold)) is not None]

        # Load bbox items for object QA
        bbox_items = []
        bbox_json_path: Path | None = None
        candidate_folder = args.bbox_json_folder
        candidate_path = candidate_folder / f"{scene_id}.json"
        fallback_path = args.data_root / scene_id / "3d_bboxes.json"
        if candidate_path.exists():
            bbox_json_path = candidate_path
        elif fallback_path.exists():
            bbox_json_path = fallback_path
        if bbox_json_path:
            try:
                bbox_items = load_bbox_items(bbox_json_path)
                print(f"加载 bbox json 成功: {bbox_json_path}, 实例数 {len(bbox_items)}")
            except Exception as exc:
                print(f"加载 bbox json 失败，跳过 object QA: {exc}")
                bbox_items = []
        else:
            print(f"场景 {scene_id}: 未提供 bbox json，跳过 object QA。")

        # Object QA
        if bbox_items:
            ignore_labels = {"wall", "floor", "ceiling"}
            
            # One Object common seen in both images, depth QA
            pairs = sample_pairs(covis, valid_mask, args.covis_threshold, args.num, rng, frames=frames, min_translation_m=0.2, min_rotation_deg=15.0)
            sampled = sample1obj_common(frames, pairs, bbox_items, ignore_labels)
            entries_object = []
            for pair, inst in sampled:
                entry = build_object_dpt_entry(
                    scene_id, intrinsics, frames, pair, args.covis_threshold,
                    bbox_json_path, bbox_items=bbox_items,
                    target_ins_id=inst.get("ins_id"), skip_labels=ignore_labels
                )
                if entry.get("objects"):
                    entries_object.append(entry)
            print(f"object depth QA 构建完成: {len(entries_object)} 条")
            
            # Two Objects across different images, distance QA
            pairs = sample_pairs(covis, valid_mask, args.covis_threshold, args.num, rng, frames=frames, min_translation_m=0.2)
            sampled = sample2obj_across(frames, pairs, bbox_items, ignore_labels, rng, max_inst_pairs_per_image_pair=3)
            entries_inter_object = []
            for pair, (inst1, inst2) in sampled:
                entry = build_object_dist_entry(
                    scene_id, intrinsics, frames, pair, args.covis_threshold,
                    inst1, inst2
                )
                entries_inter_object.append(entry)
            print(f"cross image 2 object distance QA 构建完成: {len(entries_inter_object)} 条")
            
            # Relative position QA
            pairs = sample_pairs(
                covis, valid_mask, args.covis_threshold, args.num, rng,
                frames=frames, bbox_items=bbox_items, at_least_3_objs=True
            )
            sampled = sample3obj(frames, pairs, bbox_items, ignore_labels, rng)
            entries_relpos = []
            for pair, (instA, instB, instC) in sampled:
                entry = build_object_relpos_entry(
                    scene_id, intrinsics, frames, pair, args.covis_threshold,
                    instA, instB, instC, rng=rng
                )
                if entry is not None:
                    entries_relpos.append(entry)
            print(f"relative position QA 构建完成: {len(entries_relpos)} 条")

        entries = entries_translation + entries_rotation + entries_object + entries_inter_object + entries_relpos
        all_entries.extend(entries)

        if args.output:
            out_file = args.output / f"{scene_id}.json"
            with out_file.open("w") as f:
                json.dump(entries, f, ensure_ascii=False, indent=2)
            print(f"写入 {len(entries)} 条记录到文件: {out_file}")
        else:
            print(json.dumps(entries, ensure_ascii=False, indent=2))
        
        # TODO:
        # 1. prompt type: point, mask盖在图上, 黑白mask, box, only label (language)
        # 2. asking type: 同一个问题的不同问法（语言，不同描述方式）



if __name__ == "__main__":
    main()

