import argparse
import json
import random
import sys
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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
from qa_generation.templates_object_mind_dpt import build_object_dpt_entry, build_object_orient_entry, load_bbox_items


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
):
    """Sample k index pairs with covisibility above covis_threshold and both valid, without repeating images.

    Optional motion filters (if frames provided):
      - min_translation_m: skip pairs whose translation norm is below this value (world c2w).
      - min_rotation_deg: skip pairs whose relative rotation angle is below this value.
    """
    n = covis.shape[0]
    candidates: List[Tuple[int, int, float]] = []
    for i in range(n):
        if not valid_mask[i]:
            continue
        for j in range(i + 1, n):
            if not valid_mask[j]:
                continue
            val = float(covis[i, j])
            if val <= covis_threshold:
                continue

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
                    # Relative rotation taking vectors in cam i to cam j: v_j = Rj.T @ (Ri @ v_i)
                    R_i2j_w = Rj2w_w.T @ Ri2w_w
                    # Express that rotation in camera-i coordinates
                    R_i2j_i = Ri2w_w.T @ R_i2j_w @ Ri2w_w
                    yaw_deg, pitch_deg, roll_deg = R.from_matrix(R_i2j_i).as_euler("YXZ", degrees=True)
                    if max(abs(yaw_deg), abs(pitch_deg), abs(roll_deg)) < min_rotation_deg:
                        continue

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


def main():
    parser = argparse.ArgumentParser(description="为两帧构造3D QA占位数据（基于covisibility>阈值随机抽样）。")
    parser.add_argument("--scene-id", default="all", help="例如 0a5c013435")
    # parser.add_argument("--scene-id", default="0a7cc12c0e", help="例如 0a5c013435")
    # parser.add_argument("--scene-id", default="0a5c013435", help="例如 0a5c013435")
    parser.add_argument("--data-root", default="scannetppv2/data", type=Path)
    parser.add_argument("--wai-root", default="scannetppv2_wai", type=Path)
    parser.add_argument("--covis-threshold", default=0.25, type=float, help="covisibility 阈值，用于选取图像对")
    parser.add_argument("--num", default=100, type=int, help="每类采样多少对（translation、rotation各一轮）")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--output",
        default="output_QA_yyc",
        type=Path,
        help="输出目录（文件名为 <scene_id>.json）未提供则打印到stdout",
    )
    parser.add_argument(
        "--bbox-json-folder",
        type=Path,
        default="output_3d_bounding",
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

        valid_mask = validate_images(frames, images_dir)

        # Cam Translation QA
        # entries_translation: List[Dict] = []
        # pairs = sample_pairs(covis, valid_mask, args.covis_threshold, args.num, rng, frames=frames, min_translation_m=0.25)
        # entries_translation += [build_cam_translation_main_dir_entry(scene_id, intrinsics, frames, p, args.covis_threshold) for p in pairs]
        # pairs = sample_pairs(covis, valid_mask, args.covis_threshold, args.num, rng, frames=frames, min_translation_m=0.25)
        # entries_translation += [build_cam_translation_distance_exact_entry(scene_id, intrinsics, frames, p, args.covis_threshold) for p in pairs]
        # pairs = sample_pairs(covis, valid_mask, args.covis_threshold, args.num, rng, frames=frames, min_translation_m=0.25)
        # entries_translation += [build_cam_translation_distance_threshold_entry(scene_id, intrinsics, frames, p, args.covis_threshold, rng) for p in pairs]

        # # Cam Rotation QA
        # pairs = sample_pairs(covis, valid_mask, args.covis_threshold, args.num, rng, frames=frames, min_rotation_deg=15.0)
        # entries_rotation = [build_cam_rotation_entry(scene_id, intrinsics, frames, p, args.covis_threshold) for p in pairs]

        # Object-distance QA（独立采样一轮）
        entries_object: List[Dict] = []
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

        if bbox_items:
            ignore_labels = {"wall", "floor"}
            pairs = sample_pairs(covis, valid_mask, args.covis_threshold, args.num, rng, frames=frames, min_translation_m=0.3)
            inverse_pairs = [(j, i, conf) for (i, j, conf) in pairs]
            full_pairs = pairs + inverse_pairs
            for pair in full_pairs:
                i, j, _ = pair
                fa, fb = frames[i], frames[j]
                stem_a = Path(fa.file_name).stem
                stem_b = Path(fb.file_name).stem

                single_vis = []
                for inst in bbox_items:
                    label = str(inst.get("label", ""))
                    if label.lower() in ignore_labels:
                        continue
                    images = inst.get("images", [])
                    mask_map = {Path(p).parent.name: p for p in images}
                    if stem_a in mask_map:
                        single_vis.append(inst)
                if not single_vis:
                    continue

                # NOTE(yyc): generate object distance QAs
                for inst in single_vis:
                    entry = build_object_dpt_entry(
                        scene_id,
                        intrinsics,
                        frames,
                        pair,
                        args.covis_threshold,
                        bbox_json_path,
                        bbox_items=bbox_items,
                        target_ins_id=inst.get("ins_id"),
                        skip_labels=ignore_labels,
                    )
                    if entry.get("objects"):
                        entries_object.append(entry)

                # NOTE(yyc): generate object orientation QAs
                for inst in single_vis:
                    entry = build_object_orient_entry(
                        scene_id,
                        intrinsics,
                        frames,
                        pair,
                        args.covis_threshold,
                        bbox_json_path,
                        bbox_items=bbox_items,
                        target_ins_id=inst.get("ins_id"),
                        skip_labels=ignore_labels,
                    )
                    if entry.get("objects"):
                        entries_object.append(entry)
            print(f"object QA 构建完成: {len(entries_object)} 条")

        # entries = entries_translation + entries_rotation + entries_object
        entries = entries_object
        all_entries.extend(entries)

        if args.output:
            out_file = args.output / f"{scene_id}.json"
            with out_file.open("w") as f:
                json.dump(entries, f, ensure_ascii=False, indent=2)
            print(f"写入 {len(entries)} 条记录到文件: {out_file}")
        else:
            print(json.dumps(entries, ensure_ascii=False, indent=2))



if __name__ == "__main__":
    main()

