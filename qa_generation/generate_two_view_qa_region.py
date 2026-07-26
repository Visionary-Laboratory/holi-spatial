import argparse
import itertools
import json
import random
import sys
import math
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R
from PIL import Image
import trimesh

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
from qa_generation.templates_object_dpt import build_object_dpt_entry, build_object_dir_entry, load_bbox_items
from qa_generation.templates_object_dist import (
    build_object_dist_entry,
    build_object_cross_dir_entry,
    build_object_height_entry,
    build_region_cross_dir_entry,
    build_region_object_cross_dir_entry,
)
from qa_generation.templates_object_relpos import (
    build_object_relpos_entry,
    build_region_relpos_entry,
    build_region_proxy_dir_entry,
    build_object_proxy_dir_entry,
)


@dataclass
class FrameItem:
    idx: int
    file_name: str
    transform_matrix: List[List[float]]
    is_bad: bool


PER_GPU_SIZE = 126


def load_transforms_scannetppv2(scene_id: str, data_root: Path) -> Tuple[Dict[str, float], List[FrameItem], Path]:
    """Load intrinsics/extrinsics and frame list for a scannetppv2 scene."""
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
        mat = np.array(fr["transform_matrix"], dtype=np.float32)
        # Nerfstudio(OpenGL) -> COLMAP convention.
        mat[:3, 1:3] *= -1
        frames.append(
            FrameItem(
                idx=idx,
                file_name=fr["file_path"],
                transform_matrix=mat.tolist(),
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


def load_transforms_dl3dv(scene_id: str, data_root: Path) -> Tuple[Dict[str, float], List[FrameItem], Path]:
    """Load intrinsics/extrinsics and frame list for a DL3DV scene."""
    scene_root = data_root / scene_id
    cam_dir = scene_root / "dense" / "cam"
    images_dir = scene_root / "dense" / "rgb"
    cam_files = sorted(cam_dir.glob("frame_*.npz"))
    if not cam_files:
        raise FileNotFoundError(f"未找到相机参数文件: {cam_dir}")

    frames: List[FrameItem] = []
    intrinsic = None
    for idx, cam_path in enumerate(cam_files):
        with np.load(cam_path) as data:
            intrinsic = data["intrinsic"]
            pose = data["pose"]
        file_name = f"{cam_path.stem}.png"
        frames.append(
            FrameItem(
                idx=idx,
                file_name=file_name,
                transform_matrix=pose.tolist(),
                is_bad=False,
            )
        )

    width = height = None
    first_image = images_dir / frames[0].file_name
    if first_image.exists():
        with Image.open(first_image) as img:
            width, height = img.size

    if intrinsic is None or width is None or height is None:
        raise RuntimeError(f"无法解析相机参数或图像尺寸: {scene_id}")

    intrinsics = {
        "fl_x": float(intrinsic[0, 0]),
        "fl_y": float(intrinsic[1, 1]),
        "cx": float(intrinsic[0, 2]),
        "cy": float(intrinsic[1, 2]),
        "w": int(width),
        "h": int(height),
        "camera_model": "PINHOLE",
    }
    return intrinsics, frames, images_dir


def load_covisibility(scene_id: str, wai_root: Path) -> np.ndarray | None:
    """Load the covisibility matrix; return None if missing or ambiguous."""
    cov_dir_cand = [wai_root / scene_id / "covisibility" / "v0",
               (wai_root / Path("1K_"+scene_id)),
               (wai_root / Path("2K_"+scene_id))]
    cov_dir = None
    for dir in cov_dir_cand:
        if dir.exists():
            cov_dir = dir
            break
    if cov_dir is None:
        import warnings
        warnings.warn(f"警告: 场景 {scene_id} 未找到 covisibility 目录")
        return None
        # raise FileNotFoundError(f"警告: 场景 {scene_id} 未找到 covisibility 目录")
    
    npy_files = sorted(cov_dir.glob("*.npy"))
    if len(npy_files) == 0:
        print(f"警告: 场景 {scene_id} 未找到 covisibility npy: {cov_dir}")
        return None
    if len(npy_files) != 1:
        print(f"警告: 场景 {scene_id} 期望该目录仅有一个npy文件: {cov_dir}")
        return None
    return np.load(npy_files[0])


def _stable_color_u8(key: str) -> np.ndarray:
    h = hashlib.md5(key.encode("utf-8")).digest()
    rgb = np.frombuffer(h[:3], dtype=np.uint8).astype(np.uint16)
    rgb = (rgb // 2 + 64).clip(0, 255).astype(np.uint8)
    return rgb


def _sanitize_entity_name(name: str) -> str:
    name = name.strip().replace("/", "_").replace("\\", "_").replace(" ", "_")
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in name)


def _bbox_edges_from_vertices(vertices: np.ndarray) -> np.ndarray:
    if isinstance(vertices, list) and vertices and isinstance(vertices[0], dict):
        v = np.array([[p["x"], p["y"], p["z"]] for p in vertices], dtype=np.float32)
    else:
        v = np.asarray(vertices, dtype=np.float32)
    if v.shape != (8, 3):
        raise ValueError(f"bounding_box 期望形状 (8,3)，当前 {v.shape}")

    diff = v[:, None, :] - v[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    edges: set[tuple[int, int]] = set()
    for i in range(8):
        nn = np.argsort(dist2[i])[1:4]
        for j in nn:
            a, b = (i, int(j)) if i < int(j) else (int(j), i)
            edges.add((a, b))
    edges_sorted = sorted(edges)
    segments = np.stack([np.stack([v[i], v[j]], axis=0) for i, j in edges_sorted], axis=0)
    return segments


def _load_point_cloud(ply_path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not ply_path.exists():
        return None, None

    cloud = trimesh.load(ply_path, process=False)
    if isinstance(cloud, trimesh.Scene):
        geoms = list(cloud.geometry.values())
        if not geoms:
            return None, None
        cloud = geoms[0]

    points = None
    colors = None
    if isinstance(cloud, trimesh.Trimesh):
        points = np.asarray(cloud.vertices, dtype=np.float32)
        if cloud.visual.vertex_colors is not None and len(cloud.visual.vertex_colors):
            colors = np.asarray(cloud.visual.vertex_colors)[:, :3].astype(np.uint8)
    elif isinstance(cloud, trimesh.points.PointCloud):
        points = np.asarray(cloud.vertices, dtype=np.float32)
        if cloud.colors is not None and len(cloud.colors):
            colors = np.asarray(cloud.colors)[:, :3].astype(np.uint8)
    return points, colors


def _send_blueprint(rr) -> None:
    bp = rr.blueprint.Blueprint(
        rr.blueprint.Tabs(
            rr.blueprint.Spatial3DView(name="3D", origin="/"),
        ),
        auto_views=False,
    )
    rr.send_blueprint(bp)


def save_scene_rrd(
    scene_id: str,
    intrinsics: Dict[str, float],
    frames: List[FrameItem],
    images_dir: Path,
    out_dir: Path,
    bbox_items: List[Dict],
    ply_path: Path,
    stride: int = 10,
    use_rub_view: bool = False,
) -> None:
    try:
        import rerun as rr
    except ImportError:
        print("rerun 未安装，跳过 rrd 保存")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{scene_id}.rrd"

    rr.init(f"two_view/{scene_id}", spawn=False)
    if use_rub_view:
        rr.log("/", rr.ViewCoordinates.RUB)
    _send_blueprint(rr)

    K = np.array(
        [
            [intrinsics["fl_x"], 0.0, intrinsics["cx"]],
            [0.0, intrinsics["fl_y"], intrinsics["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    resolution = (int(intrinsics["w"]), int(intrinsics["h"]))

    points, colors = _load_point_cloud(ply_path)
    if points is not None and points.size:
        log_kwargs = {}
        if colors is not None and colors.shape[0] == points.shape[0]:
            log_kwargs["colors"] = colors
        rr.log("points", rr.Points3D(points, **log_kwargs))

    for idx, inst in enumerate(bbox_items):
        label = str(inst.get("label", "unknown"))
        ins_id = str(inst.get("ins_id", idx))
        safe_label = _sanitize_entity_name(label)
        color = _stable_color_u8(f"{label}:{ins_id}")
        if "obb_transform" in inst and "obb_extents" in inst:
            transform = np.array(inst["obb_transform"], dtype=np.float32)
            extents = np.array(inst["obb_extents"], dtype=np.float32)
            center = transform[:3, 3]
            quat_xyzw = R.from_matrix(transform[:3, :3]).as_quat().astype(np.float32)
            rr.log(
                f"bbox/{safe_label}/{ins_id}",
                rr.Boxes3D(
                    centers=np.array([center], dtype=np.float32),
                    half_sizes=np.array([extents * 0.5], dtype=np.float32),
                    quaternions=[rr.Quaternion(xyzw=quat_xyzw)],
                    colors=np.array([color], dtype=np.uint8),
                ),
            )
        elif "bounding_box" in inst:
            try:
                segments = _bbox_edges_from_vertices(np.asarray(inst["bounding_box"]))
            except Exception:
                continue
            rr.log(f"bbox/{safe_label}/{ins_id}", rr.LineStrips3D(segments, colors=color))

    for fr in frames[::stride]:
        image_path = images_dir / fr.file_name
        if not image_path.exists():
            continue
        c2w = np.array(fr.transform_matrix, dtype=np.float32)

        ent = f"cameras/{Path(fr.file_name).stem}"
        rr.log(ent, rr.Transform3D(translation=c2w[:3, 3], mat3x3=c2w[:3, :3]))
        rr.log(
            ent,
            rr.Pinhole(
                resolution=resolution,
                focal_length=(float(K[0, 0]), float(K[1, 1])),
                principal_point=(float(K[0, 2]), float(K[1, 2])),
            ),
        )
        with Image.open(image_path) as img:
            rr.log(f"{ent}/image", rr.Image(img.convert("RGB")))

    rr.save(str(out_path))
    print(f"rrd 已保存: {out_path}")


def validate_images(frames: List[FrameItem], images_dir: Path) -> np.ndarray:
    """Return a boolean mask for frames that have existing image files and are not marked bad."""
    existing = {
        p.name
        for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".JPG", ".PNG"}
    }
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
    covis_range: Tuple[float, float] | None = None,
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
    
    # 预先收集所有有效的索引，并随机打乱顺序以实现随机采样
    valid_indices = [idx for idx in range(n) if valid_mask[idx]]
    rng.shuffle(valid_indices)
    
    chosen: List[Tuple[int, int, float]] = []
    used = set()
    
    # 为每个 i 预先收集所有可能的 j（j > i 且 valid），并随机打乱
    for i in valid_indices:
        if i in used:
            continue  # i 已经被使用，跳过
        
        # 收集所有可能的 j（j > i 且 valid），并随机打乱
        candidate_js = [j for j in valid_indices if j > i and j not in used]
        rng.shuffle(candidate_js)
        
        for j in candidate_js:
            if len(chosen) >= k:
                break  # 已经收集够 k 个，提前停止
            
            # 共视约束
            val = float(covis[i, j])
            if covis_range is not None:
                if val < covis_range[0] or val > covis_range[1]:
                    continue
            elif val <= covis_threshold:
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
            final_i, final_j = (j, i) if rng.random() < 0.5 else (i, j)
            chosen.append((final_i, final_j, val))
            used.add(final_i)
            used.add(final_j)
        
        if len(chosen) >= k:
            break  # 已经收集够 k 个，提前停止外层循环

    if not chosen:
        print(f"警告: 没有满足阈值的图像对，阈值 {covis_threshold}，本轮跳过。")
        return []
    
    if len(chosen) < k:
        print(f"警告: 可用的非重复图像对不足，阈值 {covis_threshold}，请求 {k} 对，仅找到 {len(chosen)} 对。")
    return chosen


def sample_pairs_region(
    covis: np.ndarray,
    valid_mask: np.ndarray,
    covis_threshold: float,
    k: int,
    rng: random.Random,
    frames: List["FrameItem"] | None = None,
    min_translation_m: float | None = None,
    min_rotation_deg: float | None = None,
    bbox_items: List[Dict] | None = None,
    region_bbox_items: List[Dict] | None = None,
    frame_to_region_bbox_ids: Dict[str, Set[str]] | None = None,
    at_least_3_objs_1_region: bool = False,
    covis_range: Tuple[float, float] | None = None,
):
    """在 sample_pairs 基础上增加 region 约束的图像对采样。

    当 at_least_3_objs_1_region=True 时，要求：
      - 图像对中的 object 实例并集数量 >= 3；
      - 图像对中的 region 实例并集数量 >= 1。

    frame_to_region_bbox_ids 用于传入“frame 与 region bbox 的关系”：
      - key: frame stem（Path(frame.file_name).stem）
      - value: 该帧可见的 region ins_id 集合
    若不传该映射且传入 region_bbox_items，会自动从 region_bbox_items 构建。
    """
    n = covis.shape[0]

    # 预构建 object: frame stem -> object ins_id 集合
    stem_to_obj_ids: Dict[str, Set[str]] = {}
    if at_least_3_objs_1_region and bbox_items is not None and frames is not None:
        for inst in bbox_items:
            ins_id = str(inst.get("ins_id", ""))
            for p_str in inst.get("images", []):
                stem = Path(p_str).parent.name
                if stem not in stem_to_obj_ids:
                    stem_to_obj_ids[stem] = set()
                stem_to_obj_ids[stem].add(ins_id)

    # 预构建/使用传入的 region: frame stem -> region ins_id 集合
    stem_to_region_ids: Dict[str, Set[str]] = {}
    if at_least_3_objs_1_region and frames is not None:
        if frame_to_region_bbox_ids is not None:
            stem_to_region_ids = frame_to_region_bbox_ids
        elif region_bbox_items is not None:
            for region_inst in region_bbox_items:
                region_id = str(region_inst.get("ins_id", ""))
                for p_str in region_inst.get("images", []):
                    stem = Path(p_str).parent.name
                    if stem not in stem_to_region_ids:
                        stem_to_region_ids[stem] = set()
                    stem_to_region_ids[stem].add(region_id)

    valid_indices = [idx for idx in range(n) if valid_mask[idx]]
    rng.shuffle(valid_indices)

    chosen: List[Tuple[int, int, float]] = []
    used = set()

    for i in valid_indices:
        if i in used:
            continue

        candidate_js = [j for j in valid_indices if j > i and j not in used]
        rng.shuffle(candidate_js)

        for j in candidate_js:
            if len(chosen) >= k:
                break

            # 共视约束
            val = float(covis[i, j])
            if covis_range is not None:
                if val < covis_range[0] or val > covis_range[1]:
                    continue
            elif val <= covis_threshold:
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

            # 约束：object 并集 >= 3 且 region 并集 >= 1
            if at_least_3_objs_1_region and frames is not None:
                fi, fj = frames[i], frames[j]
                stem_i = Path(fi.file_name).stem
                stem_j = Path(fj.file_name).stem

                union_obj_ids = stem_to_obj_ids.get(stem_i, set()) | stem_to_obj_ids.get(stem_j, set())
                if len(union_obj_ids) < 3:
                    continue

                union_region_ids = stem_to_region_ids.get(stem_i, set()) | stem_to_region_ids.get(stem_j, set())
                if len(union_region_ids) < 1:
                    continue

            final_i, final_j = (j, i) if rng.random() < 0.5 else (i, j)
            chosen.append((final_i, final_j, val))
            used.add(final_i)
            used.add(final_j)

        if len(chosen) >= k:
            break

    if not chosen:
        print("警告: 没有满足阈值与 region 约束的图像对，跳过该场景。")
        return []

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


def _bbox_to_array(bbox: List) -> Optional[np.ndarray]:
    if not bbox:
        return None
    first = bbox[0]
    if isinstance(first, dict) and {"x", "y", "z"} <= set(first.keys()):
        arr = np.array([[p["x"], p["y"], p["z"]] for p in bbox], dtype=np.float32)
    else:
        arr = np.asarray(bbox, dtype=np.float32)
    if arr.shape != (8, 3):
        return None
    return arr


def _instance_center_world(inst: Dict) -> Optional[np.ndarray]:
    arr = _bbox_to_array(inst.get("bounding_box"))
    if arr is None:
        return None
    return arr.mean(axis=0)


def _point_inside_region_bbox(point_world: np.ndarray, region_inst: Dict, eps: float = 1e-6) -> bool:
    """判断世界坐标点是否在 region bbox 内（优先 OBB，失败则退化到 AABB）。"""
    obb_t = region_inst.get("obb_transform")
    obb_extents = region_inst.get("obb_extents")
    if obb_t is not None and obb_extents is not None:
        try:
            T = np.asarray(obb_t, dtype=np.float32)
            ext = np.asarray(obb_extents, dtype=np.float32).reshape(-1)
            if T.shape == (4, 4) and ext.shape[0] >= 3:
                center = T[:3, 3]
                rot = T[:3, :3]
                local = rot.T @ (point_world - center)
                half = ext[:3] * 0.5 + eps
                return bool(np.all(np.abs(local) <= half))
        except Exception:
            pass

    # 回退：用 8 点包围盒的轴对齐范围近似判断
    arr = _bbox_to_array(region_inst.get("bounding_box"))
    if arr is None:
        return False
    mins = arr.min(axis=0) - eps
    maxs = arr.max(axis=0) + eps
    return bool(np.all(point_world >= mins) and np.all(point_world <= maxs))


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


def sample1obj_in_a(
    frames: List["FrameItem"],
    pairs: List[Tuple[int, int, float]],
    bbox_items: List[Dict],
    ignore_labels: Set[str],
) -> List[Tuple[Tuple[int, int, float], Dict]]:
    """采样在图像对的 image A 中出现的对象。返回 [(image_pair, instance), ...]"""
    results = []
    for pair in pairs:
        i, _, _ = pair
        fa = frames[i]
        stem_a = Path(fa.file_name).stem

        for inst in bbox_items:
            label = str(inst.get("label", ""))
            if label.lower() in ignore_labels:
                continue
            images = inst.get("images", [])
            mask_map = {Path(p).parent.name: p for p in images}
            if stem_a in mask_map:
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
    max_triplets_per_pair: int = 10,
) -> List[Tuple[Tuple[int, int, float], Tuple[Dict, Dict, Dict]]]:
    """从图像对的两个图像的实例并集中采样3个不同的对象。返回 [(image_pair, (instA, instB, instC)), ...]
    
    Args:
        max_triplets_per_pair: 每个图像对最多生成的3元组数量，超过则随机采样
    """
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
        
        # 打乱实例顺序，确保组合生成的随机性
        rng.shuffle(all_insts)
        
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
        
        # 收集所有满足条件的3元组
        valid_triplets = []
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
            
            valid_triplets.append((instA, instB, instC))
        
        # 如果有效3元组数量超过限制，随机采样
        if len(valid_triplets) > max_triplets_per_pair:
            valid_triplets = rng.sample(valid_triplets, max_triplets_per_pair)
        
        # 添加到结果中
        for triplet in valid_triplets:
            results.append((pair, triplet))
    
    return results


def sample2region_across(
    frames: List["FrameItem"],
    pairs: List[Tuple[int, int, float]],
    region_items: List[Dict],
    ignore_labels: Set[str],
    rng: random.Random,
    max_inst_pairs_per_image_pair: int = 3,
) -> List[Tuple[Tuple[int, int, float], Tuple[Dict, Dict]]]:
    """从图像对的两个图像中分别采样 region 对。"""
    results = []
    stem_to_regions = _build_stem_to_insts_map(region_items, frames, ignore_labels)

    for pair in pairs:
        i, j, _ = pair
        fa, fb = frames[i], frames[j]
        stem_a = Path(fa.file_name).stem
        stem_b = Path(fb.file_name).stem

        regions_a = stem_to_regions.get(stem_a, []).copy()
        regions_b = stem_to_regions.get(stem_b, []).copy()
        if not regions_a or not regions_b:
            continue

        rng.shuffle(regions_a)
        rng.shuffle(regions_b)

        count = 0
        for region_a in regions_a:
            if count >= max_inst_pairs_per_image_pair:
                break
            for region_b in regions_b:
                if count >= max_inst_pairs_per_image_pair:
                    break
                if region_a.get("ins_id") == region_b.get("ins_id"):
                    continue
                if region_a.get("label") == region_b.get("label"):
                    continue
                results.append((pair, (region_a, region_b)))
                count += 1
    return results


def sample_object_region_across(
    frames: List["FrameItem"],
    pairs: List[Tuple[int, int, float]],
    object_items: List[Dict],
    region_items: List[Dict],
    ignore_object_labels: Set[str],
    ignore_region_labels: Set[str],
    rng: random.Random,
    max_pairs_per_image_pair: int = 3,
) -> List[Tuple[Tuple[int, int, float], Tuple[Dict, Dict]]]:
    """从 image A 采样 object、从 image B 采样 region。"""
    results = []
    stem_to_objects = _build_stem_to_insts_map(object_items, frames, ignore_object_labels)
    stem_to_regions = _build_stem_to_insts_map(region_items, frames, ignore_region_labels)

    for pair in pairs:
        i, j, _ = pair
        fa, fb = frames[i], frames[j]
        stem_a = Path(fa.file_name).stem
        stem_b = Path(fb.file_name).stem

        objs_a = stem_to_objects.get(stem_a, []).copy()
        regions_b = stem_to_regions.get(stem_b, []).copy()
        if not objs_a or not regions_b:
            continue

        rng.shuffle(objs_a)
        rng.shuffle(regions_b)

        count = 0
        # 按 region 逐个尝试；如果该 region 包含了所有候选 object 的中心，则自动换下一个 region。
        for region in regions_b:
            if count >= max_pairs_per_image_pair:
                break
            valid_objs = []
            for obj in objs_a:
                if obj.get("label") == region.get("label"):
                    continue
                obj_center = _instance_center_world(obj)
                if obj_center is None:
                    continue
                if _point_inside_region_bbox(obj_center, region):
                    continue
                valid_objs.append(obj)

            if not valid_objs:
                continue

            rng.shuffle(valid_objs)
            for obj in valid_objs:
                if count >= max_pairs_per_image_pair:
                    break
                results.append((pair, (obj, region)))
                count += 1
    return results


def sample2obj1region(
    frames: List["FrameItem"],
    pairs: List[Tuple[int, int, float]],
    object_items: List[Dict],
    region_items: List[Dict],
    ignore_object_labels: Set[str],
    ignore_region_labels: Set[str],
    rng: random.Random,
    max_triplets_per_pair: int = 10,
) -> List[Tuple[Tuple[int, int, float], Tuple[Dict, Dict, Dict]]]:
    """
    从图像对中采样 (object A, object B, region C) 三元组。
    约束：region C 必须来自 image B。
    约束：region C 的 bbox 不能包含 object A 的中心点。
    """
    results: List[Tuple[Tuple[int, int, float], Tuple[Dict, Dict, Dict]]] = []
    stem_to_objects = _build_stem_to_insts_map(object_items, frames, ignore_object_labels)
    stem_to_regions = _build_stem_to_insts_map(region_items, frames, ignore_region_labels)

    for pair in pairs:
        i, j, _ = pair
        fa, fb = frames[i], frames[j]
        stem_i = Path(fa.file_name).stem
        stem_j = Path(fb.file_name).stem

        objs_i = stem_to_objects.get(stem_i, [])
        objs_j = stem_to_objects.get(stem_j, [])
        regions_j = stem_to_regions.get(stem_j, [])

        all_objs: List[Dict] = []
        seen_obj_ids = set()
        for inst in objs_i + objs_j:
            ins_id = str(inst.get("ins_id", ""))
            if ins_id in seen_obj_ids:
                continue
            seen_obj_ids.add(ins_id)
            all_objs.append(inst)

        all_regions: List[Dict] = []
        seen_region_ids = set()
        for region in regions_j:
            ins_id = str(region.get("ins_id", ""))
            if ins_id in seen_region_ids:
                continue
            seen_region_ids.add(ins_id)
            all_regions.append(region)

        if len(all_objs) < 2 or len(all_regions) < 1:
            continue

        rng.shuffle(all_objs)
        rng.shuffle(all_regions)

        valid_triplets: List[Tuple[Dict, Dict, Dict]] = []
        for instA, instB in itertools.combinations(all_objs, 2):
            labelA = instA.get("label")
            labelB = instB.get("label")
            if labelA == labelB:
                continue

            centerA = _instance_center_world(instA)
            if centerA is None:
                continue

            for regionC in all_regions:
                if _point_inside_region_bbox(centerA, regionC):
                    continue
                valid_triplets.append((instA, instB, regionC))

        if len(valid_triplets) > max_triplets_per_pair:
            valid_triplets = rng.sample(valid_triplets, max_triplets_per_pair)

        for triplet in valid_triplets:
            results.append((pair, triplet))

    return results


def sample2obj1region_for_proxy(
    frames: List["FrameItem"],
    pairs: List[Tuple[int, int, float]],
    object_items: List[Dict],
    region_items: List[Dict],
    ignore_object_labels: Set[str],
    ignore_region_labels: Set[str],
    rng: random.Random,
    max_triplets_per_pair: int = 10,
) -> List[Tuple[Tuple[int, int, float], Tuple[Dict, Dict, Dict]]]:
    """
    从图像对中采样 (object A, object B, region C) 三元组。
    约束：region C 必须来自 image B。
    用于 proxy QA，约束：region C 的 bbox 不能包含 object B 的中心点。
    """
    results: List[Tuple[Tuple[int, int, float], Tuple[Dict, Dict, Dict]]] = []
    stem_to_objects = _build_stem_to_insts_map(object_items, frames, ignore_object_labels)
    stem_to_regions = _build_stem_to_insts_map(region_items, frames, ignore_region_labels)

    for pair in pairs:
        i, j, _ = pair
        fa, fb = frames[i], frames[j]
        stem_i = Path(fa.file_name).stem
        stem_j = Path(fb.file_name).stem

        objs_i = stem_to_objects.get(stem_i, [])
        objs_j = stem_to_objects.get(stem_j, [])
        regions_j = stem_to_regions.get(stem_j, [])

        all_objs: List[Dict] = []
        seen_obj_ids = set()
        for inst in objs_i + objs_j:
            ins_id = str(inst.get("ins_id", ""))
            if ins_id in seen_obj_ids:
                continue
            seen_obj_ids.add(ins_id)
            all_objs.append(inst)

        all_regions: List[Dict] = []
        seen_region_ids = set()
        for region in regions_j:
            ins_id = str(region.get("ins_id", ""))
            if ins_id in seen_region_ids:
                continue
            seen_region_ids.add(ins_id)
            all_regions.append(region)

        if len(all_objs) < 2 or len(all_regions) < 1:
            continue

        rng.shuffle(all_objs)
        rng.shuffle(all_regions)

        valid_triplets: List[Tuple[Dict, Dict, Dict]] = []
        for instA, instB in itertools.combinations(all_objs, 2):
            labelA = instA.get("label")
            labelB = instB.get("label")
            if labelA == labelB:
                continue

            centerB = _instance_center_world(instB)
            if centerB is None:
                continue

            for regionC in all_regions:
                if _point_inside_region_bbox(centerB, regionC):
                    continue
                valid_triplets.append((instA, instB, regionC))

        if len(valid_triplets) > max_triplets_per_pair:
            valid_triplets = rng.sample(valid_triplets, max_triplets_per_pair)

        for triplet in valid_triplets:
            results.append((pair, triplet))

    return results


def main():
    parser = argparse.ArgumentParser(description="为两帧构造3D QA占位数据（基于covisibility>阈值随机抽样）。")
    parser.add_argument("--scene-id", default="all", help="例如 0a5c013435")
    # parser.add_argument("--scene-id", default="00777c41d4", help="例如 0a5c013435")
    # parser.add_argument("--scene-id", default="0b031f3119", help="例如 0a5c013435")
    # parser.add_argument("--scene-id", default="0a5c013435", help="例如 0a5c013435")
    # parser.add_argument("--scene-id", default="0d8ead0038", help="例如 0a5c013435")
    parser.add_argument(
        "--scene-id-list",
        type=str,
        default=None,
        help="场景ID列表（逗号分隔或JSON格式）。指定后仅处理这些场景；适合多卡并行",
    )
    parser.add_argument("--data-root", default="scannetppv2/data", type=Path)
    # parser.add_argument("--data-root", default="DL3DV/1K", type=Path)
    parser.add_argument("--wai-root", default="scannetppv2_wai", type=Path)
    # parser.add_argument("--wai-root", default="dl3dv_wai/", type=Path)
    parser.add_argument("--covis-threshold", default=0.25, type=float, help="covisibility 阈值，用于选取图像对")
    parser.add_argument("--num", default=100, type=int, help="每类采样多少对（translation、rotation各一轮）")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--marker-types",
        type=str,
        # default="point_marker,white_mask,mask_covering_image,2d_bbox",
        default="language_description",
        help="使用的标记方式列表，逗号分隔；可选: point_marker, white_mask, mask_covering_image, 2d_bbox, language_description",
    )
    parser.add_argument(
        "--bbox-json-folder",
        type=Path,
        # default="output_3d_bounding",
        default="output_scannetppv2_new_aabb_with_descriptions",
        # default="output_3d_bounding_scannetppv2_vllm_old_description",
        # default="Dl3dv_filter/1k",
        help="3D bbox 结果所在目录（包含description字段），文件名为 <scene_id>.json；不存在则跳过该类问题",
    )
    parser.add_argument(
        "--region-bbox-json-folder",
        type=Path,
        default="output_scannetppv2_region_new",
        help="region bbox 结果目录，文件名为 <scene_id>.json；用于 region-region / object-region QA",
    )
    parser.add_argument(
        "--output",
        # default="output_QA_lang",
        # default="output_QA_vllm_old_lang",
        # default="output_QA_dl3dv_1k",
        # default="output_QA_v11_mask",
        default="output_QA_new_lang_add_region",
        type=Path,
        help="输出目录（文件名为 <scene_id>.json）未提供则打印到stdout",
    )
    parser.add_argument("--save-rrd", action="store_true", help="保存场景相机+图片到 rrd")
    parser.add_argument("--rrd-output", type=Path, default="rerun_output_two_view")
    parser.add_argument("--ply-folder", type=Path, default="output", help="点云 ply 根目录")

    args = parser.parse_args()

    ensure_output_dir(args.output)

    # 解析 marker_types
    default_marker_types = ["point_marker", "white_mask", "mask_covering_image", "2d_bbox", "language_description"]
    marker_types = [m.strip() for m in args.marker_types.split(",") if m.strip()] if args.marker_types else default_marker_types
    if not marker_types:
        marker_types = default_marker_types

    # 解析 scene_id_list（用于多卡并行）
    def _parse_scene_id_list(s: Optional[str]) -> Optional[List[str]]:
        if not s:
            return None
        s = s.strip()
        if not s:
            return None
        if s.startswith("["):
            ids = json.loads(s)
            if not isinstance(ids, list):
                raise ValueError("--scene-id-list JSON 必须是 list")
            return [str(x) for x in ids]
        return [x.strip() for x in s.split(",") if x.strip()]

    scene_id_list = _parse_scene_id_list(args.scene_id_list)

    def iter_scene_ids() -> List[str]:
        # 1) 优先使用 scene-id-list
        if scene_id_list:
            return scene_id_list

        # 2) 如果指定单个 scene-id
        if str(args.scene_id).lower() != "all":
            return [args.scene_id]

        # 3) all: 扫描 data-root
        root = args.data_root
        out: List[str] = []
        for p in sorted(root.iterdir()):
            if not p.is_dir():
                continue
            if (p / "dslr" / "nerfstudio" / "transforms_undistorted.json").exists() or (p / "dense" /"rgb").exists():
                out.append(p.name)
        return out

    all_scene_ids = iter_scene_ids()

    # 多卡分批：当 scene-id=all 且未传 scene-id-list 时，仅打印批次并退出
    if (not scene_id_list) and str(args.scene_id).lower() == "all":
        # 先跳过 output 目录中已存在的 scene，避免重复跑
        existing = {p.stem for p in args.output.glob("*.json") if p.is_file()}
        if existing:
            all_scene_ids = [sid for sid in all_scene_ids if sid not in existing]

        all_scene_ids = sorted(all_scene_ids)
        print(f"\n按 PER_GPU_SIZE={PER_GPU_SIZE} 分组场景ID:")
        for i in range(0, len(all_scene_ids), PER_GPU_SIZE):
            batch = all_scene_ids[i : i + PER_GPU_SIZE]
            batch_num = i // PER_GPU_SIZE + 1
            print(f"\n批次 {batch_num}: {len(batch)} 个场景")
            print("批次 JSON格式（可直接复制）:")
            print(json.dumps(batch, ensure_ascii=False))
        print(f"\n总共 {len(all_scene_ids)} 个场景，分为 {(len(all_scene_ids) + PER_GPU_SIZE - 1) // PER_GPU_SIZE} 个批次")
        print("\n使用示例:")
        print('  --scene-id-list \'["scene1","scene2","scene3"]\'')
        print('  --scene-id-list "scene1,scene2,scene3"')
        return

    all_entries: List[Dict] = []
    base_rng = random.Random(args.seed)

    # 运行时也跳过已存在输出的 scene（多卡共享同一 output 目录时避免重复）
    existing = {p.stem for p in args.output.glob("*.json") if p.is_file()}
    if scene_id_list is not None:
        total_list = len(all_scene_ids)
        to_run = [sid for sid in all_scene_ids if sid not in existing]
        skipped = total_list - len(to_run)
        print(f"--scene-id-list 指定 {total_list} 个场景；output 已存在将跳过 {skipped} 个；实际处理 {len(to_run)} 个。")
        all_scene_ids = to_run

    for idx, scene_id in enumerate(all_scene_ids):
        out_file = args.output / f"{scene_id}.json"
        if out_file.exists():
            print(f"跳过场景 {scene_id}: 输出文件已存在: {out_file}")
            continue
        print(f"处理场景 {scene_id} ...")
        
        # 检查 bbox json 文件是否存在，如果不存在则跳过该 scene
        candidate_folder = args.bbox_json_folder
        candidate_path = candidate_folder / f"{scene_id}.json"
        if not candidate_path.exists():
            print(f"跳过场景 {scene_id}: bbox json 文件不存在: {candidate_path}")
            continue
        
        rng = random.Random(base_rng.random() * 1e9)

        scannet_meta = args.data_root / scene_id / "dslr" / "nerfstudio" / "transforms_undistorted.json"
        dl3dv_cam_dir = args.data_root / scene_id / "dense" / "cam"
        use_rub_view = False
        if scannet_meta.exists():
            intrinsics, frames, images_dir = load_transforms_scannetppv2(scene_id, args.data_root)
            use_rub_view = True
        elif dl3dv_cam_dir.exists():
            intrinsics, frames, images_dir = load_transforms_dl3dv(scene_id, args.data_root)
        else:
            print(f"跳过场景 {scene_id}: 未找到 scannetppv2 或 DL3DV 的相机参数文件")
            continue
        covis = load_covisibility(scene_id, args.wai_root)
        if covis is None:
            print(f"跳过场景 {scene_id}: 缺少或存在多个 covisibility npy")
            continue

        if covis.shape[0] != covis.shape[1] or covis.shape[0] != len(frames):
            print(f"跳过场景 {scene_id}: covisibility矩阵与frame数量不匹配: covis {covis.shape}, frames {len(frames)}")
            continue

        # 图像的文件必须存在
        valid_mask = validate_images(frames, images_dir)

        # Cam Translation/Rotation QA（非 region，已注释）
        entries_translation: List[Dict] = []
        entries_rotation: List[Dict] = []
        # pairs = sample_pairs(covis, valid_mask, args.covis_threshold, 2 * args.num, rng, frames=frames, min_rotation_deg=0.35)
        # entries_translation += [build_cam_translation_main_dir_entry(scene_id, intrinsics, frames, p, args.covis_threshold) for p in pairs]
        # pairs = sample_pairs(covis, valid_mask, args.covis_threshold, args.num, rng, frames=frames, min_translation_m=0.35)
        # entries_translation += [build_cam_translation_distance_exact_entry(scene_id, intrinsics, frames, p, args.covis_threshold) for p in pairs]
        # pairs = sample_pairs(covis, valid_mask, args.covis_threshold, args.num, rng, frames=frames, min_translation_m=0.35)
        # entries_translation += [build_cam_translation_distance_threshold_entry(scene_id, intrinsics, frames, p, args.covis_threshold, rng) for p in pairs]
        # pairs = sample_pairs(covis, valid_mask, args.covis_threshold, 2 * args.num, rng, frames=frames, min_rotation_deg=15.0)
        # entries_rotation = [entry for p in pairs if (entry := build_cam_rotation_entry(scene_id, intrinsics, frames, p, args.covis_threshold)) is not None]

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

        # Load bbox items for region QA
        region_bbox_items = []
        region_bbox_json_path: Path | None = None
        region_candidate_path = args.region_bbox_json_folder / f"{scene_id}.json"
        if region_candidate_path.exists():
            region_bbox_json_path = region_candidate_path
        if region_bbox_json_path:
            try:
                region_bbox_items = load_bbox_items(region_bbox_json_path)
                for region_inst in region_bbox_items:
                    region_inst["entity_type"] = "region"
                print(f"加载 region bbox json 成功: {region_bbox_json_path}, 实例数 {len(region_bbox_items)}")
            except Exception as exc:
                print(f"加载 region bbox json 失败，跳过 region QA: {exc}")
                region_bbox_items = []
        else:
            print(f"场景 {scene_id}: 未提供 region bbox json，跳过 region QA。")
        if args.save_rrd:
            ply_path = args.ply_folder / scene_id / "input.ply"
            save_scene_rrd(scene_id, intrinsics, frames, images_dir, args.rrd_output, bbox_items, ply_path, stride=10, use_rub_view=use_rub_view)

        # Object QA
        entries_object_depth: List[Dict] = []
        entries_object_dir: List[Dict] = []
        entries_inter_object: List[Dict] = []
        entries_object_height_or_length: List[Dict] = []
        entries_cross_dir: List[Dict] = []
        entries_region_cross_dir: List[Dict] = []
        entries_object_region_cross_dir: List[Dict] = []
        entries_proxy_dir: List[Dict] = []
        entries_region_proxy_dir: List[Dict] = []
        entries_relpos: List[Dict] = []
        entries_region_relpos: List[Dict] = []
        if bbox_items:
            ignore_labels = {"wall", "floor", "ceiling"}

            # 仅保留带 bbox 的帧，避免在巨大场景上随机到没有 bbox 的帧导致效率低且抽不到物体
            stems_with_bbox = {
                Path(p).parent.name
                for inst in bbox_items
                for p in inst.get("images", [])
            }
            bbox_indices = [idx for idx, fr in enumerate(frames) if Path(fr.file_name).stem in stems_with_bbox]

            if len(bbox_indices) < 2:
                print(f"场景 {scene_id}: 带 bbox 的帧过少 ({len(bbox_indices)}), 跳过 object QA。")
            else:
                frames_obj = [frames[i] for i in bbox_indices]
                covis_obj = covis[np.ix_(bbox_indices, bbox_indices)]
                valid_mask_obj = valid_mask[bbox_indices]

                if valid_mask_obj.sum() < 2:
                    print(f"场景 {scene_id}: 有效 bbox 帧不足，跳过 object QA。")
                else:
                    # One Object common seen in both images, depth QA
                    # pairs = sample_pairs(covis_obj, valid_mask_obj, args.covis_threshold, args.num, rng, frames=frames_obj, min_translation_m=0.2, min_rotation_deg=15.0)
                    # sampled_common = sample1obj_common(frames_obj, pairs, bbox_items, ignore_labels)
                    # for pair, inst in sampled_common:
                    #     entry = build_object_dpt_entry(scene_id, intrinsics, frames_obj, pair, args.covis_threshold, bbox_json_path, bbox_items=bbox_items, target_ins_id=inst.get("ins_id"), skip_labels=ignore_labels, rng=rng, marker_types=marker_types)
                    #     if entry.get("objects"):
                    #         entries_object_depth.append(entry)
                    # print(f"object depth QA 构建完成: {len(entries_object_depth)} 条")

                    # covis_low = max(0.0, args.covis_threshold - 0.20)
                    # covis_high = min(1.0, args.covis_threshold + 0.00)
                    # try:
                    #     pairs_dir = sample_pairs(covis_obj, valid_mask_obj, args.covis_threshold, args.num, rng, frames=frames_obj, min_translation_m=0.2, min_rotation_deg=15.0, covis_range=(covis_low, covis_high))
                    # except ValueError:
                    #     print(f"object direction covis 区间无可用对，回退到阈值采样: [{covis_low:.3f}, {covis_high:.3f}]")
                    #     pairs_dir = pairs

                    # sampled_in_a = sample1obj_in_a(frames_obj, pairs_dir, bbox_items, ignore_labels)
                    # for pair, inst in sampled_in_a:
                    #     entry = build_object_dir_entry(scene_id, intrinsics, frames_obj, pair, args.covis_threshold, bbox_json_path, bbox_items=bbox_items, target_ins_id=inst.get("ins_id"), skip_labels=ignore_labels, rng=rng, marker_types=marker_types)
                    #     if entry is not None:
                    #         entries_object_dir.append(entry)
                    # print(f"object direction QA 构建完成: {len(entries_object_dir)} 条")

                    # Two Objects across different images, distance QA
                    pairs = sample_pairs_region(
                        covis_obj,
                        valid_mask_obj,
                        args.covis_threshold,
                        args.num,
                        rng,
                        frames=frames_obj,
                        min_translation_m=0.2,
                        region_bbox_items=region_bbox_items,
                    )
                    # sampled = sample2obj_across(
                    #     frames_obj, pairs, bbox_items, ignore_labels, rng, max_inst_pairs_per_image_pair=3
                    # )
                    # for pair, (inst1, inst2) in sampled:
                    #     entry = build_object_dist_entry(scene_id, intrinsics, frames_obj, pair, args.covis_threshold, inst1, inst2, rng=rng, marker_types=marker_types)
                    #     entries_inter_object.append(entry)
                    # print(f"cross image 2 object distance QA 构建完成: {len(entries_inter_object)} 条")

                    # sampled = sample2obj_across(
                    #     frames_obj, pairs, bbox_items, ignore_labels, rng, max_inst_pairs_per_image_pair=3
                    # )
                    # for pair, (inst1, inst2) in sampled:
                    #     entry = build_object_height_entry(scene_id, intrinsics, frames_obj, pair, args.covis_threshold, inst1, inst2, rng=rng, marker_types=marker_types)
                    #     if entry is not None:
                    #         entries_object_height_or_length.append(entry)
                    # print(f"cross image object height/length QA 构建完成: {len(entries_object_height_or_length)} 条")

                    # sampled = sample2obj_across(
                    #     frames_obj, pairs, bbox_items, ignore_labels, rng, max_inst_pairs_per_image_pair=3
                    # )
                    # for pair, (inst1, inst2) in sampled:
                    #     entry = build_object_cross_dir_entry(scene_id, intrinsics, frames_obj, pair, args.covis_threshold, inst1, inst2, rng=rng, marker_types=marker_types)
                    #     if entry is not None:
                    #         entries_cross_dir.append(entry)
                    # print(f"cross image 2 object direction QA 构建完成: {len(entries_cross_dir)} 条")

                    if region_bbox_items:
                        sampled_object_region = sample_object_region_across(
                            frames_obj,
                            pairs,
                            bbox_items,
                            region_bbox_items,
                            ignore_labels,
                            set(),
                            rng,
                            max_pairs_per_image_pair=3,
                        )
                        for pair, (obj_inst, region_inst) in sampled_object_region:
                            entry = build_region_object_cross_dir_entry(
                                scene_id,
                                intrinsics,
                                frames_obj,
                                pair,
                                args.covis_threshold,
                                obj_inst,
                                region_inst,
                                rng=rng,
                                marker_types=marker_types,
                            )
                            if entry is not None:
                                entries_object_region_cross_dir.append(entry)
                        print(f"cross image object-region direction QA 构建完成: {len(entries_object_region_cross_dir)} 条")

                    # Proxy direction / Relative position QA
                    pairs = sample_pairs_region(
                        covis_obj,
                        valid_mask_obj,
                        args.covis_threshold,
                        args.num,
                        rng,
                        frames=frames_obj,
                        bbox_items=bbox_items,
                        region_bbox_items=region_bbox_items,
                        at_least_3_objs_1_region=True,
                    )
                    # sampled_proxy = sample3obj(frames_obj, pairs, bbox_items, ignore_labels, rng, max_triplets_per_pair=10)
                    # for pair, (instA, instB, instC) in sampled_proxy:
                    #     entry = build_object_proxy_dir_entry(scene_id, intrinsics, frames_obj, pair, args.covis_threshold, instA, instB, instC, rng=rng, marker_types=marker_types)
                    #     if entry is not None:
                    #         entries_proxy_dir.append(entry)
                    # print(f"object proxy direction QA 构建完成: {len(entries_proxy_dir)} 条")

                    if region_bbox_items:
                        sampled_region_proxy = sample2obj1region_for_proxy(
                            frames_obj,
                            pairs,
                            bbox_items,
                            region_bbox_items,
                            ignore_labels,
                            set(),
                            rng,
                            max_triplets_per_pair=10,
                        )
                        for pair, (instA, instB, regionC) in sampled_region_proxy:
                            entry = build_region_proxy_dir_entry(
                                scene_id,
                                intrinsics,
                                frames_obj,
                                pair,
                                args.covis_threshold,
                                instA,
                                instB,
                                regionC,
                                rng=rng,
                                marker_types=marker_types,
                            )
                            if entry is not None:
                                entries_region_proxy_dir.append(entry)
                        print(f"region proxy direction QA 构建完成: {len(entries_region_proxy_dir)} 条")

                    # sampled_relpos = sample3obj(frames_obj, pairs, bbox_items, ignore_labels, rng, max_triplets_per_pair=10)
                    # for pair, (instA, instB, instC) in sampled_relpos:
                    #     entry = build_object_relpos_entry(scene_id, intrinsics, frames_obj, pair, args.covis_threshold, instA, instB, instC, rng=rng, marker_types=marker_types)
                    #     if entry is not None:
                    #         entries_relpos.append(entry)
                    # print(f"relative position QA 构建完成: {len(entries_relpos)} 条")

                    if region_bbox_items:
                        sampled_region_relpos = sample2obj1region(
                            frames_obj,
                            pairs,
                            bbox_items,
                            region_bbox_items,
                            ignore_labels,
                            set(),
                            rng,
                            max_triplets_per_pair=10,
                        )
                        for pair, (instA, instB, regionC) in sampled_region_relpos:
                            entry = build_region_relpos_entry(
                                scene_id,
                                intrinsics,
                                frames_obj,
                                pair,
                                args.covis_threshold,
                                instA,
                                instB,
                                regionC,
                                rng=rng,
                                marker_types=marker_types,
                            )
                            if entry is not None:
                                entries_region_relpos.append(entry)
                        print(f"region relative position QA 构建完成: {len(entries_region_relpos)} 条")

        if region_bbox_items:
            stems_with_region_bbox = {
                Path(p).parent.name
                for inst in region_bbox_items
                for p in inst.get("images", [])
            }
            region_indices = [idx for idx, fr in enumerate(frames) if Path(fr.file_name).stem in stems_with_region_bbox]

            if len(region_indices) < 2:
                print(f"场景 {scene_id}: 带 region bbox 的帧过少 ({len(region_indices)}), 跳过 region-region QA。")
            else:
                frames_region = [frames[i] for i in region_indices]
                covis_region = covis[np.ix_(region_indices, region_indices)]
                valid_mask_region = valid_mask[region_indices]

                if valid_mask_region.sum() < 2:
                    print(f"场景 {scene_id}: 有效 region bbox 帧不足，跳过 region-region QA。")
                else:
                    pairs_region = sample_pairs(
                        covis_region,
                        valid_mask_region,
                        args.covis_threshold,
                        args.num,
                        rng,
                        frames=frames_region,
                        min_translation_m=0.2,
                    )
                    sampled_region = sample2region_across(
                        frames_region,
                        pairs_region,
                        region_bbox_items,
                        set(),
                        rng,
                        max_inst_pairs_per_image_pair=3,
                    )
                    for pair, (region_a, region_b) in sampled_region:
                        entry = build_region_cross_dir_entry(
                            scene_id,
                            intrinsics,
                            frames_region,
                            pair,
                            args.covis_threshold,
                            region_a,
                            region_b,
                            rng=rng,
                            marker_types=marker_types,
                        )
                        if entry is not None:
                            entries_region_cross_dir.append(entry)
                    print(f"cross image 2 region direction QA 构建完成: {len(entries_region_cross_dir)} 条")
        print(f"entries_translation: {len(entries_translation)}")
        print(f"entries_rotation: {len(entries_rotation)}")
        print(f"entries_object_depth: {len(entries_object_depth)}")
        print(f"entries_object_dir: {len(entries_object_dir)}")
        print(f"entries_inter_object: {len(entries_inter_object)}")
        print(f"entries_object_height_or_length: {len(entries_object_height_or_length)}")
        print(f"entries_cross_dir: {len(entries_cross_dir)}")
        print(f"entries_region_cross_dir: {len(entries_region_cross_dir)}")
        print(f"entries_object_region_cross_dir: {len(entries_object_region_cross_dir)}")
        print(f"entries_proxy_dir: {len(entries_proxy_dir)}")
        print(f"entries_region_proxy_dir: {len(entries_region_proxy_dir)}")
        print(f"entries_relpos: {len(entries_relpos)}")
        print(f"entries_region_relpos: {len(entries_region_relpos)}")
        entries = (
            # 仅保留 region 相关 QA
            entries_region_cross_dir
            + entries_object_region_cross_dir
            + entries_region_proxy_dir
            + entries_region_relpos
        )
        all_entries.extend(entries)

        if args.output:
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
