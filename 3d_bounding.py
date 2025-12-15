from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


DEFAULT_SCENE = "0a5c013435"
DEFAULT_DATA_ROOT = Path("/home/liuyifei/code/posevlm/scannetppv2/data")
DEFAULT_MASK_ROOT = Path("/home/liuyifei/code/posevlm/sam_masks")
DEFAULT_DEPTH_ROOT = Path("/home/liuyifei/code/posevlm/DptV3/data")
DEFAULT_OUTPUT_DIR = Path("/home/liuyifei/code/posevlm/sam_masks")


def setup_logger() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def load_transforms(scannet_dir: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    读取 nerfstudio transforms_undistorted.json，返回
      intr_map[stem] = 3x3 K
      c2w_map[stem] = 4x4 camera-to-world
    """
    json_path = scannet_dir / "dslr" / "nerfstudio" / "transforms_undistorted.json"
    with json_path.open("r", encoding="utf-8") as f:
        contents = json.load(f)

    fl_x, fl_y, cx, cy = contents["fl_x"], contents["fl_y"], contents["cx"], contents["cy"]
    train_frames = contents.get("frames", [])
    test_frames = contents.get("test_frames", [])

    intr_map: Dict[str, np.ndarray] = {}
    c2w_map: Dict[str, np.ndarray] = {}
    for frame in [*train_frames, *test_frames]:
        K = np.array([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]], dtype=np.float32)
        c2w = np.array(frame["transform_matrix"], dtype=np.float32)
        # OpenGL -> COLMAP 约定
        c2w[:3, 1:3] *= -1
        stem = Path(frame["file_path"]).stem
        intr_map[stem] = K
        c2w_map[stem] = c2w
    logging.info("加载位姿完成，数量: %d", len(intr_map))
    return intr_map, c2w_map


def load_mask(mask_path: Path) -> np.ndarray:
    mask = Image.open(mask_path).convert("L")
    return np.array(mask) > 0


def resize_mask_to_depth(mask: np.ndarray, depth: np.ndarray) -> np.ndarray:
    if mask.shape == depth.shape:
        return mask
    mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
    mask_resized = mask_img.resize((depth.shape[1], depth.shape[0]), resample=Image.NEAREST)
    return (np.array(mask_resized) > 0).astype(bool)


def mask_depth_to_points(
    mask: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
) -> np.ndarray:
    """将 mask 内像素 + 深度投影到世界坐标，返回 (N,3)。"""
    if mask.shape != depth.shape:
        mask = resize_mask_to_depth(mask, depth)

    ys, xs = np.where(mask)
    if ys.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    d = depth[ys, xs].astype(np.float32)
    valid = np.isfinite(d) & (d > 0)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)

    xs, ys, d = xs[valid], ys[valid], d[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x_cam = (xs - cx) / fx * d
    y_cam = (ys - cy) / fy * d
    z_cam = d

    ones = np.ones_like(z_cam)
    cam_pts = np.stack([x_cam, y_cam, z_cam, ones], axis=1)  # (N,4)
    world_pts = (cam_pts @ c2w.T)[:, :3]
    return world_pts.astype(np.float32)


def cluster_points(points: np.ndarray, voxel_size: float, min_points: int) -> List[np.ndarray]:
    """
    基于体素 6 邻域连通的简单聚类，返回每个实例的点索引数组。
    """
    if points.shape[0] == 0:
        return []
    vox = np.floor(points / voxel_size).astype(np.int64)
    voxel_to_indices: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
    for idx, v in enumerate(map(tuple, vox)):
        voxel_to_indices[v].append(idx)

    visited = set()
    components: List[List[int]] = []
    neighbors = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

    for voxel in voxel_to_indices:
        if voxel in visited:
            continue
        queue = deque([voxel])
        visited.add(voxel)
        comp_indices: List[int] = []
        while queue:
            v = queue.popleft()
            comp_indices.extend(voxel_to_indices[v])
            vx, vy, vz = v
            for dx, dy, dz in neighbors:
                nb = (vx + dx, vy + dy, vz + dz)
                if nb in voxel_to_indices and nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        if len(comp_indices) >= min_points:
            components.append(np.array(comp_indices, dtype=np.int32))
    return components


def bbox_corners(pts: np.ndarray) -> List[Dict[str, float]]:
    xyz_min = pts.min(axis=0)
    xyz_max = pts.max(axis=0)
    x0, y0, z0 = xyz_min.tolist()
    x1, y1, z1 = xyz_max.tolist()
    corners = [
        (x0, y0, z0),
        (x0, y1, z0),
        (x1, y1, z0),
        (x1, y0, z0),
        (x0, y0, z1),
        (x0, y1, z1),
        (x1, y1, z1),
        (x1, y0, z1),
    ]
    return [{"x": float(x), "y": float(y), "z": float(z)} for x, y, z in corners]


def process_scene(
    scene: str,
    data_root: Path,
    mask_root: Path,
    depth_root: Path,
    output_dir: Path,
    voxel_size: float,
    min_points: int,
) -> Path:
    scene_dir = data_root / scene
    intr_map, c2w_map = load_transforms(scene_dir)

    mask_index_path = mask_root / scene / "mask_index.json"
    with mask_index_path.open("r", encoding="utf-8") as f:
        mask_index = json.load(f)
    items = mask_index.get("items", [])

    depth_dir = depth_root / scene / "depth_da3"
    results: List[Dict] = []
    inst_counter = 1

    # 按标签聚合所有点
    label_points: Dict[str, List[np.ndarray]] = defaultdict(list)

    for item in items:
        image_name = item["image"]
        label = item["label"]
        mask_path = Path(item["mask_path"])
        stem = Path(image_name).stem

        if stem not in intr_map or stem not in c2w_map:
            logging.warning("找不到相机参数，跳过: %s", image_name)
            continue

        depth_path = depth_dir / f"{stem}.npy"
        if not depth_path.exists():
            logging.warning("找不到深度文件，跳过: %s", depth_path)
            continue

        depth = np.load(depth_path)
        mask = load_mask(mask_path)
        points = mask_depth_to_points(mask, depth, intr_map[stem], c2w_map[stem])
        if points.shape[0] == 0:
            continue
        label_points[label].append(points)

    for label, chunks in label_points.items():
        pts = np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]
        comps = cluster_points(pts, voxel_size=voxel_size, min_points=min_points)
        logging.info("标签 %s 聚类得到 %d 个实例", label, len(comps))
        for comp in comps:
            bbox = bbox_corners(pts[comp])
            results.append(
                {
                    "ins_id": str(inst_counter),
                    "label": label,
                    "bounding_box": bbox,
                }
            )
            inst_counter += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{scene}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logging.info("3D bounding boxes 已保存: %s (共 %d 个实例)", output_path, len(results))
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 2D mask 投影生成 3D bounding boxes")
    parser.add_argument("--scene", default=DEFAULT_SCENE, help="场景名，如 0a5c013435")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="scannetppv2/data 根目录")
    parser.add_argument("--mask-root", type=Path, default=DEFAULT_MASK_ROOT, help="sam mask 根目录（含 mask_index.json）")
    parser.add_argument("--depth-root", type=Path, default=DEFAULT_DEPTH_ROOT, help="深度 npy 根目录（scene/depth_da3）")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="输出 3D bbox 保存目录")
    parser.add_argument("--voxel-size", type=float, default=0.05, help="体素大小用于连通聚类（米）")
    parser.add_argument("--min-points", type=int, default=50, help="实例最少点数过滤")
    return parser.parse_args()


def main() -> None:
    setup_logger()
    args = parse_args()
    process_scene(
        scene=args.scene,
        data_root=args.data_root,
        mask_root=args.mask_root,
        depth_root=args.depth_root,
        output_dir=args.output_dir,
        voxel_size=args.voxel_size,
        min_points=args.min_points,
    )


if __name__ == "__main__":
    main()

