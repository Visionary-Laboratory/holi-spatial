#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import cv2
from PIL import Image


# ---------------------------
# Utils
# ---------------------------

def setup_logger() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

def label_color_bgr(label: str) -> Tuple[int, int, int]:
    """稳定的 label->颜色 (BGR)"""
    import hashlib
    h = hashlib.md5(label.encode("utf-8")).digest()
    return int(h[2]), int(h[1]), int(h[0])  # BGR

def normalize_image_name(name: str) -> str:
    return Path(name).stem

def parse_ins_ids(items: Optional[List[str]]) -> Optional[Set[str]]:
    """
    支持：
      --ins-ids 12 34
      --ins-ids 12,34,56
      --ins-ids "12, 34" 56
    """
    if not items:
        return None
    out: Set[str] = set()
    for x in items:
        for t in str(x).replace(" ", "").split(","):
            if t != "":
                out.add(t)
    return out if out else None

def extract_image_stems_from_paths(image_paths: List[str]) -> set[str]:
    """从 mask 路径列表里抽 stem（你样例里就是这样存的）"""
    stems = set()
    for p in image_paths:
        parts = Path(p).parts
        # 你的 json 里一般是 .../<stem>/<something>.png 这种
        for i, part in enumerate(parts):
            if i + 1 < len(parts) and parts[i + 1].lower().endswith((".png", ".jpg", ".jpeg")):
                stems.add(part)
                break
    return stems

def safe_inv(mat4: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(mat4)
    except np.linalg.LinAlgError:
        return np.eye(4, dtype=np.float32)


# ---------------------------
# Load camera intrinsics/poses (same style as your sample)
# ---------------------------

def load_transforms(scene_dir: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], str, Dict[str, Tuple[int, int]], Dict[str, str]]:
    """
    Returns:
      intr_map[stem] = 3x3 K
      c2w_map[stem]  = 4x4 camera-to-world
      scene_type     = "scannetppv2" or "dl3dv"
      size_map[stem] = (w,h)
      path_map[stem] = image_path
    """
    # ScanNet++ v2 nerfstudio style
    if (scene_dir / "dslr").exists():
        scene_type = "scannetppv2"
        json_path = scene_dir / "dslr/nerfstudio/transforms_undistorted.json"
        with json_path.open("r", encoding="utf-8") as f:
            contents = json.load(f)

        fl_x, fl_y, cx, cy = contents["fl_x"], contents["fl_y"], contents["cx"], contents["cy"]
        w, h = contents.get("w"), contents.get("h")
        frames = list(contents.get("frames", [])) + list(contents.get("test_frames", []))

        intr_map: Dict[str, np.ndarray] = {}
        c2w_map: Dict[str, np.ndarray] = {}
        size_map: Dict[str, Tuple[int, int]] = {}
        path_map: Dict[str, str] = {}

        for fr in frames:
            K = np.array([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]], dtype=np.float32)
            c2w = np.array(fr["transform_matrix"], dtype=np.float32)

            # OpenGL -> COLMAP convention (与你样例一致)
            c2w[:3, 1:3] *= -1

            stem = Path(fr["file_path"]).stem
            intr_map[stem] = K
            c2w_map[stem] = c2w

            fw = fr.get("w", w)
            fh = fr.get("h", h)
            size_map[stem] = (int(fw), int(fh))

            img_path = scene_dir / "dslr" / "resized_undistorted_images" / fr["file_path"]
            path_map[stem] = str(img_path)

        logging.info("Loaded cameras: %d", len(intr_map))
        return intr_map, c2w_map, scene_type, size_map, path_map

    # DL3DV npz style
    if (scene_dir / "dense").exists():
        scene_type = "dl3dv"
        cam_dir = scene_dir / "dense/cam"
        rgb_dir = scene_dir / "dense/rgb"
        cam_files = sorted([p for p in os.listdir(cam_dir) if p.endswith(".npz")])

        intr_map: Dict[str, np.ndarray] = {}
        c2w_map: Dict[str, np.ndarray] = {}
        size_map: Dict[str, Tuple[int, int]] = {}
        path_map: Dict[str, str] = {}

        for cam_file in cam_files:
            data = np.load(str(Path(cam_dir) / cam_file))
            if "intrinsic" in data:
                intrinsic = data["intrinsic"]
            elif "intrinsics" in data:
                intrinsic = data["intrinsics"]
            else:
                raise ValueError(f"No intrinsics in {cam_file}, keys={list(data.keys())}")

            fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            c2w = data["pose"].astype(np.float32)

            stem = Path(cam_file).stem
            intr_map[stem] = K
            c2w_map[stem] = c2w

            img_path = Path(rgb_dir) / f"{stem}.png"
            if not img_path.exists():
                img_path = Path(rgb_dir) / f"{stem}.jpg"
            path_map[stem] = str(img_path)

            if "width" in data and "height" in data:
                size_map[stem] = (int(data["width"]), int(data["height"]))
            else:
                with Image.open(img_path) as im:
                    size_map[stem] = im.size

        logging.info("Loaded cameras: %d", len(intr_map))
        return intr_map, c2w_map, scene_type, size_map, path_map

    raise NotImplementedError(f"Unsupported scene dir: {scene_dir}")


# ---------------------------
# OBB -> corners -> projection
# ---------------------------

# 8个角点的连接关系：
# 底面（z较小）：0, 1, 2, 3
# 顶面（z较大）：4, 5, 6, 7
# 对应关系：0-4, 1-5, 2-6, 3-7 在同一z轴连线上
_EDGES = [
    # 底面的4条边
    (0, 1), (1, 3), (3, 2), (2, 0),
    # 顶面的4条边
    (4, 5), (5, 7), (7, 6), (6, 4),
    # 连接底面和顶面的4条垂直边
    (0, 4), (1, 5), (2, 6), (3, 7),
]

_FACES = [
    (0, 1, 3, 2),  # 底面（z较小）
    (4, 5, 7, 6),  # 顶面（z较大）
    (0, 1, 5, 4),  # 前面（y较小）
    (2, 3, 7, 6),  # 后面（y较大）
    (0, 2, 6, 4),  # 左面（x较小）
    (1, 3, 7, 5),  # 右面（x较大）
]

_NEAR_PLANE = 1e-3

def parse_corners_from_box(box: Dict) -> Optional[np.ndarray]:
    """从 box 中读取 8 个角点，支持多种格式"""
    # 尝试 bounding_box 字段
    if "bounding_box" in box:
        bb = box["bounding_box"]
        if isinstance(bb, list) and len(bb) == 8:
            # 格式1: [{"x":..,"y":..,"z":..}, ...]
            if isinstance(bb[0], dict):
                return np.array([[float(p["x"]), float(p["y"]), float(p["z"])] for p in bb], dtype=np.float32)
            # 格式2: [[x,y,z], ...]
            elif isinstance(bb[0], (list, tuple)) and len(bb[0]) == 3:
                return np.array([[float(p[0]), float(p[1]), float(p[2])] for p in bb], dtype=np.float32)
    # 尝试 corners 字段
    if "corners" in box:
        corners = box["corners"]
        if isinstance(corners, list) and len(corners) == 8:
            if isinstance(corners[0], dict):
                return np.array([[float(p["x"]), float(p["y"]), float(p["z"])] for p in corners], dtype=np.float32)
            elif isinstance(corners[0], (list, tuple)) and len(corners[0]) == 3:
                return np.array([[float(p[0]), float(p[1]), float(p[2])] for p in corners], dtype=np.float32)
    return None

def _pca_axes(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return center c (3,) and axes matrix A (3,3) whose columns are orthonormal axes."""
    c = points.mean(axis=0)
    X = points - c
    cov = (X.T @ X) / X.shape[0]
    w, v = np.linalg.eigh(cov)  # v: columns are eigenvectors
    order = np.argsort(w)[::-1]
    v = v[:, order]
    # make right-handed
    if np.linalg.det(v) < 0:
        v[:, 2] *= -1.0
    return c, v

def upright_box_from_corners(points: np.ndarray) -> np.ndarray:
    """
    从 8 个角点创建一个有一个面平行于 xy 平面的新长方体，最小包围原始角点。
    参考 upright_box 方法。
    Returns: new_corners (8,3)
    """
    c, axes = _pca_axes(points)  # columns are local axes in world
    X = points - c
    coords = X @ axes  # local coordinates
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    lengths = maxs - mins  # full edge lengths in each PCA axis

    world_z = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # choose which original axis becomes vertical (closest to world Z)
    align = np.abs(axes.T @ world_z)  # dot(axis_i, world_z)
    i_vert = int(np.argmax(align))

    idx = [0, 1, 2]
    idx.remove(i_vert)
    i_a, i_b = idx[0], idx[1]  # the two horizontal axes (order to be decided)

    def xy_norm(v: np.ndarray) -> float:
        return np.hypot(float(v[0]), float(v[1]))

    a = axes[:, i_a]
    b = axes[:, i_b]
    # pick the horizontal axis with stronger XY projection as new_x direction
    if xy_norm(a) < xy_norm(b):
        i_a, i_b = i_b, i_a
        a, b = b, a

    new_z = world_z
    a_xy = np.array([a[0], a[1], 0.0], dtype=np.float32)
    n = np.linalg.norm(a_xy)

    if n < 1e-8:
        b_xy = np.array([b[0], b[1], 0.0], dtype=np.float32)
        n2 = np.linalg.norm(b_xy)
        if n2 < 1e-8:
            new_x = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            new_x = b_xy / n2
            i_a, i_b = i_b, i_a
    else:
        new_x = a_xy / n

    new_y = np.cross(new_z, new_x)
    ny = np.linalg.norm(new_y)
    if ny < 1e-8:
        new_x = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        new_y = np.cross(new_z, new_x)
        ny = np.linalg.norm(new_y)
    new_y = new_y / ny
    new_x = np.cross(new_y, new_z)  # re-orthonormalize

    Lx = float(lengths[i_a])
    Ly = float(lengths[i_b])
    Lz = float(lengths[i_vert])

    hx, hy, hz = Lx / 2.0, Ly / 2.0, Lz / 2.0

    # canonical order: bottom 4 then top 4
    # 确保底面4个点z值相同，顶面4个点z值相同，且对应点在同一z轴连线上
    # 底面（z = -hz）: 0, 1, 2, 3
    # 顶面（z = +hz）: 4, 5, 6, 7
    # 对应关系：0-4, 1-5, 2-6, 3-7 在同一z轴连线上
    signs = [
        (-1, -1, -1),  # 0: 底面左下
        (+1, -1, -1),  # 1: 底面右下
        (-1, +1, -1),  # 2: 底面左上
        (+1, +1, -1),  # 3: 底面右上
        (-1, -1, +1),  # 4: 顶面左下（与0在同一z轴连线）
        (+1, -1, +1),  # 5: 顶面右下（与1在同一z轴连线）
        (-1, +1, +1),  # 6: 顶面左上（与2在同一z轴连线）
        (+1, +1, +1),  # 7: 顶面右上（与3在同一z轴连线）
    ]
    corners = np.array(
        [c + sx * hx * new_x + sy * hy * new_y + sz * hz * new_z for sx, sy, sz in signs],
        dtype=np.float32,
    )
    return corners

def obb_corners_world(transform: np.ndarray, extents: np.ndarray) -> np.ndarray:
    """transform: 4x4 world, extents: (3,) full lengths"""
    half = extents.astype(np.float32) * 0.5
    corners_local = np.array(
        [[sx * half[0], sy * half[1], sz * half[2]]
         for sx in (-1.0, 1.0)
         for sy in (-1.0, 1.0)
         for sz in (-1.0, 1.0)],
        dtype=np.float32
    )
    Rw = transform[:3, :3].astype(np.float32)
    tw = transform[:3, 3].astype(np.float32)
    return corners_local @ Rw.T + tw  # (8,3)

def get_corners_world(box: Dict, make_upright: bool = False, scene_type: str = "scannetppv2") -> Optional[np.ndarray]:
    """
    优先从 box 读取 8 个角点，如果没有则从 transform 和 extents 计算。
    如果 make_upright=True，会将角点转换为有一个面平行于 xy 平面的新长方体。
    默认 make_upright=False，直接读取使用原始角点。
    
    对于 DL3DV 场景类型，直接使用 obb_transform 和 obb_extents 计算角点，跳过读取角点和 upright 处理。
    """
    # 对于 DL3DV 场景类型，直接使用 obb_transform 和 obb_extents
    if scene_type == "dl3dv":
        if "obb_transform" in box and "obb_extents" in box:
            transform = np.array(box["obb_transform"], dtype=np.float32)
            extents = np.array(box["obb_extents"], dtype=np.float32).reshape(3)
            corners = obb_corners_world(transform, extents)
            return corners
        else:
            return None
    
    # 其他场景类型：优先尝试读取 8 个角点
    corners = parse_corners_from_box(box)
    if corners is None:
        # 回退到从 transform 和 extents 计算
        if "obb_transform" in box and "obb_extents" in box:
            transform = np.array(box["obb_transform"], dtype=np.float32)
            extents = np.array(box["obb_extents"], dtype=np.float32).reshape(3)
            corners = obb_corners_world(transform, extents)
        else:
            return None
    
    # 如果需要转换为 upright box（有一个面平行于 xy 平面）
    if make_upright and corners is not None:
        corners = upright_box_from_corners(corners)
    
    return corners

def project_points(world_pts: np.ndarray, K: np.ndarray, w2c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (uv (N,2), z_cam (N,))"""
    N = world_pts.shape[0]
    pts_h = np.concatenate([world_pts, np.ones((N, 1), dtype=np.float32)], axis=1)
    cam = (pts_h @ w2c.T)[:, :3]
    return project_cam_points(cam, K)

def project_cam_points(cam_pts: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """cam_pts: (N,3) in camera coordinates -> (uv, z_cam)."""
    z = cam_pts[:, 2].astype(np.float32)
    z_safe = np.clip(z, _NEAR_PLANE, None)
    uv = (cam_pts[:, :2] / z_safe[:, None]) @ K[:2, :2].T + K[:2, 2]
    return uv.astype(np.float32), z

def world_to_camera(world_pts: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    N = world_pts.shape[0]
    pts_h = np.concatenate([world_pts, np.ones((N, 1), dtype=np.float32)], axis=1)
    return (pts_h @ w2c.T)[:, :3].astype(np.float32)

def point_inside_box(point_w: np.ndarray, corners_w: np.ndarray, margin: float = 1e-4) -> bool:
    """近似判断点是否在由 corners_w 表示的盒体内部。"""
    c, axes = _pca_axes(corners_w)
    local = (point_w - c) @ axes
    coords = (corners_w - c) @ axes
    mins = coords.min(axis=0) - margin
    maxs = coords.max(axis=0) + margin
    return bool(np.all(local >= mins) and np.all(local <= maxs))

def clip_segment_to_near_plane(p0: np.ndarray, p1: np.ndarray, near: float = _NEAR_PLANE) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """在相机坐标系下将线段裁剪到 z>=near。"""
    z0, z1 = float(p0[2]), float(p1[2])
    in0 = z0 >= near
    in1 = z1 >= near
    if in0 and in1:
        return p0, p1
    if (not in0) and (not in1):
        return None
    t = (near - z0) / (z1 - z0)
    t = float(np.clip(t, 0.0, 1.0))
    pc = p0 + t * (p1 - p0)
    if in0:
        return p0, pc
    return pc, p1

def clip_polygon_by_near_plane(cam_poly: np.ndarray, near: float = _NEAR_PLANE) -> np.ndarray:
    """
    对相机坐标系中的多边形做近裁剪面裁剪（z>=near）。
    返回裁剪后顶点，可能为空。
    """
    if cam_poly.shape[0] < 3:
        return np.empty((0, 3), dtype=np.float32)
    out: List[np.ndarray] = []
    prev = cam_poly[-1]
    prev_in = prev[2] >= near
    for cur in cam_poly:
        cur_in = cur[2] >= near
        if cur_in != prev_in:
            t = (near - float(prev[2])) / float(cur[2] - prev[2])
            t = float(np.clip(t, 0.0, 1.0))
            inter = prev + t * (cur - prev)
            out.append(inter.astype(np.float32))
        if cur_in:
            out.append(cur.astype(np.float32))
        prev = cur
        prev_in = cur_in
    if len(out) < 3:
        return np.empty((0, 3), dtype=np.float32)
    return np.stack(out, axis=0).astype(np.float32)

def bbox_overlaps_one_camera(box: Dict, K: np.ndarray, c2w: np.ndarray, size: Tuple[int, int], make_upright: bool = False, scene_type: str = "scannetppv2") -> bool:
    """使用视锥半空间检测 bbox 与相机视野重叠，减少边缘帧抖动。"""
    corners_w = get_corners_world(box, make_upright=make_upright, scene_type=scene_type)
    if corners_w is None:
        return False
    w2c = safe_inv(c2w)
    corners_c = world_to_camera(corners_w, w2c)
    cam_pos_w = c2w[:3, 3].astype(np.float32)
    if point_inside_box(cam_pos_w, corners_w):
        return True

    z = corners_c[:, 2]
    if np.all(z < _NEAR_PLANE):
        return False

    w, h = size
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    if fx <= 0 or fy <= 0:
        return False

    left = -cx / fx
    right = (float(w) - cx) / fx
    top = -cy / fy
    bottom = (float(h) - cy) / fy

    x = corners_c[:, 0]
    y = corners_c[:, 1]

    # 分离轴式 reject: 8 个角点若全部落在某个平面外侧，则一定不相交
    if np.all(x < left * z):
        return False
    if np.all(x > right * z):
        return False
    if np.all(y < top * z):
        return False
    if np.all(y > bottom * z):
        return False
    return True


# ---------------------------
# Drawing
# ---------------------------

def draw_boxes_on_image(
    img_bgr: np.ndarray,
    boxes: List[Dict],
    K: np.ndarray,
    c2w: np.ndarray,
    show_labels: bool,
    include_desc: bool,
    alpha: float,
    thickness: int,
    font_scale: float,
    render: str,  # wire | solid | both
    make_upright: bool = False,  # 是否将角点转换为 upright box（有一个面平行于 xy 平面）
    scene_type: str = "scannetppv2",  # 场景类型
) -> np.ndarray:
    w2c = safe_inv(c2w)
    img_h, img_w = img_bgr.shape[:2]
    clip_rect = (0, 0, img_w, img_h)
    cam_pos_w = c2w[:3, 3].astype(np.float32)

    # 预计算 box 深度，按远->近画（更像论文叠加效果）
    prepared = []
    for box in boxes:
        corners_w = get_corners_world(box, make_upright=make_upright, scene_type=scene_type)
        if corners_w is None:
            continue

        corners_c = world_to_camera(corners_w, w2c)
        if np.all(corners_c[:, 2] < _NEAR_PLANE):
            continue

        camera_inside = point_inside_box(cam_pos_w, corners_w)
        z_pos = corners_c[corners_c[:, 2] >= _NEAR_PLANE, 2]
        sort_depth = float(np.mean(z_pos)) if z_pos.size > 0 else float(np.mean(corners_c[:, 2]))
        prepared.append((sort_depth, box, corners_w, corners_c, camera_inside))
    prepared.sort(key=lambda x: -x[0])  # far first

    # ========== solid / both: 半透明面 ==========
    if render in ("solid", "both"):
        overlay = img_bgr.copy()
        face_items = []
        for zc, box, corners_w, corners_c, camera_inside in prepared:
            # 相机在盒体内部时，solid 面会非常不稳定，改为只画线框更稳妥
            if camera_inside:
                continue
            for f in _FACES:
                face_cam = corners_c[list(f)]
                clipped_cam = clip_polygon_by_near_plane(face_cam, _NEAR_PLANE)
                if clipped_cam.shape[0] < 3:
                    continue
                poly, zs = project_cam_points(clipped_cam, K)
                if (np.max(poly[:, 0]) < 0) or (np.min(poly[:, 0]) >= img_w) or (np.max(poly[:, 1]) < 0) or (np.min(poly[:, 1]) >= img_h):
                    continue
                depth = float(np.mean(zs))
                face_items.append((depth, box, poly))

        face_items.sort(key=lambda x: -x[0])  # far first
        for depth, box, poly in face_items:
            label = str(box.get("label", "obj"))
            col = label_color_bgr(label)
            pts = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
            cv2.fillConvexPoly(overlay, pts, col)

        img_bgr = cv2.addWeighted(overlay, float(alpha), img_bgr, float(1.0 - alpha), 0)

    # ========== wire / both: 线框 + 标签 ==========
    if render in ("wire", "both"):
        for zc, box, corners_w, corners_c, camera_inside in prepared:
            label = str(box.get("label", "obj"))
            desc = box.get("description")
            if include_desc and isinstance(desc, str) and desc.strip():
                text = f"{label}: {desc.strip()}"
            else:
                text = label

            col = label_color_bgr(label)

            # edges
            for i, j in _EDGES:
                clipped = clip_segment_to_near_plane(corners_c[i], corners_c[j], _NEAR_PLANE)
                if clipped is None:
                    continue
                p0c, p1c = clipped
                uv, _ = project_cam_points(np.stack([p0c, p1c], axis=0), K)
                p1 = tuple(np.round(uv[0]).astype(int))
                p2 = tuple(np.round(uv[1]).astype(int))
                ok, c1, c2 = cv2.clipLine(clip_rect, p1, p2)
                if not ok:
                    continue
                cv2.line(img_bgr, c1, c2, col, thickness, cv2.LINE_AA)

            # label at projected center
            if show_labels:
                center_w = corners_w.mean(axis=0)
                uv_c, zc2 = project_points(center_w[None, :], K, w2c)
                if zc2[0] > _NEAR_PLANE:
                    x, y = np.round(uv_c[0]).astype(int)
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, thickness))
                    x0, y0 = max(0, x), max(0, y - th - 4)
                    x1, y1 = min(img_bgr.shape[1] - 1, x0 + tw + 6), min(img_bgr.shape[0] - 1, y0 + th + 6)
                    cv2.rectangle(img_bgr, (x0, y0), (x1, y1), (0, 0, 0), -1)
                    cv2.putText(img_bgr, text, (x0 + 3, y0 + th + 2),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                                max(1, thickness), cv2.LINE_AA)

    # ========== solid only 但仍想要 labels ==========
    if render == "solid" and show_labels:
        for zc, box, corners_w, corners_c, camera_inside in prepared:
            label = str(box.get("label", "obj"))
            desc = box.get("description")
            if include_desc and isinstance(desc, str) and desc.strip():
                text = f"{label}: {desc.strip()}"
            else:
                text = label
            center_w = corners_w.mean(axis=0)
            uv_c, zc2 = project_points(center_w[None, :], K, w2c)
            if zc2[0] <= _NEAR_PLANE:
                continue
            x, y = np.round(uv_c[0]).astype(int)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            x0, y0 = max(0, x), max(0, y - th - 4)
            x1, y1 = min(img_bgr.shape[1] - 1, x0 + tw + 6), min(img_bgr.shape[0] - 1, y0 + th + 6)
            cv2.rectangle(img_bgr, (x0, y0), (x1, y1), (0, 0, 0), -1)
            cv2.putText(img_bgr, text, (x0 + 3, y0 + th + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    return img_bgr


def make_grid(images: List[Path], out_path: Path, cols: int, pad: int = 6, bg=(255, 255, 255)) -> None:
    if not images:
        return
    ims = [Image.open(p).convert("RGB") for p in images]
    w, h = ims[0].size
    rows = (len(ims) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * w + (cols - 1) * pad, rows * h + (rows - 1) * pad), bg)
    for idx, im in enumerate(ims):
        r, c = divmod(idx, cols)
        x = c * (w + pad)
        y = r * (h + pad)
        canvas.paste(im, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


# ---------------------------
# Main
# ---------------------------

def process_once(
    input_json: Path,
    data_root: Path,
    output_dir: Path,
    selected_stems: List[str],
    ins_id_filter: Optional[Set[str]],
    show_all_labels: bool,
    frustum_filter: bool,
    show_labels: bool,
    include_desc: bool,
    downscale: int,
    alpha: float,
    thickness: int,
    font_scale: float,
    render: str,
    tag: Optional[str] = None,
    make_grid_flag: bool = False,
    grid_cols: int = 6,
    make_upright: bool = False,  # 是否将角点转换为 upright box（有一个面平行于 xy 平面）
    use_image_association: bool = False,  # 是否使用 box["images"] 做帧级筛选（默认关闭）
) -> None:
    scene = input_json.stem
    scene_dir = data_root / scene
    intr_map, c2w_map, scene_type, _, path_map = load_transforms(scene_dir)

    with input_json.open("r", encoding="utf-8") as f:
        bboxes = json.load(f)
    logging.info("Loaded boxes: %d", len(bboxes))

    # 先按 ins_id 做全局筛一遍（更快）
    if ins_id_filter is not None:
        bboxes = [b for b in bboxes if str(b.get("ins_id", "")) in ins_id_filter]
        logging.info("After ins_id filter: %d (ins_ids=%s)", len(bboxes), sorted(list(ins_id_filter))[:20])

    run_dir = output_dir / (f"{scene}_{tag}" if tag else scene)
    run_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for stem in selected_stems:
        if stem not in intr_map or stem not in c2w_map or stem not in path_map:
            continue
        img_path = Path(path_map[stem])
        if not img_path.exists():
            logging.warning("Missing image: %s", img_path)
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)  # BGR
        if img is None:
            logging.warning("Failed to read: %s", img_path)
            continue

        K = intr_map[stem].copy()
        c2w = c2w_map[stem].copy()

        if downscale > 1:
            img = cv2.resize(img, (img.shape[1] // downscale, img.shape[0] // downscale), interpolation=cv2.INTER_AREA)
            K[0, 0] /= downscale
            K[1, 1] /= downscale
            K[0, 2] /= downscale
            K[1, 2] /= downscale

        # pick boxes for this frame
        boxes_this = []
        for box in bboxes:
            # 检查是否有 8 个角点或 transform/extents
            corners_w = get_corners_world(box, scene_type=scene_type)
            if corners_w is None:
                continue

            # 仅在显式开启时使用 box["images"] 关联；默认关闭以避免大量帧空白
            if use_image_association and ins_id_filter is None and (not show_all_labels):
                stems_in_box = extract_image_stems_from_paths(box.get("images", []))
                if stem not in stems_in_box:
                    continue

            # 默认按视锥过滤，确保无 ins_id 时也按几何可见性稳定绘制
            if frustum_filter:
                corners_w = get_corners_world(box, make_upright=make_upright, scene_type=scene_type)
                if corners_w is None:
                    continue
                if not bbox_overlaps_one_camera(box, K, c2w, (img.shape[1], img.shape[0]), make_upright=make_upright, scene_type=scene_type):
                    continue

            boxes_this.append(box)

        out = draw_boxes_on_image(
            img, boxes_this, K, c2w,
            show_labels=show_labels,
            include_desc=include_desc,
            alpha=alpha,
            thickness=thickness,
            font_scale=font_scale,
            render=render,
            make_upright=make_upright,
            scene_type=scene_type,
        )
        out_path = run_dir / f"{stem}.png"
        cv2.imwrite(str(out_path), out)
        saved.append(out_path)

    logging.info("Saved overlays: %d -> %s", len(saved), run_dir)

    if make_grid_flag and saved:
        grid_path = run_dir / f"_grid_{scene}.png"
        make_grid(saved, grid_path, cols=grid_cols)
        logging.info("Saved grid: %s", grid_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("./bbox_overlay_out"))

    # 你要的：指定 images / ins_id
    parser.add_argument("--images", nargs="*", default=None, help="指定要渲染的图像 stem（或带扩展名）。不指定=所有图")
    parser.add_argument("--ins-ids", nargs="*", default=None, help="要绘制的 ins_id 列表/逗号分隔。不指定=不过滤实例")

    parser.add_argument("--random-runs", type=int, default=0)
    parser.add_argument("--random-min", type=int, default=6)
    parser.add_argument("--random-max", type=int, default=6)
    parser.add_argument("--seed", type=int, default=32)

    parser.add_argument("--show-all-labels", action="store_true",
                        help="不按 box['images'] 关联筛 box（但仍会按视锥过滤，除非你关掉）")
    parser.add_argument(
        "--use-image-association",
        action="store_true",
        default=False,
        help="启用旧逻辑：在未指定 ins_id 且未开启 show-all-labels 时，按 box['images'] 做帧级筛选（默认关闭）",
    )
    parser.add_argument("--frustum-filter", action="store_true", default=True, help="show-all-labels 或 ins_id 模式时按视锥过滤（默认开）")
    parser.add_argument("--no-frustum-filter", action="store_false", dest="frustum_filter")

    parser.add_argument("--show-labels", action="store_true", default=True)
    parser.add_argument("--no-show-labels", action="store_false", dest="show_labels")

    parser.add_argument("--include-desc", action="store_true", default=True)
    parser.add_argument("--no-include-desc", action="store_false", dest="include_desc")

    parser.add_argument("--render", type=str, default="both", choices=["wire", "solid", "both"],
                        help="wire=只画线框；solid=只画半透明面；both=面+线")
    parser.add_argument("--downscale", type=int, default=2, help="图像下采样倍数（1=不缩放）")
    parser.add_argument("--alpha", type=float, default=0.12, help="solid/both 时面填充透明度")
    parser.add_argument("--thickness", type=int, default=2)
    parser.add_argument("--font-scale", type=float, default=0.5)

    parser.add_argument("--make-grid", action="store_true", help="把本次输出拼成 grid（方便直接放论文）")
    parser.add_argument("--grid-cols", type=int, default=6)
    
    parser.add_argument(
        "--make-upright",
        action="store_true",
        default=False,
        help="将角点转换为 upright box（有一个面平行于 xy 平面），默认关闭，直接读取使用原始角点",
    )

    args = parser.parse_args()
    setup_logger()

    scene = args.input_json.stem
    intr_map, c2w_map, _, _, _ = load_transforms(args.data_root / scene)
    available = sorted(list(c2w_map.keys()))
    if not available:
        raise RuntimeError("No cameras found.")

    rng = random.Random(args.seed)

    # image selection
    if args.images:
        stems_fixed = [normalize_image_name(x) for x in args.images]
    else:
        stems_fixed = available

    # ins_id selection
    ins_id_filter = parse_ins_ids(args.ins_ids)

    # build selected stems
    if args.random_runs and args.random_runs > 0:
        for run_idx in range(args.random_runs):
            k = rng.randint(args.random_min, args.random_max)
            k = min(k, len(stems_fixed))
            stems = rng.sample(stems_fixed, k)
            tag = f"r{run_idx+1:02d}_k{k}"
            process_once(
                args.input_json, args.data_root, args.output_dir, stems,
                ins_id_filter=ins_id_filter,
                show_all_labels=args.show_all_labels,
                frustum_filter=args.frustum_filter,
                show_labels=args.show_labels,
                include_desc=args.include_desc,
                downscale=args.downscale,
                alpha=args.alpha,
                thickness=args.thickness,
                font_scale=args.font_scale,
                render=args.render,
                tag=tag,
                make_grid_flag=args.make_grid,
                grid_cols=args.grid_cols,
                make_upright=args.make_upright,
                use_image_association=args.use_image_association,
            )
    else:
        process_once(
            args.input_json, args.data_root, args.output_dir, stems_fixed,
            ins_id_filter=ins_id_filter,
            show_all_labels=args.show_all_labels,
            frustum_filter=args.frustum_filter,
            show_labels=args.show_labels,
            include_desc=args.include_desc,
            downscale=args.downscale,
            alpha=args.alpha,
            thickness=args.thickness,
            font_scale=args.font_scale,
            render=args.render,
            tag=None,
            make_grid_flag=args.make_grid,
            grid_cols=args.grid_cols,
            make_upright=args.make_upright,
            use_image_association=args.use_image_association,
        )

if __name__ == "__main__":
    main()
