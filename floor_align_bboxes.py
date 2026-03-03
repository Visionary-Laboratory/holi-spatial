
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Floor-aligned OBB conversion script.

Input JSON format (typical):
- Top-level is a list of instances, each instance is a dict with keys:
  - "label": str
  - "bounding_box": list[8] of {"x","y","z"} corner points (order matters)
  - "obb_transform": 4x4 matrix (row-major), where the top-left 3x3 are axes
                     (columns are local axes in world coords) and the last column is center.
  - "obb_extents": [sx, sy, sz] full side lengths along the 3 local axes (NOT half-extents).

This script:
1) Finds the instance with label==floor_label (default "floor").
2) Extracts a global "up" axis from the floor box via:
   - "zclose": choose the floor axis most aligned with world_up (default [0,0,1])
   - "largest_face": choose the axis normal to the largest face (== axis with smallest extent)
3) For every NON-floor instance, adjusts its OBB so one face is perpendicular to this up axis,
   with minimal rotation (chooses which original axis becomes "up" and the best in-plane axis).
4) Writes the updated JSON to output_dir, preserving all other fields.
   The floor instance is copied unchanged.

Extent behavior (important):
- --extent_mode keep   (default): keep the original side lengths, only change the orientation (minimal change).
- --extent_mode enclose: recompute side lengths so the new OBB encloses the ORIGINAL 8 corners
                         (guarantees containment but can enlarge boxes).

Output corner order is preserved as:
  (-,-,-), (-,-,+), (-,+,-), (-,+,+), (+,-,-), (+,-,+), (+,+,-), (+,+,+)
in the local (x,y,z) frame.

Usage:
  python floor_align_bboxes.py \
      --input_dir  /path/to/jsons \
      --output_dir /path/to/output_jsons \
      --axis_method largest_face \
      --extent_mode keep \
      --floor_label floor
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


EPS = 1e-12


def _normalize(v: np.ndarray, eps: float = EPS) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return v.copy()
    return v / n


def _orthonormalize(R: np.ndarray) -> np.ndarray:
    """
    Robustly project a near-rotation matrix to SO(3) using SVD.
    Ensures det(R)=+1.
    """
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1.0
        Rn = U @ Vt
    return Rn


def _corner_signs_order() -> np.ndarray:
    """
    Corner order used by the provided JSONs:
      0 (-,-,-)
      1 (-,-,+)
      2 (-,+,-)
      3 (-,+,+)
      4 (+,-,-)
      5 (+,-,+)
      6 (+,+,-)
      7 (+,+,+)
    """
    signs = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                signs.append([sx, sy, sz])
    return np.array(signs, dtype=np.float64)


SIGNS_8 = _corner_signs_order()


def corners_from_obb(center: np.ndarray, R: np.ndarray, extents_full: np.ndarray) -> np.ndarray:
    """
    center: (3,)
    R: (3,3) with columns as local axes in world coords
    extents_full: (3,) full lengths along each local axis
    returns: (8,3) corners in the JSON order.
    """
    half = 0.5 * extents_full
    local = SIGNS_8 * half[None, :]
    world = center[None, :] + (R @ local.T).T
    return world


def parse_corners(instance: Dict[str, Any]) -> Optional[np.ndarray]:
    bb = instance.get("bounding_box", None)
    if not isinstance(bb, list) or len(bb) != 8:
        return None
    pts = []
    for p in bb:
        if not isinstance(p, dict):
            return None
        if not all(k in p for k in ("x", "y", "z")):
            return None
        pts.append([float(p["x"]), float(p["y"]), float(p["z"])])
    return np.asarray(pts, dtype=np.float64)


def parse_obb_transform_extents(instance: Dict[str, Any]) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Returns (center(3,), R(3,3), extents_full(3,)) if available, else None.
    """
    T = instance.get("obb_transform", None)
    ext = instance.get("obb_extents", None)
    if T is None or ext is None:
        return None
    try:
        T = np.asarray(T, dtype=np.float64)
        if T.shape != (4, 4):
            return None
        R = T[:3, :3]
        center = T[:3, 3]
        ext = np.asarray(ext, dtype=np.float64).reshape(3,)
        return center, R, ext
    except Exception:
        return None


def obb_from_corners_pca(corners: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fallback: estimate center, axes, extents from the 8 corners using PCA.

    PCA on the 8 corners of a box recovers its axes (up to sign/permutation) if corners are exact.
    """
    center = corners.mean(axis=0)
    X = corners - center[None, :]
    C = (X.T @ X) / float(X.shape[0])
    eigvals, eigvecs = np.linalg.eigh(C)  # ascending
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    R = _orthonormalize(eigvecs)
    local = (R.T @ X.T).T
    half = np.max(np.abs(local), axis=0)
    extents_full = 2.0 * half
    return center, R, extents_full


def choose_floor_up_axis(
    floor_instance: Dict[str, Any],
    axis_method: str,
    world_up: np.ndarray,
) -> np.ndarray:
    """
    axis_method:
      - "zclose": choose floor axis (among its 3 axes) closest to world_up
      - "largest_face": choose normal of largest face == axis with smallest extent
    """
    world_up = _normalize(world_up)

    parsed = parse_obb_transform_extents(floor_instance)
    if parsed is None:
        corners = parse_corners(floor_instance)
        if corners is None:
            raise ValueError("Floor instance has neither (obb_transform,obb_extents) nor 8-corner bounding_box.")
        _, R, ext = obb_from_corners_pca(corners)
    else:
        _, R, ext = parsed

    R = _orthonormalize(R)
    axes = [R[:, 0], R[:, 1], R[:, 2]]
    ext = np.asarray(ext, dtype=np.float64).reshape(3,)

    if axis_method == "zclose":
        idx = int(np.argmax([abs(float(np.dot(a, world_up))) for a in axes]))
    elif axis_method == "largest_face":
        idx = int(np.argmin(ext))  # smallest side length axis
    else:
        raise ValueError(f"Unknown axis_method: {axis_method}")

    a = axes[idx]
    if float(np.dot(a, world_up)) < 0.0:
        a = -a
    return _normalize(a)


def align_obb_to_up(
    center: np.ndarray,
    R: np.ndarray,
    corners_world: np.ndarray,
    up_axis: np.ndarray,
    extents_full: np.ndarray,
    extent_mode: str = "keep",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a new OBB orientation R_new whose z-axis is aligned to up_axis, with minimal rotation.

    extent_mode:
      - "keep":    keep original side lengths (only permute to match axis mapping)
      - "enclose": recompute side lengths to enclose the ORIGINAL 8 corners in the new frame
    """
    if extent_mode not in ("keep", "enclose"):
        raise ValueError(f"Unknown extent_mode: {extent_mode}")

    up_axis = _normalize(up_axis)
    R = _orthonormalize(R)

    axes = [R[:, 0], R[:, 1], R[:, 2]]
    extents_full = np.asarray(extents_full, dtype=np.float64).reshape(3,)

    dots = [abs(float(np.dot(a, up_axis))) for a in axes]
    idx_up = int(np.argmax(dots))

    # Choose sign for z (minimal change)
    z = up_axis if float(np.dot(axes[idx_up], up_axis)) >= 0.0 else -up_axis

    # --- yaw lock: 仅压平（去pitch/roll），保持绕up轴的方向 ---
    other = [i for i in range(3) if i != idx_up]
    X = corners_world - center[None, :]

    def proj_to_plane(v: np.ndarray, n: np.ndarray) -> np.ndarray:
        return v - float(np.dot(v, n)) * n

    # 选“更水平”的那个轴作为 heading 载体，避免额外绕up旋转
    p0 = proj_to_plane(axes[other[0]], z)
    p1 = proj_to_plane(axes[other[1]], z)

    if float(np.linalg.norm(p0)) >= float(np.linalg.norm(p1)):
        idx_x, idx_y = other[0], other[1]
        x = p0
    else:
        idx_x, idx_y = other[1], other[0]
        x = p1

    # 退化情况处理
    if float(np.linalg.norm(x)) < 1e-8:
        x = proj_to_plane(axes[idx_y], z)
        idx_x, idx_y = idx_y, idx_x

    if float(np.linalg.norm(x)) < 1e-8:
        world_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        x = proj_to_plane(world_x, z)
        if float(np.linalg.norm(x)) < 1e-8:
            world_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            x = proj_to_plane(world_y, z)
        x = _normalize(x)
        y = _normalize(np.cross(z, x))
    else:
        x = _normalize(x)
        # 防止180°翻转（尽量与原轴同向）
        if float(np.dot(x, axes[idx_x])) < 0.0:
            x = -x
        y = _normalize(np.cross(z, x))
        if float(np.dot(y, axes[idx_y])) < 0.0:
            x = -x
            y = -y

    R_new = _orthonormalize(np.column_stack([x, y, z]))

    if extent_mode == "keep":
        ext_new = np.array(
            [extents_full[idx_x], extents_full[idx_y], extents_full[idx_up]],
            dtype=np.float64
        )
    else:
        local_new = (R_new.T @ X.T).T
        half_new = np.max(np.abs(local_new), axis=0)
        ext_new = 2.0 * half_new

    return R_new, ext_new


def instance_to_json_corners(corners: np.ndarray) -> List[Dict[str, float]]:
    return [{"x": float(corners[i, 0]), "y": float(corners[i, 1]), "z": float(corners[i, 2])} for i in range(8)]


def instance_to_json_transform(center: np.ndarray, R: np.ndarray) -> List[List[float]]:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = center
    return [[float(v) for v in row] for row in T.tolist()]


def process_instances(
    instances: List[Dict[str, Any]],
    floor_label: str,
    axis_method: str,
    extent_mode: str,
    world_up: np.ndarray,
) -> List[Dict[str, Any]]:
    floor_candidates = [ins for ins in instances if ins.get("label") == floor_label]
    if len(floor_candidates) == 0:
        raise ValueError(f"No instance with label == '{floor_label}' found.")

    def volume(ins: Dict[str, Any]) -> float:
        parsed = parse_obb_transform_extents(ins)
        if parsed is None:
            corners = parse_corners(ins)
            if corners is None:
                return -1.0
            _, _, ext = obb_from_corners_pca(corners)
        else:
            _, _, ext = parsed
        ext = np.asarray(ext, dtype=np.float64).reshape(3,)
        return float(ext[0] * ext[1] * ext[2])

    floor_instance = sorted(floor_candidates, key=volume, reverse=True)[0]
    up_axis = choose_floor_up_axis(floor_instance, axis_method=axis_method, world_up=world_up)

    out_instances: List[Dict[str, Any]] = []
    for ins in instances:
        new_ins = json.loads(json.dumps(ins))  # deep copy

        if ins.get("label") == floor_label:
            out_instances.append(new_ins)  # keep floor unchanged
            continue

        corners = parse_corners(ins)
        parsed = parse_obb_transform_extents(ins)

        if corners is None and parsed is None:
            out_instances.append(new_ins)
            continue

        if corners is None and parsed is not None:
            center, R, ext = parsed
            corners = corners_from_obb(center, _orthonormalize(R), np.asarray(ext, dtype=np.float64))
        else:
            corners = corners

        if parsed is None:
            center, R, ext = obb_from_corners_pca(corners)
        else:
            center, R, ext = parsed
            R = _orthonormalize(R)

        R_new, ext_new = align_obb_to_up(
            center=center,
            R=R,
            corners_world=corners,
            up_axis=up_axis,
            extents_full=np.asarray(ext, dtype=np.float64),
            extent_mode=extent_mode,
        )

        corners_new = corners_from_obb(center, R_new, ext_new)

        new_ins["bounding_box"] = instance_to_json_corners(corners_new)
        new_ins["obb_transform"] = instance_to_json_transform(center, R_new)
        new_ins["obb_extents"] = [float(x) for x in ext_new.tolist()]

        out_instances.append(new_ins)

    return out_instances


def load_json_any(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def process_one_file(
    input_path: Path,
    output_path: Path,
    floor_label: str,
    axis_method: str,
    extent_mode: str,
    world_up: np.ndarray,
) -> None:
    obj = load_json_any(input_path)

    if isinstance(obj, list):
        new_obj = process_instances(
            obj,
            floor_label=floor_label,
            axis_method=axis_method,
            extent_mode=extent_mode,
            world_up=world_up,
        )
    elif isinstance(obj, dict):
        list_key = None
        for k in ("instances", "objects", "annotations", "data"):
            if k in obj and isinstance(obj[k], list):
                list_key = k
                break
        if list_key is None:
            raise ValueError(f"Unsupported JSON dict layout in {input_path} (no instance list key found).")
        obj_new = json.loads(json.dumps(obj))
        obj_new[list_key] = process_instances(
            obj[list_key],
            floor_label=floor_label,
            axis_method=axis_method,
            extent_mode=extent_mode,
            world_up=world_up,
        )
        new_obj = obj_new
    else:
        raise ValueError(f"Unsupported JSON top-level type in {input_path}: {type(obj)}")

    save_json(output_path, new_obj)


def iter_json_files(input_dir: Path) -> List[Path]:
    files: List[Path] = []
    for root, _, filenames in os.walk(str(input_dir)):
        for name in filenames:
            if name.lower().endswith(".json"):
                files.append(Path(root) / name)
    files.sort()
    return files


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str, required=True, help="Folder containing input .json files")
    p.add_argument("--output_dir", type=str, required=True, help="Folder to write converted .json files")
    p.add_argument("--floor_label", type=str, default="floor", help="Label name for the floor instance")
    p.add_argument(
        "--axis_method",
        type=str,
        default="largest_face",
        choices=["zclose", "largest_face"],
        help="How to extract the up-axis from the floor box",
    )
    p.add_argument(
        "--extent_mode",
        type=str,
        default="keep",
        choices=["keep", "enclose"],
        help="keep: keep sizes (minimal change). enclose: enlarge to enclose original corners.",
    )
    p.add_argument(
        "--world_up",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 1.0),
        help="World up direction used by axis_method=zclose (default: 0 0 1)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise SystemExit(f"input_dir does not exist: {input_dir}")

    files = iter_json_files(input_dir)
    if len(files) == 0:
        raise SystemExit(f"No .json files found under: {input_dir}")

    for in_path in files:
        rel = in_path.relative_to(input_dir)
        out_path = output_dir / rel
        try:
            process_one_file(
                input_path=in_path,
                output_path=out_path,
                floor_label=args.floor_label,
                axis_method=args.axis_method,
                extent_mode=args.extent_mode,
                world_up=np.asarray(args.world_up, dtype=np.float64),
            )
        except ValueError as e:
            # Common case: a JSON does not contain the expected floor instance.
            msg = str(e)
            if (
                f"label == '{args.floor_label}'" in msg
                or msg.startswith("No instance with label")
                or msg.startswith("Floor instance has neither")
            ):
                print(f"[SKIP] {in_path}: {msg}", file=sys.stderr)
                continue
            raise

        print(f"[OK] {in_path} -> {out_path}")


if __name__ == "__main__":
    main()
