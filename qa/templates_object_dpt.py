import json
import textwrap
from pathlib import Path
from typing import Dict, Sequence, Tuple, TYPE_CHECKING, List, Any, Optional, Iterable, Set

import numpy as np

if TYPE_CHECKING:
    # Avoid runtime circular import; only for type hints.
    from qa_generation.generate_two_view_qa import FrameItem


def load_bbox_items(json_path: Path) -> List[Dict[str, Any]]:
    if not json_path.exists():
        raise FileNotFoundError(f"未找到 3D bbox json: {json_path}")
    try:
        data = json.loads(json_path.read_text())
    except Exception as exc:  # pragma: no cover - 简单容错
        raise ValueError(f"解析 bbox json 失败: {json_path}") from exc
    if not isinstance(data, list):
        raise ValueError(f"bbox json 格式需为 list，当前为 {type(data)}")
    return data


def _bbox_center(bbox: Sequence[Sequence[float]]) -> np.ndarray:
    if len(bbox) == 0:
        raise ValueError("bounding_box 为空")
    # 兼容 [{"x":..., "y":..., "z":...}, ...] 或 [[x,y,z], ...]
    first = bbox[0]
    if isinstance(first, dict) and {"x", "y", "z"} <= set(first.keys()):
        arr = np.array([[p["x"], p["y"], p["z"]] for p in bbox], dtype=np.float32)
    else:
        arr = np.asarray(bbox, dtype=np.float32)
    if arr.shape != (8, 3):
        raise ValueError(f"bounding_box 期望形状 (8,3)，当前 {arr.shape}")
    return arr.mean(axis=0)


def _world_to_cam(point_world: np.ndarray, c2w: Sequence[Sequence[float]]) -> np.ndarray:
    mat = np.asarray(c2w, dtype=np.float32)
    if mat.shape != (4, 4):
        raise ValueError(f"transform_matrix 期望 4x4，当前 {mat.shape}")
    w2c = np.linalg.inv(mat)
    homo = np.concatenate([point_world, [1.0]], axis=0)
    cam = w2c @ homo
    return cam[:3]


def _project_cam(point_cam: np.ndarray, intrinsics: Dict[str, float]) -> Optional[Tuple[float, float]]:
    z = float(point_cam[2])
    if z <= 1e-6:
        return None
    fx, fy = intrinsics["fl_x"], intrinsics["fl_y"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    u = fx * point_cam[0] / z + cx
    v = fy * point_cam[1] / z + cy
    return float(u), float(v)


def build_object_dpt_entry(
    scene_id: str,
    intrinsics: Dict[str, float],
    frames: Sequence["FrameItem"],
    pair: Tuple[int, int, float],
    threshold: float,
    json_path: Path,
    *,
    bbox_items: Optional[List[Dict[str, Any]]] = None,
    target_ins_id: Optional[str] = None,
    skip_labels: Optional[Iterable[str]] = None,
):
    """
    基于 bbox json 与 mask 路径构造“给定 A 视角的 mask，问物体在 B 视角距离多少”QA。

    json_path: 3D bbox 结果文件，单元素格式示例：
      {
        "ins_id": "1",
        "label": "bed",
        "bounding_box": [[x,y,z] * 8],
        "images": ["/root/sam_masks_debug/<scene>/<image>/bed.png", ...]
      }
    """
    i, j, cov = pair
    fa, fb = frames[i], frames[j]
    image_a_stem = Path(fa.file_name).stem
    image_b_stem = Path(fb.file_name).stem

    if bbox_items is None:
        bbox_items = load_bbox_items(json_path)
    ignore_labels: Set[str] = {lbl.lower() for lbl in skip_labels} if skip_labels else set()

    shared_instances = []
    for inst in bbox_items:
        label = str(inst.get("label", ""))
        if label.lower() in ignore_labels:
            continue
        if target_ins_id is not None and str(inst.get("ins_id", "")) != str(target_ins_id):
            continue
        images = inst.get("images", [])
        mask_map: Dict[str, str] = {}
        for p_str in images:
            p = Path(p_str)
            mask_map[p.parent.name] = str(p)  # 目录名就是原图 stem
        if image_a_stem not in mask_map or image_b_stem not in mask_map:
            continue

        center_world = _bbox_center(inst["bounding_box"])
        center_cam_b = _world_to_cam(center_world, fb.transform_matrix)
        dist_cam_b = float(np.linalg.norm(center_cam_b))
        proj_uv = _project_cam(center_cam_b, intrinsics)

        shared_instances.append(
            {
                "ins_id": str(inst.get("ins_id", "")),
                "label": inst.get("label", ""),
                "mask_path_a": mask_map[image_a_stem],
                "mask_path_b": mask_map[image_b_stem],
                "bounding_box": inst.get("bounding_box"),
                "center_world": center_world.tolist(),
                "center_cam_b": center_cam_b.tolist(),
                "center_proj_b": proj_uv,
                "distance_cam_b": dist_cam_b,
            }
        )

    if target_ins_id is not None:
        target = next((x for x in shared_instances if x["ins_id"] == str(target_ins_id)), shared_instances[0])
    else:
        target = shared_instances[0]
    label = target["label"] or "object"
    mask_a = target["mask_path_a"]

    question = textwrap.dedent(
        f"""
        In image A, the white mask highlights the target object "{label}".
        Locate the same physical pipe in image B.
        Estimate the 3D metric distance (in meters)
        from the camera position of image B (camera center)
        to the "{label}" (to the object surface/center point).
        This is NOT pixel distance to the image center.
        Return only one number in meters (e.g., 0.7).
        Output format: <answer>NUMBER</answer>.
        """
    ).strip()
    answer = f"{target['distance_cam_b']:.1f} m"

    return {
        "scene_id": scene_id,
        "image_a": fa.file_name,
        "image_a_mask": target["mask_path_a"],
        "image_b": fb.file_name,
        "covisibility": cov,
        "threshold": threshold,
        "question_type": "object_distance",
        "sub_question_type": "bbox_center_distance",
        "question": question,
        "answer": answer,
        "selected_ins_id": target["ins_id"],
        "objects": shared_instances,
        "camera_a": {
            "intrinsics": intrinsics,
            "transform_matrix": fa.transform_matrix,
        },
        "camera_b": {
            "intrinsics": intrinsics,
            "transform_matrix": fb.transform_matrix,
        },
    }

