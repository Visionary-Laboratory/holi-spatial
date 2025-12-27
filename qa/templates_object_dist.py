import json
import textwrap
from pathlib import Path
from typing import Dict, Sequence, Tuple, TYPE_CHECKING, List, Any, Optional

import numpy as np

if TYPE_CHECKING:
    from qa_generation.generate_two_view_qa import FrameItem


def _bbox_center(bbox: Sequence[Sequence[float]]) -> np.ndarray:
    if len(bbox) == 0:
        raise ValueError("bounding_box 为空")
    first = bbox[0]
    if isinstance(first, dict) and {"x", "y", "z"} <= set(first.keys()):
        arr = np.array([[p["x"], p["y"], p["z"]] for p in bbox], dtype=np.float32)
    else:
        arr = np.asarray(bbox, dtype=np.float32)
    if arr.shape != (8, 3):
        raise ValueError(f"bounding_box 期望形状 (8,3)，当前 {arr.shape}")
    return arr.mean(axis=0)


def build_object_dist_entry(
    scene_id: str,
    intrinsics: Dict[str, float],
    frames: Sequence["FrameItem"],
    pair: Tuple[int, int, float],
    threshold: float,
    inst1: Dict[str, Any],
    inst2: Dict[str, Any],
):
    """
    构造"给定 A 视角中的物体 1 和 B 视角中的物体 2，问它们之间的 3D 距离是多少"的 QA。
    """
    i, j, cov = pair
    fa, fb = frames[i], frames[j]
    image_a_stem = Path(fa.file_name).stem
    image_b_stem = Path(fb.file_name).stem

    # 查找mask路径
    mask_map1: Dict[str, str] = {}
    for p_str in inst1.get("images", []):
        p = Path(p_str)
        mask_map1[p.parent.name] = str(p)
    
    mask_map2: Dict[str, str] = {}
    for p_str in inst2.get("images", []):
        p = Path(p_str)
        mask_map2[p.parent.name] = str(p)
    
    mask1_path = mask_map1.get(image_a_stem)
    mask2_path = mask_map2.get(image_b_stem)
    
    if mask1_path is None or mask2_path is None:
        raise ValueError(f"无法找到mask路径: inst1在{image_a_stem}或inst2在{image_b_stem}中不存在")

    center1 = _bbox_center(inst1["bounding_box"])
    center2 = _bbox_center(inst2["bounding_box"])
    
    dist_3d = float(np.linalg.norm(center1 - center2))

    label1 = inst1.get("label") or "object 1"
    label2 = inst2.get("label") or "object 2"

    question = textwrap.dedent(
        f"""
        In image A, the white mask highlights "{label1}".
        In image B, the white mask highlights "{label2}".
        Estimate the 3D metric distance (in meters) between these two physical objects.
        Return only one number in meters (e.g., 1.2).
        Output format: <answer>NUMBER</answer>.
        """
    ).strip()
    
    answer = f"{dist_3d:.2f} m"

    return {
        "scene_id": scene_id,
        "image_a": fa.file_name,
        "image_a_mask": mask1_path,
        "image_b": fb.file_name,
        "image_b_mask": mask2_path,
        "covisibility": cov,
        "threshold": threshold,
        "question_type": "object_distance",
        "sub_question_type": "inter_object_distance",
        "question": question,
        "answer": answer,
        "distance_gt": dist_3d,
        "object_1": {
            "ins_id": inst1.get("ins_id"),
            "label": label1,
            "center": center1.tolist(),
        },
        "object_2": {
            "ins_id": inst2.get("ins_id"),
            "label": label2,
            "center": center2.tolist(),
        },
        "camera_a": {
            "intrinsics": intrinsics,
            "transform_matrix": fa.transform_matrix,
        },
        "camera_b": {
            "intrinsics": intrinsics,
            "transform_matrix": fb.transform_matrix,
        },
    }

