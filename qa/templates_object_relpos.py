import textwrap
from pathlib import Path
from typing import Dict, Sequence, Tuple, TYPE_CHECKING, List, Any, Optional
import math
import random

import numpy as np

if TYPE_CHECKING:
    from qa_generation.generate_two_view_qa import FrameItem

# 8方向定义（与 mcq_direction_chatgpt.py 一致）
DIR8 = ["Front", "Front-Right", "Right", "Back-Right",
        "Back", "Back-Left", "Left", "Front-Left"]


def wrap180(deg: float) -> float:
    """Map angle to [-180, 180)."""
    x = (deg + 180.0) % 360.0 - 180.0
    return x


def classify_dir8(v_fwd: float, v_right: float, v_up: float,
                  up_dom_ratio: float = 1.2,
                  min_horiz: float = 1e-6):
    """分类为8个方向之一（与 mcq_direction_chatgpt.py 一致）。
    
    Returns:
        (label, idx, yaw_deg) 或 (None, None, None) 如果应该跳过
    """
    x, y, z = float(v_right), float(v_up), float(v_fwd)
    horiz = np.hypot(x, z)
    if horiz < min_horiz:
        return None, None, None  # skip
    if abs(y) >= up_dom_ratio * horiz:
        return None, None, None  # skip

    yaw = wrap180(np.degrees(np.arctan2(x, z)))  # right positive, front=0
    idx = int(np.round(yaw / 45.0)) % 8
    return DIR8[idx], idx, yaw


def make_mcq_random_3_of_7(true_idx: int, yaw_deg: float,
                           boundary_margin_deg: float = 3.0,
                           rng: random.Random = None):
    """
    Randomly pick 3 distractors from remaining 7,
    but if yaw is close to the sector boundary, forbid the adjacent sector
    on that boundary side.
    
    Args:
        rng: Random number generator. If None, uses global random.
    """
    if rng is None:
        rng = random.Random()

    # sector center (in degrees), keep in [-180, 180)
    center = wrap180(true_idx * 45.0)
    delta = wrap180(yaw_deg - center)  # [-180, 180)

    forbidden = set()

    # decide if near boundary (sector half-width = 22.5 deg)
    if abs(delta) >= 22.5 - boundary_margin_deg:
        # delta>0 -> near right boundary => forbid idx+1
        # delta<0 -> near left boundary  => forbid idx-1
        if delta > 0:
            forbidden.add((true_idx + 1) % 8)
        elif delta < 0:
            forbidden.add((true_idx - 1) % 8)
        # delta==0 won't happen in this branch

    candidates = [i for i in range(8) if i != true_idx and i not in forbidden]

    # if margin too large could reduce candidates; ensure we can sample 3
    if len(candidates) < 3:
        # fallback: relax forbidden
        candidates = [i for i in range(8) if i != true_idx]

    distractor_idx = rng.sample(candidates, 3)
    options_idx = distractor_idx + [true_idx]
    rng.shuffle(options_idx)

    letters = ["A", "B", "C", "D"]
    options = {letters[k]: DIR8[options_idx[k]] for k in range(4)}
    answer = [k for k, v in options.items() if v == DIR8[true_idx]][0]
    return options, answer, forbidden


def _bbox_center(bbox: Sequence[Sequence[float]]) -> np.ndarray:
    """计算3D bbox的中心点（世界坐标）。"""
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


def _build_local_frame(pA: np.ndarray, pB: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构建以A为原点、面向B的局部坐标系。
    
    返回: (forward, right, up) 三个单位向量，构成右手坐标系。
    forward: 从A指向B的方向
    right: 右手坐标系中的右方向
    up: 上方向
    """
    fwd_raw = pB - pA
    fwd_norm = np.linalg.norm(fwd_raw)
    if fwd_norm < 1e-6:
        raise ValueError("物体A和B的位置过于接近，无法确定方向")
    forward = fwd_raw / fwd_norm
    
    # 使用世界上方向 [0, 0, -1] 作为参考
    world_up = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    
    # 如果forward与world_up近乎共线，使用备选方向
    dot_fwd_up = np.abs(np.dot(forward, world_up))
    if dot_fwd_up > 0.99:  # 近乎共线
        # 使用 [0, 0, 1] 或 [1, 0, 0] 作为备选
        if abs(forward[2]) < 0.99:
            world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            world_up = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        dot_fwd_up = np.abs(np.dot(forward, world_up))
        if dot_fwd_up > 0.99:
            raise ValueError("无法构建稳定的局部坐标系：forward与备选up方向近乎共线")
    
    # 计算right: right = forward × world_up，然后归一化
    right = np.cross(forward, world_up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        raise ValueError("无法构建稳定的局部坐标系：right向量为零")
    right = right / right_norm
    
    # 重新计算up: up = right × forward（保证正交）
    up = np.cross(right, forward)
    up_norm = np.linalg.norm(up)
    if up_norm < 1e-6:
        raise ValueError("无法构建稳定的局部坐标系：up向量为零")
    up = up / up_norm
    
    return forward, right, up


def _direction_label(v_fwd: float, v_right: float, v_up: float,
                     up_dom_ratio: float = 1.2,
                     min_horiz: float = 1e-6) -> str:
    """根据局部坐标系中的分量生成8方向标签（使用 classify_dir8 逻辑）。
    
    Returns:
        方向标签字符串，如果无法分类则返回 None
    """
    label, idx, yaw = classify_dir8(v_fwd, v_right, v_up, up_dom_ratio, min_horiz)
    if label is None:
        return None
    return label


def build_object_relpos_entry(
    scene_id: str,
    intrinsics: Dict[str, float],
    frames: Sequence["FrameItem"],
    pair: Tuple[int, int, float],
    threshold: float,
    instA: Dict[str, Any],
    instB: Dict[str, Any],
    instC: Dict[str, Any],
    rng: random.Random = None,
) -> Optional[Dict[str, Any]]:
    """
    构造"站在物体A，面向物体B，问物体C的方位和距离"QA（MCQ格式）。
    
    Args:
        instA, instB, instC: bbox实例字典，需包含 "bounding_box" 字段（8个3D点）。
        rng: 随机数生成器，用于生成 MCQ 选项。如果为 None，使用全局 random。
    
    Returns:
        QA条目字典，如果无法构建（如局部坐标系无法建立或方向无法分类）则返回 None。
    """
    i, j, cov = pair
    fa, fb = frames[i], frames[j]
    
    # 获取三个物体的3D中心（世界坐标）
    centerA = _bbox_center(instA["bounding_box"])
    centerB = _bbox_center(instB["bounding_box"])
    centerC = _bbox_center(instC["bounding_box"])
    
    # 构建局部坐标系（以A为原点，面向B）
    try:
        forward, right, up = _build_local_frame(centerA, centerB)
    except ValueError as e:
        print(f"警告: 无法构建局部坐标系，跳过该三元组: {e}")
        return None
    
    # 计算C相对于A的向量
    v_AC = centerC - centerA
    dist = float(np.linalg.norm(v_AC))
    
    # 将v_AC投影到局部坐标系
    v_fwd = float(np.dot(v_AC, forward))
    v_right = float(np.dot(v_AC, right))
    v_up = float(np.dot(v_AC, up))
    
    # 使用 classify_dir8 进行分类
    direction_label, direction_idx, yaw_deg = classify_dir8(v_fwd, v_right, v_up)
    if direction_label is None:
        print(f"警告: 无法分类方向（水平分量太小或垂直分量占主导），跳过该三元组")
        return None
    
    # 生成 MCQ 选项
    if rng is None:
        rng = random.Random()
    mcq_options, mcq_answer, forbidden = make_mcq_random_3_of_7(direction_idx, yaw_deg, rng=rng)
    
    # 构建问题和答案
    labelA = instA.get("label", "object A")
    labelB = instB.get("label", "object B")
    labelC = instC.get("label", "object C")
    
    # MCQ 格式的问题
    options_text = "\n".join([f"{k}. {v}" for k, v in sorted(mcq_options.items())])
    question = textwrap.dedent(
        f"""
        You stand at {labelA} and face {labelB}.
        In which direction is {labelC} relative to you?\n{options_text}
        """
    ).strip()
    
    # 答案格式：MCQ 选项字母 + 方向标签 + 距离
    answer = f"{mcq_answer}. ({direction_label})"
    
    return {
        "scene_id": scene_id,
        "image_a": fa.file_name,
        "image_b": fb.file_name,
        "covisibility": cov,
        "threshold": threshold,
        "question_type": "object_relpos",
        "sub_question_type": "relpos_A_facing_B",
        "question": question,
        "answer": answer,
        "objects": {
            "A": {
                "ins_id": str(instA.get("ins_id", "")),
                "label": labelA,
                "center_world": centerA.tolist(),
            },
            "B": {
                "ins_id": str(instB.get("ins_id", "")),
                "label": labelB,
                "center_world": centerB.tolist(),
            },
            "C": {
                "ins_id": str(instC.get("ins_id", "")),
                "label": labelC,
                "center_world": centerC.tolist(),
            },
        },
        "relpos_details": {
            "distance_m": dist,
            "direction_label": direction_label,
            "direction_idx": direction_idx,
            "yaw_deg": yaw_deg,
            "local_coords": {
                "forward": v_fwd,
                "right": v_right,
                "up": v_up,
            },
            "local_frame": {
                "forward": forward.tolist(),
                "right": right.tolist(),
                "up": up.tolist(),
            },
            "mcq": {
                "options": mcq_options,
                "answer": mcq_answer,
                "forbidden_indices": list(forbidden),
            },
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



