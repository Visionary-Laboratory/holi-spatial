import textwrap
import json
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

# 标记方式
MARKER_TYPES = ["point_marker", "white_mask", "mask_covering_image", "2d_bbox", "language_description"]

# 物体颜色映射
OBJECT_COLORS = {
    "A": "red",
    "B": "green", 
    "C": "blue"
}


def _infer_mask_index_path(scene_id: str, images: List[str]) -> Optional[Path]:
    """从 images 路径列表中推断 mask_index.json 的路径。
    
    Args:
        scene_id: 场景 ID
        images: mask 图像路径列表（来自 bbox_json 的 images 字段）
    
    Returns:
        mask_index.json 的路径，如果无法推断或文件不存在则返回 None
    """
    if not images:
        return None
    
    # 取第一个图像路径进行分析
    first_image_path = Path(images[0])
    
    # 如果是相对路径，需要解析为绝对路径
    if not first_image_path.is_absolute():
        try:
            first_image_path = first_image_path.resolve()
        except Exception:
            return None
    
    # 在路径中查找包含 scene_id 的目录
    # 通常格式是：.../mask_folder/scene_id/.../mask.png
    parts = first_image_path.parts
    scene_id_idx = None
    for i, part in enumerate(parts):
        if part == scene_id:
            scene_id_idx = i
            break
    
    if scene_id_idx is None:
        return None
    
    # scene_id 所在的目录就是 mask 根目录
    # mask_index.json 应该在这个目录下
    mask_root = Path(*parts[:scene_id_idx + 1])
    mask_index_path = mask_root / "mask_index.json"
    
    return mask_index_path if mask_index_path.exists() else None


def _load_mask_index(scene_id: str, images: List[str]) -> Optional[Dict]:
    """从 images 路径列表中推断并加载 mask_index.json 文件。
    
    Args:
        scene_id: 场景 ID
        images: mask 图像路径列表（来自 bbox_json 的 images 字段）
    
    Returns:
        mask_index.json 的内容，如果文件不存在或无法推断则返回 None
    """
    mask_index_path = _infer_mask_index_path(scene_id, images)
    if mask_index_path is None:
        return None
    
    try:
        return json.loads(mask_index_path.read_text())
    except Exception as e:
        print(f"警告: 无法加载 mask_index.json ({mask_index_path}): {e}")
        return None


def _find_mask_rle_by_path(mask_index: Dict, mask_path: str) -> Optional[Dict]:
    """通过 mask_path 从 mask_index.json 中查找对应的 mask RLE 数据。
    
    Args:
        mask_index: mask_index.json 的内容
        mask_path: mask 路径（来自 instance 的 images 数组）
    
    Returns:
        mask RLE 数据（包含 size 和 counts），如果找不到则返回 None
    """
    if not mask_index or "items" not in mask_index or not mask_path:
        return None
    
    # 标准化路径（处理可能的路径差异）
    mask_path_normalized = str(Path(mask_path))
    
    for item in mask_index["items"]:
        item_mask_path = item.get("mask_path")
        if item_mask_path:
            item_mask_path_normalized = str(Path(item_mask_path))
            # 匹配 mask_path
            if item_mask_path_normalized == mask_path_normalized:
                return item.get("mask_rle")
    return None


def _find_mask_rle_from_instance_encodings(inst: Dict[str, Any], mask_path: Optional[str]) -> Optional[Dict]:
    """
    fallback: 当 mask_index.json 里缺失 mask_rle 时，从 bbox json 的 instance.mask_encodings 获取。
    对齐规则：优先 images 与 mask_encodings 同长度且按 index 对齐；否则尝试 basename 匹配；最后仅当 encodings 只有 1 个时使用它。
    """
    if not mask_path:
        return None
    encs = inst.get("mask_encodings")
    imgs = inst.get("images")
    if not isinstance(encs, list) or len(encs) == 0:
        return None

    def _is_rle(d) -> bool:
        return isinstance(d, dict) and "size" in d and "counts" in d

    if isinstance(imgs, list) and len(imgs) == len(encs):
        idx = None
        try:
            idx = imgs.index(mask_path)
        except ValueError:
            base = Path(mask_path).name
            for i, p in enumerate(imgs):
                if Path(p).name == base:
                    idx = i
                    break
        if idx is not None and 0 <= idx < len(encs) and _is_rle(encs[idx]):
            return encs[idx]

    if len(encs) == 1 and _is_rle(encs[0]):
        return encs[0]

    return None


def _world_to_cam(point_world: np.ndarray, c2w: Sequence[Sequence[float]]) -> np.ndarray:
    """将世界坐标转换为相机坐标"""
    mat = np.asarray(c2w, dtype=np.float32)
    if mat.shape != (4, 4):
        raise ValueError(f"transform_matrix 期望 4x4，当前 {mat.shape}")
    w2c = np.linalg.inv(mat)
    homo = np.concatenate([point_world, [1.0]], axis=0)
    cam = w2c @ homo
    return cam[:3]


def _project_cam(point_cam: np.ndarray, intrinsics: Dict[str, float]) -> Optional[Tuple[float, float]]:
    """将相机坐标投影到图像坐标"""
    z = float(point_cam[2])
    if z <= 1e-6:
        return None
    fx, fy = intrinsics["fl_x"], intrinsics["fl_y"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    u = fx * point_cam[0] / z + cx
    v = fy * point_cam[1] / z + cy
    return float(u), float(v)


def _get_2d_bbox_from_rle(mask_rle: Dict) -> Optional[Tuple[float, float, float, float]]:
    """从 RLE mask 计算 2D 边界框 (x_min, y_min, x_max, y_max)。
    
    Args:
        mask_rle: RLE 格式的 mask 字典，包含 'size' 和 'counts'
    
    Returns:
        2D 边界框 (x_min, y_min, x_max, y_max)，如果解码失败则返回 None
    """
    try:
        from pycocotools import mask as mask_utils
    except ImportError:
        return None
    
    try:
        if isinstance(mask_rle, dict) and "size" in mask_rle and "counts" in mask_rle:
            # 解码 RLE mask
            mask_array = mask_utils.decode(mask_rle)
            # 找到所有非零像素的位置
            rows, cols = np.where(mask_array > 0)
            if len(rows) == 0:
                return None
            # 计算边界框
            x_min, y_min = float(cols.min()), float(rows.min())
            x_max, y_max = float(cols.max()), float(rows.max())
            return (x_min, y_min, x_max, y_max)
    except Exception:
        pass
    return None


def _get_2d_center_from_rle(mask_rle: Dict) -> Optional[Tuple[float, float]]:
    """从 RLE mask 计算 2D 中心点。
    
    Args:
        mask_rle: RLE 格式的 mask 字典，包含 'size' 和 'counts'
    
    Returns:
        2D 中心点 (x, y)，如果解码失败则返回 None
    """
    try:
        from pycocotools import mask as mask_utils
    except ImportError:
        return None
    
    try:
        if isinstance(mask_rle, dict) and "size" in mask_rle and "counts" in mask_rle:
            # 解码 RLE mask
            mask_array = mask_utils.decode(mask_rle)
            # 找到所有非零像素的位置
            rows, cols = np.where(mask_array > 0)
            if len(rows) == 0:
                return None
            # 计算中心点（所有非零像素的平均位置）
            x_center = float(cols.mean())
            y_center = float(rows.mean())
            return (x_center, y_center)
    except Exception:
        pass
    return None


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


def _bbox_aabb(bbox: Sequence[Sequence[float]]) -> Tuple[np.ndarray, np.ndarray]:
    """计算3D bbox的轴对齐包围盒（AABB）。
    
    Returns:
        (min_corner, max_corner) 两个3D点，表示AABB的最小和最大角点
    """
    if len(bbox) == 0:
        raise ValueError("bounding_box 为空")
    first = bbox[0]
    if isinstance(first, dict) and {"x", "y", "z"} <= set(first.keys()):
        arr = np.array([[p["x"], p["y"], p["z"]] for p in bbox], dtype=np.float32)
    else:
        arr = np.asarray(bbox, dtype=np.float32)
    if arr.shape != (8, 3):
        raise ValueError(f"bounding_box 期望形状 (8,3)，当前 {arr.shape}")
    min_corner = arr.min(axis=0)
    max_corner = arr.max(axis=0)
    return min_corner, max_corner


def _bboxes_intersect(bbox1: Sequence[Sequence[float]], bbox2: Sequence[Sequence[float]], 
                      min_separation: float = 0.0) -> bool:
    """检查两个3D bbox是否有交集或距离过近。
    
    Args:
        bbox1: 第一个bbox（8个3D点）
        bbox2: 第二个bbox（8个3D点）
        min_separation: 最小分离距离（米），如果两个bbox的最近距离小于此值，则认为太近
    
    Returns:
        如果两个bbox有交集或距离过近，返回True；否则返回False
    """
    min1, max1 = _bbox_aabb(bbox1)
    min2, max2 = _bbox_aabb(bbox2)
    
    # 检查AABB是否相交（考虑最小分离距离）
    # 如果 min_separation > 0，相当于将每个bbox向外扩展 min_separation/2
    if min_separation > 0:
        expansion = min_separation / 2.0
        min1 = min1 - expansion
        max1 = max1 + expansion
        min2 = min2 - expansion
        max2 = max2 + expansion
    
    # 检查AABB是否相交
    # 如果在一个轴上不相交，则两个bbox不相交
    for i in range(3):
        if max1[i] < min2[i] or max2[i] < min1[i]:
            return False
    
    return True


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


def _dir4_label_from_plane(v_right: float, v_forward: float, min_val: float = 1e-6) -> Optional[str]:
    if abs(v_right) < min_val and abs(v_forward) < min_val:
        return None
    if abs(v_right) >= abs(v_forward):
        return "East" if v_right > 0 else "West"
    return "North" if v_forward > 0 else "South"


def _dir8_label_from_plane(
    v_right: float,
    v_forward: float,
    ratio_threshold: float = 0.5774,
    min_val: float = 1e-6,
) -> Optional[str]:
    abs_x = abs(v_right)
    abs_z = abs(v_forward)
    top_mag = max(abs_x, abs_z)
    if top_mag < min_val:
        return None

    ns = "North" if v_forward > 0 else "South"
    ew = "East" if v_right > 0 else "West"

    if abs_x >= abs_z:
        include_second = abs_z >= ratio_threshold * top_mag
        dominant = ew
    else:
        include_second = abs_x >= ratio_threshold * top_mag
        dominant = ns

    if include_second:
        return f"{ns}-{ew}"
    return dominant


def _build_dir4_mcq_options(direction: str, rng: random.Random) -> Tuple[Dict[str, str], str]:
    all_dirs = ["North", "East", "South", "West"]
    wrong = [d for d in all_dirs if d != direction]
    rng.shuffle(wrong)
    options_list = wrong[:3] + [direction]
    rng.shuffle(options_list)
    letters = ["A", "B", "C", "D"]
    options = {letters[i]: options_list[i] for i in range(4)}
    answer = next(k for k, v in options.items() if v == direction)
    return options, answer


def _build_dir8_mcq_options(direction: str, rng: random.Random) -> Tuple[Dict[str, str], str]:
    all_dirs = [
        "North",
        "North-East",
        "East",
        "South-East",
        "South",
        "South-West",
        "West",
        "North-West",
    ]
    wrong = [d for d in all_dirs if d != direction]
    rng.shuffle(wrong)
    options_list = wrong[:3] + [direction]
    rng.shuffle(options_list)
    letters = ["A", "B", "C", "D"]
    options = {letters[i]: options_list[i] for i in range(4)}
    answer = next(k for k, v in options.items() if v == direction)
    return options, answer


def build_object_proxy_dir_entry(
    scene_id: str,
    intrinsics: Dict[str, float],
    frames: Sequence["FrameItem"],
    pair: Tuple[int, int, float],
    threshold: float,
    instA: Dict[str, Any],
    instB: Dict[str, Any],
    instC: Dict[str, Any],
    rng: random.Random = None,
    marker_types: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    构造"给定 obj1 相对 obj2 的方向，问 obj3 相对 obj2 的方向"QA（MCQ）。
    使用代理坐标系（由 obj1->obj2 与 assumed_dir 确定），仅输出 N/E/S/W。
    """
    if rng is None:
        rng = random.Random()

    i, j, cov = pair
    fa, fb = frames[i], frames[j]
    image_a_name = Path(fa.file_name).name
    image_b_name = Path(fb.file_name).name

    images_for_inference = instA.get("images", [])
    if not images_for_inference:
        images_for_inference = instB.get("images", [])
    if not images_for_inference:
        images_for_inference = instC.get("images", [])

    mask_index = _load_mask_index(scene_id, images_for_inference)
    mask_index_path = _infer_mask_index_path(scene_id, images_for_inference)
    mask_index_path_str = str(mask_index_path) if mask_index_path else None

    def _find_mask_rles(inst: Dict[str, Any], image_a_name: str, image_b_name: str) -> Tuple[Optional[Dict], Optional[Dict]]:
        images = inst.get("images", [])
        image_a_stem = Path(image_a_name).stem
        image_b_stem = Path(image_b_name).stem

        mask_path_a = None
        mask_path_b = None
        for img_path_str in images:
            img_path = Path(img_path_str)
            if img_path.parent.name == image_a_stem:
                mask_path_a = img_path_str
            if img_path.parent.name == image_b_stem:
                mask_path_b = img_path_str

        mask_rle_a = _find_mask_rle_by_path(mask_index, mask_path_a) if (mask_index and mask_path_a) else None
        mask_rle_b = _find_mask_rle_by_path(mask_index, mask_path_b) if (mask_index and mask_path_b) else None

        if mask_rle_a is None:
            mask_rle_a = _find_mask_rle_from_instance_encodings(inst, mask_path_a)
        if mask_rle_b is None:
            mask_rle_b = _find_mask_rle_from_instance_encodings(inst, mask_path_b)

        return mask_rle_a, mask_rle_b

    maskA_a, maskA_b = _find_mask_rles(instA, image_a_name, image_b_name)
    maskB_a, maskB_b = _find_mask_rles(instB, image_a_name, image_b_name)
    maskC_a, maskC_b = _find_mask_rles(instC, image_a_name, image_b_name)

    # 检查三个物体之间是否有bbox交集或距离过近
    bboxA = instA["bounding_box"]
    bboxB = instB["bounding_box"]
    bboxC = instC["bounding_box"]

    if _bboxes_intersect(bboxA, bboxB, min_separation=0.0):
        print("警告: 物体A和B的bbox有交集，跳过该三元组")
        return None
    if _bboxes_intersect(bboxA, bboxC, min_separation=0.0):
        print("警告: 物体A和C的bbox有交集，跳过该三元组")
        return None
    if _bboxes_intersect(bboxB, bboxC, min_separation=0.0):
        print("警告: 物体B和C的bbox有交集，跳过该三元组")
        return None

    centerA = _bbox_center(bboxA)
    centerB = _bbox_center(bboxB)
    centerC = _bbox_center(bboxC)

    v21 = centerA - centerB
    v23 = centerC - centerB

    v21_xy = np.array([v21[0], v21[1]], dtype=np.float32)
    if np.linalg.norm(v21_xy) < 1e-6:
        return None

    assumed_dir = rng.choice(["north", "south", "east", "west"])
    dir_to_angle = {
        "north": 0.0,
        "east": math.pi / 2.0,
        "south": math.pi,
        "west": -math.pi / 2.0,
    }
    angle_world = math.atan2(float(v21_xy[1]), float(v21_xy[0]))
    delta = angle_world - dir_to_angle[assumed_dir]

    forward = np.array([math.cos(delta), math.sin(delta), 0.0], dtype=np.float32)
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-6:
        return None
    forward = forward / forward_norm
    world_up = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        return None
    right = right / right_norm

    v_fwd = float(np.dot(v23, forward))
    v_right = float(np.dot(v23, right))
    direction_label = _dir8_label_from_plane(v_right, v_fwd)
    if direction_label is None:
        return None

    mcq_options, mcq_answer = _build_dir8_mcq_options(direction_label, rng)

    candidate_marker_types = marker_types or MARKER_TYPES
    if not candidate_marker_types:
        raise ValueError("marker_types 为空，无法选择标记方式")
    marker_type = rng.choice(candidate_marker_types)

    labelA = instA.get("label", "object A")
    labelB = instB.get("label", "object B")
    labelC = instC.get("label", "object C")

    marker_descriptions = []
    marker_data = {}

    if marker_type == "point_marker":
        for obj_label, inst, mask_a, mask_b in [
            ("A", instA, maskA_a, maskA_b),
            ("B", instB, maskB_a, maskB_b),
            ("C", instC, maskC_a, maskC_b),
        ]:
            label = inst.get("label", f"object {obj_label}")
            color = OBJECT_COLORS.get(obj_label, "red")
            point_a = _get_2d_center_from_rle(mask_a) if mask_a else None
            point_b = _get_2d_center_from_rle(mask_b) if mask_b else None
            if point_a:
                marker_descriptions.append(f"In image A, a {color} point marker indicates {label}.")
                marker_data[f"image_a_point_{obj_label}"] = point_a
                marker_data[f"image_a_point_{obj_label}_color"] = color
            if point_b:
                marker_descriptions.append(f"In image B, a {color} point marker indicates {label}.")
                marker_data[f"image_b_point_{obj_label}"] = point_b
                marker_data[f"image_b_point_{obj_label}_color"] = color

    elif marker_type == "white_mask":
        for obj_label, inst, mask_a, mask_b in [
            ("A", instA, maskA_a, maskA_b),
            ("B", instB, maskB_a, maskB_b),
            ("C", instC, maskC_a, maskC_b),
        ]:
            label = inst.get("label", f"object {obj_label}")
            if mask_a:
                marker_descriptions.append(f"In image A, a separate white mask image highlights {label}.")
            if mask_b:
                marker_descriptions.append(f"In image B, a separate white mask image highlights {label}.")

    elif marker_type == "mask_covering_image":
        for obj_label, inst, mask_a, mask_b in [
            ("A", instA, maskA_a, maskA_b),
            ("B", instB, maskB_a, maskB_b),
            ("C", instC, maskC_a, maskC_b),
        ]:
            label = inst.get("label", f"object {obj_label}")
            color = OBJECT_COLORS.get(obj_label, "red")
            if mask_a:
                marker_descriptions.append(f"In image A, a {color} mask overlay indicates {label}.")
                marker_data[f"image_a_mask_rle_{obj_label}"] = mask_a
                marker_data[f"image_a_mask_rle_{obj_label}_color"] = color
            if mask_b:
                marker_descriptions.append(f"In image B, a {color} mask overlay indicates {label}.")
                marker_data[f"image_b_mask_rle_{obj_label}"] = mask_b
                marker_data[f"image_b_mask_rle_{obj_label}_color"] = color

    elif marker_type == "2d_bbox":
        for obj_label, inst, mask_a, mask_b in [
            ("A", instA, maskA_a, maskA_b),
            ("B", instB, maskB_a, maskB_b),
            ("C", instC, maskC_a, maskC_b),
        ]:
            label = inst.get("label", f"object {obj_label}")
            color = OBJECT_COLORS.get(obj_label, "red")
            bbox_a = _get_2d_bbox_from_rle(mask_a) if mask_a else None
            bbox_b = _get_2d_bbox_from_rle(mask_b) if mask_b else None
            if bbox_a:
                marker_descriptions.append(f"In image A, a {color} 2D bounding box indicates {label}.")
                marker_data[f"image_a_bbox_{obj_label}"] = bbox_a
                marker_data[f"image_a_bbox_{obj_label}_color"] = color
            if bbox_b:
                marker_descriptions.append(f"In image B, a {color} 2D bounding box indicates {label}.")
                marker_data[f"image_b_bbox_{obj_label}"] = bbox_b
                marker_data[f"image_b_bbox_{obj_label}_color"] = color

    elif marker_type == "language_description":
        masks_by_obj = {
            "A": (maskA_a, maskA_b),
            "B": (maskB_a, maskB_b),
            "C": (maskC_a, maskC_b),
        }
        for obj_label, inst in [("A", instA), ("B", instB), ("C", instC)]:
            label = inst.get("label", f"object {obj_label}")
            description = inst.get("description")
            if not description:
                raise ValueError(f"language_description 缺少描述: ins_id={inst.get('ins_id')}, label={label}")
            mask_a, mask_b = masks_by_obj.get(obj_label, (None, None))
            if not mask_a and not mask_b:
                raise ValueError(f"language_description 物体未出现在 image A/B 中: ins_id={inst.get('ins_id')}, label={label}")
            if mask_a:
                marker_descriptions.append(f"In image A, the object described as \"{description}\" is the {label}.")
            elif mask_b:
                if str(inst.get("entity_type", "")).lower() == "region":
                    marker_descriptions.append(f"The {label} is in Image B.")
                else:
                    marker_descriptions.append(f"In image B, the object described as \"{description}\" is the {label}.")
            marker_data[f"description_{obj_label}"] = description

    marker_text = "\n".join(marker_descriptions) if marker_descriptions else ""
    options_text = "\n".join([f"{k}. {v}" for k, v in sorted(mcq_options.items())])

    question_parts = []
    if marker_text:
        question_parts.append(marker_text)
    question_parts.append(f"The direction of {labelA} relative to {labelB} is {assumed_dir}.")
    question_parts.append(f"What is the direction of {labelC} relative to {labelB}?")
    question_parts.append(options_text)
    question = "\n".join(question_parts).strip()

    answer = f"{mcq_answer}. {direction_label}"

    return {
        "scene_id": scene_id,
        "image_a": fa.file_name,
        "image_b": fb.file_name,
        "covisibility": cov,
        "threshold": threshold,
        "question_type": "object_proxy_direction",
        "sub_question_type": "proxy_object_direction",
        "question": question,
        "answer": answer,
        "assumed_dir": assumed_dir,
        "marker_type": marker_type,
        "marker_data": marker_data,
        "mask_index_path": mask_index_path_str,
        "options": mcq_options,
        **({
            "image_a_mask_rle_A": maskA_a,
            "image_a_mask_rle_B": maskB_a,
            "image_a_mask_rle_C": maskC_a,
            "image_b_mask_rle_A": maskA_b,
            "image_b_mask_rle_B": maskB_b,
            "image_b_mask_rle_C": maskC_b,
        } if marker_type == "white_mask" else {}),
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
        "direction_details": {
            "label": direction_label,
            "proxy_forward": forward.tolist(),
            "proxy_right": right.tolist(),
            "proxy_up": world_up.tolist(),
            "v21_world": v21.tolist(),
            "v23_world": v23.tolist(),
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


def build_region_proxy_dir_entry(
    scene_id: str,
    intrinsics: Dict[str, float],
    frames: Sequence["FrameItem"],
    pair: Tuple[int, int, float],
    threshold: float,
    instA: Dict[str, Any],
    instB: Dict[str, Any],
    regionC: Dict[str, Any],
    rng: random.Random = None,
    marker_types: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    构造"给定 obj1 相对 obj2 的方向，问 Region 3 相对 obj2 的方向"QA（MCQ）。
    复用 object proxy 主逻辑，并在输出语义上将 C 明确为 region。
    """
    region_inst = dict(regionC)
    region_label = region_inst.get("label", "region C")
    # language_description 场景下，region 缺描述时回退到 label
    region_inst.setdefault("description", region_label)

    result = build_object_proxy_dir_entry(
        scene_id=scene_id,
        intrinsics=intrinsics,
        frames=frames,
        pair=pair,
        threshold=threshold,
        instA=instA,
        instB=instB,
        instC=region_inst,
        rng=rng,
        marker_types=marker_types,
    )
    if result is None:
        return None

    result["question_type"] = "region_proxy_direction"
    result["sub_question_type"] = "proxy_region_direction"

    # old_line = f"What is the direction of {region_label} relative to {instB.get('label', 'object B')}?"
    # new_line = f"What is the direction of region {region_label} relative to {instB.get('label', 'object B')}?"
    # result["question"] = result.get("question", "").replace(old_line, new_line)

    objects = result.get("objects", {})
    if "A" in objects:
        objects["A"]["entity_type"] = "object"
    if "B" in objects:
        objects["B"]["entity_type"] = "object"
    if "C" in objects:
        objects["C"]["entity_type"] = "region"

    return result


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
    marker_types: Optional[List[str]] = None,
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
    image_a_stem = Path(fa.file_name).stem
    image_b_stem = Path(fb.file_name).stem
    image_a_name = Path(fa.file_name).name
    image_b_name = Path(fb.file_name).name
    
    # 从 instance 的 images 字段推断 mask_index.json 的位置
    # 优先使用 instA 的 images，如果没有则尝试 instB 或 instC
    images_for_inference = instA.get("images", [])
    if not images_for_inference:
        images_for_inference = instB.get("images", [])
    if not images_for_inference:
        images_for_inference = instC.get("images", [])
    
    # 加载 mask_index.json 并获取路径
    mask_index = _load_mask_index(scene_id, images_for_inference)
    mask_index_path = _infer_mask_index_path(scene_id, images_for_inference)
    mask_index_path_str = str(mask_index_path) if mask_index_path else None
    
    # 查找三个物体在两张图像中的 mask RLE 数据
    def _find_mask_rles(inst: Dict[str, Any], image_a_name: str, image_b_name: str) -> Tuple[Optional[Dict], Optional[Dict]]:
        """返回 (mask_rle_in_a, mask_rle_in_b)
        
        通过 instance 的 images 数组中的 mask_path 来匹配 mask_index.json 中的 mask_path，
        从而找到对应的 mask_rle。
        """
        # 从 instance 的 images 数组中提取对应图像的 mask_path
        images = inst.get("images", [])
        image_a_stem = Path(image_a_name).stem  # 例如 "DSC00850"
        image_b_stem = Path(image_b_name).stem  # 例如 "DSC00853"
        
        # 查找匹配的 mask_path
        mask_path_a = None
        mask_path_b = None
        
        for img_path_str in images:
            img_path = Path(img_path_str)
            # 路径格式：.../DSC00850/door.png，提取目录名（图像 stem）
            if img_path.parent.name == image_a_stem:
                mask_path_a = img_path_str
            if img_path.parent.name == image_b_stem:
                mask_path_b = img_path_str
        
        # 通过 mask_path 查找 mask_rle
        mask_rle_a = _find_mask_rle_by_path(mask_index, mask_path_a) if (mask_index and mask_path_a) else None
        mask_rle_b = _find_mask_rle_by_path(mask_index, mask_path_b) if (mask_index and mask_path_b) else None

        # fallback: mask_index.json 缺失 mask_rle 时，从 instance.mask_encodings 获取
        if mask_rle_a is None:
            mask_rle_a = _find_mask_rle_from_instance_encodings(inst, mask_path_a)
        if mask_rle_b is None:
            mask_rle_b = _find_mask_rle_from_instance_encodings(inst, mask_path_b)
        
        return mask_rle_a, mask_rle_b
    
    maskA_a, maskA_b = _find_mask_rles(instA, image_a_name, image_b_name)
    maskB_a, maskB_b = _find_mask_rles(instB, image_a_name, image_b_name)
    maskC_a, maskC_b = _find_mask_rles(instC, image_a_name, image_b_name)
    
    # 检查三个物体之间是否有bbox交集或距离过近
    # 如果任意两个物体有交集，则不适合用来提问
    bboxA = instA["bounding_box"]
    bboxB = instB["bounding_box"]
    bboxC = instC["bounding_box"]
    
    if _bboxes_intersect(bboxA, bboxB, min_separation=0.0):
        # print(f"警告: 物体A和B的bbox有交集，跳过该三元组")
        return None
    if _bboxes_intersect(bboxA, bboxC, min_separation=0.0):
        # print(f"警告: 物体A和C的bbox有交集，跳过该三元组")
        return None
    if _bboxes_intersect(bboxB, bboxC, min_separation=0.0):
        # print(f"警告: 物体B和C的bbox有交集，跳过该三元组")
        return None
    
    # 获取三个物体的3D中心（世界坐标）
    centerA = _bbox_center(bboxA)
    centerB = _bbox_center(bboxB)
    centerC = _bbox_center(bboxC)
    
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
        # print(f"警告: 无法分类方向（水平分量太小或垂直分量占主导），跳过该三元组")
        return None
    
    # 生成 MCQ 选项
    if rng is None:
        rng = random.Random()
    mcq_options, mcq_answer, forbidden = make_mcq_random_3_of_7(direction_idx, yaw_deg, rng=rng)
    
    # 随机选择标记方式（所有物体使用同一种方式，可外部控制）
    candidate_marker_types = marker_types or MARKER_TYPES
    if not candidate_marker_types:
        raise ValueError("marker_types 为空，无法选择标记方式")
    marker_type = rng.choice(candidate_marker_types)
    
    # 构建问题和答案
    labelA = instA.get("label", "object A")
    labelB = instB.get("label", "object B")
    labelC = instC.get("label", "object C")
    
    # 根据标记方式构建说明文本
    marker_descriptions = []
    marker_data = {}  # 存储标记相关的数据（如 2D 坐标、bbox 等）
    
    if marker_type == "point_marker":
        # Point marker: 在原图上画出点标记（从 RLE mask 计算中心点）
        for obj_label, inst, mask_a, mask_b in [("A", instA, maskA_a, maskA_b), ("B", instB, maskB_a, maskB_b), ("C", instC, maskC_a, maskC_b)]:
            label = inst.get("label", f"object {obj_label}")
            color = OBJECT_COLORS.get(obj_label, "red")
            # 从 RLE mask 计算中心点
            point_a = _get_2d_center_from_rle(mask_a) if mask_a else None
            point_b = _get_2d_center_from_rle(mask_b) if mask_b else None
            if point_a:
                marker_descriptions.append(f"In image A, a {color} point marker indicates {label}.")
                marker_data[f"image_a_point_{obj_label}"] = point_a
                marker_data[f"image_a_point_{obj_label}_color"] = color
            if point_b:
                marker_descriptions.append(f"In image B, a {color} point marker indicates {label}.")
                marker_data[f"image_b_point_{obj_label}"] = point_b
                marker_data[f"image_b_point_{obj_label}_color"] = color
    
    elif marker_type == "white_mask":
        # White mask: 单独提供 mask 图像作为输入（不在原图上绘制）
        for obj_label, inst, mask_a, mask_b in [("A", instA, maskA_a, maskA_b), ("B", instB, maskB_a, maskB_b), ("C", instC, maskC_a, maskC_b)]:
            label = inst.get("label", f"object {obj_label}")
            if mask_a:
                marker_descriptions.append(f"In image A, a separate white mask image highlights {label}.")
            if mask_b:
                marker_descriptions.append(f"In image B, a separate white mask image highlights {label}.")
    
    elif marker_type == "mask_covering_image":
        # Mask covering image: mask 覆盖在原图上
        for obj_label, inst, mask_a, mask_b in [("A", instA, maskA_a, maskA_b), ("B", instB, maskB_a, maskB_b), ("C", instC, maskC_a, maskC_b)]:
            label = inst.get("label", f"object {obj_label}")
            color = OBJECT_COLORS.get(obj_label, "red")
            if mask_a:
                marker_descriptions.append(f"In image A, a {color} mask overlay indicates {label}.")
                marker_data[f"image_a_mask_rle_{obj_label}"] = mask_a
                marker_data[f"image_a_mask_rle_{obj_label}_color"] = color
            if mask_b:
                marker_descriptions.append(f"In image B, a {color} mask overlay indicates {label}.")
                marker_data[f"image_b_mask_rle_{obj_label}"] = mask_b
                marker_data[f"image_b_mask_rle_{obj_label}_color"] = color
    
    elif marker_type == "2d_bbox":
        # 2D bbox: 从 RLE mask 计算 2D 边界框
        for obj_label, inst, mask_a, mask_b in [("A", instA, maskA_a, maskA_b), ("B", instB, maskB_a, maskB_b), ("C", instC, maskC_a, maskC_b)]:
            label = inst.get("label", f"object {obj_label}")
            color = OBJECT_COLORS.get(obj_label, "red")
            # 从 RLE mask 计算 2D 边界框
            bbox_a = _get_2d_bbox_from_rle(mask_a) if mask_a else None
            bbox_b = _get_2d_bbox_from_rle(mask_b) if mask_b else None
            if bbox_a:
                marker_descriptions.append(f"In image A, a {color} 2D bounding box indicates {label}.")
                marker_data[f"image_a_bbox_{obj_label}"] = bbox_a
                marker_data[f"image_a_bbox_{obj_label}_color"] = color
            if bbox_b:
                marker_descriptions.append(f"In image B, a {color} 2D bounding box indicates {label}.")
                marker_data[f"image_b_bbox_{obj_label}"] = bbox_b
                marker_data[f"image_b_bbox_{obj_label}_color"] = color
    
    elif marker_type == "language_description":
        # Language description: 使用自然语言描述来指代物体，缺失则直接报错
        # 注意：不要使用 "In both images"（会产生歧义），应明确出现在 image A / image B
        masks_by_obj = {
            "A": (maskA_a, maskA_b),
            "B": (maskB_a, maskB_b),
            "C": (maskC_a, maskC_b),
        }
        for obj_label, inst in [("A", instA), ("B", instB), ("C", instC)]:
            label = inst.get("label", f"object {obj_label}")
            description = inst.get("description")

            if not description:
                raise ValueError(f"language_description 缺少描述: ins_id={inst.get('ins_id')}, label={label}")

            mask_a, mask_b = masks_by_obj.get(obj_label, (None, None))
            if not mask_a and not mask_b:
                raise ValueError(f"language_description 物体未出现在 image A/B 中: ins_id={inst.get('ins_id')}, label={label}")

            is_region = str(inst.get("entity_type", "")).lower() == "region"

            # 避免同一实体在 A/B 重复描述：
            # A 角色优先落在 image A；B/C 角色优先落在 image B（缺失时兜底到另一张图）。
            if obj_label == "A":
                if mask_a:
                    marker_descriptions.append(f"In image A, the object described as \"{description}\" is the {label}.")
                elif mask_b:
                    marker_descriptions.append(f"In image B, the object described as \"{description}\" is the {label}.")
            elif obj_label == "B":
                if mask_b:
                    marker_descriptions.append(f"In image B, the object described as \"{description}\" is the {label}.")
                elif mask_a:
                    marker_descriptions.append(f"In image A, the object described as \"{description}\" is the {label}.")
            else:  # C
                if is_region:
                    marker_descriptions.append(f"The {label} is in Image B.")
                else:
                    if mask_b:
                        marker_descriptions.append(f"In image B, the object described as \"{description}\" is the {label}.")
                    elif mask_a:
                        marker_descriptions.append(f"In image A, the object described as \"{description}\" is the {label}.")
            marker_data[f"description_{obj_label}"] = description
    
    marker_text = "\n".join(marker_descriptions) if marker_descriptions else ""
    
    # MCQ 格式的问题
    options_text = "\n".join([f"{k}. {v}" for k, v in sorted(mcq_options.items())])
    question_parts = []
    if marker_text:
        question_parts.append(marker_text)
    question_parts.append(f"You are positioned at {labelA} and face {labelB}.")
    question_parts.append(f"In which direction is {labelC} relative to you?")
    question_parts.append(options_text)
    
    question = "\n".join(question_parts).strip()
    
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
        "marker_type": marker_type,
        "marker_data": marker_data,
        "mask_index_path": mask_index_path_str,
        # 只有 white_mask 时才返回单独的 mask_rle（用于单独显示）
        # 其他方式（point_marker, 2d_bbox, mask_covering_image）的标记信息在 marker_data 中
        **({
            "image_a_mask_rle_A": maskA_a,
            "image_a_mask_rle_B": maskB_a,
            "image_a_mask_rle_C": maskC_a,
            "image_b_mask_rle_A": maskA_b,
            "image_b_mask_rle_B": maskB_b,
            "image_b_mask_rle_C": maskC_b,
        } if marker_type == "white_mask" else {}),
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


def build_region_relpos_entry(
    scene_id: str,
    intrinsics: Dict[str, float],
    frames: Sequence["FrameItem"],
    pair: Tuple[int, int, float],
    threshold: float,
    instA: Dict[str, Any],
    instB: Dict[str, Any],
    regionC: Dict[str, Any],
    rng: random.Random = None,
    marker_types: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    构造"站在物体A，面向物体B，问region C的方位和距离"QA（MCQ格式）。
    复用 object relpos 主逻辑，并在输出语义上将 C 明确为 region。
    """
    region_inst = dict(regionC)
    region_label = region_inst.get("label", "region C")
    # 与其它 region 模板一致：language_description 分支缺 description 时回退到 label
    region_inst.setdefault("description", region_label)

    result = build_object_relpos_entry(
        scene_id=scene_id,
        intrinsics=intrinsics,
        frames=frames,
        pair=pair,
        threshold=threshold,
        instA=instA,
        instB=instB,
        instC=region_inst,
        rng=rng,
        marker_types=marker_types,
    )
    if result is None:
        return None

    result["question_type"] = "region_relpos"
    result["sub_question_type"] = "relpos_A_facing_B_region_C"

    old_line = f"In which direction is {region_label} relative to you?"
    new_line = f"In which direction is the {region_label} relative to you?"
    result["question"] = result.get("question", "").replace(old_line, new_line)

    objects = result.get("objects", {})
    if "A" in objects:
        objects["A"]["entity_type"] = "object"
    if "B" in objects:
        objects["B"]["entity_type"] = "object"
    if "C" in objects:
        objects["C"]["entity_type"] = "region"

    return result


def build_region_proxy_dir_entry(
    scene_id: str,
    intrinsics: Dict[str, float],
    frames: Sequence["FrameItem"],
    pair: Tuple[int, int, float],
    threshold: float,
    instA: Dict[str, Any],
    instB: Dict[str, Any],
    regionC: Dict[str, Any],
    rng: random.Random = None,
    marker_types: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    构造"给定 obj1 相对 obj2 的方向，问 Region 3 相对 obj2 的方向"QA（MCQ）。
    复用 object proxy 主逻辑，并在输出语义上将 C 明确为 region。
    """
    region_inst = dict(regionC)
    region_label = region_inst.get("label", "region C")
    region_inst.setdefault("description", region_label)

    result = build_object_proxy_dir_entry(
        scene_id=scene_id,
        intrinsics=intrinsics,
        frames=frames,
        pair=pair,
        threshold=threshold,
        instA=instA,
        instB=instB,
        instC=region_inst,
        rng=rng,
        marker_types=marker_types,
    )
    if result is None:
        return None

    result["question_type"] = "region_proxy_direction"
    result["sub_question_type"] = "proxy_object_region_direction"

    labelB = (
        result.get("objects", {})
        .get("B", {})
        .get("label", instB.get("label", "object B"))
    )

    objects = result.get("objects", {})
    if "A" in objects:
        objects["A"]["entity_type"] = "object"
    if "B" in objects:
        objects["B"]["entity_type"] = "object"
    if "C" in objects:
        objects["C"]["entity_type"] = "region"

    return result
