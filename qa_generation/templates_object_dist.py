import json
import math
import textwrap
import random
from pathlib import Path
from typing import Dict, Sequence, Tuple, TYPE_CHECKING, List, Any, Optional

import numpy as np
from pycocotools import mask as mask_utils

if TYPE_CHECKING:
    from qa_generation.generate_two_view_qa import FrameItem

# 标记方式
MARKER_TYPES = ["point_marker", "white_mask", "mask_covering_image", "2d_bbox", "language_description", "language_regions"]

# 物体颜色映射
OBJECT_COLORS = {
    "1": "red",
    "2": "green",
}


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


def _bbox_top_z(bbox: Sequence[Sequence[float]]) -> float:
    if len(bbox) == 0:
        raise ValueError("bounding_box 为空")
    first = bbox[0]
    if isinstance(first, dict) and {"x", "y", "z"} <= set(first.keys()):
        arr = np.array([[p["x"], p["y"], p["z"]] for p in bbox], dtype=np.float32)
    else:
        arr = np.asarray(bbox, dtype=np.float32)
    if arr.shape != (8, 3):
        raise ValueError(f"bounding_box 期望形状 (8,3)，当前 {arr.shape}")
    return float(arr[:, 2].min())


def _bbox_longest_side(bbox: Sequence[Sequence[float]], tol: float = 1e-4) -> float:
    if len(bbox) == 0:
        raise ValueError("bounding_box 为空")
    first = bbox[0]
    if isinstance(first, dict) and {"x", "y", "z"} <= set(first.keys()):
        arr = np.array([[p["x"], p["y"], p["z"]] for p in bbox], dtype=np.float32)
    else:
        arr = np.asarray(bbox, dtype=np.float32)
    if arr.shape != (8, 3):
        raise ValueError(f"bounding_box 期望形状 (8,3)，当前 {arr.shape}")

    diff = arr[:, None, :] - arr[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    vals = np.sort(dists[np.triu_indices(8, k=1)])
    unique = []
    for v in vals:
        if v < 1e-6:
            continue
        if not unique or abs(v - unique[-1]) > tol:
            unique.append(float(v))
        if len(unique) >= 3:
            break
    if not unique:
        return 0.0
    if len(unique) >= 3:
        return max(unique[:3])
    return unique[-1]


def _world_to_cam(point_world: np.ndarray, c2w: Sequence[Sequence[float]]) -> np.ndarray:
    mat = np.asarray(c2w, dtype=np.float32)
    if mat.shape != (4, 4):
        raise ValueError(f"transform_matrix 期望 4x4，当前 {mat.shape}")
    w2c = np.linalg.inv(mat)
    homo = np.concatenate([point_world, [1.0]], axis=0)
    cam = w2c @ homo
    return cam[:3]


def _dir8_label_from_cam(
    point_cam: Sequence[float],
    ratio_threshold: float = 0.5774,
    min_val: float = 1e-6,
) -> Optional[str]:
    x = float(point_cam[0])
    z = float(point_cam[2])
    abs_x = abs(x)
    abs_z = abs(z)
    top_mag = max(abs_x, abs_z)
    if top_mag < min_val:
        return None

    ns = "North" if z > 0 else "South"
    ew = "East" if x > 0 else "West"

    include_second = False
    if abs_x >= abs_z:
        include_second = abs_z >= ratio_threshold * top_mag
        dominant = ew
    else:
        include_second = abs_x >= ratio_threshold * top_mag
        dominant = ns

    if include_second:
        return f"{ns}-{ew}"
    return dominant


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


def _build_size_mcq_options(
    label1: str,
    label2: str,
    answer_text: str,
    same_text: str,
    mode: str,
    rng: random.Random,
) -> Tuple[Dict[str, str], str]:
    if mode == "length":
        cannot_tell_variants = [
            "Cannot tell",
            "Cannot be determined.",
            f"Sometimes the {label1} is longer, sometimes the {label2} is longer.",
        ]
    else:
        cannot_tell_variants = [
            "Cannot tell",
            "Cannot be determined.",
            f"Sometimes the {label1} is taller, sometimes the {label2} is taller.",
        ]
    cannot_tell_text = rng.choice(cannot_tell_variants)
    options_list = [
        label1,
        label2,
        same_text,
        cannot_tell_text,
    ]
    rng.shuffle(options_list)
    letters = ["A", "B", "C", "D"]
    options = {letters[i]: options_list[i] for i in range(4)}
    answer = next(k for k, v in options.items() if v == answer_text)
    return options, answer


def _load_mask_index(scene_id: str, images: List[str]) -> Optional[Dict]:
    """从 images 路径列表中推断并加载 mask_index.json 文件。
    
    Args:
        scene_id: 场景 ID
        images: mask 图像路径列表（来自 bbox_json 的 images 字段）
    
    Returns:
        mask_index.json 的内容，如果文件不存在或无法推断则返回 None
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
    parts = first_image_path.parts
    scene_id_idx = None
    for i, part in enumerate(parts):
        if part == scene_id:
            scene_id_idx = i
            break
    
    if scene_id_idx is None:
        return None
    
    # scene_id 所在的目录就是 mask 根目录
    mask_root = Path(*parts[:scene_id_idx + 1])
    mask_index_path = mask_root / "mask_index.json"
    
    if not mask_index_path.exists():
        return None
    
    try:
        return json.loads(mask_index_path.read_text())
    except Exception as e:
        print(f"警告: 无法加载 mask_index.json ({mask_index_path}): {e}")
        return None


def _find_mask_rle_by_path(mask_index: Dict, mask_path: str) -> Optional[Dict]:
    """通过 mask_path 从 mask_index.json 中查找对应的 mask RLE 数据。"""
    if not mask_index or "items" not in mask_index or not mask_path:
        return None
    
    mask_path_normalized = str(Path(mask_path))
    
    for item in mask_index["items"]:
        item_mask_path = item.get("mask_path")
        if item_mask_path:
            item_mask_path_normalized = str(Path(item_mask_path))
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


def _get_2d_bbox_from_rle(mask_rle: Dict) -> Optional[Tuple[float, float, float, float]]:
    """从 RLE mask 计算 2D 边界框 (x_min, y_min, x_max, y_max)。"""
    try:
        if isinstance(mask_rle, dict) and "size" in mask_rle and "counts" in mask_rle:
            mask_array = mask_utils.decode(mask_rle)
            rows, cols = np.where(mask_array > 0)
            if len(rows) == 0:
                return None
            x_min, y_min = float(cols.min()), float(rows.min())
            x_max, y_max = float(cols.max()), float(rows.max())
            return (x_min, y_min, x_max, y_max)
    except Exception:
        pass
    return None


def _get_2d_center_from_rle(mask_rle: Dict) -> Optional[Tuple[float, float]]:
    """从 RLE mask 计算 2D 中心点。"""
    try:
        if isinstance(mask_rle, dict) and "size" in mask_rle and "counts" in mask_rle:
            mask_array = mask_utils.decode(mask_rle)
            rows, cols = np.where(mask_array > 0)
            if len(rows) == 0:
                return None
            x_center = float(cols.mean())
            y_center = float(rows.mean())
            return (x_center, y_center)
    except Exception:
        pass
    return None


def build_object_dist_entry(
    scene_id: str,
    intrinsics: Dict[str, float],
    frames: Sequence["FrameItem"],
    pair: Tuple[int, int, float],
    threshold: float,
    inst1: Dict[str, Any],
    inst2: Dict[str, Any],
    rng: random.Random = None,
    marker_types: Optional[List[str]] = None,
):
    """
    构造"给定 A 视角中的物体 1 和 B 视角中的物体 2，问它们之间的 3D 距离是多少"的 QA。
    """
    if rng is None:
        rng = random.Random()
    
    i, j, cov = pair
    fa, fb = frames[i], frames[j]
    image_a_stem = Path(fa.file_name).stem
    image_b_stem = Path(fb.file_name).stem

    # 从 instance 的 images 数组中提取对应图像的 mask_path
    images1 = inst1.get("images", [])
    images2 = inst2.get("images", [])
    
    mask_path1 = None
    mask_path2 = None
    
    for img_path_str in images1:
        img_path = Path(img_path_str)
        if img_path.parent.name == image_a_stem:
            mask_path1 = img_path_str
            break
    
    for img_path_str in images2:
        img_path = Path(img_path_str)
        if img_path.parent.name == image_b_stem:
            mask_path2 = img_path_str
            break
    
    if mask_path1 is None or mask_path2 is None:
        raise ValueError(f"无法找到mask路径: inst1在{image_a_stem}或inst2在{image_b_stem}中不存在")

    # 从 instance 的 images 字段推断 mask_index.json 的位置
    images_for_inference = images1 if images1 else images2
    
    # 加载 mask_index.json 并查找 RLE
    mask_index = _load_mask_index(scene_id, images_for_inference)
    mask_rle1 = _find_mask_rle_by_path(mask_index, mask_path1) if mask_index else None
    mask_rle2 = _find_mask_rle_by_path(mask_index, mask_path2) if mask_index else None
    if mask_rle1 is None:
        mask_rle1 = _find_mask_rle_from_instance_encodings(inst1, mask_path1)
    if mask_rle2 is None:
        mask_rle2 = _find_mask_rle_from_instance_encodings(inst2, mask_path2)

    center1 = _bbox_center(inst1["bounding_box"])
    center2 = _bbox_center(inst2["bounding_box"])
    
    dist_3d = float(np.linalg.norm(center1 - center2))

    label1 = inst1.get("label") or "object 1"
    label2 = inst2.get("label") or "object 2"

    # 随机选择标记方式（可由外部指定）
    candidate_marker_types = marker_types or MARKER_TYPES
    if not candidate_marker_types:
        raise ValueError("marker_types 为空，无法选择标记方式")
    marker_type = rng.choice(candidate_marker_types)
    color1 = OBJECT_COLORS.get("1", "red")
    color2 = OBJECT_COLORS.get("2", "green")
    
    # 根据标记方式构建说明文本
    marker_descriptions = []
    marker_data = {}
    
    if marker_type == "point_marker":
        point1 = _get_2d_center_from_rle(mask_rle1) if mask_rle1 else None
        if point1:
            marker_descriptions.append(f"In image A, a {color1} point marker indicates {label1}.")
            marker_data["image_a_point_1"] = point1
            marker_data["image_a_point_1_color"] = color1
        point2 = _get_2d_center_from_rle(mask_rle2) if mask_rle2 else None
        if point2:
            marker_descriptions.append(f"In image B, a {color2} point marker indicates {label2}.")
            marker_data["image_b_point_2"] = point2
            marker_data["image_b_point_2_color"] = color2
    
    elif marker_type == "white_mask":
        marker_descriptions.append(f"In image A, a separate white mask image highlights {label1}.")
        marker_descriptions.append(f"In image B, a separate white mask image highlights {label2}.")
    
    elif marker_type == "mask_covering_image":
        if mask_rle1:
            marker_descriptions.append(f"In image A, a {color1} mask overlay indicates {label1}.")
            marker_data["image_a_mask_rle_1"] = mask_rle1
            marker_data["image_a_mask_rle_1_color"] = color1
        if mask_rle2:
            marker_descriptions.append(f"In image B, a {color2} mask overlay indicates {label2}.")
            marker_data["image_b_mask_rle_2"] = mask_rle2
            marker_data["image_b_mask_rle_2_color"] = color2
    
    elif marker_type == "2d_bbox":
        bbox1 = _get_2d_bbox_from_rle(mask_rle1) if mask_rle1 else None
        if bbox1:
            marker_descriptions.append(f"In image A, a {color1} 2D bounding box indicates {label1}.")
            marker_data["image_a_bbox_1"] = bbox1
            marker_data["image_a_bbox_1_color"] = color1
        bbox2 = _get_2d_bbox_from_rle(mask_rle2) if mask_rle2 else None
        if bbox2:
            marker_descriptions.append(f"In image B, a {color2} 2D bounding box indicates {label2}.")
            marker_data["image_b_bbox_2"] = bbox2
            marker_data["image_b_bbox_2_color"] = color2
    
    elif marker_type in {"language_description", "language_regions"}:
        description1 = inst1.get("description")
        description2 = inst2.get("description")

        is_region1 = str(inst1.get("entity_type", "")).lower() == "region"
        is_region2 = str(inst2.get("entity_type", "")).lower() == "region"
        if is_region1:
            marker_descriptions.append(f"In the {label1} of image A.")
        else:
            marker_descriptions.append(f"In image A, the object described as \"{description1}\" is the {label1}.")
            marker_data["image_a_description_1"] = description1
        if is_region2:
            marker_descriptions.append(f"In the {label2} of image B.")
        else:
            marker_descriptions.append(f"In image B, the object described as \"{description2}\" is the {label2}.")
            marker_data["image_b_description_2"] = description2
    
    marker_text = "\n".join(marker_descriptions) if marker_descriptions else ""
    
    question = textwrap.dedent(
        f"""
        {marker_text}
        Estimate the 3D metric distance (in meters) between the centers of these two physical objects.
        Return only one number in meters (e.g., 1.2).
        Output format: <answer>NUMBER</answer>.
        """
    ).strip()
    
    answer = f"{dist_3d:.2f} m"

    result = {
        "scene_id": scene_id,
        "image_a": fa.file_name,
        "image_b": fb.file_name,
        "covisibility": cov,
        "threshold": threshold,
        "question_type": "object_distance",
        "sub_question_type": "inter_object_distance",
        "question": question,
        "answer": answer,
        "distance_gt": dist_3d,
        "marker_type": marker_type,
        "marker_data": marker_data,
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
    
    # 只有 white_mask 时才返回单独的 mask_rle（用于单独显示）
    if marker_type == "white_mask":
        result["image_a_mask_rle_1"] = mask_rle1
        result["image_b_mask_rle_2"] = mask_rle2
    
    return result


def build_object_height_entry(
    scene_id: str,
    intrinsics: Dict[str, float],
    frames: Sequence["FrameItem"],
    pair: Tuple[int, int, float],
    threshold: float,
    inst1: Dict[str, Any],
    inst2: Dict[str, Any],
    rng: random.Random = None,
    marker_types: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    构造"给定 A 视角的物体1 和 B 视角的物体2，问哪个更高/更高(看顶部)"的 QA（MCQ）。
    """
    if rng is None:
        rng = random.Random()

    i, j, cov = pair
    fa, fb = frames[i], frames[j]
    image_a_stem = Path(fa.file_name).stem
    image_b_stem = Path(fb.file_name).stem

    images1 = inst1.get("images", [])
    images2 = inst2.get("images", [])

    mask_path1 = None
    mask_path2 = None

    for img_path_str in images1:
        img_path = Path(img_path_str)
        if img_path.parent.name == image_a_stem:
            mask_path1 = img_path_str
            break

    for img_path_str in images2:
        img_path = Path(img_path_str)
        if img_path.parent.name == image_b_stem:
            mask_path2 = img_path_str
            break

    if mask_path1 is None or mask_path2 is None:
        raise ValueError(f"无法找到mask路径: inst1在{image_a_stem}或inst2在{image_b_stem}中不存在")

    images_for_inference = images1 if images1 else images2
    mask_index = _load_mask_index(scene_id, images_for_inference)
    mask_rle1 = _find_mask_rle_by_path(mask_index, mask_path1) if mask_index else None
    mask_rle2 = _find_mask_rle_by_path(mask_index, mask_path2) if mask_index else None
    if mask_rle1 is None:
        mask_rle1 = _find_mask_rle_from_instance_encodings(inst1, mask_path1)
    if mask_rle2 is None:
        mask_rle2 = _find_mask_rle_from_instance_encodings(inst2, mask_path2)

    top_z1 = _bbox_top_z(inst1["bounding_box"])
    top_z2 = _bbox_top_z(inst2["bounding_box"])
    longest1 = _bbox_longest_side(inst1["bounding_box"])
    longest2 = _bbox_longest_side(inst2["bounding_box"])

    label1 = inst1.get("label") or "object 1"
    label2 = inst2.get("label") or "object 2"

    use_length = rng.random() < 0.5
    if use_length:
        if abs(longest1 - longest2) < 1e-3:
            answer_text = "Both are about the same length"
        elif longest1 > longest2:
            answer_text = label1
        else:
            answer_text = label2
    else:
        if abs(top_z1 - top_z2) < 1e-3:
            answer_text = "Both are about the same height"
        elif top_z1 < top_z2:
            answer_text = label1
        else:
            answer_text = label2

    candidate_marker_types = marker_types or MARKER_TYPES
    if not candidate_marker_types:
        raise ValueError("marker_types 为空，无法选择标记方式")
    marker_type = rng.choice(candidate_marker_types)
    color1 = OBJECT_COLORS.get("1", "red")
    color2 = OBJECT_COLORS.get("2", "green")

    marker_descriptions = []
    marker_data = {}

    if marker_type == "point_marker":
        point1 = _get_2d_center_from_rle(mask_rle1) if mask_rle1 else None
        if point1:
            marker_descriptions.append(f"In image A, a {color1} point marker indicates {label1}.")
            marker_data["image_a_point_1"] = point1
            marker_data["image_a_point_1_color"] = color1
        point2 = _get_2d_center_from_rle(mask_rle2) if mask_rle2 else None
        if point2:
            marker_descriptions.append(f"In image B, a {color2} point marker indicates {label2}.")
            marker_data["image_b_point_2"] = point2
            marker_data["image_b_point_2_color"] = color2

    elif marker_type == "white_mask":
        marker_descriptions.append(f"In image A, a separate white mask image highlights {label1}.")
        marker_descriptions.append(f"In image B, a separate white mask image highlights {label2}.")

    elif marker_type == "mask_covering_image":
        if mask_rle1:
            marker_descriptions.append(f"In image A, a {color1} mask overlay indicates {label1}.")
            marker_data["image_a_mask_rle_1"] = mask_rle1
            marker_data["image_a_mask_rle_1_color"] = color1
        if mask_rle2:
            marker_descriptions.append(f"In image B, a {color2} mask overlay indicates {label2}.")
            marker_data["image_b_mask_rle_2"] = mask_rle2
            marker_data["image_b_mask_rle_2_color"] = color2

    elif marker_type == "2d_bbox":
        bbox1 = _get_2d_bbox_from_rle(mask_rle1) if mask_rle1 else None
        if bbox1:
            marker_descriptions.append(f"In image A, a {color1} 2D bounding box indicates {label1}.")
            marker_data["image_a_bbox_1"] = bbox1
            marker_data["image_a_bbox_1_color"] = color1
        bbox2 = _get_2d_bbox_from_rle(mask_rle2) if mask_rle2 else None
        if bbox2:
            marker_descriptions.append(f"In image B, a {color2} 2D bounding box indicates {label2}.")
            marker_data["image_b_bbox_2"] = bbox2
            marker_data["image_b_bbox_2_color"] = color2

    elif marker_type in {"language_description", "language_regions"}:
        description1 = inst1.get("description")
        description2 = inst2.get("description")
        is_region1 = str(inst1.get("entity_type", "")).lower() == "region"
        is_region2 = str(inst2.get("entity_type", "")).lower() == "region"
        if is_region1:
            marker_descriptions.append(f"In the {label1} of image A.")
        else:
            marker_descriptions.append(f"In image A, the object described as \"{description1}\" is the {label1}.")
            marker_data["image_a_description_1"] = description1
        if is_region2:
            marker_descriptions.append(f"In the {label2} of image B.")
        else:
            marker_descriptions.append(f"In image B, the object described as \"{description2}\" is the {label2}.")
            marker_data["image_b_description_2"] = description2

    marker_text = "\n".join(marker_descriptions) if marker_descriptions else ""
    if use_length:
        options, answer_letter = _build_size_mcq_options(
            label1,
            label2,
            answer_text,
            "Both are about the same length",
            "length",
            rng,
        )
    else:
        options, answer_letter = _build_size_mcq_options(
            label1,
            label2,
            answer_text,
            "Both are about the same height",
            "height",
            rng,
        )
    options_text = "\n".join([f"{k}. {v}" for k, v in sorted(options.items())])

    question_parts = []
    if marker_text:
        question_parts.append(marker_text)
    if use_length:
        question_parts.append(
            f"Which is longer (consider the longest side of the object)? {label1} or {label2}?"
        )
    else:
        question_parts.append(
            f"Which object is taller (consider the top of the objects)? {label1} or {label2}?"
        )
    question_parts.append(options_text)
    question = "\n".join(question_parts).strip()

    answer = f"{answer_letter}. {answer_text}"

    result = {
        "scene_id": scene_id,
        "image_a": fa.file_name,
        "image_b": fb.file_name,
        "covisibility": cov,
        "threshold": threshold,
        "question_type": "object_height_or_length",
        "sub_question_type": "longest_side_compare" if use_length else "top_height_compare",
        "question": question,
        "answer": answer,
        "question_variant": "length" if use_length else "height",
        "marker_type": marker_type,
        "marker_data": marker_data,
        "options": options,
        "object_1": {
            "ins_id": inst1.get("ins_id"),
            "label": label1,
            "top_z": top_z1,
            "longest_side": longest1,
        },
        "object_2": {
            "ins_id": inst2.get("ins_id"),
            "label": label2,
            "top_z": top_z2,
            "longest_side": longest2,
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

    if marker_type == "white_mask":
        result["image_a_mask_rle_1"] = mask_rle1
        result["image_b_mask_rle_2"] = mask_rle2

    return result


def build_region_cross_dir_entry(
    scene_id: str,
    intrinsics: Dict[str, float],
    frames: Sequence["FrameItem"],
    pair: Tuple[int, int, float],
    threshold: float,
    inst1: Dict[str, Any],
    inst2: Dict[str, Any],
    rng: random.Random = None,
    marker_types: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """构造 image A region + image B region 的跨图方向 QA（复用 object cross 逻辑）。"""
    inst1_region = dict(inst1)
    inst2_region = dict(inst2)
    label1 = inst1_region.get("label") or "region 1"
    label2 = inst2_region.get("label") or "region 2"
    # 不依赖 region description：仅为兼容原始 object 函数的 language_description 分支。
    inst1_region.setdefault("description", label1)
    inst2_region.setdefault("description", label2)

    result = build_object_cross_dir_entry(
        scene_id=scene_id,
        intrinsics=intrinsics,
        frames=frames,
        pair=pair,
        threshold=threshold,
        inst1=inst1_region,
        inst2=inst2_region,
        rng=rng,
        marker_types=marker_types,
    )
    if result is None:
        return None

    result["question_type"] = "region_cross_direction"
    result["sub_question_type"] = "cross_image_assumed_dir_region_region"
    result["question"] =  result["question"]
    if result.get("marker_type") == "language_description":
        old_a = f"In image A, the object described as \"{inst1_region.get('description')}\" is the {label1}."
        old_b = f"In image B, the object described as \"{inst2_region.get('description')}\" is the {label2}."
        result["question"] = result["question"].replace(old_a, f"In the {label1} of image A.")
        result["question"] = result["question"].replace(old_b, f"In the {label2} of image B.")
        marker_data = result.get("marker_data", {})
        marker_data.pop("image_a_description_1", None)
        marker_data.pop("image_b_description_2", None)

    obj1 = result.pop("object_1", None)
    obj2 = result.pop("object_2", None)
    if obj1 is not None:
        obj1["entity_type"] = "region"
        result["region_1"] = obj1
    if obj2 is not None:
        obj2["entity_type"] = "region"
        result["region_2"] = obj2
    return result


def build_region_object_cross_dir_entry(
    scene_id: str,
    intrinsics: Dict[str, float],
    frames: Sequence["FrameItem"],
    pair: Tuple[int, int, float],
    threshold: float,
    inst_object: Dict[str, Any],
    inst_region: Dict[str, Any],
    rng: random.Random = None,
    marker_types: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """构造 image A object + image B region 的跨图方向 QA（复用 object cross 逻辑）。"""
    obj = dict(inst_object)
    region = dict(inst_region)
    label2 = region.get("label") or "region 2"
    # 不依赖 region description：仅为兼容原始 object 函数的 language_description 分支。
    region.setdefault("description", label2)

    result = build_object_cross_dir_entry(
        scene_id=scene_id,
        intrinsics=intrinsics,
        frames=frames,
        pair=pair,
        threshold=threshold,
        inst1=obj,
        inst2=region,
        rng=rng,
        marker_types=marker_types,
    )
    if result is None:
        return None

    result["question_type"] = "region_object_cross_direction"
    result["sub_question_type"] = "cross_image_assumed_dir_object_region"
    # result["question"] = "Image A target is an object; image B target is a region.\n" + result["question"]
    if result.get("marker_type") == "language_description":
        # old_b = f"In image B, the object described as \"{region.get('description')}\" is the {label2}."
        # result["question"] = result["question"].replace(old_b, f"In the {label2} of image B.")
        # A 端是 object，保持原句即可；无 highlight，用 strange area；仅清理 B 端 description 元数据
        marker_data = result.get("marker_data", {})
        marker_data.pop("image_b_description_2", None)

    obj2 = result.pop("object_2", None)
    if obj2 is not None:
        obj2["entity_type"] = "region"
        result["region_2"] = obj2
    if "object_1" in result:
        result["object_1"]["entity_type"] = "object"
    return result


def build_object_cross_dir_entry(
    scene_id: str,
    intrinsics: Dict[str, float],
    frames: Sequence["FrameItem"],
    pair: Tuple[int, int, float],
    threshold: float,
    inst1: Dict[str, Any],
    inst2: Dict[str, Any],
    rng: random.Random = None,
    marker_types: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    构造"image A 的物体1 + image B 的物体2，问物体2在image B的方位"QA（MCQ）。
    """
    if rng is None:
        rng = random.Random()

    i, j, cov = pair
    fa, fb = frames[i], frames[j]
    image_a_stem = Path(fa.file_name).stem
    image_b_stem = Path(fb.file_name).stem

    images1 = inst1.get("images", [])
    images2 = inst2.get("images", [])

    mask_path1 = None
    mask_path2 = None

    for img_path_str in images1:
        img_path = Path(img_path_str)
        if img_path.parent.name == image_a_stem:
            mask_path1 = img_path_str
            break

    for img_path_str in images2:
        img_path = Path(img_path_str)
        if img_path.parent.name == image_b_stem:
            mask_path2 = img_path_str
            break

    if mask_path1 is None or mask_path2 is None:
        raise ValueError(f"无法找到mask路径: inst1在{image_a_stem}或inst2在{image_b_stem}中不存在")

    images_for_inference = images1 if images1 else images2
    mask_index = _load_mask_index(scene_id, images_for_inference)
    mask_rle1 = _find_mask_rle_by_path(mask_index, mask_path1) if mask_index else None
    mask_rle2 = _find_mask_rle_by_path(mask_index, mask_path2) if mask_index else None
    if mask_rle1 is None:
        mask_rle1 = _find_mask_rle_from_instance_encodings(inst1, mask_path1)
    if mask_rle2 is None:
        mask_rle2 = _find_mask_rle_from_instance_encodings(inst2, mask_path2)

    center1_world = _bbox_center(inst1["bounding_box"])
    center1_cam_a = _world_to_cam(center1_world, fa.transform_matrix)
    center2_world = _bbox_center(inst2["bounding_box"])
    Ta = np.asarray(fa.transform_matrix, dtype=np.float32)
    Tb = np.asarray(fb.transform_matrix, dtype=np.float32)
    cam_b_pos = Tb[:3, 3]
    v_world = center2_world - cam_b_pos
    center2_cam_a = Ta[:3, :3].T @ v_world
    center2_cam_b = _world_to_cam(center2_world, fb.transform_matrix)

    label1 = inst1.get("label") or "object 1"
    label2 = inst2.get("label") or "object 2"

    def maybe_suffix(label: str) -> str:
        if rng.random() < 0.5:
            suffix = "region" if rng.random() < 0.5 else "area"
            return f"{label} {suffix}"
        return label

    label1_q = maybe_suffix(label1)
    label2_q = label2
    assumed_dir = rng.choice(["north", "south", "east", "west"])
    dir_to_angle = {
        "north": 0.0,
        "east": math.pi / 2.0,
        "south": math.pi,
        "west": -math.pi / 2.0,
    }
    x1, z1 = float(center1_cam_a[0]), float(center1_cam_a[2])
    if abs(x1) < 1e-6 and abs(z1) < 1e-6:
        return None
    angle_cam_a = math.atan2(x1, z1)
    delta = angle_cam_a - dir_to_angle[assumed_dir]

    up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    cos_d = math.cos(delta)
    sin_d = math.sin(delta)
    forward = np.array([sin_d, 0.0, cos_d], dtype=np.float32)
    forward = forward / max(np.linalg.norm(forward), 1e-6)
    right = np.cross(forward, up)
    right = right / max(np.linalg.norm(right), 1e-6)

    v_fwd = float(np.dot(center2_cam_a, forward))
    v_right = float(np.dot(center2_cam_a, right))
    direction_label = _dir8_label_from_cam([v_right, 0.0, v_fwd])
    if direction_label is None:
        return None

    mcq_options, mcq_answer = _build_dir8_mcq_options(direction_label, rng)

    candidate_marker_types = marker_types or MARKER_TYPES
    if not candidate_marker_types:
        raise ValueError("marker_types 为空，无法选择标记方式")
    marker_type = rng.choice(candidate_marker_types)
    color1 = OBJECT_COLORS.get("1", "red")
    color2 = OBJECT_COLORS.get("2", "green")

    marker_descriptions = []
    marker_data = {}

    if marker_type == "point_marker":
        point1 = _get_2d_center_from_rle(mask_rle1) if mask_rle1 else None
        if point1:
            marker_descriptions.append(f"In image A, a {color1} point marker indicates {label1}.")
            marker_data["image_a_point_1"] = point1
            marker_data["image_a_point_1_color"] = color1
        point2 = _get_2d_center_from_rle(mask_rle2) if mask_rle2 else None
        if point2:
            marker_descriptions.append(f"In image B, a {color2} point marker indicates {label2}.")
            marker_data["image_b_point_2"] = point2
            marker_data["image_b_point_2_color"] = color2

    elif marker_type == "white_mask":
        marker_descriptions.append(f"In image A, a separate white mask image highlights {label1}.")
        marker_descriptions.append(f"In image B, a separate white mask image highlights {label2}.")

    elif marker_type == "mask_covering_image":
        if mask_rle1:
            marker_descriptions.append(f"In image A, a {color1} mask overlay indicates {label1}.")
            marker_data["image_a_mask_rle_1"] = mask_rle1
            marker_data["image_a_mask_rle_1_color"] = color1
        if mask_rle2:
            marker_descriptions.append(f"In image B, a {color2} mask overlay indicates {label2}.")
            marker_data["image_b_mask_rle_2"] = mask_rle2
            marker_data["image_b_mask_rle_2_color"] = color2

    elif marker_type == "2d_bbox":
        bbox1 = _get_2d_bbox_from_rle(mask_rle1) if mask_rle1 else None
        if bbox1:
            marker_descriptions.append(f"In image A, a {color1} 2D bounding box indicates {label1}.")
            marker_data["image_a_bbox_1"] = bbox1
            marker_data["image_a_bbox_1_color"] = color1
        bbox2 = _get_2d_bbox_from_rle(mask_rle2) if mask_rle2 else None
        if bbox2:
            marker_descriptions.append(f"In image B, a {color2} 2D bounding box indicates {label2}.")
            marker_data["image_b_bbox_2"] = bbox2
            marker_data["image_b_bbox_2_color"] = color2

    elif marker_type in {"language_description", "language_regions"}:
        description1 = inst1.get("description")
        description2 = inst2.get("description")
        is_region1 = str(inst1.get("entity_type", "")).lower() == "region"
        is_region2 = str(inst2.get("entity_type", "")).lower() == "region"
        if is_region1:
            marker_descriptions.append(f"In the {label1} of image A.")
        else:
            marker_descriptions.append(f"In image A, the object described as \"{description1}\" is the {label1}.")
            marker_data["image_a_description_1"] = description1
        if is_region2:
            marker_descriptions.append(f"In the {label2} of image B.")
        else:
            marker_descriptions.append(f"In image B, the object described as \"{description2}\" is the {label2}.")
            marker_data["image_b_description_2"] = description2

    marker_text = "\n".join(marker_descriptions) if marker_descriptions else ""
    options_text = "\n".join([f"{k}. {v}" for k, v in sorted(mcq_options.items())])

    question_parts = []
    if marker_text:
        question_parts.append(marker_text)
    question_parts.append(f"The direction of {label1_q} relative to image A is {assumed_dir}.")
    question_parts.append(f"What is the direction of {label2_q} relative to image B?")
    question_parts.append(options_text)
    question = "\n".join(question_parts).strip()

    answer = f"{mcq_answer}. {direction_label}"

    result = {
        "scene_id": scene_id,
        "image_a": fa.file_name,
        "image_b": fb.file_name,
        "covisibility": cov,
        "threshold": threshold,
        "question_type": "object_cross_direction",
        "sub_question_type": "cross_image_assumed_dir",
        "question": question,
        "answer": answer,
        "assumed_dir_a": assumed_dir,
        "marker_type": marker_type,
        "marker_data": marker_data,
        "options": mcq_options,
        "object_1": {
            "ins_id": inst1.get("ins_id"),
            "label": label1,
            "center": _bbox_center(inst1["bounding_box"]).tolist(),
        },
        "object_2": {
            "ins_id": inst2.get("ins_id"),
            "label": label2,
            "center": center2_world.tolist(),
        },
        "direction_details": {
            "label": direction_label,
            "center_cam_a_obj1": center1_cam_a.tolist(),
            "center_cam_a_obj2": center2_cam_a.tolist(),
            "center_cam_b_obj2": center2_cam_b.tolist(),
            "proxy_forward": forward.tolist(),
            "proxy_right": right.tolist(),
            "proxy_up": up.tolist(),
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

    if marker_type == "white_mask":
        result["image_a_mask_rle_1"] = mask_rle1
        result["image_b_mask_rle_2"] = mask_rle2

    return result
