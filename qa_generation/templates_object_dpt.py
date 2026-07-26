import json
import math
import textwrap
import random
from pathlib import Path
from typing import Dict, Sequence, Tuple, TYPE_CHECKING, List, Any, Optional, Iterable, Set

import numpy as np
from pycocotools import mask as mask_utils

if TYPE_CHECKING:
    # Avoid runtime circular import; only for type hints.
    from qa_generation.generate_two_view_qa import FrameItem

# 标记方式
MARKER_TYPES = ["point_marker", "white_mask", "mask_covering_image", "2d_bbox", "language_description"]


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


def _direction_info_from_cam(
    point_cam: Sequence[float],
    ratio_threshold: float = 0.5774,
    max_axes: int = 2,
) -> Optional[Dict[str, Any]]:
    x = float(point_cam[0])
    y = float(point_cam[1])
    z = float(point_cam[2])
    comps = {"x": x, "y": y, "z": z}
    order = sorted(comps.items(), key=lambda kv: abs(kv[1]), reverse=True)
    if not order:
        return None
    top_mag = abs(order[0][1])
    if top_mag <= 1e-6:
        return None

    def axis_word(axis: str, value: float) -> str:
        if axis == "z":
            return "Front" if value > 0 else "Back"  # camera looks toward +Z
        if axis == "x":
            return "Right" if value > 0 else "Left"
        if axis == "y":
            return "Down" if value > 0 else "Up"
        return "Unknown"

    parts = []
    for axis, val in order:
        if abs(val) >= ratio_threshold * top_mag:
            parts.append(axis_word(axis, val))
        if len(parts) >= max_axes:
            break

    if not parts:
        return None

    dominant_axis = order[0][0]
    direction_label = "-".join(parts)
    return {
        "direction_label": direction_label,
        "dominant_axis": dominant_axis,
        "components_cam": comps,
    }


def _build_direction_mc_options(direction: str, rng: random.Random) -> Tuple[Dict[str, str], str]:
    opposites = {
        "Front": "Back",
        "Back": "Front",
        "Left": "Right",
        "Right": "Left",
        "Up": "Down",
        "Down": "Up",
    }

    def opposite(word: str) -> str:
        return opposites.get(word, "Unknown")

    parts = direction.split("-") if direction else []
    wrong_candidates = []

    if len(parts) == 1:
        base = parts[0] if parts else "Unknown"
        alt_map = {
            "Left": ["Right", "Up", "Down"],
            "Right": ["Left", "Up", "Down"],
            "Front": ["Back", "Left", "Right"],
            "Back": ["Front", "Left", "Right"],
            "Up": ["Down", "Left", "Right"],
            "Down": ["Up", "Left", "Right"],
        }
        wrong_candidates = alt_map.get(base, ["Right", "Up", "Down"])
        correct_text = base
    elif len(parts) == 2:
        w1, w2 = parts
        wrong_candidates = [
            f"{w1}-{opposite(w2)}",
            f"{opposite(w1)}-{w2}",
            f"{opposite(w1)}-{opposite(w2)}",
        ]
        correct_text = direction
    else:
        wrong_candidates = ["Front", "Back", "Left"]
        correct_text = direction or "Unknown"

    seen = set()
    wrong_final = []
    for item in wrong_candidates:
        if item != correct_text and item not in seen:
            wrong_final.append(item)
            seen.add(item)
        if len(wrong_final) == 3:
            break

    fallback_pool = ["Front", "Back", "Left", "Right", "Up", "Down"]
    for item in fallback_pool:
        if len(wrong_final) == 3:
            break
        if item != correct_text and item not in seen:
            wrong_final.append(item)
            seen.add(item)

    options_list = wrong_final + [correct_text]
    rng.shuffle(options_list)
    letters = ["A", "B", "C", "D"]
    options = {letter: text for letter, text in zip(letters, options_list)}
    correct_letter = next(k for k, v in options.items() if v == correct_text)
    return options, correct_letter


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
    rng: random.Random = None,
    marker_types: Optional[List[str]] = None,
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
        if image_a_stem not in mask_map:
            continue
        mask_path_b = mask_map.get(image_b_stem)

        center_world = _bbox_center(inst["bounding_box"])
        center_cam_b = _world_to_cam(center_world, fb.transform_matrix)
        dist_cam_b = float(np.linalg.norm(center_cam_b))
        proj_uv = _project_cam(center_cam_b, intrinsics)

        shared_instances.append(
            {
                "ins_id": str(inst.get("ins_id", "")),
                "label": inst.get("label", ""),
                "mask_path_a": mask_map[image_a_stem],
                "mask_path_b": mask_path_b,
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
    mask_path_a = target["mask_path_a"]
    
    if rng is None:
        rng = random.Random()
    
    # 从 bbox_items 的 images 字段推断 mask_index.json 的位置
    images_for_inference = []
    for inst in bbox_items:
        images_for_inference.extend(inst.get("images", []))
        if images_for_inference:
            break
    
    # 加载 mask_index.json 并查找 RLE
    mask_index = _load_mask_index(scene_id, images_for_inference)
    mask_rle_a = _find_mask_rle_by_path(mask_index, mask_path_a) if mask_index else None
    if mask_rle_a is None:
        # fallback: mask_index.json 可能缺失 mask_rle 字段
        target_inst = next((it for it in bbox_items if str(it.get("ins_id", "")) == target["ins_id"]), None)
        if target_inst is not None:
            mask_rle_a = _find_mask_rle_from_instance_encodings(target_inst, mask_path_a)
    
    # 随机选择标记方式（可由外部指定）
    candidate_marker_types = marker_types or MARKER_TYPES
    if not candidate_marker_types:
        raise ValueError("marker_types 为空，无法选择标记方式")
    marker_type = rng.choice(candidate_marker_types)
    color = "red"  # 单个物体使用红色
    
    # 根据标记方式构建说明文本
    marker_descriptions = []
    marker_data = {}
    
    if marker_type == "point_marker":
        point_a = _get_2d_center_from_rle(mask_rle_a) if mask_rle_a else None
        if point_a:
            marker_descriptions.append(f"In image A, a {color} point marker indicates {label}.")
            marker_data["image_a_point"] = point_a
            marker_data["image_a_point_color"] = color
    
    elif marker_type == "white_mask":
        marker_descriptions.append(f"In image A, a separate white mask image highlights {label}.")
    
    elif marker_type == "mask_covering_image":
        if mask_rle_a:
            marker_descriptions.append(f"In image A, a {color} mask overlay indicates {label}.")
            marker_data["image_a_mask_rle"] = mask_rle_a
            marker_data["image_a_mask_rle_color"] = color
    
    elif marker_type == "2d_bbox":
        bbox_a = _get_2d_bbox_from_rle(mask_rle_a) if mask_rle_a else None
        if bbox_a:
            marker_descriptions.append(f"In image A, a {color} 2D bounding box indicates {label}.")
            marker_data["image_a_bbox"] = bbox_a
            marker_data["image_a_bbox_color"] = color
    
    elif marker_type == "language_description":
        # 从 bbox_items 中查找对应的 instance 的描述，若缺失则直接报错
        description = None
        for inst in bbox_items:
            if str(inst.get("ins_id", "")) == target["ins_id"]:
                description = inst.get("description")
                break
        
        if not description:
            raise ValueError(f"language_description 缺少描述，scene_id={scene_id}, ins_id={target['ins_id']}")
        
        marker_descriptions.append(f"In image A, the object described as \"{description}\" is the {label}.")
        marker_data["image_a_description"] = description
    
    marker_text = "\n".join(marker_descriptions) if marker_descriptions else ""

    question = textwrap.dedent(
        f"""
        {marker_text}
        Locate the same physical object in image B.
        Estimate the 3D metric distance (in meters)
        from the camera position of image B (camera center)
        to the "{label}" (to the object surface/center point).
        This is NOT pixel distance to the image center.
        Return only one number in meters (e.g., 0.7).
        Output format: <answer>NUMBER</answer>.
        """
    ).strip()
    answer = f"{target['distance_cam_b']:.1f} m"

    result = {
        "scene_id": scene_id,
        "image_a": fa.file_name,
        "image_b": fb.file_name,
        "covisibility": cov,
        "threshold": threshold,
        "question_type": "object_distance",
        "sub_question_type": "bbox_center_distance",
        "question": question,
        "answer": answer,
        "selected_ins_id": target["ins_id"],
        "marker_type": marker_type,
        "marker_data": marker_data,
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
    
    # 只有 white_mask 时才返回单独的 mask_rle（用于单独显示）
    if marker_type == "white_mask":
        result["image_a_mask_rle"] = mask_rle_a
    
    return result


def build_object_dir_entry(
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
    rng: random.Random = None,
    marker_types: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    基于 bbox json 与 mask 路径构造“给定 A 视角的标记，问物体在 B 视角相对方向”QA。
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
            }
        )

    if not shared_instances:
        return None

    if target_ins_id is not None:
        target = next((x for x in shared_instances if x["ins_id"] == str(target_ins_id)), shared_instances[0])
    else:
        target = shared_instances[0]
    label = target["label"] or "object"
    mask_path_a = target["mask_path_a"]

    direction_info = _direction_info_from_cam(target["center_cam_b"])
    if direction_info is None:
        return None
    direction_label = direction_info["direction_label"]

    if rng is None:
        rng = random.Random()
    mcq_options, mcq_answer = _build_direction_mc_options(direction_label, rng)

    images_for_inference = []
    for inst in bbox_items:
        images_for_inference.extend(inst.get("images", []))
        if images_for_inference:
            break

    mask_index = _load_mask_index(scene_id, images_for_inference)
    mask_rle_a = _find_mask_rle_by_path(mask_index, mask_path_a) if mask_index else None
    if mask_rle_a is None:
        target_inst = next((it for it in bbox_items if str(it.get("ins_id", "")) == target["ins_id"]), None)
        if target_inst is not None:
            mask_rle_a = _find_mask_rle_from_instance_encodings(target_inst, mask_path_a)

    candidate_marker_types = marker_types or MARKER_TYPES
    if not candidate_marker_types:
        raise ValueError("marker_types 为空，无法选择标记方式")
    marker_type = rng.choice(candidate_marker_types)
    color = "red"

    marker_descriptions = []
    marker_data = {}

    if marker_type == "point_marker":
        point_a = _get_2d_center_from_rle(mask_rle_a) if mask_rle_a else None
        if point_a:
            marker_descriptions.append(f"In image A, a {color} point marker indicates {label}.")
            marker_data["image_a_point"] = point_a
            marker_data["image_a_point_color"] = color

    elif marker_type == "white_mask":
        marker_descriptions.append(f"In image A, a separate white mask image highlights {label}.")

    elif marker_type == "mask_covering_image":
        if mask_rle_a:
            marker_descriptions.append(f"In image A, a {color} mask overlay indicates {label}.")
            marker_data["image_a_mask_rle"] = mask_rle_a
            marker_data["image_a_mask_rle_color"] = color

    elif marker_type == "2d_bbox":
        bbox_a = _get_2d_bbox_from_rle(mask_rle_a) if mask_rle_a else None
        if bbox_a:
            marker_descriptions.append(f"In image A, a {color} 2D bounding box indicates {label}.")
            marker_data["image_a_bbox"] = bbox_a
            marker_data["image_a_bbox_color"] = color

    elif marker_type == "language_description":
        description = None
        for inst in bbox_items:
            if str(inst.get("ins_id", "")) == target["ins_id"]:
                description = inst.get("description")
                break
        if not description:
            raise ValueError(f"language_description 缺少描述，scene_id={scene_id}, ins_id={target['ins_id']}")
        marker_descriptions.append(f"In image A, the object described as \"{description}\" is the {label}.")
        marker_data["image_a_description"] = description

    marker_text = "\n".join(marker_descriptions) if marker_descriptions else ""
    options_text = "\n".join([f"{k}. {v}" for k, v in sorted(mcq_options.items())])

    question_parts = []
    if marker_text:
        question_parts.append(marker_text)
    question_templates = [
        "Which direction is the \"{label}\" relative to you when taking image B?",
        "When you were taking the photo in Image B, where is the {label} area relative to you?",
    ]
    question_parts.append(rng.choice(question_templates).format(label=label))
    question_parts.append(options_text)
    question = "\n".join(question_parts).strip()

    answer = f"{mcq_answer}. {direction_label}"

    result = {
        "scene_id": scene_id,
        "image_a": fa.file_name,
        "image_b": fb.file_name,
        "covisibility": cov,
        "threshold": threshold,
        "question_type": "object_direction",
        "sub_question_type": "bbox_center_direction",
        "question": question,
        "answer": answer,
        "selected_ins_id": target["ins_id"],
        "marker_type": marker_type,
        "marker_data": marker_data,
        "options": mcq_options,
        "objects": shared_instances,
        "direction_details": {
            "label": direction_label,
            "center_cam_b": target["center_cam_b"],
            "dominant_axis": direction_info["dominant_axis"],
            "components_cam": direction_info["components_cam"],
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
        result["image_a_mask_rle"] = mask_rle_a

    return result
