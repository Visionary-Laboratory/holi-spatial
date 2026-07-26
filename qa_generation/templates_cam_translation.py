from typing import Dict, Sequence, Tuple, TYPE_CHECKING
import itertools
import math
import random
import numpy as np

if TYPE_CHECKING:
    # Avoid runtime circular import; only for type hints.
    from qa_generation.generate_two_view_qa import FrameItem


def build_cam_translation_main_dir_entry(
    scene_id: str,
    intrinsics: Dict[str, float],
    frames: Sequence["FrameItem"],
    pair: Tuple[int, int, float],
    threshold: float,
):
    """Main-direction template for camera translation QA."""
    i, j, cov = pair
    fa, fb = frames[i], frames[j]
    info = _translation_info(frames, pair)
    direction = info["direction_label"]
    options, correct_letter = _build_direction_mc_options(direction, random)
    options_text = "\n".join([f"{k}) {v}" for k, v in options.items()])
    return {
        "scene_id": scene_id,
        "image_a": fa.file_name,
        "image_b": fb.file_name,
        "covisibility": cov,
        "threshold": threshold,
        "question_type": "cam_translation",
        "sub_question_type": "main_dir",
        "question": (
            "What is the primary camera motion direction from view A to view B in view A's coordinate?\n"
            f"{options_text}\n"
            "Reply with only the option letter (A/B/C/D)."
        ),
        "answer": correct_letter,
        "options": options,
        "camera_a": {
            "intrinsics": intrinsics,
            "transform_matrix": fa.transform_matrix,
        },
        "camera_b": {
            "intrinsics": intrinsics,
            "transform_matrix": fb.transform_matrix,
        },
        "translation_details": info,
    }


def build_cam_translation_distance_exact_entry(
    scene_id: str,
    intrinsics: Dict[str, float],
    frames: Sequence["FrameItem"],
    pair: Tuple[int, int, float],
    threshold: float,
):
    """Exact distance question (numeric answer, meters)."""
    info = _translation_info(frames, pair)
    i, j, cov = pair
    fa, fb = frames[i], frames[j]
    dist_m = info["translation_cam_norm"]
    return {
        "scene_id": scene_id,
        "image_a": fa.file_name,
        "image_b": fb.file_name,
        "covisibility": cov,
        "threshold": threshold,
        "question_type": "cam_translation",
        "sub_question_type": "distance_exact",
        "question": "What is the camera translation distance from view A to view B (meters)?",
        "answer": f"{dist_m:.3f}",
        "camera_a": {
            "intrinsics": intrinsics,
            "transform_matrix": fa.transform_matrix,
        },
        "camera_b": {
            "intrinsics": intrinsics,
            "transform_matrix": fb.transform_matrix,
        },
        "translation_details": info,
    }


def build_cam_translation_distance_threshold_entry(
    scene_id: str,
    intrinsics: Dict[str, float],
    frames: Sequence["FrameItem"],
    pair: Tuple[int, int, float],
    threshold: float,
    rng: random.Random,
):
    """Thresholded distance yes/no question with auto/semi-random threshold near the true distance."""
    info = _translation_info(frames, pair)
    i, j, cov = pair
    fa, fb = frames[i], frames[j]
    dist_m = info["translation_cam_norm"]

    # Choose a factor not too close to 1 (avoid trivial), not too far (avoid 0 or huge).
    factor = rng.uniform(0.6, 1.4)
    if 0.85 < factor < 1.15:  # push away from near equality
        factor = 1.15 if factor >= 1.0 else 0.85
    dist_threshold_m = max(0.05, dist_m * factor)

    passed = dist_m > dist_threshold_m
    return {
        "scene_id": scene_id,
        "image_a": fa.file_name,
        "image_b": fb.file_name,
        "covisibility": cov,
        "threshold": threshold,
        "question_type": "cam_translation",
        "sub_question_type": "distance_threshold",
        "question": (
            "Using ONLY the provided camera transform matrices, decide whether the camera translation "
            f"from view A to view B exceeds {dist_threshold_m:.2f} meters. "
            "Compute the translation distance as the Euclidean norm of the translation vector between "
            "the two camera centers (in meters). Return ONLY one token: 'yes' or 'no'. "
            "Outputformat: <answer>yes</answer> or <answer>no</answer> (no extra text)."
        ),
        "answer": "yes" if passed else "no",
        "camera_a": {
            "intrinsics": intrinsics,
            "transform_matrix": fa.transform_matrix,
        },
        "camera_b": {
            "intrinsics": intrinsics,
            "transform_matrix": fb.transform_matrix,
        },
        "translation_details": {**info, "distance_threshold_m": dist_threshold_m},
    }


# Helpers
def _translation_info(frames: Sequence["FrameItem"], pair: Tuple[int, int, float]):
    i, j, cov = pair
    fa, fb = frames[i], frames[j]
    # world translation (c2w assumed)
    Ta = np.array(fa.transform_matrix, dtype=float)
    Tb = np.array(fb.transform_matrix, dtype=float)
    delta_w = Tb[:3, 3] - Ta[:3, 3]
    dx_w, dy_w, dz_w = delta_w.tolist()
    norm_w = math.sqrt(dx_w * dx_w + dy_w * dy_w + dz_w * dz_w)

    # camera-A coordinates: delta_cam = R_a^T * delta_w (R_a from c2w)
    R_a = Ta[:3, :3]
    delta_cam = R_a.T @ delta_w
    dx, dy, dz = delta_cam.tolist()
    norm_cam = math.sqrt(dx * dx + dy * dy + dz * dz)
    axes = ["x", "y", "z"]
    dominant_idx = max(range(3), key=lambda k: [abs(dx), abs(dy), abs(dz)][k])
    dominant_axis = axes[dominant_idx]

    def axis_word(axis: str, value: float) -> str:
        if axis == "z":
            return "forward" if value > 0 else "backward"  # camera looks toward +Z
        if axis == "x":
            return "right" if value > 0 else "left"
        if axis == "y":
            return "down" if value > 0 else "up"
        return "unknown"

    comps = {"x": dx, "y": dy, "z": dz}
    order = sorted(comps.items(), key=lambda kv: abs(kv[1]), reverse=True)
    
    top_mag = abs(order[0][1]) if order else 0.0
    parts = []
    for axis, val in order:
        if top_mag == 0:
            break
        # keep components that are reasonably large relative to the dominant one
        if abs(val) >= 0.5774 * top_mag:
            parts.append(axis_word(axis, val))
    direction_label = "-".join(parts) if parts else "stationary"

    return {
        "translation_world": [dx_w, dy_w, dz_w],
        "translation_world_norm": norm_w,
        "translation_cam": [dx, dy, dz],
        "translation_cam_norm": norm_cam,
        "dominant_axis": dominant_axis,
        "primary_direction": axis_word(dominant_axis, comps[dominant_axis]),
        "direction_label": direction_label,
        "covisibility": cov,
    }


def _build_direction_mc_options(direction: str, rng: random.Random):
    """构造单选选项，随机放置正确答案。"""
    opposites = {
        "forward": "backward",
        "backward": "forward",
        "left": "right",
        "right": "left",
        "up": "down",
        "down": "up",
    }

    def opposite(word: str) -> str:
        return opposites.get(word, "unknown")

    parts = direction.split("-") if direction else []
    wrong_candidates = []

    if len(parts) == 1:
        base = parts[0] if parts else "unknown"
        alt_map = {
            "left": ["right", "up", "down"],
            "right": ["left", "up", "down"],
            "forward": ["backward", "left", "right"],
            "backward": ["forward", "left", "right"],
            "up": ["down", "left", "right"],
            "down": ["up", "left", "right"],
        }
        wrong_candidates = alt_map.get(base, ["right", "up", "down"])
        correct_text = base
    elif len(parts) == 2:
        w1, w2 = parts
        wrong_candidates = [
            f"{w1}-{opposite(w2)}",
            f"{opposite(w1)}-{w2}",
            f"{opposite(w1)}-{opposite(w2)}",
        ]
        correct_text = direction
    elif len(parts) >= 3:
        combos = ["-".join(c) for c in itertools.combinations(parts, 2)]
        rng.shuffle(combos)
        wrong_candidates = combos[:3]
        correct_text = direction
    else:
        wrong_candidates = ["forward", "backward", "left"]
        correct_text = direction or "unknown"

    # 去重并填充到 3 个错误项
    seen = set()
    wrong_final = []
    for item in wrong_candidates:
        if item != correct_text and item not in seen:
            wrong_final.append(item)
            seen.add(item)
        if len(wrong_final) == 3:
            break

    fallback_pool = ["forward", "backward", "left", "right", "up", "down"]
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
