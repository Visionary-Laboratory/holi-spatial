from typing import Dict, Sequence, Tuple, TYPE_CHECKING, Optional
import numpy as np
from scipy.spatial.transform import Rotation as R
import random
import hashlib

if TYPE_CHECKING:  # Only for type hints to avoid runtime import cycles.
    from qa_generation.generate_two_view_qa import FrameItem


def _stable_seed(*parts: str) -> int:
    s = "|".join(parts).encode("utf-8")
    h = hashlib.md5(s).hexdigest()
    return int(h[:8], 16)


def _wrap180(deg: float) -> float:
    # Not strictly needed for this MCQ, but useful if you later add yaw sector logic.
    return (deg + 180.0) % 360.0 - 180.0


def build_cam_rotation_entry(
    scene_id: str,
    intrinsics: Dict[str, float],
    frames: Sequence["FrameItem"],
    pair: Tuple[int, int, float],
    threshold: float,
    *,
    # --- Rule knobs (feel free to tune) ---
    min_deg: float = 5.0,          # skip if the largest axis magnitude is below this
    dual_ratio: float = 0.5,       # dual if a2/a1 >= dual_ratio (plus buffer)
    ratio_buffer: float = 0.1,     # grey zone around dual_ratio -> skip
    dual_min_deg: float = 5.0,     # a2 also must be >= this to qualify as dual
) -> Optional[dict]:
    """
    Camera rotation MCQ:
      - Keep your verified R_rel and YXZ as_euler.
      - Choose two most significant axes among yaw/pitch/roll.
      - Decide single_dominant vs dual_dominant.
      - Generate ABCD choices accordingly (no "cannot determine"/"no rotation").
      - Record yaw/pitch/roll and meta (difficulty, axis_pair, mode, etc.) in JSON.

    Returns None if skipped by rules.
    """
    i, j, cov = pair
    fa, fb = frames[i], frames[j]

    # c2w rotations (unchanged)
    Ri2w_w = np.array(fa.transform_matrix, dtype=float)[:3, :3]
    Rj2w_w = np.array(fb.transform_matrix, dtype=float)[:3, :3]

    # relative rotation (UNCHANGED, per your request)
    R_rel = Ri2w_w.T @ Rj2w_w

    # Euler extraction (UNCHANGED, per your request)
    yaw_deg, pitch_deg, roll_deg = R.from_matrix(R_rel).as_euler("YXZ", degrees=True)

    # --- axis selection: drop the least significant, keep top-2 ---
    ay, ap, ar = abs(float(yaw_deg)), abs(float(pitch_deg)), abs(float(roll_deg))
    axes = [("yaw", ay), ("pitch", ap), ("roll", ar)]
    axes_sorted = sorted(axes, key=lambda t: t[1], reverse=True)

    (axis1, a1), (axis2, a2) = axes_sorted[0], axes_sorted[1]
    axis_drop = axes_sorted[2][0]

    # Skip if overall too small (unstable / not meaningful)
    if a1 < min_deg:
        return None

    # Decide single vs dual with grey-zone skipping
    ratio = a2 / (a1 + 1e-9)
    dual_hi = dual_ratio + ratio_buffer
    dual_lo = dual_ratio - ratio_buffer

    if a2 < dual_min_deg:
        mode = "single_dominant"
    else:
        if ratio >= dual_hi:
            mode = "dual_dominant"
        elif ratio <= dual_lo:
            mode = "single_dominant"
        else:
            # grey zone: avoid borderline cases that flip label easily
            return None

    # Difficulty buckets (stored in JSON)
    if a1 >= 30.0:
        difficulty = "easy"
    elif a1 >= 10.0:
        difficulty = "medium"
    else:
        difficulty = "hard"  # since a1 >= min_deg

    # Helpers: map sign to direction phrases (your verified conventions)
    def axis_dir_word(axis_name: str, val_deg: float) -> str:
        if axis_name == "yaw":
            return "turn right" if val_deg > 0 else "turn left"
        if axis_name == "pitch":
            return "tilt up" if val_deg > 0 else "tilt down"
        if axis_name == "roll":
            return "clockwise roll" if val_deg > 0 else "counter-clockwise roll"
        raise ValueError(f"Unknown axis: {axis_name}")

    def axis_two_options(axis_name: str) -> Tuple[str, str]:
        # (+) option then (-) option wording, consistent with axis_dir_word
        if axis_name == "yaw":
            return ("turn right", "turn left")
        if axis_name == "pitch":
            return ("tilt up", "tilt down")
        if axis_name == "roll":
            return ("clockwise roll", "counter-clockwise roll")
        raise ValueError(f"Unknown axis: {axis_name}")

    # Get axis values
    val_map = {"yaw": float(yaw_deg), "pitch": float(pitch_deg), "roll": float(roll_deg)}
    a1_val = val_map[axis1]
    a2_val = val_map[axis2]

    # Build choices
    seed = _stable_seed(scene_id, str(i), str(j), "camrot")
    rng = random.Random(seed)

    letters = ["A", "B", "C", "D"]

    if mode == "dual_dominant":
        # 2x2 combinations (axis1 dir, axis2 dir) => 4 options
        a1_pos, a1_neg = axis_two_options(axis1)
        a2_pos, a2_neg = axis_two_options(axis2)

        combos = [
            (a1_pos, a2_pos),
            (a1_pos, a2_neg),
            (a1_neg, a2_pos),
            (a1_neg, a2_neg),
        ]

        def combo_to_text(c):
            # If roll is included, phrasing still works:
            # "turn left and counter-clockwise roll"
            return f"{c[0]} and {c[1]}"

        options_list = [combo_to_text(c) for c in combos]
        truth = combo_to_text((axis_dir_word(axis1, a1_val), axis_dir_word(axis2, a2_val)))

        rng.shuffle(options_list)
        options = {letters[k]: options_list[k] for k in range(4)}
        answer = next(L for L, txt in options.items() if txt == truth)

        question = (
            "Given view A and view B, consider the relative rotation from A to B expressed in view A’s camera frame. "
            f"Which option best describes the rotation direction using TWO components: {axis1} and {axis2}? "
            "Choose exactly one option (A/B/C/D)."
        )

    else:
        # single_dominant: four single-direction options from axis1 and axis2 (2 + 2)
        # correct is ALWAYS axis1 direction (since axis1 is the dominant axis)
        a1_pos, a1_neg = axis_two_options(axis1)
        a2_pos, a2_neg = axis_two_options(axis2)

        options_list = [a1_pos, a1_neg, a2_pos, a2_neg]
        truth = axis_dir_word(axis1, a1_val)

        rng.shuffle(options_list)
        options = {letters[k]: options_list[k] for k in range(4)}
        answer = next(L for L, txt in options.items() if txt == truth)

        question = (
            "Given view A and view B, consider the relative rotation from A to B expressed in view A’s camera frame. "
            "Which SINGLE rotation direction is the most prominent? "
            f"(This question focuses on the dominant axis among {axis1} and {axis2}.) "
            "Choose exactly one option (A/B/C/D)."
        )

    rotation_details = {
        "R_rel_camA_to_camB": R_rel.tolist(),
        # record raw degrees (as you requested)
        "yaw_deg_left_right": float(yaw_deg),     # + => right
        "pitch_deg_up_down": float(pitch_deg),    # + => up
        "roll_deg_ccw_cw": float(roll_deg),       # + => CW
    }

    rotation_meta = {
        "axis_pair": f"{axis1}+{axis2}",
        "axis_drop": axis_drop,
        "mode": mode,  # single_dominant | dual_dominant
        "a1_deg": float(a1),
        "a2_deg": float(a2),
        "ratio_a2_a1": float(ratio),
        "difficulty": difficulty,
        "rules": {
            "min_deg": float(min_deg),
            "dual_ratio": float(dual_ratio),
            "ratio_buffer": float(ratio_buffer),
            "dual_min_deg": float(dual_min_deg),
        },
    }

    return {
        "scene_id": scene_id,
        "image_a": fa.file_name,
        "image_b": fb.file_name,
        "covisibility": cov,
        "threshold": threshold,
        "question_type": "cam_rotation_mcq",
        "sub_question_type": f"{rotation_meta['axis_pair']}_{mode}",
        "difficulty": difficulty,
        "question": question,
        "options": options,   # A/B/C/D -> text
        "answer": answer,     # "A" / "B" / "C" / "D"
        "camera_a": {
            "intrinsics": intrinsics,
            "transform_matrix": fa.transform_matrix,
        },
        "camera_b": {
            "intrinsics": intrinsics,
            "transform_matrix": fb.transform_matrix,
        },
        "rotation_details": rotation_details,
        "rotation_meta": rotation_meta,
    }
