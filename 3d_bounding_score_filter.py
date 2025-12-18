#!/usr/bin/env python3
"""
读取 output_yifei_dl3dv/<scene>.json 中的 3D instance，
依据 mask_dl3dv/<scene>/mask_index.json 中的 score 过滤：
- 每个 instance 的 images 字段列出其来源的 mask 图片路径
- 对应的 mask_index.json 已记录每个 mask 的 score
- 只要任一关联 mask 的 score >= 阈值（默认 0.85），则保留该实例
- 输出 <scene>_filter.json
- 若安装了 rerun，可额外生成仅包含过滤后 bbox 的 <scene>_new.rrd
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按 mask 置信度过滤 3D bbox")
    parser.add_argument(
        "--input-json",
        type=Path,
        required=True,
        help="原始 3D bbox json 路径，例如 output_yifei_dl3dv/<scene>.json",
    )
    parser.add_argument(
        "--mask-root",
        type=Path,
        default=Path("mask_dl3dv"),
        help="mask 根目录（内含 <scene>/mask_index.json）",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="判定有效的最低 mask 置信度",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="过滤后的 json 输出路径，默认 <input>_filter.json",
    )
    parser.add_argument(
        "--output-rrd",
        type=Path,
        default=None,
        help="过滤后 rrd 输出路径，默认 <input>_new.rrd；需安装 rerun",
    )
    return parser.parse_args()


def parse_image_info(image_path: str) -> Tuple[str, str]:
    """
    从 mask 路径解析 scene、相对 mask 路径（frame_xxxxx/label_nn.png）。
    预期形如：sam_dl3dv/<scene>/frame_xxxxx/<label>_NN.png
    """
    p = Path(image_path)
    parts = p.parts
    if "sam_dl3dv" not in parts:
        raise ValueError(f"路径缺少 sam_dl3dv: {image_path}")
    base_idx = parts.index("sam_dl3dv")
    try:
        scene = parts[base_idx + 1]
        frame = parts[base_idx + 2]
    except IndexError as exc:  # noqa: BLE001
        raise ValueError(f"路径缺少 scene/frame: {image_path}") from exc
    rel_mask = f"{frame}/{p.name}"
    return scene, rel_mask


def load_mask_index_scores(mask_root: Path, scene: str) -> Dict[str, float]:
    """
    读取 mask_index.json 中每个 mask 的 score。
    返回键为相对路径 (frame_xxxxx/label_nn.png)，值为 score。
    """
    mask_index_path = mask_root / scene / "mask_index.json"
    if not mask_index_path.exists():
        logging.info("mask_index.json 不存在，跳过：%s", mask_index_path)
        return {}
    try:
        with mask_index_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        items = data.get("items", [])
        result = {}
        for item in items:
            path = item.get("mask_path")
            score = item.get("score")
            if not isinstance(path, str) or not isinstance(score, (int, float)):
                continue
            p = Path(path)
            rel = f"{p.parent.name}/{p.name}"
            result[rel] = float(score)
        logging.info("从 mask_index.json 读取到 %d 条记录", len(result))
        return result
    except Exception as exc:  # noqa: BLE001
        logging.warning("读取 mask_index.json 失败: %s", exc)
        return {}


def instance_has_high_score(
    images: Iterable[str],
    threshold: float,
    scores_map: Dict[str, float],
) -> bool:
    for img in images:
        try:
            _, rel = parse_image_info(img)
        except ValueError as exc:
            logging.warning("跳过无法解析的图片路径 %s: %s", img, exc)
            continue
        score = scores_map.get(rel)
        if score is not None and score >= threshold:
            return True
    return False


def rotation_matrix_to_quaternion_xyzw(mat: "np.ndarray") -> "np.ndarray":
    """3x3 旋转矩阵转 xyzw 四元数，避免依赖 scipy。"""
    import numpy as np

    m = mat
    t = np.trace(m)
    if t > 0.0:
        s = (t + 1.0) ** 0.5 * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = (1.0 + m[0, 0] - m[1, 1] - m[2, 2]) ** 0.5 * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = (1.0 + m[1, 1] - m[0, 0] - m[2, 2]) ** 0.5 * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = (1.0 + m[2, 2] - m[0, 0] - m[1, 1]) ** 0.5 * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    return np.array([qx, qy, qz, qw], dtype=np.float32)


def label_color(label: str) -> Tuple[int, int, int]:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return digest[0], digest[1], digest[2]


def export_rerun_rrd(instances: List[dict], scene: str, output_rrd: Path) -> bool:
    """使用 rerun 只导出过滤后的 bbox。"""
    try:
        import numpy as np
        import rerun as rr
    except Exception as exc:  # noqa: BLE001
        logging.warning("未安装 rerun，跳过 rrd 生成：%s", exc)
        return False

    output_rrd.parent.mkdir(parents=True, exist_ok=True)
    rr.init(f"filtered_bbox/{scene}", spawn=False)
    try:
        rr.save(str(output_rrd))
    except Exception as exc:  # noqa: BLE001
        logging.warning("创建 rrd 失败（仍继续尝试写入日志）: %s", exc)

    for inst in instances:
        try:
            transform = np.array(inst["obb_transform"], dtype=np.float32)
            extents = np.array(inst["obb_extents"], dtype=np.float32) * 0.5
            center = transform[:3, 3]
            rot = transform[:3, :3]
            quat = rotation_matrix_to_quaternion_xyzw(rot)
            col = np.array(label_color(inst.get("label", "")), dtype=np.uint8)

            rr.log(
                f"instances/{inst.get('label', 'unknown')}/{inst.get('ins_id', '0')}/bbox",
                rr.Boxes3D(
                    centers=np.array([center], dtype=np.float32),
                    half_sizes=np.array([extents], dtype=np.float32),
                    quaternions=[rr.Quaternion(xyzw=quat)],
                    colors=np.array([col], dtype=np.uint8),
                    labels=[f"{inst.get('label', 'unknown')}:{inst.get('ins_id', '0')}"],
                ),
            )
        except Exception as exc:  # noqa: BLE001
            logging.warning("记录实例 %s 失败: %s", inst.get("ins_id"), exc)

    logging.info("过滤后 bbox 已写入 rrd: %s", output_rrd)
    return True


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parse_args()

    input_json = args.input_json
    if not input_json.exists():
        raise FileNotFoundError(f"输入 json 不存在: {input_json}")

    output_json = (
        args.output_json
        if args.output_json is not None
        else input_json.with_name(input_json.stem + "_filter.json")
    )
    output_rrd = (
        args.output_rrd
        if args.output_rrd is not None
        else input_json.with_name(input_json.stem + "_new.rrd")
    )

    with input_json.open("r", encoding="utf-8") as f:
        instances = json.load(f)
    if not isinstance(instances, list):
        raise ValueError("输入 json 顶层应为列表")

    scene_name = input_json.stem
    mask_scores = load_mask_index_scores(args.mask_root, scene_name)

    kept: List[dict] = []
    for inst in instances:
        images = inst.get("images", [])
        if not images:
            logging.debug("实例 %s 无 images，丢弃", inst.get("ins_id"))
            continue
        if instance_has_high_score(images, args.threshold, mask_scores):
            kept.append(inst)

    logging.info("共 %d 个实例，保留 %d 个（阈值 %.2f）", len(instances), len(kept), args.threshold)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)
    logging.info("已写出过滤结果: %s", output_json)

    if kept:
        export_rerun_rrd(kept, scene=scene_name, output_rrd=output_rrd)
    else:
        logging.warning("无实例通过阈值，跳过 rrd 生成。")


if __name__ == "__main__":
    main()
