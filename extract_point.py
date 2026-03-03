from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import rerun.dataframe as rr_df


def get_instance_points(rec, label: str, ins_id: str) -> Optional[np.ndarray]:
    """
    从 RRD 记录中读取某个实例的点云。

    在 3d_bounding_instance_gs_rerun*.py 中，点云是以：
        f"instances/{label}/{ins_id}/points"
    的路径写入 rerun 的。
    这里参考 3d_2d.py 的实现，同时尝试带/不带前导斜杠的路径。
    """
    # 先尝试带前导斜杠
    entity_path = f"/instances/{label}/{ins_id}/points"
    try:
        view = rec.view(index="log_time", contents=entity_path)
        data = view.select()
        df = data.read_pandas()
        if df.empty:
            # 再尝试不带前导斜杠
            entity_path = f"instances/{label}/{ins_id}/points"
            view = rec.view(index="log_time", contents=entity_path)
            data = view.select()
            df = data.read_pandas()
            if df.empty:
                return None

        # 找到包含 positions 的列
        col = [c for c in df.columns if "positions" in c]
        if not col:
            return None
        val = df[col[0]].iloc[-1]

        if isinstance(val, np.ndarray) and val.dtype == object:
            # 数组的数组，stack 一下
            pts = np.stack(val)
        else:
            pts = np.array(val)

        if pts.ndim == 1:
            if pts.size % 3 == 0 and pts.size > 0:
                pts = pts.reshape(-1, 3)
            elif pts.size == 3:
                pts = pts.reshape(1, 3)

        if pts.ndim != 2 or pts.shape[1] != 3:
            return None
        return pts.astype(np.float32)
    except Exception as e:  # noqa: BLE001
        print(f"Error reading points for {label}_{ins_id}: {e}")
        return None


def save_points_as_ply(points: np.ndarray, out_path: Path) -> None:
    """
    将 (N,3) 点云保存为 ASCII PLY（仅 xyz）。
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = points.shape[0]
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n")
        for x, y, z in points:
            f.write(f"{float(x)} {float(y)} {float(z)}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 RRD 中按 ins_id 提取实例点云，并保存为 PLY。"
    )
    parser.add_argument(
        "rrd",
        type=Path,
        help="输入 RRD 路径，例如 output_3d_bounding_scannet_evaluation_new/0a7cc12c0e.rrd",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="对应的 JSON 路径（包含 ins_id 和 label）。默认与 RRD 同名但后缀为 .json。",
    )
    parser.add_argument(
        "--ins-ids",
        nargs="+",
        required=True,
        help="需要提取的 ins_id 列表，可以指定多个。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./extracted_points"),
        help="输出 PLY 保存目录。",
    )

    args = parser.parse_args()

    rrd_path: Path = args.rrd
    if not rrd_path.exists():
        print(f"RRD 不存在: {rrd_path}")
        return

    json_path: Path
    if args.json is not None:
        json_path = args.json
    else:
        json_path = rrd_path.with_suffix(".json")
    if not json_path.exists():
        print(f"JSON 不存在: {json_path}")
        return

    ins_ids: List[str] = [str(x) for x in args.ins_ids]
    out_dir: Path = args.output_dir

    print(f"加载 JSON: {json_path}")
    with json_path.open("r", encoding="utf-8") as f:
        instances: List[Dict] = json.load(f)

    # 构建 ins_id -> label 映射
    ins_to_label: Dict[str, str] = {}
    for inst in instances:
        ins = str(inst.get("ins_id", ""))
        lbl = inst.get("label", "")
        if ins:
            ins_to_label[ins] = lbl

    missing: List[str] = [ins for ins in ins_ids if ins not in ins_to_label]
    if missing:
        print(f"警告：以下 ins_id 在 JSON 中未找到，将跳过: {missing}")

    print(f"加载 RRD: {rrd_path}")
    try:
        archive = rr_df.load_archive(str(rrd_path))
        rec = archive.all_recordings()[0]
    except Exception as e:  # noqa: BLE001
        print(f"加载 RRD 失败: {e}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有 ins_id 的点云
    all_points: List[np.ndarray] = []
    valid_ins_ids: List[str] = []
    
    for ins_id in ins_ids:
        if ins_id not in ins_to_label:
            continue
        label = ins_to_label[ins_id]
        print(f"提取实例 points: ins_id={ins_id}, label={label}")
        pts = get_instance_points(rec, label, ins_id)
        if pts is None or pts.size == 0:
            print(f"  未找到点云或为空，跳过 ins_id={ins_id}")
            continue
        all_points.append(pts)
        valid_ins_ids.append(ins_id)
        print(f"  已读取: ins_id={ins_id}, N={pts.shape[0]}")
    
    if not all_points:
        print("没有找到任何点云，退出")
        return
    
    # 合并所有点云
    merged_points = np.vstack(all_points)
    print(f"合并后总点数: {merged_points.shape[0]}")
    
    # 生成输出文件名：基于所有 ins_id
    ins_str = "_".join(valid_ins_ids)
    out_path = out_dir / f"{rrd_path.stem}_ins_{ins_str}.ply"
    save_points_as_ply(merged_points, out_path)
    print(f"已保存合并的点云: {out_path} (N={merged_points.shape[0]})")


if __name__ == "__main__":
    main()


