#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replace Boxes3D (bounding boxes) in existing Rerun .rrd files using updated JSON bboxes.

Scenario:
- old_dir contains many *.rrd and (optionally) *.json
- new_json_dir contains updated *.json (same stem as the rrd)
- The .rrd boxes were originally logged from the old json; we now overwrite the Boxes3D
  at entity paths like: /instances/{label}/{ins_id}/bbox

Behavior:
- If new json exists for an rrd stem: re-save a new .rrd that preserves everything
  (Points3D, ViewCoordinates, etc.) but overwrites Boxes3D transforms/sizes.
- If new json does NOT exist: copy the original rrd and append "_failed" to filename.

Notes:
- This script uses rerun-sdk 0.24.x `rerun.dataframe` to load .rrd.
- We do NOT attempt to preserve original log_time/log_tick; we only preserve content.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

import rerun as rr
import rerun.dataframe as rr_df


def _rr_entity_path(path: str) -> str:
    """
    Convert an archive entity_path (usually starting with '/') into a string safe for rr.log.

    Rerun entity paths require escaping whitespace. We escape any whitespace characters
    so that paths like 'instances/blue machine/1/bbox' keep the exact same entity name.
    """
    if path == "/":
        return "/"
    s = path.lstrip("/")
    # Rerun uses backslash to escape the next character in entity paths.
    # We only add escapes for whitespace that isn't already escaped, and we
    # preserve existing escape sequences from the archive.
    out: list[str] = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "\\" and i + 1 < len(s):
            # Preserve existing escape sequence.
            out.append("\\")
            out.append(s[i + 1])
            i += 2
            continue
        if ch.isspace():
            out.append("\\")
            out.append(ch)
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _rr_unescape_component(s: str) -> str:
    """
    Undo rerun path escaping for a single path component.
    E.g. 'cleaning\\ supplies' -> 'cleaning supplies'
    """
    out: list[str] = []
    i = 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s):
            out.append(s[i + 1])
            i += 2
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def _stable_color_u8(key: str) -> np.ndarray:
    import hashlib

    h = hashlib.md5(key.encode("utf-8")).digest()
    rgb = np.frombuffer(h[:3], dtype=np.uint8).astype(np.uint16)
    rgb = (rgb // 2 + 64).clip(0, 255).astype(np.uint8)
    return rgb  # (3,) uint8


def _decode_packed_u32_to_rgb(packed: np.ndarray) -> np.ndarray:
    """
    rerun.dataframe stores Color components as packed uint32 of form 0xRRGGBBAA.
    On little-endian machines bytes are [AA, BB, GG, RR].
    """
    packed = np.asarray(packed, dtype=np.uint32).reshape(-1)
    # view as bytes in native endianness (little-endian here)
    u8 = packed.view(np.uint8).reshape(-1, 4)  # [AA, BB, GG, RR]
    rgb = u8[:, [3, 2, 1]].copy()
    return rgb.astype(np.uint8)


def _stack_positions_object_array(pos_obj: np.ndarray) -> np.ndarray:
    """
    Positions from rerun.dataframe often come as (N,) dtype=object with each element (3,).
    Convert to (N,3) float32.
    """
    if isinstance(pos_obj, np.ndarray) and pos_obj.dtype == object:
        if pos_obj.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        return np.stack(list(pos_obj), axis=0).astype(np.float32)
    arr = np.asarray(pos_obj)
    if arr.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr.astype(np.float32)
    # Fallback: best effort
    return np.array(arr, dtype=np.float32).reshape(-1, 3)


def _unwrap_maybe_object_array(val: Any, dtype: np.dtype) -> np.ndarray:
    """
    rerun.dataframe may wrap vectors in an object array, e.g. [array([x,y,z],dtype=float32)].
    This unwraps/stack such structures into a numeric ndarray.
    """
    arr = np.asarray(val)
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        if arr.size == 0:
            return np.empty((0,), dtype=dtype)
        arr = np.stack([np.asarray(x) for x in arr], axis=0)
    return np.asarray(arr, dtype=dtype)


@dataclass(frozen=True)
class BoxParams:
    center: np.ndarray  # (3,) float32
    half_sizes: np.ndarray  # (3,) float32
    quat_xyzw: np.ndarray  # (4,) float32


def _quat_xyzw_from_mat3(m: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix to quaternion (xyzw), numpy-only.
    Assumes m is a proper rotation (orthonormal, det=+1) or close to it.
    """
    m = np.asarray(m, dtype=np.float32).reshape(3, 3)
    tr = float(m[0, 0] + m[1, 1] + m[2, 2])
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0  # s = 4*w
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = np.sqrt(1.0 + float(m[0, 0]) - float(m[1, 1]) - float(m[2, 2])) * 2.0  # s = 4*x
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + float(m[1, 1]) - float(m[0, 0]) - float(m[2, 2])) * 2.0  # s = 4*y
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + float(m[2, 2]) - float(m[0, 0]) - float(m[1, 1])) * 2.0  # s = 4*z
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    q = np.asarray([x, y, z, w], dtype=np.float32)
    # Normalize for robustness.
    n = float(np.linalg.norm(q))
    if n > 1e-12:
        q /= n
    return q


def _box_from_json_entry(entry: Mapping[str, Any]) -> BoxParams:
    T = np.asarray(entry["obb_transform"], dtype=np.float32)
    ext = np.asarray(entry["obb_extents"], dtype=np.float32).reshape(3,)
    center = T[:3, 3].astype(np.float32)
    half = (ext * 0.5).astype(np.float32)
    quat = _quat_xyzw_from_mat3(T[:3, :3])  # xyzw
    return BoxParams(center=center, half_sizes=half, quat_xyzw=quat)


def _iter_json_instances(obj: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(obj, list):
        for it in obj:
            if isinstance(it, dict):
                yield it
        return
    if isinstance(obj, dict):
        for k in ("instances", "objects", "annotations", "data"):
            v = obj.get(k)
            if isinstance(v, list):
                for it in v:
                    if isinstance(it, dict):
                        yield it
                return
    raise ValueError("Unsupported JSON layout: expected list or dict containing a list of instances.")


def _load_new_boxes_by_key(json_path: Path) -> Dict[Tuple[str, str], BoxParams]:
    with json_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    out: Dict[Tuple[str, str], BoxParams] = {}
    for inst in _iter_json_instances(obj):
        label = str(inst.get("label", ""))
        ins_id = str(inst.get("ins_id", ""))
        if not label or not ins_id:
            continue
        if "obb_transform" not in inst or "obb_extents" not in inst:
            continue
        try:
            out[(label, ins_id)] = _box_from_json_entry(inst)
        except Exception:
            continue
    return out


def _extract_instance_key_from_entity_path(entity_path: str) -> Optional[Tuple[str, str]]:
    """
    Try to parse /.../instances/{label}/{ins_id}/bbox
    Works with optional prefixes, e.g. /world/instances/{label}/{ins_id}/bbox
    """
    pp = PurePosixPath(entity_path)
    parts = list(pp.parts)
    # Typical parts: ('/', 'instances', 'bag', '21', 'bbox')
    # If label contains '/', it becomes multiple segments:
    # ('/', 'instances', 'machine', 'pipe', '215', 'bbox') -> label='machine/pipe', ins_id='215'
    for i, p in enumerate(parts):
        if p != "instances":
            continue
        # Find the last 'bbox' segment after this 'instances'
        bbox_idx = None
        for j in range(i + 1, len(parts)):
            if parts[j] == "bbox":
                bbox_idx = j
        if bbox_idx is None:
            continue
        if bbox_idx - (i + 1) < 2:
            # need at least label + ins_id + bbox
            continue
        ins_id = str(parts[bbox_idx - 1])
        label_parts = [str(x) for x in parts[i + 1 : bbox_idx - 1]]
        if not label_parts:
            continue
        label_raw = "/".join(label_parts)
        label = _rr_unescape_component(label_raw)
        ins_id = _rr_unescape_component(ins_id)
        return (label, ins_id)
    return None


def _read_entity_df(rec: Any, entity_path: str) -> "np.ndarray | Any":
    view = rec.view(index="log_tick", contents=entity_path)
    if entity_path == "/__properties":
        try:
            return view.select_static().read_pandas()
        except Exception:
            return None
    try:
        df = view.select().read_pandas()
        if df is not None and not getattr(df, "empty", True):
            return df
    except Exception:
        pass
    # Some entities are purely static; fall back to static selection to avoid warnings/empties.
    try:
        df = view.select_static().read_pandas()
        return df
    except Exception:
        return None


def _get_last(row_df, col_name: str) -> Any:
    if col_name not in row_df.columns:
        raise KeyError(col_name)
    return row_df.iloc[-1][col_name]


def _log_points3d(entity_path: str, df) -> None:
    pos_col = f"{entity_path}:Points3D:positions"
    col_col = f"{entity_path}:Points3D:colors"
    positions_obj = _get_last(df, pos_col)
    positions = _stack_positions_object_array(positions_obj)
    colors = None
    if col_col in df.columns:
        packed = _get_last(df, col_col)
        if packed is not None:
            packed = np.asarray(packed)
            if packed.size == positions.shape[0]:
                colors = _decode_packed_u32_to_rgb(packed)
    rr.log(
        _rr_entity_path(entity_path),
        rr.Points3D(positions, colors=colors) if colors is not None else rr.Points3D(positions),
    )


def _log_view_coordinates(entity_path: str, df) -> None:
    # entity_path should be "/"
    col = f"{entity_path}:ViewCoordinates:xyz"
    xyz = _get_last(df, col)
    xyz_arr = np.asarray(xyz)
    # rerun.dataframe may give object array like: [array([3,1,6], dtype=uint8)]
    if isinstance(xyz_arr, np.ndarray) and xyz_arr.dtype == object:
        if xyz_arr.size == 1:
            xyz_arr = np.asarray(xyz_arr[0])
        else:
            xyz_arr = np.concatenate([np.asarray(x) for x in xyz_arr], axis=0)
    xyz_arr = np.asarray(xyz_arr, dtype=np.uint8).reshape(-1)
    if xyz_arr.size != 3:
        xyz_arr = xyz_arr[:3]
    rr.log("/", rr.ViewCoordinates(xyz=xyz_arr))


def _log_recording_info(entity_path: str, df) -> None:
    # This is optional; rr.init already sets recording properties. Keep best-effort.
    col = "property:RecordingInfo:start_time"
    if col not in df.columns:
        return
    start_time = _get_last(df, col)
    # There isn't a public python API to set RecordingInfo:start_time directly on save.
    # Skip silently.
    _ = start_time


def _log_boxes3d(
    entity_path: str,
    df,
    new_box: Optional[BoxParams],
    fallback_label: Optional[str],
    fallback_ins_id: Optional[str],
) -> None:
    centers_col = f"{entity_path}:Boxes3D:centers"
    half_col = f"{entity_path}:Boxes3D:half_sizes"
    quat_col = f"{entity_path}:Boxes3D:quaternions"
    colors_col = f"{entity_path}:Boxes3D:colors"
    labels_col = f"{entity_path}:Boxes3D:labels"

    cols_obj = getattr(df, "columns", None)
    df_cols = set(cols_obj) if cols_obj is not None else set()

    if new_box is None:
        if centers_col not in df_cols or half_col not in df_cols or quat_col not in df_cols:
            return
        centers = _unwrap_maybe_object_array(_get_last(df, centers_col), np.float32).reshape(-1, 3)
        half_sizes = _unwrap_maybe_object_array(_get_last(df, half_col), np.float32).reshape(-1, 3)
        quats = _unwrap_maybe_object_array(_get_last(df, quat_col), np.float32).reshape(-1, 4)
    else:
        centers = np.asarray([new_box.center], dtype=np.float32)
        half_sizes = np.asarray([new_box.half_sizes], dtype=np.float32)
        quats = np.asarray([new_box.quat_xyzw], dtype=np.float32)

    # Colors/labels: preserve from old if present; otherwise generate.
    colors_rgb = None
    if colors_col in df_cols:
        try:
            packed = np.asarray(_get_last(df, colors_col))
            if packed.size > 0:
                colors_rgb = _decode_packed_u32_to_rgb(packed)  # (N,3)
        except Exception:
            colors_rgb = None

    labels = None
    if labels_col in df_cols:
        try:
            labels = _get_last(df, labels_col)
            # labels may be list[str] or np.ndarray dtype=object
            if isinstance(labels, np.ndarray):
                labels = [str(x) for x in labels.tolist()]
            elif isinstance(labels, list):
                labels = [str(x) for x in labels]
            else:
                labels = [str(labels)]
        except Exception:
            labels = None

    if colors_rgb is None:
        key = f"{fallback_label or 'unknown'}:{fallback_ins_id or ''}"
        colors_rgb = np.asarray([_stable_color_u8(key)], dtype=np.uint8)

    if labels is None:
        if fallback_label is not None and fallback_ins_id is not None:
            labels = [f"{fallback_label}:{fallback_ins_id}"]
        else:
            labels = None

    rr.log(
        _rr_entity_path(entity_path),
        rr.Boxes3D(
            centers=centers,
            half_sizes=half_sizes,
            quaternions=[rr.Quaternion(xyzw=q.astype(np.float32)) for q in quats],
            colors=colors_rgb.astype(np.uint8),
            labels=labels,
        ),
    )


def rewrite_one_rrd(
    old_rrd_path: Path,
    new_json_path: Path,
    out_rrd_path: Path,
) -> Tuple[int, int]:
    """
    Returns: (num_boxes_overwritten, num_boxes_added)
    """
    archive = rr_df.load_archive(str(old_rrd_path))
    recs = archive.all_recordings()
    if not recs:
        raise RuntimeError(f"No recordings found in {old_rrd_path}")
    rec = recs[0]

    new_boxes = _load_new_boxes_by_key(new_json_path)
    used_new_keys: set[Tuple[str, str]] = set()
    overwritten = 0

    rr.init(str(getattr(rec, "application_id", None) or f"rrd_replace/{old_rrd_path.stem}"), spawn=False)

    schema = rec.schema()
    entity_paths = sorted({d.entity_path for d in schema.component_columns()})

    for ent in entity_paths:
        try:
            df = _read_entity_df(rec, ent)
        except Exception:
            continue
        if df is None or getattr(df, "empty", False):
            continue

        # Identify archetype by column suffix.
        cols = set(df.columns)
        if f"{ent}:Points3D:positions" in cols:
            _log_points3d(ent, df)
            continue
        if f"{ent}:Boxes3D:centers" in cols:
            key = _extract_instance_key_from_entity_path(ent)
            label = key[0] if key else None
            ins_id = key[1] if key else None
            new_box = None
            if key is not None and key in new_boxes:
                new_box = new_boxes[key]
                used_new_keys.add(key)
                overwritten += 1
            _log_boxes3d(ent, df, new_box=new_box, fallback_label=label, fallback_ins_id=ins_id)
            continue
        if ent == "/" and f"{ent}:ViewCoordinates:xyz" in cols:
            _log_view_coordinates(ent, df)
            continue
        if ent == "/__properties" and "property:RecordingInfo:start_time" in cols:
            _log_recording_info(ent, df)
            continue

        # Unknown/unsupported archetype: skip silently.

    # Add any boxes present in new json but missing from old rrd.
    added = 0
    for (label, ins_id), bp in new_boxes.items():
        if (label, ins_id) in used_new_keys:
            continue
        ent = f"/instances/{label}/{ins_id}/bbox"
        _log_boxes3d(ent, df=None, new_box=bp, fallback_label=label, fallback_ins_id=ins_id)
        added += 1

    out_rrd_path.parent.mkdir(parents=True, exist_ok=True)
    rr.save(str(out_rrd_path))
    return overwritten, added


def iter_rrd_files(input_dir: Path) -> List[Path]:
    files: List[Path] = []
    for p in input_dir.rglob("*.rrd"):
        if p.is_file():
            files.append(p)
    files.sort()
    return files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old_dir", type=Path, default=Path("output_scannetppv2_new"), help="Folder containing original .rrd files")
    ap.add_argument("--new_json_dir", type=Path, default=Path("output_scannetppv2_new_aabb"), help="Folder containing updated .json files")
    ap.add_argument("--out_dir", type=Path, default=Path("output_scannetppv2_new_rrd_aabb"), help="Output folder for new .rrd files")
    ap.add_argument("--failed_suffix", type=str, default="failed", help="Suffix appended before .rrd when json missing")
    ap.add_argument("--limit", type=int, default=0, help="Process only first N rrd files (0=all)")
    args = ap.parse_args()

    old_dir: Path = args.old_dir
    new_json_dir: Path = args.new_json_dir
    out_dir: Path = args.out_dir

    rrd_files = iter_rrd_files(old_dir)
    if args.limit and args.limit > 0:
        rrd_files = rrd_files[: args.limit]

    if not rrd_files:
        raise SystemExit(f"No .rrd files found under: {old_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    failed = 0
    overwritten_total = 0
    added_total = 0

    for old_rrd in rrd_files:
        stem = old_rrd.stem
        new_json = new_json_dir / f"{stem}.json"
        rel = old_rrd.relative_to(old_dir)
        out_rrd = out_dir / rel

        if not new_json.exists():
            failed += 1
            out_rrd_failed = out_rrd.with_name(f"{out_rrd.stem}_{args.failed_suffix}{out_rrd.suffix}")
            out_rrd_failed.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old_rrd, out_rrd_failed)
            print(f"[FAILED_JSON_MISSING] {old_rrd} -> {out_rrd_failed}")
            continue

        try:
            overwritten, added = rewrite_one_rrd(old_rrd, new_json, out_rrd)
        except Exception as e:
            failed += 1
            out_rrd_failed = out_rrd.with_name(f"{out_rrd.stem}_{args.failed_suffix}{out_rrd.suffix}")
            out_rrd_failed.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old_rrd, out_rrd_failed)
            print(f"[FAILED_REWRITE] {old_rrd} -> {out_rrd_failed} ({type(e).__name__}: {e})")
            continue

        ok += 1
        overwritten_total += overwritten
        added_total += added
        print(f"[OK] {old_rrd} -> {out_rrd} (overwrite={overwritten}, add={added})")

    print(
        f"Done. ok={ok}, failed={failed}, overwritten_total={overwritten_total}, added_total={added_total}, out_dir={out_dir}"
    )


if __name__ == "__main__":
    main()

