#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


SH_C0 = 0.28209479177387814


@dataclass(frozen=True)
class PlyHeader:
    fmt: Literal["binary_little_endian"]
    vertex_count: int
    vertex_properties: list[str]
    header_bytes: int


def _parse_ply_header(ply_path: Path) -> PlyHeader:
    with ply_path.open("rb") as f:
        line = f.readline().decode("ascii", "ignore").strip()
        if line != "ply":
            raise ValueError(f"Not a PLY file: {ply_path}")

        fmt: str | None = None
        vertex_count: int | None = None
        in_vertex_element = False
        vertex_properties: list[str] = []

        while True:
            raw = f.readline()
            if not raw:
                raise ValueError(f"Unexpected EOF before end_header: {ply_path}")
            line = raw.decode("ascii", "ignore").strip()

            if line.startswith("format "):
                # e.g. "format binary_little_endian 1.0"
                parts = line.split()
                if len(parts) < 3:
                    raise ValueError(f"Malformed format line in {ply_path}: {line}")
                fmt = parts[1]
                if fmt != "binary_little_endian":
                    raise ValueError(f"Unsupported PLY format {fmt} in {ply_path}")
                in_vertex_element = False
                continue

            if line.startswith("element "):
                # e.g. "element vertex 5942054"
                parts = line.split()
                if len(parts) != 3:
                    raise ValueError(f"Malformed element line in {ply_path}: {line}")
                elem_name, count_str = parts[1], parts[2]
                in_vertex_element = elem_name == "vertex"
                if in_vertex_element:
                    vertex_count = int(count_str)
                continue

            if line.startswith("property "):
                if not in_vertex_element:
                    continue
                parts = line.split()
                if len(parts) != 3:
                    raise ValueError(f"Malformed property line in {ply_path}: {line}")
                prop_type, prop_name = parts[1], parts[2]
                if prop_type != "float":
                    raise ValueError(
                        f"Unsupported vertex property type {prop_type} for {prop_name} in {ply_path}"
                    )
                vertex_properties.append(prop_name)
                continue

            if line == "end_header":
                header_bytes = f.tell()
                break

        if fmt is None or vertex_count is None or not vertex_properties:
            raise ValueError(f"Incomplete PLY header: {ply_path}")

        return PlyHeader(
            fmt="binary_little_endian",
            vertex_count=vertex_count,
            vertex_properties=vertex_properties,
            header_bytes=header_bytes,
        )


def _sample_indices(
    n: int, *, max_points: int, sample: Literal["stride", "random"], seed: int
) -> np.ndarray:
    if max_points <= 0 or max_points >= n:
        return np.arange(n, dtype=np.int64)

    if sample == "stride":
        step = max(1, n // max_points)
        return np.arange(0, n, step, dtype=np.int64)[:max_points]

    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_points, replace=False).astype(np.int64))


def load_gaussian_point_cloud(
    ply_path: Path,
    *,
    max_points: int = 300_000,
    sample: Literal["stride", "random"] = "stride",
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    header = _parse_ply_header(ply_path)
    prop_names = header.vertex_properties
    n_props = len(prop_names)

    required = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"]
    missing = [p for p in required if p not in prop_names]
    if missing:
        raise ValueError(f"Missing properties {missing} in {ply_path}")

    idx_x = prop_names.index("x")
    idx_y = prop_names.index("y")
    idx_z = prop_names.index("z")
    idx_dc0 = prop_names.index("f_dc_0")
    idx_dc1 = prop_names.index("f_dc_1")
    idx_dc2 = prop_names.index("f_dc_2")

    data = np.memmap(
        ply_path,
        dtype="<f4",
        mode="r",
        offset=header.header_bytes,
        shape=(header.vertex_count, n_props),
    )

    indices = _sample_indices(header.vertex_count, max_points=max_points, sample=sample, seed=seed)
    positions = np.empty((indices.shape[0], 3), dtype=np.float32)
    positions[:, 0] = data[indices, idx_x]
    positions[:, 1] = data[indices, idx_y]
    positions[:, 2] = data[indices, idx_z]

    f_dc = np.empty((indices.shape[0], 3), dtype=np.float32)
    f_dc[:, 0] = data[indices, idx_dc0]
    f_dc[:, 1] = data[indices, idx_dc1]
    f_dc[:, 2] = data[indices, idx_dc2]

    rgb = np.clip(0.5 + SH_C0 * f_dc, 0.0, 1.0)
    colors = (rgb * 255.0).astype(np.uint8)
    return positions, colors


def _bbox_edges_from_vertices(vertices: list[list[float]]) -> np.ndarray:
    v = np.asarray(vertices, dtype=np.float32)
    if v.shape != (8, 3):
        raise ValueError(f"Expected 8x3 vertices, got {v.shape}")

    diff = v[:, None, :] - v[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)

    edges: set[tuple[int, int]] = set()
    for i in range(8):
        nn = np.argsort(dist2[i])[1:4]  # 3 nearest neighbors (excluding itself)
        for j in nn:
            a, b = (i, int(j)) if i < int(j) else (int(j), i)
            edges.add((a, b))

    edges_sorted = sorted(edges)
    segments = np.stack([np.stack([v[i], v[j]], axis=0) for i, j in edges_sorted], axis=0)
    return segments


def _stable_color_u8(key: str) -> np.ndarray:
    h = hashlib.md5(key.encode("utf-8")).digest()
    # avoid too-dark colors
    rgb = np.frombuffer(h[:3], dtype=np.uint8).astype(np.uint16)
    rgb = (rgb // 2 + 64).clip(0, 255).astype(np.uint8)
    return rgb


def _sanitize_entity_name(name: str) -> str:
    name = name.strip().replace(os.sep, "_").replace("/", "_").replace("\\", "_")
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in name)


def visualize_scene_to_rrd(
    *,
    scene: str,
    bbox_json_path: Path,
    gs_ply_path: Path,
    out_rrd_path: Path,
    max_points: int,
    sample: Literal["stride", "random"],
    seed: int,
    overwrite: bool,
) -> None:
    if out_rrd_path.exists() and not overwrite:
        logging.info("Skip existing: %s", out_rrd_path)
        return

    try:
        import rerun as rr
    except ImportError as exc:
        raise RuntimeError("rerun is not installed") from exc

    with bbox_json_path.open("r", encoding="utf-8") as f:
        bbox_data = json.load(f)
    objects = bbox_data.get("objects", [])

    logging.info("Loading point cloud: %s", gs_ply_path)
    points, colors = load_gaussian_point_cloud(
        gs_ply_path, max_points=max_points, sample=sample, seed=seed
    )

    rr.init(f"dl3dv_eval_gt/{scene}", spawn=False)
    rr.log("gaussians/points", rr.Points3D(points, colors=colors))

    for idx, obj in enumerate(objects):
        name = str(obj.get("name", "unknown"))
        verts = obj.get("vertices")
        if not isinstance(verts, list):
            continue
        try:
            segments = _bbox_edges_from_vertices(verts)
        except Exception as exc:  # noqa: BLE001
            logging.warning("Skip bbox %s[%d] due to invalid vertices: %s", name, idx, exc)
            continue

        ent = f"bboxes/{idx:03d}_{_sanitize_entity_name(name)}"
        col = _stable_color_u8(f"{name}:{idx}")
        try:
            rr.log(ent, rr.LineStrips3D(segments, colors=col))
        except Exception:  # noqa: BLE001
            rr.log(ent, rr.LineStrips3D(segments))

    out_rrd_path.parent.mkdir(parents=True, exist_ok=True)
    rr.save(str(out_rrd_path))
    logging.info("Saved: %s", out_rrd_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize DL3DV eval GT bboxes + GS point cloud to .rrd")
    parser.add_argument("--bbox-dir", type=Path, default=Path("3dBounding_dl3dv_eval_gt"))
    parser.add_argument("--gs-root", type=Path, default=Path("output_DL3DV/1K"))
    parser.add_argument("--out-dir", type=Path, default=Path("output_rerun_dl3dv_eval_gt"))
    parser.add_argument("--scene", type=str, default=None, help="Process only one scene id (json stem)")
    parser.add_argument("--max-points", type=int, default=4_000_000)
    parser.add_argument("--sample", type=str, choices=["stride", "random"], default="stride")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    bbox_dir: Path = args.bbox_dir
    if args.scene:
        json_paths = [bbox_dir / f"{args.scene}.json"]
    else:
        json_paths = sorted(
            p for p in bbox_dir.glob("*.json") if p.name != "_classes.json"
        )

    if not json_paths:
        raise SystemExit(f"No bbox json found in {bbox_dir}")

    for json_path in json_paths:
        scene = json_path.stem
        gs_ply_path = args.gs_root / scene / "point_cloud/iteration_30000/point_cloud.ply"
        if not gs_ply_path.exists():
            logging.warning("Missing point cloud, skip scene %s: %s", scene, gs_ply_path)
            continue
        out_rrd_path = args.out_dir / f"{scene}.rrd"
        visualize_scene_to_rrd(
            scene=scene,
            bbox_json_path=json_path,
            gs_ply_path=gs_ply_path,
            out_rrd_path=out_rrd_path,
            max_points=args.max_points,
            sample=args.sample,
            seed=args.seed,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()

