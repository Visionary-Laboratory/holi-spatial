from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import os
import numpy as np
import numpy.typing as npt
import torch
from PIL import Image

PGSR_ROOT = Path(__file__).parent / "PGSR"
if str(PGSR_ROOT) not in sys.path:
    sys.path.append(str(PGSR_ROOT))

from arguments import ModelParams, PipelineParams  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from scene import Scene  # noqa: E402
from scene.gaussian_model import GaussianModel  # noqa: E402


DEFAULT_SCENE = "0a5c013435"
DEFAULT_DATA_ROOT = Path("/home/liuyifei/code/posevlm/scannetppv2/data")
DEFAULT_MASK_ROOT = Path("/home/liuyifei/code/posevlm/sam_masks_debug")
DEFAULT_OUTPUT_DIR = Path("/home/liuyifei/code/posevlm/sam_masks_debug")
DEFAULT_GS_MODEL = Path("/home/liuyifei/code/posevlm/output") / DEFAULT_SCENE
MAX_POINT_COUNT = 5_000_000


def setup_logger() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def load_transforms(scannet_dir: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    读取 nerfstudio transforms_undistorted.json，返回
      intr_map[stem] = 3x3 K
      c2w_map[stem] = 4x4 camera-to-world
    """
    json_path = scannet_dir / "dslr" / "nerfstudio" / "transforms_undistorted.json"
    with json_path.open("r", encoding="utf-8") as f:
        contents = json.load(f)

    fl_x, fl_y, cx, cy = contents["fl_x"], contents["fl_y"], contents["cx"], contents["cy"]
    train_frames = contents.get("frames", [])
    test_frames = contents.get("test_frames", [])

    intr_map: Dict[str, np.ndarray] = {}
    c2w_map: Dict[str, np.ndarray] = {}
    for frame in [*train_frames, *test_frames]:
        K = np.array([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]], dtype=np.float32)
        c2w = np.array(frame["transform_matrix"], dtype=np.float32)
        # OpenGL -> COLMAP 约定
        c2w[:3, 1:3] *= -1
        stem = Path(frame["file_path"]).stem
        intr_map[stem] = K
        c2w_map[stem] = c2w
    logging.info("加载位姿完成，数量: %d", len(intr_map))
    return intr_map, c2w_map


def load_mask(mask_path: Path) -> np.ndarray:
    mask = Image.open(mask_path).convert("L")
    return np.array(mask) > 0


def resize_mask_to_depth(mask: np.ndarray, depth: np.ndarray) -> np.ndarray:
    if mask.shape == depth.shape:
        return mask
    mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
    mask_resized = mask_img.resize((depth.shape[1], depth.shape[0]), resample=Image.NEAREST)
    return (np.array(mask_resized) > 0).astype(bool)


def mask_depth_to_points(
    mask: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
) -> np.ndarray:
    """将 mask 内像素 + 深度投影到世界坐标，返回 (N,3)。"""
    if mask.shape != depth.shape:
        mask = resize_mask_to_depth(mask, depth)

    ys, xs = np.where(mask)
    if ys.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    d = depth[ys, xs].astype(np.float32)
    valid = np.isfinite(d) & (d > 0)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)

    xs, ys, d = xs[valid], ys[valid], d[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x_cam = (xs - cx) / fx * d
    y_cam = (ys - cy) / fy * d
    z_cam = d

    ones = np.ones_like(z_cam)
    cam_pts = np.stack([x_cam, y_cam, z_cam, ones], axis=1)  # (N,4)
    world_pts = (cam_pts @ c2w.T)[:, :3]
    return world_pts.astype(np.float32)


def cluster_points(points: np.ndarray, voxel_size: float, min_points: int) -> List[np.ndarray]:
    """
    基于体素 6 邻域连通的简单聚类，返回每个实例的点索引数组。
    """
    if points.shape[0] == 0:
        return []
    vox = np.floor(points / voxel_size).astype(np.int64)
    voxel_to_indices: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
    for idx, v in enumerate(map(tuple, vox)):
        voxel_to_indices[v].append(idx)

    visited = set()
    components: List[List[int]] = []
    neighbors = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

    for voxel in voxel_to_indices:
        if voxel in visited:
            continue
        queue = deque([voxel])
        visited.add(voxel)
        comp_indices: List[int] = []
        while queue:
            v = queue.popleft()
            comp_indices.extend(voxel_to_indices[v])
            vx, vy, vz = v
            for dx, dy, dz in neighbors:
                nb = (vx + dx, vy + dy, vz + dz)
                if nb in voxel_to_indices and nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        if len(comp_indices) >= min_points:
            components.append(np.array(comp_indices, dtype=np.int32))
    return components


def bbox_corners(pts: np.ndarray) -> List[Dict[str, float]]:
    xyz_min = pts.min(axis=0)
    xyz_max = pts.max(axis=0)
    x0, y0, z0 = xyz_min.tolist()
    x1, y1, z1 = xyz_max.tolist()
    corners = [
        (x0, y0, z0),
        (x0, y1, z0),
        (x1, y1, z0),
        (x1, y0, z0),
        (x0, y0, z1),
        (x0, y1, z1),
        (x1, y1, z1),
        (x1, y0, z1),
    ]
    return [{"x": float(x), "y": float(y), "z": float(z)} for x, y, z in corners]


def depth_to_points(
    depth: npt.NDArray[np.floating],
    K: np.ndarray,
    c2w: np.ndarray,
    color: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """将整幅深度图投影到世界坐标，同时可返回颜色。

    Returns:
        pts: (N,3) float32
        cols: (N,3) uint8 或 None
    """
    H, W = depth.shape
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    d = depth.astype(np.float32).reshape(-1)
    valid = np.isfinite(d) & (d > 0)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), None
    xs = xs.reshape(-1)[valid].astype(np.float32)
    ys = ys.reshape(-1)[valid].astype(np.float32)
    d = d[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (xs - cx) / fx * d
    y_cam = (ys - cy) / fy * d
    z_cam = d
    ones = np.ones_like(z_cam)
    cam_pts = np.stack([x_cam, y_cam, z_cam, ones], axis=1)
    world_pts = (cam_pts @ c2w.T)[:, :3]
    cols = None
    if color is not None:
        col = color.reshape(-1, color.shape[-1])[valid]
        if col.dtype != np.uint8:
            col = (col * 255.0).clip(0, 255).astype(np.uint8)
        cols = col[:, :3]
    return world_pts.astype(np.float32), cols


def load_gaussian_cfg(model_path: Path) -> argparse.Namespace:
    cfg_path = model_path / "cfg_args"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg_ns = eval(f.read(), {"Namespace": argparse.Namespace})

    cfg = vars(cfg_ns).copy()
    cfg["model_path"] = str(model_path)

    repo_root = Path(__file__).parent

    def _resolve(path_str: str) -> str:
        p = Path(path_str)
        if p.is_absolute():
            return str(p)
        cand_repo = (repo_root / p).resolve()
        if os.path.exists(cand_repo):
            return str(cand_repo)
        return str((PGSR_ROOT / p).resolve())
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    if "source_path" in cfg:
        cfg["source_path"] = os.path.join(curr_dir,cfg["source_path"]) 
    if "images" in cfg:
        cfg["images"] = os.path.join(curr_dir,cfg["images"])
    return argparse.Namespace(**cfg)


def build_pipeline_defaults() -> argparse.Namespace:
    pipeline_params = PipelineParams(argparse.ArgumentParser(),)
    vals = {k.lstrip("_"): v for k, v in vars(pipeline_params).items() if not k.startswith("__")}
    return argparse.Namespace(**vals)


def render_depths_with_gaussians(model_path: Path, iteration: int) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    使用 3DGS 渲染深度与颜色，返回
      depth_map: {image_name: depth_numpy}
      color_map: {image_name: color_numpy(H,W,3) in [0,1]}
    """
    cfg = load_gaussian_cfg(model_path)
    dataset = ModelParams(argparse.ArgumentParser(), sentinel=True).extract(cfg)
    pipeline = PipelineParams(argparse.ArgumentParser()).extract(build_pipeline_defaults())

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")

    depth_map: Dict[str, np.ndarray] = {}
    color_map: Dict[str, np.ndarray] = {}
    cameras = list(scene.getTrainCameras()) + list(scene.getTestCameras())
    with torch.no_grad():
        for cam in cameras:
            out = render(cam, gaussians, pipeline, background)
            depth_map[cam.image_name] = out["plane_depth"].squeeze().detach().cpu().numpy()
            color = out["render"].permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy()
            color_map[cam.image_name] = color
    logging.info("3DGS 深度渲染完成，数量: %d", len(depth_map))
    return depth_map, color_map


def export_pointcloud_and_bboxes(
    points: np.ndarray,
    bboxes: List[Dict],
    output_dir: Path,
    scene: str,
    colors: Optional[np.ndarray] = None,
    include_labels: Optional[set[str]] = None,
) -> Path | None:
    try:
        import trimesh
    except ImportError:
        logging.warning("未安装 trimesh，跳过 glb 导出")
        return None

    if colors is None or colors.shape[0] != points.shape[0]:
        cols_rgba = np.full((points.shape[0], 4), [200, 200, 200, 255], dtype=np.uint8)
    else:
        cols_rgba = np.concatenate([colors[:, :3], np.full((colors.shape[0], 1), 255, dtype=np.uint8)], axis=1)
    pc = trimesh.points.PointCloud(points, colors=cols_rgba)
    scene_tm = trimesh.Scene()
    scene_tm.add_geometry(pc, node_name="points")

    def label_color(label: str) -> np.ndarray:
        h = abs(hash(label))
        r = 50 + (h % 206)
        g = 50 + ((h // 256) % 206)
        b = 50 + ((h // 65536) % 206)
        return np.array([r, g, b, 255], dtype=np.uint8)

    for idx, box in enumerate(bboxes):
        if include_labels is not None and box["label"] not in include_labels:
            continue
        corners = np.array([[c["x"], c["y"], c["z"]] for c in box["bounding_box"]], dtype=np.float32)
        xyz_min = corners.min(axis=0)
        xyz_max = corners.max(axis=0)
        extents = xyz_max - xyz_min
        center = (xyz_min + xyz_max) * 0.5
        # 线框 bbox
        mesh_box = trimesh.creation.box(extents=extents)
        mesh_box.apply_translation(center)
        edges = mesh_box.edges_unique
        edge_verts = mesh_box.vertices[edges]
        path = trimesh.load_path(edge_verts.reshape(-1, 3))
        col = label_color(box["label"])
        path.colors = np.tile(col[None, :], (len(path.entities), 1))
        path.metadata = {"label": box["label"]}
        scene_tm.add_geometry(path, node_name=f"bbox_{idx+1}_{box['label']}")

        # 小球标记中心，便于在查看器里看到标签颜色
        try:
            sph = trimesh.creation.icosphere(subdivisions=2, radius=max(extents) * 0.01)
            sph.apply_translation(center)
            sph.visual.vertex_colors = np.tile(col[None, :], (len(sph.vertices), 1))
            sph.metadata = {"label": box["label"]}
            scene_tm.add_geometry(sph, node_name=f"bbox_center_{idx+1}_{box['label']}")
        except Exception:
            pass

    glb_path = output_dir / f"{scene}_points_bbox.glb"
    scene_tm.export(glb_path)
    logging.info("点云+BBox glb 已保存: %s (点数: %d, bbox: %d)", glb_path, points.shape[0], len(bboxes))
    return glb_path


def process_scene(
    scene: str,
    data_root: Path,
    mask_root: Path,
    model_path: Path,
    iteration: int,
    output_dir: Path,
    voxel_size: float,
    min_points: int,
    include_labels: Optional[set[str]],
) -> Path:
    scene_dir = data_root / scene
    intr_map, c2w_map = load_transforms(scene_dir)
    depth_map, color_map = render_depths_with_gaussians(model_path, iteration)

    mask_index_path = mask_root / scene / "mask_index.json"
    with mask_index_path.open("r", encoding="utf-8") as f:
        mask_index = json.load(f)
    items = mask_index.get("items", [])

    results: List[Dict] = []
    inst_counter = 1

    all_depth_points: List[np.ndarray] = []
    all_depth_colors: List[np.ndarray] = []

    # 按标签聚合所有点
    label_points: Dict[str, List[np.ndarray]] = defaultdict(list)

    for stem, depth in depth_map.items():
        if stem not in intr_map or stem not in c2w_map:
            continue
        color = color_map.get(stem)
        pts_full, cols_full = depth_to_points(depth, intr_map[stem], c2w_map[stem], color=color)
        if pts_full.shape[0] > 0:
            all_depth_points.append(pts_full)
            if cols_full is not None:
                all_depth_colors.append(cols_full)

    for item in items:
        image_name = item["image"]
        label = item["label"]
        mask_path = Path(item["mask_path"])
        stem = Path(image_name).stem

        if stem not in intr_map or stem not in c2w_map:
            logging.warning("找不到相机参数，跳过: %s", image_name)
            continue

        if stem not in depth_map:
            logging.warning("渲染深度缺失，跳过: %s", stem)
            continue

        depth = depth_map[stem]
        mask = load_mask(mask_path)
        points = mask_depth_to_points(mask, depth, intr_map[stem], c2w_map[stem])
        if points.shape[0] == 0:
            continue
        label_points[label].append(points)

    for label, chunks in label_points.items():
        pts = np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]
        comps = cluster_points(pts, voxel_size=voxel_size, min_points=min_points)
        logging.info("标签 %s 聚类得到 %d 个实例", label, len(comps))
        for comp in comps:
            bbox = bbox_corners(pts[comp])
            results.append(
                {
                    "ins_id": str(inst_counter),
                    "label": label,
                    "bounding_box": bbox,
                }
            )
            inst_counter += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{scene}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logging.info("3D bounding boxes 已保存: %s (共 %d 个实例)", output_path, len(results))

    if all_depth_points:
        pts_all = np.concatenate(all_depth_points, axis=0)
        cols_all = None
        if all_depth_colors and len(all_depth_colors) == len(all_depth_points):
            try:
                cols_all = np.concatenate(all_depth_colors, axis=0)
            except Exception:
                cols_all = None
        orig_n = pts_all.shape[0]
        if orig_n > MAX_POINT_COUNT:
            idx = np.random.choice(orig_n, MAX_POINT_COUNT, replace=False)
            pts_all = pts_all[idx]
            if cols_all is not None and cols_all.shape[0] == orig_n:
                cols_all = cols_all[idx]
            logging.info("点云下采样: %d -> %d", orig_n, pts_all.shape[0])
        export_pointcloud_and_bboxes(
            pts_all,
            results,
            output_dir,
            scene,
            colors=cols_all,
            include_labels=include_labels,
        )

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 2D mask 投影生成 3D bounding boxes")
    parser.add_argument("--scene", default=DEFAULT_SCENE, help="场景名，如 0a5c013435")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="scannetppv2/data 根目录")
    parser.add_argument("--mask-root", type=Path, default=DEFAULT_MASK_ROOT, help="sam mask 根目录（含 mask_index.json）")
    parser.add_argument("--model-path", "-m", type=Path, default=DEFAULT_GS_MODEL, help="3DGS 模型目录（含 cfg_args）")
    parser.add_argument("--iteration", type=int, default=-1, help="加载的迭代编号，-1 表示自动最新")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="输出 3D bbox 保存目录")
    parser.add_argument("--voxel-size", type=float, default=0.05, help="体素大小用于连通聚类（米）")
    parser.add_argument("--min-points", type=int, default=50, help="实例最少点数过滤")
    parser.add_argument(
        "--bbox-labels",
        type=str,
        default="",
        help="仅导出指定类别的 bbox 到 glb，逗号分隔，空则导出全部",
    )
    return parser.parse_args()


def main() -> None:
    setup_logger()
    args = parse_args()
    process_scene(
        scene=args.scene,
        data_root=args.data_root,
        mask_root=args.mask_root,
        model_path=args.model_path,
        iteration=args.iteration,
        output_dir=args.output_dir,
        voxel_size=args.voxel_size,
        min_points=args.min_points,
        include_labels=set([s for s in args.bbox_labels.split(",") if s.strip()]) if args.bbox_labels else None,
    )


if __name__ == "__main__":
    main()

