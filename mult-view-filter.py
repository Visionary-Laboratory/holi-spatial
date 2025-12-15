from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm
import cv2
import sys

PGSR_ROOT = Path(__file__).parent / "PGSR"
if str(PGSR_ROOT) not in sys.path:
    sys.path.append(str(PGSR_ROOT))

from arguments import ModelParams, PipelineParams  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from scene import Scene  # noqa: E402
from scene.gaussian_model import GaussianModel  # noqa: E402
from utils.general_utils import safe_state  # noqa: E402


DEFAULT_SCENE = "0a5c013435"
DEFAULT_MODEL_PATH = Path("/home/liuyifei/code/posevlm/output") / DEFAULT_SCENE
DEFAULT_MASK_OUTPUT = Path("/home/liuyifei/code/posevlm/output") / DEFAULT_SCENE / "mv_masks"


def setup_logger() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_multi_view(model_path: Path) -> Dict[str, List[str]]:
    mv_path = model_path / "multi_view.json"
    mapping: Dict[str, List[str]] = {}
    if not mv_path.exists():
        logging.warning("multi_view.json 未找到: %s", mv_path)
        return mapping
    with mv_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            ref = data.get("ref_name") or data.get("ref") or data.get("ref_name" "")
            nearest = data.get("nearest_name") or []
            if ref:
                mapping[ref] = nearest
    logging.info("multi_view.json 加载完成，条目: %d", len(mapping))
    return mapping


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
        cfg["source_path"] = os.path.join(curr_dir, cfg["source_path"])
    if "images" in cfg:
        cfg["images"] = os.path.join(curr_dir, cfg["images"])
    return argparse.Namespace(**cfg)


def build_pipeline_defaults() -> argparse.Namespace:
    pipeline_params = PipelineParams(argparse.ArgumentParser())
    vals = {k.lstrip("_"): v for k, v in vars(pipeline_params).items() if not k.startswith("__")}
    return argparse.Namespace(**vals)


def render_depths_with_gaussians(model_path: Path, iteration: int) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Scene]:
    cfg = load_gaussian_cfg(model_path)
    dataset = ModelParams(argparse.ArgumentParser(), sentinel=True).extract(cfg)
    pipeline = PipelineParams(argparse.ArgumentParser()).extract(build_pipeline_defaults())

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")

    depth_map: Dict[str, np.ndarray] = {}
    color_map: Dict[str, np.ndarray] = {}
    cameras = list(scene.getTrainCameras())
    with torch.no_grad():
        for cam in tqdm(cameras, desc="render depth"):
            out = render(cam, gaussians, pipeline, background)
            depth_map[cam.image_name] = out["plane_depth"].squeeze().detach().cpu().numpy()
            color = out["render"].permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy()
            color_map[cam.image_name] = color
    logging.info("深度渲染完成，数量: %d", len(depth_map))
    return depth_map, color_map, scene


def get_points_from_depth(cam, depth: np.ndarray, scale: int = 1) -> torch.Tensor:
    st = int(max(int(scale / 2) - 1, 0))
    depth_view = depth.squeeze()[st::scale, st::scale]
    rays_d = cam.get_rays(scale=scale)
    depth_view = depth_view[: rays_d.shape[0], : rays_d.shape[1]]
    pts = (rays_d * torch.from_numpy(depth_view).to(rays_d.device)[..., None]).reshape(-1, 3)
    R = torch.tensor(cam.R).float().to(rays_d.device)
    T = torch.tensor(cam.T).float().to(rays_d.device)
    pts = (pts - T) @ R.transpose(-1, -2)
    return pts


def filter_masks(depth_map: Dict[str, np.ndarray], scene: Scene, multi_view: Dict[str, List[str]], pixel_thred: float, output_dir: Path) -> None:
    cams = scene.getTrainCameras()
    name_to_cam = {c.image_name: c for c in cams}
    os.makedirs(output_dir, exist_ok=True)

    depth_tensors = {k: torch.from_numpy(v).float().cuda() for k, v in depth_map.items()}

    for view in tqdm(cams, desc="multi-view filter"):
        if view.image_name not in multi_view:
            continue
        neighbors = [n for n in multi_view[view.image_name] if n in name_to_cam and n in depth_tensors]
        if len(neighbors) == 0:
            continue

        ref_depth = depth_tensors[view.image_name]
        H, W = ref_depth.shape
        ix, iy = torch.meshgrid(torch.arange(W, device="cuda"), torch.arange(H, device="cuda"), indexing="xy")
        pixels = torch.stack([ix, iy], dim=-1).float()

        pts = get_points_from_depth(view, ref_depth.cpu().numpy()).cuda()
        mask_accum = torch.zeros((H, W), dtype=torch.int32, device="cuda")

        for nb_name in neighbors:
            nb_cam = name_to_cam[nb_name]
            nb_depth = depth_tensors[nb_name]
            nb_W, nb_H = nb_cam.image_width, nb_cam.image_height

            w2c_nb = torch.tensor(nb_cam.world_view_transform.cpu().numpy(), device="cuda").float()
            pts_cam_nb = pts @ w2c_nb[:3, :3].T + w2c_nb[3, :3]
            z = pts_cam_nb[:, 2]
            valid = z > 0.1
            if not torch.any(valid):
                continue
            pts_cam_nb = pts_cam_nb[valid]
            z = z[valid]
            u = (nb_cam.Fx * pts_cam_nb[:, 0] / z + nb_cam.Cx)
            v = (nb_cam.Fy * pts_cam_nb[:, 1] / z + nb_cam.Cy)
            u_int = u.long()
            v_int = v.long()
            inside = (u_int >= 0) & (u_int < nb_W) & (v_int >= 0) & (v_int < nb_H)
            if not torch.any(inside):
                continue
            u = u[inside]
            v = v[inside]
            z = z[inside]
            depth_nb_vals = nb_depth[v_int[inside], u_int[inside]].cuda()
            # 仅使用像素重投影偏移过滤
            proj_u = (view.Fx * pts_cam_nb[inside][:, 0] / z + view.Cx)
            proj_v = (view.Fy * pts_cam_nb[inside][:, 1] / z + view.Cy)
            proj_u = proj_u.clamp(0, W - 1)
            proj_v = proj_v.clamp(0, H - 1)
            du = (proj_u - pixels[..., 0].reshape(-1)[valid][inside]).abs()
            dv = (proj_v - pixels[..., 1].reshape(-1)[valid][inside]).abs()
            ok = (du <= pixel_thred) & (dv <= pixel_thred)
            proj_u = proj_u[ok].long()
            proj_v = proj_v[ok].long()
            mask_accum[proj_v, proj_u] += 1

        mask_final = (mask_accum > 0).cpu().numpy().astype(np.uint8)
        cv2.imwrite(str(output_dir / f"{view.image_name}.png"), mask_final * 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="multi-view depth consistency filter (Gaussian)")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="3DGS 模型目录 (包含 cfg_args, multi_view.json)")
    parser.add_argument("--iteration", type=int, default=-1, help="加载迭代，-1 表示最新")
    parser.add_argument("--pixel-thred", type=float, default=5.0, help="像素重投影阈值")
    parser.add_argument("--mask-output", type=Path, default=DEFAULT_MASK_OUTPUT, help="输出一致性 mask 目录")
    return parser.parse_args()


def main() -> None:
    setup_logger()
    args = parse_args()
    safe_state(False)

    depth_map, color_map, scene = render_depths_with_gaussians(args.model_path, args.iteration)
    multi_view = load_multi_view(args.model_path)
    filter_masks(depth_map, scene, multi_view, args.pixel_thred, args.mask_output)
    logging.info("完成，多视图一致性 mask 已输出到: %s", args.mask_output)


if __name__ == "__main__":
    main()
