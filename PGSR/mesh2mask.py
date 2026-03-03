#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
import json
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import cv2
import open3d as o3d
from scene.app_model import AppModel
import copy
from collections import deque
from Mesh2DepthHelper import DepthRenderer,Load_ply_resource,Build_Ply_Render_Camera_Parameters_colmap,Build_Ply_Render_Camera_Parameters_default,show_depth_preview, Build_Ply_Render_Camera_Parameters_colmap_correct
import math
def render_depth_from_mesh(mesh, view, H, W):
    """
    Render depth from mesh using Open3D ray casting
    """
    # Create scene with mesh
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    
    # Get camera parameters
    fx, fy = view.Fx, view.Fy
    cx, cy = view.Cx, view.Cy
    
    # Create camera intrinsic
    intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
    
    # Create camera pose (world to camera)
    pose = np.identity(4)
    R_np = view.R.transpose(-1, -2).cpu().numpy() if hasattr(view.R, 'cpu') else view.R.transpose(-1, -2)
    T_np = view.T.cpu().numpy() if hasattr(view.T, 'cpu') else view.T
    pose[:3, :3] = R_np
    pose[:3, 3] = T_np
    
    # Convert to Open3D format (camera to world)
    pose_o3d = np.linalg.inv(pose)
    
    # Create rays
    rays = scene.create_rays_pinhole(
        intrinsic_matrix=intrinsic.intrinsic_matrix,
        extrinsic_matrix=pose_o3d,
        width_px=W,
        height_px=H
    )
    
    # Cast rays
    ans = scene.cast_rays(rays)
    
    # Get depth (distance along ray)
    depth = ans['t_hit'].numpy()
    
    # Set invalid hits (inf or nan) to 0
    depth[~np.isfinite(depth)] = 0
    
    return depth

def generate_masks(model_path, source_path, mesh_path, iteration, views, scene, gaussians, pipeline, background, 
                   app_model=None, max_depth=5.0, voxel_size=0.006):
    """
    Generate masks by comparing rendered depth with mesh depth
    """
    # Load mesh
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")
    
    print(f"Loading mesh from {mesh_path}")


    
    # Create mask output directory
    # If source_path contains dslr/nerfstudio/, replace it with mask
    if "dslr/nerfstudio" in source_path:
        # Replace dslr/nerfstudio with mask
        mask_output_dir = source_path.replace("dslr/nerfstudio", "mask")
    elif "dslr\\nerfstudio" in source_path:
        # Handle Windows path separator
        mask_output_dir = source_path.replace("dslr\\nerfstudio", "mask")
    else:
        mask_output_dir = os.path.join(source_path, "mask")
    makedirs(mask_output_dir, exist_ok=True)
    
    # Threshold for mask (4 * voxel_size as requested)
    depth_threshold = voxel_size
    
    print(f"Generating masks with depth threshold: {depth_threshold:.6f}")
    device = "cuda"
    mesh = Load_ply_resource(mesh_path,'cuda')
    renderer = DepthRenderer(device=device)
    for idx, view in enumerate(tqdm(views, desc="Generating masks")):
        # Render depth from Gaussian Splatting (same as render.py)
        out = render(view, gaussians, pipeline, background, app_model=app_model)
        rendering = out["render"].clamp(0.0, 1.0)
        _, H, W = rendering.shape
        
        depth_gs = out["plane_depth"].squeeze()
        depth_gs_np = depth_gs.detach().cpu().numpy()
        
        # Render depth from mesh
        
        # Set up rasterization configuration
        tanfovx = math.tan(view.FoVx * 0.5)
        tanfovy = math.tan(view.FoVy * 0.5)
        
        # start = time.time()

        camera_parameters = Build_Ply_Render_Camera_Parameters_colmap_correct(view.Fx, view.Fy, view.Cx, view.Cy, view.image_width, view.image_height, 0.01, 100.0, \
        view.R.transpose(-1,-2), view.T, 'cuda')

       


        depth_mesh, mask_inf = renderer.render_depth_batched(
            mesh = mesh,camera_parameters=camera_parameters,  max_tris_per_pass=1_000_000, flip_y=True,linear_depth=True
        )
        
        # Convert depth_mesh to numpy array if it's a tensor
        if isinstance(depth_mesh, torch.Tensor):
            depth_mesh_np = depth_mesh.detach().cpu().numpy()
        else:
            depth_mesh_np = np.array(depth_mesh)
        
        # Create mask: where depth difference is larger than threshold
        # Also mask out invalid depths (0 or too large)
        valid_gs = (depth_gs_np > 0) & (depth_gs_np < max_depth)
        valid_mesh = (depth_mesh_np > 0) & (depth_mesh_np < max_depth)
        valid_both = valid_gs & valid_mesh
        
        # Calculate depth difference
        depth_diff = np.abs(depth_gs_np - depth_mesh_np)
        
        # Mask: 1 for valid points with small error, 0 for invalid or large error
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[valid_both & (depth_diff <= depth_threshold)] = 255
        
        # Also mask out points where mesh has no hit but GS has depth (likely outliers)
        mask[(valid_gs & ~valid_mesh)] = 0
        
        # Save mask
        mask_filename = view.image_name + ".png"
        mask_path = os.path.join(mask_output_dir, mask_filename)
        cv2.imwrite(mask_path, mask)
        
        # Optional: save visualization of depth difference for debugging
        if idx == 0:  # Save first image as example
            depth_diff_vis = depth_diff.copy()
            depth_diff_vis[~valid_both] = 0
            depth_diff_vis = (depth_diff_vis - depth_diff_vis.min()) / (depth_diff_vis.max() - depth_diff_vis.min() + 1e-20)
            depth_diff_vis = (depth_diff_vis * 255).clip(0, 255).astype(np.uint8)
            depth_diff_color = cv2.applyColorMap(depth_diff_vis, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(mask_output_dir, view.image_name + "_depth_diff.jpg"), depth_diff_color)
    
    print(f"Masks saved to: {mask_output_dir}")

def main():
    torch.set_num_threads(8)
    # Set up command line argument parser
    parser = ArgumentParser(description="Generate masks from mesh depth comparison")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--max_depth", default=6.0, type=float)
    parser.add_argument("--voxel_size", default=0.006, type=float)
    parser.add_argument("--mesh_path", type=str, required=True, help="Path to mesh file (e.g., mesh/tsdf_fusion_post.ply)")
    # parser.add_argument("-s", "--source_path", type=str, default=None, help="Source path for saving masks (default: model_path)")

    args = get_combined_args(parser)
    print("Generating masks for " + args.model_path)
    
    # Use source_path if provided, otherwise use model_path
    if args.source_path is None:
        args.source_path = args.model_path
    else:
        args.source_path = os.path.abspath(args.source_path)
    
    # Resolve mesh path (can be relative to model_path or absolute)
    if not os.path.isabs(args.mesh_path):
        mesh_path = os.path.join(args.model_path, args.mesh_path)
    else:
        mesh_path = args.mesh_path
    
    # Initialize system state (RNG)
    safe_state(args.quiet)
    print(f"multi_view_num {model.multi_view_num}")
    
    with torch.no_grad():
        gaussians = GaussianModel(model.extract(args).sh_degree)
        scene = Scene(model.extract(args), gaussians, load_iteration=args.iteration, shuffle=False)
        # app_model = AppModel()
        # app_model.load_weights(scene.model_path)
        # app_model.eval()
        # app_model.cuda()

        dataset = model.extract(args)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        # Generate masks for train cameras
        generate_masks(
            dataset.model_path, 
            args.source_path,
            mesh_path,
            scene.loaded_iter, 
            scene.getTrainCameras(), 
            scene, 
            gaussians, 
            pipeline.extract(args), 
            background,
            app_model=None,
            max_depth=args.max_depth,
            voxel_size=args.voxel_size
        )

if __name__ == "__main__":
    main()

