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

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from tqdm import tqdm

class CameraInfo(NamedTuple):
    uid: int
    global_id: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    fx: float
    fy: float

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def load_poses(pose_path, num):
    poses = []
    with open(pose_path, "r") as f:
        lines = f.readlines()
    for i in range(num):
        line = lines[i]
        c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
        c2w[:3,3] = c2w[:3,3] * 10.0
        w2c = np.linalg.inv(c2w)
        w2c = w2c
        poses.append(w2c)
    poses = np.stack(poses, axis=0)
    return poses

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]

        cam_info = CameraInfo(uid=uid, global_id=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                              image_path=image_path, image_name=image_name, 
                              width=width, height=height, fx=focal_length_x, fy=focal_length_y)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    try:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    except:
        colors = np.random.rand(positions.shape[0], positions.shape[1])
    try:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    except:
        normals = np.random.rand(positions.shape[0], positions.shape[1])
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    # cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : int(x.image_name.split('_')[-1]))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    
    js_file = f"{path}/split.json"
    train_list = None
    test_list = None
    if os.path.exists(js_file):
        with open(js_file) as file:
            meta = json.load(file)
            train_list = meta["train"]
            test_list = meta["test"]
            print(f"train_list {len(train_list)}, test_list {len(test_list)}")

    if train_list is not None:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in train_list]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in test_list]
        print(f"train_cam_infos {len(train_cam_infos)}, test_cam_infos {len(test_cam_infos)}")
    elif eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/points3D.ply")
    bin_path = os.path.join(path, "sparse/points3D.bin")
    txt_path = os.path.join(path, "sparse/points3D.txt")
    if not os.path.exists(ply_path) or True:
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
            print(f"xyz {xyz.shape}")
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, images=None, extension=".png", test=False):
    cam_infos = []
    reading_dir = "images" if images == None else images
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        try:
            fovx = contents["camera_angle_x"]
        except:
            fovx = None
        if test == True:
            frames = contents["test_frames"]
        else:
            frames = contents["frames"]
        fl_x = contents["fl_x"]
        fl_y = contents["fl_y"]

        for idx, frame in enumerate(frames):
            if len(frames) > 2000:
                if idx%2 == 0:
                    continue

            cam_name = os.path.join(reading_dir, frame["file_path"])

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = cam_name
            image_name = Path(cam_name).stem

            try:
                image = Image.open(image_path)
            except:
                print(f'{image_path} not exit')
                continue

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")


            if fovx is not None:
                fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
                FovY = fovy 
                FovX = fovx
            else:
                # given focal in pixel unit
                FovY = focal2fov(fl_y, image.size[1])
                FovX = focal2fov(fl_x, image.size[0])

            cam_infos.append(CameraInfo(uid=idx, global_id=idx, R=R, T=T, FovY=FovY, FovX=FovX, fx=fl_x, fy=fl_y,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, ply_path=None, images=None, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_undistorted.json", white_background, images, extension)
    # print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_undistorted.json", white_background, images, extension,test = True)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if ply_path == None:
        print("ply_path is None, make sure this is eval")
        ply_path = "output_scannet/scannetppv2/0b031f3119/input.ply"

    # ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readDL3DVCameras(path, images_folder="rgb"):
    """
    Read cameras from DL3DV format
    DL3DV structure:
    - dense/cam/ directory: contains .npz files with intrinsics and pose
    - dense/rgb/ or dense/images/ directory: contains image files
    """
    cam_infos = []
    
    # DL3DV data is in dense/ subdirectory
    dense_path = os.path.join(path, "dense")
    data_root = dense_path  # Store data root for later use
    cam_dir = os.path.join(dense_path, "cam")
    image_dir = os.path.join(dense_path, images_folder)

    if not os.path.exists(cam_dir):
        raise ValueError(f"DL3DV cam directory not found: {cam_dir}")
    if not os.path.exists(image_dir):
        raise ValueError(f"DL3DV image directory not found: {image_dir}")
    
    # Get list of camera files
    cam_files = sorted([f for f in os.listdir(cam_dir) if f.endswith('.npz')])
    
    for idx, cam_file in tqdm(
        enumerate(cam_files),
        total=len(cam_files),
    ):
        cam_file_path = os.path.join(cam_dir, cam_file)
        image_name = cam_file.replace('.npz', '')
        
        # Try to find image file
        image_file = os.path.join(image_dir, f"{image_name}.jpg")
        if not os.path.exists(image_file):
            image_file = os.path.join(image_dir, f"{image_name}.png")
        if not os.path.exists(image_file):
            print(f"Warning: Image not found for camera {cam_file}, skipping")
            continue
        
        try:
            # Load camera parameters from .npz file
            cam_data = np.load(cam_file_path)
            
            # Extract intrinsics - DL3DV uses 'intrinsic' (singular) and it's a 3x3 matrix
            if 'intrinsic' in cam_data:
                intrinsic = cam_data['intrinsic']
                fx = intrinsic[0, 0]
                fy = intrinsic[1, 1]
                cx = intrinsic[0, 2]
                cy = intrinsic[1, 2]
            elif 'intrinsics' in cam_data:
                # Fallback for ScanNet-like format
                intrinsics = cam_data['intrinsics']
                fx = intrinsics[0, 0]
                fy = intrinsics[1, 1]
                cx = intrinsics[0, 2]
                cy = intrinsics[1, 2]
            else:
                raise ValueError(f"No intrinsics found in {cam_file}. Available keys: {list(cam_data.keys())}")
            
            # Load image to get dimensions
            image = Image.open(image_file)
            width, height = image.size
            
            # Convert focal length to FoV
            FovX = focal2fov(fx, width)
            FovY = focal2fov(fy, height)
            
            # Extract extrinsics
            # 'pose' is the camera-to-world matrix (C2W)
            c2w = cam_data['pose']
            
            # Convert to world-to-camera (W2C)
            w2c = np.linalg.inv(c2w)
            
            # Extract R and T
            # R should be transposed for the format expected by Camera class
            R = np.transpose(w2c[:3, :3])
            T = w2c[:3, 3]
            
            cam_info = CameraInfo(
                uid=idx,
                global_id=idx,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image_path=image_file,
                image_name=image_name,
                width=width,
                height=height,
                fx=fx,
                fy=fy,
            )
            
            # Release memory
            image.close()
            image = None
            
            cam_infos.append(cam_info)
            
        except Exception as e:
            print(f"Error loading camera {cam_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return cam_infos

def _get_scannet_scene_part_paths(path):
    scene_path = os.path.abspath(path)
    scene_name = os.path.basename(os.path.normpath(scene_path))
    scene_root = os.path.dirname(scene_path)

    if not os.path.isdir(scene_path):
        raise ValueError(f"ScanNet scene directory not found: {scene_path}")

    if "_" not in scene_name:
        return [scene_path]

    scene_prefix = scene_name.rsplit("_", 1)[0]
    part_paths = []
    for candidate_name in sorted(os.listdir(scene_root)):
        candidate_path = os.path.join(scene_root, candidate_name)
        if not os.path.isdir(candidate_path) or "_" not in candidate_name:
            continue

        candidate_prefix, candidate_suffix = candidate_name.rsplit("_", 1)
        if candidate_prefix != scene_prefix or not candidate_suffix.isdigit():
            continue

        if (
            os.path.exists(os.path.join(candidate_path, "pose"))
            and os.path.exists(os.path.join(candidate_path, "intrinsic", "intrinsic_color.txt"))
            and os.path.exists(os.path.join(candidate_path, "color"))
        ):
            part_paths.append(candidate_path)

    return part_paths or [scene_path]

def _find_scannet_ply_path(path):
    pointcloud_da3_path = os.path.join(path, "pointcloud_da3.ply")
    if os.path.exists(pointcloud_da3_path):
        return pointcloud_da3_path

    ply_files = [f for f in os.listdir(path) if f.endswith('_vh_clean_2.ply')]
    if ply_files:
        return os.path.join(path, ply_files[0])

    ply_files = [f for f in os.listdir(path) if f.endswith('.ply')]
    if ply_files:
        return os.path.join(path, ply_files[0])

    return None

def _merge_point_clouds(point_clouds):
    if not point_clouds:
        return None

    if len(point_clouds) == 1:
        return point_clouds[0]

    merged_pcd = BasicPointCloud(
        points=np.concatenate([pcd.points for pcd in point_clouds], axis=0),
        colors=np.concatenate([pcd.colors for pcd in point_clouds], axis=0),
        normals=np.concatenate([pcd.normals for pcd in point_clouds], axis=0),
    )

    downsample_stride = max(len(point_clouds), 1)
    if downsample_stride > 1:
        merged_pcd = BasicPointCloud(
            points=merged_pcd.points[::downsample_stride],
            colors=merged_pcd.colors[::downsample_stride],
            normals=merged_pcd.normals[::downsample_stride],
        )

    return merged_pcd

def readScannetCameras(path, images_folder="color", image_name_prefix=None, uid_offset=0):
    """
    Read cameras from ScanNet format
    ScanNet structure:
    - intrinsic/intrinsic_color.txt: shared color intrinsics
    - pose/ directory: contains per-frame camera-to-world matrices
    - color/ directory: contains image files (.jpg)
    """
    cam_infos = []
    
    pose_dir = os.path.join(path, "pose")
    intrinsic_path = os.path.join(path, "intrinsic", "intrinsic_color.txt")
    image_dir = os.path.join(path, images_folder)

    if not os.path.exists(pose_dir):
        raise ValueError(f"ScanNet pose directory not found: {pose_dir}")
    if not os.path.exists(intrinsic_path):
        raise ValueError(f"ScanNet intrinsic file not found: {intrinsic_path}")
    if not os.path.exists(image_dir):
        raise ValueError(f"ScanNet image directory not found: {image_dir}")

    intrinsics = np.loadtxt(intrinsic_path)
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    image_files = sorted(
        [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png"))],
        key=lambda name: int(Path(name).stem) if Path(name).stem.isdigit() else Path(name).stem,
    )

    for idx, image_file_name in tqdm(
        enumerate(image_files),
        total=len(image_files),
    ):
        image_file = os.path.join(image_dir, image_file_name)
        image_name = Path(image_file_name).stem
        unique_image_name = image_name
        if image_name_prefix is not None:
            unique_image_name = f"{image_name_prefix}_{image_name}"

        pose_path = os.path.join(pose_dir, f"{image_name}.txt")
        if not os.path.exists(pose_path):
            print(f"Warning: Pose not found for image {image_file_name}, skipping")
            continue

        try:
            # ScanNet stores camera-to-world matrices in pose/<frame_id>.txt
            c2w = np.loadtxt(pose_path)
            if c2w.shape != (4, 4):
                print(f"Warning: Invalid pose shape {c2w.shape} for {pose_path}, skipping")
                continue
            if not np.isfinite(c2w).all():
                print(f"Warning: Non-finite pose values in {pose_path}, skipping")
                continue
            
            # Convert to world-to-camera (W2C)
            w2c = np.linalg.inv(c2w)

            # Load image only after pose passes validation.
            image = Image.open(image_file)
            width, height = image.size
            
            # Convert focal length to FoV
            FovX = focal2fov(fx, width)
            FovY = focal2fov(fy, height)
            
            # Extract R and T
            # R should be transposed for the format expected by Camera class
            R = np.transpose(w2c[:3, :3])
            T = w2c[:3, 3]
            
            cam_info = CameraInfo(
                uid=uid_offset + idx,
                global_id=uid_offset + idx,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image_path=image_file,
                image_name=unique_image_name,
                width=width,
                height=height,
                fx=fx,
                fy=fy,
            )
            
            # Release memory
            image.close()
            image = None
            
            cam_infos.append(cam_info)
            
        except Exception as e:
            print(f"Error loading camera {image_file_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return cam_infos


def _load_matrix_4x4_txt(file_path: str) -> np.ndarray:
    mat = np.loadtxt(file_path, dtype=np.float32)
    if mat.shape == (4, 4):
        return mat
    if mat.size == 16:
        return mat.reshape(4, 4)
    raise ValueError(f"矩阵文件不是 4x4: {file_path}, shape={mat.shape}")


def _scannetv2_sort_key(cam: CameraInfo):
    try:
        return (0, int(cam.image_name))
    except Exception:
        return (1, cam.image_name)


def _scannetv2_name_sort_key(name: str):
    stem = Path(name).stem
    try:
        return (0, int(stem))
    except Exception:
        return (1, stem)


def readScannetV2Cameras(path, images_folder="color"):
    """
    Read cameras from ScanNetV2 format.
    ScanNetV2 structure:
    - color/ directory: image files
    - pose/ directory: c2w txt files (4x4)
    - intrinsic/intrinsic_color.txt: camera intrinsic matrix
    """
    cam_infos = []

    pose_dir = os.path.join(path, "pose")
    image_dir = os.path.join(path, images_folder)
    intrinsic_dir = os.path.join(path, "intrinsic")
    intrinsic_file = os.path.join(intrinsic_dir, "intrinsic_color.txt")

    if not os.path.exists(pose_dir):
        raise ValueError(f"ScanNetV2 pose directory not found: {pose_dir}")
    if not os.path.exists(image_dir):
        raise ValueError(f"ScanNetV2 image directory not found: {image_dir}")
    if not os.path.exists(intrinsic_file):
        raise ValueError(f"ScanNetV2 intrinsic file not found: {intrinsic_file}")

    K = _load_matrix_4x4_txt(intrinsic_file)
    fx = float(K[0, 0])
    fy = float(K[1, 1])

    pose_files = sorted([f for f in os.listdir(pose_dir) if f.endswith(".txt")], key=_scannetv2_name_sort_key)
    for idx, pose_file in tqdm(
        enumerate(pose_files),
        total=len(pose_files),
    ):
        frame_stem = Path(pose_file).stem
        pose_file_path = os.path.join(pose_dir, pose_file)

        image_file = None
        for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]:
            cand = os.path.join(image_dir, f"{frame_stem}{ext}")
            if os.path.exists(cand):
                image_file = cand
                break
        if image_file is None:
            continue

        try:
            c2w = _load_matrix_4x4_txt(pose_file_path)
            if not np.isfinite(c2w).all():
                continue
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])
            T = w2c[:3, 3]

            image = Image.open(image_file)
            width, height = image.size
            image.close()

            FovX = focal2fov(fx, width)
            FovY = focal2fov(fy, height)

            cam_info = CameraInfo(
                uid=idx,
                global_id=idx,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image_path=image_file,
                image_name=frame_stem,
                width=width,
                height=height,
                fx=fx,
                fy=fy,
            )
            cam_infos.append(cam_info)
        except Exception as e:
            print(f"Error loading ScanNetV2 frame {pose_file}: {e}")
            continue

    return cam_infos

def readDL3DVSceneInfo(path, images="rgb", eval=False, llffhold=8, ply_path="pointcloud_da3.ply"):
    """
    Read scene information from DL3DV format
    """
    cam_infos_unsorted = readDL3DVCameras(path, images_folder=images)
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)
    
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
    
    nerf_normalization = getNerfppNorm(train_cam_infos)
    
    # For DL3DV, we might not have initial point cloud
    # If not provided, we'll create a random one or use from pretrained model
    if not os.path.exists(ply_path):
        ply_path = os.path.join(path, "dense", ply_path)
    assert os.path.exists(ply_path), f"请把depthanything V3 的输出点云放在{ply_path}."
    
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    
    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info

def readScannetSceneInfo(path, images="color", eval=False, llffhold=8, ply_path=None):
    """
    Read scene information from ScanNet format
    """
    scene_part_paths = _get_scannet_scene_part_paths(path)
    print(f"Loading ScanNet scene parts: {[os.path.basename(p) for p in scene_part_paths]}")

    cam_infos_unsorted = []
    uid_offset = 0
    for scene_part_path in scene_part_paths:
        scene_part_name = os.path.basename(os.path.normpath(scene_part_path))
        part_cam_infos = readScannetCameras(
            scene_part_path,
            images_folder=images,
            image_name_prefix=scene_part_name,
            uid_offset=uid_offset,
        )
        cam_infos_unsorted.extend(part_cam_infos)
        uid_offset += len(part_cam_infos)

    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)
    cam_infos = [
        cam_info._replace(uid=idx, global_id=idx)
        for idx, cam_info in enumerate(cam_infos)
    ]
    
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
    
    nerf_normalization = getNerfppNorm(train_cam_infos)
    
    if ply_path is not None:
        ply_paths = [ply_path]
    else:
        ply_paths = [_find_scannet_ply_path(scene_part_path) for scene_part_path in scene_part_paths]

    valid_ply_paths = []
    point_clouds = []
    for cur_ply_path in ply_paths:
        if not cur_ply_path or not os.path.exists(cur_ply_path):
            continue
        try:
            point_clouds.append(fetchPly(cur_ply_path))
            valid_ply_paths.append(cur_ply_path)
        except:
            continue

    pcd = _merge_point_clouds(point_clouds)
    if len(valid_ply_paths) > 1:
        print(
            f"Merged {len(valid_ply_paths)} ScanNet point clouds and downsampled with stride {len(valid_ply_paths)}."
        )
    
    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=valid_ply_paths[0] if len(valid_ply_paths) == 1 else None,
    )
    return scene_info


def readScannetV2SceneInfo(path, images="color", eval=False, llffhold=8, ply_path="pointcloud_da3.ply"):
    """
    Read scene information from ScanNetV2 format.
    """
    cam_infos_unsorted = readScannetV2Cameras(path, images_folder=images)
    cam_infos = sorted(cam_infos_unsorted.copy(), key=_scannetv2_sort_key)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if ply_path is None:
        ply_path = os.path.join(path, "pointcloud_da3.ply")
    elif not os.path.isabs(ply_path):
        ply_path = os.path.join(path, ply_path)

    if ply_path and os.path.exists(ply_path):
        try:
            pcd = fetchPly(ply_path)
        except Exception:
            pcd = None
    else:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path if ply_path and os.path.exists(ply_path) else None,
    )
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "DL3DV": readDL3DVSceneInfo,
    "scannet": readScannetSceneInfo,
    "scannetv2": readScannetV2SceneInfo,
}