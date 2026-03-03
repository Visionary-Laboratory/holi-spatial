import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
import math
import argparse
from pathlib import Path
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.read_write_model import read_model, qvec2rotmat
import json

def load_scannet_poses(scannet_dir, images_folder='color'):
    """
    从 ScanNet 数据中加载相机 pose 和内参
    
    Args:
        scannet_dir: ScanNet 场景目录路径 (例如: "processed_scannet/scans_test/scene0707_00")
        images_folder: 图像文件夹名称，默认为 'color'
    
    Returns:
        image_files: 图像文件路径列表
        extrinsics: 外参矩阵 (N, 4, 4) - world to camera
        intrinsics: 内参矩阵 (N, 3, 3)
    """
    path = scannet_dir
    # ScanNet data structure: cam/ and color/ directories are at the scene root
    cam_dir = os.path.join(path, "cam")
    image_dir = os.path.join(path, images_folder)

    if not os.path.exists(cam_dir):
        raise ValueError(f"相机目录不存在: {cam_dir}")
    if not os.path.exists(image_dir):
        raise ValueError(f"图像目录不存在: {image_dir}")

    # Get list of camera files
    cam_files = sorted([f for f in os.listdir(cam_dir) if f.endswith('.npz')])

    if len(cam_files) == 0:
        raise ValueError(f"在 {cam_dir} 中未找到相机文件")

    image_files = []
    extrinsics_list = []
    intrinsics_list = []

    for idx, cam_file in tqdm(enumerate(cam_files), total=len(cam_files)):
        cam_file_path = os.path.join(cam_dir, cam_file)
        image_name = cam_file.replace('.npz', '')
        
        # Try to find image file
        image_file = os.path.join(image_dir, f"{image_name}.jpg")
        if not os.path.exists(image_file):
            image_file = os.path.join(image_dir, f"{image_name}.png")
        if not os.path.exists(image_file):
            print(f"警告: 未找到图像文件 {cam_file}，跳过")
            continue
        image_files.append(image_file)
        
        try:
            # Load camera parameters from .npz file
            cam_data = np.load(cam_file_path)
            
            # Extract intrinsics - ScanNet uses 'intrinsics' (plural) and it's a 3x3 matrix
            if 'intrinsics' in cam_data:
                intrinsics = cam_data['intrinsics']
                fx = intrinsics[0, 0]
                fy = intrinsics[1, 1]
                cx = intrinsics[0, 2]
                cy = intrinsics[1, 2]
            elif 'intrinsic' in cam_data:
                # Fallback for DL3DV-like format
                intrinsic = cam_data['intrinsic']
                fx = intrinsic[0, 0]
                fy = intrinsic[1, 1]
                cx = intrinsic[0, 2]
                cy = intrinsic[1, 2]
            else:
                raise ValueError(f"在 {cam_file} 中未找到内参。可用键: {list(cam_data.keys())}")
            intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            intrinsics_list.append(intrinsic)

            # Extract extrinsics
            # 'pose' is the camera-to-world matrix (C2W)
            if 'pose' not in cam_data:
                raise ValueError(f"在 {cam_file} 中未找到pose。可用键: {list(cam_data.keys())}")
            c2w = cam_data['pose']
            w2c = np.linalg.inv(c2w)
            # w2c = c2w
            extrinsics_list.append(w2c)
            
        except Exception as e:
            print(f"加载相机文件 {cam_file} 时出错: {e}")
            raise ValueError(f"无法加载相机。尝试过的路径: {cam_file_path}")
    
    if len(image_files) == 0:
        raise ValueError(f"在 ScanNet 数据中未找到有效的图像文件")
    
    print(f"从 ScanNet 数据中加载了 {len(image_files)} 张图像")
    
    return image_files, np.array(extrinsics_list), np.array(intrinsics_list)


def load_dl3dv_poses(dl3dv_dir, images_folder='rgb'):
    path = dl3dv_dir
    # DL3DV data is in dense/ subdirectory
    dense_path = os.path.join(path, "dense")
    cam_dir = os.path.join(dense_path, "cam")
    image_dir = os.path.join(dense_path, images_folder)

    # Get list of camera files
    cam_files = sorted([f for f in os.listdir(cam_dir) if f.endswith('.npz')])

    image_files = []
    extrinsics_list = []
    intrinsics_list = []

    for idx, cam_file in tqdm(enumerate(cam_files),total=len(cam_files)):
        cam_file_path = os.path.join(cam_dir, cam_file)
        image_name = cam_file.replace('.npz', '')
        
        # Try to find image file
        image_file = os.path.join(image_dir, f"{image_name}.jpg")
        if not os.path.exists(image_file):
            image_file = os.path.join(image_dir, f"{image_name}.png")
        if not os.path.exists(image_file):
            print(f"Warning: Image not found for camera {cam_file}, skipping")
            continue
        image_files.append(image_file)
        
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
            intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            intrinsics_list.append(intrinsic)

            # Extract extrinsics
            # 'pose' is the camera-to-world matrix (C2W)
            c2w = cam_data['pose']
            w2c = np.linalg.inv(c2w)
            extrinsics_list.append(w2c)
            
        except Exception as e:
            print(f"Error loading camera {cam_file}: {e}")
            raise ValueError(f"无法加载相机。尝试过的路径: {cam_file_path}")
        
    return image_files, np.array(extrinsics_list), np.array(intrinsics_list)



def load_colmap_poses(colmap_dir):
    """
    从 COLMAP 数据中加载相机 pose 和内参
    
    Args:
        colmap_dir: COLMAP sparse 重建目录路径 (例如: "proxy-gs/berlin/sparse/0")
    
    Returns:
        image_files: 图像文件路径列表
        extrinsics: 外参矩阵 (N, 4, 4) - world to camera
        intrinsics: 内参矩阵 (N, 3, 3)
    """
    # 读取 COLMAP 模型
    cameras, images, points3D = read_model(colmap_dir, ext="")
    
    # 获取图像目录（尝试多个可能的路径）
    possible_paths = [
        os.path.join(os.path.dirname(os.path.dirname(colmap_dir)), "images"),
        os.path.join(os.path.dirname(colmap_dir), "images"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(colmap_dir))), "images"),
    ]
    
    images_dir = None
    for path in possible_paths:
        if os.path.exists(path):
            images_dir = path
            break
    
    if images_dir is None:
        # 如果所有路径都不存在，尝试在当前目录查找
        images_dir = "images"
        if not os.path.exists(images_dir):
            raise ValueError(f"无法找到图像目录。尝试过的路径: {possible_paths}")
    
    image_files = []
    extrinsics = []
    intrinsics = []
    
    # 按图像 ID 排序以确保顺序一致
    sorted_images = sorted(images.items(), key=lambda x: x[0])
    
    # 限制只使用前250张图片
    max_images = 1000
    if len(sorted_images) > max_images:
        sorted_images = sorted_images[:max_images]
        print(f"限制使用前 {max_images} 张图片（总共 {len(images)} 张）")
    
    for image_id, image_data in sorted_images:
        image_name = image_data.name
        image_path = os.path.join(images_dir, image_name)
        
        if not os.path.exists(image_path):
            print(f"警告: 图像文件不存在: {image_path}")
            continue
        
        image_files.append(image_path)
        
        # 获取相机参数
        camera = cameras[image_data.camera_id]
        
        # 将四元数转换为旋转矩阵
        R = qvec2rotmat(image_data.qvec)
        t = image_data.tvec
        
        # 创建外参矩阵 (world to camera, COLMAP 格式)
        extrinsic = np.eye(4, dtype=np.float32)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = t
        extrinsics.append(extrinsic)
        
        # 创建内参矩阵
        if camera.model == "PINHOLE":
            fx, fy, cx, cy = camera.params
        elif camera.model == "SIMPLE_PINHOLE":
            f, cx, cy = camera.params
            fx = fy = f
        else:
            # 对于其他模型，使用基本针孔近似
            fx = fy = camera.params[0] if len(camera.params) > 0 else 1000
            cx = camera.width / 2
            cy = camera.height / 2
        
        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        intrinsics.append(intrinsic)
    
    if not image_files:
        raise ValueError(f"在 COLMAP 数据中未找到有效的图像文件")
    
    print(f"从 COLMAP 数据中加载了 {len(image_files)} 张图像")
    
    return image_files, np.array(extrinsics), np.array(intrinsics)


def depth_to_pointcloud(depth, intrinsics, extrinsics, images, conf=None, conf_threshold=0.5):
    """
    从深度图生成点云
    
    Args:
        depth: 深度图 (N, H, W)
        intrinsics: 内参矩阵 (N, 3, 3)
        extrinsics: 外参矩阵 (N, 4, 4) - world to camera
        images: 图像 (N, H, W, 3) - 用于颜色
        conf: 置信度图 (N, H, W) - 可选
        conf_threshold: 置信度阈值
    
    Returns:
        points: 点云坐标 (M, 3)
        colors: 点云颜色 (M, 3)
    """
    N, H, W = depth.shape
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    ones = np.ones_like(us)
    pix = np.stack([us, vs, ones], axis=-1).reshape(-1, 3)  # (H*W, 3)
    
    points_all = []
    colors_all = []
    
    for i in tqdm(range(N)):
        d = depth[i]  # (H, W)
        valid = np.isfinite(d) & (d > 0)
        
        if conf is not None:
            valid &= conf[i] >= conf_threshold
        
        if not np.any(valid):
            continue
        
        d_flat = d.reshape(-1)
        vidx = np.flatnonzero(valid.reshape(-1))
        
        # 计算逆变换
        K_inv = np.linalg.inv(intrinsics[i])  # (3, 3)
        c2w = np.linalg.inv(extrinsics[i])  # (4, 4) - camera to world
        
        # 将像素坐标转换为相机坐标
        rays = K_inv @ pix[vidx].T  # (3, M)
        Xc = rays * d_flat[vidx][None, :]  # (3, M)
        Xc_h = np.vstack([Xc, np.ones((1, Xc.shape[1]))])  # (4, M)
        
        # 转换为世界坐标
        Xw = (c2w @ Xc_h)[:3].T.astype(np.float32)  # (M, 3)
        
        # 提取颜色
        cols = images[i].reshape(-1, 3)[vidx].astype(np.uint8)  # (M, 3)
        
        points_all.append(Xw)
        colors_all.append(cols)
    
    if len(points_all) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
    
    return np.concatenate(points_all, 0), np.concatenate(colors_all, 0)


def depth_to_pointcloud_torch_batched(
    depth, intrinsics, extrinsics, images, conf=None,
    conf_threshold=0.5,
    batch_size=8,
    device="cuda",
):
    """
    depth: (N, H, W)
    intrinsics: (N, 3, 3)
    extrinsics: (N, 4, 4)  -- world-to-camera
    images: (N, H, W, 3)
    conf: (N, H, W) or None
    """

    # Move inputs to GPU
    depth = torch.as_tensor(depth, device=device)
    intrinsics = torch.as_tensor(intrinsics, device=device)
    extrinsics = torch.as_tensor(extrinsics, device=device)
    images = torch.as_tensor(images, device=device).to(torch.uint8)

    if conf is not None:
        conf = torch.as_tensor(conf, device=device)

    N, H, W = depth.shape
    HW = H * W

    # Create a single (H*W, 3) pixel grid once, reused for all batches
    us, vs = torch.meshgrid(
        torch.arange(W, device=device),
        torch.arange(H, device=device),
        indexing="xy"
    )
    pix = torch.stack([us, vs, torch.ones_like(us)], dim=-1).reshape(-1, 3)  # (HW, 3)

    all_points = []
    all_colors = []

    # Process in batches of N
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        B = end - start

        # Slice batch
        d = depth[start:end]              # (B, H, W)
        imgs = images[start:end]          # (B, H, W, 3)
        K = intrinsics[start:end]         # (B, 3, 3)
        W2C = extrinsics[start:end]       # (B, 4, 4)
        C2W = torch.inverse(W2C)          # (B, 4, 4)

        # Flatten
        d_flat = d.reshape(B, -1)         # (B, HW)
        img_flat = imgs.reshape(B, -1, 3) # (B, HW, 3)

        # Valid depth mask
        valid = torch.isfinite(d_flat) & (d_flat > 0)

        if conf is not None:
            c_flat = conf[start:end].reshape(B, -1)
            valid &= (c_flat >= conf_threshold)

        # For each image in batch, extract only valid pixels
        for bi in range(B):
            vidx = valid[bi].nonzero(as_tuple=True)[0]      # (M,)
            if vidx.numel() == 0:
                continue

            # Compute K^-1 once
            K_inv = torch.inverse(K[bi])                   # (3, 3)

            # pix coords
            pix_sel = pix[vidx].T                          # (3, M)

            # rays = K^-1 * pix
            rays = K_inv @ pix_sel                         # (3, M)

            # scale by depth
            depths_sel = d_flat[bi][vidx]                  # (M,)
            Xc = rays * depths_sel.unsqueeze(0)            # (3, M)

            # Homogeneous → world
            Xc_h = torch.cat([Xc, torch.ones((1, Xc.shape[1]), device=device)], dim=0)
            Xw = (C2W[bi] @ Xc_h)[:3].T                    # (M, 3)

            # Color
            cols = img_flat[bi][vidx]                     # (M, 3)

            all_points.append(Xw)
            all_colors.append(cols)

    if len(all_points) == 0:
        return (torch.zeros((0, 3), dtype=torch.float32, device=device),
                torch.zeros((0, 3), dtype=torch.uint8, device=device))

    return torch.cat(all_points, dim=0), torch.cat(all_colors, dim=0)



def downsample_pointcloud(points, colors, target_num_points=20000000):
    """
    下采样点云到指定的点数
    
    Args:
        points: 点云坐标 (N, 3)
        colors: 点云颜色 (N, 3)
        target_num_points: 目标点数，默认 20 million
    
    Returns:
        points_downsampled: 下采样后的点云坐标
        colors_downsampled: 下采样后的点云颜色
    """
    num_points = len(points)
    
    if num_points <= target_num_points:
        print(f"点云点数 ({num_points}) 已小于等于目标点数 ({target_num_points})，无需下采样")
        return points, colors
    
    # 随机采样索引
    indices = np.random.choice(num_points, size=target_num_points, replace=False)
    indices = np.sort(indices)  # 排序以保持顺序
    
    points_downsampled = points[indices]
    colors_downsampled = colors[indices]
    
    print(f"点云已从 {num_points} 个点下采样到 {target_num_points} 个点")
    
    return points_downsampled, colors_downsampled


def save_pointcloud_ply(points, colors, filename):
    """
    保存点云为 PLY 格式
    
    Args:
        points: 点云坐标 (N, 3)
        colors: 点云颜色 (N, 3)
        filename: 输出文件名
    """
    import struct
    
    num_points = len(points)
    
    with open(filename, 'wb') as f:
        # PLY 头部
        f.write(b'ply\n')
        f.write(b'format binary_little_endian 1.0\n')
        f.write(f'element vertex {num_points}\n'.encode())
        f.write(b'property float x\n')
        f.write(b'property float y\n')
        f.write(b'property float z\n')
        f.write(b'property uchar red\n')
        f.write(b'property uchar green\n')
        f.write(b'property uchar blue\n')
        f.write(b'end_header\n')
        
        # 写入点云数据
        for i in range(num_points):
            f.write(struct.pack('<fffBBB', 
                              float(points[i, 0]),
                              float(points[i, 1]),
                              float(points[i, 2]),
                              int(colors[i, 0]),
                              int(colors[i, 1]),
                              int(colors[i, 2])))
    
    print(f"点云已保存到: {filename} (共 {num_points} 个点)")

def chunk_inference(image_files, image_names, intrinsics, extrinsics):
    orig_w, orig_h = Image.open(image_files[0]).size
    upper_bound_res = max(orig_w, orig_h)
    
    print(f"外参形状: {extrinsics.shape}")  # (N, 4, 4)
    print(f"内参形状: {intrinsics.shape}")  # (N, 3, 3)
    
    # 运行推理
    print("正在运行推理...")
    output_dir = f"./output_{scannet_dir.split('/')[-1]}"

    LOAD_FFROM_RESULTS = False
    # Either predict
    if not LOAD_FFROM_RESULTS:
        prediction = model.inference(
            image_files,
            extrinsics=extrinsics,  # (N, 4, 4)
            intrinsics=intrinsics,  # (N, 3, 3)
            align_to_input_ext_scale=True,
            # process_res=upper_bound_res,  # this can lead to OOM issue.
            # export_format="npz-glb",
            # export_dir=output_dir,
        )
    else:
        # Or load saved predictions
        prediction = np.load(output_dir+'/exports/npz/results.npz')
        # convert to namedtuple for consistent attribute access
        from collections import namedtuple
        replace_map = {"image": "processed_images"}
        Prediction = namedtuple('Prediction', [replace_map.get(k, k) for k in prediction.keys()])
        prediction = Prediction(**{replace_map.get(k, k): v for k, v in prediction.items()})
    
    # 打印预测结果形状
    print("\n预测结果:")
    print(f"processed_images: {prediction.processed_images.shape}")  # [N, H, W, 3]
    print(f"depth: {prediction.depth.shape}")  # [N, H, W]
    print(f"conf: {prediction.conf.shape}")  # [N, H, W]
    print(f"extrinsics: {prediction.extrinsics.shape}")  # [N, 3, 4] 或 [N, 4, 4]
    print(f"intrinsics: {prediction.intrinsics.shape}")  # [N, 3, 3]
    
    
    # 确保 extrinsics 是 4x4 格式
    if prediction.extrinsics.shape[-1] == 4 and prediction.extrinsics.shape[-2] == 3:
        # 如果是 (N, 3, 4)，转换为 (N, 4, 4)
        N = prediction.extrinsics.shape[0]
        extrinsics_4x4 = np.eye(4, dtype=np.float32)[None].repeat(N, axis=0)
        extrinsics_4x4[:, :3, :] = prediction.extrinsics
        prediction_extrinsics = extrinsics_4x4
    else:
        prediction_extrinsics = prediction.extrinsics
    assert (prediction.intrinsics - intrinsics < 1e-6).all(), "the predicted intrinsics should be consistent with input intrinsics."
    assert (prediction_extrinsics - extrinsics < 1e-6).all(), "the predicted extrinsics should be consistent with input extrinsics."

    # Resize img, depth, conf to original image size
    img_th = torch.from_numpy(prediction.processed_images).permute(0,3,1,2).float()
    img_th = F.interpolate(img_th, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
    if LOAD_FFROM_RESULTS:
        prediction = prediction._replace(processed_images = img_th.permute(0,2,3,1).cpu().numpy())
    else:
        prediction.processed_images =img_th.permute(0,2,3,1).cpu().numpy()
    depth_th = torch.from_numpy(prediction.depth).unsqueeze(1).float()
    depth_th = F.interpolate(depth_th, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
    if LOAD_FFROM_RESULTS:
        prediction = prediction._replace(depth = depth_th.squeeze(1).cpu().numpy())
    else:
        prediction.depth = depth_th.squeeze(1).cpu().numpy()
    conf_th = torch.from_numpy(prediction.conf).unsqueeze(1).float()
    conf_th = F.interpolate(conf_th, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
    if LOAD_FFROM_RESULTS:
        prediction = prediction._replace(conf = conf_th.squeeze(1).cpu().numpy())
    else:
        prediction.conf = conf_th.squeeze(1).cpu().numpy()

    # 保存深度图到 input_folder/dense/depth_da3 路径
    depth_save_dir = os.path.join(scannet_dir, 'depth_da3')
    conf_save_dir = os.path.join(scannet_dir, 'conf_da3')
    os.makedirs(depth_save_dir, exist_ok=True)
    os.makedirs(conf_save_dir, exist_ok=True)
    for id, d in tqdm(enumerate(prediction.depth, start=0)):
        np.save(f'{depth_save_dir}/{image_names[id]}.npy', d)
    for id, d in tqdm(enumerate(prediction.conf, start=0)):
        np.save(f'{conf_save_dir}/{image_names[id]}.npy', d)
    print(f'成功保存深度图到路径{depth_save_dir}')

    # 从深度图生成点云
    print("\n正在生成点云...")
    points, colors = depth_to_pointcloud(
        prediction.depth,
        intrinsics,
        prediction_extrinsics,
        prediction.processed_images,
        conf=prediction.conf,
        conf_threshold=0.3  # 可以根据需要调整置信度阈值
    )
    
    print(f"生成的点云: {points.shape[0]} 个点")
    return points, colors


# 主程序
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="使用 Depth Anything 3 对 ScanNet 数据集进行深度估计和点云生成"
    )

    # Add arguments
    parser.add_argument("--scannet_dirs", type=str, default="",
                        help="Path to multiple input scenes, e.g. /path/to/scans_test")
    parser.add_argument("--scannet_dir", type=str, default="",
                        help="Path to single input scene, e.g. /path/to/scans_test/scene0707_00")
    parser.add_argument("--scannet_dir_list", nargs="+", default="",
                        help="list of specified scene dirs, like /path/to/scans_test/scene1 /path/to/scans_test/scene2")
    parser.add_argument('--load_colmap', action='store_true', help='load from colmap pose.')
    parser.add_argument('--images_folder', type=str, default='color', help='images folder name, default: color')

    args = parser.parse_args()
    assert not (args.scannet_dirs == "" and args.scannet_dir == "" and args.scannet_dir_list == ""), "Please provide the path to ScanNet scenes."
    assert not (len(args.scannet_dirs) > 0 and len(args.scannet_dir) > 0), "Do not simultaneously provide multi folder and single folder args."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    print("正在加载模型...")
    model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
    model = model.to(device=device)
    print("模型加载完成")
    
    # 从 ScanNet 数据加载 pose
    if args.scannet_dirs is not None and args.scannet_dirs != "":
        scannet_dirs = [os.path.join(args.scannet_dirs, d) for d in os.listdir(args.scannet_dirs) if os.path.isdir(os.path.join(args.scannet_dirs, d))]
        scannet_dirs = sorted(scannet_dirs)
    elif args.scannet_dir is not None and args.scannet_dir != "":
        scannet_dirs = []
        scannet_dirs.append(args.scannet_dir)
    else:
        scannet_dirs = args.scannet_dir_list

    print(f"处理前有 {len(scannet_dirs)} 个 ScanNet 场景")
    for d in scannet_dirs[:]:
        assert os.path.exists(d), f"场景目录不存在: {d}"
        # 检查点云文件是否已存在（保存在场景根目录，与color文件夹并行）
        target_3dgs_path = os.path.join(d, 'pointcloud_da3.ply')
        scannet_dirs.remove(d) if os.path.exists(target_3dgs_path) else None
    print(f"过滤后有 {len(scannet_dirs)} 个 ScanNet 场景需要处理")
    PER_GPU_SIZE = 283  # = num of scenes / num of gpus
    for start_idx in range(0, len(scannet_dirs), PER_GPU_SIZE):
        end_idx = min(len(scannet_dirs), start_idx + PER_GPU_SIZE)
        print(f'索引 {start_idx} 到 {end_idx}')
        print(scannet_dirs[start_idx:end_idx])


    for scannet_dir in tqdm(scannet_dirs):
        target_3dgs_path = os.path.join(scannet_dir, 'pointcloud_da3.ply')
        if os.path.exists(target_3dgs_path):
            print(f'{target_3dgs_path} 已存在，跳过。')
            continue

        print(f"正在从 {scannet_dir} 加载 COLMAP 数据...")
        load_func = load_colmap_poses if args.load_colmap else load_scannet_poses
        if args.load_colmap:
            image_files, extrinsics, intrinsics = load_func(scannet_dir)
        else:
            image_files, extrinsics, intrinsics = load_func(scannet_dir, images_folder=args.images_folder)
        image_names = [Path(f).stem for f in image_files]

        MAX_CHUNK_SIZE=800  # tune this to avoid OOM
        def balanced_chunks(N, max_chunk=MAX_CHUNK_SIZE):
            """
            Split N items into chunks <= max_chunk, but as even as possible.
            Returns a list of (start, end) index ranges.
            """
            import math
            # minimal number of chunks needed
            num_chunks = math.ceil(N / max_chunk)
            # balanced chunk size
            chunk_size = math.ceil(N / num_chunks)
            ranges = []
            start = 0
            while start < N:
                end = min(N, start + chunk_size)
                ranges.append((start, end))
                start = end
            return ranges

        if len(image_files) < MAX_CHUNK_SIZE:
            points, colors = chunk_inference(image_files, image_names, intrinsics, extrinsics)
        else:
            rand_idx = torch.randperm(len(image_files))
            points = []
            colors = []
            ranges = balanced_chunks(len(image_files), max_chunk=MAX_CHUNK_SIZE)
            for start_idx, end_idx in ranges:
                indices = rand_idx[start_idx : end_idx].to(dtype=torch.long, device=device).tolist()
                point, color = chunk_inference([image_files[i] for i in indices], [image_names[i] for i in indices], intrinsics[indices], extrinsics[indices])
                point, color = downsample_pointcloud(point, color, target_num_points=4000000)
                points.append(point)
                colors.append(color)
            points = np.concatenate(points, 0)
            colors = np.concatenate(colors, 0)

        
        # 下采样点云到 20 million 个点
        print("\n正在下采样点云...")
        points, colors = downsample_pointcloud(points, colors, target_num_points=4000000)
        
        # 保存点云
        output_file = os.path.join(scannet_dir, 'pointcloud_da3.ply')
        save_pointcloud_ply(points, colors, output_file)
        
        print("\n完成!")