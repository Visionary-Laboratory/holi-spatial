import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
import math
import argparse
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.read_write_model import read_model, qvec2rotmat

def load_dl3dv_poses(dl3dv_dir, images_folder='rgb'):
    path = dl3dv_dir
    # 检测是 DL3DV 格式还是 ScanNet 格式
    # DL3DV: dense/cam/ 和 dense/rgb/
    # ScanNet: cam/ 和 color/
    dense_path = os.path.join(path, "dense")
    is_dl3dv = os.path.isdir(os.path.join(dense_path, "cam")) and os.path.isdir(
        os.path.join(dense_path, images_folder)
    )
    is_scannet = os.path.isdir(os.path.join(path, "cam")) and os.path.isdir(
        os.path.join(path, "color")
    )
    
    if is_scannet:
        # ScanNet 格式
        cam_dir = os.path.join(path, "cam")
        image_dir = os.path.join(path, "color")
    elif is_dl3dv:
        # DL3DV 格式
        cam_dir = os.path.join(dense_path, "cam")
        image_dir = os.path.join(dense_path, images_folder)
    else:
        raise ValueError(f"无法识别数据格式。路径: {path}")

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


# 主程序
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Your program description here"
    )

    # Add arguments
    parser.add_argument("--dl3dv_dirs", type=str, default="",
                        help="Path to multiple input file, e.g. /path/to/1K")
    parser.add_argument("--dl3dv_dir", type=str, default="",
                        help="Path to input file, e.g. /path/to/1K/0a1b7c20a92c43c6b8954b1ac909fb2f0fa8b2997b80604bc8bbec80a1cb2da3")
    parser.add_argument("--dl3dv_dir_list", nargs="+", default="",
                        help="list of specified dirs, like /path/to/1K/scene1 /path/to/1K/scene2")
    parser.add_argument('--load_colmap', action='store_true', help='load from colmap pose.')

    args = parser.parse_args()
    assert not (args.dl3dv_dirs == "" and args.dl3dv_dir == "" and args.dl3dv_dir_list == ""), "Please provide the path to dl3dv."
    assert not (len(args.dl3dv_dirs) > 0 and len(args.dl3dv_dir) > 0), "Do not simultaneouse provide multi folder and single folder args."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    print("正在加载模型...")
    model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
    model = model.to(device=device)
    print("模型加载完成")
    
    # 从 COLMAP 数据加载 pose
    # folder = '/mnt/shared-storage-user/solution/liuyifei/datasets/processed_dl3dv_ours/1K'
    # dl3dv_dirs = [os.path.join(folder, d) for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
    # dl3dv_dirs = sorted(dl3dv_dirs)[:10]
    if args.dl3dv_dirs is not None and args.dl3dv_dirs != "":
        dl3dv_dirs = [os.path.join(args.dl3dv_dirs, d) for d in os.listdir(args.dl3dv_dirs) if os.path.isdir(os.path.join(args.dl3dv_dirs, d))]
        dl3dv_dirs = sorted(dl3dv_dirs)
    elif args.dl3dv_dir is not None and args.dl3dv_dir != "":
        dl3dv_dirs = []
        dl3dv_dirs.append(args.dl3dv_dir)
    else:
        dl3dv_dirs = args.dl3dv_dir_list
    
    # Remove bad extrinsic data
    bad = ["3671f05fc0771dcd7751d2a397ae01a05aa509ca48f3ca1d9295053512608a92"]
    dl3dv_dirs = [d for d in dl3dv_dirs if not any(b in d for b in bad)]

    print(f"there are {len(dl3dv_dirs)} dl3dv dirs to process before filtering")
    for d in dl3dv_dirs[:]:
        assert os.path.exists(d)
        # 检测数据格式以确定点云路径
        dense_path = os.path.join(d, "dense")
        is_dl3dv = os.path.isdir(os.path.join(dense_path, "cam")) and os.path.isdir(
            os.path.join(dense_path, "rgb")
        )
        is_scannet = os.path.isdir(os.path.join(d, "cam")) and os.path.isdir(
            os.path.join(d, "color")
        )
        
        if is_scannet:
            target_3dgs_path = os.path.join(d, 'pointcloud_da3.ply')
        elif is_dl3dv:
            target_3dgs_path = os.path.join(d, 'dense/pointcloud_da3.ply')
        else:
            target_3dgs_path = None
        
        if target_3dgs_path and os.path.exists(target_3dgs_path):
            dl3dv_dirs.remove(d)
    print(f"there are {len(dl3dv_dirs)} dl3dv dirs to process after filtering")

    PER_GPU_SIZE=283  # = num of scenes / num of gpus
    for start_idx in range(0, len(dl3dv_dirs), PER_GPU_SIZE):
        end_idx = min(len(dl3dv_dirs), start_idx + PER_GPU_SIZE)
        print(f'From indx {start_idx} to {end_idx} ')
        print(dl3dv_dirs[start_idx:end_idx])

    for dl3dv_dir in tqdm(dl3dv_dirs):
        # 检测数据格式以确定点云路径
        dense_path = os.path.join(dl3dv_dir, "dense")
        is_dl3dv = os.path.isdir(os.path.join(dense_path, "cam")) and os.path.isdir(
            os.path.join(dense_path, "rgb")
        )
        is_scannet = os.path.isdir(os.path.join(dl3dv_dir, "cam")) and os.path.isdir(
            os.path.join(dl3dv_dir, "color")
        )
        
        if is_scannet:
            target_3dgs_path = os.path.join(dl3dv_dir, 'pointcloud_da3.ply')
        elif is_dl3dv:
            target_3dgs_path = os.path.join(dl3dv_dir, 'dense/pointcloud_da3.ply')
        else:
            print(f'无法识别数据格式，跳过: {dl3dv_dir}')
            continue
        
        if os.path.exists(target_3dgs_path):
            print(f'{target_3dgs_path} already exists, skipping.')
            continue

        print(f"正在从 {dl3dv_dir} 加载 COLMAP 数据...")
        load_func = load_colmap_poses if args.load_colmap else load_dl3dv_poses
        image_files, extrinsics, intrinsics = load_func(dl3dv_dir)

        orig_w, orig_h = Image.open(image_files[0]).size
        upper_bound_res = max(orig_w, orig_h)
        
        print(f"外参形状: {extrinsics.shape}")  # (N, 4, 4)
        print(f"内参形状: {intrinsics.shape}")  # (N, 3, 3)
        print("data load finished")
        # 运行推理
        print("正在运行推理...")
        output_dir = f"./output_{dl3dv_dir.split('/')[-1]}"

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
        assert (prediction.intrinsics - intrinsics < 1e-6).all(), f"{dl3dv_dir=} the predicted intrinsics should be consistent with input intrinsics."
        assert (prediction_extrinsics - extrinsics < 1e-6).all(), f"{dl3dv_dir=} the predicted extrinsics should be consistent with input extrinsics."

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

        # 检测数据格式以确定保存路径
        dense_path = os.path.join(dl3dv_dir, "dense")
        is_dl3dv = os.path.isdir(os.path.join(dense_path, "cam")) and os.path.isdir(
            os.path.join(dense_path, "rgb")
        )
        is_scannet = os.path.isdir(os.path.join(dl3dv_dir, "cam")) and os.path.isdir(
            os.path.join(dl3dv_dir, "color")
        )
        
        if is_scannet:
            # ScanNet 格式：保存到 scene_dir/depth_da3/
            depth_save_dir = os.path.join(dl3dv_dir, 'depth_da3')
            conf_save_dir = os.path.join(dl3dv_dir, 'conf_da3')
            pointcloud_save_path = os.path.join(dl3dv_dir, 'pointcloud_da3.ply')
        elif is_dl3dv:
            # DL3DV 格式：保存到 scene_dir/dense/depth_da3/
            depth_save_dir = os.path.join(dl3dv_dir, 'dense', 'depth_da3')
            conf_save_dir = os.path.join(dl3dv_dir, 'dense', 'conf_da3')
            pointcloud_save_path = os.path.join(dl3dv_dir, 'dense', 'pointcloud_da3.ply')
        else:
            raise ValueError(f"无法识别数据格式。路径: {dl3dv_dir}")
        
        os.makedirs(depth_save_dir, exist_ok=True)
        os.makedirs(conf_save_dir, exist_ok=True)
        
        # 保存深度图和置信度图，使用原始文件名（不带 frame_ 前缀）
        for idx, (d, c) in enumerate(tqdm(zip(prediction.depth, prediction.conf), total=len(prediction.depth))):
            # 从 image_files 中提取原始文件名
            image_name = os.path.splitext(os.path.basename(image_files[idx]))[0]
            np.save(os.path.join(depth_save_dir, f'{image_name}.npy'), d)
            np.save(os.path.join(conf_save_dir, f'{image_name}.npy'), c)
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
        
        # 下采样点云到 20 million 个点
        print("\n正在下采样点云...")
        points, colors = downsample_pointcloud(points, colors, target_num_points=4000000)
        
        # 保存点云
        save_pointcloud_ply(points, colors, pointcloud_save_path)
        
        print("\n完成!")