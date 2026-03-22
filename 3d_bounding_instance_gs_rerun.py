from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Literal
import os
import numpy as np
import numpy.typing as npt
import torch
import cv2
from PIL import Image
from skimage import morphology
from tqdm import tqdm
import trimesh
from scipy.spatial.transform import Rotation as R
import pycocotools.mask as mask_utils
from transformers import AutoModelForVision2Seq, AutoProcessor
import base64
import io

from sam3.agent.helpers.visualizer import Visualizer
from sam3.agent.helpers.zoom_in import render_zoom_in

PGSR_ROOT = Path(__file__).parent / "PGSR"
if str(PGSR_ROOT) not in sys.path:
    sys.path.append(str(PGSR_ROOT))

from arguments import ModelParams, PipelineParams  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from scene import Scene  # noqa: E402
from scene.gaussian_model import GaussianModel  # noqa: E402


DEFAULT_SCENE = "0a5c013435"
DEFAULT_DATA_ROOT = Path("scannetppv2/data")
DEFAULT_MASK_ROOT = Path("sam_masks_debug")
DEFAULT_OUTPUT_DIR = Path("output_3d_bounding")
DEFAULT_GS_MODEL = Path("output") / DEFAULT_SCENE
DEFAULT_RERUN = False
MAX_POINT_COUNT = 5_000_000
DEFAULT_VLM_MODEL_PATH = "/mnt/shared-storage-user/intern7shared/share_ckpt_hf/models--Qwen--Qwen3-VL-30B-A3B-Thinking/snapshots/7e9bbfa2c1b2059edd18160793fd421194da2c10/"


def label_color(label: str) -> np.ndarray:
    """根据标签生成稳定的伪随机颜色。"""
    h = abs(hash(label))
    r = 50 + (h % 206)
    g = 50 + ((h // 256) % 206)
    b = 50 + ((h // 65536) % 206)
    return np.array([r, g, b, 255], dtype=np.uint8)


def setup_logger(log_path: Optional[Path] = None) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
        
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
        handlers=handlers
    )


def load_vlm_model(model_path: str):
    logging.info("加载 VLM 模型: %s", model_path)
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def verify_label_with_vlm(
    model,
    processor,
    raw_image: Image.Image,
    overlay_image: Image.Image,
    zoomed_image: Image.Image,
    category: str,
) -> bool:
    """使用本地 Qwen 模型验证标签是否正确。"""
    prompt = (
        "You are a visual label verifier for a single masked object.\n"
        "You will be given three images: (1) the original image, (2) an overlay image where the target object is highlighted with a colored mask, and (3) a zoomed-in pair showing the original object and the highlighted object.\n"
        f'The user-provided label/category to verify is: "{category}".\n\n'

        "Task:\n"
        "Decide whether the given label correctly describes the object indicated by the HIGHLIGHTED region.\n"
        "If it is correct, ACCEPT. If it is incorrect (wrong category), REJECT and provide the correct label you believe fits best.\n\n"

        "Guidelines (consistency-and-rewrite mindset):\n"
        "1) Focus ONLY on the object covered by the HIGHLIGHTED region. Ignore other objects outside the mask.\n"
        "2) Identify the core visible noun/object class of the masked object (the best label).\n"
        "3) Treat extra modifiers in the label (color/material/size/position/relations) as constraints ONLY if they are clearly visible.\n"
        "   If modifiers are wrong or unverifiable but the core object class is correct, still ACCEPT.\n"
        "4) If the mask covers only a part of an object but the object class is still clear, judge by the most likely full object.\n"
        "5) If the masked region is ambiguous, too small, or does not correspond to a recognizable object, REJECT and set predicted_label to \"unknown\".\n"
        "6) Use common-sense synonyms/hypernyms: accept reasonable equivalents (e.g., \"sofa\" vs \"couch\").\n"
        "   If the label is too specific and not verifiable (e.g., exact brand/model/species), prefer a more general correct label.\n\n"

        "Output format (VERY IMPORTANT):\n"
        "Return ONLY a JSON object on a single line with these keys:\n"
        "  - decision: \"ACCEPT\" or \"REJECT\"\n"
        "  - predicted_label: a short English noun phrase for the masked object class (e.g., \"chair\", \"person\", \"car\").\n"
        "Rules:\n"
        "  - If decision is ACCEPT, predicted_label should be exactly the provided label (category) or its closest normalized form.\n"
        "  - If decision is REJECT, predicted_label must be your best guess of the correct label, or \"unknown\" if unclear.\n"
        "Do not output any extra text after the JSON object. The JSON should be the final part of your response."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": raw_image.convert("RGB")},
                {"type": "image", "image": overlay_image.convert("RGB")},
                {"type": "image", "image": zoomed_image.convert("RGB")},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    try:
        # 准备模型输入
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        with torch.inference_mode():
            # 再次增加 tokens 限制，某些复杂场景 Thinking 需要极长的篇幅
            generated_ids = model.generate(**inputs, max_new_tokens=2048)

        # 提取生成的文本
        trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # 尝试从输出中提取 JSON
        import json
        import re
        
        # 寻找 JSON 块：匹配最外层的 {}
        json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                data = json.loads(json_str)
                decision = data.get("decision", "REJECT")
                logging.info("Qwen 验证结果 [%s]: %s -> %s", category, decision, data.get("predicted_label", ""))
                return decision == "ACCEPT"
            except json.JSONDecodeError:
                logging.warning("Qwen 返回了无效的 JSON 字符串: %s", json_str)
        else:
            logging.warning("Qwen 未能输出 JSON 格式结果，输出预览: %s...", output_text[:200])
    except Exception as e:
        logging.warning("Qwen 验证过程出错: %s", e)

    return False


def load_mask(mask_path: Path) -> np.ndarray:
    mask = Image.open(mask_path).convert("L")
    return np.array(mask) > 0


def encode_mask(mask: np.ndarray) -> dict:
    """将 mask 编码为 RLE 格式，与 agent.py 中的保存方式一致"""
    rle = mask_utils.encode(np.asarray(mask, order="F"))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def load_transforms(
    scene_dir: Path, image_names: Optional[set[str]] = None
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    str,
    Dict[str, Tuple[int, int]],
    Dict[str, str],
    Dict[str, np.ndarray],
]:
    """
    读取相机与图像路径，返回
      intr_map[stem] = 3x3 K
      c2w_map[stem] = 4x4 camera-to-world
      scene_type: str（scannetppv2 / dl3dv / scannet）
      size_map[stem] = (width, height)
      path_map[stem] = image路径
      mask_map[stem] = image mask (bool)

    支持：ScanNet++ nerfstudio、DL3DV、ScanNet v2（color + intrinsic + pose）、
    ScanNet 变体（cam npz + color）。
    """
    # scannetppv2
    if (scene_dir / "dslr").exists():
        scene_type = "scannetppv2"
        json_path = scene_dir / "dslr/nerfstudio/transforms_undistorted.json"
        with json_path.open("r", encoding="utf-8") as f:
            contents = json.load(f)

        fl_x, fl_y, cx, cy = contents["fl_x"], contents["fl_y"], contents["cx"], contents["cy"]
        w, h = contents.get("w"), contents.get("h")
        train_frames = contents.get("frames", [])
        test_frames = contents.get("test_frames", [])

        intr_map: Dict[str, np.ndarray] = {}
        c2w_map: Dict[str, np.ndarray] = {}
        size_map: Dict[str, Tuple[int, int]] = {}
        path_map: Dict[str, str] = {}
        mask_map: Dict[str, np.ndarray] = {}

        mask_dir = scene_dir / "mask"

        for frame in [*train_frames, *test_frames]:
            stem = Path(frame["file_path"]).stem
            if image_names is not None and stem not in image_names:
                continue

            K = np.array([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]], dtype=np.float32)
            c2w = np.array(frame["transform_matrix"], dtype=np.float32)
            # OpenGL -> COLMAP 约定
            c2w[:3, 1:3] *= -1
            intr_map[stem] = K
            c2w_map[stem] = c2w

            frame_w = frame.get("w", w)
            frame_h = frame.get("h", h)
            size_map[stem] = (frame_w, frame_h)
            path_map[stem] = str(
                scene_dir / "dslr/" / "resized_undistorted_images" / frame["file_path"]
            )

            # 加载 mask
            mask_path = mask_dir / f"{stem}.png"
            if mask_path.exists():
                mask = load_mask(mask_path)
                # 调整 mask 尺寸以匹配图像尺寸
                if mask.shape[:2] != (frame_h, frame_w):
                    mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
                    mask_resized = mask_img.resize((frame_w, frame_h), resample=Image.NEAREST)
                    mask = (np.array(mask_resized) > 0).astype(bool)
                mask_map[stem] = mask
            else:
                print(f"No mask found for {stem}")
                # 如果没有 mask，创建一个全为 True 的 mask
                mask_map[stem] = np.ones((frame_h, frame_w), dtype=bool)

        logging.info("加载位姿完成，数量: %d", len(intr_map))
        return intr_map, c2w_map, scene_type, size_map, path_map, mask_map
    # dl3dv
    elif (scene_dir / "dense").exists():
        scene_type = "dl3dv"
        cam_dir = scene_dir / "dense" / "cam"
        rgb_dir = scene_dir / "dense" / "rgb"
        cam_files = sorted([f for f in os.listdir(cam_dir) if f.endswith(".npz")])
        intr_map: Dict[str, np.ndarray] = {}
        c2w_map: Dict[str, np.ndarray] = {}
        size_map: Dict[str, Tuple[int, int]] = {}
        path_map: Dict[str, str] = {}
        mask_map: Dict[str, np.ndarray] = {}

        mask_dir = scene_dir / "mask"

        for idx, cam_file in tqdm(enumerate(cam_files), total=len(cam_files)):
            stem = Path(cam_file).stem
            if image_names is not None and stem not in image_names:
                continue

            cam_file_path = os.path.join(cam_dir, cam_file)
            cam_data = np.load(cam_file_path)
            if "intrinsic" in cam_data:
                intrinsic = cam_data["intrinsic"]
                fx = intrinsic[0, 0]
                fy = intrinsic[1, 1]
                cx = intrinsic[0, 2]
                cy = intrinsic[1, 2]
            elif "intrinsics" in cam_data:
                # Fallback for ScanNet-like format
                intrinsics = cam_data["intrinsics"]
                fx = intrinsics[0, 0]
                fy = intrinsics[1, 1]
                cx = intrinsics[0, 2]
                cy = intrinsics[1, 2]
            else:
                raise ValueError(
                    f"No intrinsics found in {cam_file}. Available keys: {list(cam_data.keys())}"
                )
            c2w = cam_data["pose"]
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            intr_map[stem] = K
            c2w_map[stem] = c2w

            # DL3DV 通常在 npz 中包含尺寸，或者需要读取图片
            img_path = rgb_dir / f"{stem}.png"
            if not img_path.exists():
                img_path = rgb_dir / f"{stem}.jpg"

            if "width" in cam_data and "height" in cam_data:
                img_w, img_h = int(cam_data["width"]), int(cam_data["height"])
                size_map[stem] = (img_w, img_h)
            else:
                # 如果没有尺寸，尝试读取第一张图获取尺寸
                with Image.open(img_path) as img:
                    img_w, img_h = img.size
                    size_map[stem] = (img_w, img_h)
            path_map[stem] = str(img_path)

            # 加载 mask
            mask_path = mask_dir / f"{stem}.png"
            if mask_path.exists():
                mask = load_mask(mask_path)
                # 调整 mask 尺寸以匹配图像尺寸
                if mask.shape[:2] != (img_h, img_w):
                    mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
                    mask_resized = mask_img.resize((img_w, img_h), resample=Image.NEAREST)
                    mask = (np.array(mask_resized) > 0).astype(bool)
                mask_map[stem] = mask
            else:
                # 如果没有 mask，创建一个全为 True 的 mask
                mask_map[stem] = np.ones((img_h, img_w), dtype=bool)

        logging.info("加载位姿完成，数量: %d", len(intr_map))
        return intr_map, c2w_map, scene_type, size_map, path_map, mask_map
    # ScanNet v2 官方结构: color/, intrinsic/intrinsic_color.txt, pose/*.txt
    # 需在 cam+color 分支之前，避免与仅有 cam npz 的变体混淆
    elif (
        (scene_dir / "color").exists()
        and (scene_dir / "intrinsic").exists()
        and (scene_dir / "pose").exists()
    ):
        scene_type = "scannet"
        rgb_dir = scene_dir / "color"
        pose_dir = scene_dir / "pose"
        intrinsic_dir = scene_dir / "intrinsic"
        intr_file = intrinsic_dir / "intrinsic_color.txt"
        if not intr_file.exists():
            intr_file = intrinsic_dir / "intrinsic_depth.txt"
        if not intr_file.exists():
            raise FileNotFoundError(f"Intrinsic file not found in {intrinsic_dir}")
        intr_4x4 = np.loadtxt(str(intr_file), dtype=np.float32)
        if intr_4x4.size == 16:
            intr_4x4 = intr_4x4.reshape(4, 4)
        K = intr_4x4[:3, :3].astype(np.float32)

        intr_map: Dict[str, np.ndarray] = {}
        c2w_map: Dict[str, np.ndarray] = {}
        size_map: Dict[str, Tuple[int, int]] = {}
        path_map: Dict[str, str] = {}
        mask_map: Dict[str, np.ndarray] = {}
        mask_dir = scene_dir / "mask"

        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for img_path in sorted(rgb_dir.glob(ext)):
                stem = img_path.stem
                if image_names is not None and stem not in image_names:
                    continue
                pose_file = pose_dir / f"{stem}.txt"
                if not pose_file.exists():
                    continue
                c2w = np.loadtxt(str(pose_file), dtype=np.float32)
                if c2w.size == 16:
                    c2w = c2w.reshape(4, 4)
                if c2w.shape != (4, 4):
                    continue
                intr_map[stem] = K.copy()
                c2w_map[stem] = c2w.astype(np.float32)
                path_map[stem] = str(img_path)
                with Image.open(img_path) as img:
                    img_w, img_h = img.size
                size_map[stem] = (img_w, img_h)

                mask_path = mask_dir / f"{stem}.png"
                if mask_path.exists():
                    mask = load_mask(mask_path)
                    if mask.shape[:2] != (img_h, img_w):
                        mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
                        mask_resized = mask_img.resize((img_w, img_h), resample=Image.NEAREST)
                        mask = (np.array(mask_resized) > 0).astype(bool)
                    mask_map[stem] = mask
                else:
                    mask_map[stem] = np.ones((img_h, img_w), dtype=bool)

        if not intr_map:
            raise ValueError(f"No valid color/pose pairs in {scene_dir}")
        logging.info(
            "加载位姿完成 (ScanNet v2: color + intrinsic + pose)，数量: %d",
            len(intr_map),
        )
        return intr_map, c2w_map, scene_type, size_map, path_map, mask_map
    # scannet (cam npz + color)
    else:
        scene_type = "scannet"
        cam_dir = scene_dir / "cam"
        rgb_dir = scene_dir / "color"
        cam_files = sorted([f for f in os.listdir(cam_dir) if f.endswith(".npz")])
        intr_map: Dict[str, np.ndarray] = {}
        c2w_map: Dict[str, np.ndarray] = {}
        size_map: Dict[str, Tuple[int, int]] = {}
        path_map: Dict[str, str] = {}
        mask_map: Dict[str, np.ndarray] = {}

        mask_dir = scene_dir / "mask"

        for idx, cam_file in tqdm(enumerate(cam_files), total=len(cam_files)):
            stem = Path(cam_file).stem
            if image_names is not None and stem not in image_names:
                continue

            cam_file_path = os.path.join(cam_dir, cam_file)
            cam_data = np.load(cam_file_path)
            if "intrinsic" in cam_data:
                intrinsic = cam_data["intrinsic"]
                fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
            elif "intrinsics" in cam_data:
                intrinsics = cam_data["intrinsics"]
                fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
            else:
                raise ValueError(
                    f"No intrinsics found in {cam_file}. Available keys: {list(cam_data.keys())}"
                )

            c2w = cam_data["pose"]
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            intr_map[stem] = K
            c2w_map[stem] = c2w

            # 查找图片，支持 png, jpg 和 npz
            img_path = rgb_dir / f"{stem}.png"
            if not img_path.exists():
                img_path = rgb_dir / f"{stem}.jpg"
            if not img_path.exists():
                img_path = rgb_dir / f"{stem}.npz"

            if "width" in cam_data and "height" in cam_data:
                img_w, img_h = int(cam_data["width"]), int(cam_data["height"])
                size_map[stem] = (img_w, img_h)
            else:
                if img_path.suffix in [".png", ".jpg"]:
                    with Image.open(img_path) as img:
                        img_w, img_h = img.size
                        size_map[stem] = (img_w, img_h)
                elif img_path.suffix == ".npz":
                    img_data = np.load(img_path)
                    # 尝试常见的 key
                    if "image" in img_data:
                        h, w = img_data["image"].shape[:2]
                    elif "color" in img_data:
                        h, w = img_data["color"].shape[:2]
                    else:
                        raise ValueError(f"No image found in {img_path}")
                    img_w, img_h = w, h
                    size_map[stem] = (img_w, img_h)
            path_map[stem] = str(img_path)

            # 加载 mask
            mask_path = mask_dir / f"{stem}.png"
            if mask_path.exists():
                mask = load_mask(mask_path)
                # 调整 mask 尺寸以匹配图像尺寸
                if mask.shape[:2] != (img_h, img_w):
                    mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
                    mask_resized = mask_img.resize((img_w, img_h), resample=Image.NEAREST)
                    mask = (np.array(mask_resized) > 0).astype(bool)
                mask_map[stem] = mask
            else:
                # 如果没有 mask，创建一个全为 True 的 mask
                mask_map[stem] = np.ones((img_h, img_w), dtype=bool)

        logging.info("加载位姿完成，数量: %d", len(intr_map))
        return intr_map, c2w_map, scene_type, size_map, path_map, mask_map


def resize_mask_to_depth(mask: np.ndarray, depth: np.ndarray) -> np.ndarray:
    if mask.shape == depth.shape:
        return mask
    mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
    mask_resized = mask_img.resize((depth.shape[1], depth.shape[0]), resample=Image.NEAREST)
    return (np.array(mask_resized) > 0).astype(bool)


def remove_flying_pixels(depth_m: np.ndarray, k: int = 5, thr_m: float = 0.01) -> np.ndarray:
    """
    利用中值滤波删除深度飞点，返回与输入同尺寸的米单位深度。
    飞点会被置为 0。
    """
    if k <= 1:
        return depth_m.astype(np.float32)
    # 处理 NaN / inf
    depth_safe = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
    depth_mm = np.clip(depth_safe * 1000.0, 0, 65535).astype(np.uint16)
    med_mm = cv2.medianBlur(depth_mm, k)

    depth = depth_mm.astype(np.float32) / 1000.0
    med = med_mm.astype(np.float32) / 1000.0

    valid = depth > 0
    outlier = valid & (np.abs(depth - med) > thr_m)

    out = depth.copy()
    out[outlier] = 0.0  # 或 np.nan，根据下游需求
    return out


def mask_depth_to_points(
    mask: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
) -> np.ndarray:
    """将 mask 内像素 + 深度投影到世界坐标，返回 (N,3)。"""
    if mask.shape != depth.shape:
        mask = resize_mask_to_depth(mask, depth)

    depth_filtered = depth.copy()
    depth_filtered[~mask] = 0
    depth_filtered = remove_flying_pixels(depth_filtered)

    ys, xs = np.where(mask)
    if ys.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    d = depth_filtered[ys, xs].astype(np.float32)
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


def compute_obb(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """计算 OBB（定向包围盒），返回 (transform, extents)。
    
    Returns:
        transform: 4x4 变换矩阵（从 OBB 局部坐标系到世界坐标系）
        extents: (3,) OBB 的尺寸
    """
    if pts.shape[0] < 3:
        raise ValueError(f"OBB 计算至少需要 3 个点，当前: {pts.shape[0]}")

    T, extents = trimesh.bounds.oriented_bounds(pts)
    # oriented_bounds 返回的是“将点变到 OBB 局部坐标系”的矩阵，需要取逆才能作为 box 的 world transform
    T = np.linalg.inv(T).astype(np.float32)
    extents = extents.astype(np.float32)
    return T, extents


def obb_corners(transform: np.ndarray, extents: np.ndarray) -> np.ndarray:
    """计算 OBB 的 8 个角点（世界坐标）。
    
    Args:
        transform: 4x4 变换矩阵
        extents: (3,) OBB 尺寸
    
    Returns:
        corners: (8, 3) 8 个角点的世界坐标
    """
    # OBB 局部坐标系的 8 个角点
    half = extents * 0.5
    local_corners = np.array([
        [-half[0], -half[1], -half[2]],
        [-half[0], -half[1],  half[2]],
        [-half[0],  half[1], -half[2]],
        [-half[0],  half[1],  half[2]],
        [ half[0], -half[1], -half[2]],
        [ half[0], -half[1],  half[2]],
        [ half[0],  half[1], -half[2]],
        [ half[0],  half[1],  half[2]],
    ], dtype=np.float32)
    
    # 变换到世界坐标
    ones = np.ones((8, 1), dtype=np.float32)
    local_corners_homo = np.concatenate([local_corners, ones], axis=1)
    world_corners = (local_corners_homo @ transform.T)[:, :3]
    return world_corners


def bbox_corners_from_obb(transform: np.ndarray, extents: np.ndarray) -> List[Dict[str, float]]:
    """将 OBB (transform, extents) 转成 8 个角点的 JSON 格式。"""
    corners = obb_corners(transform, extents)
    return [{"x": float(x), "y": float(y), "z": float(z)} for x, y, z in corners]


def obb_overlap(
    transform_a: np.ndarray,
    extents_a: np.ndarray,
    transform_b: np.ndarray,
    extents_b: np.ndarray,
    intersect_ratio: float = 0.33,
) -> bool:
    """计算两个 OBB 的交叠体积，判断是否重叠。
    
    Args:
        transform_a: 4x4 变换矩阵
        extents_a: (3,) OBB 尺寸
        transform_b: 4x4 变换矩阵
        extents_b: (3,) OBB 尺寸
        intersect_ratio: 交叠体积占较小盒体积的最小比例
    
    Returns:
        是否重叠
    """
    obb_a = trimesh.creation.box(extents=extents_a, transform=transform_a)
    obb_b = trimesh.creation.box(extents=extents_b, transform=transform_b)
    inter = obb_a.intersection(obb_b)

    vol_inter = inter.volume if inter.is_volume else 0.0
    if vol_inter <= 0:
        return False

    vol_a = float(np.prod(extents_a))
    vol_b = float(np.prod(extents_b))
    min_vol = max(1e-9, min(vol_a, vol_b))
    return vol_inter / min_vol >= intersect_ratio


def obb_contains(
    transform_a: np.ndarray,
    extents_a: np.ndarray,
    transform_b: np.ndarray,
    extents_b: np.ndarray,
) -> bool:
    """判断 OBB A 是否包含 OBB B：B 的 8 个角点都落在 A 的局部盒内。"""
    corners_b = obb_corners(transform_b, extents_b)
    T_a_inv = np.linalg.inv(transform_a)
    ones = np.ones((8, 1), dtype=np.float32)
    corners_b_homo = np.concatenate([corners_b, ones], axis=1)
    corners_b_local = (corners_b_homo @ T_a_inv.T)[:, :3]
    half_a = extents_a * 0.5
    return np.all(np.abs(corners_b_local) <= half_a + 1e-6)


def filter_outliers(points: np.ndarray, z_thresh: float = 3.5) -> np.ndarray:
    """
    用中位数+MAD 计算逐轴 Z 分数，删除 |Z| 最大的 10% 点。
    这样不依赖固定阈值，按比例裁掉尾部离群。
    """
    if points.shape[0] < 5:
        return points
    med = np.median(points, axis=0)  # (3,)
    mad = np.median(np.abs(points - med), axis=0)  # (3,)
    mad[mad < 1e-6] = 1e-6  # 避免除零
    z = 0.6745 * (points - med) / mad  # (N,3)
    # 依据每个点的最大 |z| 进行排序，删除最靠后的 10%
    max_abs_z = np.abs(z).max(axis=1)  # (N,)
    n = points.shape[0]
    drop = int(max(1, round(n * 0.10)))
    if drop <= 0 or n - drop < 1:
        return points
    # 获取需要保留的索引（按 max_abs_z 升序保留前 n-drop 个）
    keep_idx = np.argsort(max_abs_z)[: n - drop]
    return points[keep_idx]


def erode_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return mask
    selem = morphology.square(max(1, int(pixels)))
    return morphology.binary_erosion(mask, selem)


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


def voxelize(points: np.ndarray, voxel_size: float) -> set[Tuple[int, int, int]]:
    if points.shape[0] == 0:
        return set()
    vox = np.floor(points / voxel_size).astype(np.int64)
    return set(map(tuple, vox))


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


def render_depths_with_gaussians(
    model_path: Path, 
    iteration: int, 
    intr_map: Dict[str, np.ndarray], 
    c2w_map: Dict[str, np.ndarray],
    size_map: Dict[str, Tuple[int, int]],
    path_map: Dict[str, str],
    image_names: Optional[set[str]] = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    使用 3DGS 渲染深度与颜色，不再初始化完整的 Scene，而是直接构造 Camera。
    """
    from utils.graphics_utils import fov2focal, focal2fov
    from scene.cameras import Camera

    cfg = load_gaussian_cfg(model_path)
    dataset = ModelParams(argparse.ArgumentParser(), sentinel=True).extract(cfg)
    pipeline = PipelineParams(argparse.ArgumentParser()).extract(build_pipeline_defaults())

    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.load_ply(str(model_path / "point_cloud" / f"iteration_30000" / "point_cloud.ply"))
    
    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")

    depth_map: Dict[str, np.ndarray] = {}
    color_map: Dict[str, np.ndarray] = {}
    
    # 确定需要渲染的 stems
    if image_names is not None:
        target_stems = {Path(name).stem for name in image_names}
        stems_to_render = [s for s in target_stems if s in c2w_map]
    else:
        stems_to_render = list(c2w_map.keys())

    logging.info("准备渲染相机数量: %d", len(stems_to_render))

    with torch.no_grad():
        for stem in tqdm(stems_to_render, desc="Rendering depths"):
            # 获取位姿和内参
            c2w = c2w_map[stem]
            K = intr_map[stem]
            width, height = size_map[stem]
            image_path = path_map[stem]

            # 3DGS 渲染需要 world2view (R, T)
            # 根据 dataset_readers.py 的逻辑：
            # 1. R 应该是 c2w 的旋转矩阵 (因为 Camera 内部会做 R.transpose() 得到 w2v 的旋转部分)
            # 2. T 应该是 w2v 的平移向量 (即 -R_c2w.T @ T_c2w)
            R_c2w = c2w[:3, :3]
            T_c2w = c2w[:3, 3]
            
            R_w2v = R_c2w.T
            T_w2v = -R_w2v @ T_c2w
            
            # 计算 FoV (与 dataset_readers.py 274-275行一致)
            fx = K[0, 0]
            fy = K[1, 1]
            fovx = focal2fov(fx, width)
            fovy = focal2fov(fy, height)

            # 构造 Camera 对象，确保参数与 PGSR 内部创建的一致
            cam = Camera(
                colmap_id=0,
                R=R_c2w,      # 传入 c2w 旋转部分
                T=T_w2v,      # 传入 w2v 平移部分
                FoVx=fovx,
                FoVy=fovy,
                image_width=width,
                image_height=height,
                image_path=image_path,
                image_name=stem,
                uid=0,
                preload_img=False, # 渲染不需要预加载图片
                data_device="cuda"
            )

            out = render(cam, gaussians, pipeline, background)
            depth_np = out["plane_depth"].squeeze().detach().cpu().numpy()
            depth_map[stem] = depth_np
            color = out["render"].permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy()
            color_map[stem] = color

            # --- 临时可视化深度图 ---
            # debug_depth_dir = Path("debug_depth_vis")
            # debug_depth_dir.mkdir(exist_ok=True)
            # valid_depth = depth_np[depth_np > 0]
            # if valid_depth.size > 0:
            #     d_min, d_max = valid_depth.min(), valid_depth.max()
            #     depth_vis = (depth_np - d_min) / (d_max - d_min + 1e-8)
            #     depth_vis = (np.clip(depth_vis, 0, 1) * 255).astype(np.uint8)
            #     Image.fromarray(depth_vis).save(debug_depth_dir / f"{stem}_depth.png")
            # -----------------------

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
    if include_labels is None:
        return None
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

    for idx, box in enumerate(bboxes):
        if include_labels is not None and box["label"] not in include_labels:
            continue
        col = label_color(box["label"])

        transform = np.array(box["obb_transform"], dtype=np.float32)
        extents = np.array(box["obb_extents"], dtype=np.float32)
        mesh_box = trimesh.creation.box(extents=extents, transform=transform)
        
        # 线框 bbox
        edges = mesh_box.edges_unique
        edge_verts = mesh_box.vertices[edges]
        path = trimesh.load_path(edge_verts.reshape(-1, 3))
        path.colors = np.tile(col[None, :], (len(path.entities), 1))
        path.metadata = {"label": box["label"]}
        scene_tm.add_geometry(path, node_name=f"bbox_{idx+1}_{box['label']}")

        # 小球标记中心，便于在查看器里看到标签颜色
        try:
            sph = trimesh.creation.icosphere(subdivisions=2, radius=float(max(extents)) * 0.01)
            sph.apply_translation(transform[:3, 3])
            sph.visual.vertex_colors = np.tile(col[None, :], (len(sph.vertices), 1))
            sph.metadata = {"label": box["label"]}
            scene_tm.add_geometry(sph, node_name=f"bbox_center_{idx+1}_{box['label']}")
        except Exception:
            pass

    glb_path = output_dir / f"{scene}_points_bbox.glb"
    scene_tm.export(glb_path)
    logging.info("点云+BBox glb 已保存: %s (点数: %d, bbox: %d)", glb_path, points.shape[0], len(bboxes))
    return glb_path


def log_rerun_scene(
    points_all: Optional[np.ndarray],
    colors_all: Optional[np.ndarray],
    bboxes: List[Dict],
    instance_clouds: List[Tuple[str, str, np.ndarray]],
    c2w_map: Dict[str, np.ndarray],
    intr_map: Dict[str, np.ndarray],
    color_map: Dict[str, np.ndarray],
    cam_max_size: int,
    scene: str,
    scene_type: Literal["scannetppv2", "dl3dv"],
    include_labels: Optional[set[str]],
    rerun_addr: Optional[str],
    rerun_save_path: Optional[Path],
) -> None:
    """将点云与 3D bbox 发往 rerun 0.24.1 进行可视化。"""
    try:
        import rerun as rr
    except ImportError:
        logging.warning("未安装 rerun，跳过可视化")
        return

    rr.init(f"3d_bounding_instance_gs/{scene}", spawn=False)
    # nerfstudio/ScanNet++（OpenGL 相机，前向 -Z，Y 向上）推荐坐标系，避免视角倒置
    if scene_type == "scannetppv2":
        rr.log("/", rr.ViewCoordinates.RUB)  # Right, Up, Back

    if rerun_addr:
        try:
            rr.connect(rerun_addr)
            logging.info("rerun 已连接: %s", rerun_addr)
        except Exception as exc:  # noqa: BLE001
            logging.warning("rerun 连接失败 (%s): %s", rerun_addr, exc)

    # 整体点云（others）
    if points_all is not None and points_all.shape[0] > 0:
        log_kwargs = {}
        if colors_all is not None and colors_all.shape[0] == points_all.shape[0]:
            log_kwargs["colors"] = colors_all
        rr.log("others", rr.Points3D(points_all, **log_kwargs))

    # 按实例分别记录，便于在 rerun 中单独开/关
    for box in bboxes:
        if include_labels is not None and box["label"] not in include_labels:
            continue
        ins_id = box["ins_id"]
        label = box["label"]
        col = label_color(label)[:3]

        transform = np.array(box["obb_transform"], dtype=np.float32)
        extents = np.array(box["obb_extents"], dtype=np.float32)
        center = transform[:3, 3]
        quat_xyzw = R.from_matrix(transform[:3, :3]).as_quat().astype(np.float32)  # xyzw
        rr.log(
            f"instances/{label}/{ins_id}/bbox",
            rr.Boxes3D(
                centers=np.array([center], dtype=np.float32),
                half_sizes=np.array([extents * 0.5], dtype=np.float32),
                quaternions=[rr.Quaternion(xyzw=quat_xyzw)],
                colors=np.array([col], dtype=np.uint8),
                labels=[f"{label}:{ins_id}"],
            ),
        )
        matched = [pts for lbl, iid, pts in instance_clouds if lbl == label and iid == ins_id]
        if matched:
            pts = matched[0]
            rr.log(
                f"instances/{label}/{ins_id}/points",
                rr.Points3D(
                    pts,
                    colors=np.tile(col[None, :], (pts.shape[0], 1)),
                ),
            )

    # 相机与渲染彩色图（可选下采样）
    # for stem, c2w in c2w_map.items():
    #     if stem not in intr_map or stem not in color_map:
    #         continue
    #     K = intr_map[stem]
    #     fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    #     img = color_map[stem]
    #     img_uint8 = (img * 255.0).clip(0, 255).astype(np.uint8)
    #     max_side = max(img_uint8.shape[0], img_uint8.shape[1])
    #     if cam_max_size > 0 and max_side > cam_max_size:
    #         scale = cam_max_size / float(max_side)
    #         new_size = (max(1, int(round(img_uint8.shape[1] * scale))), max(1, int(round(img_uint8.shape[0] * scale))))
    #         pil_img = Image.fromarray(img_uint8)
    #         pil_img = pil_img.resize(new_size, resample=Image.BILINEAR)
    #         img_uint8 = np.array(pil_img, dtype=np.uint8)
    #     H, W = img_uint8.shape[0], img_uint8.shape[1]
    #     rr.log(
    #         f"cameras/{stem}",
    #         rr.Transform3D(
    #             translation=c2w[:3, 3].astype(np.float32),
    #             mat3x3=c2w[:3, :3].astype(np.float32),
    #         ),
    #     )
    #     rr.log(
    #         f"cameras/{stem}/pinhole",
    #         rr.Pinhole(
    #             focal_length=np.array([fx, fy], dtype=np.float32),
    #             principal_point=np.array([cx, cy], dtype=np.float32),
    #             resolution=np.array([W, H], dtype=np.float32),
    #         ),
    #     )
    #     rr.log(f"cameras/{stem}/image", rr.Image(img_uint8))

    # 在所有 log 操作完成后执行 save
    if rerun_save_path:
        rerun_save_path = Path(rerun_save_path)
        rerun_save_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            rr.save(str(rerun_save_path))
            logging.info("rerun 日志已保存到: %s", rerun_save_path)
        except Exception as exc:  # noqa: BLE001
            logging.warning("保存 rerun 日志失败: %s", exc)


def process_scene(
    scene: str,
    data_root: Path,
    mask_root: Path,
    scene_json_dir: Path,
    model_path: Path,
    iteration: int,
    output_dir: Path,
    voxel_size: float,
    min_points: int,
    erode_pixels: int,
    instance_max_points: int,
    include_labels: Optional[set[str]],
    rerun_enabled: bool,
    rerun_addr: Optional[str],
    rerun_save_path: Optional[Path],
    cam_max_size: int,
) -> Path:
    scene_dir = data_root / scene

    # 尝试加载包含 100 张图的 JSON（先于 load_transforms，用于筛选视角）
    scene_json_path = scene_json_dir / f"{scene}.json"
    image_names = None
    if scene_json_path.exists():
        logging.info("加载场景图片列表: %s", scene_json_path)
        with scene_json_path.open("r", encoding="utf-8") as f:
            scene_data = json.load(f)
            per_image = scene_data.get("per_image", {})
            image_names = {Path(name).stem for name in per_image.keys()}
            logging.info("场景包含 %d 张图片", len(image_names))
    else:
        logging.warning("场景图片列表不存在: %s，将渲染所有相机", scene_json_path)

    intr_map, c2w_map, scene_type, size_map, path_map, mask_map = load_transforms(
        scene_dir, image_names=image_names
    )
    print("scene_type:", scene_type)

    depth_map, color_map = render_depths_with_gaussians(
        model_path, iteration, 
        intr_map=intr_map, c2w_map=c2w_map, 
        size_map=size_map, path_map=path_map,
        image_names=image_names
    )

    mask_index_path = mask_root / scene / "mask_index.json"
    with mask_index_path.open("r", encoding="utf-8") as f:
        mask_index = json.load(f)
    items = mask_index.get("items", [])
    mask_score_map: Dict[str, float] = {}
    mask_to_item: Dict[str, Dict] = {}
    for it in items:
        path = it.get("mask_path")
        score = it.get("score")
        if isinstance(path, str) and isinstance(score, (int, float)):
            mask_score_map[path] = float(score)
            mask_to_item[path] = it

    # 加载本地 Qwen 模型
    vlm_model, vlm_processor = load_vlm_model(DEFAULT_VLM_MODEL_PATH)

    results: List[Dict] = []
    inst_counter = 1

    all_depth_points: List[np.ndarray] = []
    all_depth_colors: List[np.ndarray] = []
    pts_all: Optional[np.ndarray] = None
    cols_all: Optional[np.ndarray] = None
    instance_clouds: List[Tuple[str, str, np.ndarray]] = []

    for stem, depth in depth_map.items():
        if stem not in intr_map or stem not in c2w_map:
            continue
        color = color_map.get(stem)
        pts_full, cols_full = depth_to_points(depth, intr_map[stem], c2w_map[stem], color=color)
        if pts_full.shape[0] > 0:
            all_depth_points.append(pts_full)
            if cols_full is not None:
                all_depth_colors.append(cols_full)

    # 按标签分组，逐标签处理，互不干扰
    label_groups: Dict[str, List[Dict]] = defaultdict(list)
    for item in items:
        label_groups[item["label"]].append(item)

    for label, label_items in label_groups.items():
        logging.info("处理标签 %s，共 %d 个 mask", label, len(label_items))
        instances: List[Dict] = []
        if len(label_items) > 1000:
            # 实例过多时均匀抽取 500 个进行重叠检查以加速
            sampled = np.linspace(0, len(label_items) - 1, num=500, dtype=int)
            label_items = [label_items[i] for i in sorted(set(int(x) for x in sampled))]
        for item in tqdm(label_items, desc=f"Processing {label}"):



            image_name = item["image"]
            mask_path = Path(item["mask_path"])
            # if mask_score_map.get(item["mask_path"]) <= 0.75 :
            #     continue

            stem = Path(image_name).stem


            if stem not in intr_map or stem not in c2w_map:
                logging.warning("找不到相机参数，跳过: %s", image_name)
                continue

            if stem not in depth_map:
                logging.warning("渲染深度缺失，跳过: %s", stem)
                continue

            depth = depth_map[stem]
            # 从 item 中读取 mask_rle，如果不存在则退而求其次加载 PNG
            if "mask_rle" in item:
                rle = item["mask_rle"]
                mask = mask_utils.decode(rle).astype(bool)
            else:
                mask = load_mask(mask_path)

            mask = erode_mask(mask, erode_pixels)
            # 与 image mask 求与，被 mask 的部分不要投射
            if stem in mask_map:
                image_mask = mask_map[stem]
                if image_mask.shape != mask.shape:
                    mask_img = Image.fromarray(image_mask.astype(np.uint8) * 255)
                    mask_resized = mask_img.resize(
                        (mask.shape[1], mask.shape[0]), resample=Image.NEAREST
                    )
                    image_mask = (np.array(mask_resized) > 0).astype(bool)
                mask = mask & image_mask

            points = mask_depth_to_points(mask, depth, intr_map[stem], c2w_map[stem])
            points = filter_outliers(points)
            if points.shape[0] < min_points:
                continue

            # 计算 OBB
            transform, extents = compute_obb(points)
            # 编码 mask 并保存
            mask_encoding = encode_mask(mask)
            new_inst = {
                "points": [points],
                "obb_transform": transform,
                "obb_extents": extents,
                "images": {str(mask_path)},
                "mask_encodings": {str(mask_path): mask_encoding},
            }

            # 判断是否与已有实例合并
            def share_same_frame(img_a: str, img_b: str) -> bool:
                return Path(img_a).parent.name == Path(img_b).parent.name

            merge_indices = []
            indices_to_check = range(len(instances))
            if len(instances) > 1000:
                # 实例过多时均匀抽取 500 个进行重叠检查以加速
                sampled = np.linspace(0, len(instances) - 1, num=500, dtype=int)
                indices_to_check = sorted(set(int(x) for x in sampled))

            for idx in indices_to_check:
                inst = instances[idx]
                overlap = obb_overlap(
                    inst["obb_transform"],
                    inst["obb_extents"],
                    transform,
                    extents,
                )
                # 检查新 mask 是否与该实例冲突
                # conflict_with_new = any(
                #     share_same_frame(img, next(iter(new_inst["images"])))
                #     for img in inst["images"]
                # )
                
                # if overlap and not conflict_with_new:
                #     # 还需要检查：如果要合并这个实例，会不会导致 merge_indices 里的实例互相冲突？
                #     can_merge_with_others = True
                #     for already_selected_idx in merge_indices:
                #         if any(share_same_frame(img1, img2) 
                #                for img1 in instances[already_selected_idx]["images"] 
                #                for img2 in inst["images"]):
                #             can_merge_with_others = False
                #             break
                    
                if overlap:
                    merge_indices.append(idx)

            if not merge_indices:
                instances.append(new_inst)
            else:
                merged = new_inst
                for idx in sorted(merge_indices, reverse=True):
                    inst = instances.pop(idx)
                    merged["points"].extend(inst["points"])
                    merged["images"] |= inst["images"]
                    # 合并 mask 编码
                    if "mask_encodings" not in merged:
                        merged["mask_encodings"] = {}
                    if "mask_encodings" in inst:
                        merged["mask_encodings"].update(inst["mask_encodings"])
                # 重新计算合并后的 OBB
                all_pts = np.concatenate(merged["points"], axis=0)
                if instance_max_points > 0 and all_pts.shape[0] > instance_max_points:
                    orig_n = all_pts.shape[0]
                    idx = np.random.choice(orig_n, instance_max_points, replace=False)
                    all_pts = all_pts[idx]
                    # logging.info("实例合并点云下采样: %d -> %d", orig_n, all_pts.shape[0])
                merged["obb_transform"], merged["obb_extents"] = compute_obb(all_pts)
                instances.append(merged)

        logging.info("标签 %s 初步合并得到 %d 个实例", label, len(instances))

        # 1. 置信度过滤与 VLM 验证
        qualified_instances = []
        for inst in tqdm(instances, desc=f"Verifying {label} instances"):
            max_score = max((mask_score_map.get(img) or 0.0) for img in inst["images"])
            # 如果有 mask 分数 >= 0.9，直接保留
            if max_score >= 0.9:
                qualified_instances.append(inst)
                continue
            
            # 只验证最高score在0.75和0.9之间的
            if max_score < 0.75:
                continue

            # 否则，取分数最高的 mask 进行 VLM 验证
            highest_conf_mask_path = max(inst["images"], key=lambda img: mask_score_map.get(img, 0.0))
            item = mask_to_item[highest_conf_mask_path]
            image_name = item["image"]
            
            # 获取原始图片路径
            if scene_type == "scannetppv2":
                raw_image_path = data_root / scene / "dslr" / "resized_undistorted_images" / image_name
            elif scene_type == "dl3dv":
                raw_image_path = data_root / scene / "dense" / "rgb" / image_name
            else:
                raw_image_path = data_root / scene /  "resized_undistorted_images" / image_name

            if not raw_image_path.exists():
                logging.warning("原始图片不存在，跳过 VLM 验证: %s", raw_image_path)
                continue

            try:
                raw_image = Image.open(raw_image_path).convert("RGB")
                
                if "mask_rle" in item:
                    rle = item["mask_rle"]
                    mask_np = mask_utils.decode(rle).astype(bool)
                else:
                    mask_image = Image.open(highest_conf_mask_path).convert("L")
                    mask_np = np.array(mask_image) > 0
                
                # 准备 RLE 和 BBox
                orig_h, orig_w = mask_np.shape
                # 如果没有从 item 获取到 rle，则重新编码
                if "mask_rle" not in item:
                    rle = mask_utils.encode(np.asfortranarray(mask_np.astype(np.uint8)))
                    rle['counts'] = rle['counts'].decode('ascii')
                else:
                    rle = item["mask_rle"]
                
                bbox = mask_utils.toBbox(rle)  # [x, y, w, h]
                bbox_xyxy = np.array([[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]])

                # 1. 生成 Zoom-in 图像并获取颜色
                object_data = {
                    "labels": [{"noun_phrase": label}],
                    "segmentation": rle,
                }
                zoomed_image, color_hex = render_zoom_in(object_data, raw_image, mask_alpha=0.5)

                # 2. 使用相同颜色生成 Overlay 图像
                viz = Visualizer(np.array(raw_image))
                viz.overlay_instances(
                    boxes=bbox_xyxy,
                    masks=[rle],
                    binary_masks=[mask_np],
                    assigned_colors=[color_hex],
                    alpha=0.5,
                )
                overlay_image = Image.fromarray(viz.output.get_image())
                
                if verify_label_with_vlm(vlm_model, vlm_processor, raw_image, overlay_image, zoomed_image, label):
                    logging.info("VLM 召回了低分实例 (label: %s, highest_score: %.2f, image: %s)", 
                                 label, mask_score_map.get(highest_conf_mask_path, 0.0), raw_image_path)
                    qualified_instances.append(inst)
            except Exception as e:
                logging.error("VLM 验证过程出错: %s", e)
                import traceback
                traceback.print_exc()

        # 2. 对同一类别的实例进行二次 overlap 检测 (阈值 0.8)
        # final_instances = []
        # for q_inst in qualified_instances:
        #     q_transform = q_inst["obb_transform"]
        #     q_extents = q_inst["obb_extents"]
            
        #     merged = False
        #     for f_idx, f_inst in enumerate(final_instances):
        #         # 如果重叠度 > 0.8，则融合
        #         if obb_overlap(q_transform, q_extents, f_inst["obb_transform"], f_inst["obb_extents"], intersect_ratio=0.8):
        #             f_inst["points"].extend(q_inst["points"])
        #             f_inst["images"] |= q_inst["images"]
        #             # 合并 mask 编码
        #             if "mask_encodings" not in f_inst:
        #                 f_inst["mask_encodings"] = {}
        #             if "mask_encodings" in q_inst:
        #                 f_inst["mask_encodings"].update(q_inst["mask_encodings"])
                    
        #             # 重新计算融合后的 OBB
        #             all_pts = np.concatenate(f_inst["points"], axis=0)
        #             if instance_max_points > 0 and all_pts.shape[0] > instance_max_points:
        #                 idx = np.random.choice(all_pts.shape[0], instance_max_points, replace=False)
        #                 all_pts = all_pts[idx]
        #             f_inst["obb_transform"], f_inst["obb_extents"] = compute_obb(all_pts)
        #             merged = True
        #             break
            
        #     if not merged:
        #         final_instances.append(q_inst)
        final_instances = qualified_instances 
        logging.info("标签 %s 最终过滤与vlm召回后剩余 %d 个实例", label, len(qualified_instances))

        # 3. 记录最终结果
        for inst in final_instances:
            pts = np.concatenate(inst["points"], axis=0)
            transform = inst["obb_transform"]
            extents = inst["obb_extents"]
            bbox = bbox_corners_from_obb(transform, extents)
            
            # 准备 mask 编码，按 images 顺序排序
            mask_encodings_dict = inst.get("mask_encodings", {})
            mask_encodings_list = []
            
            sorted_images = sorted(inst["images"])
            
            # 使用相对路径保存图像路径 (相对于当前运行目录，通常是项目根目录)
            def to_rel(p):
                try:
                    # 如果已经是相对路径则直接返回，否则转为相对于当前目录的路径
                    if os.path.isabs(p):
                        return os.path.relpath(p, os.getcwd())
                    return p
                except Exception:
                    return str(p)

            rel_images = [to_rel(img) for img in sorted_images]
            highest_conf_mask = max(inst["images"], key=lambda img: mask_score_map.get(img, 0.0))
            rel_highest_conf_mask = to_rel(highest_conf_mask)

            for img_path in sorted_images:
                if img_path in mask_encodings_dict:
                    mask_encodings_list.append(mask_encodings_dict[img_path])
            
            results.append(
                {
                    "ins_id": str(inst_counter),
                    "label": label,
                    "bounding_box": bbox,
                    "obb_transform": transform.tolist(),
                    "obb_extents": extents.tolist(),
                    "images": rel_images,
                    "highest_confidence_mask": rel_highest_conf_mask,
                    "mask_encodings": mask_encodings_list,
                }
            )
            
            # 用于 rerun 可视化的点云
            pts_vis = pts
            if instance_max_points > 0 and pts.shape[0] > instance_max_points:
                idx = np.random.choice(pts.shape[0], instance_max_points, replace=False)
                pts_vis = pts[idx]
            instance_clouds.append((label, str(inst_counter), pts_vis))
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

    if rerun_enabled:
        log_rerun_scene(
            pts_all,
            cols_all,
            results,
            instance_clouds,
            c2w_map,
            intr_map,
            color_map,
            cam_max_size,
            scene,
            scene_type=scene_type,
            include_labels=include_labels,
            rerun_addr=rerun_addr if rerun_addr else None,
            rerun_save_path=rerun_save_path,
        )

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 2D mask 投影生成 3D bounding boxes")
    parser.add_argument("--scene", default=DEFAULT_SCENE, help="场景名，如 0a5c013435")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="scannetppv2/data 根目录")
    parser.add_argument("--mask-root", type=Path, default=DEFAULT_MASK_ROOT, help="sam mask 根目录（含 mask_index.json）")
    parser.add_argument(
        "--scene-json-dir",
        type=Path,
        default=Path("scene_objects_Qwen3-VL-30B-A3B-Instruct"),
        help="每场景 JSON 所在目录（读取 {scene}.json 中的 per_image 图片列表）",
    )
    parser.add_argument("--model-path", "-m", type=Path, default=DEFAULT_GS_MODEL, help="3DGS 模型目录（含 cfg_args）")
    parser.add_argument("--iteration", type=int, default=-1, help="加载的迭代编号，-1 表示自动最新")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="输出 3D bbox 保存目录")
    parser.add_argument("--voxel-size", type=float, default=0.005, help="（保留参数，未使用）")
    parser.add_argument("--min-points", type=int, default=30, help="实例最少点数过滤")
    parser.add_argument("--erode-pixels", type=int, default=15, help="mask 形态学腐蚀像素，过滤边缘")
    parser.add_argument(
        "--instance-max-points",
        type=int,
        default=100000,
        help="每个实例点云的上限，用于 rerun 可视化下采样；<=0 表示不限制",
    )
    parser.add_argument(
        "--rerun-cam-max-size",
        type=int,
        default=128,
        help="相机图像长边的最大尺寸（像素），<=0 表示不缩放",
    )
    parser.add_argument(
        "--bbox-labels",
        type=str,
        # default="cleaning product",
        default=None,
        help="仅导出指定类别的 bbox 到 glb，逗号分隔，空则导出全部",
    )
    def str_to_bool(v: str) -> bool:
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser.add_argument(
        "--rerun",
        nargs="?",
        const=True,
        default=DEFAULT_RERUN,
        type=str_to_bool,
        help=f"启用 rerun 0.24.1 可视化（点云与 3D bbox），默认: {DEFAULT_RERUN}",
    )
    parser.add_argument(
        "--rerun-addr",
        type=str,
        default="",
        help="rerun viewer 地址，例如 0.0.0.0:9876；为空则不尝试连接",
    )
    parser.add_argument(
        "--rerun-save-rrd",
        type=Path,
        default=None,
        help="可选，将日志保存为 rrd 文件（便于远程下载/离线回放）",
    )
    return parser.parse_args()


def main() -> None:
    setup_logger()  # 初始控制台输出
    args = parse_args()
    
    # 重新配置日志到文件 logging/scene_name.json
    log_dir = Path("logging")
    log_file = log_dir / f"{args.scene}.json"
    setup_logger(log_file)
    logging.info("日志已同步保存至: %s", log_file)
    
    # 如果启用了 rerun 且未指定保存路径，自动设置为 output_dir/scene.rrd
    rerun_save_path = args.rerun_save_rrd
    if args.rerun and rerun_save_path is None:
        rerun_save_path = args.output_dir / f"{args.scene}.rrd"
    
    process_scene(
        scene=args.scene,
        data_root=args.data_root,
        mask_root=args.mask_root,
        scene_json_dir=args.scene_json_dir,
        model_path=args.model_path,
        iteration=args.iteration,
        output_dir=args.output_dir,
        min_points=args.min_points,
        voxel_size=args.voxel_size,
        erode_pixels=args.erode_pixels,
        instance_max_points=args.instance_max_points,
        include_labels=set([s for s in args.bbox_labels.split(",") if s.strip()]) if args.bbox_labels else None,
        rerun_enabled=args.rerun,
        rerun_addr=args.rerun_addr,
        rerun_save_path=rerun_save_path,
        cam_max_size=args.rerun_cam_max_size,
    )


if __name__ == "__main__":
    main()

# TODOs:
# 1. rerun中添加相机
# 2. 除了保存depth_points，还保存eroded mask points及他们的label，以便后续可视化。没有label的点，label赋值为others

# vlm进行判断，消除sam的错误mask

