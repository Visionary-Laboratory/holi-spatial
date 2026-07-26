#!/usr/bin/env python3
"""
为3D instance生成描述脚本

该脚本会：
1. 读取3D bounding box JSON文件（如output_3d_bounding/*.json）
2. 对每个instance，找到其highest_confidence_mask
3. 从mask_index.json中找到对应的mask_rle
4. 将mask叠加到原图上（画轮廓）
5. 调用VLM生成3D一致的描述（不包含2D图像位置信息）
6. 保存描述到JSON文件中
"""

import os
import json
import argparse
import base64
import io
from pathlib import Path
import numpy as np
import cv2
import re
from tqdm import tqdm

try:
    from pycocotools import mask as mask_util
except ImportError:
    print("Warning: pycocotools not installed, mask decoding may fail")
    mask_util = None

from PIL import Image

from openai import OpenAI

# 每个GPU处理的场景数量
PER_GPU_SIZE = 89


def resolve_vllm_api_url(vllm_api_url: str | None) -> str:
    return vllm_api_url or "http://localhost:8000/v1/chat/completions"


def init_vllm_client(vllm_api_url: str) -> OpenAI:
    api_base = vllm_api_url
    if api_base.endswith("/chat/completions"):
        api_base = api_base[: -len("/chat/completions")]
    return OpenAI(base_url=api_base, api_key="not-needed")


def decode_rle_mask(mask_encoding):
    """
    解码RLE格式的mask
    """
    if mask_util is None:
        raise ImportError("pycocotools is required for mask decoding")
    
    size = mask_encoding.get("size", [])
    counts = mask_encoding.get("counts", "")
    
    if isinstance(counts, str):
        # RLE字符串格式
        rle = {
            "size": size,
            "counts": counts
        }
        mask = mask_util.decode(rle)
    else:
        # 已经是数组格式
        rle = {
            "size": size,
            "counts": counts
        }
        mask = mask_util.decode(rle)
    
    return mask


def apply_mask_to_image(image_path, mask, line_width=3):
    """
    将mask应用到图像上，用红色轮廓圈出物体（只绘制轮廓，不填充）
    """
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    
    # 确保mask是二值化的uint8格式
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255
    else:
        mask = (mask > 0).astype(np.uint8) * 255
    
    # 调整mask尺寸以匹配图像尺寸
    img_h, img_w = img_array.shape[:2]
    mask_h, mask_w = mask.shape
    if mask_h != img_h or mask_w != img_w:
        mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    
    # 复制图像，在其上绘制轮廓
    masked_image = img_array.copy()
    
    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 绘制红色轮廓
    cv2.drawContours(masked_image, contours, -1, (255, 0, 0), line_width)  # 红色 (BGR格式)
    
    return Image.fromarray(masked_image)


def extract_scene_id_from_mask_path(mask_path):
    """
    从mask路径中提取场景ID
    例如: sam_masks_debug/0a7cc12c0e/DSC05865/bed.png -> 0a7cc12c0e
    """
    parts = mask_path.split('/')
    # 找到sam_masks_debug目录后的第一个目录名
    for i, part in enumerate(parts):
        if part in ['sam_masks_debug', 'sam_masks_debug_dl3dv'] and i + 1 < len(parts):
            return parts[i + 1]
    return None


def extract_image_name_from_mask_path(mask_path):
    """
    从mask路径中提取图片名称（不含扩展名）
    例如: sam_masks_debug/0a7cc12c0e/DSC05865/bed.png -> DSC05865
    例如: sam_masks_debug/c285c82ade/02dd3b53_DSC07112/blind_02.png -> 02dd3b53_DSC07112
    例如: sam_masks_debug_dl3dv/scene/frame_00098/chair_02.png -> frame_00098
    """
    parts = mask_path.split('/')
    # 找到最后一个目录名（图片名）
    if len(parts) >= 2:
        return parts[-2]
    return None


def find_original_image_path(dataset_root, mask_path, scene_id):
    """
    根据mask路径和场景ID找到原图路径
    """
    if not dataset_root:
        return None

    image_name = extract_image_name_from_mask_path(mask_path)
    if not image_name:
        return None

    dataset_root = os.path.normpath(str(dataset_root))
    base_name = Path(image_name).stem
    ext = Path(image_name).suffix
    ext_candidates = []
    if ext:
        ext_candidates.append(ext)
    for cand in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        if cand not in ext_candidates:
            ext_candidates.append(cand)

    scene_root = os.path.join(dataset_root, scene_id)
    candidate_roots = [scene_root]
    # 兼容常见子目录布局
    candidate_roots.append(os.path.join(scene_root, "dslr", "resized_undistorted_images"))
    candidate_roots.append(os.path.join(scene_root, "dense", "rgb"))
    # scannetv2: scannetv2/scans/scene0000_01/color
    candidate_roots.append(os.path.join(scene_root, "color"))

    for root in candidate_roots:
        for ext_cand in ext_candidates:
            cand = os.path.join(root, f"{base_name}{ext_cand}")
            if os.path.exists(cand):
                return cand

    return None


def find_mask_rle_from_mask_index(mask_index_path, image_name, label):
    """
    从mask_index.json中找到对应的mask_rle
    """
    if not os.path.exists(mask_index_path):
        return None
    
    with open(mask_index_path, 'r', encoding='utf-8') as f:
        mask_index_data = json.load(f)
    
    items = mask_index_data.get('items', [])
    for item in items:
        if item.get('image') == image_name and item.get('label') == label:
            return item.get('mask_rle')
    
    return None


def find_mask_rle_for_instance(instance: dict, image_name: str, label: str, mask_path_abs: str):
    """
    优先从 mask_index.json 获取 mask_rle；如果缺失（比如某些场景 mask_index.json 没有 mask_rle 字段），
    则回退到 bbox json 中该 instance 的 mask_encodings（与 highest_confidence_mask 对齐）。
    """
    # mask_rle = find_mask_rle_from_mask_index(mask_index_path, image_name, label)
    # if mask_rle:
    #     return mask_rle

    # fallback: use instance["mask_encodings"]
    encs = instance.get("mask_encodings", None)
    if not isinstance(encs, list) or len(encs) == 0:
        return None

    imgs = instance.get("images", None)
    if isinstance(imgs, list) and len(imgs) == len(encs):
        # best effort: align by exact path match
        idx = None
        try:
            idx = imgs.index(mask_path_abs)
        except ValueError:
            # fallback: align by path suffix (frame_dir/filename) to avoid basename collisions
            target_path = Path(mask_path_abs)
            target_suffix = os.path.join(target_path.parent.name, target_path.name)
            for i, p in enumerate(imgs):
                p_path = Path(p)
                cand_suffix = os.path.join(p_path.parent.name, p_path.name)
                if cand_suffix == target_suffix:
                    idx = i
                    break
        if idx is not None and 0 <= idx < len(encs):
            cand = encs[idx]
            if isinstance(cand, dict) and "size" in cand and "counts" in cand:
                return cand

    # fallback: if only one encoding present, use it
    if len(encs) == 1 and isinstance(encs[0], dict) and "size" in encs[0] and "counts" in encs[0]:
        return encs[0]

    return None


def image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def resize_image_max_side(image: Image.Image, max_image_side: int | None) -> Image.Image:
    if not max_image_side or max_image_side <= 0:
        return image
    width, height = image.size
    max_dim = max(width, height)
    if max_dim <= max_image_side:
        return image
    scale = max_image_side / max_dim
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
    return image.resize(new_size, resample=resample)


def call_vlm_for_3d_instance_description(image_path, label, client: OpenAI, max_image_side: int | None):
    """
    调用vLLM生成3D instance的描述
    注意：描述应该是3D一致的，不包含2D图像中的位置信息
    """
    prompt = f"""You are a visual description assistant for 3D scene understanding. Look at the image carefully. A {label} is marked with a RED CONTOUR/OUTLINE (red border line) in the image.

IMPORTANT: Focus ONLY on the object that is outlined in RED. The red contour clearly marks the boundary of this specific {label}.

Your task is to generate a SHORT, PRECISE description that helps uniquely identify this {label} as a 3D object in the scene. The description must be 3D-CONSISTENT, meaning it should describe the object's intrinsic properties that remain the same regardless of the viewing angle or camera position.

CRITICAL: DO NOT include 2D image-specific location information (such as "on the left", "on the right", "in the center", "at the front", "in the corner", etc.). These are view-dependent and will change when viewing from different angles. Instead, focus on:

1. **INTRINSIC VISUAL FEATURES** (REQUIRED): Describe key characteristics that make this object unique and distinguishable as a 3D object:
   - **Basic attributes**: Color, material, texture, shape, size (be specific: "dark grey" not just "grey", "metallic silver" not just "silver")
   - **Surface details**: Patterns, textures, decorative elements, surface finish (e.g., "wooden with visible grain", "metallic with scratches and wear marks", "smooth glossy white")
   - **Text and markings**: Any visible text, labels, logos, signs, numbers, or written content on the object (e.g., "with 'EXIT' sign", "labeled 'A1'", "with FedEx logo", "with 'PUSH' text")
   - **Structural features**: Windows, handles, buttons, panels, openings, frames, hinges, or other distinctive structural elements (e.g., "with glass window", "with brass handle", "with push bar")
   - **Decorative elements**: Patterns, designs, artwork, engravings, or any decorative features visible on the object
   - **Unique details**: Any unusual or distinctive features that make this object stand out from others of the same type
   - Focus on features that help distinguish it from other objects of the same type - be as specific and unique as possible

2. **OBJECT TYPE/SUBTYPE** (if applicable): If the object has a specific type or subtype, include it:
   - Examples: "elevator door", "sliding door", "glass door", "vending machine", "ATM machine", "office chair", "dining table", etc.
   - Include the FULL type name in your description

3. **DIFFERENTIATION** (if multiple objects of same type exist): If there are multiple {label}s visible, describe features that distinguish this one:
   - **Use feature comparison**: "the one with [distinctive feature]", "the one without [feature]", "the one that has [unique detail]"
   - **Use size comparison**: "the larger one", "the smaller one", "the taller one", "the wider one"
   - **Use relative positioning to other objects**: "next to [other object]", "behind [other object]", "in front of [other object]"
   - **Combine multiple distinguishing factors**: Combine visual features with relative positioning (e.g., "dark grey with window, next to the white door")

4. **CONCISENESS**: Keep the description SHORT (maximum 10-15 words total)
   - Format: [object type/subtype if applicable], [specific visual features], [relative positioning if needed]
   - Use commas to separate different parts
   - DO NOT include the word "{label}" in the output unless it's part of a compound type name
   - Examples of good descriptions:
     * "blue door with glass window" (for a door with window)
     * "dark grey metallic elevator door" (for an elevator door)
     * "light grey door with glass window and brass handle" (for a door with specific features)
     * "brown wooden door with visible grain texture" (for a wooden door)
     * "large white vending machine with red FedEx logo" (for a vending machine with logo)
     * "small rectangular metal kiosk with 'EXIT' sign" (for a kiosk with text)
     * "grey metallic door with decorative pattern and push bar" (for a door with multiple features)
     * "white door with 'PUSH' label and glass window" (for a door with text and window)
     * "teal vending machine with postcard display grid" (for a vending machine with specific features)
     * "dark grey door with window, next to the white door" (if multiple doors exist)

OUTPUT FORMAT:
- Output ONLY the final description text (10-15 words maximum)
- DO NOT include any thinking process, reasoning, or intermediate steps
- DO NOT include 2D image position words like "left", "right", "center", "corner", "front", "back" (unless referring to the front/back of the object itself)
- DO NOT include view-dependent location phrases like "on the left", "on the right", "in the center", "at the front", "in the corner"
- Focus on 3D-consistent properties that remain the same from any viewing angle
- If the object has a specific type or subtype, include the FULL type name
- Start your response directly with the description, without any prefix like "The description is:" or "It's a"

CRITICAL: If you are a thinking model, do NOT output your thinking process. Output ONLY the final concise description (10-15 words).

Describe the {label} outlined in RED (object type/subtype if applicable, and distinctive 3D-consistent features):"""

    # 加载图像
    try:
        image = Image.open(image_path).convert("RGB")
        image = resize_image_max_side(image, max_image_side)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
    
    image_url = image_to_base64(image)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    
    try:
        resp = client.chat.completions.create(
            model="",
            messages=messages,
            max_tokens=256,
            temperature=0.0,
        )
        response = (resp.choices[0].message.content or "").strip()
        
        if response is None or not response.strip():
            print("Warning: 模型返回空响应")
            return None
        
        description = response.strip()
        # 清理可能的额外文本（移除可能包含的label词汇）
        description = description.replace(f"the {label}", "").strip()
        description = description.replace(f"a {label}", "").strip()
        description = description.replace(f"an {label}", "").strip()
        # 注意：不要完全移除label，因为可能是复合词的一部分（如"elevator door"）
        
        # 清理多余的标点和空格
        description = description.strip(".,;:!?")
        description = " ".join(description.split())  # 合并多个空格
        
        # 如果清理后为空，返回None
        if not description or description.strip() == "":
            return None
        
        return description
    except Exception as e:
        print(f"Error calling vLLM: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_single_json(
    json_path,
    output_dir,
    client: OpenAI,
    max_image_side: int | None,
    scene_id=None,
    dataset_root: str | None = None,
):
    """
    处理单个JSON文件，为每个instance生成描述
    """
    print(f"\n处理文件: {json_path}")
    
    # 读取JSON文件
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            instances = json.load(f)
    except Exception as e:
        error_msg = f"错误: 无法读取JSON文件 {json_path}: {e}"
        print(error_msg)
        raise RuntimeError(error_msg) from e
    
    if not isinstance(instances, list):
        error_msg = f"错误: JSON文件格式不正确，期望list，得到{type(instances)}"
        print(error_msg)
        raise ValueError(error_msg)
    
    # 如果没有提供scene_id，尝试从文件名推断
    if scene_id is None:
        scene_id = os.path.splitext(os.path.basename(json_path))[0]
    
    print(f"场景ID: {scene_id}")
    print(f"找到 {len(instances)} 个instances")
    
    # 处理每个instance
    processed_count = 0
    success_count = 0
    
    for idx, instance in enumerate(instances):
        ins_id = instance.get('ins_id', '')
        label = instance.get('label', '')
        highest_confidence_mask = instance.get('highest_confidence_mask', '')
        
        if not highest_confidence_mask:
            print(f"  跳过 instance {idx} (ins_id={ins_id}, label={label}): 没有highest_confidence_mask")
            continue
        
        processed_count += 1
        print(f"\n  处理 instance {idx}/{len(instances)}: ins_id={ins_id}, label={label}")
        print(f"    highest_confidence_mask: {highest_confidence_mask}")
        
        # 如果highest_confidence_mask是相对路径，需要先解析
        mask_path_abs = highest_confidence_mask
        if not os.path.isabs(highest_confidence_mask):
            mask_path_abs = str(Path(highest_confidence_mask))
        
        # 提取场景ID和图片名
        scene_id_from_mask = extract_scene_id_from_mask_path(mask_path_abs)
        if scene_id_from_mask:
            scene_id = scene_id_from_mask
        
        image_name = extract_image_name_from_mask_path(mask_path_abs)
        if not image_name:
            error_msg = f"错误: 无法从mask路径提取图片名: {mask_path_abs}"
            print(error_msg)
            raise ValueError(error_msg)
        
        print(f"    图片名: {image_name}")
        
        # 找到原图路径
        original_image_path = find_original_image_path(dataset_root, mask_path_abs, scene_id)
        if not original_image_path or not os.path.exists(original_image_path):
            error_msg = (
                f"错误: 找不到原图: {original_image_path}\n尝试的mask路径: {mask_path_abs}"
                f"\n场景ID: {scene_id}\n图片名: {image_name}\n数据集根目录: {dataset_root}"
            )
            print(error_msg)
            raise FileNotFoundError(error_msg)
        
        print(f"    原图路径: {original_image_path}")
        
        # 找到mask_rle
        mask_rle = find_mask_rle_for_instance(instance, image_name, label, mask_path_abs)
        if not mask_rle:
            error_msg = (
                f"错误: 找不到mask_rle（先查mask_index.json的(image,label)，再回退instance.mask_encodings）"
                f" (image={image_name}, label={label}, ins_id={ins_id})"
            )
            print(error_msg)
            raise ValueError(error_msg)
        
        # 解码mask
        try:
            mask = decode_rle_mask(mask_rle)
        except Exception as e:
            error_msg = f"错误: 无法解码mask: {e}"
            print(error_msg)
            raise RuntimeError(error_msg) from e
        
        # 将mask叠加到原图上
        try:
            masked_image = apply_mask_to_image(original_image_path, mask)
        except Exception as e:
            error_msg = f"错误: 无法应用mask到图像: {e}"
            print(error_msg)
            raise RuntimeError(error_msg) from e
        
        # 保存临时图片
        temp_dir = output_dir if output_dir else os.path.dirname(json_path)
        os.makedirs(temp_dir, exist_ok=True)
        temp_image_path = os.path.join(temp_dir, f"temp_masked_{scene_id}_{ins_id}_{label}.jpg")
        try:
            masked_image.save(temp_image_path, format='JPEG', quality=95)
        except Exception as e:
            error_msg = f"错误: 无法保存临时图片: {e}"
            print(error_msg)
            raise RuntimeError(error_msg) from e
        
        # 调用vLLM生成描述
        print(f"    正在调用vLLM生成描述...")
        description = call_vlm_for_3d_instance_description(temp_image_path, label, client, max_image_side)
        
        if description:
            print(f"    生成的描述: {description}")
            instance['description'] = description
            success_count += 1
        else:
            print(f"    警告: VLM返回None，跳过描述")
        
        # 清理临时文件
        try:
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
        except:
            pass
    
    print(f"\n处理完成: {processed_count} 个instances处理，{success_count} 个成功生成描述")
    
    # 保存更新后的JSON
    output_path = os.path.join(output_dir, os.path.basename(json_path)) if output_dir else json_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(instances, f, indent=2, ensure_ascii=False)
    
    print(f"已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='为3D instance生成描述')
    parser.add_argument('--input_dir', type=str, default="output_scannetppv2_new_aabb",
                        help='输入目录（包含JSON文件的目录，如output_3d_bounding）。如果提供了--json_file，则不需要此参数')
    parser.add_argument('--output_dir', type=str, default="output_scannetppv2_new_aabb_with_descriptions",
                        help='输出目录（如果为None，则覆盖原文件）')
    parser.add_argument('--scene_id', type=str, default=None,
                        help='场景ID（如果为None，则从文件名推断）')
    parser.add_argument('--json_file', type=str, default=None,
                        help='单个JSON文件路径（如果指定，则只处理该文件，不需要--input_dir）')
    parser.add_argument(
        '--model_path',
        type=str,
        default='',
        help='兼容旧参数（vLLM模式下不使用）'
    )
    parser.add_argument('--scene_id_list', type=str, 
                        # default='["001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f"]',
                        default=None,
                        help='场景ID列表（逗号分隔或JSON格式），如果指定，则只处理列表中的场景')
    parser.add_argument('--vllm-api-url', type=str, default=None,
                        help='vLLM API URL（/v1/chat/completions）')
    parser.add_argument('--max-image-side', type=int, default=1024,
                        help='发送给 vLLM 的图像最长边，<=0 表示不缩放')
    parser.add_argument(
        '--dataset_root',
        type=str,
        default="scannetppv2/data",
        help='数据集根目录（包含scene_id的父目录，例如 DL3DV/1K 或 scannetppv2/data）',
    )
    
    args = parser.parse_args()
    
    # 验证参数
    if not args.json_file and not args.input_dir:
        parser.error("必须提供 --input_dir 或 --json_file 参数之一")
    
    vllm_api_url = resolve_vllm_api_url(args.vllm_api_url)
    client = init_vllm_client(vllm_api_url)
    
    # 检查pycocotools
    if mask_util is None:
        print("错误: pycocotools 未安装，无法解码mask")
        print("请运行: pip install pycocotools")
        return
    
    # 处理文件
    if args.json_file:
        # 处理单个文件
        process_single_json(
            args.json_file,
            args.output_dir,
            client,
            args.max_image_side,
            args.scene_id,
            dataset_root=args.dataset_root,
        )
    else:
        # 处理目录中的所有JSON文件
        json_files = [f for f in os.listdir(args.input_dir) if f.endswith('.json')]
        print(f"找到 {len(json_files)} 个JSON文件")
        
        # 提取所有场景ID
        all_scene_ids = [os.path.splitext(f)[0] for f in json_files]
        all_scene_ids.sort()
        
        # 检查输出目录中已存在的场景，跳过已处理的场景
        if args.output_dir and os.path.exists(args.output_dir):
            existing_scenes = set()
            for existing_file in os.listdir(args.output_dir):
                if existing_file.endswith('.json'):
                    existing_scenes.add(os.path.splitext(existing_file)[0])
            
            if existing_scenes:
                print(f"输出目录中已存在 {len(existing_scenes)} 个场景，将跳过这些场景")
                json_files = [f for f in json_files if os.path.splitext(f)[0] not in existing_scenes]
                all_scene_ids = [sid for sid in all_scene_ids if sid not in existing_scenes]
                print(f"过滤后，需要处理 {len(json_files)} 个场景")
        
        # 如果指定了scene_id_list，只处理列表中的场景
        if args.scene_id_list:
            try:
                # 尝试解析为JSON格式
                if args.scene_id_list.startswith('['):
                    scene_id_set = set(json.loads(args.scene_id_list))
                else:
                    # 解析为逗号分隔的字符串
                    scene_id_set = set([s.strip() for s in args.scene_id_list.split(',')])
                
                # 过滤出需要处理的场景
                json_files = [f for f in json_files if os.path.splitext(f)[0] in scene_id_set]
                print(f"根据 --scene_id_list 过滤后，需要处理 {len(json_files)} 个场景")
            except Exception as e:
                print(f"错误: 无法解析 --scene_id_list: {e}")
                return
        else:
            # 如果没有指定scene_id_list，按PER_GPU_SIZE分组并打印
            print(f"\n按 PER_GPU_SIZE={PER_GPU_SIZE} 分组场景ID:")
            for i in range(0, len(all_scene_ids), PER_GPU_SIZE):
                scene_id_batch = all_scene_ids[i:i+PER_GPU_SIZE]
                # 打印为JSON格式（双引号），方便直接复制到--scene_id_list
                batch_num = i//PER_GPU_SIZE + 1
                print(f"\n批次 {batch_num}: {len(scene_id_batch)} 个场景")
                print(f"批次 {batch_num} JSON格式（可直接复制）:")
                print(json.dumps(scene_id_batch, ensure_ascii=False))
            print(f"\n总共 {len(all_scene_ids)} 个场景，分为 {(len(all_scene_ids) + PER_GPU_SIZE - 1) // PER_GPU_SIZE} 个批次")
            print("\n使用示例:")
            print('  --scene_id_list \'["scene1","scene2","scene3"]\'')
            print("或者使用逗号分隔格式:")
            print('  --scene_id_list "scene1,scene2,scene3"')
            return
        
        # 处理过滤后的文件
        for json_file in tqdm(json_files):
            json_path = os.path.join(args.input_dir, json_file)
            process_single_json(
                json_path,
                args.output_dir,
                client,
                args.max_image_side,
                args.scene_id,
                dataset_root=args.dataset_root,
            )


if __name__ == "__main__":
    main()
