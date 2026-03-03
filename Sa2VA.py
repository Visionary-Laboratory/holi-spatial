import torch
from transformers import AutoTokenizer, AutoModel, AutoProcessor
from PIL import Image
import numpy as np
import os
import logging
from typing import Optional, List, Dict, Tuple
import cv2


def load_sa2va_model(
    model_path: str = "/mnt/shared-storage-user/solution/huggingface/hub/models--ByteDance--Sa2VA-Qwen2_5-VL-7B/snapshots/4cd6709067cf257235e922c201a6f265292f5fc8",
    device: str = "cuda"
):
    """加载 Sa2VA 模型和处理器"""
    if 'qwen' in model_path.lower():
        print("Using AutoProcessor for Qwen-VL model.")
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        tokenizer = None
    else:
        processor = None
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
    
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        local_files_only=True,
        trust_remote_code=True
    ).eval()
    
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
    
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    
    return model, tokenizer, processor


def predict_mask_sa2va(
    model,
    tokenizer: Optional,
    processor: Optional,
    image: Image.Image,
    text_prompt: str,
    mask_prompts: Optional[List] = None
) -> Dict:
    """
    使用 Sa2VA 模型进行 mask 预测
    
    Args:
        model: Sa2VA 模型
        tokenizer: tokenizer（如果使用）
        processor: processor（如果使用）
        image: PIL Image
        text_prompt: 文本提示，例如 "Segment the {label} in the image"
        mask_prompts: mask prompts（可选）
    
    Returns:
        包含预测结果的字典，可能包含 "prediction"（文本）和 "masks"（mask 数组）
    """
    # 构建输入字典
    input_dict = {
        'image': image,
        'text': text_prompt,
        'past_text': '',
        'mask_prompts': mask_prompts,
        'tokenizer': tokenizer,
        'processor': processor,
    }
    
    # 进行预测
    return_dict = model.predict_forward(**input_dict)
    
    return return_dict


def extract_masks_from_sa2va_output(return_dict: Dict, image_size: Tuple[int, int]) -> List[np.ndarray]:
    """
    从 Sa2VA 输出中提取 masks
    
    Args:
        return_dict: Sa2VA 模型的输出字典
        image_size: (height, width) 图像尺寸
    
    Returns:
        mask 列表，每个 mask 是 (H, W) 的 numpy 数组，值为 0 或 255
    """
    masks = []
    
    # 从 prediction_masks 中提取 mask
    if "prediction_masks" in return_dict:
        mask_data = return_dict["prediction_masks"]
    elif "masks" in return_dict:
        mask_data = return_dict["masks"]
    elif "mask" in return_dict:
        mask_data = return_dict["mask"]
    elif "segmentation" in return_dict:
        mask_data = return_dict["segmentation"]
    else:
        # 如果没有直接的 mask，尝试从 prediction 中解析
        # 或者返回空列表
        logging.warning(f"未找到 mask 数据，return_dict 的键: {list(return_dict.keys())}")
        return masks
    
    # 处理 mask 数据
    # Sa2VA 返回的格式是 list(np.array(1, h, w), ...)
    if isinstance(mask_data, list):
        # 列表格式，每个元素是 (1, h, w) 的数组
        for mask_item in mask_data:
            if isinstance(mask_item, torch.Tensor):
                mask_item = mask_item.detach().cpu().numpy()
            
            if isinstance(mask_item, np.ndarray):
                # 去掉第一个维度 (1, h, w) -> (h, w)
                if mask_item.ndim == 3 and mask_item.shape[0] == 1:
                    mask_item = mask_item[0]  # 从 (1, h, w) 变成 (h, w)
                elif mask_item.ndim == 2:
                    pass  # 已经是 (h, w) 格式
                else:
                    logging.warning(f"意外的 mask 形状: {mask_item.shape}")
                    continue
                
                # 转换为二值 mask
                if mask_item.dtype != bool:
                    mask_item = mask_item > 0.5
                mask_uint8 = mask_item.astype(np.uint8) * 255
                
                # 调整尺寸
                if mask_uint8.shape[0] != image_size[0] or mask_uint8.shape[1] != image_size[1]:
                    mask_uint8 = cv2.resize(mask_uint8, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
                
                masks.append(mask_uint8)
    elif isinstance(mask_data, torch.Tensor):
        mask_data = mask_data.detach().cpu().numpy()
    
    if isinstance(mask_data, np.ndarray):
        # 处理不同维度的 mask
        if mask_data.ndim == 4:
            # (batch, num_masks, H, W)
            mask_data = mask_data[0]  # 取第一个 batch
        if mask_data.ndim == 3:
            # (num_masks, H, W) 或 (H, W, num_masks)
            if mask_data.shape[0] < mask_data.shape[2]:
                # 可能是 (H, W, num_masks)
                mask_data = np.transpose(mask_data, (2, 0, 1))
        
        # 转换为二值 mask
        for i in range(mask_data.shape[0]):
            mask = mask_data[i]
            if mask.dtype != bool:
                mask = mask > 0.5
            mask_uint8 = mask.astype(np.uint8) * 255
            
            # 调整尺寸
            if mask_uint8.shape[0] != image_size[0] or mask_uint8.shape[1] != image_size[1]:
                mask_uint8 = cv2.resize(mask_uint8, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
            
            masks.append(mask_uint8)
    
    return masks


# 示例用法
if __name__ == "__main__":
    # 加载模型
    model, tokenizer, processor = load_sa2va_model()
    
    # 测试图像聊天
    image_path = "dl3dv-demo/dense/rgb/frame_00001.png"
    if os.path.exists(image_path):
        text_prompts = "<image>Please describe the image."
        image = Image.open(image_path).convert('RGB')
        return_dict = predict_mask_sa2va(model, tokenizer, processor, image, text_prompts)
        answer = return_dict.get("prediction", "")  # the text format answer
        print(f"Answer: {answer}")