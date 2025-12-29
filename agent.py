from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pycocotools.mask as mask_utils
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from tqdm import tqdm

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.agent.client_sam3 import sam3_inference
from sam3.agent.helpers.mask_overlap_removal import remove_overlapping_masks
from sam3.agent.viz import visualize

# 允许 Ampere 上的 tf32
# https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

# 默认使用单卡
# os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

DEFAULT_MODEL_PATH = "/mnt/shared-storage-user/intern7shared/share_ckpt_hf/models--Qwen--Qwen3-VL-30B-A3B-Thinking/snapshots/7e9bbfa2c1b2059edd18160793fd421194da2c10"
DEFAULT_DATA_ROOT = Path("/home/liuyifei/code/posevlm/scannetppv2/data")
DEFAULT_SCENE_JSON = Path("scene_objects_Qwen3-VL-30B-A3B-Instruct/0a5c013435.json")
DEFAULT_SAM_OUTPUT_DIR = Path("sam_vis_debug_new")
DEFAULT_MASK_OUTPUT_DIR = Path("sam_masks_debug_new")


def setup_logger() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def sanitize_filename(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", text.strip().lower())
    return cleaned or "unknown"


def load_scene_config(json_path: Path) -> tuple[str, Dict[str, List[str]]]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    scene_name = data.get("scene")
    per_image: Dict[str, List[str]] = data.get("per_image", {})
    if not scene_name:
        raise ValueError(f"JSON {json_path} 中缺少 scene 字段")
    return scene_name, per_image


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


def prepare_sam_response(
    sam_processor: Sam3Processor,
    image_path: Path,
    category: str,
    sam_output_dir: Path,
    scene_name: str,
) -> dict:
    raw_output = sam3_inference(sam_processor, str(image_path), category)
    raw_output = remove_overlapping_masks(raw_output)

    serialized: dict = {
        # 绝对路径避免在不同工作目录下读取失败
        "original_image_path": str(image_path.resolve()),
        "orig_img_h": raw_output.get("orig_img_h"),
        "orig_img_w": raw_output.get("orig_img_w"),
        "pred_boxes": raw_output.get("pred_boxes", []),
        "pred_masks": raw_output.get("pred_masks", []),
        "pred_scores": raw_output.get("pred_scores", []),
    }

    if serialized["pred_scores"]:
        order = sorted(
            range(len(serialized["pred_scores"])),
            key=lambda i: serialized["pred_scores"][i],
            reverse=True,
        )
        serialized["pred_scores"] = [serialized["pred_scores"][i] for i in order]
        serialized["pred_boxes"] = [serialized["pred_boxes"][i] for i in order]
        serialized["pred_masks"] = [serialized["pred_masks"][i] for i in order]

    pred_masks = serialized.get("pred_masks", []) or []
    pred_boxes = serialized.get("pred_boxes", []) or [None] * len(pred_masks)
    pred_scores = serialized.get("pred_scores", []) or [1.0] * len(pred_masks)

    valid_masks, valid_boxes, valid_scores = [], [], []
    for mask_item, box_item, score_item in zip(pred_masks, pred_boxes, pred_scores):
        if len(mask_item) > 4:
            valid_masks.append(mask_item)
            valid_boxes.append(box_item)
            valid_scores.append(score_item)

    serialized["pred_masks"] = valid_masks
    serialized["pred_boxes"] = valid_boxes
    serialized["pred_scores"] = valid_scores

    safe_category = sanitize_filename(category)
    output_image_path = (
        sam_output_dir / scene_name / image_path.stem / f"{safe_category}.png"
    )
    output_image_path.parent.mkdir(parents=True, exist_ok=True)

    viz_image = visualize(serialized)
    viz_image.save(output_image_path)

    output_json_path = output_image_path.with_suffix(".json")
    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump({**serialized, "category": category}, f, ensure_ascii=False, indent=2)

    serialized["output_image_path"] = str(output_image_path)
    serialized["category"] = category
    return serialized


def build_validation_messages(
    raw_image: Image.Image, overlay_image: Image.Image, category: str
) -> List[Dict]:


    prompt = (
    "You are a visual label verifier for a single masked object.\n"
    "You will be given two images: (1) the original image, and (2) a black-and-white mask image.\n"
    "In the mask image, WHITE pixels indicate the target object region; BLACK pixels are background.\n"
    f'The user-provided label/category to verify is: "{category}".\n\n'

    "Task:\n"
    "Decide whether the given label correctly describes the object indicated by the WHITE mask region.\n"
    "If it is correct, ACCEPT. If it is incorrect (wrong category), REJECT and provide the correct label you believe fits best.\n\n"

    "Guidelines (consistency-and-rewrite mindset):\n"
    "1) Focus ONLY on the object covered by the WHITE mask region. Ignore other objects outside the mask.\n"
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
    "Do not output any extra text."
)



    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": raw_image},
                {"type": "image", "image": overlay_image},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def parse_indices(text: str) -> List[int]:
    lowered = text.lower()
    if "none" in lowered:
        return []
    numbers = re.findall(r"\d+", lowered)
    unique_sorted = sorted({int(n) for n in numbers})
    return unique_sorted


def ask_vlm_for_indices(
    model,
    processor,
    raw_image_path: Path,
    overlay_image_path: Path,
    category: str,
    max_new_tokens: int = 64,
) -> List[int]:
    raw_image = Image.open(raw_image_path).convert("RGB")
    overlay_image = Image.open(overlay_image_path).convert("RGB")
    messages = build_validation_messages(raw_image, overlay_image, category)

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    indices = parse_indices(output_text)
    logging.info("类别 %s 大模型返回: %s -> %s", category, output_text, indices)
    return indices


def save_masks(
    serialized: dict,
    approved_indices: Sequence[int],
    mask_output_dir: Path,
    scene_name: str,
    image_name: str,
    image_stem: str,
    category: str,
) -> tuple[List[Path], List[Dict]]:
    if not approved_indices:
        return [], []

    pred_masks = serialized.get("pred_masks", [])
    pred_boxes = serialized.get("pred_boxes", []) or []
    pred_scores = serialized.get("pred_scores", []) or []
    h = int(serialized["orig_img_h"])
    w = int(serialized["orig_img_w"])

    saved_paths: List[Path] = []
    records: List[Dict] = []
    for idx in approved_indices:
        zero_idx = idx - 1
        if zero_idx < 0 or zero_idx >= len(pred_masks):
            logging.warning(
                "索引 %d 超出可用掩码数量 %d，跳过 (%s / %s / %s)",
                idx,
                len(pred_masks),
                scene_name,
                image_stem,
                category,
            )
            continue

        rle = pred_masks[zero_idx]
        mask = mask_utils.decode({"size": (h, w), "counts": rle})
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        binary_mask = (mask > 0).astype(np.uint8) * 255
        output_path = (
            mask_output_dir
            / scene_name
            / image_stem
            / f"{sanitize_filename(category)}_{idx}.png"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(binary_mask).save(output_path)
        saved_paths.append(output_path)
        records.append(
            {
                "image": image_name,
                "label": category,
                "mask_path": str(output_path),
                "bbox": pred_boxes[zero_idx] if zero_idx < len(pred_boxes) else None,
                "score": pred_scores[zero_idx] if zero_idx < len(pred_scores) else None,
                "mask_index": idx,
            }
        )
    return saved_paths, records


def process_scene(
    scene_name: str,
    per_image: Dict[str, List[str]],
    data_root: Path,
    sam_processor: Sam3Processor,
    vlm_model,
    vlm_processor,
    sam_output_dir: Path,
    mask_output_dir: Path,
    image_limit: int | None = None,
    category_limit: int | None = None,
) -> None:
    data_root = data_root.expanduser().resolve()
    data_root_str = str(data_root)
    if "scannetppv2" in data_root_str:
        image_root = data_root / scene_name / "dslr" / "resized_undistorted_images"
    elif "dl3dv" in data_root_str:
        image_root = data_root / scene_name / "dense" / "rgb"
    else:
        # 默认使用 scannetppv2 的路径
        image_root = data_root / scene_name / "dslr" / "resized_undistorted_images"
    # image_root = data_root / scene_name / "dslr" / "resized_undistorted_images"
    if not image_root.exists():
        raise FileNotFoundError(f"找不到图片目录: {image_root}")

    items = list(per_image.items())
    if image_limit:
        items = items[:image_limit]

    all_records: List[Dict] = []
    missing_images: List[str] = []

    for image_idx, (image_name, categories) in enumerate(tqdm(items, desc="处理图片")):
        image_path = image_root / image_name
        if image_name != "DSC01860.JPG"  and image_name !="DSC01810.JPG":
            continue

        if not image_path.exists():
            logging.warning("图片不存在，跳过: %s", image_path)
            missing_images.append(image_name)
            continue

        logging.info("(%d/%d) 处理图片 %s", image_idx + 1, len(items), image_name)
        categories_to_use = categories if category_limit is None else categories[:category_limit]
        for category in categories_to_use:
            sam_result = prepare_sam_response(
                sam_processor, image_path, category, sam_output_dir, scene_name
            )

            if not sam_result.get("pred_masks"):
                logging.info("类别 %s 未检测到掩码，跳过。", category)
                continue

            overlay_image_path = Path(sam_result["output_image_path"])
            approved = ask_vlm_for_indices(
                vlm_model, vlm_processor, image_path, overlay_image_path, category
            )
            saved_paths, records = save_masks(
                sam_result,
                approved,
                mask_output_dir,
                scene_name,
                image_name,
                image_path.stem,
                category,
            )
            all_records.extend(records)
            logging.info(
                "类别 %s 保存掩码 %d 个，路径示例: %s",
                category,
                len(saved_paths),
                saved_paths[0] if saved_paths else "无",
            )

    # 保存索引
    mask_root = mask_output_dir / scene_name
    index_path = mask_root / "mask_index.json"
    index = {
        "scene": scene_name,
        "image_root": str(image_root),
        "mask_root": str(mask_root),
        "items": all_records,
        "missing_images": missing_images,
    }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    logging.info("mask 索引已保存到 %s，总计 %d 个 mask，缺失图片 %d", index_path, len(all_records), len(missing_images))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="调用 SAM + VLM 过滤掩码")
    parser.add_argument("--scene-json", type=Path, default=DEFAULT_SCENE_JSON)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--sam-output-dir", type=Path, default=DEFAULT_SAM_OUTPUT_DIR)
    parser.add_argument("--mask-output-dir", type=Path, default=DEFAULT_MASK_OUTPUT_DIR)
    parser.add_argument("--image-limit", type=int, default=None, help="可选：限制处理前 N 张图片")
    parser.add_argument(
        "--category-limit", type=int, default=None, help="可选：每张图只取前 N 个类别"
    )
    return parser.parse_args()


def main() -> None:
    setup_logger()
    args = parse_args()
    # 使用绝对路径，避免相对路径在子进程/不同工作目录下读取失败
    args.data_root = args.data_root.expanduser().resolve()

    scene_name, per_image = load_scene_config(args.scene_json)
    sam_model = build_sam3_image_model(checkpoint_path="/mnt/shared-storage-user/solution/huggingface/hub/models--facebook--sam3/snapshots/2afe64078f4420bdfbc063162d1336003efadc81/sam3.pt")
    sam_processor = Sam3Processor(sam_model, confidence_threshold=0.5)

    vlm_model, vlm_processor = load_vlm_model(args.model_path)

    process_scene(
        scene_name=scene_name,
        per_image=per_image,
        data_root=args.data_root,
        sam_processor=sam_processor,
        vlm_model=vlm_model,
        vlm_processor=vlm_processor,
        sam_output_dir=args.sam_output_dir,
        mask_output_dir=args.mask_output_dir,
        image_limit=args.image_limit,
        category_limit=args.category_limit,
    )


if __name__ == "__main__":
    main()
