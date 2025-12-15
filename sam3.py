from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


DEFAULT_DATA_ROOT = Path("/home/liuyifei/code/posevlm/scannetppv2/data")
DEFAULT_SCENE_JSON = Path("scene_objects_Qwen3-VL-30B-A3B-Instruct/0a5c013435.json")
DEFAULT_OUTPUT_DIR = Path("/home/liuyifei/code/posevlm/sam_masks_debug")


def setup_logger() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def slugify_label(label: str) -> str:
    safe = re.sub(r"[^\w\-]+", "_", label.strip().lower())
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe or "label"


def load_scene_json(json_path: Path) -> tuple[str, Dict[str, List[str]]]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    scene_name = data.get("scene")
    if not scene_name:
        raise ValueError(f"JSON {json_path} 中缺少 scene 字段")
    per_image: Dict[str, List[str]] = data.get("per_image") or {}
    return scene_name, per_image


def ensure_image_path(image_root: Path, image_name: str) -> Optional[Path]:
    """尽量兼容大小写和扩展名差异寻找图片路径。"""
    candidates: Sequence[Path] = [
        image_root / image_name,
        image_root / image_name.lower(),
        image_root / image_name.upper(),
    ]
    stem = Path(image_name).stem
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        candidates.append(image_root / f"{stem}{ext}")
    for path in candidates:
        if path.exists():
            return path
    return None


def normalize_masks_boxes(output: Dict) -> List[Dict]:
    # 避免直接用 `or` 在 tensor 上求布尔导致报错
    masks = output.get("masks", None)
    if masks is None:
        masks = output.get("out_binary_masks", None)
    boxes = output.get("boxes")
    scores = output.get("scores")

    if masks is None:
        return []

    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().numpy()
    if boxes is not None and isinstance(boxes, torch.Tensor):
        boxes = boxes.detach().cpu().numpy()
    if scores is not None and isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
        
    assert len(masks) == len(boxes) and len(boxes) == len(scores)

    if masks.ndim == 4:
        masks = masks.reshape(-1, masks.shape[-2], masks.shape[-1])
    elif masks.ndim == 3:
        pass
    elif masks.ndim == 2:
        masks = masks[None, ...]
    else:
        return []

    boxes_list: List[Optional[Sequence[float]]] = []
    if boxes is None:
        boxes_list = [None] * len(masks)
    else:
        if boxes.ndim == 3:
            boxes = boxes.reshape(-1, boxes.shape[-1])
        boxes_list = [boxes[i].tolist() for i in range(min(len(boxes), len(masks)))]
        if len(boxes_list) < len(masks):
            boxes_list.extend([None] * (len(masks) - len(boxes_list)))

    score_list: List[Optional[float]] = []
    if scores is None:
        score_list = [None] * len(masks)
    else:
        scores = scores.reshape(-1)
        score_list = [float(scores[i]) for i in range(min(len(scores), len(masks)))]
        if len(score_list) < len(masks):
            score_list.extend([None] * (len(masks) - len(score_list)))

    results: List[Dict] = []
    for mask, box, score in zip(masks, boxes_list, score_list):
        mask_bool = mask > 0.5 if mask.dtype != bool else mask
        results.append({"mask": mask_bool, "box": box, "score": score})
    return results


def pick_best_mask(items: Iterable[Dict]) -> Optional[Dict]:
    best: Optional[Dict] = None
    for item in items:
        if best is None:
            best = item
            continue
        if item.get("score") is None:
            continue
        if best.get("score") is None or item["score"] > best["score"]:
            best = item
    return best


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask_img = Image.fromarray((mask.astype(np.uint8)) * 255)
    mask_img.save(path)


def process_image(
    processor: Sam3Processor,
    image_path: Path,
    labels: Iterable[str],
    mask_root: Path,
) -> List[Dict]:
    image = Image.open(image_path).convert("RGB")
    results: List[Dict] = []
    unique_labels = sorted(set(labels))

    for label in unique_labels:
        state = processor.set_image(image)
        output = processor.set_text_prompt(state=state, prompt=label)
        masks_info = normalize_masks_boxes(output)
        if not masks_info:
            logging.warning("未获得 mask: %s - %s", image_path.name, label)
            continue

        safe_label = slugify_label(label)
        multi = len(masks_info) > 1
        for idx, item in enumerate(masks_info, 1):
            fname = f"{safe_label}_{idx:02d}.png" if multi else f"{safe_label}.png"
            mask_path = mask_root / image_path.stem / fname
            save_mask(item["mask"], mask_path)

            results.append(
                {
                    "image": image_path.name,
                    "label": label,
                    "mask_path": str(mask_path),
                    "bbox": item.get("box"),
                    "score": item.get("score"),
                }
            )
        logging.info("已保存 %d 个 mask (%s)", len(masks_info), label)
    return results


def process_scene(
    scene_json: Path,
    data_root: Path,
    output_dir: Path,
) -> Path:
    scene_name, per_image = load_scene_json(scene_json)
    image_root = data_root / scene_name / "dslr" / "resized_undistorted_images"

    if not image_root.exists():
        raise FileNotFoundError(f"找不到图片目录: {image_root}")

    logging.info("加载模型...")
    model = build_sam3_image_model()
    processor = Sam3Processor(model, confidence_threshold=0.5)

    all_results: List[Dict] = []
    missing_images: List[str] = []

    mask_root = output_dir / scene_name
    for idx, (image_name, labels) in enumerate(per_image.items(), 1):
        image_path = ensure_image_path(image_root, image_name)
        if image_path is None:
            logging.warning("图片不存在，跳过: %s", image_name)
            missing_images.append(image_name)
            continue
        logging.info("(%d/%d) 处理图片 %s，标签数 %d", idx, len(per_image), image_name, len(labels))
        image_results = process_image(processor, image_path, labels, mask_root)
        all_results.extend(image_results)

    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = mask_root / "mask_index.json"
    index = {
        "scene": scene_name,
        "image_root": str(image_root),
        "mask_root": str(mask_root),
        "items": all_results,
        "missing_images": missing_images,
    }
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    logging.info("索引保存到 %s，总计 %d 个 mask", index_path, len(all_results))
    return index_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于 scene_objects JSON + SAM3 生成标签 mask")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="原始图片数据根目录（包含 <scene>/dslr/resized_undistorted_images）",
    )
    parser.add_argument(
        "--scene-json",
        type=Path,
        default=DEFAULT_SCENE_JSON,
        help="scene_objects 下的分类结果 JSON 路径",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="mask 及索引输出目录",
    )
    return parser.parse_args()


def main() -> None:
    setup_logger()
    args = parse_args()
    process_scene(args.scene_json, args.data_root, args.output_dir)


if __name__ == "__main__":
    main()