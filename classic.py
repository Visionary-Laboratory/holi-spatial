from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from tqdm import tqdm
from sam3.agent.client_sam3 import call_sam_service as call_sam_service_orig

# DEFAULT_MODEL_PATH = "/mnt/shared-storage-user/solution/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5/"
DEFAULT_MODEL_PATH = "/mnt/shared-storage-user/intern7shared/share_ckpt_hf/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/4b184fbdab8886057d8d80c09f35bcfc65fe640e"
DEFAULT_DATA_ROOT = "/home/liuyifei/code/posevlm/scannetppv2/data"
DEFAULT_OUTPUT_DIR = "/home/liuyifei/code/posevlm/scene_objects_Qwen3-VL-30B-A3B-Instruct"


def setup_logger() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def load_model(model_path: str):
    logging.info("加载模型: %s", model_path)
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype="auto",
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def discover_scenes(data_root: Path, scene_names: Sequence[str] | None) -> List[Path]:
    if scene_names:
        return [data_root / name for name in scene_names]
    scenes: List[Path] = []
    for scene_dir in data_root.iterdir():
        if scene_dir.is_dir() and (scene_dir / "dslr"/"resized_undistorted_images").exists():
            scenes.append(scene_dir)
    return scenes


def collect_images(image_root: Path, max_images: int) -> List[Path]:
    if not image_root.exists():
        logging.warning("找不到图片目录: %s", image_root)
        return []
    image_files = sorted(
        [p for p in image_root.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    if not image_files:
        logging.warning("目录中没有找到图片: %s", image_root)
        return []
    if len(image_files) <= max_images:
        return image_files

    # 均匀抽样到 max_images 张
    indices = {
        round(i * (len(image_files) - 1) / (max_images - 1))
        for i in range(max_images)
    }
    return [image_files[i] for i in sorted(indices)]


def build_messages(image: Image.Image, previous_labels: Sequence[str]) -> List[Dict]:
    previous_text = ", ".join(previous_labels) if previous_labels else "none"
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": (
                        "Identify clear, specific object classes in this image using concise, "
                        "lowercase English nouns (e.g., chair, table, sofa). "
                        "Do not use vague or generic descriptions; if an object cannot be "
                        "described precisely, skip it. "
                        f"Existing labels from previous frames: {previous_text}. "
                        "If any of these labeled objects also appear here, do not describe "
                        "them again for this image. "
                        "Return only the new object class names, lowercase, comma-separated, "
                        "with no explanations."
                    ),
                },
            ],
        }
    ]


def parse_categories(text: str) -> List[str]:
    tokens = re.split(r"[,\n;]", text)
    cleaned: List[str] = []
    for token in tokens:
        item = re.sub(r"[^a-zA-Z\s\-]", "", token).strip().lower()
        if item:
            cleaned.append(item)
    if cleaned:
        return cleaned

    # 兜底：提取单词
    fallback = re.findall(r"[a-zA-Z]+", text.lower())
    return fallback


def run_inference_on_image(
    model,
    processor,
    image_path: Path,
    previous_labels: Sequence[str],
) -> List[str]:
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:  # pylint: disable=broad-except
        logging.warning("读取图片失败 %s: %s", image_path, exc)
        return []

    messages = build_messages(image, previous_labels)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=96)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return parse_categories(output_text)


def save_scene_result(
    scene_name: str,
    output_dir: Path,
    categories: Iterable[str],
    per_image: Dict[str, List[str]],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{scene_name}.json"
    data = {
        "scene": scene_name,
        "categories": sorted(set(categories)),
        "per_image": per_image,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=True)
    return output_path


def process_scene(
    scene_dir: Path,
    model,
    processor,
    max_images: int,
    output_dir: Path,
) -> None:
    image_root = scene_dir / "dslr/resized_undistorted_images"
    images = collect_images(image_root, max_images)
    if not images:
        logging.warning("场景 %s 无可用图片，跳过。", scene_dir.name)
        return

    aggregated: Set[str] = set()
    per_image: Dict[str, List[str]] = {}

    for idx, image_path in tqdm(enumerate(images, 1), total=len(images)):
        # logging.info("(%d/%d) 处理图片: %s", idx, len(images), image_path.name)
        categories = run_inference_on_image(model, processor, image_path, sorted(aggregated))
        if categories:
            aggregated.update(categories)
        per_image[image_path.name] = categories

    output_path = save_scene_result(scene_dir.name, output_dir, aggregated, per_image)
    logging.info("场景 %s 完成，去重后类别数 %d，结果保存到 %s", scene_dir.name, len(aggregated), output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量调用 VLM 抽取场景物体类别")
    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help="包含场景目录的根路径（每个场景应有 resized_undistorted_images 子目录）。",
    )
    parser.add_argument(
        "--scene",
        action="append",
        dest="scenes",
        help="指定要处理的场景名称，可多次提供；不指定则自动扫描 data-root 下的场景。",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=100,
        help="每个场景最多抽样的图片数量（均匀抽样）。",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Hugging Face 模型路径或名称。",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="每个场景 JSON 结果的输出目录。",
    )
    return parser.parse_args()


def main() -> None:
    setup_logger()
    args = parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    scenes = discover_scenes(data_root, args.scenes)

    if not scenes:
        logging.error("未找到任何场景目录，请检查 data-root 或 scene 配置。")
        return

    model, processor = load_model(args.model_path)
    for scene_dir in tqdm(scenes):
        process_scene(scene_dir, model, processor, args.max_images, output_dir)


if __name__ == "__main__":
    main()