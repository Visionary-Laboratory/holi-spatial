from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set
import os
import base64
import io
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
# from transformers import AutoModelForVision2Seq, AutoProcessor
from tqdm import tqdm
from sam3.agent.client_sam3 import call_sam_service as call_sam_service_orig
from api import CallBoyueModel

# DEFAULT_MODEL_PATH = "/mnt/shared-storage-user/solution/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5/"
# DEFAULT_MODEL_PATH = "/mnt/shared-storage-user/intern7shared/share_ckpt_hf/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/4b184fbdab8886057d8d80c09f35bcfc65fe640e"
DEFAULT_DATA_ROOT = "scannetppv2"
DEFAULT_OUTPUT_DIR = "gemini3_pro_labels_output"


def setup_logger() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def encode_image_to_base64(image: Image.Image) -> str:
    """将 PIL Image 转换为 base64 字符串。"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def load_gemini_client():
    MODEL_NAME = "gemini-3-flash-preview"
    MAX_TOKENS = 4096
    conf = {
        "model_name": MODEL_NAME,
        "temperature": 0.7,
        "top_p": 1.0,
        "max_tokens": MAX_TOKENS,
        "stream": False
    }
    api_key = "sk-rx3WxxkfCaePAKTP8tzvkIB9eZBTpqJmYvhHw2JwfUYtExDC"
    server_url = os.environ.get("BOYUE_SERVER_URL", "http://35.220.164.252:3888/v1/")
    client = CallBoyueModel(
        conf=conf,
        api_key=api_key,
        base_url=server_url
    )
    return client


def discover_scenes(data_root: Path, scene_names: Sequence[str] | None) -> List[Path]:
    if scene_names:
        return [data_root / name for name in scene_names]
    scenes: List[Path] = []
    
    # 根据 data_root 名称判断使用哪个路径
    data_root_str = str(data_root)
    if "scannetppv2" in data_root_str:
        image_path = "dslr/resized_undistorted_images"
    elif "DL3DV" in data_root_str:
        image_path = "dense/rgb"
    else:
        # 默认使用 scannetppv2 的路径
        image_path = "dslr/resized_undistorted_images"
    
    for scene_dir in data_root.iterdir():
        if scene_dir.is_dir() and (scene_dir / image_path).exists():
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


def run_inference_on_image(
    client: CallBoyueModel,
    image_path: Path,
) -> List[str]:
    """并行阶段：识别图中所有物体。"""
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        logging.warning("读取图片失败 %s: %s", image_path, exc)
        return []

    prompt = (
        "Task: Identify every distinct, clear object in this image using concise, lowercase English nouns.\n"
        "Guidelines:\n"
        "1. Use singular nouns (e.g., 'chair', 'bottle', 'box'). No plurals.\n"
        "2. Use whole-object, practical categories. Do NOT split an object into parts (e.g., use 'broom', NOT 'broom handle').\n"
        "3. Use common categories and avoid adjectives or fine-grained subtypes (e.g., use 'box' for all boxes, 'chair' for all chairs, 'bottle' for all brands).\n"
        "4. Use canonical terms only; no synonyms (e.g., use 'sofa' instead of 'couch'; 'bag' instead of 'backpack').\n"
        "5. Exhaustive labeling: Label every distinct object you see. If unsure of a subtype, use the closest concrete supercategory.\n"
        "6. Do NOT use non-informative words like 'object', 'thing', 'stuff', or 'unknown'.\n"
        "7. Output format: Return ONLY a JSON array of strings. No explanations or markdown headers."
    )

    base64_img = encode_image_to_base64(image)
    processed_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    try:
        api_params = {
            "model": "gemini-3-flash-preview",
            "messages": processed_messages,
            "temperature": 0.2, # 降低随机性
            "max_tokens": 1024,
            "stream": False
        }
        response = client.client.chat.completions.create(**api_params)
        if response.choices:
            text = response.choices[0].message.content
            # 简单解析逻辑
            start, end = text.find('['), text.rfind(']')
            if start != -1 and end != -1:
                return json.loads(text[start:end+1])
    except Exception as e:
        logging.error("API Error for %s: %s", image_path.name, e)
    return []


def consolidate_labels_globally(client: CallBoyueModel, raw_labels: Set[str]) -> Dict[str, str]:
    """全局规范化阶段：合并同义词、处理单复数、消除包含关系。"""
    if not raw_labels:
        return {}
    
    labels_list = sorted(list(raw_labels))
    prompt = (
        "You are a label cleaner. I will give you a list of object labels found in a 3D scene.\n"
        "Your task is to merge them into a minimal set of unique, concise, singular English nouns.\n"
        "Rules:\n"
        "1. Convert everything to SINGULAR (e.g., 'valves' -> 'valve').\n"
        "2. REMOVE REDUNDANCY: If one label is a more specific version of another, merge it into the broader one (e.g., 'cardboard box' -> 'box', 'power wire' -> 'wire').\n"
        "3. MERGE SYNONYMS: (e.g., 'couch' and 'sofa' -> 'sofa').\n"
        "4. Output format: Return ONLY a JSON object mapping the ORIGINAL label to the CLEANED label.\n"
        f"Labels to process: {json.dumps(labels_list)}"
    )

    try:
        response = client.client.chat.completions.create(
            model="gemini-3-flash-preview",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )
        mapping = json.loads(response.choices[0].message.content)
        return mapping
    except Exception as e:
        logging.error("Global consolidation failed: %s", e)
        return {l: l for l in raw_labels}


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
    client: CallBoyueModel,
    max_images: int,
    output_dir: Path,
    data_root: Path,
) -> None:
    # 路径处理逻辑保持不变
    data_root_str = str(data_root)
    if "scannetppv2" in data_root_str:
        image_root = scene_dir / "resized_undistorted_images"
    elif "DL3DV" in data_root_str:
        image_root = scene_dir / "dense/rgb"
    else:
        # 默认使用 scannetppv2 的路径
        image_root = scene_dir / "dslr/resized_undistorted_images"
    
    images = collect_images(image_root, max_images)
    if not images: return

    # 阶段 1：并行抽取
    raw_per_image: Dict[str, List[str]] = {}
    all_raw_labels: Set[str] = set()
    
    max_workers = 15 # 并行加速
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_img = {executor.submit(run_inference_on_image, client, p): p for p in images}
        for future in tqdm(as_completed(future_to_img), total=len(images), desc="Extracting"):
            img_p = future_to_img[future]
            labels = future.result()
            raw_per_image[img_p.name] = labels
            all_raw_labels.update(labels)

    # 阶段 2：全局去重与规范化
    logging.info("Starting global label consolidation for %d unique labels...", len(all_raw_labels))
    label_map = consolidate_labels_globally(client, all_raw_labels)

    # 阶段 3：应用映射并清理
    final_per_image: Dict[str, List[str]] = {}
    final_aggregated: Set[str] = set()
    
    for img_name, labels in raw_per_image.items():
        mapped = set()
        for l in labels:
            # 应用大模型的全局映射，若没映射上则简单处理
            clean_l = label_map.get(l, l).lower().strip()
            if clean_l:
                mapped.add(clean_l)
        final_per_image[img_name] = sorted(list(mapped))
        final_aggregated.update(mapped)

    output_path = save_scene_result(scene_dir.name, output_dir, final_aggregated, final_per_image)
    logging.info("Scene %s completed. Canonical categories: %d", scene_dir.name, len(final_aggregated))


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

    client = load_gemini_client()
    for scene_dir in tqdm(scenes):
        process_scene(scene_dir, client, args.max_images, output_dir, data_root)


if __name__ == "__main__":
    main()
