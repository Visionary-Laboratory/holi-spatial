from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set

from openai import OpenAI
from PIL import Image
from tqdm import tqdm
from sam3.agent.client_sam3 import call_sam_service as call_sam_service_orig

DEFAULT_API_BASE = "http://localhost:8000/v1"
DEFAULT_DATA_ROOT = "scannetppv2/data"
DEFAULT_OUTPUT_DIR = "scene_objects_Qwen3-VL-30B-A3B-Instruct-scannetppv2"


def setup_logger() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def init_vllm_client(api_base: str) -> OpenAI:
    logging.info("初始化 vLLM API 客户端: %s", api_base)
    client = OpenAI(
        base_url=api_base,
        api_key="not-needed",  # vLLM 不需要真实的 API key
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


def image_to_base64(image: Image.Image) -> str:
    """将 PIL Image 转换为 base64 编码的字符串"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def build_messages(image: Image.Image, previous_labels: Sequence[str]) -> List[Dict]:
    previous_text = ", ".join(previous_labels) if previous_labels else "none"
    image_url = image_to_base64(image)
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
                {
                "type": "text",
                "text": (
                    "You are zoning a video frame into functional regions (activity areas), NOT objects.\n\n"
                    "Task:\n"
                    "- Identify all distinct functional regions visible in the image (e.g., sleep area, study area, cooking area, dining area, bathroom area, hallway, entry area, storage area).\n"
                    "- A region is an area of the scene defined by its intended use, not by individual objects.\n"
                    "- Do NOT list object names.\n\n"
                    "Region vocabulary rules:\n"
                    "- Use concise, lowercase English region labels.\n"
                    "- Use base label only (no adjectives, no materials).\n"
                    "- If unsure, choose the closest reasonable supercategory (e.g., 'living area' instead of a specific style).\n"
                    "- Avoid 'other/unknown'.\n\n"
                    "Consistency with previous frames (canonical vocabulary):\n"
                    f"- Existing region labels from previous frames: {previous_text}\n"
                    "- If a region matches a previous label in meaning (including synonyms, singular/plural, or minor wording differences), "
                    "you MUST reuse the previous label exactly and MUST NOT introduce a new synonym.\n\n"
                    "Granularity:\n"
                    "- Prefer larger, functional zones over tiny subdivisions.\n"
                    "- Merge areas with the same function into one label.\n\n"
                    "Output:\n"
                    "- Return a single line of lowercase, comma-separated region labels only.\n"
                    "- Include labels from previous frames only if they also appear in the current image.\n"
                    "- No explanations, no extra text."
                        )
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
    client: OpenAI,
    image_path: Path,
    previous_labels: Sequence[str],
) -> List[str]:
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:  # pylint: disable=broad-except
        logging.warning("读取图片失败 %s: %s", image_path, exc)
        return []

    messages = build_messages(image, previous_labels)
    
    try:
        response = client.chat.completions.create(
            model="",  # vLLM 不需要指定模型名称
            messages=messages,
            max_tokens=400,
            temperature=0.0,  # 使用确定性输出
        )
        output_text = response.choices[0].message.content
        print(output_text)
        cats = set(parse_categories(output_text))
        # cats.update(["wall", "floor", "ceiling"])
        return sorted(cats)
        # return parse_categories(output_text)
    except Exception as exc:  # pylint: disable=broad-except
        logging.error("API 调用失败 %s: %s", image_path, exc)
        return []


def save_scene_result(
    scene_name: str,
    output_dir: Path,
    categories: Iterable[str],
    per_image: Dict[str, List[str]],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{scene_name}.json"
    categories_set = set(categories) 
    data = {
        "scene": scene_name,
        "categories": sorted(categories_set),
        "per_image": per_image,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=True)
    return output_path


def process_scene(
    scene_dir: Path,
    client: OpenAI,
    max_images: int,
    output_dir: Path,
    data_root: Path,
) -> None:
    # 根据 data_root 名称判断使用哪个路径
    data_root_str = str(data_root)
    if "scannetppv2" in data_root_str:
        image_root = scene_dir / "dslr/resized_undistorted_images"
    elif "scannetv2" in data_root_str:
        # ScanNet v2: scans/<scene_id>/color/*.jpg
        image_root = scene_dir / "color"
    elif "DL3DV" in data_root_str:
        image_root = scene_dir / "dense/rgb"
    else:
        # 默认使用 scannetppv2 的路径
        image_root = scene_dir / "dslr/resized_undistorted_images"
    
    images = collect_images(image_root, max_images)
    if not images:
        logging.warning("场景 %s 无可用图片，跳过。", scene_dir.name)
        return

    aggregated: Set[str] = set()
    per_image: Dict[str, List[str]] = {}

    for idx, image_path in tqdm(enumerate(images, 1), total=len(images)):
        # logging.info("(%d/%d) 处理图片: %s", idx, len(images), image_path.name)
        categories = run_inference_on_image(client, image_path, sorted(aggregated))
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
        "--api-base",
        default=DEFAULT_API_BASE,
        help="vLLM API 服务的基础 URL（默认: http://localhost:8000/v1）。",
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

    client = init_vllm_client(args.api_base)
    for scene_dir in tqdm(scenes):
        process_scene(scene_dir, client, args.max_images, output_dir, data_root)


if __name__ == "__main__":
    main()