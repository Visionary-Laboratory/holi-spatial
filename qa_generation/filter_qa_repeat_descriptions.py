#!/usr/bin/env python3
"""
过滤 language_description QA：当同一张图中存在多个同 label 物体时，判断目标描述是否可区分。

输入：
- --bbox-json-folder: output_3d_bounding_with_descriptions（每个 scene 一个 json，包含 ins_id/label/images/description）
- --output: output_QA_lang（每个 scene 一个 QA json）

输出：
- --output-filter-repeat: 过滤后的 QA json（每个 scene 一个）

规则（保守）：
- 仅处理 marker_type == language_description 的 QA；其它类型原样保留
- 若同一张图里该 label 只有一个实例：保留
- 若存在其它同 label 实例：
  - 若其它实例里有 description 与目标完全相同：直接 DROP
- 否则调用 vLLM 服务（文本模式）判断是否能“唯一指代”该实例；不确定则 DROP
"""

import argparse
import base64
import io
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm

from openai import OpenAI
from PIL import Image


@dataclass(frozen=True)
class Ref:
    scene_id: str
    image_stem: str
    image_name: str
    ins_id: str
    label: str
    description: str


def load_json(path: Path):
    return json.loads(path.read_text())


def build_image_label_index(bbox_items: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, List[Tuple[str, str]]]], Dict[str, Dict[str, Dict[str, str]]]]:
    """
    Returns:
      - stem -> label -> list[(ins_id, description)]
      - stem -> ins_id -> {"label": ..., "description": ...}
    """
    stem_label_map: Dict[str, Dict[str, List[Tuple[str, str]]]] = {}
    stem_ins_map: Dict[str, Dict[str, Dict[str, str]]] = {}

    for inst in bbox_items:
        ins_id = str(inst.get("ins_id", ""))
        label = str(inst.get("label", ""))
        desc = inst.get("description")
        if not desc:
            # 这里不强制报错：让过滤脚本能指出哪些 QA 会触发缺失（由 QA 生成阶段决定是否允许）
            desc = ""

        for p_str in inst.get("images", []) or []:
            stem = Path(p_str).parent.name  # 与 generate_two_view_qa.py 约定一致
            stem_label_map.setdefault(stem, {}).setdefault(label, []).append((ins_id, desc))
            stem_ins_map.setdefault(stem, {})[ins_id] = {"label": label, "description": desc}

    return stem_label_map, stem_ins_map


def extract_refs_from_entry(entry: Dict[str, Any], scene_id: str, stem_ins_map: Dict[str, Dict[str, Dict[str, str]]]) -> List[Ref]:
    """
    从 QA entry 中提取需要判别唯一性的引用（Ref）。
    """
    marker_type = entry.get("marker_type")
    if marker_type != "language_description":
        return []

    image_a = str(entry.get("image_a", ""))
    image_b = str(entry.get("image_b", ""))
    stem_a = Path(image_a).stem if image_a else ""
    stem_b = Path(image_b).stem if image_b else ""

    refs: List[Ref] = []
    md = entry.get("marker_data") or {}

    def is_object_entity(obj: Dict[str, Any]) -> bool:
        # 兼容旧数据：未显式给 entity_type 时按 object 处理
        entity_type = str(obj.get("entity_type", "object")).strip().lower()
        return (not entity_type) or entity_type == "object"

    # object depth (bbox_center_distance): description 在 image A
    if "selected_ins_id" in entry and "image_a_description" in md:
        ins_id = str(entry["selected_ins_id"])
        label = ""
        # 从 objects 列表里反查 label
        for obj in entry.get("objects", []) or []:
            if str(obj.get("ins_id", "")) == ins_id:
                label = str(obj.get("label", ""))
                break
        desc = str(md.get("image_a_description", "")).strip()
        if stem_a and ins_id and label and desc:
            refs.append(Ref(scene_id, stem_a, Path(image_a).name, ins_id, label, desc))
        return refs

    # inter object distance: image A 的 object_1 + image B 的 object_2
    if "object_1" in entry and "object_2" in entry and "image_a_description_1" in md and "image_b_description_2" in md:
        o1 = entry["object_1"] or {}
        o2 = entry["object_2"] or {}
        ins1 = str(o1.get("ins_id", ""))
        lab1 = str(o1.get("label", ""))
        desc1 = str(md.get("image_a_description_1", "")).strip()
        if is_object_entity(o1) and stem_a and ins1 and lab1 and desc1:
            refs.append(Ref(scene_id, stem_a, Path(image_a).name, ins1, lab1, desc1))

        ins2 = str(o2.get("ins_id", ""))
        lab2 = str(o2.get("label", ""))
        desc2 = str(md.get("image_b_description_2", "")).strip()
        if is_object_entity(o2) and stem_b and ins2 and lab2 and desc2:
            refs.append(Ref(scene_id, stem_b, Path(image_b).name, ins2, lab2, desc2))
        return refs

    # 兼容新版 region-object 混合条目：
    # object_n / region_n + image_[a|b]_description_n，仅提取 object 的描述用于判别。
    indexed_refs: List[Ref] = []
    for key, obj in entry.items():
        if not isinstance(obj, dict):
            continue
        m = re.fullmatch(r"(object|region)_(\d+)", key)
        if not m:
            continue
        idx = m.group(2)
        if not is_object_entity(obj):
            continue
        ins_id = str(obj.get("ins_id", ""))
        label = str(obj.get("label", ""))
        desc_a = str(md.get(f"image_a_description_{idx}", "")).strip()
        desc_b = str(md.get(f"image_b_description_{idx}", "")).strip()
        if stem_a and ins_id and label and desc_a:
            indexed_refs.append(Ref(scene_id, stem_a, Path(image_a).name, ins_id, label, desc_a))
        if stem_b and ins_id and label and desc_b:
            indexed_refs.append(Ref(scene_id, stem_b, Path(image_b).name, ins_id, label, desc_b))
    if indexed_refs:
        return indexed_refs

    # relpos/proxy 等：objects dict(A/B/C) + marker_data description_A/B/C
    if isinstance(entry.get("objects"), dict):
        objs = entry["objects"]
        for key in ["A", "B", "C"]:
            obj = objs.get(key) or {}
            if not is_object_entity(obj):
                continue
            ins_id = str(obj.get("ins_id", ""))
            label = str(obj.get("label", ""))
            desc = str(md.get(f"description_{key}", "")).strip()
            if not (ins_id and label and desc):
                continue
            # 只在该物体确实出现在对应 image stem 时才检查
            if stem_a and ins_id in (stem_ins_map.get(stem_a) or {}):
                refs.append(Ref(scene_id, stem_a, Path(image_a).name, ins_id, label, desc))
            if stem_b and ins_id in (stem_ins_map.get(stem_b) or {}):
                refs.append(Ref(scene_id, stem_b, Path(image_b).name, ins_id, label, desc))
        return refs

    return refs


def normalize_vllm_api_url(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return ""
    if not (u.startswith("http://") or u.startswith("https://")):
        u = "http://" + u
    if "/v1/" not in u:
        u = u.rstrip("/") + "/v1/chat/completions"
    return u


def resolve_vllm_api_urls(vllm_api_urls: Optional[str], vllm_api_url: Optional[str]) -> List[str]:
    if vllm_api_urls:
        parts = [normalize_vllm_api_url(p) for p in vllm_api_urls.split(",")]
        urls = [p for p in parts if p]
        if urls:
            return urls
    url = normalize_vllm_api_url(vllm_api_url or "")
    if url:
        return [url]
    return ["http://localhost:8000/v1/chat/completions"]


def init_vllm_client(vllm_api_url: str) -> OpenAI:
    api_base = vllm_api_url
    if api_base.endswith("/chat/completions"):
        api_base = api_base[: -len("/chat/completions")]
    return OpenAI(base_url=api_base, api_key="not-needed")


_thread_local = threading.local()


def get_thread_client(vllm_api_url: str) -> OpenAI:
    client_map = getattr(_thread_local, "client_map", None)
    if client_map is None:
        client_map = {}
        _thread_local.client_map = client_map
    client = client_map.get(vllm_api_url)
    if client is None:
        client = init_vllm_client(vllm_api_url)
        client_map[vllm_api_url] = client
    return client


def resolve_image_path(data_root: Path, scene_id: str, image_name: str) -> Path:
    """
    QA 里 image_a/image_b 一般是 "DSCxxxxx.JPG" 或 "0.jpg" 等。
    支持的路径格式:
      - scannetppv2: <data_root>/<scene_id>/dslr/resized_undistorted_images/<image_name>
      - ScanNetV2:   <data_root>/<scene_id>/color/<image_name>
    """
    name = Path(image_name).name
    candidates = [
        data_root / scene_id / "dslr" / "resized_undistorted_images" / name,
        data_root / scene_id / "color" / name,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"找不到原图，已尝试: {candidates}")


def call_text_llm_json(client: OpenAI, prompt: str, max_new_tokens: int = 256) -> Dict[str, Any]:
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    resp = client.chat.completions.create(
        model="",
        messages=messages,
        max_tokens=max_new_tokens,
        temperature=0.0,
    )
    text = (resp.choices[0].message.content or "").strip()

    # 提取 JSON
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {"decision": "DROP", "reason": f"parse_error: no_json: {text[:200]}"}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {"decision": "DROP", "reason": f"parse_error: bad_json: {text[:200]}"}


def image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def resize_image_max_side(image: Image.Image, max_image_side: Optional[int]) -> Image.Image:
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


def call_vlm_json(
    client: OpenAI,
    image_path: Path,
    prompt: str,
    max_new_tokens: int = 256,
    max_image_side: Optional[int] = None,
) -> Dict[str, Any]:
    image = Image.open(str(image_path)).convert("RGB")
    image = resize_image_max_side(image, max_image_side)
    image_url = image_to_base64(image)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    resp = client.chat.completions.create(
        model="",
        messages=messages,
        max_tokens=max_new_tokens,
        temperature=0.0,
    )
    text = (resp.choices[0].message.content or "").strip()

    # 提取 JSON
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {"decision": "DROP", "reason": f"parse_error: no_json: {text[:200]}"}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {"decision": "DROP", "reason": f"parse_error: bad_json: {text[:200]}"}


def build_disambiguation_prompt(ref: Ref, other_descs: List[str]) -> str:
    others = [d for d in other_descs if d.strip()]
    others_str = "\n".join([f"- {d}" for d in others]) if others else "(none)"
    return (
        "You are a strict dataset QA filter.\n"
        "You are given an image and multiple language descriptions of objects of the SAME category in this image.\n"
        "Task: Decide if the TARGET description can uniquely identify exactly one object among OTHER objects of that category in the image.\n"
        "Be conservative: if there is any plausible ambiguity, answer DROP.\n\n"
        f"Scene: {ref.scene_id}\n"
        f"Image: {ref.image_name}\n"
        f"Category(label): {ref.label}\n\n"
        f"TARGET description:\n{ref.description}\n\n"
        "OTHER descriptions (same label in same image):\n"
        f"{others_str}\n\n"
        "Output JSON on a single line with keys:\n"
        '  - decision: "KEEP" or "DROP"\n'
        '  - reason: short English reason\n'
        "Do NOT output any other text."
    )


def process_scene(
    qa_path: Path,
    args: argparse.Namespace,
    vllm_api_url: str,
    cache_img: Dict[Tuple[str, str, str, str, Tuple[str, ...]], str],
    cache_txt: Dict[Tuple[str, str, Tuple[str, ...]], str],
    cache_lock: threading.Lock,
    show_progress: bool,
) -> Tuple[str, int, int, int]:
    scene_id = qa_path.stem
    bbox_path = args.bbox_json_folder / f"{scene_id}.json"
    if not bbox_path.exists():
        raise FileNotFoundError(f"缺少 bbox json（需含description字段）: {bbox_path}")

    qa_entries = load_json(qa_path)
    bbox_items = load_json(bbox_path)
    if not isinstance(qa_entries, list):
        raise ValueError(f"QA json 期望 list: {qa_path}")
    if not isinstance(bbox_items, list):
        raise ValueError(f"bbox json 期望 list: {bbox_path}")

    stem_label_map, stem_ins_map = build_image_label_index(bbox_items)
    client = get_thread_client(vllm_api_url)

    kept: List[Dict[str, Any]] = []
    total_in = total_out = total_drop = 0

    for e in tqdm(qa_entries, desc=f"{scene_id}", disable=not show_progress):
        total_in += 1

        if args.only_language_description and e.get("marker_type") != "language_description":
            total_drop += 1
            continue

        refs = extract_refs_from_entry(e, scene_id, stem_ins_map)
        if not refs:
            kept.append(e)
            total_out += 1
            continue

        keep_entry = True
        for ref in refs:
            others = [
                d
                for (iid, d) in (stem_label_map.get(ref.image_stem, {}).get(ref.label, []) or [])
                if iid != ref.ins_id
            ]
            if not others:
                continue

            if any(o.strip() == ref.description.strip() for o in others):
                keep_entry = False
                break

            others_key = tuple(sorted([o.strip() for o in others]))
            decision: Optional[str] = None

            if args.use_image:
                key_img = (ref.scene_id, ref.image_name, ref.label, ref.description.strip(), others_key)
                with cache_lock:
                    decision = cache_img.get(key_img)
                if decision is None:
                    prompt = build_disambiguation_prompt(ref, others)
                    image_path = resolve_image_path(args.data_root, ref.scene_id, ref.image_name)
                    out = call_vlm_json(
                        client,
                        image_path,
                        prompt,
                        max_image_side=args.max_image_side,
                    )
                    decision = str(out.get("decision", "DROP")).upper()
                    if decision not in {"KEEP", "DROP"}:
                        decision = "DROP"
                    with cache_lock:
                        cache_img[key_img] = decision
            else:
                key_txt = (ref.label, ref.description.strip(), others_key)
                with cache_lock:
                    decision = cache_txt.get(key_txt)
                if decision is None:
                    prompt = build_disambiguation_prompt(ref, others) + "\n\nIMPORTANT: You do NOT have access to the image. Use text only."
                    out = call_text_llm_json(client, prompt)
                    decision = str(out.get("decision", "DROP")).upper()
                    if decision not in {"KEEP", "DROP"}:
                        decision = "DROP"
                    with cache_lock:
                        cache_txt[key_txt] = decision

            if decision != "KEEP":
                keep_entry = False
                break

        if keep_entry:
            kept.append(e)
            total_out += 1
        else:
            total_drop += 1

    out_path = args.output_filter_repeat / qa_path.name
    out_path.write_text(json.dumps(kept, ensure_ascii=False, indent=2))
    return scene_id, total_in, total_out, total_drop


def main():
    parser = argparse.ArgumentParser(description="过滤 language_description 下的同类重复物体歧义 QA")
    parser.add_argument("--bbox-json-folder", type=Path, default="output_scannetppv2_new_aabb_with_descriptions")
    parser.add_argument("--output", type=Path, default="output_QA_new_lang_add_region")
    parser.add_argument("--output-filter-repeat", type=Path, default="output_QA_new_add_region_filter_repeat")
    parser.add_argument("--data-root", type=Path, default="scannetppv2/data", help="scannetppv2 数据根目录（包含各 scene 的 dslr/resized_undistorted_images）")
    parser.add_argument("--vllm-api-url", type=str, default=None, help="覆盖默认 vLLM API URL（/v1/chat/completions）")
    parser.add_argument("--vllm-api-urls", type=str, default=None, help="逗号分隔多个 vLLM API URL（/v1/chat/completions），优先于 --vllm-api-url")
    parser.add_argument("--max-workers", type=int, default=12, help="并行处理场景数量")
    parser.add_argument("--scenes", nargs="*", default=None, help="仅处理指定场景ID（可多个）")
    parser.add_argument("--max-image-side", type=int, default=1024, help="发送给 vLLM 的图像最长边，<=0 表示不缩放")
    parser.add_argument(
        "--no-use-image",
        action="store_true",
        help="不使用图片，仅用文字进行判别（更快但可能不准）",
    )
    parser.add_argument("--only-language-description", action="store_true", help="只处理 marker_type==language_description 的条目；其它条目丢弃")
    args = parser.parse_args()
    args.use_image = not args.no_use_image

    args.output_filter_repeat.mkdir(parents=True, exist_ok=True)

    vllm_api_urls = resolve_vllm_api_urls(args.vllm_api_urls, args.vllm_api_url)

    qa_files = sorted([p for p in args.output.glob("*.json") if p.is_file()])
    if not qa_files:
        raise FileNotFoundError(f"--output 下未找到任何 .json: {args.output}")

    if args.scenes:
        scene_set = set(args.scenes)
        qa_files = [p for p in qa_files if p.stem in scene_set]

    # 如果输出已存在则跳过
    qa_files = [
        p for p in qa_files
        if not (args.output_filter_repeat / p.name).exists()
    ]
    print(f"待处理场景数量: {len(qa_files)}")

    # cache key:
    #  - use-image: (scene_id, image_name, label, target, tuple(sorted(others)))
    #  - text-only: (label, target, tuple(sorted(others)))
    cache_img: Dict[Tuple[str, str, str, str, Tuple[str, ...]], str] = {}
    cache_txt: Dict[Tuple[str, str, Tuple[str, ...]], str] = {}
    cache_lock = threading.Lock()

    total_in = total_out = total_drop = 0

    if args.max_workers <= 1:
        for idx, qa_path in enumerate(qa_files):
            vllm_api_url = vllm_api_urls[idx % len(vllm_api_urls)]
            scene_id, in_count, out_count, drop_count = process_scene(
                qa_path, args, vllm_api_url, cache_img, cache_txt, cache_lock, True
            )
            total_in += in_count
            total_out += out_count
            total_drop += drop_count
            print(
                f"[{scene_id}] 输入 {in_count} 条，输出 {out_count} 条，丢弃 {drop_count} 条 -> "
                f"{args.output_filter_repeat / qa_path.name}"
            )
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(
                    process_scene,
                    qa_path,
                    args,
                    vllm_api_urls[idx % len(vllm_api_urls)],
                    cache_img,
                    cache_txt,
                    cache_lock,
                    True,
                ): qa_path
                for idx, qa_path in enumerate(qa_files)
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="scenes"):
                qa_path = futures[fut]
                scene_id, in_count, out_count, drop_count = fut.result()
                total_in += in_count
                total_out += out_count
                total_drop += drop_count
                print(
                    f"[{scene_id}] 输入 {in_count} 条，输出 {out_count} 条，丢弃 {drop_count} 条 -> "
                    f"{args.output_filter_repeat / qa_path.name}"
                )

    print(f"总计：输入 {total_in} 条，输出 {total_out} 条，丢弃 {total_drop} 条。")


if __name__ == "__main__":
    main()
