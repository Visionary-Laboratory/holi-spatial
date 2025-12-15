import os
import json
import argparse
from collections import defaultdict
import numpy as np

from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForVision2Seq, AutoProcessor


# ----------------------------------------------------------------------
# Load local GPT OSS-20B
# ----------------------------------------------------------------------
def load_local_llm(model_path):
    print(f"Loading local LLM: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, tokenizer


def ask_llm(model, tokenizer, label_list):
    if len(label_list) == 0:
        raise NotImplementedError
    elif len(label_list) == 1:
        canonical = label_list[0]
    else:
        """Ask local GPT model for canonical name."""
        prompt = (
            f'将以下名词视为同一物体的不同叫法：{label_list}。请从中选出一个最能代表该物体的英文名词，用 JSON 格式输出，不要输出多余内容。输出格式如下（不要换行，不要解释）：["label": "<答案>"]'
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=False
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)
        ]
        text = tokenizer.decode(generated_ids_trimmed[0], skip_special_tokens=True)
        canonical = json.loads(text)['label'].lower().strip()

    return canonical


# ----------------------------------------------------------------------
# Clustering (open-set, semantic)
# ----------------------------------------------------------------------
def cluster_labels(labels, threshold=0.3):
    model = SentenceTransformer("/mnt/shared-storage-user/intern7shared/share_ckpt_hf/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf")
    emb = model.encode(labels, normalize_embeddings=True)

    clustering = AgglomerativeClustering(
        metric="cosine",
        linkage="average",
        distance_threshold=threshold,
        n_clusters=None
    )
    cluster_ids = clustering.fit_predict(emb)

    groups = defaultdict(list)
    for label, cid in zip(labels, cluster_ids):
        groups[cid].append(label)

    return groups, cluster_ids


# ----------------------------------------------------------------------
# Process a single JSON file
# ----------------------------------------------------------------------
def process_one_json(input_path, output_path, llm, tokenizer):
    print(f"\n=== Processing: {os.path.basename(input_path)} ===")

    data = json.load(open(input_path))
    categories = data["categories"]
    per_image = data["per_image"]

    # ---------------- Global Clustering on categories ----------------
    groups, cluster_ids = cluster_labels(categories)
    print(f"→ Found {len(groups)} semantic clusters")

    # ---------------- Canonicalization using local LLM ----------------
    cid_to_canonical = {}
    for cid, words in groups.items():
        canon = ask_llm(llm, tokenizer, words)
        cid_to_canonical[cid] = canon
        print(f"[Cluster {cid}] {words} → '{canon}'")

    # Map global categories → canonical
    canonical_categories = [cid_to_canonical[cid] for cid in cluster_ids]
    cleaned_categories = sorted(set(canonical_categories))

    # ---------------- Per-image normalization ----------------
    print("→ Normalizing per-image predictions...")
    label_to_canonical = {}
    for cid, words in groups.items():
        canon = cid_to_canonical[cid]
        for w in words:
            label_to_canonical[w] = canon
    normalized_per_image = {}
    for img, labels in per_image.items():
        new_labels = [label_to_canonical[l] for l in labels if l in label_to_canonical]
        normalized_per_image[img] = sorted(set(new_labels))

    # ---------------- Save output ----------------
    out = {
        "scene": data["scene"],
        "categories": cleaned_categories,
        "per_image": normalized_per_image
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    json.dump(out, open(output_path, "w"), indent=2)
    print(f"✔ Saved: {output_path}")


# ----------------------------------------------------------------------
# Process entire folder
# ----------------------------------------------------------------------
def process_folder(input_dir, output_dir, llm_path):
    # Load model once
    llm, tokenizer = load_local_llm(llm_path)

    # List JSON files
    all_files = [f for f in os.listdir(input_dir) if f.endswith(".json")]
    all_files.sort()

    print(f"\nFound {len(all_files)} JSON files in {input_dir}")

    for fname in all_files:
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)
        if os.path.exists(out_path):
            print(f'{out_path} already exists, skipping.')

        process_one_json(in_path, out_path, llm, tokenizer)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default='./scene_objects_Qwen3-VL-30B-A3B-Instruct')
    parser.add_argument("--output_folder", type=str, default='./scene_canonical_objects_Qwen3-VL-30B-A3B-Instruct')
    parser.add_argument("--llm_path", type=str, default="/mnt/shared-storage-user/intern7shared/share_ckpt_hf/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe")
    args = parser.parse_args()

    process_folder(args.input_folder, args.output_folder, args.llm_path)

