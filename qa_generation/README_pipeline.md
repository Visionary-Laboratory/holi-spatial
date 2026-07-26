# QA Generation Pipeline

This directory turns 3D object/region annotations into spatial QA and LLaMA-Factory training data.

## Inputs

- AABB post-processed object annotation JSONs: `output_scannetppv2_new_aabb/<scene_id>.json`
- Region annotation JSONs: `output_scannetppv2_region_new/<scene_id>.json`
- Dataset images and camera metadata: `scannetppv2/data/<scene_id>/...`
- Covisibility metadata: `scannetppv2_wai/<scene_id>/covisibility/v0/*.npy`
- OpenAI-compatible vLLM endpoint: default `http://localhost:8000/v1/chat/completions`

## Steps

1. Generate instance descriptions:

```bash
python qa_generation/generate_3d_instance_description.py \
  --input_dir output_scannetppv2_new_aabb \
  --output_dir output_scannetppv2_new_aabb_with_descriptions \
  --dataset_root scannetppv2/data \
  --vllm-api-url http://localhost:8000/v1/chat/completions
```

2. Generate two-view QA with object and region annotations:

```bash
python qa_generation/generate_two_view_qa_region.py \
  --data-root scannetppv2/data \
  --wai-root scannetppv2_wai \
  --marker-types language_description \
  --bbox-json-folder output_scannetppv2_new_aabb_with_descriptions \
  --region-bbox-json-folder output_scannetppv2_region_new \
  --output output_QA_new_lang_add_region
```

For object-only QA, use `generate_two_view_qa.py` and omit `--region-bbox-json-folder`.

3. Filter repeated descriptions:

```bash
python qa_generation/filter_qa_repeat_descriptions.py \
  --data-root scannetppv2/data \
  --bbox-json-folder output_scannetppv2_new_aabb_with_descriptions \
  --output output_QA_new_lang_add_region \
  --output-filter-repeat output_QA_new_add_region_filter_repeat \
  --vllm-api-url http://localhost:8000/v1/chat/completions
```

Use `--vllm-api-urls host1:8000,host2:8000` to shard filtering across multiple vLLM servers.

4. Convert to LLaMA-Factory JSONL:

```bash
python qa_generation/convert_qa_to_llamafactory.py \
  --image-root scannetppv2/data \
  --qa-json-dir output_QA_new_add_region_filter_repeat \
  --output-dir output_llamafactory_QA_add_region_lang
```

## Outputs

- `output_scannetppv2_new_aabb_with_descriptions/<scene_id>.json`
- `output_QA_new_lang_add_region/<scene_id>.json`
- `output_QA_new_add_region_filter_repeat/<scene_id>.json`
- `output_llamafactory_QA_add_region_lang/dataset.jsonl`
