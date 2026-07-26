# 3D Instance Description

`generate_3d_instance_description.py` adds a short, 3D-consistent description to each object instance JSON.

The script reads each 3D annotation, finds the highest-confidence source mask/image, overlays the mask, and asks a VLM to describe the object without 2D-only phrases such as "left side" or "center of the image".

## Usage

```bash
python qa_generation/generate_3d_instance_description.py \
  --input_dir output_scannetppv2_new_aabb \
  --output_dir output_scannetppv2_new_aabb_with_descriptions \
  --dataset_root scannetppv2/data \
  --vllm-api-url http://localhost:8000/v1/chat/completions
```

Single scene:

```bash
python qa_generation/generate_3d_instance_description.py \
  --json_file output_scannetppv2_new_aabb/0a5c013435.json \
  --output_dir output_scannetppv2_new_aabb_with_descriptions \
  --dataset_root scannetppv2/data \
  --vllm-api-url http://localhost:8000/v1/chat/completions
```

## Expected Input

Each input JSON should come from `3d_bounding_instance_gs_rerun_da3.py` and include instance records with source images and mask metadata.

## Output

The output JSON keeps the original fields and adds an instance-level `description` field. These descriptions are consumed by `generate_two_view_qa.py`, `generate_two_view_qa_region.py`, and `filter_qa_repeat_descriptions.py`.
