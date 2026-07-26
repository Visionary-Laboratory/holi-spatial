# Convert QA to LLaMA-Factory

`convert_qa_to_llamafactory.py` converts per-scene QA JSON files into a multimodal `dataset.jsonl` plus referenced images.

## Usage

```bash
python qa_generation/convert_qa_to_llamafactory.py \
  --image-root scannetppv2/data \
  --qa-json-dir output_QA_new_add_region_filter_repeat \
  --output-dir output_llamafactory_QA_add_region_lang
```

Single scene:

```bash
python qa_generation/convert_qa_to_llamafactory.py \
  --image-root scannetppv2/data \
  --qa-json-dir output_QA_new_add_region_filter_repeat \
  --output-dir output_llamafactory_QA_add_region_lang \
  --scene-id 0a5c013435
```

## Output

```text
output_llamafactory_QA_add_region_lang/
  dataset.jsonl
  images/<scene_id>/
```

`dataset.jsonl` contains LLaMA-Factory style records with `conversations` and one or more image paths. Marker-based QA entries create rendered marker images; `language_description` entries reuse original images.
