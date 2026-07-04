# Holi-Spatial Pipeline

This repository contains the data curation pipeline used to turn ScanNet, ScanNet++, or DL3DV scenes into 3DGS geometry, mesh-guided masks, object/region 3D annotations, spatial QA, and LLaMA-Factory training data.

The pipeline follows the three stages described in the Holi-Spatial paper:

1. Geometric optimization: DA3 depth/point cloud initialization, then PGSR/3DGS training.
2. Image-level perception: VLM class or region discovery, then SAM3 mask generation.
3. Scene-level lift and refinement: lift masks into 3D, merge/filter boxes, caption instances, and synthesize spatial QA.

Paper: https://arxiv.org/abs/2603.07660

## Repository Layout

- `inference_da3_scannetppv2.py`, `run_da3.sh`: unified DA3 preprocessing for ScanNet, ScanNet++, and DL3DV.
- `PGSR/`: PGSR training, rendering, and mesh-to-mask code used by this pipeline.
- `3dgs_train.sh`: batch PGSR/3DGS training entrypoint for ScanNet, ScanNet++, and DL3DV.
- `mesh.sh`: render meshes and generate mesh-guided masks for ScanNet v2, ScanNet++, and DL3DV.
- `classic_vllm.py`, `classic_region.py`: VLM-based object and functional-region label discovery.
- `sam3.py`: SAM3 text-prompted mask generation.
- `3d_bounding_instance_gs_rerun_da3.py`: object 2D-to-3D lifting/refinement.
- `3d_bounding_instance_gs_region.py`: functional-region 2D-to-3D lifting/refinement.
- `postprocess_3d_bbox_aabb.py`: floor-aligned AABB-style box post-processing before QA generation.
- `scannetppv2_new.sh`, `scannetppv2_region.sh`, `scannetv2_all.sh`, `scannetv2_region.sh`, `DL3DV_new.sh`, `dl3dv_new_2k.sh`: dataset-specific annotation batch scripts.
- `qa_generation/`: instance captions, spatial QA generation, QA filtering, and LLaMA-Factory conversion.

## Setup

Create a CUDA-enabled Python environment, then install the common dependencies:

```bash
pip install -r requirements.txt
pip install -e PGSR/submodules/diff-plane-rasterization
pip install -e PGSR/submodules/simple-knn
```

Install or expose the external model packages required by the pipeline:

- Depth Anything 3: provides `depth_anything_3.api.DepthAnything3`.
- SAM3 checkpoint/assets: `sam3.py` downloads the default checkpoint and BPE asset from `facebook/sam3` when local paths are not provided.
- A VLM served through an OpenAI-compatible vLLM endpoint. All scripts default to `http://localhost:8000/v1` or `http://localhost:8000/v1/chat/completions`.

## Expected Data Layout

ScanNet v2:

```text
scannetv2/scans/<scene_id>/
  color/
  pose/
  intrinsic/
```

ScanNet++:

```text
scannetppv2/data/<scene_id>/
  dslr/nerfstudio/transforms_undistorted.json
  dslr/resized_undistorted_images/
```

DL3DV:

```text
processed_dl3dv_ours/1K/<scene_id>/
  dense/cam/
  dense/rgb/
```

The QA stage also needs a covisibility root, for example `scannetppv2_wai/<scene_id>/covisibility/v0/*.npy` or `dl3dv_wai/1K_<scene_id>/*.npy`.

## End-to-End Pipeline

### 1. DA3 Depth and Point Cloud

Process a directory of scenes:

```bash
SCENE_ROOT=scannetppv2/data bash run_da3.sh
```

Process a single scene:

```bash
SCENE_DIR=scannetppv2/data/0a5c013435 bash run_da3.sh
```

Outputs per scene:

- `depth_da3/<image_stem>.npy`
- `pointcloud_da3.ply`

DL3DV example:

```bash
SCENE_ROOT=processed_dl3dv_ours/1K bash run_da3.sh
```

For pre-split scene lists, such as ScanNet v2 lists:

```bash
SPLIT_DIR=scannetv2/splits NUM_GPUS=8 bash run_da3.sh
```

### 2. PGSR / 3DGS Training

The generic batch script supports ScanNet v2, ScanNet++, and DL3DV. Set roots explicitly for your dataset.

ScanNet v2 example:

```bash
SCENE_ROOT=scannetv2/scans \
OUTPUT_ROOT=pgsr_scannetv2_all \
LOG_FILE=train_pgsr_progress_scannetv2.json \
bash 3dgs_train.sh
```

ScanNet++ example:

```bash
SCENE_ROOT=scannetppv2/data \
OUTPUT_ROOT=pgsr_scannetppv2_all \
LOG_FILE=train_pgsr_progress_scannetppv2.json \
bash 3dgs_train.sh
```

The expected trained model output is:

```text
<OUTPUT_ROOT>/<scene_id>/point_cloud/iteration_30000/point_cloud.ply
```

### 3. Mesh and Mesh-Guided Masks

ScanNet v2 example:

```bash
DATA_ROOT=scannetv2/scans \
OUTPUT_ROOT=pgsr_scannetv2_all \
bash mesh.sh
```

ScanNet++ example:

```bash
DATA_ROOT=scannetppv2/data \
OUTPUT_ROOT=pgsr_scannetppv2_all \
bash mesh.sh
```

Outputs:

- `<3dgs_output>/<scene_id>/mesh/tsdf_fusion_post.ply`
- `<data_root>/<scene_id>/mask/`

### 4. Object and Region Annotation

Start an OpenAI-compatible vLLM server, then run object annotation.

ScanNet++ object instances:

```bash
DATA_ROOT=scannetppv2/data \
SCENES_DIR=pgsr_scannetppv2_all \
CLASS_JSON_DIR=Qwen3VL-32B-Scannetppv2 \
MASK_ROOT=sam3_masks_scannetppv2_new \
OUTPUT_JSON_DIR=output_scannetppv2_new \
VLLM_API_BASE=http://localhost:8000/v1 \
VLLM_CHAT_URL=http://localhost:8000/v1/chat/completions \
bash scannetppv2_new.sh
```

ScanNet++ functional regions:

```bash
DATA_ROOT=scannetppv2/data \
SCENES_DIR=pgsr_scannetppv2_all \
CLASS_JSON_DIR=Qwen3VL-32B-Scannetppv2_region \
MASK_ROOT=sam3_masks_scannetppv2_region \
OUTPUT_JSON_DIR=output_scannetppv2_region_new \
VLLM_API_BASE=http://localhost:8000/v1 \
VLLM_CHAT_URL=http://localhost:8000/v1/chat/completions \
bash scannetppv2_region.sh
```

The same pattern applies to `scannetv2_all.sh`, `scannetv2_region.sh`, `DL3DV_new.sh`, and `dl3dv_new_2k.sh`.

Annotation outputs are one JSON per scene, containing instance labels, source images, masks, 3D boxes, and metadata needed by QA generation.

### 5. AABB Box Post-Processing

Convert the raw 3D object boxes into floor-aligned AABB-style boxes before QA generation:

```bash
python postprocess_3d_bbox_aabb.py \
  --input_dir output_scannetppv2_new \
  --output_dir output_scannetppv2_new_aabb \
  --floor_label floor \
  --axis_method largest_face \
  --extent_mode keep
```

For ScanNet v2, use the corresponding object annotation directory:

```bash
python postprocess_3d_bbox_aabb.py \
  --input_dir output_scannetv2_new \
  --output_dir output_scannetv2_new_aabb \
  --floor_label floor
```

The region JSONs can be kept as generated by the region annotation step.

### 6. Instance Descriptions and QA

Generate object descriptions:

```bash
python qa_generation/generate_3d_instance_description.py \
  --input_dir output_scannetppv2_new_aabb \
  --output_dir output_scannetppv2_new_aabb_with_descriptions \
  --dataset_root scannetppv2/data \
  --vllm-api-url http://localhost:8000/v1/chat/completions
```

Generate object + region spatial QA:

```bash
python qa_generation/generate_two_view_qa_region.py \
  --data-root scannetppv2/data \
  --wai-root scannetppv2_wai \
  --marker-types language_description \
  --bbox-json-folder output_scannetppv2_new_aabb_with_descriptions \
  --region-bbox-json-folder output_scannetppv2_region_new \
  --output output_QA_new_lang_add_region
```

Filter repeated/ambiguous descriptions:

```bash
python qa_generation/filter_qa_repeat_descriptions.py \
  --data-root scannetppv2/data \
  --bbox-json-folder output_scannetppv2_new_aabb_with_descriptions \
  --output output_QA_new_lang_add_region \
  --output-filter-repeat output_QA_new_add_region_filter_repeat \
  --vllm-api-url http://localhost:8000/v1/chat/completions
```

### 7. Convert to LLaMA-Factory

```bash
python qa_generation/convert_qa_to_llamafactory.py \
  --image-root scannetppv2/data \
  --qa-json-dir output_QA_new_add_region_filter_repeat \
  --output-dir output_llamafactory_QA_add_region_lang
```

The final output contains `dataset.jsonl` and copied/rendered images in the format expected by LLaMA-Factory multimodal SFT.

## Main Outputs

- DA3: `depth_da3/`, `pointcloud_da3.ply`
- PGSR: `pgsr_*/<scene>/point_cloud/iteration_30000/point_cloud.ply`
- Mesh/masks: `mesh/tsdf_fusion_post.ply`, `<data_root>/<scene>/mask/`
- Object boxes: `output_scannetppv2_new/<scene>.json`
- AABB object boxes: `output_scannetppv2_new_aabb/<scene>.json`
- Region boxes: `output_scannetppv2_region_new/<scene>.json`
- Descriptions: `output_scannetppv2_new_aabb_with_descriptions/<scene>.json`
- QA: `output_QA_new_add_region_filter_repeat/<scene>.json`
- LLaMA-Factory: `output_llamafactory_QA_add_region_lang/dataset.jsonl`

## Notes

- All large datasets, model checkpoints, masks, QA outputs, and 3DGS outputs are ignored by `.gitignore`.
- The batch scripts skip scenes whose final output JSON or trained point cloud already exists.
- For single-scene debugging, call the Python scripts directly with `--scene`, `--data-root`, `--scene-json`, `--mask-root`, and `--model-path`.
