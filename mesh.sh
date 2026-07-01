#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
SCANNET_IMAGE_SUBDIR="${SCANNET_IMAGE_SUBDIR:-color}"
SCANNETPP_NERFSTUDIO_SUBDIR="${SCANNETPP_NERFSTUDIO_SUBDIR:-dslr/nerfstudio}"
SCANNETPP_IMAGE_SUBDIR="${SCANNETPP_IMAGE_SUBDIR:-dslr/resized_undistorted_images}"
DL3DV_IMAGE_SUBDIR="${DL3DV_IMAGE_SUBDIR:-rgb}"
RENDER_MAX_DEPTH="${RENDER_MAX_DEPTH:-4.0}"
MASK_MAX_DEPTH="${MASK_MAX_DEPTH:-6.0}"
VOXEL_SIZE="${VOXEL_SIZE:-0.006}"
NUM_CLUSTER="${NUM_CLUSTER:-1}"
MAX_PARALLEL="${MAX_PARALLEL:-}"
NUM_GPUS="${NUM_GPUS:-}"
FORCE="${FORCE:-0}"
USE_DEPTH_FILTER="${USE_DEPTH_FILTER:-0}"
MESH_PATH_REL="${MESH_PATH_REL:-mesh/tsdf_fusion_post.ply}"
DONE_PLY_REL="point_cloud/iteration_30000/point_cloud.ply"

usage() {
    cat <<'EOF'
Render PGSR/3DGS meshes and generate mesh-guided masks for ScanNet v2, ScanNet++, or DL3DV.

Required:
  DATA_ROOT=/path/to/scenes        Directory containing scene folders.
  OUTPUT_ROOT=/path/to/3dgs        Directory containing trained 3DGS outputs.

Optional:
  NUM_GPUS=8                       Number of GPUs. Defaults to nvidia-smi count.
  MAX_PARALLEL=8                   Max concurrent scenes. Defaults to NUM_GPUS.
  RENDER_MAX_DEPTH=4.0             Max depth for TSDF fusion.
  MASK_MAX_DEPTH=6.0               Max depth for mesh-to-mask filtering.
  VOXEL_SIZE=0.006                 TSDF voxel size and mask depth threshold.
  NUM_CLUSTER=1                    Number of connected mesh clusters to keep.
  USE_DEPTH_FILTER=1               Enable PGSR render depth filtering.
  FORCE=1                          Re-render mesh and masks even if outputs exist.

Examples:
  DATA_ROOT=scannetv2/scans OUTPUT_ROOT=pgsr_scannetv2_all bash mesh.sh
  DATA_ROOT=scannetppv2/data OUTPUT_ROOT=pgsr_scannetppv2_all bash mesh.sh
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

if [ -z "$DATA_ROOT" ] || [ -z "$OUTPUT_ROOT" ]; then
    usage >&2
    echo "Error: DATA_ROOT and OUTPUT_ROOT are required." >&2
    exit 2
fi

if [ ! -d "$DATA_ROOT" ]; then
    echo "Error: DATA_ROOT does not exist or is not a directory: $DATA_ROOT" >&2
    exit 2
fi

if [ ! -d "$OUTPUT_ROOT" ]; then
    echo "Error: OUTPUT_ROOT does not exist or is not a directory: $OUTPUT_ROOT" >&2
    exit 2
fi

if [ -z "$NUM_GPUS" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l)
    else
        NUM_GPUS=1
    fi
fi

if [ "$NUM_GPUS" -lt 1 ]; then
    echo "Error: NUM_GPUS must be >= 1." >&2
    exit 2
fi

if [ -z "$MAX_PARALLEL" ]; then
    MAX_PARALLEL="$NUM_GPUS"
fi

scene_layout() {
    local scene_dir="$1"
    if [ -f "$scene_dir/$SCANNETPP_NERFSTUDIO_SUBDIR/transforms_undistorted.json" ] && \
       [ -d "$scene_dir/$SCANNETPP_IMAGE_SUBDIR" ]; then
        echo "scannetpp"
    elif [ -d "$scene_dir/dense/cam" ] && [ -d "$scene_dir/dense/$DL3DV_IMAGE_SUBDIR" ]; then
        echo "dl3dv"
    elif [ -d "$scene_dir/pose" ] && [ -d "$scene_dir/$SCANNET_IMAGE_SUBDIR" ]; then
        echo "scannet"
    else
        return 1
    fi
}

set_scene_args() {
    local scene_dir="$1"
    local layout="$2"
    SOURCE_PATH="$scene_dir"
    IMAGE_ARG=()

    case "$layout" in
        scannetpp)
            SOURCE_PATH="$scene_dir/$SCANNETPP_NERFSTUDIO_SUBDIR"
            IMAGE_ARG=(-i "$scene_dir/$SCANNETPP_IMAGE_SUBDIR")
            ;;
        dl3dv)
            IMAGE_ARG=(-i "$DL3DV_IMAGE_SUBDIR")
            ;;
        scannet)
            IMAGE_ARG=(-i "$SCANNET_IMAGE_SUBDIR")
            ;;
        *)
            echo "Error: unknown scene layout: $layout" >&2
            return 1
            ;;
    esac
}

SCENES=()
for scene_dir in "$DATA_ROOT"/*/; do
    [ -d "$scene_dir" ] || continue
    scene=$(basename "$scene_dir")
    if ! layout=$(scene_layout "$scene_dir"); then
        echo "Skip unrecognized scene layout: $scene"
        continue
    fi
    if [ ! -f "$OUTPUT_ROOT/$scene/$DONE_PLY_REL" ]; then
        echo "Skip scene without trained 3DGS: $scene"
        continue
    fi
    SCENES+=("$scene:$layout")
done

echo "Found ${#SCENES[@]} scenes to process."

process_scene() {
    local scene="$1"
    local layout="$2"
    local gpu_id="$3"
    local scene_dir="$DATA_ROOT/$scene"
    local model_dir="$OUTPUT_ROOT/$scene"
    local mesh_file="$model_dir/$MESH_PATH_REL"
    local mask_dir="$scene_dir/mask"
    local -a render_args
    local -a mask_args

    export CUDA_VISIBLE_DEVICES="$gpu_id"
    set_scene_args "$scene_dir" "$layout"

    echo "[GPU $gpu_id] Start scene=$scene layout=$layout"

    if [ "$FORCE" != "1" ] && [ -f "$mesh_file" ]; then
        echo "[GPU $gpu_id] Mesh exists, skip render: $mesh_file"
    else
        render_args=(
            -s "$SOURCE_PATH"
            -m "$model_dir"
            --skip_test
            --max_depth "$RENDER_MAX_DEPTH"
            --voxel_size "$VOXEL_SIZE"
            --num_cluster "$NUM_CLUSTER"
        )
        if [ "$USE_DEPTH_FILTER" = "1" ]; then
            render_args+=(--use_depth_filter)
        fi
        render_args+=("${IMAGE_ARG[@]}")

        python PGSR/render.py "${render_args[@]}"
    fi

    if [ ! -f "$mesh_file" ]; then
        echo "[GPU $gpu_id] Error: mesh was not generated: $mesh_file" >&2
        return 1
    fi

    if [ "$FORCE" != "1" ] && [ -d "$mask_dir" ]; then
        echo "[GPU $gpu_id] Mask directory exists, skip mesh2mask: $mask_dir"
        return 0
    fi

    mask_args=(
        -m "$model_dir"
        -s "$SOURCE_PATH"
        --mesh_path "$MESH_PATH_REL"
        --max_depth "$MASK_MAX_DEPTH"
        --voxel_size "$VOXEL_SIZE"
    )
    mask_args+=("${IMAGE_ARG[@]}")

    python PGSR/mesh2mask.py "${mask_args[@]}"
    echo "[GPU $gpu_id] Done scene=$scene"
}

current_slot=0
failed=0

for item in "${SCENES[@]}"; do
    scene="${item%%:*}"
    layout="${item#*:}"

    while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
        if ! wait -n; then
            failed=$((failed + 1))
        fi
    done

    gpu_id=$((current_slot % NUM_GPUS))
    process_scene "$scene" "$layout" "$gpu_id" &
    echo "Launched scene=$scene layout=$layout GPU=$gpu_id PID=$!"
    current_slot=$((current_slot + 1))
done

while [ "$(jobs -rp | wc -l)" -gt 0 ]; do
    if ! wait -n; then
        failed=$((failed + 1))
    fi
done

if [ "$failed" -ne 0 ]; then
    echo "Mesh/mask processing finished with $failed failed job(s)." >&2
    exit 1
fi

echo "Mesh/mask processing finished successfully."
