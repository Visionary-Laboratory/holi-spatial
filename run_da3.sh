#!/usr/bin/env bash
set -euo pipefail

SCRIPT="${SCRIPT:-inference_da3_scannetppv2.py}"
NUM_GPUS="${NUM_GPUS:-1}"

usage() {
    cat <<'EOF'
Run DA3 preprocessing for ScanNet v2, ScanNet++, or DL3DV scenes.

Set exactly one input mode:
  SCENE_ROOT=/path/to/scenes       Directory containing scene folders.
  SCENE_DIR=/path/to/scene         Single scene folder.
  SCENE_LIST=/path/to/scenes.txt   Text file with one scene path per line.
  SPLIT_DIR=/path/to/splits        Directory containing scenes_part1.txt, scenes_part2.txt, ...

Optional:
  NUM_GPUS=8                       Number of split files/GPU processes when SPLIT_DIR is used.
  LOAD_COLMAP=1                    Load poses from COLMAP sparse format.

Examples:
  SCENE_ROOT=scannetppv2/data bash run_da3.sh
  SCENE_ROOT=processed_dl3dv_ours/1K bash run_da3.sh
  SCENE_DIR=scannetppv2/data/0a5c013435 bash run_da3.sh
  SPLIT_DIR=scannetv2/splits NUM_GPUS=8 bash run_da3.sh
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

input_count=0
[ -n "${SCENE_ROOT:-}" ] && input_count=$((input_count + 1))
[ -n "${SCENE_DIR:-}" ] && input_count=$((input_count + 1))
[ -n "${SCENE_LIST:-}" ] && input_count=$((input_count + 1))
[ -n "${SPLIT_DIR:-}" ] && input_count=$((input_count + 1))

if [ "$input_count" -ne 1 ]; then
    usage >&2
    echo "Error: set exactly one of SCENE_ROOT, SCENE_DIR, SCENE_LIST, or SPLIT_DIR." >&2
    exit 2
fi

common_args=()
if [ "${LOAD_COLMAP:-0}" = "1" ]; then
    common_args+=(--load_colmap)
fi

if [ -n "${SPLIT_DIR:-}" ]; then
    pids=()
    for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
        split_file="$SPLIT_DIR/scenes_part$((gpu_id + 1)).txt"
        if [ ! -f "$split_file" ]; then
            echo "Skip missing split file: $split_file"
            continue
        fi
        CUDA_VISIBLE_DEVICES="$gpu_id" \
            python "$SCRIPT" --scene_list_txt "$split_file" "${common_args[@]}" &
        pids+=($!)
        echo "Started DA3 split=$split_file GPU=$gpu_id PID=$!"
    done

    if [ "${#pids[@]}" -eq 0 ]; then
        echo "Error: no split files found in $SPLIT_DIR." >&2
        exit 2
    fi

    for pid in "${pids[@]}"; do
        wait "$pid"
    done
    echo "All DA3 split jobs finished."
elif [ -n "${SCENE_ROOT:-}" ]; then
    python "$SCRIPT" --scannetppv2_dirs "$SCENE_ROOT" "${common_args[@]}"
elif [ -n "${SCENE_DIR:-}" ]; then
    python "$SCRIPT" --scannetppv2_dir "$SCENE_DIR" "${common_args[@]}"
else
    python "$SCRIPT" --scene_list_txt "$SCENE_LIST" "${common_args[@]}"
fi
