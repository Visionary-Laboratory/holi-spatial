


#!/usr/bin/env bash
set -euo pipefail

# 多卡多线程训练所有场景：每张 GPU 同时训练最多 2 个场景。
# 有场景完成后立刻在同一张卡上补下一个，直到跑完全部场景。
# 训练完成的场景会记录到 train_pgsr_progress.json，重复运行会跳过已完成场景。

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENE_ROOT="${SCENE_ROOT:-$ROOT_DIR/scannet/scans_test}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/output_scans_test/}"
LOG_FILE="${LOG_FILE:-$ROOT_DIR/train_pgsr_progress_scans_test.json}"
SLEEP_INTERVAL=5   # 轮询作业完成的间隔（秒）
MAX_JOBS_PER_GPU=3
DONE_PLY_REL="point_cloud/iteration_30000/point_cloud.ply"

# 线程安全地更新日志，维护四类状态：pending/running/completed/failed
update_log() {
  local mode="$1"  # init / start / done_ok / done_fail
  shift
  exec 200>"${LOG_FILE}.lock"
  flock 200
  python - "$LOG_FILE" "$mode" "$@" <<'PY'
import json, os, sys

log_path, mode, *rest = sys.argv[1:]
state = {"pending": [], "running": [], "completed": [], "failed": []}
if os.path.exists(log_path):
    try:
        loaded = json.load(open(log_path))
        if isinstance(loaded, dict):
            for k in state:
                if isinstance(loaded.get(k), list):
                    state[k] = loaded[k]
    except Exception:
        pass

def save():
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

def drop_all(name):
    for k in state:
        if name in state[k]:
            state[k].remove(name)

if mode == "init":
    # rest: pending scenes list
    state["pending"] = rest
    state["running"] = []
    # keep completed/failed from previous log if存在
    save()
    sys.exit(0)

if not rest:
    save()
    sys.exit(0)

scene = rest[0]
if mode == "start":
    drop_all(scene)
    state["running"].append(scene)
elif mode == "done_ok":
    drop_all(scene)
    state["completed"].append(scene)
elif mode == "done_fail":
    drop_all(scene)
    state["failed"].append(scene)

save()
PY
  flock -u 200
}

# 收集待训练场景：跳过 log 已完成、已有 point_cloud.ply、缺必要目录的
collect_scenes() {
  python - "$SCENE_ROOT" "$OUTPUT_ROOT" "$DONE_PLY_REL" "$LOG_FILE" <<'PY'
import json, os, sys
scene_root, output_root, done_rel, log_file = sys.argv[1:5]

done = set()
if os.path.exists(log_file):
    try:
        data = json.load(open(log_file))
        # 支持两种日志格式：
        # 1. 新格式：{"pending": [], "running": [], "completed": [], "failed": []}
        # 2. 旧格式：[{"scene": "...", "status": "ok"}, ...]
        if isinstance(data, dict):
            # 新格式：从 completed 列表中提取
            done = set(data.get("completed", []))
        elif isinstance(data, list):
            # 旧格式：从列表中提取已完成的场景
            done = {d["scene"] for d in data if d.get("status") == "ok" and "scene" in d}
    except Exception:
        done = set()

keep = []
skip_done = 0
skip_missing = 0

for name in sorted(os.listdir(scene_root)):
    d = os.path.join(scene_root, name)
    if not os.path.isdir(d):
        continue
    if name in done:
        skip_done += 1
        continue
    if os.path.isfile(os.path.join(output_root, name, done_rel)):
        skip_done += 1
        continue
    # 检测 DL3DV 格式：dense/cam/ 和 dense/rgb/
    is_dl3dv = os.path.isdir(os.path.join(d, "dense", "cam")) and os.path.isdir(
        os.path.join(d, "dense", "rgb")
    )
    # 检测 ScanNet 格式：cam/ 和 color/
    is_scannet = os.path.isdir(os.path.join(d, "cam")) and os.path.isdir(
        os.path.join(d, "color")
    )
    if is_dl3dv or is_scannet:
        keep.append(name)
    else:
        skip_missing += 1

for n in keep:
    print(n)

print(f"#stats keep={len(keep)} skip_done={skip_done} skip_missing={skip_missing}", file=sys.stderr)
PY
}

run_scene() {
  local scene="$1"
  local gpu="$2"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    python PGSR/train.py \
      -s "$SCENE_ROOT/$scene" \
      -m "$OUTPUT_ROOT/$scene" \
      --resolution 1
    local code=$?
    exit $code
  ) &
}

main() {
  echo "SCENE_ROOT=$SCENE_ROOT"
  echo "OUTPUT_ROOT=$OUTPUT_ROOT"
  echo "LOG_FILE=$LOG_FILE"
  # 获取 GPU 数量
  mapfile -t gpu_list < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
  local gpu_count=${#gpu_list[@]}
  if (( gpu_count == 0 )); then
    echo "未检测到 GPU，退出。" >&2
    exit 1
  fi
  echo "检测到 ${gpu_count} 张 GPU：${gpu_list[*]}"

  mapfile -t scenes < <(collect_scenes)
  # 提取并打印统计行
  local stats_line
  for ((i=${#scenes[@]}-1; i>=0; i--)); do
    if [[ "${scenes[$i]}" == \#stats* ]]; then
      stats_line="${scenes[$i]}"
      unset 'scenes[$i]'
    fi
  done
  if [[ -n "${stats_line:-}" ]]; then
    echo "场景过滤统计: ${stats_line#\#stats }"
  fi

  if (( ${#scenes[@]} == 0 )); then
    echo "没有待训练的场景（可能全部已完成）。"
    exit 0
  fi

  echo "待训练场景数：${#scenes[@]}"
  if (( ${#scenes[@]} > 0 )); then
    echo "示例场景（前 10 个）：${scenes[*]:0:10}"
  fi

  # 初始化日志的 pending 列表
  update_log init "${scenes[@]}"

  # GPU 运行计数与 PID 映射
  declare -A gpu_running=()
  for g in "${gpu_list[@]}"; do
    gpu_running[$g]=0
  done
  declare -A pid_gpu=()
  declare -A pid_scene=()

  running=0
  max_total=$(( ${#gpu_list[@]} * MAX_JOBS_PER_GPU ))

  start_one() {
    # 选择当前最空闲的 GPU
    local best_gpu=
    local min_jobs=9999
    for g in "${gpu_list[@]}"; do
      local j=${gpu_running[$g]}
      if (( j < MAX_JOBS_PER_GPU )) && (( j < min_jobs )); then
        best_gpu=$g
        min_jobs=$j
      fi
    done
    [[ -z "$best_gpu" ]] && return 1

    local scene="${scenes[0]}"
    scenes=("${scenes[@]:1}")
    echo "启动场景 ${scene} 在 GPU ${best_gpu}（槽位 $((${gpu_running[$best_gpu]}+1))/${MAX_JOBS_PER_GPU}）"
    update_log start "$scene"
    run_scene "$scene" "$best_gpu"
    local pid=$!
    pid_gpu[$pid]=$best_gpu
    pid_scene[$pid]=$scene
    ((gpu_running[$best_gpu]++))
    ((running++))
    return 0
  }

  # 先尽量填满所有槽位
  while (( running < max_total )) && (( ${#scenes[@]} > 0 )); do
    start_one || break
  done

  # 循环等待已完成的任务并补位
  while (( running > 0 )); do
    # wait -n 等待任意一个任务结束
    local finished_pid
    if ! wait -n; then
      : # 非零退出，后续根据 wait $pid 获取真实退出码
    fi
    # 找到已结束的 pid 并拿到退出码
    for pid in "${!pid_gpu[@]}"; do
      if ! kill -0 "$pid" 2>/dev/null; then
        finished_pid=$pid
        break
      fi
    done
    if [[ -n "${finished_pid:-}" ]]; then
      if wait "$finished_pid"; then
        local exit_code=0
      else
        local exit_code=$?
      fi
      local g=${pid_gpu[$finished_pid]}
      local sc=${pid_scene[$finished_pid]}
      unset pid_gpu["$finished_pid"]
      unset pid_scene["$finished_pid"]
      ((gpu_running[$g]--))
      ((running--))
      if (( exit_code == 0 )); then
        update_log done_ok "$sc"
        echo "完成场景 ${sc} (GPU ${g}, PID ${finished_pid}, status=ok)"
      else
        update_log done_fail "$sc"
        echo "场景失败 ${sc} (GPU ${g}, PID ${finished_pid}, code=${exit_code})"
      fi
    fi

    # 补位：尽量填满所有槽位
    while (( running < max_total )) && (( ${#scenes[@]} > 0 )); do
      start_one || break
    done

    echo "当前运行作业数：$running，队列剩余：${#scenes[@]}"
  done

  echo "全部场景训练流程结束。进度记录在 ${LOG_FILE}。"
}

main "$@"