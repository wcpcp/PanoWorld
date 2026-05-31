#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"
echo "Switched to directory: $SCRIPT_DIR"

GPU_DEVICES="${GPU_DEVICES:-6,7}"
GPU_NUM="${GPU_NUM:-$(echo "$GPU_DEVICES" | awk -F',' '{print NF}')}"
echo "Available GPU NUM = $GPU_NUM"

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
CONFIG_PATH="${CONFIG_PATH:-config/config.yaml}"

OUTPUT_DIR="${OUTPUT_DIR:-$(
  python3 - <<'PY' "$CONFIG_PATH"
import sys

try:
    import yaml
except Exception:
    print("./outputs")
    raise SystemExit(0)

path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    print(raw.get("training", {}).get("output_dir", "./outputs"))
except Exception:
    print("./outputs")
PY
)}"
mkdir -p "$OUTPUT_DIR"

export NCCL_NVLS_ENABLE=0
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

CUDA_VISIBLE_DEVICES="$GPU_DEVICES" torchrun \
  --nproc_per_node="$GPU_NUM" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  train.py \
  --config "$CONFIG_PATH" \
  "$@" \
  2>&1 | tee "$OUTPUT_DIR/train.log"
