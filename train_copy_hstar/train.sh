#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"
echo "Switched to directory: $SCRIPT_DIR"

GPU_DEVICES="${GPU_DEVICES:-6,7}"
GPU_NUM="${GPU_NUM:-$(echo "$GPU_DEVICES" | awk -F',' '{print NF}')}"
echo "Available GPU NUM = $GPU_NUM"

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
if [[ -z "${MASTER_PORT+x}" ]]; then
    MASTER_PORT="29500"
    AUTO_SELECT_MASTER_PORT=1
else
    MASTER_PORT="${MASTER_PORT}"
    AUTO_SELECT_MASTER_PORT=0
fi
CONFIG_PATH="${CONFIG_PATH:-config/config.yaml}"

if [[ "$AUTO_SELECT_MASTER_PORT" -eq 1 ]]; then
    MASTER_PORT="$({
        python3 - <<'PY' "$MASTER_PORT"
import socket
import sys

preferred = int(sys.argv[1])

def is_free(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        sock.close()

selected = None
for port in range(preferred, preferred + 201):
    if is_free(port):
        selected = port
        break

if selected is None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    selected = sock.getsockname()[1]
    sock.close()

print(selected)
PY
    })"
fi
echo "Using master endpoint: ${MASTER_ADDR}:${MASTER_PORT}"

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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

CUDA_VISIBLE_DEVICES="$GPU_DEVICES" torchrun \
  --nproc_per_node="$GPU_NUM" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  train.py \
  --config "$CONFIG_PATH" \
  "$@" \
  2>&1 | tee "$OUTPUT_DIR/train.log"
