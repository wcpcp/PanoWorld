#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$ROOT/.venv/bin/python"
MODEL_LINK="$ROOT/DAP/checkpoints/model.pth"
MODEL_FILE="$ROOT/models/hf_model/model.pth"
OUTPUT_DIR="${1:-$ROOT/outputs/demo}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Virtualenv not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -f "$MODEL_FILE" ]]; then
  echo "Model weights not found: $MODEL_FILE" >&2
  exit 1
fi

mkdir -p "$ROOT/DAP/checkpoints"
ln -sf ../../models/hf_model/model.pth "$MODEL_LINK"

cd "$ROOT/DAP"
"$PYTHON_BIN" test/infer.py \
  --config config/infer_local.yaml \
  --txt datasets/demo_inputs.txt \
  --output "$OUTPUT_DIR" \
  --device auto

"$PYTHON_BIN" "$ROOT/make_demo_preview.py" \
  --input-dir "$ROOT/inputs/official_examples/hfdemo" \
  --depth-dir "$OUTPUT_DIR/depth_vis_color_100m" \
  --output "$OUTPUT_DIR/demo_contact_sheet.png"
