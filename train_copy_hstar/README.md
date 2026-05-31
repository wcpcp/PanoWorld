# H* / Thinking-in-360 Training

This directory is the H* / Thinking-in-360 training variant of the PanoWorld ERP trainer. It shares the same Qwen3.5-VL ERP adapter stack as `train_copy/`, but its default configuration targets H* / Thinking-in-360 data and the `thinking360_success_rate` evaluation metric.

## What Is Different from `train_copy/`

- The default data paths point to H* / Thinking-in-360 training and benchmark files.
- The default metric is `thinking360_success_rate`.
- The example run name and checkpoint directory are H*-specific.
- `generation_eval_trainer.py`, `data/`, and `utils.py` include parsing and scoring logic for Thinking-in-360 style answers.

Everything else follows the same base trainer design: Qwen3.5-VL full fine-tuning, ERP spherical geometry features, additive or cross-attention adapters, and DeepSpeed support.

## Directory Layout

| Path | Description |
| --- | --- |
| `train.py` | Main training entry point. |
| `generation_eval_trainer.py` | Generation evaluation trainer with H* metric support. |
| `utils.py` | Model, tokenizer, parsing, metric, and utility helpers. |
| `config/config.yaml` | H* training template. Replace local paths before running. |
| `data/` | Dataset and collator code. |
| `modeling/` | ERP geometry and Qwen adapter implementation. |
| `deepspeed/` | ZeRO-0, ZeRO-2, and ZeRO-3 configs. |
| `train.sh` | Example launch script. |

## ERP Adapter

The adapter configuration is controlled by the `erp` block:

```yaml
erp:
  enabled: true
  pos_mode: "paper"
  stage: "patch"
  target: "pooler"
  adapter_type: "da2_cross_attn"
  cross_attn_embed_type: "fourier"
  hidden_dim: 512
  num_heads: 8
  gate_init: 0.01
  use_layernorm: true
```

For faster experiments, switch to:

```yaml
erp:
  enabled: true
  stage: "patch"
  adapter_type: "additive"
  pos_mode: "paper"
```

For a stronger but more expensive geometry path, keep `da2_cross_attn` and consider moving the adapter to `stage: merger` if patch-stage training is too slow.

## Data Format

Training records should follow the same multimodal chat structure as the base trainer:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "/path/to/pano.jpg"},
        {"type": "text", "text": "Thinking-in-360 question"}
      ]
    },
    {
      "role": "assistant",
      "content": "Answer text"
    }
  ],
  "images": ["/path/to/pano.jpg"]
}
```

The benchmark file is expected to contain panorama image paths, questions, and gold answers in the H* / Thinking-in-360 format consumed by `data/data.py` and `utils.py`.

## Configuration

Update `config/config.yaml` before running:

- `model.name_or_path`: base model or a previously trained PanoWorld checkpoint.
- `data.train_jsonl`: H* / Thinking-in-360 training data.
- `data.eval_jsonl`: benchmark file.
- `training.output_dir`: checkpoint output directory.
- `training.save_steps` and `training.eval_steps`: validation/checkpoint cadence.
- `wandb.mode`: `online`, `offline`, or `disabled`.

The committed paths are examples from the original experiment environment and should be replaced for a new machine.

## Environment

Use the same training environment as `train_copy/`:

```bash
cd ../train_copy
conda env create -f environment.yml
conda activate vln
pip install -r requirements.txt
```

Then return to `train_copy_hstar/` for H* / Thinking-in-360 fine-tuning.

## Run Training

From `train_copy_hstar/`:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 train.py \
  --config config/config.yaml
```

Or edit and run:

```bash
bash train.sh
```

## Evaluation

The default H* evaluation uses generation:

```yaml
data:
  eval_method: "generation"
  eval_metric: "thinking360_success_rate"
  eval_generation_max_new_tokens: 512
  eval_generation_do_sample: false
  eval_generation_num_beams: 1
```

Generation evaluation is slower than teacher-forced scoring but better matches benchmark usage. Reduce `eval_steps`, `eval_max_samples`, or `eval_generation_max_new_tokens` for quick debugging runs.

## Practical Notes

- Do not commit checkpoints, W&B runs, generated outputs, or local caches.
- Keep `strict_image_checks: true` for final runs; use `skip_missing_images` only for exploratory data checks.
- Cross-attention adapters can increase memory use substantially. Start with small evaluation subsets when changing ERP stages or adapter types.
