# PanoWorld ERP Base-Model Training

This directory contains the PanoWorld base-model training code for Qwen3.5-VL with ERP-aware spherical geometry adapters. It is intended for full fine-tuning on ERP image-language data and for generation-based validation on panorama reasoning benchmarks.

## What This Trainer Provides

- Qwen3.5-VL training with ERP geometry injected into the visual stream.
- Full fine-tuning and configurable trainable module selection.
- ERP spherical position features based on yaw and pitch.
- Additive and DA2-style cross-attention adapter variants.
- Generation-based validation for multiple-choice and BFOV-style answers.
- DeepSpeed ZeRO configuration files for multi-GPU training.

The code has been trimmed to focus on ERP image training. Older VLN-specific action metrics, video branches, navigation data flattening, and unrelated training options have been removed from the public path.

## Directory Layout

| Path | Description |
| --- | --- |
| `train.py` | Main training entry point. |
| `generation_eval_trainer.py` | Trainer extension for generation-time evaluation. |
| `utils.py` | Model, tokenizer, evaluation, and utility helpers. |
| `config/config.yaml` | Default training configuration. Replace local paths before running. |
| `config/config.py` | Structured configuration loader. |
| `data/` | Dataset and collator code. |
| `modeling/` | ERP geometry and Qwen adapter implementation. |
| `deepspeed/` | ZeRO-0, ZeRO-2, and ZeRO-3 configs. |
| `train.sh` | Example launch script. |

## ERP Adapter

The default ERP feature is:

```python
[sin(yaw), cos(yaw), sin(pitch), cos(pitch)]
```

These features are computed for image patches and injected through one of two adapter families:

- `additive`: a lightweight MLP residual adapter.
- `da2_cross_attn`: a spherical-embedding cross-attention adapter inspired by DA2/SphereViT-style geometry modeling.

The main ERP options live under `erp` in `config/config.yaml`:

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

Recommended starting points:

- Use `adapter_type: additive` for the lightest baseline.
- Use `stage: merger` when trying cross-attention with lower token cost.
- Use `stage: patch` for the strongest early visual geometry injection, with higher memory and runtime cost.

## Data Formats

Training data is expected as JSON or JSONL records containing chat messages and image paths. A typical sample follows the Qwen multimodal chat style:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "/path/to/pano.jpg"},
        {"type": "text", "text": "Question text"}
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

Evaluation data is usually JSONL with fields such as:

```json
{
  "question": "Question text",
  "options": ["A", "B", "C", "D"],
  "answer": "A",
  "image_path": "/path/to/pano.jpg"
}
```

For BFOV-style tasks, generated coordinate answers are parsed during evaluation when the corresponding metric is enabled.

## Configuration

Before running, update these fields in `config/config.yaml`:

- `model.name_or_path`: local Qwen3.5-VL base model or checkpoint.
- `data.train_jsonl`: training data path.
- `data.eval_jsonl`: evaluation data path.
- `data.system_prompt_path`: optional system prompt file.
- `training.output_dir`: checkpoint output directory.
- `training.deepspeed`: DeepSpeed config path.
- `wandb.mode`: set to `online`, `offline`, or `disabled` as needed.

The committed configuration is an example and contains environment-specific paths. Treat it as a template.

## Run Training

From `train_copy/`:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 train.py \
  --config config/config.yaml
```

Or edit `train.sh` and run:

```bash
bash train.sh
```

## Evaluation Behavior

The default evaluation mode is generation-based:

```yaml
data:
  eval_method: "generation"
  eval_metric: "choice_accuracy"
  eval_generation_max_new_tokens: 512
  eval_generation_do_sample: false
  eval_generation_num_beams: 1
```

During generation evaluation, the trainer temporarily enables KV cache and disables gradient checkpointing for faster decoding, then restores the training state.

## Practical Notes

- `da2_cross_attn` is more expensive than the additive adapter, especially at the patch stage.
- Keep generated checkpoints, W&B runs, and model artifacts outside git.
- Use `skip_missing_images: true` only for exploratory runs; strict image checks are safer for final experiments.
- The code assumes recent `transformers`, PyTorch, and Qwen3.5-VL support.
