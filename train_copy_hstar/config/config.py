from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ERPConfig:
    enabled: bool = False
    pos_mode: str = "paper"
    stage: str = "output"
    target: str = "pooler"
    adapter_type: str = "additive"
    cross_attn_embed_type: str = "fourier"
    hidden_dim: int = 512
    num_heads: int = 8
    gate_init: float = 0.01
    use_layernorm: bool = True


@dataclass
class ModelConfig:
    name_or_path: str
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"
    attn_implementation: Optional[str] = "flash_attention_2"
    cache_dir: Optional[str] = None
    image_token: str = "<image>"
    model_max_length: int = 2048
    trainable_modules: Optional[Dict[str, bool]] = None


@dataclass
class DataConfig:
    train_jsonl: str
    eval_jsonl: Optional[str] = None
    image_root: Optional[str] = None
    train_max_samples: Optional[int] = None
    eval_max_samples: Optional[int] = None
    shuffle: bool = True
    eval_shuffle: bool = False
    prompt_format: str = "chat_template"
    image_processor: Optional[Dict[str, Any]] = None
    system_prompt: Optional[str] = None
    system_prompt_path: Optional[str] = None
    auto_insert_media_placeholders: bool = True
    strict_image_checks: bool = True
    skip_missing_images: bool = True
    eval_method: str = "generation"
    eval_metric: str = "auto"
    eval_use_generation: bool = True
    eval_disable_thinking: bool = True
    eval_print_predictions: bool = False
    eval_generation_max_new_tokens: int = 16
    eval_generation_do_sample: bool = False
    eval_generation_num_beams: int = 1


@dataclass
class TrainingConfig:
    output_dir: str = "./outputs"
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    num_train_epochs: int = 3
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    max_steps: int = -1
    eval_strategy: str = "no"
    save_strategy: str = "steps"
    max_shard_size: str = "5GB"
    load_best_model_at_end: bool = False
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    save_total_limit: int = 2
    warmup_steps: float = 0.03
    lr_scheduler_type: str = "cosine"
    fp16: bool = False
    bf16: bool = True
    optim: str = "adamw_torch"
    report_to: Optional[List[str]] = None
    run_name: Optional[str] = None
    seed: int = 42
    remove_unused_columns: bool = False
    dataloader_num_workers: int = 4
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    deepspeed: Optional[str] = None


@dataclass
class WandbConfig:
    project: Optional[str] = None
    entity: Optional[str] = None
    name: Optional[str] = None
    tags: Optional[List[str]] = None
    mode: Optional[str] = None
    group: Optional[str] = None
    notes: Optional[str] = None
    id: Optional[str] = None
    resume: Optional[str] = None


@dataclass
class RunConfig:
    do_train: bool = True
    do_eval: bool = False
    resume_from_checkpoint: Optional[Any] = None
    eval_before_train: bool = False
    eval_before_train_on_resume: bool = False


@dataclass
class TrainConfig:
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    run: RunConfig
    erp: ERPConfig
    wandb: Optional[WandbConfig] = None


def load_config(path: str) -> TrainConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    model = raw.get("model", {})
    data = raw.get("data", {})
    training = raw.get("training", {})
    run = raw.get("run", {})
    erp = raw.get("erp", {})
    wandb = raw.get("wandb", None)

    return TrainConfig(
        model=ModelConfig(**model),
        data=DataConfig(**data),
        training=TrainingConfig(**training),
        run=RunConfig(**run),
        erp=ERPConfig(**erp),
        wandb=WandbConfig(**wandb) if wandb else None,
    )
