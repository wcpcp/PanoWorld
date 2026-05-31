import argparse
import os
import shutil

from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from config.config import load_config
from data.collator import MultiModalDataCollator
from data.data import SupervisedDataset
from generation_eval_trainer import GenerationEvalTrainer
from utils import (
    build_choice_accuracy_metrics,
    build_exact_match_metrics,
    build_thinking360_interval_metrics,
    init_wandb,
    load_model,
    load_processor_and_tokenizer,
    preprocess_logits_for_metrics,
    print_model_parameters,
    rank0_print,
    set_model,
    set_seed,
)

RANK = int(os.environ.get("RANK", "0"))


def _resolve_eval_method(cfg) -> str:
    method = str(getattr(cfg.data, "eval_method", "") or "").strip().lower()
    if method:
        return method
    if bool(getattr(cfg.data, "eval_use_generation", False)):
        return "generation"
    return "teacher_forcing"


def _validate_required_paths(cfg):
    required_paths = {
        "model.name_or_path": cfg.model.name_or_path,
    }
    do_train = bool(getattr(cfg.run, "do_train", True))
    do_eval = bool(getattr(cfg.run, "do_eval", False))
    if not do_train and not do_eval:
        raise ValueError("At least one of `run.do_train` or `run.do_eval` must be true.")
    if do_train:
        required_paths["data.train_jsonl"] = cfg.data.train_jsonl
    if do_eval:
        if not cfg.data.eval_jsonl:
            raise ValueError("`data.eval_jsonl` is required when `run.do_eval` is true.")
        required_paths["data.eval_jsonl"] = cfg.data.eval_jsonl
    if cfg.training.deepspeed:
        required_paths["training.deepspeed"] = cfg.training.deepspeed

    missing = [
        f"{name}: {path}"
        for name, path in required_paths.items()
        if path and not os.path.exists(path)
    ]
    if missing:
        raise FileNotFoundError("Missing required path(s):\n" + "\n".join(missing))

    os.makedirs(cfg.training.output_dir, exist_ok=True)


def safe_save_model_for_hf_trainer(
    trainer: Trainer,
    output_dir: str,
    max_shard_size: str = "5GB",
):
    if trainer.accelerator is None:
        return

    trainer.accelerator.wait_for_everyone()
    state_dict_model = trainer.model_wrapped if trainer.is_deepspeed_enabled else trainer.model
    state_dict = trainer.accelerator.get_state_dict(state_dict_model)
    model_to_save = trainer.accelerator.unwrap_model(trainer.model)

    if trainer.args.should_save:
        safe_serialization = getattr(trainer.args, "save_safetensors", True)
        model_to_save.save_pretrained(
            output_dir,
            state_dict=state_dict,
            safe_serialization=safe_serialization,
            max_shard_size=max_shard_size,
        )

    trainer.accelerator.wait_for_everyone()


def _build_dataset(cfg, processor, tokenizer, *, split: str):
    is_train = split == "train"
    data_path = cfg.data.train_jsonl if is_train else cfg.data.eval_jsonl
    if not data_path:
        return None
    eval_method = _resolve_eval_method(cfg)

    return SupervisedDataset(
        jsonl_path=data_path,
        processor=processor,
        tokenizer=tokenizer,
        image_root=cfg.data.image_root,
        image_token=cfg.model.image_token,
        model_max_length=cfg.model.model_max_length,
        max_samples=cfg.data.train_max_samples if is_train else cfg.data.eval_max_samples,
        shuffle=cfg.data.shuffle if is_train else cfg.data.eval_shuffle,
        prompt_format=cfg.data.prompt_format,
        image_processor_cfg=cfg.data.image_processor,
        system_prompt=cfg.data.system_prompt,
        auto_insert_media_placeholders=cfg.data.auto_insert_media_placeholders,
        strict_image_checks=cfg.data.strict_image_checks,
        skip_missing_images=cfg.data.skip_missing_images,
        generation_eval=(not is_train and eval_method in {"generation", "choice_scoring"}),
        raw_generation_eval=(not is_train and eval_method == "generation"),
        disable_thinking=(not is_train and bool(getattr(cfg.data, "eval_disable_thinking", True))),
    )


def _has_thinking360_interval_fields(item) -> bool:
    if not isinstance(item, dict):
        return False
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    yaw = item.get("target_yaw_interval", metadata.get("target_yaw_interval"))
    pitch = item.get("target_pitch_interval", metadata.get("target_pitch_interval"))
    return isinstance(yaw, (list, tuple)) and isinstance(pitch, (list, tuple))


def _should_use_thinking360_metrics(eval_metric: str, eval_items, is_choice_eval: bool) -> bool:
    has_thinking360_intervals = any(
        _has_thinking360_interval_fields(item)
        for item in eval_items[: min(len(eval_items), 256)]
    )
    return (
        eval_metric in {"thinking360", "thinking360_success", "thinking360_success_rate"}
        or (eval_metric == "auto" and has_thinking360_intervals)
        or (eval_metric == "choice_accuracy" and has_thinking360_intervals and not is_choice_eval)
    )


def _resolve_compute_metrics(cfg, tokenizer, eval_dataset):
    if eval_dataset is None:
        return None
    eval_method = _resolve_eval_method(cfg)

    eval_metric = str(getattr(cfg.data, "eval_metric", "auto") or "auto").lower()
    eval_items = getattr(eval_dataset, "items", [])
    is_choice_eval = any(
        isinstance(item, dict) and (
            item.get("answer_format") in {"4_way_multiple_choice", "multiple_choice", "5_way_multiple_choice"} or
            bool(item.get("options"))
        )
        for item in eval_items[: min(len(eval_items), 64)]
    )

    if _should_use_thinking360_metrics(eval_metric, eval_items, is_choice_eval):
        return build_thinking360_interval_metrics(
            tokenizer,
            generation_mode=eval_method in {"generation", "choice_scoring"},
            eval_items=eval_items,
            print_predictions=bool(getattr(cfg.data, "eval_print_predictions", False)),
        )

    if eval_metric == "choice_accuracy" or (eval_metric == "auto" and is_choice_eval):
        valid_keys = []
        for item in eval_items:
            options = item.get("options")
            if not isinstance(options, list):
                continue
            for option in options:
                if not isinstance(option, dict):
                    continue
                key = option.get("key")
                if key:
                    valid_keys.append(str(key).upper())
        valid_keys = sorted(set(valid_keys)) or ["A", "B", "C", "D", "E"]
        return build_choice_accuracy_metrics(
            tokenizer,
            valid_keys=valid_keys,
            generation_mode=eval_method in {"generation", "choice_scoring"},
            eval_items=eval_items,
        )

    return build_exact_match_metrics(
        tokenizer,
        generation_mode=eval_method in {"generation", "choice_scoring"},
    )


def _resolve_resume_checkpoint(resume_from_checkpoint, output_dir: str):
    if resume_from_checkpoint in (None, False, "false", "False", "0", 0):
        return None

    if isinstance(resume_from_checkpoint, str):
        value = resume_from_checkpoint.strip()
        if not value:
            return None
        if value.lower() in {"true", "auto", "1", "yes"}:
            return get_last_checkpoint(output_dir)
        return value

    if resume_from_checkpoint:
        return get_last_checkpoint(output_dir)
    return None


def _resolve_warmup_args(warmup_steps_value):
    if isinstance(warmup_steps_value, float) and 0.0 < warmup_steps_value < 1.0:
        return 0, warmup_steps_value
    return warmup_steps_value, 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir_override = os.environ.get("OUTPUT_DIR")
    if output_dir_override:
        cfg.training.output_dir = output_dir_override
    _validate_required_paths(cfg)
    set_seed(cfg.training.seed)

    processor, tokenizer = load_processor_and_tokenizer(cfg)
    do_train = bool(getattr(cfg.run, "do_train", True))
    do_eval = bool(getattr(cfg.run, "do_eval", False))
    train_dataset = _build_dataset(cfg, processor, tokenizer, split="train") if do_train else None
    if do_train and train_dataset is None:
        raise ValueError("`data.train_jsonl` 不能为空。")
    eval_dataset = _build_dataset(cfg, processor, tokenizer, split="eval") if do_eval else None

    load_best_model_at_end = bool(do_train and cfg.training.load_best_model_at_end and eval_dataset is not None)
    metric_for_best_model = cfg.training.metric_for_best_model if load_best_model_at_end else None
    eval_items = getattr(eval_dataset, "items", []) if eval_dataset is not None else []
    is_choice_eval = any(
        isinstance(item, dict) and (
            item.get("answer_format") in {"4_way_multiple_choice", "multiple_choice", "5_way_multiple_choice"} or
            bool(item.get("options"))
        )
        for item in eval_items[: min(len(eval_items), 64)]
    )
    eval_metric = str(getattr(cfg.data, "eval_metric", "auto") or "auto").lower()
    if load_best_model_at_end and not metric_for_best_model:
        if _should_use_thinking360_metrics(eval_metric, eval_items, is_choice_eval):
            metric_for_best_model = "thinking360_success_rate"
        elif eval_metric == "choice_accuracy" or (eval_metric == "auto" and is_choice_eval):
            metric_for_best_model = "choice_accuracy"
        else:
            metric_for_best_model = "exact_match"
    greater_is_better = cfg.training.greater_is_better if load_best_model_at_end else None
    warmup_steps, warmup_ratio = _resolve_warmup_args(cfg.training.warmup_steps)

    if not do_train:
        cfg.training.gradient_checkpointing = False

    model = load_model(cfg)
    if do_train:
        set_model(cfg, model)
    else:
        for param in model.parameters():
            param.requires_grad = False

    training_args = TrainingArguments(
        output_dir=cfg.training.output_dir,
        run_name=cfg.training.run_name,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        num_train_epochs=cfg.training.num_train_epochs,
        logging_steps=cfg.training.logging_steps,
        save_steps=cfg.training.save_steps,
        eval_steps=cfg.training.eval_steps,
        max_steps=cfg.training.max_steps,
        eval_strategy=cfg.training.eval_strategy if do_train and eval_dataset is not None else "no",
        save_strategy=cfg.training.save_strategy,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        save_total_limit=cfg.training.save_total_limit,
        warmup_steps=warmup_steps,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        fp16=cfg.training.fp16,
        bf16=cfg.training.bf16,
        optim=cfg.training.optim,
        report_to=cfg.training.report_to,
        remove_unused_columns=cfg.training.remove_unused_columns,
        dataloader_num_workers=cfg.training.dataloader_num_workers,
        max_grad_norm=cfg.training.max_grad_norm,
        deepspeed=cfg.training.deepspeed,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    if RANK == 0:
        rank0_print(
            RANK,
            f"ERP adapter enabled={cfg.erp.enabled}, stage={cfg.erp.stage}, "
            f"target={cfg.erp.target}, pos_mode={cfg.erp.pos_mode}, "
            f"adapter_type={cfg.erp.adapter_type}, "
            f"cross_attn_embed_type={cfg.erp.cross_attn_embed_type}",
        )
        print_model_parameters(model)

    init_wandb(cfg.wandb, training_args, RANK)
    data_collator = MultiModalDataCollator(tokenizer)

    eval_method = _resolve_eval_method(cfg)
    eval_prompt_only = eval_dataset is not None and eval_method in {"generation", "choice_scoring"}
    trainer_cls = GenerationEvalTrainer if eval_prompt_only else Trainer
    trainer_kwargs = {}
    if eval_prompt_only:
        trainer_kwargs.update(
            eval_use_generation=(eval_method == "generation"),
            eval_method=eval_method,
            eval_print_predictions=bool(getattr(cfg.data, "eval_print_predictions", False)),
            generation_max_new_tokens=int(getattr(cfg.data, "eval_generation_max_new_tokens", 16)),
            generation_do_sample=bool(getattr(cfg.data, "eval_generation_do_sample", False)),
            generation_num_beams=int(getattr(cfg.data, "eval_generation_num_beams", 1)),
            generation_processor=processor,
        )

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=_resolve_compute_metrics(cfg, tokenizer, eval_dataset),
        preprocess_logits_for_metrics=(
            None if eval_prompt_only
            else (preprocess_logits_for_metrics if eval_dataset is not None else None)
        ),
        **trainer_kwargs,
    )

    eval_before_train = bool(getattr(cfg.run, "eval_before_train", False))

    if eval_before_train or (not do_train and eval_dataset is not None):
        if eval_dataset is None:
            rank0_print(RANK, "skip initial eval: eval_before_train=True but eval_dataset is None")
        else:
            eval_label = "evaluation" if not do_train else "initial evaluation before training"
            rank0_print(RANK, f"running {eval_label}...")
            initial_metrics = trainer.evaluate()
            if RANK == 0:
                rank0_print(RANK, f"eval metrics: {initial_metrics}")

    if not do_train:
        if RANK == 0:
            rank0_print(RANK, "run.do_train is false; finished evaluation without training.")
        report_to = cfg.training.report_to or []
        if isinstance(report_to, str):
            report_to = [report_to]
        if "wandb" in report_to:
            import wandb

            wandb.finish()
        return

    rank0_print(RANK, "calling trainer.train()")

    resume_checkpoint = _resolve_resume_checkpoint(
        cfg.run.resume_from_checkpoint,
        training_args.output_dir,
    )
    if resume_checkpoint:
        rank0_print(RANK, f"resuming from checkpoint: {resume_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        trainer.train()

    trainer.save_state()

    if RANK == 0 and getattr(trainer.state, "best_model_checkpoint", None):
        best_ckpt_file = os.path.join(training_args.output_dir, "best_checkpoint.txt")
        with open(best_ckpt_file, "w", encoding="utf-8") as f:
            f.write(trainer.state.best_model_checkpoint + "\n")
            if getattr(trainer.state, "best_metric", None) is not None:
                f.write(f"best_metric={trainer.state.best_metric}\n")

    source_path = os.path.join(cfg.model.name_or_path, "chat_template.json")
    template_path = os.path.join(training_args.output_dir, "chat_template.json")
    if os.path.exists(source_path):
        shutil.copy(source_path, template_path)

    model.config.use_cache = True
    safe_save_model_for_hf_trainer(
        trainer=trainer,
        output_dir=training_args.output_dir,
        max_shard_size=cfg.training.max_shard_size,
    )

    if RANK == 0:
        processor.save_pretrained(cfg.training.output_dir)
        tokenizer.save_pretrained(cfg.training.output_dir)
        report_to = cfg.training.report_to or []
        if isinstance(report_to, str):
            report_to = [report_to]
        if "wandb" in report_to:
            import wandb

            wandb.finish()


if __name__ == "__main__":
    main()
