from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from datasets import Image, Sequence, load_dataset
from peft import LoraConfig
from transformers import AutoProcessor
from trl import SFTConfig, SFTTrainer


@dataclass(frozen=True)
class TrainConfig:
    model_name_or_path: str
    train_file: str
    output_dir: str
    eval_file: str | None = None
    num_train_epochs: float = 1.0
    max_steps: int = -1
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    expected_world_size: int | None = None
    expected_global_batch_size: int | None = None
    max_length: int = 32000
    assistant_only_loss: bool = True
    gradient_checkpointing: bool = True
    bf16: bool = True
    tf32: bool = True
    logging_steps: int = 5
    save_steps: int = 500
    eval_steps: int | None = 500
    save_total_limit: int | None = None
    seed: int = 3407
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str | list[str] = "all-linear"
    lora_exclude_modules: str | list[str] | None = None
    freeze_vision_tower: bool = True
    deepspeed: str | None = None
    report_to: str = "none"


def load_config(path: str) -> TrainConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("SFT config must be a YAML object")
    config = TrainConfig(**payload)
    world_size = int(__import__("os").environ.get("WORLD_SIZE", "1"))
    if config.expected_world_size is not None and world_size != config.expected_world_size:
        raise ValueError(
            f"this profile requires {config.expected_world_size} processes, got WORLD_SIZE={world_size}"
        )
    effective_batch = world_size * config.per_device_train_batch_size * config.gradient_accumulation_steps
    if config.expected_global_batch_size is not None and effective_batch != config.expected_global_batch_size:
        raise ValueError(
            f"global batch must be {config.expected_global_batch_size}, got {effective_batch}"
        )
    print(f"SFT effective batch: {world_size} x {config.per_device_train_batch_size} x {config.gradient_accumulation_steps} = {effective_batch}")
    return config


def load_split(path: str):
    dataset = load_dataset("json", data_files=path, split="train")
    if "messages" not in dataset.column_names:
        raise ValueError(f"{path} must contain a messages column")
    if "images" in dataset.column_names:
        dataset = dataset.cast_column("images", Sequence(Image(decode=True)))
    return dataset


def freeze_vision(model: Any) -> None:
    for name, parameter in model.named_parameters():
        if any(part in name.lower() for part in ("visual", "vision_tower", "vision_model")):
            parameter.requires_grad = False


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the multimodal tool-use policy with supervised trajectories.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default=None, help="Override model_name_or_path from YAML.")
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    model_name = args.model or cfg.model_name_or_path

    train_dataset = load_split(cfg.train_file)
    eval_dataset = load_split(cfg.eval_file) if cfg.eval_file else None
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    peft_config = None
    if cfg.use_lora:
        peft_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
            exclude_modules=cfg.lora_exclude_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

    train_args = SFTConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        max_length=cfg.max_length,
        assistant_only_loss=cfg.assistant_only_loss,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=cfg.bf16,
        tf32=cfg.tf32,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=cfg.eval_steps,
        save_total_limit=cfg.save_total_limit,
        seed=cfg.seed,
        report_to=cfg.report_to,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        model_init_kwargs={"trust_remote_code": True, "dtype": "bfloat16"},
        deepspeed=cfg.deepspeed,
    )
    trainer = SFTTrainer(
        model=model_name,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        peft_config=peft_config,
    )
    if cfg.freeze_vision_tower:
        freeze_vision(trainer.model)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(cfg.output_dir)
    processor.save_pretrained(cfg.output_dir)


if __name__ == "__main__":
    main()
