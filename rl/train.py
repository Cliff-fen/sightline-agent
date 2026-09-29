from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from datasets import Image, Sequence, load_dataset
import torch
from peft import LoraConfig, PeftConfig, PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor
from trl import GRPOConfig, GRPOTrainer

try:
    from .environment import SearchEnvironment
    from .rewards import answer_reward, fatal_review_reward, final_answer_reward
    from .tool_protocol import configure_tool_response_parser
except ImportError:  # Direct execution: python rl/train.py
    from environment import SearchEnvironment
    from rewards import answer_reward, fatal_review_reward, final_answer_reward
    from tool_protocol import configure_tool_response_parser


@dataclass(frozen=True)
class TrainConfig:
    model_name_or_path: str
    train_file: str
    output_dir: str
    eval_file: str | None = None
    max_steps: int = -1
    num_train_epochs: float = 1.0
    learning_rate: float = 1e-6
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    num_generations: int = 8
    steps_per_generation: int | None = None
    max_completion_length: int = 2048
    max_tool_calling_iterations: int = 6
    temperature: float = 0.7
    top_p: float = 1.0
    beta: float = 0.0
    loss_type: str = "dapo"
    scale_rewards: str = "group"
    mask_truncated_completions: bool = True
    gradient_checkpointing: bool = True
    bf16: bool = True
    tf32: bool = True
    logging_steps: int = 1
    log_completions: bool = False
    num_completions_to_print: int | None = None
    save_steps: int = 500
    eval_steps: int | None = 500
    save_total_limit: int | None = None
    seed: int = 3407
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str | list[str] = "all-linear"
    lora_exclude_modules: str | list[str] | None = None
    use_vllm: bool = True
    vllm_mode: str = "colocate"
    vllm_gpu_memory_utilization: float = 0.35
    vllm_max_model_length: int = 10240
    deepspeed: str | None = None
    report_to: str = "none"


def load_config(path: str) -> TrainConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("GRPO config must be a YAML object")
    config = TrainConfig(**payload)
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    effective_batch = world_size * config.per_device_train_batch_size * config.gradient_accumulation_steps
    generation_batch = world_size * config.per_device_train_batch_size * (
        config.steps_per_generation or config.gradient_accumulation_steps
    )
    if effective_batch % config.num_generations:
        raise ValueError("effective batch must be divisible by num_generations")
    if generation_batch % config.num_generations:
        raise ValueError("generation batch must be divisible by num_generations")
    print(f"GRPO effective batch: {world_size} x {config.per_device_train_batch_size} x {config.gradient_accumulation_steps} = {effective_batch}")
    print(f"GRPO generation batch: {generation_batch}")
    return config


def load_split(path: str):
    dataset = load_dataset("json", data_files=path, split="train")
    required = {"prompt", "answer"}
    missing = required.difference(dataset.column_names)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if "images" in dataset.column_names:
        dataset = dataset.cast_column("images", Sequence(Image(decode=True)))
    return dataset


def load_policy(model_name: str, cfg: TrainConfig):
    adapter_path = Path(model_name) / "adapter_config.json"
    processor = configure_tool_response_parser(
        AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    )
    if adapter_path.is_file():
        adapter_config = PeftConfig.from_pretrained(model_name)
        base_model = AutoModelForImageTextToText.from_pretrained(
            adapter_config.base_model_name_or_path,
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )
        policy = PeftModel.from_pretrained(base_model, model_name, is_trainable=True)
        return policy, processor, None

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
    return model_name, processor, peft_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a multimodal tool-use policy with GRPO.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default=None, help="Override model_name_or_path from YAML.")
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    model_name = args.model or cfg.model_name_or_path
    train_dataset = load_split(cfg.train_file)
    eval_dataset = load_split(cfg.eval_file) if cfg.eval_file else None
    policy, processor, peft_config = load_policy(model_name, cfg)
    train_args = GRPOConfig(
        output_dir=cfg.output_dir,
        max_steps=cfg.max_steps,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_generations=cfg.num_generations,
        steps_per_generation=cfg.steps_per_generation,
        max_completion_length=cfg.max_completion_length,
        max_tool_calling_iterations=cfg.max_tool_calling_iterations,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        beta=cfg.beta,
        loss_type=cfg.loss_type,
        scale_rewards=cfg.scale_rewards,
        mask_truncated_completions=cfg.mask_truncated_completions,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=cfg.bf16,
        tf32=cfg.tf32,
        logging_steps=cfg.logging_steps,
        log_completions=cfg.log_completions,
        num_completions_to_print=cfg.num_completions_to_print,
        save_steps=cfg.save_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=cfg.eval_steps,
        save_total_limit=cfg.save_total_limit,
        seed=cfg.seed,
        report_to=cfg.report_to,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        use_vllm=cfg.use_vllm,
        vllm_mode=cfg.vllm_mode,
        vllm_gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
        vllm_max_model_length=cfg.vllm_max_model_length,
        model_init_kwargs={"trust_remote_code": True, "dtype": "bfloat16"} if isinstance(policy, str) else None,
        deepspeed=cfg.deepspeed,
    )
    trainer = GRPOTrainer(
        model=policy,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        peft_config=peft_config,
        reward_funcs=[answer_reward, final_answer_reward, fatal_review_reward],
        environment_factory=SearchEnvironment,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(cfg.output_dir)
    processor.save_pretrained(cfg.output_dir)


if __name__ == "__main__":
    main()
