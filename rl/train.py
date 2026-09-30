from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from datasets import Image, Sequence, load_dataset
import torch
from peft import LoraConfig, PeftConfig, PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor
from trl import GRPOConfig, GRPOTrainer

from data.prompts import DEEP_RESEARCH_SYSTEM_PROMPT, MAX_AGENT_TURNS

try:
    from .environment import SearchEnvironment
    from .rewards import trajectory_reward, validate_judge_config
    from .tool_protocol import configure_tool_response_parser
except ImportError:  # Direct execution: python rl/train.py
    from environment import SearchEnvironment
    from rewards import trajectory_reward, validate_judge_config
    from tool_protocol import configure_tool_response_parser


@dataclass(frozen=True)
class TrainConfig:
    model_name_or_path: str
    train_file: str
    output_dir: str
    eval_file: str | None = None
    max_steps: int = 500
    num_train_epochs: float = 1.0
    learning_rate: float = 1e-6
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    expected_world_size: int | None = None
    expected_global_batch_size: int | None = None
    num_generations: int = 8
    num_generations_eval: int = 1
    steps_per_generation: int | None = None
    max_prompt_length: int = 4096
    max_context_length: int = 64000
    max_completion_length: int = 70000
    max_new_tokens_per_turn: int = 4096
    max_tool_calling_iterations: int = MAX_AGENT_TURNS
    temperature: float = 0.7
    top_p: float = 1.0
    eval_top_p: float = 0.95
    top_k: int = 0
    beta: float = 0.0
    epsilon_high: float = 0.28
    loss_type: str = "grpo"
    scale_rewards: str = "none"
    mask_truncated_completions: bool = True
    gradient_checkpointing: bool = True
    bf16: bool = True
    tf32: bool = True
    ddp_timeout_seconds: int = 21600
    logging_steps: int = 1
    log_completions: bool = False
    num_completions_to_print: int | None = None
    save_steps: int = 100
    eval_steps: int | None = 100
    save_total_limit: int | None = None
    seed: int = 3407
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str | list[str] = "all-linear"
    lora_exclude_modules: str | list[str] | None = None
    use_vllm: bool = True
    vllm_mode: str = "colocate"
    vllm_gpu_memory_utilization: float = 0.35
    vllm_max_model_length: int = 74576
    vllm_tensor_parallel_size: int = 1
    deepspeed: str | None = None
    report_to: str = "none"


def load_config(path: str) -> TrainConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("GRPO config must be a YAML object")
    config = TrainConfig(**payload)
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    if config.expected_world_size is not None and world_size != config.expected_world_size:
        raise ValueError(
            f"this profile requires {config.expected_world_size} processes, got WORLD_SIZE={world_size}"
        )
    effective_batch = world_size * config.per_device_train_batch_size * config.gradient_accumulation_steps
    if config.expected_global_batch_size is not None and effective_batch != config.expected_global_batch_size:
        raise ValueError(
            f"global batch must be {config.expected_global_batch_size}, got {effective_batch}"
        )
    generation_batch = world_size * config.per_device_train_batch_size * (
        config.steps_per_generation or config.gradient_accumulation_steps
    )
    if effective_batch % config.num_generations:
        raise ValueError("effective batch must be divisible by num_generations")
    if generation_batch % config.num_generations:
        raise ValueError("generation batch must be divisible by num_generations")
    if config.max_tool_calling_iterations != MAX_AGENT_TURNS:
        raise ValueError(
            f"max_tool_calling_iterations must match the shared Agent contract ({MAX_AGENT_TURNS})"
        )
    if config.max_new_tokens_per_turn != 4096:
        raise ValueError("max_new_tokens_per_turn must remain 4096 for method parity")
    if config.max_completion_length != 70000:
        raise ValueError("max_completion_length must remain 70000 for method parity")
    if config.max_context_length != 64000:
        raise ValueError("max_context_length must remain 64000 for method parity")
    if config.max_prompt_length != 4096:
        raise ValueError("max_prompt_length must remain 4096 for method parity")
    if config.loss_type != "grpo" or config.scale_rewards != "none":
        raise ValueError("TRL must use sequence-normalized loss and unscaled group advantages")
    if config.vllm_max_model_length < config.max_completion_length + config.max_prompt_length:
        raise ValueError("vllm_max_model_length must cover prompt plus trajectory completion budget")
    if config.num_generations_eval != 1 or config.eval_top_p != 0.95:
        raise ValueError("evaluation must use one completion per prompt with top_p=0.95")
    print(f"GRPO effective batch: {world_size} x {config.per_device_train_batch_size} x {config.gradient_accumulation_steps} = {effective_batch}")
    print(f"GRPO generation batch: {generation_batch}")
    return config


def _normalize_prompt(messages):
    prompt = []
    for source_message in messages:
        message = dict(source_message)
        content = message.get("content", "")
        if isinstance(content, str):
            message["content"] = [{"type": "text", "text": content}]
        prompt.append(message)
    if not prompt or prompt[0].get("role") != "system":
        prompt.insert(
            0,
            {
                "role": "system",
                "content": [{"type": "text", "text": DEEP_RESEARCH_SYSTEM_PROMPT}],
            },
        )
    return prompt


def load_split(path: str):
    dataset = load_dataset("json", data_files=path, split="train")
    required = {"prompt", "answer"}
    missing = required.difference(dataset.column_names)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if "images" in dataset.column_names:
        dataset = dataset.cast_column("images", Sequence(Image(decode=True)))
    def ensure_system(example):
        return {"prompt": _normalize_prompt(example["prompt"])}
    dataset = dataset.map(ensure_system)
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


class TrajectoryGRPOTrainer(GRPOTrainer):
    """Keep a 70k trajectory budget while limiting each model turn to 4096 tokens."""

    def __init__(
        self,
        *args,
        max_new_tokens_per_turn: int,
        eval_top_p: float,
        generation_only_eval: bool = False,
        **kwargs,
    ):
        self.max_new_tokens_per_turn = max_new_tokens_per_turn
        self.eval_top_p = eval_top_p
        self.generation_only_eval = generation_only_eval
        super().__init__(*args, **kwargs)

    def _generate_single_turn(self, *args, **kwargs):
        top_p = self.top_p if self.model.training else self.eval_top_p
        if self.use_vllm:
            self.vllm_generation.max_completion_length = self.max_new_tokens_per_turn
            self.vllm_generation.top_p = top_p
        else:
            self.generation_config.max_new_tokens = self.max_new_tokens_per_turn
            self.generation_config.top_p = top_p
            self.generation_kwargs["max_new_tokens"] = self.max_new_tokens_per_turn
            self.generation_kwargs["top_p"] = top_p
        return super()._generate_single_turn(*args, **kwargs)

    def _tool_call_loop(self, *args, **kwargs):
        # The initial rollout expands each prompt into `num_generations`
        # candidates. After a tool result, every active candidate is already a
        # distinct trajectory and must receive exactly one continuation.
        num_generations = self.num_generations
        self.num_generations = 1
        try:
            return super()._tool_call_loop(*args, **kwargs)
        finally:
            self.num_generations = num_generations

    def _generate_and_score_completions(self, inputs):
        importance_sampling = self.vllm_importance_sampling_correction
        if self.generation_only_eval and not self.model.training:
            self.vllm_importance_sampling_correction = False
        try:
            output = super()._generate_and_score_completions(inputs)
        finally:
            self.vllm_importance_sampling_correction = importance_sampling
        # With scale_rewards=none, TRL computes r_i - group_mean. Multiplying
        # by G/(G-1) converts it to the leave-one-out baseline used by RLOO.
        group_size = self.num_generations if self.model.training else self.num_generations_eval
        if group_size > 1:
            output["advantages"] = output["advantages"] * (group_size / (group_size - 1))
        if self.model.training and self.environments is not None:
            fatal_mask = output["advantages"].new_tensor(
                [bool(getattr(environment, "_sightline_fatal", False)) for environment in self.environments],
                dtype=torch.bool,
            )
            if fatal_mask.numel() != output["advantages"].numel():
                raise RuntimeError("fatal mask and local advantage batch are misaligned")
            output["advantages"] = torch.where(
                fatal_mask,
                output["advantages"].clamp(min=0),
                output["advantages"],
            )
        if self.model.training:
            # TRL's `grpo` loss averages tokens within each sequence. Scaling
            # by the learnable-token count makes it sequence-mean/token-sum;
            # tool-result tokens remain excluded by TRL's tool mask.
            token_mask = output["completion_mask"]
            if "tool_mask" in output:
                token_mask = token_mask * output["tool_mask"]
            output["advantages"] = output["advantages"] * token_mask.sum(-1).clamp(min=1)
        return output

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if not self.generation_only_eval:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        prepared = self._prepare_inputs(inputs)
        return prepared["advantages"].new_zeros(()), None, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a multimodal tool-use policy with GRPO.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default=None, help="Override model_name_or_path from YAML.")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Generate and score evaluation episodes without running an optimizer step.",
    )
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=None,
        help="Evaluate only the first N examples from the selected evaluation split.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    validate_judge_config()
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
        num_generations_eval=cfg.num_generations_eval,
        steps_per_generation=cfg.steps_per_generation,
        max_completion_length=cfg.max_completion_length,
        # TRL counts only post-initial tool-loop iterations; a 50-call budget
        # therefore maps to 49 additional iterations after the first call.
        max_tool_calling_iterations=cfg.max_tool_calling_iterations - 1,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        beta=cfg.beta,
        epsilon_high=cfg.epsilon_high,
        loss_type=cfg.loss_type,
        scale_rewards=cfg.scale_rewards,
        mask_truncated_completions=cfg.mask_truncated_completions,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=cfg.bf16,
        tf32=cfg.tf32,
        ddp_timeout=cfg.ddp_timeout_seconds,
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
        vllm_tensor_parallel_size=cfg.vllm_tensor_parallel_size,
        model_init_kwargs={"trust_remote_code": True, "dtype": "bfloat16"} if isinstance(policy, str) else None,
        deepspeed=cfg.deepspeed,
    )
    trainer = TrajectoryGRPOTrainer(
        model=policy,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        peft_config=peft_config,
        reward_funcs=[trajectory_reward],
        environment_factory=SearchEnvironment,
        max_new_tokens_per_turn=cfg.max_new_tokens_per_turn,
        eval_top_p=cfg.eval_top_p,
        generation_only_eval=args.eval_only,
    )
    if args.eval_only:
        evaluation_dataset = eval_dataset if eval_dataset is not None else train_dataset
        if args.eval_limit is not None:
            if args.eval_limit <= 0:
                raise ValueError("--eval-limit must be a positive integer")
            evaluation_dataset = evaluation_dataset.select(
                range(min(args.eval_limit, len(evaluation_dataset)))
            )
        metrics = trainer.evaluate(eval_dataset=evaluation_dataset, metric_key_prefix="eval")
        trainer.save_metrics("eval", metrics)
        print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))
        return
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(cfg.output_dir)
    processor.save_pretrained(cfg.output_dir)


if __name__ == "__main__":
    main()
