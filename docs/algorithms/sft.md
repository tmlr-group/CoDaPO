---
id: sft
title: Supervised Fine-Tuning (SFT)
description: SFT pipeline in AlphaApollo — config reference, LoRA support, multi-turn data format, and the SFT-to-RL handoff.
sidebar_label: "Supervised Fine-Tuning"
sidebar_position: 2
---

# Supervised Fine-Tuning (SFT)

Supervised Fine-Tuning (SFT) is typically the first stage of LLM post-training. The model learns from curated instruction-response pairs using standard cross-entropy loss. AlphaApollo's SFT trainer is built on PyTorch FSDP and supports sequence parallelism, LoRA, and multi-turn conversation data.

## Overview

The SFT pipeline:

1. Loads a pretrained model and instruction-response dataset
2. Fine-tunes the model using standard cross-entropy loss
3. Supports both single-turn and multi-turn conversation formats
4. Outputs a checkpoint compatible with the RL training pipeline

Entry point:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=<N_GPUS> \
    -m verl.trainer.fsdp_sft_trainer \
    model.partial_pretrain=<MODEL_PATH> \
    data.train_files=<TRAIN_DATA> \
    ...
```

## Config Reference

The SFT config is defined in `verl/trainer/config/sft_trainer.yaml`.

### Data

```yaml
data:
  train_batch_size: 256
  micro_batch_size_per_gpu: 4
  train_files: ~/data/gsm8k/train.parquet
  val_files: ~/data/gsm8k/test.parquet
  # Single-turn settings
  prompt_key: question
  response_key: answer
  prompt_dict_keys: ['question']
  response_dict_keys: ['answer']
  # Multi-turn settings
  multiturn:
    enable: false
    messages_key: messages
  max_length: 1024
  truncation: error
  balance_dp_token: False
  chat_template: null
  custom_cls:
    path: null
    name: null
```

| Parameter               | Type   | Default        | Description                                                                      |
| ----------------------- | ------ | -------------- | -------------------------------------------------------------------------------- |
| `train_batch_size`      | int    | `256`          | Global training batch size.                                                      |
| `micro_batch_size_per_gpu` | int | `4`           | Per-GPU batch size for forward/backward pass (gradient accumulation).            |
| `train_files`           | str    | —              | Path to training data (parquet format).                                          |
| `val_files`             | str    | —              | Path to validation data (parquet format).                                        |
| `prompt_key`            | str    | `question`     | Column name for prompts.                                                         |
| `response_key`          | str    | `answer`       | Column name for responses.                                                       |
| `prompt_dict_keys`      | list   | `['question']` | Keys to extract from prompt dict if the column contains dicts.                   |
| `response_dict_keys`    | list   | `['answer']`   | Keys to extract from response dict.                                              |
| `max_length`            | int    | `1024`         | Maximum sequence length (prompt + response).                                     |
| `truncation`            | str    | `error`        | Truncation strategy: `error`, `left`, `right`.                                   |
| `balance_dp_token`      | bool   | `False`        | Balance tokens across data-parallel ranks.                                       |
| `chat_template`         | str    | `null`         | Custom chat template. `null` uses the model's default.                           |
| `multiturn.enable`      | bool   | `false`        | Enable multi-turn conversation format.                                           |
| `multiturn.messages_key`| str    | `messages`     | Column name for multi-turn messages list.                                        |

### Model

```yaml
model:
  partial_pretrain: ~/models/gemma-1.1-7b-it
  strategy: fsdp2
  fsdp_config:
    wrap_policy:
      min_num_params: 0
    cpu_offload: False
    offload_params: False
  external_lib: null
  enable_gradient_checkpointing: False
  trust_remote_code: False
  lora_rank: 0
  lora_alpha: 16
  target_modules: all-linear
  use_liger: False
```

| Parameter                        | Type       | Default       | Description                                                   |
| -------------------------------- | ---------- | ------------- | ------------------------------------------------------------- |
| `partial_pretrain`               | str        | —             | Path to pretrained model (HuggingFace format).                |
| `strategy`                       | str        | `fsdp2`       | FSDP strategy: `fsdp` or `fsdp2`.                            |
| `fsdp_config.cpu_offload`        | bool       | `False`       | Enable CPU offloading for FSDP.                               |
| `enable_gradient_checkpointing`  | bool       | `False`       | Enable gradient checkpointing to reduce memory.               |
| `trust_remote_code`              | bool       | `False`       | Allow loading remote code models.                             |
| `lora_rank`                      | int        | `0`           | LoRA rank. Set > 0 to enable LoRA fine-tuning.                |
| `lora_alpha`                     | int        | `16`          | LoRA scaling factor.                                          |
| `target_modules`                 | str / list | `all-linear`  | LoRA target modules.                                          |
| `use_liger`                      | bool       | `False`       | Use Liger kernel for memory-efficient computation.            |

### Optimizer

```yaml
optim:
  lr: 1e-5
  betas: [0.9, 0.95]
  weight_decay: 0.01
  warmup_steps_ratio: 0.1
  clip_grad: 1.0
  lr_scheduler: cosine     # cosine or wsd
```

| Parameter           | Type  | Default       | Description                                                                            |
| ------------------- | ----- | ------------- | -------------------------------------------------------------------------------------- |
| `lr`                | float | `1e-5`        | Learning rate.                                                                         |
| `betas`             | list  | `[0.9, 0.95]` | Adam optimizer beta parameters.                                                        |
| `weight_decay`      | float | `0.01`        | Weight decay coefficient.                                                              |
| `warmup_steps_ratio`| float | `0.1`         | Fraction of total steps used for learning rate warmup.                                 |
| `clip_grad`         | float | `1.0`         | Gradient clipping norm.                                                                |
| `lr_scheduler`      | str   | `cosine`      | LR scheduler: `cosine` (cosine annealing) or `wsd` (warmup-stable-decay).             |

### Trainer

```yaml
trainer:
  default_local_dir: /tmp/sft_model
  project_name: gsm8k-sft
  experiment_name: test
  total_epochs: 4
  total_training_steps: null
  logger: ['console']
  seed: 1
```

| Parameter              | Type | Default | Description                                                     |
| ---------------------- | ---- | ------- | --------------------------------------------------------------- |
| `default_local_dir`    | str  | —       | Directory to save checkpoints.                                  |
| `project_name`         | str  | —       | Project name for logging.                                       |
| `total_epochs`         | int  | `4`     | Number of training epochs.                                      |
| `total_training_steps` | int  | `null`  | Alternative: stop after N steps (overrides `total_epochs`).     |
| `logger`               | list | `['console']` | Logging backends.                                         |
| `seed`                 | int  | `1`     | Random seed for reproducibility.                                |

### Sequence Parallelism

```yaml
ulysses_sequence_parallel_size: 1
use_remove_padding: False
```

- `ulysses_sequence_parallel_size`: Degree of Ulysses sequence parallelism. Set to `2` or more to split long sequences across GPUs.
- `use_remove_padding`: Remove padding tokens before computation for better efficiency.

## Examples

### Basic SFT on GSM8K

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.micro_batch_size=4 \
    optim.lr=1e-4 \
    model.partial_pretrain=Qwen/Qwen2.5-0.5B-Instruct \
    trainer.default_local_dir=/tmp/sft_output \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=qwen-0.5b-sft \
    trainer.logger=['console'] \
    trainer.total_epochs=4 \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true
```

### SFT with LoRA

For parameter-efficient fine-tuning:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    -m verl.trainer.fsdp_sft_trainer \
    model.partial_pretrain=Qwen/Qwen2.5-7B-Instruct \
    model.lora_rank=32 \
    model.lora_alpha=16 \
    model.target_modules=all-linear \
    optim.lr=1e-4 \
    data.train_files=$HOME/data/train.parquet \
    data.val_files=$HOME/data/test.parquet \
    trainer.total_epochs=3
```

### Multi-Turn SFT

For training on conversation data:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/data/multiturn/train.parquet \
    data.val_files=$HOME/data/multiturn/test.parquet \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.max_length=2048 \
    model.partial_pretrain=Qwen/Qwen2.5-0.5B-Instruct \
    trainer.total_epochs=4
```

## Data Format

### Single-Turn Data

The training data should be in parquet format with prompt and response columns:

| prompt (or question) | response (or answer) |
| -------------------- | -------------------- |
| "What is 2 + 3?"     | "2 + 3 = 5"          |
| "Solve: 4x = 12"     | "x = 3"              |

When using `prompt_dict_keys` and `response_dict_keys`, the column can contain JSON dicts:

```json
{
  "extra_info": {
    "question": "What is 2 + 3?",
    "answer": "2 + 3 = 5"
  }
}
```

### Multi-Turn Data

For multi-turn format, provide a `messages` column with a list of role-content pairs:

```json
{
  "messages": [
    {"role": "user", "content": "What is 2+3?"},
    {"role": "assistant", "content": "2+3 = 5"},
    {"role": "user", "content": "And 5+7?"},
    {"role": "assistant", "content": "5+7 = 12"}
  ]
}
```

## Data Preparation

Use the provided preprocessing scripts to prepare datasets:

```bash
# GSM8K
python3 -m examples.data_preprocess.gsm8k

# Informal Math
python3 -m examples.data_preprocess.prepare_informal_math \
    --data_source DigitalLearningGmbH/MATH-lighteval

# Multi-turn data
python3 -m examples.data_preprocess.multiturn
```

## SFT → RL Pipeline

The SFT checkpoint can be directly used as the starting point for RL training:

```bash
# Step 1: SFT
torchrun ... -m verl.trainer.fsdp_sft_trainer \
    model.partial_pretrain=Qwen/Qwen2.5-1.5B-Instruct \
    trainer.default_local_dir=/tmp/sft_ckpt \
    ...

# Step 2: RL Training (using SFT checkpoint)
python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.model.path=/tmp/sft_ckpt \
    algorithm.adv_estimator=grpo \
    ...
```

:::info
The SFT checkpoint directory can be passed directly to `actor_rollout_ref.model.path` — no format conversion needed.
:::

## Related Pages

- [RL Training](./rl-training.md) — RL algorithms for post-training
- [Evolving Pipeline](./evolving-pipeline.md) — Inference-time self-improvement via policy-verifier loops
- [RL Training Config](../configuration/rl_config.md) — Detailed RL parameter reference
- [Configuration Overview](../configuration/index.md) — Hydra basics and CLI overrides
