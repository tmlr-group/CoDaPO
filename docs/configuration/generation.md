---
id: generation-config
title: Generation Config
description: Parameter reference for generation.yaml — offline inference, data collection, and multi-turn environment interaction with verl.trainer.main_generation.
sidebar_label: "Generation Config"
sidebar_position: 3
---

# Generation Config

This page documents the `generation.yaml` configuration used by `verl.trainer.main_generation`. The generation pipeline runs offline inference on a given dataset. It can produce standalone model responses or interact with environments for multi-turn tasks such as tool-augmented math reasoning.

## Overview

The generation pipeline is used to:

1. **Collect training data** — Generate solutions for problems that can be used for SFT or further analysis
2. **Evaluate models** — Produce responses for benchmarks and compute metrics
3. **Multi-turn inference** — Interact with environments (e.g., informal math with Python code execution)

Launch with:

```bash
python3 -m verl.trainer.main_generation \
    model.path=Qwen/Qwen2.5-1.5B-Instruct \
    data.path=~/data/test.parquet \
    rollout.temperature=0.6 \
    ...
```

## Config Structure

```yaml
trainer:
  nnodes: 1
  n_gpus_per_node: 8

data:
  path: ~/data/rlhf/math/test.parquet
  prompt_key: prompt
  n_samples: 5
  output_path: /opt/tiger/math_output.parquet
  batch_size: 128
  return_raw_chat: True
  max_prompt_length: ${rollout.prompt_length}
  max_response_length: ${rollout.response_length}
  truncation: error
  save2json: False
  json_output_path: ???

model:
  path: ~/models/Qwen2-7B-Instruct
  external_lib: null

rollout:
  name: vllm
  mode: sync
  temperature: 1.0
  top_k: -1
  top_p: 0.7
  prompt_length: 1536
  response_length: 512
  dtype: bfloat16
  gpu_memory_utilization: 0.5
  tensor_model_parallel_size: 1
  max_num_batched_tokens: 8192
  max_num_seqs: 1024
  n: 1
  enforce_eager: True
  free_cache_engine: True
  load_format: dummy_dtensor
  enable_chunked_prefill: True

env:
  env_name: informal_math_training
  seed: 0
  max_steps: 1
  history_length: 2
  ...
```

## Data

| Parameter            | Type  | Default  | Description                                                                                |
| -------------------- | ----- | -------- | ------------------------------------------------------------------------------------------ |
| `path`               | str   | —        | Path to input dataset (parquet format).                                                    |
| `prompt_key`         | str   | `prompt` | Column name for prompts in the dataset.                                                    |
| `n_samples`          | int   | `5`      | Number of responses to generate per prompt.                                                |
| `output_path`        | str   | —        | Path to save output parquet file.                                                          |
| `batch_size`         | int   | `128`    | Batch size for generation. Also sets `train_batch_size` and `val_batch_size`.              |
| `return_raw_chat`    | bool  | `True`   | Return raw chat without applying template. Useful for multi-turn environments.             |
| `max_prompt_length`  | int   | —        | Maximum prompt length (defaults to `rollout.prompt_length`).                               |
| `max_response_length`| int   | —        | Maximum response length (defaults to `rollout.response_length`).                           |
| `truncation`         | str   | `error`  | Truncation strategy: `error`, `left`, `right`, `middle`.                                   |
| `save2json`          | bool  | `False`  | Also save outputs in JSON format (in addition to parquet).                                 |
| `json_output_path`   | str   | —        | Path for JSON output (required when `save2json=True`).                                     |

## Model

| Parameter      | Type  | Description                                                                     |
| -------------- | ----- | ------------------------------------------------------------------------------- |
| `path`         | str   | HuggingFace model path or local checkpoint path.                                |
| `external_lib` | str   | Additional Python packages to import for model registration.                    |

## Rollout

The rollout section controls the inference engine and sampling parameters.

| Parameter                    | Type  | Default    | Description                                                             |
| ---------------------------- | ----- | ---------- | ----------------------------------------------------------------------- |
| `name`                       | str   | `vllm`     | Inference engine: `vllm`, `sglang`, or `hf`.                           |
| `mode`                       | str   | `sync`     | `sync` for standard `LLM`, `async` for `AsyncLLM`.                     |
| `temperature`                | float | `1.0`      | Sampling temperature. Higher = more random.                             |
| `top_k`                      | int   | `-1`       | Top-k sampling. `-1` for vLLM (disabled), `0` for HF (disabled).      |
| `top_p`                      | float | `0.7`      | Nucleus sampling threshold.                                             |
| `prompt_length`              | int   | `1536`     | Maximum prompt token length.                                            |
| `response_length`            | int   | `512`      | Maximum response token length.                                          |
| `dtype`                      | str   | `bfloat16` | Model precision. Should align with training precision.                  |
| `gpu_memory_utilization`     | float | `0.5`      | Fraction of GPU memory for vLLM. Increase for larger models.           |
| `tensor_model_parallel_size` | int   | `1`        | Tensor parallelism degree. Increase for larger models.                  |
| `max_num_batched_tokens`     | int   | `8192`     | Maximum batched tokens for vLLM scheduler.                              |
| `n`                          | int   | `1`        | Number of responses per prompt per batch.                               |
| `enable_chunked_prefill`     | bool  | `True`     | Enable chunked prefill for better throughput.                           |

## Environment

The generation pipeline supports environment interaction for multi-turn tasks.

```yaml
env:
  env_name: informal_math_training
  seed: 0
  max_steps: 8
  history_length: 8
  resources_per_worker:
    num_cpus: 0.1
    num_gpus: 0
  informal_math:
    memory_type: simple
    enable_python_code: true
    enable_local_rag: true
    python_code_timeout: 30
```

| Parameter                        | Type  | Description                                                              |
| -------------------------------- | ----- | ------------------------------------------------------------------------ |
| `env_name`                       | str   | Environment to use. Set to the desired environment name.                 |
| `max_steps`                      | int   | Maximum interaction steps. For single-turn generation, set to `1`.       |
| `history_length`                 | int   | Number of past turns included in observations.                           |
| `informal_math.enable_python_code` | bool | Enable Python code execution tool.                                      |
| `informal_math.enable_local_rag` | bool  | Enable local RAG retrieval tool.                                         |

## Example: Informal Math Generation

```bash
python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=1 \
    data.path=~/data/HuggingFaceH4/MATH-500/test.parquet \
    data.prompt_key=prompt \
    data.n_samples=2 \
    data.batch_size=32 \
    data.return_raw_chat=True \
    data.output_path=~/data/output.parquet \
    data.save2json=true \
    data.json_output_path=~/data/output.json \
    model.path=Qwen/Qwen2.5-1.5B-Instruct \
    rollout.temperature=0.6 \
    rollout.top_k=20 \
    rollout.top_p=0.95 \
    rollout.prompt_length=2048 \
    rollout.response_length=8192 \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.75 \
    rollout.name=vllm \
    env.env_name=informal_math_training \
    env.max_steps=8 \
    env.history_length=8 \
    env.informal_math.enable_python_code=true \
    env.informal_math.enable_local_rag=true
```

## Example: Single-Turn Generation (No Tool)

For generation without environment interaction:

```bash
python3 -m verl.trainer.main_generation \
    trainer.n_gpus_per_node=1 \
    data.path=~/data/test.parquet \
    data.n_samples=5 \
    model.path=Qwen/Qwen2.5-1.5B-Instruct \
    rollout.temperature=0.7 \
    rollout.response_length=4096 \
    env.max_steps=1
```

:::tip Single-turn generation
Set `env.max_steps=1` to disable environment interaction and run single-turn generation only.
:::

## Differences from RL Training Config

| Aspect                  | Generation                     | RL Training                  |
| ----------------------- | ------------------------------ | ---------------------------- |
| Entry point             | `verl.trainer.main_generation` | `verl.trainer.main_ppo`      |
| Actor training          | No                             | Yes                          |
| Critic model            | No                             | Yes (PPO only)               |
| Reward computation      | No                             | Yes                          |
| Environment interaction | Optional                       | Yes                          |
| Output                  | Parquet / JSON file            | Model checkpoints            |

## Related Pages

- [Configuration Overview](./index.md) — Hydra basics and CLI override syntax
- [RL Training Config](./rl_config.md) — Full RL training configuration reference
- [Evolving Config](./evolving.md) — Self-evolving pipeline configuration
