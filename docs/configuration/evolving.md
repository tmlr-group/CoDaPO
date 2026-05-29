---
id: evolving-config
title: Evolving Config
description: Configuration reference for the Evolving Pipeline — dataset, environment, model endpoints, concurrency, memory, and K-branch multi-model setup.
sidebar_label: "Evolving Config"
sidebar_position: 4
---

# Evolving Config

This page documents configuration for the **Evolving Pipeline**. The Evolving Pipeline is AlphaApollo's inference-time self-improvement framework. It runs a policy–verifier loop across multiple rounds to iteratively refine solutions — without updating model weights.

:::info
Unlike RL training configs (which use Hydra via `verl.trainer.main_ppo`), the evolving pipeline uses standalone YAML configs loaded by `examples/evolving/evolving_main.py` or `examples/evolving/evolving_multi_models.py`.
:::

## Overview

The evolving config controls:

1. **Dataset loading** — Which problems to evolve on
2. **Environment settings** — How policy and verifier environments behave
3. **Model endpoints** — vLLM server configurations for policy and verifier
4. **Evolving loop** — Number of rounds, parallelism, memory depth

## Single-Model Config

File: `configs/vllm_informal_math.yaml`

```yaml
run:
  tag: default
  dataset_name: aime24
  file_name: test.parquet
  data_root: ./data/math-ai/
  policy_env_num: 1
  verifier_env_num: 5
  test_times: 1

env:
  name: informal_math_evolving
  group_n: 1
  config:
    informal_math_evolving:
      evolving_round: 10
      log_requests: false
      python_code_timeout: 300
      enable_verify: true
      concurrency:
        verifier_max_workers: 5
        problem_max_workers: 30
      nd_memory:
        scored_history_length: 3
      policy_env:
        max_steps: 4
        history_length: 4
        memory_type: simple
        enable_python_code: true
        enable_local_rag: true
      verifier_env:
        max_steps: 4
        history_length: 4
        memory_type: simple
        enable_python_code: true
        enable_local_rag: true

vllm_config: &vllm_config
  model_name: "qwen3_4b_inst"
  base_url: "http://localhost:8000/v1"
  api_key: "EMPTY"

policy_model_cfg:
  <<: *vllm_config
  temperature: 0.7
  max_tokens: 8192
  system_prompt: ""

verifier_cfg:
  <<: *vllm_config
  temperature: 0.4
  max_tokens: 4096
  system_prompt: ""
```

## Run Section

| Parameter        | Type   | Default          | Description                                                                                          |
| ---------------- | ------ | ---------------- | ---------------------------------------------------------------------------------------------------- |
| `tag`            | str    | `default`        | Experiment tag for output directory naming.                                                          |
| `dataset_name`   | str    | `aime24`         | Sub-folder inside `data_root` containing the dataset.                                                |
| `file_name`      | str    | `test.parquet`   | Dataset file to evaluate.                                                                            |
| `data_root`      | str    | `./data/math-ai/`| Base directory for datasets.                                                                         |
| `policy_env_num` | int    | `1`              | Number of parallel policy environments.                                                              |
| `verifier_env_num`| int   | `5`              | Number of parallel verifier environments. Use **odd numbers** to ensure majority voting works correctly. |
| `test_times`     | int    | `1`              | Number of passes over the dataset.                                                                   |

:::tip
Use an **odd number** for `verifier_env_num` (e.g., 3, 5, 7) to ensure majority voting always produces a definitive result.
:::

## Environment Configuration

### Top-Level Environment

| Parameter | Type | Description                                                   |
| --------- | ---- | ------------------------------------------------------------- |
| `name`    | str  | Environment identifier. Use `informal_math_evolving` for math tasks. |
| `group_n` | int  | Batch size for grouped environments.                          |

### Evolving-Specific Settings

```yaml
env.config.informal_math_evolving:
  evolving_round: 10
  log_requests: false
  python_code_timeout: 300
  enable_verify: true
```

| Parameter             | Type  | Default | Description                                                                                                   |
| --------------------- | ----- | ------- | ------------------------------------------------------------------------------------------------------------- |
| `evolving_round`      | int   | `10`    | Number of self-evolution iterations per problem. Each round the policy generates a new solution, informed by previous attempts. |
| `log_requests`        | bool  | `false` | Log all environment/model interactions for debugging.                                                         |
| `python_code_timeout` | int   | `300`   | Timeout (seconds) for Python code execution tool. Evolving allows longer timeouts than training.              |
| `enable_verify`       | bool  | `true`  | Enable verifier orchestration. When `true`, a separate verifier agent evaluates policy solutions.             |

### Concurrency Settings

```yaml
concurrency:
  verifier_max_workers: 5
  problem_max_workers: 30
```

| Parameter              | Type | Default | Description                                                                              |
| ---------------------- | ---- | ------- | ---------------------------------------------------------------------------------------- |
| `verifier_max_workers` | int  | `5`     | Max parallel workers for verifier actions. `0` = sequential execution.                   |
| `problem_max_workers`  | int  | `30`    | Max parallel workers for problem-level execution. Controls how many problems are processed simultaneously. |

### N-Dimensional Memory

```yaml
nd_memory:
  scored_history_length: 3
```

| Parameter              | Type | Description                                                                                              |
| ---------------------- | ---- | -------------------------------------------------------------------------------------------------------- |
| `scored_history_length`| int  | Number of scored past solutions retained in memory. The memory maintains the top-scoring solutions from previous evolving rounds. |

### Policy Environment

```yaml
policy_env:
  max_steps: 4
  history_length: 4
  memory_type: simple
  enable_python_code: true
  enable_local_rag: true
```

| Parameter            | Type  | Default  | Description                                                                                                  |
| -------------------- | ----- | -------- | ------------------------------------------------------------------------------------------------------------ |
| `max_steps`          | int   | `4`      | Maximum interaction steps per evolving round for the policy agent.                                           |
| `history_length`     | int   | `4`      | Number of past turns included in the observation.                                                            |
| `memory_type`        | str   | `simple` | Memory implementation: `simple` (recent history), `score` (score-ranked), `ndimensional` (multi-dimensional scoring). |
| `enable_python_code` | bool  | `true`   | Allow the policy to execute Python code via the `<python_code>` tool.                                        |
| `enable_local_rag`   | bool  | `true`   | Allow the policy to retrieve relevant information via local RAG.                                             |

### Verifier Environment

Same parameters as `policy_env`, configured independently:

```yaml
verifier_env:
  max_steps: 4
  history_length: 4
  memory_type: simple
  enable_python_code: true
  enable_local_rag: true
```

The verifier inspects `<answer>` tags and returns structured `<report>` feedback that is injected back into the policy's memory.

## Model Configuration

Models are configured as vLLM-compatible OpenAI API endpoints using YAML anchors for reuse:

```yaml
vllm_config: &vllm_config
  model_name: "qwen3_4b_inst"
  base_url: "http://localhost:8000/v1"
  api_key: "EMPTY"
```

### Policy Model

```yaml
policy_model_cfg:
  <<: *vllm_config
  temperature: 0.7
  max_tokens: 8192
  system_prompt: ""
```

| Parameter    | Type  | Description                                                             |
| ------------ | ----- | ----------------------------------------------------------------------- |
| `model_name` | str   | Model identifier (matches vLLM server `--served-model-name`).           |
| `base_url`   | str   | vLLM server endpoint.                                                   |
| `temperature`| float | Sampling temperature. Policy typically uses higher temperature (0.7) for diversity. |
| `max_tokens` | int   | Maximum tokens per generation.                                          |

### Verifier Model

```yaml
verifier_cfg:
  <<: *vllm_config
  temperature: 0.4
  max_tokens: 4096
  system_prompt: ""
```

:::note
The verifier uses a **lower temperature** (0.4) for more deterministic evaluation compared to the policy (0.7).
:::

## Multi-Model Config (K-Branch)

File: `configs/vllm_informal_math_multi_models.yaml`

The multi-model config extends the single-model setup with K-branch parallel evolution:

```yaml
# Model definitions (YAML anchors)
vllm_qwen3: &vllm_qwen3
  model_name: "qwen3_4b_inst"
  base_url: "http://localhost:8000/v1"
  api_key: "EMPTY"

vllm_qwen25: &vllm_qwen25
  model_name: "qwen2_5_14b_inst"
  base_url: "http://localhost:8001/v1"
  api_key: "EMPTY"

# Shared verifier (fallback for branches without their own)
default_verifier_cfg:
  <<: *vllm_qwen3
  temperature: 0.4
  max_tokens: 8192

# K-Branch configuration
branches:
  - branch_id: "qwen25_branch_temp_0.7"
    policy_model_cfg:
      <<: *vllm_qwen25
      temperature: 0.7
      max_tokens: 8192
    verifier_cfg:
      <<: *vllm_qwen25
      temperature: 0.4
      max_tokens: 8192

  - branch_id: "qwen3_branch_temp_0.7"
    policy_model_cfg:
      <<: *vllm_qwen3
      temperature: 0.7
      max_tokens: 8192
    verifier_cfg:
      <<: *vllm_qwen3
      temperature: 0.4
      max_tokens: 8192
```

### Multi-Model Specific Settings

| Parameter                       | Type  | Description                                                                               |
| ------------------------------- | ----- | ----------------------------------------------------------------------------------------- |
| `branches`                      | list  | List of branch configurations. Each branch runs independently with its own policy and verifier. |
| `branches[].branch_id`          | str   | Unique identifier for the branch (used in logging and output).                            |
| `branches[].policy_model_cfg`   | dict  | Policy model configuration for this branch.                                               |
| `branches[].verifier_cfg`       | dict  | Verifier model configuration (optional, falls back to `default_verifier_cfg`).            |
| `default_verifier_cfg`          | dict  | Shared verifier used when a branch doesn't specify its own.                               |
| `concurrency.branch_max_workers`| int   | Max parallel workers for K-branch execution.                                              |

All branches share a **thread-safe solution memory** (`ThreadSafeSolutionMemory`) for real-time cross-pollination between branches.

## Hosting vLLM Servers

Before running the evolving pipeline, you need to host the model(s) via vLLM:

```bash
# Single model
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --served-model-name qwen25_7b_inst \
    --port 8000 \
    --tensor-parallel-size 1

# Multiple models (on different ports)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Instruct \
    --served-model-name qwen3_4b_inst \
    --port 8000 &

python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-14B-Instruct \
    --served-model-name qwen2_5_14b_inst \
    --port 8001 &
```

## Running the Evolving Pipeline

### Single-Model Evolving

```bash
python examples/evolving/evolving_main.py \
    --config configs/vllm_informal_math.yaml
```

### Multi-Model K-Branch Evolving

```bash
python examples/evolving/evolving_multi_models.py \
    --config configs/vllm_informal_math_multi_models.yaml
```

## Output Structure

The evolving pipeline produces rich outputs under:

```
outputs/<dataset>/<tag>/<model>/test_*/
├── step_outputs/      # Per-step trajectory data
├── problem_results/   # Per-problem final results
└── metrics.json       # Aggregate success rates and verifier accuracy
```

## Related Pages

- [Configuration Overview](./index.md) — Hydra basics and CLI override syntax
- [RL Training Config](./rl_config.md) — Full RL training configuration reference
- [Generation Config](./generation.md) — Offline generation configuration
- [Evolving Pipeline](../algorithms/evolving-pipeline.md) — How the evolving algorithm works
orithm works
