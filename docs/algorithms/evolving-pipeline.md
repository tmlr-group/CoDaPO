---
id: evolving-pipeline
title: Evolving Pipeline
description: AlphaApollo's Evolving Pipeline — policy-verifier loops, solution memory, and single- or multi-model K-branch setups for inference-time improvement.
sidebar_label: "Evolving Pipeline"
sidebar_position: 3
---

# Evolving Pipeline

The **Evolving Pipeline** is AlphaApollo's inference-time self-improvement framework. It runs a **policy–verifier loop** across multiple rounds: the Policy Agent generates a candidate solution, the Verifier Agent evaluates it, and the result is stored in **Solution Memory** for the next round. No model weights are updated — improvement comes entirely from structured feedback and accumulated solution history.

:::note Terminology
This feature is called the **Evolving Pipeline** throughout AlphaApollo's documentation. External papers and scripts may also use "self-improvement loop", "self-evolving", or "evo" — all refer to the same system.
:::

## Overview

The evolving pipeline implements a multi-round self-improvement loop:

```
┌─────────────────────────────────────────────────────────────┐
│                    Evolving Loop                             │
│                                                             │
│  For each problem × evolving_round:                         │
│                                                             │
│  ┌──────────┐    ┌───────────────┐    ┌──────────────────┐  │
│  │  Policy   │───▶│  Environment  │───▶│    Verifier      │  │
│  │  Agent    │    │  (tool use,   │    │  (evaluate &     │  │
│  │          │◀───│   code exec)  │◀───│   report)        │  │
│  └──────────┘    └───────────────┘    └──────────────────┘  │
│       │                                        │            │
│       ▼                                        ▼            │
│  ┌────────────────────────────────────────────────────┐     │
│  │              Solution Memory                        │     │
│  │  (retains top solutions across rounds)              │     │
│  └────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## Key Concepts

### Policy Agent

The policy agent generates solutions to problems. It can use tools (Python code execution, RAG retrieval) and iterates across multiple steps within each round.

### Verifier Agent

The verifier agent evaluates policy-generated solutions. It:
- Inspects `<answer>` tags in the policy's output
- Can execute Python code to check mathematical correctness
- Returns a structured `<report>` with pass/fail judgment
- Multiple verifier instances run in parallel for **majority voting**

### Solution Memory

The solution memory retains top-scoring solutions across evolving rounds, enabling the policy to learn from its best past attempts:

| Memory Type     | Description                                                                |
| --------------- | -------------------------------------------------------------------------- |
| `simple`        | Retains recent history (FIFO)                                              |
| `score`         | Retains highest-scoring solutions                                          |
| `ndimensional`  | Multi-dimensional scoring with `scored_history_length` control             |

### Execution Modes

| Mode           | Description                                                                                                                                   |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **Agentic**    | Interleaves verification when `<answer>` tags appear. Force-verifies on the final step of each round. Advances to the next round after verification. |
| **Mechanical** | Runs a fixed number of steps without interleaved verification. Performs a single verifier session at the end of each round.                    |

## Pipeline Steps

### 1. Data Loading

Problems are loaded from processed parquet files via `examples/evolving/utils/dataset_loader.py`:

```python
# Load problems from parquet
problems = load_dataset(
    data_root="./data/math-ai/",
    dataset_name="aime24",
    file_name="test.parquet"
)
```

### 2. Environment Setup

The pipeline creates separate environments for policy and verifier roles:

```python
# Policy environment: generates solutions
policy_env = InformalMathEnvironmentManager(
    max_steps=4,
    history_length=4,
    memory_type="simple",
    enable_python_code=True,
    enable_local_rag=True
)

# Verifier environment: evaluates solutions
verifier_env = InformalMathEnvironmentManager(
    max_steps=4,
    history_length=4,
    memory_type="simple",
    enable_python_code=True
)
```

### 3. Evolving Loop

For each problem and each evolving round:

```
Round 1:  Policy generates solution → Verifier evaluates → Update memory
Round 2:  Policy generates (with memory) → Verifier evaluates → Update memory
  ...
Round N:  Policy generates (with rich memory) → Verifier evaluates → Final result
```

Each round benefits from the accumulated solution memory, allowing the policy to refine its approach based on previous attempts and verifier feedback.

### 4. Verification

The verifier runs multiple instances in parallel and uses majority voting:

```yaml
verifier_env_num: 5  # Odd number for majority voting
concurrency:
  verifier_max_workers: 5
```

:::tip
Use an **odd number** for `verifier_env_num` (e.g., 3, 5, 7) to ensure majority voting always produces a definitive result.
:::

### 5. Output Collection

The pipeline produces detailed outputs:

```
outputs/<dataset>/<tag>/<model>/test_*/
├── step_outputs/        # Per-step trajectory data
├── problem_results/     # Per-problem results with correctness
└── metrics.json         # Aggregate metrics
```

Metrics tracked:
- **Policy accuracy**: Does the policy's answer match ground truth?
- **Verifier accuracy**: Does the verifier's judgment agree with ground truth?
- **Success rate**: Fraction of problems solved across evolving rounds

## Single-Model Evolving

The standard single-model setup uses one policy model and one verifier model (can be the same model):

```bash
# 1. Host the model
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --served-model-name qwen25_7b_inst \
    --port 8000

# 2. Run evolving
python examples/evolving/evolving_main.py \
    --config configs/vllm_informal_math.yaml
```

Config highlights:

```yaml
env.config.informal_math_evolving:
  evolving_round: 10        # 10 rounds of self-improvement
  enable_verify: true        # Enable verifier
  policy_env.max_steps: 4    # 4 steps per round
  policy_env.enable_python_code: true

policy_model_cfg:
  temperature: 0.7           # Higher for diversity
  max_tokens: 8192

verifier_cfg:
  temperature: 0.4           # Lower for precision
  max_tokens: 4096
```

## Multi-Model K-Branch Evolving

K-branch evolving runs multiple policy models in parallel, with a shared solution memory for cross-pollination:

```
┌─────────────────────────────────────────────────┐
│           Shared Solution Memory                 │
│     (ThreadSafeSolutionMemory)                  │
├─────────┬────────────┬────────────┬─────────────┤
│         │            │            │             │
│  Branch 1     Branch 2     Branch 3    ...     │
│  (Qwen-7B)   (Qwen-14B)  (Qwen-3B)           │
│  Policy +     Policy +     Policy +             │
│  Verifier     Verifier     Verifier             │
└─────────┴────────────┴────────────┴─────────────┘
```

**Key benefits:**
- Different models bring different strengths and perspectives
- Cross-pollination via shared memory accelerates convergence
- Branches run in parallel for efficiency

```bash
# 1. Host multiple models
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Instruct \
    --served-model-name qwen3_4b_inst \
    --port 8000 &

python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-14B-Instruct \
    --served-model-name qwen2_5_14b_inst \
    --port 8001 &

# 2. Run multi-model evolving
python examples/evolving/evolving_multi_models.py \
    --config configs/vllm_informal_math_multi_models.yaml
```

Config highlights:

```yaml
concurrency:
  branch_max_workers: 2     # Run 2 branches in parallel

branches:
  - branch_id: "qwen25_branch"
    policy_model_cfg:
      model_name: "qwen2_5_14b_inst"
      base_url: "http://localhost:8001/v1"
      temperature: 0.7
  - branch_id: "qwen3_branch"
    policy_model_cfg:
      model_name: "qwen3_4b_inst"
      base_url: "http://localhost:8000/v1"
      temperature: 0.7
```

## Concurrency and Parallelism

The evolving pipeline supports multiple levels of parallelism:

| Level             | Parameter              | Description                                       |
| ----------------- | ---------------------- | ------------------------------------------------- |
| Problem-level     | `problem_max_workers`  | Process multiple problems simultaneously           |
| Verifier-level    | `verifier_max_workers` | Run multiple verifier instances per problem        |
| Branch-level      | `branch_max_workers`   | Run multiple model branches in parallel            |
| Environment-level | `policy_env_num`       | Multiple policy environments per problem           |

Recommended settings for a single GPU:

```yaml
concurrency:
  verifier_max_workers: 5
  problem_max_workers: 30
  branch_max_workers: 2  # multi-model only
```

## Related Pages

- [Evolving Config](../configuration/evolving.md) — Detailed configuration reference
- [RL Training](./rl-training.md) — RL algorithms for post-training
- [SFT](./sft.md) — Supervised Fine-Tuning process
- [Configuration Overview](../configuration/index.md) — Hydra basics and CLI overrides
