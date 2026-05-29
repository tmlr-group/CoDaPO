---
id: algorithms
title: Algorithms
description: Overview of AlphaApollo's training algorithms — SFT, RL training (PPO/GRPO), and the Evolving Pipeline.
sidebar_position: 1
---

# Algorithms

AlphaApollo supports multiple training and inference paradigms for LLM post-training. This section covers the core algorithms and pipelines available in the framework.

## Training Pipelines

| Pipeline                                          | Description                                                   | Entry Point                              |
| ------------------------------------------------- | ------------------------------------------------------------- | ---------------------------------------- |
| [Supervised Fine-Tuning (SFT)](./sft.md)          | Train on curated instruction-response pairs                   | `verl.trainer.fsdp_sft_trainer`          |
| [RL Training](./rl-training.md)                   | Reinforcement learning with PPO, GRPO, etc.           | `verl.trainer.main_ppo`                  |
| [Evolving Pipeline](./evolving-pipeline.md)       | Inference-time self-improvement via policy-verifier loops     | `examples/evolving/evolving_main.py`     |

## Supported RL Algorithms

| Algorithm | `adv_estimator`         | Critic Required | Group Rollouts   | Key Feature                              |
| --------- | ----------------------- | --------------- | ---------------- | ---------------------------------------- |
| PPO       | `gae`                   | Yes             | No               | Standard RL with value function          |
| GRPO      | `grpo`                  | No              | Yes (`n > 1`)    | Group-relative advantage estimation      |
| DAPO      | `grpo` + filter         | No              | Yes (`n > 1`)    | Group filtering for better learning signal|
| RLOO      | `rloo`                  | No              | Yes (`n > 1`)    | Leave-one-out baseline                   |

## Typical Workflow

A typical AlphaApollo post-training workflow follows these stages:

```
┌─────────┐     ┌──────────────┐     ┌──────────────────┐
│   SFT   │────▶│ RL Training │────▶│     Evolving     │
│         │     │ (GRPO)       │     │  (Self-Improve)  │
└─────────┘     └──────────────┘     └──────────────────┘
  Stage 1           Stage 2              Stage 3
```

1. **SFT** — Fine-tune a pretrained model on task-specific instruction-response data
2. **RL Training** — Further optimize the model using environment rewards (GRPO, PPO)
3. **Evolving** — Iteratively improve solutions at inference time through policy-verifier self-improvement loops

:::tip
Each stage is optional — you can start from any point depending on your needs.
:::

## Example Scripts

AlphaApollo provides ready-to-use scripts for various environments and algorithms:

### RL Training

```bash
# GRPO on MATH-lighteval with Qwen2.5-3B-Instruct and evaluate on MATH-500
cd examples/rl
bash run_rl_informal_math_tool.sh
```

### SFT

```bash
# SFT on NuminaMath-TIR with Qwen2.5-3B-Instruct
bash examples/sft/run_sft_informal_math_tool.sh
```

### Evolving

```bash
# Before running the self-evolution scripts, make sure to serve the corresponding number of models.
python alphaapollo/utils/ray_serve_llm.py --model_path Qwen/Qwen3-4B-Instruct-2507 --gpus "0,1" --port 8000 --model_id "qwen3_4b_inst"

# single-model evolution
python3 -m alphaapollo.workflows.evo \
  --preprocess.data_source=math-ai/aime24 \
  --run.dataset_name=aime24 \
  --policy_model_cfg.model_name=qwen3_4b_inst \
  --policy_model_cfg.base_url=http://localhost:8000/v1 \
  --verifier_cfg.model_name=qwen3_4b_inst \
  --verifier_cfg.base_url=http://localhost:8000/v1
```

## Related Pages

- [RL Training Config](../configuration/rl_config.md) — Detailed RL parameter reference
- [Generation Config](../configuration/generation.md) — Offline generation configuration
- [Evolving Config](../configuration/evolving.md) — Evolving pipeline configuration
- [Configuration Overview](../configuration/index.md) — Hydra basics and CLI overrides
