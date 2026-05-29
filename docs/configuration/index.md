---
id: configuration
title: Configuration
description: How AlphaApollo uses Hydra for config management — entry points, CLI overrides, variable interpolation, and environment variables.
sidebar_position: 1
---

# Configuration

AlphaApollo uses [Hydra](https://hydra.cc/) as its configuration management framework. All training, generation, and evaluation entry points are driven by composable YAML config files with full support for CLI overrides.

## Hydra Basics

Each trainer entry point is decorated with `@hydra.main`, which loads a default YAML config and merges any command-line overrides on top.

| Entry Point                    | Config File                               | Description                       |
| ------------------------------ | ----------------------------------------- | --------------------------------- |
| `verl.trainer.main_ppo`        | `verl/trainer/config/ppo_trainer.yaml`    | PPO / GRPO RL training            |
| `verl.trainer.main_generation` | `verl/trainer/config/generation.yaml`     | Offline generation / inference    |
| `verl.trainer.fsdp_sft_trainer`| `verl/trainer/config/sft_trainer.yaml`    | Supervised Fine-Tuning            |
| `verl.trainer.main_eval`       | `verl/trainer/config/evaluation.yaml`     | Evaluation pipeline               |

For example, the PPO trainer is launched as:

```python
# verl/trainer/main_ppo.py
@hydra.main(config_path="config", config_name="ppo_trainer")
def main(config):
    ...
```

## CLI Overrides

Hydra allows you to override any config parameter directly from the command line using dot-separated paths. This is the primary way to customize training runs without editing YAML files.

### Basic Override Syntax

```bash
python3 -m verl.trainer.main_ppo \
    data.train_batch_size=16 \
    data.max_prompt_length=4096 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    algorithm.adv_estimator=grpo
```

### Common Override Patterns

**Nested keys** — use dot notation to reach any depth:

```bash
actor_rollout_ref.actor.optim.lr=1e-6
actor_rollout_ref.rollout.val_kwargs.temperature=0.4
```

**Lists** — use bracket syntax:

```bash
trainer.logger=['console','wandb']
```

**Boolean values**:

```bash
actor_rollout_ref.actor.use_kl_loss=True
env.informal_math.enable_python_code=true
```

**Adding new keys** — prefix with `+`:

```bash
+data.response_dict_keys=['answer']
```

**Null values**:

```bash
trainer.default_hdfs_dir=null
```

## Variable Interpolation

Config files use OmegaConf's `${...}` syntax for cross-referencing values, keeping configs DRY:

```yaml
critic:
  rollout_n: ${actor_rollout_ref.rollout.n}
  ppo_epochs: ${actor_rollout_ref.actor.ppo_epochs}
  shuffle: ${actor_rollout_ref.actor.shuffle}
  loss_agg_mode: ${actor_rollout_ref.actor.loss_agg_mode}
```

This means changing `actor_rollout_ref.rollout.n` automatically updates `critic.rollout_n` as well.

## Config File Structure

A typical RL training config (`ppo_trainer.yaml`) is organized into the following top-level sections:

```yaml
data:              # Dataset paths, tokenization, batch sizes
actor_rollout_ref: # Actor model, rollout engine, reference model
critic:            # Critic model (for PPO)
reward_model:      # Reward model settings
algorithm:         # RL algorithm hyperparameters
trainer:           # Training loop, logging, checkpointing
env:               # Environment configuration
ray_init:          # Ray cluster settings
```

See the dedicated pages for detailed breakdowns:

- [RL Training Config](./rl_config.md) — Full `ppo_trainer.yaml` parameter reference (PPO, GRPO)
- [Generation Config](./generation.md) — `generation.yaml` parameter reference for offline inference
- [Evolving Config](./evolving.md) — Evolving pipeline configuration for self-improvement loops

## Recipe Configs

Algorithm-specific recipes inherit from the base `ppo_trainer.yaml` using Hydra's `defaults` mechanism:

```yaml
# recipe/dapo/config/dapo_trainer.yaml
defaults:
  - ppo_trainer
  - _self_
```

This allows recipes to only specify the parameters that differ. For example:

| Recipe | Key Overrides                             |
| ------ | ----------------------------------------- |
| DAPO   | `filter_groups.enable=True`, custom reward manager |
| SPPO   | `sppo_eta` parameter                      |
| SPIN   | `dpo_beta` parameter                      |
| PRIME  | RLOO advantage estimator, custom reward model |

## Environment Variables

Several environment variables affect training behavior:

```bash
# Debugging
export HYDRA_FULL_ERROR=1          # Show full Hydra error traces

# Ray
export RAY_DISABLE_DASHBOARD=1     # Disable Ray dashboard
export RAY_USAGE_STATS_ENABLED=0   # Disable telemetry
export RAY_memory_usage_threshold=0.98  # OOM kill threshold

# vLLM
export VLLM_ATTENTION_BACKEND=XFORMERS  # Attention backend
export CUDA_VISIBLE_DEVICES=0,1,2,3     # GPU selection
```

## Typical Training Launch

A complete GRPO training launch typically looks like:

```bash
set -x
ENGINE=${1:-vllm}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/train.parquet \
    data.val_files=$HOME/data/test.parquet \
    data.train_batch_size=8 \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    env.env_name=informal_math_training \
    env.max_steps=4 \
    env.rollout.n=8 \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.total_epochs=1
```

:::tip Debugging Hydra errors
Set `HYDRA_FULL_ERROR=1` before launching to get the full Python traceback instead of Hydra's condensed error output.
:::

## Related Pages

- [RL Training Config](./rl_config.md) — Detailed PPO/GRPO config reference
- [Generation Config](./generation.md) — Offline generation parameter reference
- [Evolving Config](./evolving.md) — Evolving Pipeline configuration reference
- [RL Training Algorithm](../algorithms/rl-training.md) — How the RL algorithms work
- [SFT](../algorithms/sft.md) — Supervised Fine-Tuning process
