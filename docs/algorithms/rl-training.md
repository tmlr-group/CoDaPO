---
id: rl-training
title: RL Training
description: Reinforcement learning algorithms available in AlphaApollo — PPO, GRPO, DAPO, and RLOO — with training architecture and configuration examples.
sidebar_label: "RL Training"
sidebar_position: 1
---

# RL Training

AlphaApollo integrates with [verl](https://github.com/volcengine/verl) (Versatile RL) for production-grade RL-based LLM post-training. It exposes PPO, GRPO, DAPO, and RLOO through a single unified entry point. This page covers each algorithm, the shared training architecture, and key configuration parameters.

## Supported Algorithms

AlphaApollo supports multiple RL algorithms through a unified entry point (`verl.trainer.main_ppo`). The algorithm is selected via the `algorithm.adv_estimator` config parameter.

### PPO (Proximal Policy Optimization)

PPO is the standard RL algorithm for LLM post-training. It uses a critic model to estimate the value function and computes advantages via Generalized Advantage Estimation (GAE).

**Key characteristics:**
- Requires a **critic model** in addition to the actor
- Uses GAE for advantage estimation
- Supports clipped surrogate objective with configurable clip ratio
- Single rollout per prompt (`n=1`)

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    algorithm.gamma=1.0 \
    algorithm.lam=1.0 \
    actor_rollout_ref.rollout.n=1 \
    ...
```

:::tip When to use PPO
- When you have a trained reward model
- When sample efficiency matters (PPO can learn from fewer samples)
- When you need fine-grained value estimation at each token
:::

### GRPO (Group Relative Policy Optimization)

GRPO eliminates the critic by estimating advantages from a **group of rollouts** for each prompt. For each prompt, multiple responses are generated, and their rewards are compared within the group to compute relative advantages.

**Key characteristics:**
- **No critic model** needed (reduces memory and compute)
- Generates multiple responses per prompt (`n > 1`)
- Uses KL divergence loss against reference model
- Advantages are normalized within each group

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    env.rollout.n=8 \
    ...
```

:::tip When to use GRPO
- When you want simpler training without a critic
- When GPU memory is a constraint
- For agentic tasks where group comparison is natural
:::

### DAPO (Data-Augmented Policy Optimization)

DAPO builds on GRPO with an additional **group filtering** mechanism that regenerates rollouts when a group provides no learning signal (all rewards are the same).

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.max_num_gen_batches=10 \
    ...
```

### RLOO (Relative Likelihood Optimization)

RLOO uses a leave-one-out baseline for advantage estimation across multiple rollouts.

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=rloo \
    env.rollout.n=8 \
    ...
```

## Algorithm Comparison

| Feature                      | PPO      | GRPO     | DAPO     | RLOO     |
| ---------------------------- | -------- | -------- | -------- | -------- |
| Critic required              | Yes      | No       | No       | No       |
| Group rollouts (`n > 1`)     | No       | Yes      | Yes      | Yes      |
| KL loss                      | Optional | Yes      | Yes      | Optional |
| Step-level advantage         | No       | No       | No       | No       |
| Discount factor (`gamma < 1`)| Via GAE  | N/A      | N/A      | N/A      |
| Group filtering              | No       | No       | Yes      | No       |

## Training Architecture

AlphaApollo uses verl's **HybridFlow** architecture that combines single-controller orchestration with multi-controller execution:

```
┌──────────────────────────────────────────────────────┐
│                   PPO Ray Trainer                     │
│              (Single Controller / Driver)             │
├──────────────────────────────────────────────────────┤
│                                                      │
│   ┌─────────────┐   ┌─────────────┐                 │
│   │    Actor     │   │   Rollout   │                 │
│   │   (FSDP)    │◄──│   (vLLM)   │                 │
│   │  Training   │   │ Generation  │                 │
│   └──────┬──────┘   └──────┬──────┘                 │
│          │                  │                         │
│   ┌──────┴──────┐   ┌──────┴──────┐                 │
│   │  Reference  │   │   Critic    │                 │
│   │   Model     │   │  (PPO only) │                 │
│   └─────────────┘   └─────────────┘                 │
│                                                      │
│   ┌─────────────┐   ┌─────────────┐                 │
│   │   Reward    │   │ Environment │                 │
│   │   Model/Fn  │   │  Workers    │                 │
│   └─────────────┘   └─────────────┘                 │
└──────────────────────────────────────────────────────┘
```

### Training Loop

For each training iteration:

1. **Rollout**: Generate responses using the rollout engine (vLLM/SGLang) within the environment
2. **Reward**: Compute rewards using reward model and/or custom reward functions
3. **Advantage**: Estimate advantages (GAE for PPO, group-relative for GRPO)
4. **Update**: Update actor (and critic for PPO) using the clipped surrogate objective
5. **Sync**: Synchronize updated weights to the rollout engine

### Multi-Turn Interaction

For agentic tasks, the rollout phase involves multi-turn environment interaction:

```
┌───────┐     ┌───────────┐     ┌─────────────┐
│ Model │────▶│Environment│────▶│   Reward   │
│(Actor)│     │ (e.g.,    │     │ (per-step   │
│       │◀────│  Math)   │◀────│  or final)  │
└───────┘     └───────────┘     └─────────────┘
   │               │
   │  step 1..N    │
   │◀─────────────▶│
```

Each episode consists of multiple steps where the model generates actions, the environment processes them, and observations are returned.

## Key Training Parameters

### Learning Rate & Optimization

```bash
actor_rollout_ref.actor.optim.lr=1e-6
actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1
actor_rollout_ref.actor.optim.warmup_style=cosine  # or constant
actor_rollout_ref.actor.optim.weight_decay=0.01
actor_rollout_ref.actor.grad_clip=1.0
```

### KL Divergence Control

KL divergence prevents the policy from drifting too far from the reference model:

```bash
# Option 1: KL loss in the actor objective (recommended for GRPO)
actor_rollout_ref.actor.use_kl_loss=True
actor_rollout_ref.actor.kl_loss_coef=0.01
actor_rollout_ref.actor.kl_loss_type=low_var_kl

# Option 2: KL penalty in the reward
algorithm.use_kl_in_reward=True
algorithm.kl_ctrl.type=fixed
algorithm.kl_ctrl.kl_coef=0.001
```

KL loss types:
- `kl` (k1): Standard KL divergence
- `abs`: Absolute difference
- `mse` (k2): Mean squared error
- `low_var_kl` (k3): Low-variance KL estimator (recommended)
- `full`: Full KL divergence

### Invalid Action Penalty

For agentic environments, penalize malformed actions:

```bash
actor_rollout_ref.actor.use_invalid_action_penalty=True
actor_rollout_ref.actor.invalid_action_penalty_coef=0.1
```

## Data Preparation

Before training, prepare your dataset using the data preprocessing scripts:

```bash
# Prepare informal math dataset
python3 -m examples.data_preprocess.prepare_informal_math \
    --data_source DigitalLearningGmbH/MATH-lighteval

# Prepare text-based environment dataset
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size 16 \
    --val_data_size 128
```

Datasets are expected in **parquet format** with at least a `prompt` column.

## Multi-GPU and Multi-Node Training

AlphaApollo supports distributed training across multiple GPUs and nodes:

```bash
# Single node, 2 GPUs
trainer.n_gpus_per_node=2 trainer.nnodes=1

# Multi-node (4 nodes, 8 GPUs each)
trainer.n_gpus_per_node=8 trainer.nnodes=4
```

Key distributed training settings:

```bash
# Tensor parallelism for rollout
actor_rollout_ref.rollout.tensor_model_parallel_size=2

# Sequence parallelism for training
actor_rollout_ref.actor.ulysses_sequence_parallel_size=2

# FSDP offloading (trade speed for memory)
actor_rollout_ref.actor.fsdp_config.param_offload=True
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
actor_rollout_ref.ref.fsdp_config.param_offload=True  # recommended for ref model >7B
```

:::tip Memory optimization for large models
For reference models larger than 7B, enable `param_offload` to reduce peak GPU memory usage during training.
:::

## Related Pages

- [RL Training Config](../configuration/rl_config.md) — Detailed parameter reference
- [Configuration Overview](../configuration/index.md) — Hydra basics and CLI overrides
- [SFT](./sft.md) — Supervised Fine-Tuning process
- [Evolving Pipeline](./evolving-pipeline.md) — Inference-time self-improvement via policy-verifier loops
