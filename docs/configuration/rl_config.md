---
id: rl-config
title: RL Training Config
description: Full parameter reference for ppo_trainer.yaml — data, actor, rollout, critic, reward, algorithm, environment, and trainer sections.
sidebar_label: "RL Training Config"
sidebar_position: 2
---

# RL Training Config

This page provides a detailed breakdown of the `ppo_trainer.yaml` configuration file used by `verl.trainer.main_ppo`. This single config drives PPO, GRPO, DAPO and other RL algorithm variants.

## Data

```yaml
data:
  tokenizer: null
  train_files: ~/data/rlhf/gsm8k/train.parquet
  val_files: ~/data/rlhf/gsm8k/test.parquet
  prompt_key: prompt
  reward_fn_key: data_source
  max_prompt_length: 512
  max_response_length: 512
  train_batch_size: 1024
  val_batch_size: null
  return_raw_input_ids: False
  return_raw_chat: False
  return_full_prompt: False
  shuffle: True
  filter_overlong_prompts: False
  filter_overlong_prompts_workers: 1
  truncation: error
  image_key: images
  video_key: videos
  trust_remote_code: False
  custom_cls:
    path: null
    name: null
```

| Parameter                      | Type       | Description                                                                                     |
| ------------------------------ | ---------- | ----------------------------------------------------------------------------------------------- |
| `train_files`                  | str / list | Training data parquet path(s). Supports local or HDFS paths.                                    |
| `val_files`                    | str / list | Validation data parquet path(s).                                                                |
| `prompt_key`                   | str        | Column name for prompts in the dataset. Default: `prompt`.                                      |
| `reward_fn_key`                | str        | Column name for reward function dispatch. Default: `data_source`.                               |
| `max_prompt_length`            | int        | Maximum prompt token length. Prompts are left-padded to this length.                            |
| `max_response_length`          | int        | Maximum response token length for rollout generation.                                           |
| `train_batch_size`             | int        | Global batch size per training iteration.                                                       |
| `val_batch_size`               | int        | Validation batch size. `null` defaults to `train_batch_size`.                                  |
| `return_raw_input_ids`         | bool       | Return un-templated input IDs. Set `True` when policy and RM use different chat templates.      |
| `return_raw_chat`              | bool       | Return raw chat prompt without applying chat template.                                          |
| `shuffle`                      | bool       | Shuffle training data in the dataloader.                                                        |
| `filter_overlong_prompts`      | bool       | Filter out prompts exceeding `max_prompt_length`.                                               |
| `truncation`                   | str        | Truncation strategy: `error` (raise error), `left`, `right`, or `middle`.                      |
| `trust_remote_code`            | bool       | Allow remote tokenizer code.                                                                    |
| `custom_cls.path`              | str        | Path to custom dataset class file.                                                              |
| `custom_cls.name`              | str        | Name of the custom dataset class.                                                               |

## Actor / Rollout / Reference Policy

### Model Configuration

```yaml
actor_rollout_ref:
  hybrid_engine: True
  model:
    path: ~/models/deepseek-llm-7b-chat
    external_lib: null
    override_config: {}
    enable_gradient_checkpointing: True
    enable_activation_offload: False
    use_remove_padding: False
    lora_rank: 0
    lora_alpha: 16
    target_modules: all-linear
    use_liger: False
    trust_remote_code: False
```

| Parameter                         | Type       | Description                                                                    |
| --------------------------------- | ---------- | ------------------------------------------------------------------------------ |
| `hybrid_engine`                   | bool       | Enable hybrid engine (actor + rollout on same GPUs). Currently only `True` is supported. |
| `model.path`                      | str        | HuggingFace model path (local or remote).                                      |
| `model.external_lib`              | str        | Extra Python packages to import for model/tokenizer registration.              |
| `model.override_config`           | dict       | Override model config values (e.g., attention implementation).                 |
| `model.enable_gradient_checkpointing` | bool   | Enable gradient checkpointing to save memory.                                  |
| `model.use_remove_padding`        | bool       | Remove padding tokens for better efficiency.                                   |
| `model.lora_rank`                 | int        | LoRA rank. Set > 0 to enable LoRA training.                                    |
| `model.lora_alpha`                | int        | LoRA scaling factor.                                                           |
| `model.target_modules`            | str / list | LoRA target modules. `all-linear` or explicit list.                            |

### Actor Training

```yaml
  actor:
    strategy: fsdp               # fsdp or fsdp2
    ppo_mini_batch_size: 256     # global mini-batch size for PPO updates
    ppo_micro_batch_size_per_gpu: null  # gradient accumulation granularity
    use_dynamic_bsz: False
    ppo_max_token_len_per_gpu: 16384
    grad_clip: 1.0
    clip_ratio: 0.2              # PPO clip ratio
    clip_ratio_low: 0.2
    clip_ratio_high: 0.2
    clip_ratio_c: 3.0            # Dual-clip PPO lower bound
    loss_agg_mode: "token-mean"  # or "seq-mean-token-sum" / "seq-mean-token-mean"
    entropy_coeff: 0.001
    use_kl_loss: False           # True for GRPO
    kl_loss_coef: 0.001
    kl_loss_type: low_var_kl     # kl, abs, mse, low_var_kl, full
    use_invalid_action_penalty: True
    invalid_action_penalty_coef: 0.1
    ppo_epochs: 1
    shuffle: False
    ulysses_sequence_parallel_size: 1
    use_torch_compile: True
    optim:
      lr: 1e-6
      lr_warmup_steps: -1
      lr_warmup_steps_ratio: 0.
      min_lr_ratio: 0.0
      warmup_style: constant    # constant or cosine
      weight_decay: 0.01
    fsdp_config:
      wrap_policy:
        min_num_params: 0
      param_offload: False
      optimizer_offload: False
      fsdp_size: -1
    checkpoint:
      contents: ["model", "optimizer", "extra"]
```

**Key parameters:**

- `ppo_mini_batch_size`: The global mini-batch size. One training batch is split into sub-batches of this size for PPO updates.
- `ppo_micro_batch_size_per_gpu`: Per-GPU forward pass batch size (gradient accumulation). Smaller = less memory, slower.
- `clip_ratio`: Standard PPO clipping range `[1-clip, 1+clip]`. Controls policy update aggressiveness.
- `loss_agg_mode`: How to aggregate per-token losses: `token-mean` (default), `seq-mean-token-sum`, or `seq-mean-token-mean`.
- `use_kl_loss`: Enable KL divergence loss against the reference model. **Set to `True` for GRPO.**
- `kl_loss_type`: KL estimation method. `low_var_kl` (k3) is recommended for lower variance.
- `use_invalid_action_penalty`: Penalize invalid actions in agentic environments.

### Rollout Engine

```yaml
  rollout:
    name: vllm                 # vllm, sglang, or hf
    mode: sync                 # sync (LLM) or async (AsyncLLM)
    temperature: 1.0
    top_k: -1
    top_p: 1
    prompt_length: ${data.max_prompt_length}
    response_length: ${data.max_response_length}
    dtype: bfloat16
    gpu_memory_utilization: 0.5
    ignore_eos: False
    enforce_eager: True
    free_cache_engine: True
    load_format: dummy_dtensor
    tensor_model_parallel_size: 2
    max_num_batched_tokens: 8192
    max_num_seqs: 1024
    n: 1                       # >1 for GRPO (group size)
    enable_chunked_prefill: True
    val_kwargs:
      top_k: -1
      top_p: 1.0
      temperature: 0
      n: 1
      do_sample: False
    multi_turn:
      enable: False
      max_turns: null
      tool_config_path: null
      format: chatml
```

| Parameter                  | Type  | Description                                                                  |
| -------------------------- | ----- | ---------------------------------------------------------------------------- |
| `name`                     | str   | Inference engine: `vllm`, `sglang`, or `hf`.                                |
| `mode`                     | str   | `sync` for standard `LLM`, `async` for `AsyncLLM`.                          |
| `temperature`              | float | Sampling temperature during training rollout.                                |
| `n`                        | int   | Number of responses per prompt. Set > 1 for GRPO/RLOO.                       |
| `gpu_memory_utilization`   | float | Fraction of GPU memory for vLLM.                                             |
| `tensor_model_parallel_size` | int | Tensor parallelism degree for rollout.                                       |
| `free_cache_engine`        | bool  | Offload KV cache after generation to save memory.                            |
| `load_format`              | str   | Weight loader: `dummy_dtensor`, `dtensor`, `hf`, `auto`.                    |
| `multi_turn.enable`        | bool  | Enable multi-turn tool interaction.                                          |
| `val_kwargs`               | dict  | Override sampling parameters for validation (typically greedy).              |

### Reference Model

```yaml
  ref:
    strategy: fsdp
    fsdp_config:
      param_offload: False
      wrap_policy:
        min_num_params: 0
    log_prob_micro_batch_size_per_gpu: null
```

The reference model is enabled when `actor.use_kl_loss=True` or `algorithm.use_kl_in_reward=True`.

:::tip Memory optimization
For models larger than 7B, enable `actor_rollout_ref.ref.fsdp_config.param_offload=True` on the reference model to reduce peak GPU memory pressure.
:::

## Critic Model

```yaml
critic:
  strategy: fsdp
  model:
    path: ~/models/deepseek-llm-7b-chat
    enable_gradient_checkpointing: True
  optim:
    lr: 1e-5
    warmup_style: constant
    weight_decay: 0.01
  ppo_mini_batch_size: ${actor_rollout_ref.actor.ppo_mini_batch_size}
  ppo_max_token_len_per_gpu: 32768
  grad_clip: 1.0
  cliprange_value: 0.5
```

:::note
The critic is used only for PPO (GAE advantage estimation). For GRPO, the critic is not required.
:::

## Reward Model

```yaml
reward_model:
  enable: False
  model:
    input_tokenizer: ${actor_rollout_ref.model.path}
    path: ~/models/FsfairX-LLaMA3-RM-v0.1
    trust_remote_code: False
  reward_manager: episode   # episode, naive, prime
  launch_reward_fn_async: False
custom_reward_function:
  path: null
  name: compute_score
```

- `enable`: Set `True` to use a model-based RM. When `False`, only custom reward functions are used.
- `reward_manager`: Determines how rewards are computed. `episode` for agentic tasks, `naive` for standard RLHF, `prime` for parallel verification.
- `custom_reward_function.path`: Path to your custom reward function file. Implement a `compute_score` function.

## Algorithm

```yaml
algorithm:
  gamma: 1.0
  lam: 1.0
  adv_estimator: gae           # gae, grpo, rloo
  norm_adv_by_std_in_grpo: True
  use_kl_in_reward: False
  kl_penalty: kl
  kl_ctrl:
    type: fixed
    kl_coef: 0.001
  filter_groups:               # DAPO configuration
    enable: False
    max_num_gen_batches: 10
```

| Parameter                  | Type  | Description                                                                       |
| -------------------------- | ----- | --------------------------------------------------------------------------------- |
| `adv_estimator`            | str   | Advantage estimator: `gae` (PPO), `grpo`, `rloo`, `reinforce_plus_plus`.          |
| `gamma`                    | float | Discount factor. `1.0` for episodic tasks, < 1.0 for continuing tasks (e.g., `0.9`–`0.95`). |
| `lam`                      | float | GAE lambda. Trade-off between bias and variance.                                  |
| `use_kl_in_reward`         | bool  | Add KL penalty to the reward signal.                                              |
| `kl_ctrl.type`             | str   | `fixed` or `adaptive` KL controller.                                             |
| `filter_groups.enable`     | bool  | Enable DAPO-style group filtering.                                                |

### Choosing an Algorithm

| Algorithm | `adv_estimator`                        | `rollout.n` / `env.rollout.n` | `actor.use_kl_loss` | Critic Required |
| --------- | -------------------------------------- | ----------------------------- | ------------------- | --------------- |
| PPO       | `gae`                                  | 1                             | False               | Yes             |
| GRPO      | `grpo`                                 | > 1 (e.g., 8)                 | True                | No              |
| RLOO      | `rloo`                                 | > 1                           | False               | No              |
| DAPO      | `grpo` + `filter_groups.enable=True`   | > 1                           | True                | No              |

## Environment

```yaml
env:
  env_name: alfworld/AlfredTWEnv
  seed: 0
  max_steps: 50
  history_length: 2
  resources_per_worker:
    num_cpus: 0.1
    num_gpus: 0
  rollout:
    n: 1                        # Group size for GRPO
  informal_math:
    memory_type: simple          # simple, score, ndimensional
    enable_python_code: true
    enable_local_rag: true
    python_code_timeout: 30
  mol_optim:
    memory_type: simple
    timeout: 5
    use_intermediate_reward: False
```

| Parameter               | Type  | Description                                                                                     |
| ----------------------- | ----- | ----------------------------------------------------------------------------------------------- |
| `env_name`              | str   | Environment identifier. Options: `alfworld/AlfredTWEnv`, `informal_math_training`, `mol_optim`, `sokoban`, `webshop`, etc. |
| `max_steps`             | int   | Maximum interaction steps per episode.                                                          |
| `history_length`        | int   | Number of past turns included in the observation.                                               |
| `rollout.n`             | int   | Number of parallel environment groups per prompt (for GRPO).                             |
| `resources_per_worker`  | dict  | Ray resource allocation per environment worker.                                                 |

### Environment-Specific Settings

Each environment has its own sub-config. For example, `informal_math` supports:
- `memory_type`: Memory implementation (`simple`, `score`, `ndimensional`)
- `enable_python_code`: Enable Python code execution tool
- `enable_local_rag`: Enable local RAG retrieval tool
- `python_code_timeout`: Timeout for code execution (seconds)

## Trainer

```yaml
trainer:
  total_epochs: 30
  total_training_steps: null
  project_name: verl_examples
  experiment_name: gsm8k
  logger: ["console", "wandb"]
  nnodes: 1
  n_gpus_per_node: 8
  save_freq: -1
  test_freq: -1
  val_before_train: True
  critic_warmup: 0
  resume_mode: auto           # auto, disable, resume_path
  resume_from_path: null
  default_local_dir: checkpoints/${trainer.project_name}/${trainer.experiment_name}
  max_actor_ckpt_to_keep: null
```

| Parameter               | Type  | Description                                                                                       |
| ----------------------- | ----- | ------------------------------------------------------------------------------------------------- |
| `total_epochs`          | int   | Number of training epochs.                                                                        |
| `total_training_steps`  | int   | Alternative to epochs: stop after this many steps.                                                |
| `logger`                | list  | Logging backends: `console`, `wandb`, `tensorboard`, `mlflow`.                                   |
| `save_freq`             | int   | Save checkpoint every N iterations. `-1` = never.                                                |
| `test_freq`             | int   | Run validation every N iterations.                                                                |
| `resume_mode`           | str   | `auto` resumes from latest checkpoint; `disable` starts fresh; `resume_path` uses `resume_from_path`. |
| `max_actor_ckpt_to_keep`| int   | Max checkpoints to retain. `null` = keep all.                                                    |

## Full Example

A complete GRPO training command for informal math:

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/train.parquet \
    data.val_files=$HOME/data/test.parquet \
    data.train_batch_size=8 \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    env.env_name=informal_math_training \
    env.max_steps=4 \
    env.history_length=4 \
    env.rollout.n=8 \
    env.informal_math.enable_python_code=true \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.total_epochs=1
```

## Related Pages

- [Configuration Overview](./index.md) — Hydra basics and CLI override syntax
- [Generation Config](./generation.md) — Offline generation configuration
- [Evolving Config](./evolving.md) — Evolving pipeline configuration
- [RL Training Algorithm](../algorithms/rl-training.md) — How the algorithms work
