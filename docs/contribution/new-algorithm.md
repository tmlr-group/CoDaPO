---
sidebar_label: "Adding a New Algorithm"
sidebar_position: 4
---

# Adding a New Algorithm / Workflow

This guide explains how to add a new training or inference algorithm to AlphaApollo. In AlphaApollo's terminology an **algorithm** is surfaced as a **workflow** — a CLI entry point that loads a YAML config, runs data preprocessing, and launches a trainer or inference loop.

:::info Prerequisite reading

- [agent-system.md](../core-modules/agent-system.md) for the environment loop
- [evolution.md](../core-modules/evolution.md) for the evolution pipeline
- [tools.md](../core-modules/tools.md) for tool integration
  :::

## Workflow Architecture

```text
CLI entry point                    YAML config
  (workflows/my_algo.py)             (examples/configs/my_algo.yaml)
        │                                  │
        ▼                                  ▼
  parse_standard_args()            load_config()
        │                                  │
        └──────────┬───────────────────────┘
                   ▼
             api.my_algo()
                   │
          ┌────────┼────────────┐
          ▼        ▼            ▼
     set env    run_modules   run_trainer / custom entrypoint
     variables  (preprocess)  (training / inference)
```

### Existing Workflows

| Workflow | Entry Point         | `api.py` Function | Runner Module                  | Launcher              | Flow                       |
| -------- | ------------------- | ----------------- | ------------------------------ | --------------------- | -------------------------- |
| **RL**   | `workflows/rl.py`   | `api.rl()`        | `verl.trainer.main_ppo`        | `python`              | `_run_standard_workflow`   |
| **SFT**  | `workflows/sft.py`  | `api.sft()`       | configurable                   | `python` / `torchrun` | `_run_standard_workflow`   |
| **Test** | `workflows/test.py` | `api.test()`      | `verl.trainer.main_generation` | `python`              | `_run_standard_workflow`   |
| **Evo**  | `workflows/evo.py`  | `api.evo()`       | `evolving.evolving_main`       | `python`              | Custom (entrypoint module) |

All standard workflows (RL, SFT, Test) share the same three-phase pipeline implemented by `_run_standard_workflow()`:

1. **Load config** → `load_config(config_path)`
2. **Set env vars** → `env_with_overrides(cfg["env"])`
3. **Preprocess** → `run_modules(cfg["preprocess"], env)` — runs each data prep script as `python -m <module> <args>`
4. **Train / Infer** → `run_trainer(cfg["runner"], env, extra_overrides)` — launches the runner module via `python -m` or `torchrun`

The Evolution workflow uses a custom flow that calls a configurable `entrypoint_module` instead of `run_trainer`.

:::info Config args conversion
The `preprocess` section’s `args` dict is converted to CLI arguments by `normalize_cli_args()` in `api.py` — e.g., `{data_source: "X", local_dir: "./data"}` becomes `--data_source X --local_dir ./data`. List values are joined with spaces.
:::

## Choosing a Workflow Type

Before implementing, decide which pattern fits your algorithm:

| Scenario | Recommended Flow | Reason |
| --- | --- | --- |
| Standard gradient-based training (PPO, GRPO, DPO, etc.) | Standard (`_run_standard_workflow`) | Reuses verl’s distributed trainer infrastructure |
| Supervised fine-tuning on expert data | Standard with `torchrun` | Leverages multi-GPU data parallelism |
| Iterative inference-time refinement | Custom (like `evo()`) | Needs custom orchestration beyond preprocess → train |
| Evaluation / benchmarking | Standard (`test` pattern) | Runs generation without gradient updates |

## Key Components

### YAML Configuration Schema

Every workflow is driven by a single YAML file with three sections:

```yaml
# 1. Environment variables (set before any subprocess)
env:
  CUDA_VISIBLE_DEVICES: "0,1"
  VLLM_ATTENTION_BACKEND: XFORMERS

# 2. Data preprocessing steps (run sequentially)
preprocess:
  - module: alphaapollo.data_preprocess.prepare_rl_training_data
    args:
      data_source: your-org/your-dataset
      local_dir: ./data

# 3. Training / inference runner
runner:
  launcher: python # "python" or "torchrun"
  module: your.training.module
  overrides: # Hydra-style CLI overrides
    - trainer.total_epochs=5
    - env.env_name=my_domain
    - data.train_batch_size=16
```

### CLI Argument Parser

`workflows/common.py` provides a reusable parser:

```python
from alphaapollo.workflows.common import parse_standard_args

config, overrides = parse_standard_args(
    description="Run my algorithm.",
    default_config="examples/configs/my_algo.yaml",
)
# config   → path to YAML file (--config flag)
# overrides → list of extra CLI strings appended to runner.overrides
```

### RewardManager

The `EpisodeRewardManager` (in `core/reward_manager/episode.py`) converts per-episode rewards into a token-level reward tensor compatible with verl's PPO trainer:

```python
class EpisodeRewardManager:
    def __init__(self, tokenizer, num_examine, normalize_by_length=False): ...
    def __call__(self, data: DataProto, return_dict=False) -> torch.Tensor: ...
```

- Reads `episode_rewards` and `episode_lengths` from `data.non_tensor_batch`.
- Places the scalar reward at the last valid response token position.
- Supports `normalize_by_length` to divide the reward by episode length.- When `return_dict=True`, returns `{"reward_tensor": ..., "reward_extra_info": {}}` instead of just the tensor.
If your algorithm uses a fundamentally different reward structure, you can subclass `EpisodeRewardManager` or implement a new reward manager.

## Step-by-Step Guide

### Step 1 — Implement the Training / Inference Module

Create your algorithm's entry point under `alphaapollo/core/generation/`:

```text
alphaapollo/core/generation/my_algorithm/
├── __init__.py
└── main.py          ← your training / inference logic
```

The entry point must be runnable via `python -m alphaapollo.core.generation.my_algorithm.main`. It typically:

1. Parses Hydra-style config overrides from `sys.argv`.
2. Builds the environment via `make_envs(config)`.
3. Runs the training or inference loop.
4. Saves results / checkpoints.

```python
# alphaapollo/core/generation/my_algorithm/main.py

import hydra
from omegaconf import DictConfig
from alphaapollo.core.environments.env_manager import make_envs


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(config: DictConfig):
    envs, val_envs = make_envs(config)

    # Your training / inference loop
    for epoch in range(config.trainer.total_epochs):
        # ... collect trajectories, compute rewards, update model ...
        pass

    envs.close()
    val_envs.close()


if __name__ == "__main__":
    main()
```

:::note
If your algorithm does not use Hydra, you can use `argparse` or any other CLI framework — just make sure the runner overrides are passed as CLI arguments.
:::

### Step 2 — Create the Workflow Entry Point

Create `alphaapollo/workflows/my_algo.py` following the established pattern:

```python
# alphaapollo/workflows/my_algo.py

from __future__ import annotations

from alphaapollo.workflows import api
from alphaapollo.workflows.common import parse_standard_args


def main() -> None:
    config, overrides = parse_standard_args(
        description="Run AlphaApollo my_algo workflow.",
        default_config=api.DEFAULT_CONFIGS["my_algo"],
    )
    api.my_algo(config_path=config, extra_overrides=overrides)


if __name__ == "__main__":
    main()
```

### Step 3 — Register in `api.py`

Open `alphaapollo/workflows/api.py` and make two changes:

**A. Add a default config path:**

```python
DEFAULT_CONFIGS = {
    "rl": "examples/configs/rl_informal_math_tool.yaml",
    "sft": "examples/configs/sft_informal_math_tool.yaml",
    "test": "examples/configs/test_informal_math.yaml",
    "evo": "examples/configs/vllm_informal_math.yaml",
    "my_algo": "examples/configs/my_algo.yaml",   # ← add
}
```

**B. Add the workflow function.** Choose one of two patterns:

**Standard flow** (if your algorithm follows the preprocess → runner pattern):

```python
def my_algo(
    config_path: str = DEFAULT_CONFIGS["my_algo"],
    extra_overrides: Optional[List[str]] = None,
) -> None:
    _run_standard_workflow(config_path=config_path, extra_overrides=extra_overrides)
```

**Custom flow** (if your algorithm needs a different launch sequence — like evolution):

```python
def my_algo(
    config_path: str = DEFAULT_CONFIGS["my_algo"],
    extra_overrides: Optional[List[str]] = None,
) -> None:
    cfg = load_config(config_path)
    env = env_with_overrides(cfg.get("env"))
    run_modules(cfg.get("preprocess", []), env=env)

    # Custom launch logic
    module = cfg.get("entrypoint_module", "alphaapollo.core.generation.my_algorithm.main")
    cmd = [sys.executable, "-m", module, "--config", config_path]
    if extra_overrides:
        cmd.extend(extra_overrides)
    run_cmd(cmd, env=env)
```

### Step 4 — Export in `__init__.py`

Open `alphaapollo/__init__.py` and add the new workflow:

```python
from .workflows.api import evo, rl, sft, test, my_algo

__all__ = ["rl", "sft", "test", "evo", "my_algo"]
```

### Step 5 — Create the YAML Configuration

Create `examples/configs/my_algo.yaml`:

```yaml
env:
  CUDA_VISIBLE_DEVICES: "0,1"

preprocess:
  - module: alphaapollo.data_preprocess.prepare_rl_training_data
    args:
      data_source: your-org/your-dataset

runner:
  launcher: python
  module: alphaapollo.core.generation.my_algorithm.main
  overrides:
    - env.env_name=my_domain
    - env.seed=42
    - env.max_steps=5
    - data.train_batch_size=8
    - data.val_batch_size=64
    - trainer.total_epochs=3
    - trainer.save_freq=100
```

### Step 6 — Create the Run Script

Create `examples/my_algo/run_my_algo.sh`:

```bash
#!/bin/bash
set -euo pipefail

python -m alphaapollo.workflows.my_algo \
    --config examples/configs/my_algo.yaml \
    "$@"
```

Make it executable:

```bash
chmod +x examples/my_algo/run_my_algo.sh
```

## Customizing the Reward Function

There are two levels at which you can inject custom reward logic:

### Environment-Level Reward

Override `_get_reward(done)` in your domain environment (see [new-environment.md](new-environment.md)):

```python
def _get_reward(self, done: bool) -> float:
    if not done:
        return 0.0
    # Your custom scoring logic
    return my_custom_score(self.chat_history, self.ground_truth)
```

### Trainer-Level Reward

Subclass `EpisodeRewardManager` to add reward shaping, bonuses, or penalties at the token level:

```python
from alphaapollo.core.reward_manager.episode import EpisodeRewardManager

class MyRewardManager(EpisodeRewardManager):
    def __call__(self, data, return_dict=False):
        reward_tensor = super().__call__(data, return_dict=False)
        # Add a length penalty
        for i in range(len(data)):
            resp_len = data[i].batch['attention_mask'][data[i].batch['prompts'].shape[-1]:].sum()
            reward_tensor[i, resp_len - 1] -= 0.01 * resp_len
        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": {}}
        return reward_tensor
```

Then register it in your training module's config or code.

## Using `torchrun` for Distributed Training

If your algorithm requires multi-GPU data-parallel training, set `launcher: torchrun` in the YAML config:

```yaml
runner:
  launcher: torchrun
  module: alphaapollo.core.generation.my_algorithm.main
  torchrun:
    standalone: true
    nnodes: 1
    nproc_per_node: 4
  overrides:
    - trainer.total_epochs=3
```

The `run_trainer()` function in `api.py` will automatically construct the correct `torchrun` command.

## Verification

1. **Config validation** — load and inspect:

   ```python
   from alphaapollo.workflows.api import load_config
   cfg = load_config("examples/configs/my_algo.yaml")
   print(cfg)
   ```

2. **Dry-run preprocess** — verify data scripts run:

   ```bash
   python -m alphaapollo.data_preprocess.prepare_rl_training_data \
       --data_source your-org/your-dataset
   ```

3. **End-to-end** — launch the full workflow:

   ```bash
   python -m alphaapollo.workflows.my_algo \
       --config examples/configs/my_algo.yaml
   ```

4. **Verify exports** — confirm the module is importable:

   ```python
   from alphaapollo import my_algo
   ```

## Checklist

| #   | Item                                             | Where                                  |
| --- | ------------------------------------------------ | -------------------------------------- |
| 1   | Implement training / inference module            | `core/generation/my_algorithm/main.py` |
| 2   | Create workflow entry point                      | `workflows/my_algo.py`                 |
| 3   | Register in `api.py` (default config + function) | `workflows/api.py`                     |
| 4   | Export in `__init__.py`                          | `alphaapollo/__init__.py`              |
| 5   | Create YAML config                               | `examples/configs/my_algo.yaml`        |
| 6   | Create run script                                | `examples/my_algo/run_my_algo.sh`      |
| 7   | (Optional) Custom reward manager                 | `core/reward_manager/`                 |
