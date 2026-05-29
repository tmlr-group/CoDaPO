---
sidebar_label: "Adding a New Environment"
sidebar_position: 3
---

# Adding a New Environment

This guide explains how to extend AlphaApollo with a new task environment. A new environment lets you plug a completely different problem domain (e.g. code generation, scientific reasoning, game playing) into the existing training, evolution, and testing pipelines.

:::info Prerequisite reading

- [agent-system.md](../core-modules/agent-system.md) for the layered architecture
- [tools.md](../core-modules/tools.md) for the tool framework that environments depend on
  :::

## Architecture Overview

AlphaApollo environments follow a five-layer stack. When you add a new domain you provide implementations for layers 2–5; layer 1 is reusable as-is.

```text
Env (core.py)                                ← 1. Abstract Gym-style interface
 └── BaseTextEnv (base_text_env.py)          ← 2. Tool registration & dispatch
      └── MyDomainEnv (env.py)               ← 3. Domain-specific logic
           └── MyDomainMultiProcessEnv       ← 4. Vectorized parallelism
               (envs.py)
                └── MyDomainEnvironmentManager  ← 5. Prompts, memory, projection
                    (env_manager.py)
```

| Layer             | Class                        | Key Methods                                                                                                  |
| ----------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------ |
| 1 — Abstract base | `Env[ObsType, ActType]`      | `step(action)`, `init(**kw)`, `close()`                                                                      |
| 2 — Text base     | `BaseTextEnv(Env[str, str])` | `init_tool_groups(groups)`, `_execute_tool(group, name, input)`                                              |
| 3 — Domain env    | `MyDomainEnv(BaseTextEnv)`   | `reset(extras)`, `step(action, text_actions)`, `_parse_action(action)`, `_is_done(...)`, `_get_reward(done)` |
| 4 — Multi-process | `MyDomainMultiProcessEnv`    | `reset(kwargs)`, `step(actions, text_actions)`, `close()`                                                    |
| 5 — Manager       | `MyDomainEnvironmentManager` | `reset(kwargs)`, `step(text_actions)`, `build_text_obs(...)`, `success_evaluator(...)`                       |

## Step-by-Step Guide

### Step 1 — Create the Environment Package

Create a new directory under `alphaapollo/core/environments/`:

```text
alphaapollo/core/environments/my_domain/
├── __init__.py
├── core.py            # (optional — can reuse the existing core.py)
├── base_text_env.py   # (optional — can reuse the existing base_text_env.py)
├── env.py             # ← your domain environment
├── envs.py            # ← vectorized wrapper + factory function
├── projection.py      # ← LLM output → structured action
└── utils/             # ← domain-specific helpers (scoring, parsing, …)
```

:::tip Shortcut
If your domain uses the same text-in/text-out pattern as informal math, you can import `Env` and `BaseTextEnv` from `informal_math_training` instead of copying them.
:::

### Step 2 — Implement the Domain Environment

This is the core of your extension. Subclass `BaseTextEnv` and implement the domain-specific logic.

```python
# alphaapollo/core/environments/my_domain/env.py

import re
import json
from typing import Any, Dict, List, Optional, Tuple
from omegaconf import DictConfig

# Reuse the shared base classes
from alphaapollo.core.environments.informal_math_training.base_text_env import (
    BaseTextEnv, BaseTextEnvStepOutput, ConversationType
)
from alphaapollo.core.tools import InformalMathToolGroup  # or your custom ToolGroup

# ──────────────────────────────────────────────
# Tool patterns — one entry per tool the model can call.
# Format: (tool_name, regex_pattern)
# ──────────────────────────────────────────────
TOOL_PATTERNS = [
    ("python_code", r"<python_code>(.*?)</python_code>"),
    # add more as needed
]


class MyDomainEnv(BaseTextEnv):
    """Single-instance environment for the My Domain task."""

    def __init__(self, env_config: DictConfig):
        super().__init__()

        tool_config = {
            "enable_python_code": getattr(env_config, "enable_python_code", True),
            "enable_local_rag": getattr(env_config, "enable_local_rag", False),
            "python_code_timeout": getattr(env_config, "python_code_timeout", 30),
            "rag_cfg": getattr(env_config, "rag", None),
        }
        self.tool_group = InformalMathToolGroup(
            log_requests=getattr(env_config, "log_requests", False),
            tool_config=tool_config,
        )
        self.init_tool_groups([self.tool_group])

    # ── reset ──────────────────────────────────
    def reset(self, extras: Optional[Dict[str, Any]] = None) -> None:
        extras = extras or {}
        self.question = extras["question"]
        self.ground_truth = extras["ground_truth"]
        self.max_steps = extras.get("max_steps", 3)
        self.data_source = extras.get("data_source", "unknown")

        # Inject ground truth into the tool group (for verification tools)
        self.tool_group.set_ground_truth(self.ground_truth)

        self.chat_history: ConversationType = []
        self.done = False
        self.turns = 0

    # ── action parsing ─────────────────────────
    def _parse_action(self, action: str) -> List[Tuple[Optional[str], Optional[str]]]:
        """Extract tool calls from the raw LLM action using TOOL_PATTERNS."""
        tool_calls = []
        for tool_name, pattern in TOOL_PATTERNS:
            if f"<{tool_name}>" in action and f"</{tool_name}>" in action:
                m = re.search(pattern, action, re.DOTALL)
                if m:
                    tool_calls.append((tool_name, m.group(1).strip()))
        return tool_calls if tool_calls else [(None, None)]

    # ── termination ────────────────────────────
    def _is_done(self, tool_calls) -> bool:
        if self.turns >= self.max_steps:
            return True
        if not tool_calls or all(tc == (None, None) for tc in tool_calls):
            return True
        return False

    # ── reward ─────────────────────────────────
    def _get_reward(self, done: bool) -> float:
        if not done:
            return 0.0
        # Concatenate full trajectory and score it
        trajectory = "".join(
            item["text_actions"] for item in self.chat_history
        )
        return self._compute_score(trajectory, self.ground_truth)

    @staticmethod
    def _compute_score(solution: str, ground_truth: str) -> float:
        """Domain-specific scoring function. Replace with your own."""
        # Example: exact-match binary reward
        return 1.0 if ground_truth.strip() in solution else 0.0

    # ── step ───────────────────────────────────
    def step(self, action, text_actions) -> BaseTextEnvStepOutput:
        self.turns += 1
        self.chat_history.append({
            "role": "assistant",
            "content": action,
            "text_actions": text_actions,
        })

        raw = text_actions if isinstance(text_actions, str) else action
        tool_calls = self._parse_action(raw)
        self.done = self._is_done(tool_calls)
        reward = self._get_reward(self.done)

        if self.done:
            return BaseTextEnvStepOutput(
                observations=[],
                reward=reward,
                done=True,
                metadata={"data_source": self.data_source, "tool_calling": False},
                postprocessed_action=action,
            )

        observations, tool_infos = [], []
        for tool_name, tool_input in tool_calls:
            if tool_name is None:
                continue
            # Dispatch to the right tool
            obs = self._execute_tool_wrapped(tool_name, tool_input)
            new_obs = {"role": "user", "content": obs, "text_actions": text_actions}
            self.chat_history.append(new_obs)
            observations.append(new_obs)
            tool_infos.append({
                "tool_calling": True,
                "tool_name": tool_name,
                "tool_input": tool_input,
                "data_source": self.data_source,
            })

        return BaseTextEnvStepOutput(
            observations=observations,
            reward=reward,
            done=False,
            metadata=tool_infos,
            postprocessed_action=action,
        )

    def _execute_tool_wrapped(self, tool_name: str, tool_input: str) -> str:
        """Call the tool group and wrap the result in <tool_response> tags."""
        tool_output = super()._execute_tool(
            self.tool_group.name, tool_name, {"query": tool_input}
        )
        text_result = tool_output.get("text_result", "")
        return f"\n<tool_response>{text_result}</tool_response>\n"
```

### Step 3 — Implement the Projection Function

The projection function maps raw LLM outputs to structured actions and a validity mask. It lives in `projection.py`.

```python
# alphaapollo/core/environments/my_domain/projection.py

import re
from typing import List, Tuple

# Tools the model can call — add new tags here
TOOL_CALLING_TOKENS = [
    "python_code",
    # ... add your domain tools
]


def _postprocess_action(action: str) -> str:
    """Trim at the first closing </answer> or </tool> tag."""
    answer_pos = action.find("</answer>")
    if answer_pos != -1:
        return action[: answer_pos] + "</answer>"

    earliest = len(action)
    tag = None
    for token in TOOL_CALLING_TOKENS:
        pos = action.find(f"</{token}>")
        if pos != -1 and pos < earliest:
            earliest = pos
            tag = f"</{token}>"
    if tag:
        return action[: earliest] + tag
    return action


def my_domain_projection(actions: List[str]) -> Tuple[List[str], List[int]]:
    """
    Project LLM actions into (results, valids).

    Extraction priority:
        1. First complete <answer>…</answer> block.
        2. First complete <TOOL>…</TOOL> block.
        3. Empty string (invalid).

    Validity is 0 when the action mixes tool and answer tags,
    or contains duplicate tags.
    """
    results: List[str] = []
    valids: List[int] = [1] * len(actions)

    re_tool = {
        t: re.compile(f"<{t}>(.*?)</{t}>", re.I | re.S) for t in TOOL_CALLING_TOKENS
    }
    re_tool_tag = {
        t: re.compile(f"<{t}>", re.I) for t in TOOL_CALLING_TOKENS
    }
    re_answer = re.compile(r"<answer>(.*?)</answer>", re.I | re.S)
    re_answer_tag = re.compile(r"<answer>", re.I)

    for i, action in enumerate(actions):
        trimmed = _postprocess_action(action)

        m = re_answer.search(trimmed)
        if m:
            results.append(f"<answer>{m.group(1).strip()}</answer>")
        else:
            found = False
            for t in TOOL_CALLING_TOKENS:
                m = re_tool[t].search(trimmed)
                if m:
                    results.append(f"<{t}>{m.group(1).strip()}</{t}>")
                    found = True
                    break
            if not found:
                results.append("")
                valids[i] = 0

        # Validity checks
        n_tool = sum(len(re_tool_tag[t].findall(action)) for t in TOOL_CALLING_TOKENS)
        n_ans = len(re_answer_tag.findall(action))
        if (n_tool and n_ans) or n_tool > 1 or n_ans > 1:
            valids[i] = 0

    return results, valids
```

### Step 4 — Implement the Multi-Process Wrapper

Wrap multiple environment instances for batch-parallel execution using a `ThreadPoolExecutor`. Export a `build_*_envs()` factory function.

```python
# alphaapollo/core/environments/my_domain/envs.py

import asyncio
import concurrent.futures
from typing import Dict, List
from copy import deepcopy

import gymnasium as gym
import numpy as np
from omegaconf import DictConfig


class MyDomainMultiProcessEnv(gym.Env):
    def __init__(self, seed=0, env_num=1, group_n=1, is_train=True, env_config=None):
        super().__init__()
        from alphaapollo.core.environments.my_domain.env import MyDomainEnv

        self.batch_size = env_num * group_n
        self.max_steps = env_config.max_steps

        cfg = env_config.my_domain
        self.envs = [MyDomainEnv(deepcopy(cfg)) for _ in range(self.batch_size)]

        workers = min(self.batch_size, 256)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        self._loop = asyncio.new_event_loop()

    # -- internal helpers ---------------------------------------------------
    def _sync_reset(self, env, kw):
        env.reset(kw)
        return kw["question"], {"data_source": kw.get("data_source", "unknown")}

    def _sync_step(self, env, action, text_actions):
        out = env.step(action, text_actions)
        obs = "" if not out["observations"] else out["observations"][0]["content"]
        reward = out["reward"]
        done = out["done"]
        info = {"tool_infos": out.get("metadata", []),
                "postprocessed_action": out.get("postprocessed_action"),
                "won": bool(done and reward > 0.0)}
        return obs, reward, done, info

    # -- public API ---------------------------------------------------------
    def reset(self, kwargs: List[Dict]):
        tasks = [
            self._loop.run_in_executor(self._executor, self._sync_reset, env, kw)
            for env, kw in zip(self.envs, kwargs)
        ]
        results = self._loop.run_until_complete(asyncio.gather(*tasks))
        obs, infos = map(list, zip(*results))
        return obs, infos

    def step(self, actions, text_actions):
        tasks = [
            self._loop.run_in_executor(self._executor, self._sync_step, e, a, t)
            for e, a, t in zip(self.envs, actions, text_actions)
        ]
        results = self._loop.run_until_complete(asyncio.gather(*tasks))
        obs, rews, dones, infos = map(list, zip(*results))
        return obs, rews, dones, infos

    def close(self):
        for e in self.envs:
            e.close()
        self._executor.shutdown(wait=True)
        self._loop.close()


def build_my_domain_envs(seed=0, env_num=1, group_n=1, is_train=True, env_config=None):
    return MyDomainMultiProcessEnv(
        seed=seed, env_num=env_num, group_n=group_n,
        is_train=is_train, env_config=env_config,
    )
```

### Step 5 — Create the Package `__init__.py`

Export exactly two symbols — the factory function and the projection function:

```python
# alphaapollo/core/environments/my_domain/__init__.py

from alphaapollo.core.environments.my_domain.projection import my_domain_projection
from alphaapollo.core.environments.my_domain.envs import build_my_domain_envs
```

### Step 6 — Register in `make_envs()`

Open `alphaapollo/core/environments/env_manager.py` and add an `elif` branch inside `make_envs()`:

```python
def make_envs(config):
    ...
    if "informal_math_training" in config.env.env_name.lower():
        ...
    elif "informal_math_evolving" in config.env.env_name.lower():
        ...

    # ↓ your new environment
    elif "my_domain" in config.env.env_name.lower():
        from .my_domain import build_my_domain_envs, my_domain_projection
        _envs = build_my_domain_envs(
            seed=config.env.seed,
            env_num=config.data.train_batch_size,
            group_n=group_n, is_train=True, env_config=config.env,
        )
        _val_envs = build_my_domain_envs(
            seed=config.env.seed + 1000,
            env_num=config.data.val_batch_size,
            group_n=1, is_train=False, env_config=config.env,
        )
        projection_f = partial(my_domain_projection)
        envs = MyDomainEnvironmentManager(_envs, projection_f, config)
        val_envs = MyDomainEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs

    else:
        raise ValueError(f"Environment {config.env.env_name} not supported")
```

### Step 7 — Implement the EnvironmentManager

Subclass `EnvironmentManagerBase` and implement `build_text_obs()` to construct the prompt the model will see.

```python
# Add to env_manager.py or a separate file

class MyDomainEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        # Choose memory backend
        mem_type = config.env.my_domain.memory_type
        if mem_type == "simple":
            self.memory = SimpleMemory()
        elif mem_type == "score":
            self.memory = EvolvingMemory(sort_key="score", descending=True)
        else:
            raise ValueError(f"Unknown memory type: {mem_type}")
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs):
        obs, infos = self.envs.reset(kwargs=kwargs)
        self.tasks = obs
        self.memory.reset(batch_size=len(obs))
        return {
            "text": self.build_text_obs(obs, init=True),
            "image": None,
            "anchor": obs.copy(),
        }, infos

    def step(self, text_actions):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions, text_actions)

        self.memory.store({
            "text_obs": next_obs,
            "action": text_actions,
        })

        next_observations = {
            "text": self.build_text_obs(next_obs),
            "image": None,
            "anchor": next_obs.copy(),
        }
        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])

        return next_observations, to_numpy(rewards), to_numpy(dones), infos

    def build_text_obs(self, text_obs, init=False):
        """Construct the text prompt for the model."""
        result = []
        for i, obs in enumerate(text_obs):
            if init:
                # First turn — include system instructions + question
                prompt = f"Solve the following problem:\n{self.tasks[i]}"
            else:
                # Subsequent turns — append observation to history
                prompt = obs
            result.append(prompt)
        return result
```

## Prompt Templates

Place your prompt templates in `alphaapollo/core/environments/prompts/my_domain.py` and import them in the `prompts/__init__.py`. Templates should define at the minimum:

- **System / initial prompt** — problem statement, available tools, format constraints.
- **Follow-up prompt** — memory context, step counter, instructions for how to continue.

See the existing `prompts/` directory for conventions.

## Memory Integration

AlphaApollo supports pluggable memory backends configured via `config.env.my_domain.memory_type`:

| Type           | Class                | Behaviour                                |
| -------------- | -------------------- | ---------------------------------------- |
| `simple`       | `SimpleMemory`       | FIFO buffer, no scoring                  |
| `search`       | `SearchMemory`       | Retrieval-based semantic search          |
| `score`        | `EvolvingMemory`     | Ranks stored entries by score            |
| `ndimensional` | `NDimensionalMemory` | Multi-dimensional Pareto-optimal ranking |

All memory classes implement the `BaseMemory` interface:

```python
class BaseMemory:
    def reset(self, batch_size: int): ...
    def store(self, data: Dict[str, Any]): ...
    def fetch(self, n: int, obs_key: str, action_key: str) -> Tuple[List[str], ...]: ...
```

To add a custom memory backend, subclass `BaseMemory`, implement the three methods above, and add a branch in your `EnvironmentManager.__init__`.

## Configuration

Add a section to your YAML config:

```yaml
runner:
  overrides:
    - env.env_name=my_domain
    - env.seed=42
    - env.max_steps=5
    - env.history_length=5
    - env.my_domain.memory_type=simple
    - env.my_domain.enable_python_code=true
    - env.my_domain.log_requests=false
```

## Verification

1. **Smoke test** — instantiate the environment directly:

   ```python
   from alphaapollo.core.environments.my_domain.env import MyDomainEnv
   env = MyDomainEnv(cfg)
   env.reset({"question": "What is 2+2?", "ground_truth": "4"})
   out = env.step("<answer>4</answer>", "<answer>4</answer>")
   assert out["done"] is True and out["reward"] == 1.0
   ```

2. **Batch test** — run through `make_envs()`:

   ```python
   from alphaapollo.core.environments.env_manager import make_envs
   envs, val_envs = make_envs(config)
   obs, infos = envs.reset(kwargs)
   ```

3. **End-to-end** — use the test workflow:

   ```bash
   python -m alphaapollo.workflows.test \
       --config examples/configs/test_my_domain.yaml
   ```

## Evolving Variant

If your environment should support the **self-evolution** workflow (policy-verifier iterative refinement), you also need to create an evolving variant:

```text
alphaapollo/core/environments/my_domain_evolving/
├── __init__.py
├── env.py             ← extends MyDomainEnv with <report> tag, done_reason, policy_solution
├── envs.py            ← vectorized wrapper with previous_solutions support
└── projection.py      ← adds <report> tag support (highest priority)
```

Key differences from the training variant:

| Feature | Training | Evolving |
| --- | --- | --- |
| Termination tags | `<answer>` only | `<answer>` + `<report>` |
| Roles | Policy only | Policy + Verifier |
| Extra fields | — | `policy_solution`, `done_reason`, `previous_solutions` |
| Force-done | — | Empty action triggers termination |

Register the evolving variant as a separate `elif` branch in `make_envs()` with an `InformalMathEvolvingEnvironmentManager`-style manager. See [`informal_math_evolving/`](https://github.com/tmlr-group/AlphaApollo) in the source for a complete reference implementation.

## Data Preprocessing

A new environment typically needs a corresponding data preprocessing script in `alphaapollo/data_preprocess/`. See [Dataset Pipeline](../core-modules/dataset.md) for the standard pattern. At minimum you need to:

1. Create `prepare_my_domain_data.py` following the `prepare_rl_training_data.py` template.
2. Define `QUESTION_KEYS` and `GROUND_TRUTH_KEYS` appropriate for your domain’s datasets.
3. Register the script in your workflow YAML under the `preprocess` section.

## Architecture Decisions

| Decision | Guidance |
| --- | --- |
| Reuse `InformalMathToolGroup` vs. create new ToolGroup | Reuse if your domain uses the same tools (Python, RAG, verify). Create a new group if you need domain-specific tools — see [Adding a New Tool](./new-tool.md). |
| Reuse `BaseTextEnv` vs. start from `Env` | Always subclass `BaseTextEnv` if your domain is text-in/text-out with tool calls. Only subclass `Env` directly for non-text domains (e.g., image-based). |
| Training variant only vs. training + evolving | Start with training only. Add an evolving variant when you want multi-round self-improvement with verifier feedback. |

## Checklist

| #   | Item                                 | Where                                       |
| --- | ------------------------------------ | ------------------------------------------- |
| 1   | Create environment package directory | `core/environments/my_domain/`              |
| 2   | Implement `MyDomainEnv`              | `core/environments/my_domain/env.py`        |
| 3   | Implement projection function        | `core/environments/my_domain/projection.py` |
| 4   | Implement multi-process wrapper      | `core/environments/my_domain/envs.py`       |
| 5   | Export in `__init__.py`              | `core/environments/my_domain/__init__.py`   |
| 6   | Register in `make_envs()`            | `core/environments/env_manager.py`          |
| 7   | Implement `EnvironmentManager`       | `core/environments/env_manager.py`          |
| 8   | Add prompt templates                 | `core/environments/prompts/`                |
| 9   | Create data preprocessing script     | `data_preprocess/prepare_my_domain_data.py` |
| 10  | Create YAML config                   | `examples/configs/`                         |
| 11  | (Optional) Add evolving variant      | `core/environments/my_domain_evolving/`     |
