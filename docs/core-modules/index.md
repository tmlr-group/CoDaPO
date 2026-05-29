---
sidebar_position: 3
---

# Core Modules

This section explains how the AlphaApollo system is built — the internal architecture, key abstractions, and how components interact at runtime.

## Overview

AlphaApollo is organized around four core pillars:

| Module | Description | Key Directory |
|---|---|---|
| [Agent System](./agent-system.md) | The environment-driven, multi-turn agentic reasoning loop | `alphaapollo/core/environments/` |
| [Self-Evolution](./evolution.md) | Iterative policy-verifier self-improvement at inference time | `alphaapollo/core/generation/evolving/` |
| [Dataset Pipeline](./dataset.md) | Data preprocessing scripts for all workflows | `alphaapollo/data_preprocess/` |
| [Tools](./tools.md) | Extensible tool framework for code execution, verification, and RAG | `alphaapollo/core/tools/` |

## Architecture at a Glance

```
┌──────────────────────────────────────────────────────┐
│                   AlphaApollo                        │
│                                                      │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Tools     │  │   Prompts    │  │   Memory     │ │
│  │ (code, RAG) │  │ (templates,  │  │ (simple,     │ │
│  │             │  │  formatting) │  │  score, ND)  │ │
│  └──────┬──────┘  └──────┬───────┘  └──────┬───────┘ │
│         │                │                 │         │
│         └────────────────┼─────────────────┘         │
│                          ▼                           │
│              ┌───────────────────────┐               │
│              │    Environment Loop   │               │
│              │  (Gym-style step/     │               │
│              │   reset interface)    │               │
│              └───────────┬───────────┘               │
│                          │                           │
│              ┌───────────┴───────────┐               │
│              ▼                       ▼               │
│   ┌──────────────────┐   ┌──────────────────┐        │
│   │   RL Training    │   │   Self-Evolution │        │
│   │  (PPO, GRPO,     │   │  (policy-verifier│        │
│   │   DAPO)          │   │   loops)         │        │
│   └──────────────────┘   └──────────────────┘        │
└──────────────────────────────────────────────────────┘
```

## Related Pages

- [Algorithms](../algorithms/index.md) — Training and inference pipelines
- [Configuration](../configuration/index.md) — YAML config reference
- [Adding a New Tool](../contribution/new-tool.md) — Extend the tool framework
- [Adding a New Environment](../contribution/new-environment.md) — Plug in a new domain
- [Adding a New Algorithm](../contribution/new-algorithm.md) — Create a new workflow
