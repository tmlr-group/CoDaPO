---
sidebar_position: 1
---

# Contribution Guide

This section provides step-by-step guides for extending AlphaApollo with new components. Whether you want to add a new tool, plug in a new task domain, or implement a new training algorithm, these guides walk you through the full process.

## Extension Points

| Guide                                            | What You'll Build                                                                    | Difficulty      |
| ------------------------------------------------ | ------------------------------------------------------------------------------------ | --------------- |
| [Adding a New Tool](./new-tool.md)               | A custom tool the LLM can invoke during reasoning (e.g., web search, database query) | ⭐ Easy         |
| [Adding a New Environment](./new-environment.md) | A new task domain with its own reward, parsing, and projection logic                 | ⭐⭐ Medium     |
| [Adding a New Algorithm](./new-algorithm.md)     | A new training or inference workflow with CLI entry point and YAML config            | ⭐⭐⭐ Advanced |

## Prerequisites

Before diving into these guides, make sure you are familiar with:

- [Agent System](../core-modules/agent-system.md) — The layered environment architecture
- [Tools](../core-modules/tools.md) — The decorator-based tool framework
- [Self-Evolution](../core-modules/evolution.md) — The self-improvement pipeline
- [Configuration](../configuration/index.md) — Hydra configs and YAML structure

## Design Principles

AlphaApollo follows several conventions that all extensions should respect:

1. **Gym-style interface** — Environments implement `reset()` / `step()` / `close()`.
2. **Decorator-based registration** — Tools use `@tool`; no manual registry needed.
3. **XML tag protocol** — The LLM communicates tool calls via `<tool_name>...</tool_name>` tags.
4. **Config-driven** — All behavior is controlled by YAML configs with CLI overrides.
5. **Workflow pattern** — Every algorithm is a workflow: config → preprocess → train/infer.
