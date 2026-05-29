---
id: quick-start
title: Quick Start
description: Run AlphaApollo reasoning, learning, and evolution with either one-line commands or full pipeline scripts.
sidebar_position: 3
---

# Quick Start

This page maps the `examples/` scripts to copy-paste commands.

> **Tip**  
> Each workflow has two entry styles:
> - **Method 1**: one-line workflow command (fast and minimal)
> - **Method 2**: script entrypoint (for explicit pipeline flow)

## Agentic Reasoning (test)

### Method 1: one-line workflow entrypoint

```bash
# no-tool reasoning
python3 -m alphaapollo.workflows.test \
  --model.path=Qwen/Qwen2.5-3B-Instruct \
  --preprocess.data_source=math-ai/aime24
```

```bash
# tool-integrated reasoning
python3 -m alphaapollo.workflows.test \
  --model.path=Qwen/Qwen2.5-3B-Instruct \
  --preprocess.data_source=math-ai/aime24 \
  --env.informal_math.enable_python_code=true \
  --env.informal_math.enable_local_rag=false \
  --env.max_steps=4
```

```bash
# Select specific dataset samples (e.g., the 0th AIME test question) and test
python3 -m alphaapollo.workflows.test \
  --model.path=Qwen/Qwen2.5-3B-Instruct \
  --preprocess.module=alphaapollo.data_preprocess.prepare_custom_data \
  --preprocess.data_source=math-ai/aime24 \
  --preprocess.splits=test \
  --preprocess.sample_indices=0 \
  --data.path=~/data/custom_data/test.parquet
```

```bash
# Directly evaluate a plain text question (not from a dataset)
python3 -m alphaapollo.workflows.test \
  --model.path=Qwen/Qwen2.5-3B-Instruct \
  --preprocess.module=alphaapollo.data_preprocess.prepare_single_question \
  --preprocess.question_text="What is the sum of integers from 1 to 1000?" \
  --preprocess.ground_truth="500500" \
  --data.path=~/data/single_question/test.parquet
```

### Method 2: script entrypoints

```bash
bash examples/test/run_test_informal_math_no_tool.sh
```

```bash
bash examples/test/run_test_informal_math.sh
```

## Agentic Learning (SFT + RL)

### Method 1: one-line workflow entrypoint

```bash
# multi-turn SFT
python3 -m alphaapollo.workflows.sft \
  --model.partial_pretrain=Qwen/Qwen2.5-3B-Instruct \
  --preprocess.data_source=AI-MO/NuminaMath-TIR
```

```bash
# multi-turn RL
python3 -m alphaapollo.workflows.rl \
  --model.path=Qwen/Qwen2.5-3B-Instruct \
  --preprocess.data_source=HuggingFaceH4/MATH-500 \
  --algorithm.adv_estimator=grpo
```

### Method 2: full pipeline scripts (data prep + training)

```bash
bash examples/sft/run_sft_informal_math_no_tool.sh
```

```bash
bash examples/sft/run_sft_informal_math_tool.sh
```

```bash
bash examples/rl/run_rl_informal_math_no_tool.sh
```

```bash
bash examples/rl/run_rl_informal_math_tool.sh
```

> **Info**  
> The RL full scripts contain explicit preprocessing and trainer launch steps, while Method 1 uses the workflow module entrypoint.

## Self-Evolution (evo)

> **Warning — Required order**  
> For self-evolution, you **must** start model serving first; otherwise evo commands will fail.

### Step 1 (Terminal A): launch model service

```bash
python alphaapollo/utils/ray_serve_llm.py --model_path <model_path> --gpus <gpus> --port <port> --model_id <model_id>
```

Example:

```bash
python alphaapollo/utils/ray_serve_llm.py --model_path Qwen/Qwen3-4B-Instruct-2507 --gpus "4,5" --port 9876 --model_id qwen3_4b_inst
```

### Step 2 (Terminal B): run evolution

Method 1: one-line workflow entrypoint

```bash
# single-model evolution
python3 -m alphaapollo.workflows.evo \
  --preprocess.data_source=math-ai/aime24 \
  --run.dataset_name=aime24 \
  --policy_model_cfg.model_name=qwen3_4b_inst \
  --policy_model_cfg.base_url=http://localhost:8000/v1 \
  --verifier_cfg.model_name=qwen3_4b_inst \
  --verifier_cfg.base_url=http://localhost:8000/v1
```

Method 2: script entrypoints

```bash
bash examples/evo/run_evo_informal_math.sh
```

```bash
bash examples/evo/run_evo_informal_math_multi_models.sh
```

## Optional Demo

If you keep demo assets in your branch, you can also run terminal/web demos from `examples/demo/`.



