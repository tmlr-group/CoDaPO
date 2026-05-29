---
sidebar_label: "Dataset Pipeline"
sidebar_position: 3
---

# Dataset Pipeline

AlphaApollo provides a set of preprocessing scripts that download datasets from HuggingFace Hub, normalize them into a unified schema, and output parquet files ready for each workflow (evolving, RL training, RL validation, SFT). All scripts live in `alphaapollo/data_preprocess/`.

## Common Pipeline

Every preprocessing script follows the same pattern:

```text
HuggingFace Hub  →  Field Extraction  →  Normalization  →  Parquet Output
```

### Adaptive Field Extraction

Datasets from different sources use different column names. The scripts handle this with fallback key lists:

```python
QUESTION_KEYS    = ["question", "problem", "prompt", "Problem", "instruction"]
GROUND_TRUTH_KEYS = ["answer", "solution", "ground_truth", "final_answer", "target", "boxed_answer"]
SOLUTION_KEYS     = ["solution", "detailed_solution", "rationale", "chain_of_thought", "cot"]
```

For each example, the script tries each key in order and uses the first match.

### Answer Extraction

`extract_solution(solution_str)` extracts the final answer from a solution string by locating the last `\boxed{...}` expression and handling nested braces correctly. Internally it delegates to `verl.utils.reward_score.math.last_boxed_only_string` and `remove_boxed`. This is used for both ground-truth answers and model outputs.

### Normalization

`_normalise_text(value)` converts raw values to clean strings, handling multiple input types:
- `None` → empty string
- `str` → stripped string  
- `list` / `tuple` → newline-joined items
- `dict` → JSON serialization

`_filter_metadata(example, used_keys)` strips already-extracted fields from the raw example, keeping only unknown metadata for downstream inspection.

`process_example(example, data_source)` maps a raw HuggingFace example into the target schema, applying field extraction, answer extraction, and metadata tagging.

## Preprocessing Scripts

### Evolving Data

**Script**: `alphaapollo/data_preprocess/prepare_evolving_data.py`

**Purpose**: Prepares data for the self-evolution workflow.

**Default data source**: `math-ai/aime24`

**Output schema**:

| Field | Type | Description |
| --- | --- | --- |
| `data_source` | str | Origin dataset identifier |
| `prompt` | list | Chat-format prompt `[{role, content}]` |
| `ability` | str | Task category (e.g., `"math"`) |
| `reward_model` | dict | Reward config `{style, ground_truth}` |
| `extra_info` | dict | Additional metadata |
| `metadata` | dict | Source metadata |
| `env_kwargs` | dict | Environment kwargs `{question, ground_truth}` |

**Usage**:

```bash
python3 -m alphaapollo.data_preprocess.prepare_evolving_data \
  --data_source math-ai/aime24 \
  --local_dir ./data
```

:::tip Local data & HDFS
All preprocessing scripts also accept:
- **Local paths**: if `--data_source` points to an existing local directory, it loads parquet files from there instead of downloading from HuggingFace Hub.
- **`--hdfs_dir`**: if provided, processed files are mirrored to the specified HDFS directory via `verl.utils.hdfs_io`.
:::

### RL Training Data

**Script**: `alphaapollo/data_preprocess/prepare_rl_training_data.py`

**Purpose**: Prepares data for reinforcement learning training.

**Default data source**: `math-ai/aime24`

**Output schema**: Same as evolving data — contains `prompt`, `reward_model`, `env_kwargs`, etc.

**Usage**:

```bash
python3 -m alphaapollo.data_preprocess.prepare_rl_training_data \
  --data_source math-ai/aime24 \
  --local_dir ./data
```

:::note
`prepare_evolving_data.py` and `prepare_rl_training_data.py` share the same field-extraction logic and output schema. They are intentionally kept as separate scripts for workflow isolation, but their core processing pipeline is identical.
:::

### RL Validation Data

**Script**: `alphaapollo/data_preprocess/prepare_rl_validation_data.py`

**Purpose**: Prepares validation sets for evaluating RL-trained models.

**Default data source**: `math-ai/aime24`

**Key difference**: Supports **multiple data sources** via the `--data_sources` argument (accepts a list), and includes a `metadata` string field for downstream analysis.

**Usage**:

```bash
python3 -m alphaapollo.data_preprocess.prepare_rl_validation_data \
  --data_sources math-ai/aime24 math-ai/aime25 \
  --local_dir ./data
```

### SFT Data (No Tool)

**Script**: `alphaapollo/data_preprocess/prepare_sft_no_tool.py`

**Purpose**: Prepares data for vanilla supervised fine-tuning (no tool use).

**Output schema** (simplified):

| Field | Type | Description |
| --- | --- | --- |
| `question` | str | System prompt + question text |
| `answer` | str | Ground-truth trajectory (`gt_traj`) |

The system prompt does **not** mention any tools — the model is expected to reason purely in text.

**Usage**:

```bash
python3 -m alphaapollo.data_preprocess.prepare_sft_no_tool \
  --data_source math-ai/aime24 \
  --local_dir ./data
```

### SFT Data (With Tool)

**Script**: `alphaapollo/data_preprocess/prepare_sft_tool.py`

**Purpose**: Prepares data for multi-turn SFT with tool-use demonstrations.

**Output schema** (chat format):

| Field | Type | Description |
| --- | --- | --- |
| `messages` | list | `[{role: "system", content: ...}, {role: "user", content: ...}, {role: "assistant", content: ...}]` |

This follows the standard chat-format expected by most fine-tuning frameworks.

**Usage**:

```bash
python3 -m alphaapollo.data_preprocess.prepare_sft_tool \
  --data_source math-ai/aime24 \
  --local_dir ./data
```

## Output Data Example

A single processed record looks like this (JSON representation of the parquet row):

```json
{
  "data_source": "dummy_question",
  "prompt": [
    {"role": "user", "content": "Find all prime factors of 120."}
  ],
  "ability": "math",
  "reward_model": {"style": "rule", "ground_truth": "2, 3, 5"},
  "extra_info": {"split": "train", "index": 0, "question": "Find all prime factors of 120.", "ground_truth": "2, 3, 5", "gt_traj": "", "data_source": "dummy_question"},
  "metadata": null,
  "env_kwargs": {"question": "Find all prime factors of 120.", "ground_truth": "2, 3, 5", "gt_traj": "", "data_source": "dummy_question"}
}
```

## Output Schema Comparison

| Script | Key Output Fields | Format | Tool Prompt |
| --- | --- | --- | --- |
| `prepare_evolving_data` | prompt, reward_model, env_kwargs | Structured dict | Yes |
| `prepare_rl_training_data` | prompt, reward_model, env_kwargs | Structured dict | Yes |
| `prepare_rl_validation_data` | prompt, reward_model, env_kwargs, metadata | Structured dict | Yes |
| `prepare_sft_no_tool` | question, answer | Simple key-value | No |
| `prepare_sft_tool` | messages | Chat messages list | No |

## Dependencies

The preprocessing scripts depend on:
- `datasets` — HuggingFace `load_dataset`
- `pandas` — Parquet serialization
- `verl.utils.hdfs_io` — HDFS mirroring (optional)
- `verl.utils.reward_score.math` — `\boxed{}` answer extraction

## Integration with Workflows

The preprocessing scripts are automatically invoked by the workflow system. When running a workflow (e.g., `alphaapollo.workflows.rl`), the `preprocess` section of the YAML config specifies which scripts to run:

```yaml
preprocess:
  - module: alphaapollo.data_preprocess.prepare_rl_training_data
    args:
      data_source: DigitalLearningGmbH/MATH-lighteval
      local_dir: ./data
  - module: alphaapollo.data_preprocess.prepare_rl_validation_data
    args:
      data_source: HuggingFaceH4/MATH-500
      splits: test
      local_dir: ./data
```

The workflow API (`alphaapollo/workflows/api.py`) loads the config, runs each preprocessing module, and then launches the trainer.

## Custom Data Sources

To use a custom dataset:

1. Host it on HuggingFace Hub (or use a local path).
2. Ensure it contains columns matching at least one key in `QUESTION_KEYS` and `GROUND_TRUTH_KEYS`.
3. Pass the dataset identifier via `--data_source`:

```bash
python3 -m alphaapollo.data_preprocess.prepare_rl_training_data \
  --data_source your-org/your-dataset \
  --local_dir ./data
```

The adaptive field extraction will automatically detect the correct columns.

For guidance on creating a fully custom environment that consumes its own data format, see [new-environment.md](../contribution/new-environment.md).
