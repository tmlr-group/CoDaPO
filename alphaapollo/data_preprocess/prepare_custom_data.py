# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from datasets import load_dataset

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed


# Matches GSM8K's "#### <number>" answer marker; same regex used by
# alphaapollo/core/environments/.../qwen_math.py.
_GSM8K_ANSWER_RE = re.compile(r"####\s*(-?[0-9\.\,]+)")


def extract_solution(solution_str):
    # MATH-style: "$\boxed{42}$"
    try:
        return remove_boxed(last_boxed_only_string(solution_str))
    except Exception:
        pass

    # GSM8K-style: "<reasoning>\n#### 42"
    gsm8k_matches = _GSM8K_ANSWER_RE.findall(solution_str)
    if gsm8k_matches:
        return gsm8k_matches[-1].replace(",", "").replace("$", "")

    print(f"No boxed/#### answer found, return the original solution string: {solution_str}")
    return str(solution_str)


# ---------------------------------------------------------------------------
# Defaults & logging
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

USER_PREFIX = ""
ABILITY = "math"

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

QUESTION_KEYS: List[str] = [
    "question",
    "problem",
    "prompt",
    "Problem",
    "instruction",
]
GROUND_TRUTH_KEYS: List[str] = [
    "answer",
    "ground_truth",
    "final_answer",
    "target",
    "boxed_answer",
    "solution",
]
SOLUTION_KEYS: List[str] = [
    "solution",
    "detailed_solution",
    "rationale",
    "chain_of_thought",
    "cot",
]


def _normalise_text(value: Any) -> str:
    """Convert a raw value to a clean string."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        items = [str(item).strip() for item in value if item is not None and str(item).strip()]
        return "\n".join(items)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _first_non_empty(example: Dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        if key in example:
            value = _normalise_text(example[key])
            if value:
                return value
    return ""


def _build_prompt(question: str) -> List[Dict[str, str]]:
    content = f"{USER_PREFIX}{question}".strip()
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def _filter_metadata(example: Dict[str, Any], used_keys: Iterable[str]) -> Dict[str, Any]:
    used = set(used_keys)
    return {k: v for k, v in example.items() if k not in used}


def _parse_sample_indices(sample_indices: str) -> Optional[List[int]]:
    sample_indices = sample_indices.strip()
    if not sample_indices:
        return None
    indices = [int(x.strip()) for x in sample_indices.split(",") if x.strip()]
    if any(i < 0 for i in indices):
        raise ValueError("sample indices must be non-negative")
    return indices


def process_example(
    example: Dict[str, Any],
    *,
    idx: int,
    split: str,
    data_source: str,
) -> Optional[Dict[str, Any]]:
    """Convert a raw dataset example into verl's expected schema."""

    question = _first_non_empty(example, QUESTION_KEYS)
    if not question:
        LOG.warning("Skipping index %s (split=%s): missing question field", idx, split)
        return None

    ground_truth = _first_non_empty(example, GROUND_TRUTH_KEYS)
    if not ground_truth:
        LOG.warning(
            "Skipping index %s (split=%s): missing ground-truth/answer field", idx, split
        )
        return None

    ground_truth = extract_solution(ground_truth)

    gt_traj = _first_non_empty(example, SOLUTION_KEYS)

    prompt = _build_prompt(question)

    extra_info = {
        "split": split,
        "index": idx,
        "question": question,
        "ground_truth": ground_truth,
        "gt_traj": gt_traj,
        "data_source": data_source,
    }

    env_kwargs = {
        "question": question,
        "ground_truth": ground_truth,
        "gt_traj": gt_traj,
        "data_source": data_source,
    }

    metadata = _filter_metadata(
        example,
        used_keys=set(QUESTION_KEYS + GROUND_TRUTH_KEYS + SOLUTION_KEYS),
    )
    metadata_str = json.dumps(metadata, ensure_ascii=False, sort_keys=True)

    record = {
        "data_source": data_source,
        "prompt": prompt,
        "ability": ABILITY,
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": extra_info,
        "metadata": metadata_str,
        "env_kwargs": env_kwargs,
    }

    return record


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and normalise informal math datasets."
    )
    parser.add_argument(
        "--data_source",
        default="math-ai/aime24",
        help="HuggingFace dataset repository ID (default: math-ai/aime24).",
    )
    parser.add_argument(
        "--splits",
        default="train,test",
        help="Comma separated list of dataset splits to export (default: train,test).",
    )
    parser.add_argument(
        "--local_dir",
        default="~/data",
        help="Directory to store the processed parquet files.",
    )
    parser.add_argument(
        "--hdfs_dir",
        default=None,
        help="Optional HDFS directory (if provided, processed files are mirrored there).",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity level.",
    )
    parser.add_argument(
        "--sample_indices",
        type=str,
        default="",
        help="Comma-separated sample indices, e.g. 0,1,2.",
    )
    return parser.parse_args()


def load_and_process_dataset(
    data_source: str,
    splits: List[str],
    args: argparse.Namespace,
    sample_indices: Optional[List[int]] = None,
) -> Dict[str, pd.DataFrame]:
    
    # Allow "<repo_id>:<config_name>" to disambiguate HF datasets that require a config
    # (e.g. "openai/gsm8k:main"). HF repo ids cannot contain ':', so splitting is safe.
    if ":" in data_source:
        repo_id, config_name = data_source.split(":", 1)
    else:
        repo_id, config_name = data_source, None

    LOG.info(
        "Loading dataset '%s'%s",
        repo_id,
        f" (config={config_name})" if config_name else "",
    )

    try:
        dataset_dict = load_dataset(repo_id, name=config_name)
    except Exception as e:
        LOG.error("Failed to load dataset '%s': %s", data_source, e)
        return {}
    
    split_dfs = {}
    
    for split in splits:
        if split not in dataset_dict:
            LOG.warning("Split '%s' not found in dataset %s", split, data_source)
            continue
            
        dataset = dataset_dict[split]
        source_indices = list(range(len(dataset)))
        if sample_indices is not None:
            selected_indices = [i for i in sample_indices if i < len(dataset)]
            if not selected_indices:
                continue
            dataset = dataset.select(selected_indices)
            source_indices = selected_indices
        records: List[Dict[str, Any]] = []
        
        for idx, example in enumerate(dataset):
            processed = process_example(
                example,
                idx=source_indices[idx],
                split=split,
                data_source=data_source,
            )
            if processed is not None:
                records.append(processed)
        
        if records:
            df = pd.DataFrame.from_records(records)
            split_dfs[split] = df
            LOG.info("Processed %d examples from %s (%s split)", len(df), data_source, split)
        else:
            LOG.warning("No valid examples were processed for %s (%s split)", data_source, split)
    
    return split_dfs


def main():
    args = parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    data_source = args.data_source
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    sample_indices = _parse_sample_indices(args.sample_indices)

    LOG.info("Processing data source: %s", data_source)
    LOG.info("Target splits: %s", splits)

    local_dir = os.path.expanduser(args.local_dir)
    split_dfs = load_and_process_dataset(
        data_source,
        splits,
        args,
        sample_indices=sample_indices,
    )
    output_dir = os.path.join(local_dir, "custom_data")
    os.makedirs(output_dir, exist_ok=True)

    processed_files: List[str] = []
    for split, df in split_dfs.items():
        output_file = os.path.join(output_dir, f"{split}.parquet")
        df.to_parquet(output_file, index=False)
        processed_files.append(output_file)
        LOG.info("Saved %s (%s split) to %s: %d examples",
                 data_source, split, output_file, len(df))

        info = {
            "split": split,
            "total_examples": len(df),
            "source_dataset": data_source,
            "generated_at": str(pd.Timestamp.now(tz='Asia/Shanghai'))
        }
        info_file = os.path.join(output_dir, f"{split}_info.json")
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)

    if not processed_files:
        LOG.warning("No files were written. Please check dataset availability and filters.")
        return

    if args.hdfs_dir:
        try:
            makedirs(args.hdfs_dir)
            copy(src=output_dir, dst=args.hdfs_dir)
            LOG.info("Copied processed files to HDFS: %s", args.hdfs_dir)
        except Exception as exc:
            LOG.error("Failed to copy files to HDFS: %s", exc)


if __name__ == "__main__":
    main()
