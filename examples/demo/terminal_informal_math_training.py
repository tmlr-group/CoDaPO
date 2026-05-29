#!/usr/bin/env python3
"""Interactive terminal demo for informal_math_training environment.

This demo supports two LLM backends:
- vLLM (OpenAI-compatible local endpoint)
- API (hosted OpenAI-compatible endpoint)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

from omegaconf import OmegaConf
from openai import OpenAI

# Ensure repository root is on PYTHONPATH so local packages can be imported.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alphaapollo.core.environments.env_manager import InformalMathTrainingEnvironmentManager
from alphaapollo.core.environments.informal_math_training import (
    build_informal_math_training_envs,
    informal_math_training_projection,
)


class LLMClient:
    def __init__(self, model_name: str, base_url: str, api_key: str, temperature: float, max_tokens: int):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=300.0,
            max_retries=3,
        )

    def generate(self, user_prompt: str, system_prompt: str = "") -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            n=1,
            stop=None,
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""


def merge_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = {
        "backend": args.backend,
        "llm": {
            "model_name": args.model,
            "base_url": args.base_url,
            "api_key": args.api_key,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "system_prompt": args.system_prompt,
        },
        "env": {
            "seed": args.seed,
            "max_steps": args.max_steps,
            "history_length": args.history_length,
            "memory_type": args.memory_type,
            "enable_python_code": args.enable_python_code,
            "enable_local_rag": args.enable_local_rag,
            "python_code_timeout": args.python_code_timeout,
            "log_requests": args.log_requests,
        },
        "demo": {
            "ground_truth": args.ground_truth,
            "data_source": args.data_source,
            "show_full_prompts": args.show_full_prompts,
        },
    }

    if args.config:
        file_cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
        cfg = OmegaConf.to_container(OmegaConf.merge(OmegaConf.create(file_cfg), OmegaConf.create(cfg)), resolve=True)

    env_api_key = os.environ.get("OPENAI_API_KEY")
    if not cfg["llm"]["api_key"]:
        cfg["llm"]["api_key"] = env_api_key or "EMPTY"

    if not cfg["llm"]["base_url"]:
        if cfg["backend"] == "vllm":
            cfg["llm"]["base_url"] = "http://localhost:8000/v1"
        else:
            cfg["llm"]["base_url"] = "https://api.openai.com/v1"

    return cfg


def build_manager(cfg: Dict[str, Any]) -> InformalMathTrainingEnvironmentManager:
    # Env constructor reads env_config.max_steps + env_config.informal_math_training.
    env_runtime_cfg = OmegaConf.create(
        {
            "max_steps": cfg["env"]["max_steps"],
            "informal_math_training": {
                "enable_python_code": cfg["env"]["enable_python_code"],
                "enable_local_rag": cfg["env"]["enable_local_rag"],
                "python_code_timeout": cfg["env"]["python_code_timeout"],
                "log_requests": cfg["env"]["log_requests"],
            },
        }
    )

    envs = build_informal_math_training_envs(
        seed=cfg["env"]["seed"],
        env_num=1,
        group_n=1,
        is_train=False,
        env_config=env_runtime_cfg,
    )

    # Manager currently reads config.env.informal_math.* in the training class.
    manager_cfg = OmegaConf.create(
        {
            "env": {
                "env_name": "informal_math_training",
                "history_length": cfg["env"]["history_length"],
                "max_steps": cfg["env"]["max_steps"],
                "informal_math": {
                    "memory_type": cfg["env"]["memory_type"],
                    "enable_python_code": cfg["env"]["enable_python_code"],
                    "enable_local_rag": cfg["env"]["enable_local_rag"],
                    "execution_mode": "agentic",
                },
            }
        }
    )

    return InformalMathTrainingEnvironmentManager(envs, informal_math_training_projection, manager_cfg)


def format_block(title: str, content: str, full_text: bool = True, max_chars: int = 1200) -> str:
    text = content if content is not None else ""
    if not full_text and len(text) > max_chars:
        text = text[:max_chars] + "\n... [truncated]"
    return f"\n[{title}]\n{text}\n"


def run_episode(question: str, client: LLMClient, manager: InformalMathTrainingEnvironmentManager, cfg: Dict[str, Any]) -> None:
    reset_kwargs = [
        {
            "question": question,
            "ground_truth": cfg["demo"]["ground_truth"],
            "gt_traj": "",
            "data_source": cfg["demo"]["data_source"],
        }
    ]

    obs, _ = manager.reset(kwargs=reset_kwargs)
    done = False
    step_id = 0
    reward = 0.0

    while not done and step_id < cfg["env"]["max_steps"]:
        step_id += 1
        prompt_text = obs["text"][0]
        print(format_block(f"Step {step_id} Prompt", prompt_text, cfg["demo"]["show_full_prompts"]))

        action = client.generate(prompt_text, cfg["llm"]["system_prompt"])
        print(format_block(f"Step {step_id} Model Action", action, True))

        obs, rewards, dones, infos = manager.step([action])
        reward = float(rewards[0])
        done = bool(dones[0])
        info = infos[0] if infos else {}

        is_valid = info.get("is_action_valid")
        print(f"[Step {step_id} Env] done={done}, reward={reward}, is_action_valid={is_valid}")

        anchor = obs.get("anchor", [""])[0] if isinstance(obs, dict) else ""
        if anchor:
            print(format_block(f"Step {step_id} Tool/Env Feedback", anchor, True))

    print(f"\n[Episode End] total_steps={step_id}, final_reward={reward}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive terminal demo for informal_math_training.")
    parser.add_argument("--config", type=str, default="", help="Optional YAML config path.")
    parser.add_argument("--backend", choices=["vllm", "api"], default="vllm")

    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--base-url", type=str, default="")
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--system-prompt", type=str, default="")

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--history-length", type=int, default=4)
    parser.add_argument("--memory-type", choices=["simple", "score", "ndimensional"], default="simple")
    parser.add_argument("--enable-python-code", action="store_true", default=True)
    parser.add_argument("--disable-python-code", dest="enable_python_code", action="store_false")
    parser.add_argument("--enable-local-rag", action="store_true", default=False)
    parser.add_argument("--python-code-timeout", type=int, default=30)
    parser.add_argument("--log-requests", action="store_true", default=False)

    parser.add_argument("--ground-truth", type=str, default="\\boxed{0}", help="Placeholder GT for demo reward calculation.")
    parser.add_argument("--data-source", type=str, default="terminal_demo")
    parser.add_argument("--show-full-prompts", action="store_true", default=False)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = merge_config(args)

    client = LLMClient(
        model_name=cfg["llm"]["model_name"],
        base_url=cfg["llm"]["base_url"],
        api_key=cfg["llm"]["api_key"],
        temperature=cfg["llm"]["temperature"],
        max_tokens=cfg["llm"]["max_tokens"],
    )
    manager = build_manager(cfg)

    print("Terminal demo started. Type a math problem, or type 'exit' to quit.")
    print(f"backend={cfg['backend']} model={cfg['llm']['model_name']} base_url={cfg['llm']['base_url']}")

    try:
        while True:
            user_input = input("\nUser> ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                break
            run_episode(user_input, client, manager, cfg)
    finally:
        manager.envs.close()


if __name__ == "__main__":
    main()
