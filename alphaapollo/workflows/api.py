from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIGS = {
    "rl": "examples/configs/rl_informal_math_tool.yaml",
    "sft": "examples/configs/sft_informal_math_tool.yaml",
    "test": "examples/configs/test_informal_math_no_tool.yaml",
    "evo": "examples/configs/vllm_informal_math.yaml",
}


def to_plain(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def load_config(config_path: str) -> Dict[str, Any]:
    if not os.path.isabs(config_path):
        config_path = str(PROJECT_ROOT / config_path)
    cfg = OmegaConf.load(config_path)
    data = to_plain(cfg)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return data


def env_with_overrides(overrides: Optional[Dict[str, Any]]) -> Dict[str, str]:
    env = os.environ.copy()
    base_pythonpath = env.get("PYTHONPATH", "")
    pythonpath_entries = [
        str(PROJECT_ROOT),
        str(PROJECT_ROOT / "alphaapollo" / "core" / "generation"),
    ]
    if base_pythonpath:
        pythonpath_entries.append(base_pythonpath)
    env["PYTHONPATH"] = ":".join(pythonpath_entries)
    
    if not overrides:
        return env
    for key, value in overrides.items():
        env[str(key)] = str(value)
    return env


def normalize_cli_args(args: Any) -> List[str]:
    if args is None:
        return []
    if isinstance(args, dict):
        cli_args: List[str] = []
        for key, value in args.items():
            cli_args.append(f"--{key}")
            if value is None:
                continue
            if isinstance(value, (list, tuple)):
                cli_args.extend(str(x) for x in value)
            else:
                cli_args.append(str(value))
        return cli_args
    if isinstance(args, list):
        return [str(x) for x in args]
    raise ValueError(f"Unsupported CLI args format: {type(args)}")


def run_cmd(cmd: List[str], env: Dict[str, str]) -> None:
    print(f"[alphaapollo] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT), env=env)


def run_modules(steps: Iterable[Dict[str, Any]], env: Dict[str, str]) -> None:
    for step in steps:
        module = step.get("module")
        if not module:
            raise ValueError("Each preprocess step must define `module`.")
        args = normalize_cli_args(step.get("args"))
        cmd = [sys.executable, "-m", str(module), *args]
        run_cmd(cmd, env=env)


def run_trainer(runner: Dict[str, Any], env: Dict[str, str], extra_overrides: Optional[List[str]]) -> None:
    launcher = runner.get("launcher", "python")
    module = runner.get("module")
    if not module:
        raise ValueError("Runner config must define `module`.")

    overrides = [str(x) for x in (runner.get("overrides") or [])]
    if extra_overrides:
        overrides.extend(extra_overrides)

    if launcher == "python":
        cmd = [sys.executable, "-m", str(module), *overrides]
    elif launcher == "torchrun":
        torchrun_cfg = runner.get("torchrun") or {}
        standalone = bool(torchrun_cfg.get("standalone", True))
        nnodes = int(torchrun_cfg.get("nnodes", 1))
        nproc_per_node = int(torchrun_cfg.get("nproc_per_node", 1))
        cmd = ["torchrun"]
        if standalone:
            cmd.append("--standalone")
        cmd.extend([f"--nnodes={nnodes}", f"--nproc_per_node={nproc_per_node}", "-m", str(module), *overrides])
    else:
        raise ValueError(f"Unsupported launcher: {launcher}")

    run_cmd(cmd, env=env)


def extract_preprocess_overrides(
    preprocess_steps: List[Dict[str, Any]],
    extra_overrides: Optional[List[str]],
) -> List[str]:
    if not extra_overrides:
        return []

    def _set_preprocess_arg(arg_key: str, arg_value: Any) -> None:
        for step in preprocess_steps:
            args = step.setdefault("args", {})
            if isinstance(args, dict):
                args[arg_key] = arg_value

    remaining: List[str] = []
    for override in extra_overrides:
        module_match = re.match(r"^preprocess\.(.+?)=(.*)$", override)
        if module_match:
            lhs = module_match.group(1)
            value = module_match.group(2)

            # Support module-scoped overrides:
            # preprocess.<module_path>.<arg_key>=value
            # e.g. preprocess.alphaapollo.data_preprocess.prepare_rl_training_data.data_source=...
            module_key_matched = False
            for step in preprocess_steps:
                module = str(step.get("module", ""))
                prefix = f"{module}."
                if lhs.startswith(prefix):
                    arg_key = lhs[len(prefix):]
                    if arg_key:
                        args = step.setdefault("args", {})
                        if isinstance(args, dict):
                            args[arg_key] = value
                            module_key_matched = True
            if module_key_matched:
                continue

        step_match = re.match(r"^preprocess\.(\d+)\.(.+?)=(.*)$", override)
        if step_match:
            step_idx = int(step_match.group(1))
            key = step_match.group(2)
            value = step_match.group(3)
            if step_idx < 0 or step_idx >= len(preprocess_steps):
                raise ValueError(
                    f"Invalid preprocess step index in override: {override}. "
                    f"Available steps: 0..{max(len(preprocess_steps) - 1, 0)}"
                )
            if key == "module":
                preprocess_steps[step_idx]["module"] = value
            else:
                args = preprocess_steps[step_idx].setdefault("args", {})
                if isinstance(args, dict):
                    args[key] = value
            continue

        if override.startswith("preprocess.module="):
            value = override.split("=", 1)[1]
            for step in preprocess_steps:
                step["module"] = value
            continue

        if override.startswith("preprocess.data_source="):
            value = override.split("=", 1)[1]
            _set_preprocess_arg("data_source", value)
            continue

        if override.startswith("preprocess.data_sources="):
            value = override.split("=", 1)[1]
            values = [x for x in value.split(",") if x]
            _set_preprocess_arg("data_sources", values)
            continue

        if override.startswith("preprocess."):
            raw_key, value = override.split("=", 1)
            arg_key = raw_key[len("preprocess.") :]
            _set_preprocess_arg(arg_key, value)
            continue

        remaining.append(override)

    return remaining


def run_standard_workflow(config_path: str, extra_overrides: Optional[List[str]]) -> None:
    cfg = load_config(config_path)
    env = env_with_overrides(cfg.get("env"))
    preprocess_steps = cfg.get("preprocess", [])
    runner_overrides = extract_preprocess_overrides(preprocess_steps, extra_overrides)
    run_modules(preprocess_steps, env=env)
    run_trainer(cfg["runner"], env=env, extra_overrides=runner_overrides)


def rl(
    config_path: str = DEFAULT_CONFIGS["rl"],
    extra_overrides: Optional[List[str]] = None,
) -> None:
    run_standard_workflow(config_path=config_path, extra_overrides=extra_overrides)


def sft(
    config_path: str = DEFAULT_CONFIGS["sft"],
    extra_overrides: Optional[List[str]] = None,
) -> None:
    run_standard_workflow(config_path=config_path, extra_overrides=extra_overrides)


def test(
    config_path: str = DEFAULT_CONFIGS["test"],
    extra_overrides: Optional[List[str]] = None,
) -> None:
    run_standard_workflow(config_path=config_path, extra_overrides=extra_overrides)


def evo(
    config_path: str = DEFAULT_CONFIGS["evo"],
    extra_overrides: Optional[List[str]] = None,
) -> None:
    cfg = load_config(config_path)
    env = env_with_overrides(cfg.get("env"))
    preprocess_steps = cfg.get("preprocess", [])
    remaining_overrides = extract_preprocess_overrides(preprocess_steps, extra_overrides)
    run_modules(preprocess_steps, env=env)
    module = cfg.get("entrypoint_module", "alphaapollo.core.generation.evolving.evolving_main")
    if not remaining_overrides:
        run_cmd([sys.executable, "-m", module, "--config", config_path], env=env)
        return
    config_file = config_path if os.path.isabs(config_path) else str(PROJECT_ROOT / config_path)
    merged = OmegaConf.merge(OmegaConf.load(config_file), OmegaConf.from_dotlist(remaining_overrides))
    fd, temp_config = tempfile.mkstemp(prefix="alphaapollo_evo_", suffix=".yaml")
    os.close(fd)
    try:
        OmegaConf.save(merged, temp_config)
        run_cmd([sys.executable, "-m", module, "--config", temp_config], env=env)
    finally:
        if os.path.exists(temp_config):
            os.remove(temp_config)

