from __future__ import annotations

import argparse
from typing import List


def build_standard_parser(description: str, default_config: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        default=default_config,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "extra_overrides",
        nargs="*",
        help="Optional trainer overrides, e.g. trainer.total_epochs=2",
    )
    return parser


def normalize_unknown_overrides(tokens: List[str]) -> List[str]:
    overrides: List[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if not token.startswith("--"):
            overrides.append(token)
            i += 1
            continue
        key = token[2:]
        if "=" in key:
            overrides.append(key)
            i += 1
            continue
        if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            overrides.append(f"{key}={tokens[i + 1]}")
            i += 2
        else:
            i += 1

    return overrides


def parse_standard_args(description: str, default_config: str) -> tuple[str, List[str]]:
    parser = build_standard_parser(description, default_config)
    args, unknown = parser.parse_known_args()
    return args.config, [*args.extra_overrides, *normalize_unknown_overrides(unknown)]


