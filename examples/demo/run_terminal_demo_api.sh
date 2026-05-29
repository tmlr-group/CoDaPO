#!/usr/bin/env bash
set -euo pipefail

: "${OPENAI_API_KEY:?Please set OPENAI_API_KEY before running API demo.}"

python3 examples/demo/terminal_informal_math_training.py \
  --config examples/configs/demo_terminal_api.yaml
