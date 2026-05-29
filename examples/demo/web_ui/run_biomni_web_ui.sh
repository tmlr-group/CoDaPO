#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible wrapper name; now launches the local examples/demo web UI.
MODE=web bash examples/demo/run_terminal_demo_vllm.sh "$@"
