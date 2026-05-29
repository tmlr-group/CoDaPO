#!/usr/bin/env bash
set -euo pipefail

VLLM_PID_FILE="${VLLM_PID_FILE:-/tmp/alphaapollo_vllm.pid}"
WEB_PID_FILE="${WEB_PID_FILE:-/tmp/alphaapollo_web.pid}"

stop_from_pid_file() {
  local label="$1"
  local file="$2"
  if [[ -f "$file" ]]; then
    local pid
    pid="$(cat "$file" || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" || true
      echo "Stopped $label PID=$pid"
    else
      echo "$label not running (pid file: $file)"
    fi
    rm -f "$file"
  else
    echo "$label pid file not found: $file"
  fi
}

stop_from_pid_file "web" "$WEB_PID_FILE"
stop_from_pid_file "vllm" "$VLLM_PID_FILE"
