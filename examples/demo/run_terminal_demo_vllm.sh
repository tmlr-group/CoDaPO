#!/usr/bin/env bash
set -euo pipefail

# Unified launcher for vLLM + local demos in examples/demo.
# Modes:
#   MODE=terminal -> run terminal demo
#   MODE=web      -> run local web demo (no Biomni-Web dependency)
MODE="${MODE:-terminal}"
START_VLLM="${START_VLLM:-0}"
DETACH="${DETACH:-0}"

# vLLM settings
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
# The model identifier used by chat completions. Defaults to the served model name.
VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-${VLLM_MODEL}}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_API_BASE="${VLLM_API_BASE:-http://${VLLM_HOST}:${VLLM_PORT}/v1}"
VLLM_LOG="${VLLM_LOG:-/tmp/vllm_terminal_demo.log}"
WAIT_SECS="${WAIT_SECS:-180}"

# Local web demo settings
WEB_HOST="${WEB_HOST:-127.0.0.1}"
WEB_PORT="${WEB_PORT:-7860}"
WEB_LOG="${WEB_LOG:-/tmp/alphaapollo_web_demo.log}"
VLLM_PID_FILE="${VLLM_PID_FILE:-/tmp/alphaapollo_vllm.pid}"
WEB_PID_FILE="${WEB_PID_FILE:-/tmp/alphaapollo_web.pid}"

VLLM_PID=""
WEB_PID=""

# vllm serve Qwen/Qwen2.5-3B-Instruct

kill_listeners_on_port() {
  local port="$1"
  local label="$2"
  local pids=""

  # Prefer lsof for portability in local dev shells.
  if command -v lsof >/dev/null 2>&1; then
    pids="$(lsof -tiTCP:"${port}" -sTCP:LISTEN 2>/dev/null | tr '\n' ' ' | xargs echo -n || true)"
  elif command -v fuser >/dev/null 2>&1; then
    pids="$(fuser -n tcp "${port}" 2>/dev/null | tr '\n' ' ' | xargs echo -n || true)"
  fi

  if [[ -z "${pids}" ]]; then
    return 0
  fi

  echo "[run_terminal_demo_vllm] Found existing ${label} listener(s) on port ${port}: ${pids}"
  kill ${pids} 2>/dev/null || true

  local deadline=$((SECONDS + 5))
  while (( SECONDS < deadline )); do
    sleep 0.5
    local alive=()
    local pid=""
    for pid in ${pids}; do
      if kill -0 "${pid}" 2>/dev/null; then
        alive+=("${pid}")
      fi
    done
    if (( ${#alive[@]} == 0 )); then
      echo "[run_terminal_demo_vllm] Cleared port ${port}."
      return 0
    fi
  done

  echo "[run_terminal_demo_vllm] Force killing stubborn process(es) on port ${port}: ${pids}"
  kill -9 ${pids} 2>/dev/null || true
}

cleanup() {
  if [[ -n "${WEB_PID}" ]] && kill -0 "${WEB_PID}" 2>/dev/null; then
    kill "${WEB_PID}" || true
  fi
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    kill "${VLLM_PID}" || true
  fi
}
trap cleanup EXIT

if [[ "${START_VLLM}" == "1" ]]; then
  echo "[run_terminal_demo_vllm] Starting vLLM in background..."
  python -m vllm.entrypoints.openai.api_server \
    --host "${VLLM_HOST}" \
    --port "${VLLM_PORT}" \
    --model "${VLLM_MODEL}" >"${VLLM_LOG}" 2>&1 &
  VLLM_PID=$!
  echo "${VLLM_PID}" > "${VLLM_PID_FILE}"
  echo "[run_terminal_demo_vllm] vLLM PID=${VLLM_PID}, log=${VLLM_LOG}"
fi

echo "[run_terminal_demo_vllm] Waiting for ${VLLM_API_BASE}/models ..."
for ((i=1; i<=WAIT_SECS; i++)); do
  if curl -fsS "${VLLM_API_BASE}/models" >/dev/null; then
    echo "[run_terminal_demo_vllm] vLLM is ready."
    break
  fi
  if [[ "$i" -eq "${WAIT_SECS}" ]]; then
    echo "[run_terminal_demo_vllm] ERROR: vLLM did not become ready in ${WAIT_SECS}s"
    echo "[run_terminal_demo_vllm] Check logs: ${VLLM_LOG}"
    exit 1
  fi
  sleep 1
done

if [[ "${MODE}" == "terminal" ]]; then
  python3 examples/demo/terminal_informal_math_training.py \
    --config examples/configs/demo_terminal_vllm.yaml \
    --model "${VLLM_MODEL_NAME}" \
    --base-url "${VLLM_API_BASE}"
  exit 0
fi

if [[ "${MODE}" == "web" ]]; then
  kill_listeners_on_port "${WEB_PORT}" "web"
  echo "[run_terminal_demo_vllm] Starting local web UI at http://${WEB_HOST}:${WEB_PORT}"
  echo "[run_terminal_demo_vllm] Web log: ${WEB_LOG}"
  python3 examples/demo/web_informal_math_training.py \
    --config examples/configs/demo_terminal_vllm.yaml \
    --model "${VLLM_MODEL_NAME}" \
    --host "${WEB_HOST}" \
    --port "${WEB_PORT}" \
    --base-url "${VLLM_API_BASE}" >"${WEB_LOG}" 2>&1 &
  WEB_PID=$!
  echo "${WEB_PID}" > "${WEB_PID_FILE}"
  echo "[run_terminal_demo_vllm] Web demo PID=${WEB_PID}"
  echo "[run_terminal_demo_vllm] Open: http://${WEB_HOST}:${WEB_PORT}"
  echo "[run_terminal_demo_vllm] PID files: ${WEB_PID_FILE} ${VLLM_PID_FILE}"
  if [[ "${DETACH}" == "1" ]]; then
    echo "[run_terminal_demo_vllm] Detached mode enabled; leaving processes in background."
    echo "[run_terminal_demo_vllm] Tail logs with:"
    echo "  tail -f ${WEB_LOG}"
    echo "  tail -f ${VLLM_LOG}"
    trap - EXIT
    exit 0
  fi
  echo "[run_terminal_demo_vllm] Press Ctrl+C to stop web demo and vLLM."
  wait "${WEB_PID}"
  exit 0
fi

echo "[run_terminal_demo_vllm] ERROR: Unsupported MODE=${MODE}. Use MODE=terminal or MODE=web."
exit 1
