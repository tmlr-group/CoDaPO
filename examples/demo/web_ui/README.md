# Local Web Interface (No Biomni Dependency)

This web interface is implemented directly in `examples/demo` and uses:

- `examples/demo/web_informal_math_training.py`
- `alphaapollo/core/environments/informal_math_training`

## Quick start

```bash
MODE=web bash examples/demo/run_terminal_demo_vllm.sh
```

Shortcut:

```bash
bash examples/demo/web_ui/run_web_ui.sh
```

Open `http://127.0.0.1:7860` by default.

## Common overrides

```bash
MODE=web \
VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct \
VLLM_PORT=8000 \
WEB_PORT=7860 \
bash examples/demo/run_terminal_demo_vllm.sh
```

If your vLLM endpoint exposes a different served model id, set:

```bash
MODE=web VLLM_MODEL_NAME=your-served-model-id bash examples/demo/run_terminal_demo_vllm.sh
```

If vLLM is already running:

```bash
MODE=web START_VLLM=0 VLLM_API_BASE=http://127.0.0.1:8000/v1 \
bash examples/demo/run_terminal_demo_vllm.sh
```

## Logs

- vLLM: `/tmp/vllm_terminal_demo.log`
- Web demo: `/tmp/alphaapollo_web_demo.log`
