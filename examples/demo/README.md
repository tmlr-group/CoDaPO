# Terminal Demo: `informal_math_training`

This demo lets a user type a math question in terminal and run a multi-step loop with AlphaApollo's `informal_math_training` environment.

## Files

- `examples/demo/terminal_informal_math_training.py`
- `examples/configs/demo_terminal_vllm.yaml`
- `examples/configs/demo_terminal_api.yaml`
- `examples/demo/run_terminal_demo_vllm.sh`
- `examples/demo/run_terminal_demo_api.sh`

## Run with vLLM

1. Start an OpenAI-compatible vLLM endpoint (example):

```bash
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-3B-Instruct --port 8000
```

2. Run demo:

```bash
bash examples/demo/run_terminal_demo_vllm.sh
```

### Run Local Web UI with vLLM

```bash
MODE=web bash examples/demo/run_terminal_demo_vllm.sh
```

Open `http://127.0.0.1:7860` by default.

See `examples/demo/web_ui/README.md` for full options.

## Run with external API (NOT TESTING YET)

```bash
export OPENAI_API_KEY=your_key_here
bash examples/demo/run_terminal_demo_api.sh
```

## Optional overrides

```bash
python3 examples/demo/terminal_informal_math_training.py \
  --config examples/configs/demo_terminal_vllm.yaml \
  --model Qwen/Qwen2.5-3B-Instruct \
  --base-url http://localhost:8000/v1 \
  --max-steps 8 \
  --show-full-prompts
```

If your vLLM server uses a custom served model name, pass it through:

```bash
VLLM_MODEL=Qwen/Qwen2.5-3B-Instruct \
VLLM_MODEL_NAME=my-served-name \
bash examples/demo/run_terminal_demo_vllm.sh
```

Type `exit` or `quit` to stop.
