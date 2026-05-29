---
sidebar_label: "Troubleshooting"
sidebar_position: 3
---

# Troubleshooting & FAQ

Common issues and solutions when working with AlphaApollo.

### Network errors (ModelScope / Hugging Face download failures)

Use a stable network + proxy settings first.

- configure `http_proxy`, `https_proxy`, `no_proxy`
- pre-download models to local cache, then run training/eval

### OOM solutions

Try these first:

- reduce batch sizes / rollout group size / sequence lengths
- if still OOM, use a smaller model or fewer concurrent workers

### FlashAttention (`flash-attn`) error

Recommended approach: install from a matching release wheel.

- download a `flash-attn` wheel from the official release that matches your **PyTorch + CUDA + Python**
- install it directly, for example:

```bash
pip install /path/to/flash_attn-<version>-<python>-<platform>.whl
```
