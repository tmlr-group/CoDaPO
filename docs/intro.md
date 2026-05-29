---
sidebar_position: 1
slug: /intro
---

# Welcome to AlphaApollo

**AlphaApollo** is a flexible, efficient, and production-ready RL training framework for LLM post-training. It follows the [HybridFlow](https://arxiv.org/pdf/2409.19256) architecture and adds project-specific extensions.

## Why AlphaApollo?

AlphaApollo is designed to make RL training for LLMs accessible, flexible, and scalable:

### 🚀 Easy to Use

- **Simple API**: Build complex RL dataflows with just a few lines of code
- **Diverse RL Algorithms**: Support for PPO, GRPO, and more (via [verl](https://github.com/volcengine/verl) integration)
- **Ready-to-Use Examples**: Out-of-the-box scripts for various environments

### 🔧 Flexible & Modular

- **Seamless Integration**: Works with PyTorch FSDP, Megatron-LM, vLLM
- **Customizable Components**: Easy to extend with custom environments, rewards, and memory systems
- **Flexible Device Mapping**: Efficient resource utilization across different cluster sizes

## Quick Start

### Installation

```bash
conda create -n alphaapollo python==3.12 -y
conda activate alphaapollo

git clone https://github.com/tmlr-group/AlphaApollo.git
cd AlphaApollo

bash installation.sh
```

### Run Your First Training

Check out our examples for different workflows:

```bash
# RL Training (with tool use)
cd examples/rl
bash run_rl_informal_math_tool.sh

# Self-Evolution
cd examples/evo
bash run_evo_informal_math.sh

# SFT (with tool use)
cd examples/sft
bash run_sft_informal_math_tool.sh
```

## Architecture

AlphaApollo uses a hybrid architecture that enables:

1. **Flexible Dataflow**: Define complex RL training pipelines
2. **Efficient Execution**: Optimize computation across multiple GPUs
3. **Modular Design**: Easy to customize and extend components

## What's Next?

- [Installation Guide](./getting-started/Installation.md) - Detailed setup instructions
- [Quick Start Tutorial](./getting-started/quick-start.md) - Your first AlphaApollo training job
- [Core Modules](./core-modules/index.md) - Agent system, evolution, data pipeline, and tools
- [Algorithms](./algorithms/index.md) - Explore supported RL algorithms
- [Configuration Reference](./configuration/index.md) - Detailed API and runtime configuration documentation
- [Contribution Guide](./contribution/index.md) - Add your own tools, environments, and algorithms

## Documentation Structure

This documentation is organized into the following sections:

### Getting Started
- **[Installation](./getting-started/Installation.md)** - Environment setup, dependencies, and troubleshooting
- **[Quick Start](./getting-started/quick-start.md)** - Run core workflows and example scripts
- **[Troubleshooting & FAQ](./getting-started/troubleshooting.md)** - Common issues and solutions

### Core Modules
- **[Core Modules Overview](./core-modules/index.md)** - High-level architecture map
- **[Agent System](./core-modules/agent-system.md)** - Multi-turn environment and manager flow
- **[Self-Evolution](./core-modules/evolution.md)** - Policy-verifier iterative refinement
- **[Dataset Pipeline](./core-modules/dataset.md)** - Data preprocessing and schema normalization
- **[Tools](./core-modules/tools.md)** - Tool registration, execution, and built-ins

### Configuration
- **[Configuration Overview](./configuration/index.md)** - Hydra basics and CLI overrides
- **[RL Training Config](./configuration/rl_config.md)** - PPO/GRPO parameter details
- **[Generation Config](./configuration/generation.md)** - Offline generation settings
- **[Evolving Config](./configuration/evolving.md)** - Self-evolution runtime settings

### Algorithms
- **[Algorithms Overview](./algorithms/index.md)** - End-to-end workflow map
- **[RL Training](./algorithms/rl-training.md)** - verl integration and RL training flow
- **[SFT](./algorithms/sft.md)** - Supervised fine-tuning pipeline
- **[Evolving Pipeline](./algorithms/evolving-pipeline.md)** - Inference-time self-improvement

### Contribution
- **[Contribution Guide](./contribution/index.md)** - Extension points and conventions
- **[Adding a New Tool](./contribution/new-tool.md)** - Implement and register custom tools
- **[Adding a New Environment](./contribution/new-environment.md)** - Add a new task domain
- **[Adding a New Algorithm](./contribution/new-algorithm.md)** - Add a new workflow

## Recommended Reading Path

If you're new to AlphaApollo, we recommend reading in this order:

1. Start with **Installation** and **Quick Start** to run a working baseline
2. Read **Core Modules Overview** and **Agent System** to understand runtime flow
3. Choose your path: **RL Training** / **SFT** / **Evolving Pipeline**
4. Use **Configuration** pages to tune behavior and scale
5. Follow **Contribution Guide** to extend the framework safely

## Community & Support

- **GitHub**: [AlphaApollo](https://github.com/tmlr-group/AlphaApollo)
- **Paper**: [AlphaApollo on arXiv](https://arxiv.org/pdf/2510.06261)

## Contributing

We welcome contributions! AlphaApollo is open-source under the Apache 2.0 license. Check out our [contributing guide](https://github.com/tmlr-group/AlphaApollo) to get started.

---

Ready to get started? Head over to the [Installation Guide](./getting-started/Installation.md) to begin your journey with AlphaApollo!
