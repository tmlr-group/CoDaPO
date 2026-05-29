---
id: installation
title: Installation
description: Set up AlphaApollo with Conda, install dependencies, and troubleshoot common runtime issues.
sidebar_position: 2
---

# Installation

This page covers environment setup, dependency installation, and common installation/runtime issues.

## Requirements

- Linux with NVIDIA GPU(s) (recommended for training/evolution workloads)
- CUDA-compatible PyTorch environment
- Python 3.12 (project default)

> **Tip**  
> If you only want to get started quickly, complete this page and then jump to `quick-start`.

## Install AlphaApollo

### 1) Clone repository

```bash
git clone https://github.com/tmlr-group/AlphaApollo.git
cd AlphaApollo
```

### 2) Create and activate environment

```bash
conda create -n alphaapollo python==3.12 -y
conda activate alphaapollo
```

### 3) Run project installer

```bash
bash installation.sh
```

This script is expected to install dependencies used by AlphaApollo workflows (including Verl-related runtime requirements).

If no matching wheel exists for your environment, disable flash-attn path and continue with a standard attention backend.
