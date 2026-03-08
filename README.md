# Language Modeling using Transformers (LMT)

[![CI](https://github.com/michaelellis003/LMT/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/michaelellis003/LMT/actions/workflows/ci-cd.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

An educational PyTorch library for understanding how modern transformer architectures work -- from attention mechanisms to full language models. Every component is written to be **understood**, with clear code, detailed docstrings, and mathematical notation that maps directly to the papers.

## Features

**Attention & Sequence Mixing**

| Component | Description | Paper |
|-----------|-------------|-------|
| Multi-Head Attention | Standard scaled dot-product attention | Vaswani et al., 2017 |
| Grouped Query Attention | Shared KV heads for efficiency | Ainslie et al., 2023 |
| Sliding Window Attention | Local attention with fixed window | Beltagy et al., 2020 |
| Multi-Head Latent Attention | KV compression + decoupled RoPE | DeepSeek-AI, 2024 |
| Flash Attention | Tiled attention with online softmax (educational) | Dao et al., 2022 |
| Gated Delta Networks | Linear attention with delta rule | Yang et al., 2024 |
| SSD | Structured State Space Duality (Mamba-2) | Dao & Gu, 2024 |

**Feed-Forward Networks**

| Component | Description |
|-----------|-------------|
| SwiGLU | Gated FFN with Swish activation (LLaMA, Mixtral) |
| Mixture of Experts | Top-k sparse routing with load balancing loss |

**Other Components**: RMSNorm, RoPE, YaRN, ALiBi, KV Cache, Multi-Token Prediction, Mixture of Depths

**Model Architectures**

| Model | Key Components |
|-------|---------------|
| GPT | Multi-head attention + GELU FFN + learned position embeddings |
| LLaMA | RMSNorm + RoPE + SwiGLU + GQA |
| Mamba | Selective State Space Model (no attention) |
| Mixtral | LLaMA + MoE FFN + sliding window attention |
| DeepSeek-V2 | MLA + MoE + decoupled RoPE |
| Qwen3 | LLaMA-style + QK-Norm + weight tying |
| Gemma | Interleaved local/global attention |
| Kimi | DeepSeek-style MLA + MoE |

## Installation

```bash
pip install pylmt
```

Or install from source for development:

```bash
git clone https://github.com/michaelellis003/LMT.git
cd LMT
pip install uv
uv sync
```

## Quick Start

```python
import torch
from lmt.models.config import ModelConfig
from lmt.models.llama import LLaMA

config = ModelConfig(
    vocab_size=32000,
    embed_dim=512,
    num_heads=8,
    num_kv_heads=4,     # GQA: 4 KV heads shared across 8 query heads
    num_layers=6,
    context_length=1024,
    dropout=0.0,
)

model = LLaMA(config)
x = torch.randint(0, config.vocab_size, (1, 128))
logits = model(x)  # [1, 128, 32000]
```

### Using Individual Layers

```python
from lmt.layers.attention import GroupedQueryAttention
from lmt.layers.ffn import SwiGLU
from lmt.layers.normalization import RMSNorm

norm = RMSNorm(d_model=512)
attn = GroupedQueryAttention(config)
ffn = SwiGLU(d_model=512)

x = torch.randn(1, 64, 512)
x = x + attn(norm(x))  # Pre-norm attention
x = x + ffn(norm(x))   # Pre-norm FFN
```

### Mixture of Experts

```python
from lmt.models.mixtral import Mixtral

config = ModelConfig(
    vocab_size=32000, embed_dim=512, num_heads=8,
    num_kv_heads=4, num_layers=8,
    context_length=2048, window_size=256, dropout=0.0,
)

model = Mixtral(config, num_experts=8, top_k=2)
logits = model(x)
aux_loss = model.aux_loss  # load balancing loss for training
```

## Project Structure

```
src/lmt/
  layers/
    attention/     # MHA, GQA, SWA, MLA, Flash, GDN, SSD, KV Cache
    ffn/           # SwiGLU, MoE (Router + Experts)
    normalization/ # RMSNorm
    positional/    # RoPE, YaRN, ALiBi
    ssm/           # Selective SSM, Mamba Block
    blocks/        # ConfigurableBlock, Mixture of Depths
  models/
    gpt/           # GPT (decoder-only transformer)
    llama/         # LLaMA (RMSNorm + RoPE + SwiGLU + GQA)
    mamba/         # Mamba (SSM, no attention)
    mixtral/       # Mixtral (LLaMA + MoE + sliding window)
    deepseek/      # DeepSeek-V2 (MLA + MoE)
    qwen3/         # Qwen3 (QK-Norm + weight tying)
    gemma/         # Gemma (interleaved attention)
    kimi/          # Kimi (MLA + MoE)
  training/        # Trainer, DPO, GRPO, evaluation, configs
  tokenizer/       # BPE, char, naive tokenizers
  generate.py      # Text generation utilities
```

## Development

```bash
uv run pytest tests/ -v       # Run tests (500+ passing)
uv run ruff check src/ tests/ # Lint
uv run ruff format src/ tests/ # Format
uv run pyright src/            # Type check
uv run mkdocs serve            # Local docs server
```

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
