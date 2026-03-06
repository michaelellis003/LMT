# LMT - Language Modeling using Transformers

An educational PyTorch library for understanding how modern transformer
architectures work -- from attention mechanisms to full language models.

## Why LMT?

Most transformer implementations prioritize performance over readability.
LMT takes the opposite approach: every component is written to be
**understood**, with clear code, detailed docstrings, and mathematical
notation that maps directly to the papers.

## What's Inside

### Attention Mechanisms

| Component | Description | Paper |
|-----------|-------------|-------|
| Multi-Head Attention | Standard scaled dot-product attention | Vaswani et al., 2017 |
| Grouped Query Attention | Shared KV heads for efficiency | Ainslie et al., 2023 |
| Sliding Window Attention | Local attention with fixed window | Beltagy et al., 2020 |
| Multi-Head Latent Attention | KV compression via low-rank latent | DeepSeek-AI, 2024 |

### Feed-Forward Networks

| Component | Description |
|-----------|-------------|
| SwiGLU | Gated FFN with Swish activation |
| Mixture of Experts | Sparse top-k routing with load balancing |

### Model Architectures

| Model | Key Components |
|-------|---------------|
| GPT | Multi-head attention + standard FFN |
| LLaMA | RMSNorm + RoPE + SwiGLU + GQA |
| Mixtral | LLaMA + MoE FFN + sliding window attention |

## Quick Start

```bash
pip install pylmt
```

```python
import torch
from lmt.models.config import ModelConfig
from lmt.models.llama import LLaMA

config = ModelConfig(
    vocab_size=32000,
    embed_dim=512,
    num_heads=8,
    num_kv_heads=4,  # GQA: 4 KV heads shared across 8 query heads
    num_layers=6,
    context_length=1024,
    dropout=0.0,
)

model = LLaMA(config)
x = torch.randint(0, config.vocab_size, (1, 128))
logits = model(x)  # [1, 128, 32000]
```

## Interactive Explorations

Learn by experimenting, not just reading:

- [**Attention Pattern Explorer**](visualizations/attention-patterns.md) --
  see how causal, sliding window, and sparse attention masks actually look

## Design Principles

1. **Readability first** -- code should read like a textbook
2. **Math in docstrings** -- LaTeX formulas that match the papers
3. **Test-driven** -- every component has a comprehensive test suite
4. **Composable** -- mix and match layers to build new architectures
