# LMT - Language Modeling using Transformers

An educational PyTorch library for understanding how modern transformer
architectures work -- from attention mechanisms to full language models.

## Why LMT?

Most transformer implementations prioritize performance over readability.
LMT takes the opposite approach: every component is written to be
**understood**, with clear code, detailed docstrings, and mathematical
notation that maps directly to the papers.

## What's Inside

### Attention & Sequence Mixing

| Component | Description | Paper |
|-----------|-------------|-------|
| Multi-Head Attention | Standard scaled dot-product attention | Vaswani et al., 2017 |
| Grouped Query Attention | Shared KV heads for efficiency | Ainslie et al., 2023 |
| Sliding Window Attention | Local attention with fixed window | Beltagy et al., 2020 |
| Multi-Head Latent Attention | KV compression via low-rank latent | DeepSeek-AI, 2024 |
| Flash Attention | Tiled attention with online softmax (educational) | Dao et al., 2022 |
| Gated Delta Networks | Linear attention with delta rule | Yang et al., 2024 |
| SSD | Structured State Space Duality (Mamba-2) | Dao & Gu, 2024 |

### Positional Encodings

| Component | Description |
|-----------|-------------|
| RoPE | Rotary Position Embeddings |
| YaRN | RoPE context extension via NTK-aware scaling |
| ALiBi | Attention with Linear Biases (no positional embeddings) |

### Feed-Forward Networks

| Component | Description |
|-----------|-------------|
| SwiGLU | Gated FFN with Swish activation |
| Mixture of Experts | Sparse top-k routing with shared experts and load balancing |

### Advanced Techniques

| Component | Description |
|-----------|-------------|
| Mixture of Depths | Dynamic compute routing -- skip layers for easy tokens |
| Multi-Token Prediction | Train on multiple future tokens simultaneously |
| KV Cache | Inference-time caching for autoregressive generation |

### Model Architectures

| Model | Key Components |
|-------|---------------|
| GPT | Multi-head attention + learned positional embeddings |
| LLaMA | RMSNorm + RoPE + SwiGLU + GQA |
| Mixtral | LLaMA + MoE FFN + sliding window attention |
| Mamba | Selective State Space Model (no attention) |
| Configurable | Mix any attention + FFN + norm via `BlockConfig` |

### Tokenizers

| Tokenizer | Description |
|-----------|-------------|
| CharTokenizer | Character-level, no dependencies, great for experiments |
| BPETokenizer | Byte Pair Encoding via tiktoken (GPT-2/GPT-4 vocabularies) |
| NaiveTokenizer | Simple word-level tokenizer for testing |

## Quick Start

```bash
pip install pylmt
```

```python
import torch
from lmt import ModelConfig, LLaMA

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
- [**RoPE Explorer**](visualizations/rope-explorer.md) --
  visualize rotation circles, frequency spectrum, and dot-product decay
- [**Positional Encoding Comparison**](visualizations/positional-encoding.md) --
  compare RoPE, ALiBi, and sinusoidal encodings side-by-side

## Design Principles

1. **Readability first** -- code should read like a textbook
2. **Math in docstrings** -- LaTeX formulas that match the papers
3. **Test-driven** -- every component has a comprehensive test suite (~400 tests)
4. **Composable** -- mix and match layers to build new architectures
