# LLaMA

Meta's LLaMA architecture (Touvron et al., 2023) combines several
modern improvements over GPT into a cleaner, more efficient design.

## What Changed From GPT

| Aspect | GPT | LLaMA |
|--------|-----|-------|
| Normalization | Pre-norm LayerNorm | **Pre-norm RMSNorm** |
| Position encoding | Learned absolute | **RoPE** (relative) |
| FFN activation | GELU | **SwiGLU** |
| Attention | Standard MHA | **Grouped Query Attention** |
| Position embedding | Added to input | **Applied inside attention** |
| Bias terms | Yes | **No** (all linear layers) |

## Architecture

```
Tokens -> Token Embedding (no positional embedding!)
  -> N x LlamaBlock:
       RMSNorm -> LlamaAttention(GQA + RoPE) -> residual
       RMSNorm -> SwiGLU -> residual
  -> RMSNorm -> Linear head -> logits
```

## Usage

```python
from lmt.models.llama import LLaMA
from lmt.models.config import ModelConfig

config = ModelConfig(
    vocab_size=32000,
    embed_dim=512,
    num_heads=8,
    num_kv_heads=4,   # GQA: 4 KV heads, 2 queries per group
    num_layers=8,
    context_length=2048,
    dropout=0.0,
)

model = LLaMA(config)
```

## Weight Initialization

LLaMA uses **scaled residual initialization**: output projections
(`out_proj` in attention, `w2` in SwiGLU) are initialized with a
reduced standard deviation:

\[
\sigma = \frac{0.02}{\sqrt{2 \cdot N_{\text{layers}}}}
\]

This prevents the residual stream from growing too large in deep models.

## API Reference

::: lmt.models.llama.LLaMA
    options:
      show_source: true
      members:
        - __init__
        - forward

## References

- Touvron et al., [*LLaMA: Open and Efficient Foundation Language Models*](https://arxiv.org/abs/2302.13971) (2023) -- LLaMA
- Touvron et al., [*Llama 2: Open Foundation and Fine-Tuned Chat Models*](https://arxiv.org/abs/2307.09288) (2023) -- LLaMA 2 (introduced GQA)
