# Qwen3

Alibaba's Qwen3 architecture (Qwen Team, 2025) is a modern LLaMA-family
model with two key additions that improve training stability at scale.

## What Changed From LLaMA

| Aspect | LLaMA | Qwen3 |
|--------|-------|-------|
| Q/K normalization | None | **QK-Norm** (RMSNorm per-head) |
| Weight tying | Separate embeddings | **Tied** input/output embeddings |
| Everything else | Same | Same |

## Why QK-Norm?

Without normalization, the dot product $Q \cdot K^T$ can grow unbounded
with network depth. This pushes softmax into saturation (near-zero
gradients), causing training instability. QK-Norm applies RMSNorm
to Q and K per-head before the dot product, keeping attention logits
bounded regardless of input magnitude.

Used by Qwen3, Gemma 3, and Kimi K2.

## Why Weight Tying?

The token embedding ("what does this token mean?") and the output head
("which token should come next?") answer the same question from different
directions. Sharing their weights reduces parameter count by
`vocab_size * embed_dim` with minimal quality loss. Standard practice
for models under ~1B parameters.

## Usage

```python
from lmt import Qwen3, ModelConfig

config = ModelConfig(
    vocab_size=32000,
    embed_dim=512,
    num_heads=8,
    num_kv_heads=4,
    num_layers=8,
    context_length=2048,
    dropout=0.0,
)

model = Qwen3(config)
# qk_norm and tie_weights are forced True automatically
```

## API Reference

::: lmt.models.qwen3.Qwen3
    options:
      show_source: true
      members:
        - __init__
        - forward

## References

- Qwen Team, [*Qwen3 Technical Report*](https://arxiv.org/abs/2505.09388) (2025)
