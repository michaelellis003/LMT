# Gemma

Google's Gemma architecture (Gemma Team, 2024/2025) extends the
LLaMA family with interleaved local/global attention and embedding
scaling.

## What Changed From LLaMA

| Aspect | LLaMA | Gemma 3 |
|--------|-------|---------|
| Attention pattern | Global (full) every layer | **Interleaved local/global** |
| Q/K normalization | None | **QK-Norm** |
| Weight tying | Separate | **Tied** |
| Embedding scaling | None | **Multiply by** $\sqrt{d_{model}}$ |

## Interleaved Local/Global Attention

The key architectural innovation in Gemma 3. Most layers use
**sliding window** (local) attention with a fixed window size,
and every Nth layer uses **full** (global) attention:

```
Layer 0: local  (window=512)
Layer 1: local  (window=512)
Layer 2: local  (window=512)
Layer 3: GLOBAL (full attention)
Layer 4: local  (window=512)
...
```

**Why this works**: Local attention is O(T * w) instead of O(T^2),
making it much cheaper for long sequences. But purely local attention
loses long-range dependencies. By inserting periodic global layers,
the model maintains long-range connectivity while keeping most
computation efficient.

The effective receptive field grows linearly with depth through the
global layers, similar to how dilated convolutions work.

## Embedding Scaling

Gemma multiplies the embedding output by $\sqrt{d_{model}}$ before
passing it to the transformer blocks. This normalizes the embedding
magnitude relative to the model dimension, stabilizing training
across different model scales.

## Usage

```python
from lmt import Gemma, ModelConfig

config = ModelConfig(
    vocab_size=32000,
    embed_dim=512,
    num_heads=8,
    num_kv_heads=4,
    num_layers=12,
    context_length=4096,
    dropout=0.0,
)

model = Gemma(
    config,
    local_window=512,   # sliding window size for local layers
    global_every=4,     # global attention every 4th layer
)
```

## API Reference

::: lmt.models.gemma.Gemma
    options:
      show_source: true
      members:
        - __init__
        - forward
        - forward_hidden

## References

- Gemma Team, [*Gemma: Open Models Based on Gemini Research and Technology*](https://arxiv.org/abs/2403.08295) (2024)
- Gemma Team, [*Gemma 3 Technical Report*](https://arxiv.org/abs/2503.19786) (2025)
