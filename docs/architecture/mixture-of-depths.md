# Mixture of Depths

## Overview

Standard transformers process every token through every layer — regardless of whether
that token is a simple function word like "the" or a complex reasoning step.
**Mixture of Depths** (MoD) changes this by adding a learned router at each layer
that decides which tokens get processed and which skip through via the residual
connection only.

This is analogous to [Mixture of Experts](ffn.md#mixture-of-experts-moe) but applied
at the **block level** instead of the FFN level:

| Pattern | Routing Granularity | What Gets Routed |
|---------|-------------------|-----------------|
| MoE | FFN level | Tokens → different expert FFNs |
| MoD | Block level | Tokens → process vs skip entire block |

## Architecture

```
                    ┌──────────────┐
                    │  DepthRouter │
                    │   (learned)  │
                    └──────┬───────┘
                           │
                    ┌──────┴──────┐
                    │             │
               Selected      Skipped
              (top C%)      (1 - C%)
                    │             │
                    ▼             │
            ┌──────────────┐     │
            │ Attn + FFN   │     │
            │ (full block) │     │
            └──────┬───────┘     │
                   │             │
                   ▼             ▼
              x + w·Δ           x
                   │             │
                   └─────┬───────┘
                         ▼
                      Output
```

Where $\Delta = \text{Block}(x) - x$ is the block's contribution, and $w$ is
the router's continuous weight (preserves gradient flow).

## DepthRouter

The router is a simple linear projection from $d_\text{model}$ to a scalar score,
followed by top-k selection:

$$\text{score}_t = \sigma(x_t \cdot w_\text{gate})$$

The top $\lfloor C \cdot L \rfloor$ tokens (by score) are selected for processing,
where $C$ is the capacity fraction and $L$ is the sequence length.

### Why Not All-or-Nothing?

The router's continuous weight $w_t$ is multiplied into the block's output for
selected tokens. This serves two purposes:

1. **Gradient flow**: A hard binary mask would block gradients for the routing
   decision. The continuous weight allows the router to learn.
2. **Soft gating**: The model can learn to partially attenuate block contributions
   even for selected tokens.

## LMT Usage

```python
from lmt.layers.blocks import MoDBlock, BlockConfig
from lmt.models.config import ModelConfig

config = ModelConfig(embed_dim=512, num_heads=8, ...)
block_config = BlockConfig(attention='gqa', ffn='swiglu', norm='rmsnorm')

# 50% of tokens processed, 50% skip
mod_block = MoDBlock(config, block_config, capacity=0.5)
output = mod_block(x)

# Access routing auxiliary loss
aux_loss = mod_block.aux_loss
```

## Capacity Tuning

The `capacity` parameter controls the compute/quality tradeoff:

| Capacity | Behavior | Use Case |
|----------|----------|----------|
| 0.0 | Pure residual (identity) | Debugging |
| 0.25 | 25% tokens processed | Maximum efficiency |
| 0.5 | 50% tokens processed | Balanced (paper default) |
| 1.0 | All tokens processed | Standard transformer |

## Design Insight

MoD demonstrates that the "routing" pattern is more general than just MoE.
The same top-k selection mechanism can be applied at different granularities:

- **Token → Expert** (MoE): Which FFN processes this token?
- **Token → Layer** (MoD): Should this token be processed at all?
- **Token → Depth** (Early Exit): Should this token stop processing early?

LMT's composable architecture makes it easy to experiment with all of these.

## References

- [Mixture-of-Depths: Dynamically allocating compute in transformer-based language models](https://arxiv.org/abs/2404.02258)
  (Raposo et al., 2024)
