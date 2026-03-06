# YaRN: Context Window Extension

## The Problem

[RoPE](positional.md) encodes position by rotating query and key vectors at position-dependent angles.
The rotation frequencies are precomputed for a fixed maximum sequence length (e.g., 2048 or 4096).
When you try to use the model at positions **beyond** this limit, the frequencies haven't been
calibrated — the model's attention patterns break down and perplexity spikes.

## Why Not Just Interpolate?

The simplest approach, **linear position interpolation**, scales all position indices by
$L_\text{original} / L_\text{extended}$. This works but has a problem: it compresses
**all** frequency dimensions equally, including the high-frequency ones that encode
fine-grained local patterns. These local patterns don't need adjustment — they work
the same regardless of total sequence length.

## YaRN's Key Insight

Different RoPE dimensions operate at different **wavelengths**:

- **High-frequency** dimensions (short wavelength): Encode nearby relationships
  (e.g., "the word right next to me"). These don't change when context extends.
- **Low-frequency** dimensions (long wavelength): Encode global position
  (e.g., "I'm near the beginning/end of the sequence"). These need to stretch.

YaRN applies a **per-dimension interpolation ramp**:

$$\text{freq}'_i = \text{freq}_i \cdot \frac{1}{(1 - \gamma_i) + \gamma_i / s}$$

where $s$ is the scale factor and $\gamma_i$ is the ramp function:

$$\gamma_i = \text{clamp}\left(\frac{\lambda_i - L \cdot \beta_\text{slow}}{L \cdot (\beta_\text{fast} - \beta_\text{slow})}, 0, 1\right)$$

| Dimension Type | Wavelength | $\gamma$ | Effect |
|---|---|---|---|
| High frequency | $< L \cdot \beta_\text{slow}$ | 0 | **Preserved** (no interpolation) |
| Mid frequency | Between bounds | 0-1 | **Blended** (smooth transition) |
| Low frequency | $> L \cdot \beta_\text{fast}$ | 1 | **Fully interpolated** |

## Attention Temperature

Extending context also changes the **entropy** of attention distributions. With more
positions to attend to, the attention becomes more diffuse. YaRN compensates with a
temperature correction:

$$t = 0.1 \ln(s) + 1$$

Attention logits are scaled by $1/t$ to sharpen the distribution back to training-time entropy.

## LMT Usage

```python
from lmt.layers.positional import YaRNRoPE

# Extend a 2048-context model to 8192
yarn = YaRNRoPE(
    d_model=64,
    original_max_seq_len=2048,
    extended_max_seq_len=8192,
    beta_fast=32,   # default
    beta_slow=1,    # default
)

# Use exactly like standard RoPE
q_rot, k_rot = yarn.apply_rotary_emb(q, k)

# Access temperature for attention scaling
temperature = yarn.attn_temperature  # ~1.14 for 4x extension
```

### With a LLaMA Model

YaRN is a drop-in replacement for standard RoPE in any model that uses it:

```python
from lmt.layers.positional import YaRNRoPE
from lmt.layers.blocks import BlockConfig
from lmt.models.base import BaseModel
from lmt.models.config import ModelConfig

config = ModelConfig(context_length=8192, ...)  # Extended context
yarn = YaRNRoPE(d_model=head_dim, original_max_seq_len=2048,
                extended_max_seq_len=8192)
block_config = BlockConfig(attention='gqa', ffn='swiglu',
                          norm='rmsnorm', rope=yarn)
model = BaseModel(config, block_config=block_config)
```

## Comparison of RoPE Extension Methods

| Method | Approach | Strengths | Weaknesses |
|--------|----------|-----------|------------|
| Linear Interpolation | Scale all positions by L_orig/L_ext | Simple | Hurts local patterns |
| NTK-aware | Adjust base frequency | Preserves high freq | Hard to tune |
| **YaRN** | Per-dimension ramp + temperature | Best of both worlds | Slightly more complex |
| NTK-by-parts | Different treatment per freq band | Good quality | No temperature fix |

## References

- [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071)
  (Peng et al., 2023, ICLR 2024)
- [Extending the RoPE — EleutherAI Blog](https://blog.eleuther.ai/yarn/)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
  (Su et al., 2021) — original RoPE paper
